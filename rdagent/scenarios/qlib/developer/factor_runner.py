from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
import numpy as np

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle

pandarallel.initialize(verbose=1)

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf_baseline.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_ic_cross_section(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.Series:
        """
        多标的：同一截面上的截面相关性（原逻辑）
        """
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size), dtype=float)
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def _to_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
            if "instrument" in df.index.names:
                # 单标的：直接丢掉 instrument 这一层，避免层级/顺序问题
                df = df.reset_index("instrument", drop=True)
        return df.sort_index()

    def _canon_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names)
            if "datetime" in names and "instrument" in names:
                front = ["datetime", "instrument"]
                rest = [n for n in names if n not in front]
                df = df.reorder_levels(front + rest)
        return df.sort_index()

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        is_multi_index = isinstance(SOTA_feature.index, pd.MultiIndex)
        n_instruments = (
            SOTA_feature.index.get_level_values("instrument").nunique()
            if is_multi_index and "instrument" in SOTA_feature.index.names
            else 1
        )

        if n_instruments > 1:
            # 你原来的多标的逻辑保持不动
            concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
            IC_series = (
                concat_feature.groupby("datetime")
                .parallel_apply(
                    lambda x: self.calculate_ic_cross_section(
                        x, SOTA_feature.shape[1], new_feature.shape[1]
                    )
                )
                .mean()
            )
            IC_series.index = pd.MultiIndex.from_product(
                [range(SOTA_feature.shape[1]), range(new_feature.shape[1])]
            )
            IC_max = IC_series.unstack().abs().max(axis=0)
            keep_mask = (IC_max.fillna(0.0) < 0.99).values
            return new_feature.iloc[:, keep_mask]

        # ===== 单标的逻辑（加最小输出）=====
        dbg = True

        old_cols = list(SOTA_feature.columns)
        new_cols = list(new_feature.columns)

        SOTA_feature = self._canon_index(SOTA_feature)
        new_feature = self._canon_index(new_feature)

        SOTA_aligned, new_aligned = SOTA_feature.align(new_feature, join="inner", axis=0)

        if dbg and (len(SOTA_aligned) == 0 or len(new_aligned) == 0):
            print("[DEDUP][TS] align produced empty intersection.")
            print(f"  SOTA rows={len(SOTA_feature)}, new rows={len(new_feature)}")
            print(f"  SOTA index.names={getattr(SOTA_feature.index,'names',None)}")
            print(f"  new  index.names={getattr(new_feature.index,'names',None)}")

        # 对齐后为空：宁可不去重（避免误删）
        if len(SOTA_aligned) == 0 or len(new_aligned) == 0:
            return new_feature

        n_old = SOTA_aligned.shape[1]
        concat = pd.concat([SOTA_aligned, new_aligned], axis=1)

        corr = concat.corr()
        corr_block = corr.iloc[:n_old, n_old:]  # old x new

        # 关键：NaN 当 0，避免“算不出相关性 -> 误删”
        IC_max = corr_block.abs().max(axis=0)
        IC_max_filled = IC_max.fillna(0.0)
        keep_mask = (IC_max_filled < 0.99)
        kept = list(IC_max_filled[keep_mask].index)
        removed = list(IC_max_filled[~keep_mask].index)

        # 只在“发生删列 / 或出现 NaN / 或有效样本极少”时输出
        need_print = dbg and (
            len(removed) > 0
            or IC_max.isna().any()
        )

        if need_print:
            print(f"[DEDUP][TS] rows(aligned)={len(SOTA_aligned)} old={len(old_cols)} new={len(new_cols)}")
            if IC_max.isna().any():
                na_cols = list(IC_max[IC_max.isna()].index)
                print(f"[DEDUP][TS] IC_max has NaN for new cols: {na_cols}")

            # 对每个被删的新因子，打印：最相似旧因子、max|corr|、overlap、std
            for c in removed:
                # 哪个旧因子最像
                s = corr_block[c].abs()
                best_old = s.idxmax() if s.notna().any() else None
                best_corr = float(s.max()) if s.notna().any() else float("nan")

                # overlap：该 best_old 与 c 的有效重叠样本数
                overlap = None
                if best_old is not None:
                    overlap = int((SOTA_aligned[best_old].notna() & new_aligned[c].notna()).sum())

                # 新因子自身 std（看是否近似常数）
                std_new = float(new_aligned[c].std(skipna=True))

                print(f"  [REMOVED] {c}: max|corr|={best_corr:.6f} vs old={best_old}, overlap={overlap}, std={std_new:.6g}")

            print(f"[DEDUP][TS] kept={len(kept)} removed={len(removed)}")
            if len(removed) > 0:
                print(f"[DEDUP][TS] removed_list={removed}")

        return new_feature.loc[:, kept]
    # def calculate_information_coefficient(
    #     self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    # ) -> pd.DataFrame:
    #     res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
    #     for col1 in range(SOTA_feature_column_size):
    #         for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
    #             res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
    #                 :, col1
    #             ].corr(concat_feature.iloc[:, col2])
    #     return res

    # def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
    #     # calculate the IC between each column of SOTA_feature and new_feature
    #     # if the IC is larger than a threshold, remove the new_feature column
    #     # return the new_feature

    #     concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
    #     IC_max = (
    #         concat_feature.groupby("datetime")
    #         .parallel_apply(
    #             lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
    #         )
    #         .mean()
    #     )
    #     IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
    #     IC_max = IC_max.unstack().max(axis=0)
    #     return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info(f"Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            logger.info(f"New factor processing ...")
            # Process the new factors data
            new_factors = process_factor_data(exp)

            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                before_cols = list(new_factors.columns)
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                after_cols = list(new_factors.columns)
                removed = [c for c in before_cols if c not in set(after_cols)]
                logger.info(f"[DEDUP] before={len(before_cols)} after={len(after_cols)} removed={removed}")
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # Sort and nest the combined factors under 'feature'
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            num_features = RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns)
            logger.info(f"Factor data processing completed.")

            # Due to the rdagent and qlib docker image in the numpy version of the difference,
            # the `combined_factors_df.pkl` file could not be loaded correctly in qlib dokcer,
            # so we changed the file type of `combined_factors_df` from pkl to parquet.
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # Save the combined factors to the workspace
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # If model exp exists in the previous experiment
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info(f"Experiment execution ...")
            if exist_sota_model_exp:
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]}
                )
                env_to_use = {"PYTHONPATH": "./"}
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                if sota_training_hyperparameters:
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        }
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                if sota_model_type == "TimeSeries":
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                    )
                elif sota_model_type == "Tabular":
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # model + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use
                )
            else:
                # LGBM + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name=(
                        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                    )
                )
        else:
            logger.info(f"Experiment execution ...")
            result, stdout = exp.experiment_workspace.execute(
                qlib_config_name=(
                    f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                )
            )

        if result is None:
            logger.error(f"Failed to run this experiment, because {stdout}")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        exp.result = result
        exp.stdout = stdout

        return exp
