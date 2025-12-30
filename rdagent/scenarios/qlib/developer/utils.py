from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

def process_factor_data(exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
    """
    Debug 代码，同时也是重要的 log
    和 rdagent/scenarios/qlib/developer/factor_runner.py 中的 [dedup] 联合使用，输出哪些因子被正常生成，哪些因子因为各种原因没有生成
    """
    if isinstance(exp_or_list, QlibFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []

    # 收集每个 experiment 下成功生成的 factor df
    for exp in exp_or_list:
        if isinstance(exp, QlibFactorExperiment):
            if len(exp.sub_tasks) > 0:
                # 有 sub_tasks 代表是“设计任务开发”的实验：应当有 feedback 可用
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                
                # 重要 log：检查 sub_workspace_list 和 prop_dev_feedback 长度是否一致，哪些成功了，哪些失败了
                pairs = list(zip(exp.sub_workspace_list, exp.prop_dev_feedback))
                logger.info(f"[FACTOR-MERGE] sub_workspace_list={len(exp.sub_workspace_list)}, "
                            f"prop_dev_feedback={len(exp.prop_dev_feedback)}, zipped={len(pairs)}")

                selected = [(impl, fb) for impl, fb in pairs if impl and fb]
                logger.info(f"[FACTOR-MERGE] selected_for_execute={len(selected)}, "
                            f"skipped_in_zip_or_filter={len(exp.sub_workspace_list) - len(selected)}")
                
                message_and_df_list = multiprocessing_wrapper(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ],  # only execute successfully feedback
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                error_message = ""
                for message, df in message_and_df_list:
                    # 成功标准：df 不为空，且 index 包含 datetime（Qlib 标准 MultiIndex）
                    if df is not None and "datetime" in df.index.names:
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(df)
                            logger.info(
                                f"Factor data from {exp.hypothesis.concise_justification} is successfully generated."
                            )
                        else:
                            logger.warning(f"Factor data from {exp.hypothesis.concise_justification} is not generated.")
                    else:
                        error_message += f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                        logger.warning(
                            f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                        )

    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        raise FactorEmptyError(
            f"No valid factor data found to merge (in process_factor_data) because of {error_message}."
        )
