from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_brief_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        return (
            md5_hash(data_type + self.file_dict["factor.py"])
            if ("factor.py" in self.file_dict and not self.raise_exception)
            else None
        )

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. link all the source data to the workspace path folder
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import the factor.py dump get the factor value to result.h5
        5. read the factor value from the output file in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe


        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])

        """
        self.before_execute()
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None
        with FileLock(self.workspace_path / "execution.lock"):
            if self.target_task.version == 1:
                source_data_path = (
                    Path(
                        FACTOR_COSTEER_SETTINGS.data_folder_debug,
                    )
                    if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                    else Path(
                        FACTOR_COSTEER_SETTINGS.data_folder,
                    )
                )
            elif self.target_task.version == 2:
                # TODO you can change the name of the data folder for a better understanding
                source_data_path = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / f"factor.py"

            self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)

            # 修改代码的目的很清晰：将高阶因子硬编码到代码中。runtime 计算并不安全友好，但可以快速落地。
            # 该流程已经被修改，现在会在实验之前，在 h5 文件中预先计算好这些高阶因子，供因子代码直接使用。参考RD-Agent/rdagent/scenarios/qlib/experiment/factor_data_template/generate.py和RD-Agent/rdagent/scenarios/qlib/experiment/factor_data_template/enrich_daily_py.py


            # ------------------------------
            # ★ 强制注入所有基础简单因子（最快速修复）
            # ------------------------------

            #       - **Calendar / seasonality features**: $year, $month, $dayofweek, $dayofyear  
            #     Capture seasonal cycles, weekday effects, and time-based structure.

            #   - **Position / participation features**: $open_interest, $OIMOM5, $OIMOM20  
            #     Reflect market participation, funding pressure, and crowding risk.

            #   - **Price trend & deviation features**: $RESI5, $RESI10, $RSQR5, $RSQR10, $RSQR20, $RSQR60  
            #     Describe short- to long-term trend strength and deviations from trend.

            #   - **Volatility & activity features**: $WVMA5, $WVMA60, $VSTD5, $STD5  
            #     Represent volatility bursts and abnormal price–volume activity.

            #   - **Price–volume relation features**: $CORR5, $CORR10, $CORR20, $CORR60, $CORD5, $CORD10, $CORD60  
            #     Measure how price interacts with volume or volume changes.

            #   - **K-line structure features**: $KLEN, $KLOW  
            #     Capture intraday reversal structure and swing amplitude.

            #   - **Momentum-related features**: $PMOM5, $PMOM10, $PMOM20, $VMOM5  
            #     Reflect short- and medium-term momentum behavior in price and volume.
            # import numpy as np

            # h5_path = self.workspace_path / "daily_pv.h5"
            # if h5_path.exists():
            #     df = pd.read_hdf(h5_path, key="data")

            #     eps = 1e-12

            #     # ========= RESI: Rolling residual ratio ==========
            #     def _resi(x):
            #         n = len(x)
            #         t = np.arange(n)
            #         coef = np.polyfit(t, x, 1)
            #         pred = coef[0] * t + coef[1]
            #         return (x[-1] - pred[-1]) / (x[-1] + eps)

            #     # 仅保留你新清单中的 RESI10/RESI20
            #     df["$RESI10"] = df["$close"].rolling(10).apply(_resi, raw=True)
            #     df["$RESI20"] = df["$close"].rolling(20).apply(_resi, raw=True)

            #     # ========== RSQUARE ==========
            #     def _rsq(x):
            #         n = len(x)
            #         t = np.arange(n)
            #         coef = np.polyfit(t, x, 1)
            #         pred = coef[0] * t + coef[1]
            #         ss_res = ((x - pred) ** 2).sum()
            #         ss_tot = ((x - x.mean()) ** 2).sum() + eps
            #         return 1 - ss_res / ss_tot

            #     df["$RSQR5"]  = df["$close"].rolling(5).apply(_rsq, raw=True)
            #     df["$RSQR10"] = df["$close"].rolling(10).apply(_rsq, raw=True)
            #     df["$RSQR20"] = df["$close"].rolling(20).apply(_rsq, raw=True)
            #     df["$RSQR60"] = df["$close"].rolling(60).apply(_rsq, raw=True)

            #     # ========== KLEN / KLOW ==========
            #     df["$KLEN"] = (df["$high"] - df["$low"]) / (df["$open"] + eps)
            #     df["$KLOW"] = (np.minimum(df["$open"], df["$close"]) - df["$low"]) / (df["$open"] + eps)

            #     # ========== PMOM (5/10/20/60) ==========
            #     df["$PMOM5"]  = df["$close"] / df["$close"].shift(5)  - 1
            #     df["$PMOM10"] = df["$close"] / df["$close"].shift(10) - 1
            #     df["$PMOM20"] = df["$close"] / df["$close"].shift(20) - 1
            #     df["$PMOM60"] = df["$close"] / df["$close"].shift(60) - 1

            #     # ========== VMOM (5/10/20/60) ==========
            #     df["$VMOM5"]  = df["$volume"] / df["$volume"].shift(5)  - 1
            #     df["$VMOM10"] = df["$volume"] / df["$volume"].shift(10) - 1
            #     df["$VMOM20"] = df["$volume"] / df["$volume"].shift(20) - 1
            #     df["$VMOM60"] = df["$volume"] / df["$volume"].shift(60) - 1

            #     # ========== OIMOM (5/10/20/60) ==========
            #     df["$OIMOM5"]  = df["$open_interest"] / df["$open_interest"].shift(5)  - 1
            #     df["$OIMOM10"] = df["$open_interest"] / df["$open_interest"].shift(10) - 1
            #     df["$OIMOM20"] = df["$open_interest"] / df["$open_interest"].shift(20) - 1
            #     df["$OIMOM60"] = df["$open_interest"] / df["$open_interest"].shift(60) - 1

            #     # ========== STD (price vol proxy, normalized by price) ==========
            #     df["$STD5"]  = df["$close"].rolling(5).std()  / (df["$close"] + eps)
            #     df["$STD10"] = df["$close"].rolling(10).std() / (df["$close"] + eps)
            #     df["$STD20"] = df["$close"].rolling(20).std() / (df["$close"] + eps)
            #     df["$STD60"] = df["$close"].rolling(60).std() / (df["$close"] + eps)

            #     # ========== VSTD (volume vol proxy, normalized by volume) ==========
            #     # 与你 YAML: Std($volume, N)/($volume+eps) 一致
            #     df["$VSTD5"]  = df["$volume"].rolling(5).std()  / (df["$volume"] + eps)
            #     df["$VSTD20"] = df["$volume"].rolling(20).std() / (df["$volume"] + eps)

            #     # ========== CORR (10/20/60) ==========
            #     logv = np.log(df["$volume"] + 1)
            #     df["$CORR10"] = df["$close"].rolling(10).corr(logv)
            #     df["$CORR20"] = df["$close"].rolling(20).corr(logv)
            #     df["$CORR60"] = df["$close"].rolling(60).corr(logv)

            #     # ========== CORD (price ratio vs volume ratio, 10/20/60) ==========
            #     pr = df["$close"] / df["$close"].shift(1)  # 与你原实现保持一致
            #     vr = np.log(df["$volume"] / df["$volume"].shift(1) + 1)
            #     df["$CORD10"] = pr.rolling(10).corr(vr)
            #     df["$CORD20"] = pr.rolling(20).corr(vr)
            #     df["$CORD60"] = pr.rolling(60).corr(vr)

            #     # ========== WVMA (5/20/60) ==========
            #     # 你当前这个写法更像“(价量活动项)的 std/mean 比率”，与 YAML 中 WVMA 的用法一致，就保持该口径并补齐 20
            #     act = (df["$close"] / df["$close"].shift(1) - 1).abs() * df["$volume"]
            #     for w in (5, 20, 60):
            #         df[f"$WVMA{w}"] = act.rolling(w).std() / (act.rolling(w).mean() + eps)

            #     # ========== Loop-1 derived factors (write back as columns) ==========
            #     # 1) OIMOM20_vol_scaled_VSTD5 = OIMOM20 / (1 + VSTD5)
            #     df["$OIMOM20_vol_scaled_VSTD5"] = df["$OIMOM20"] / (1.0 + df["$VSTD5"])

            #     # 2) PMOM5_vol_scaled_STD5 = PMOM5 / (1 + STD5)
            #     df["$PMOM5_vol_scaled_STD5"] = df["$PMOM5"] / (1.0 + df["$STD5"])

            #     # 3) CORR20_vol_scaled_WVMA5 = CORR20 / (1 + WVMA5)
            #     df["$CORR20_vol_scaled_WVMA5"] = df["$CORR20"] / (1.0 + df["$WVMA5"])

            #     # 4) KLOW_OI_pressure = KLOW / (1 + open_interest)
            #     df["$KLOW_OI_pressure"] = df["$KLOW"] / (1.0 + df["$open_interest"])

            #     # 5) VMOM5_tanh_squash = tanh(gamma_5 * VMOM5 / (1 + VSTD5)), gamma_5 = 1.0
            #     gamma_5 = 1.0
            #     df["$VMOM5_tanh_squash"] = np.tanh(gamma_5 * (df["$VMOM5"] / (1.0 + df["$VSTD5"])))
                
            #     # 回写到 daily_pv.h5
            #     df.to_hdf(h5_path, key="data", mode="w")

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            if self.target_task.version == 1:
                execution_code_path = code_path
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            try:
                subprocess.check_output(
                    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                )
                execution_success = True
            except subprocess.CalledProcessError as e:
                import site

                execution_feedback = (
                    e.output.decode()
                    .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)

            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
