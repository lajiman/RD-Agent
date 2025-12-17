# import json
# from pathlib import Path
# from typing import Dict

# import pandas as pd

# from rdagent.core.experiment import Experiment
# from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
# from rdagent.log import rdagent_logger as logger
# from rdagent.oai.llm_utils import APIBackend
# from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
# from rdagent.utils import convert2bool
# from rdagent.utils.agent.tpl import T

# DIRNAME = Path(__file__).absolute().resolve().parent

# IMPORTANT_METRICS = [
#     "IC",
#     "1day.excess_return_with_cost.annualized_return",
#     "1day.excess_return_with_cost.max_drawdown",
# ]


# def process_results(current_result, sota_result):
#     # Convert the results to dataframes
#     current_df = pd.DataFrame(current_result)
#     sota_df = pd.DataFrame(sota_result)

#     # Set the metric as the index
#     current_df.index.name = "metric"
#     sota_df.index.name = "metric"

#     # Rename the value column to reflect the result type
#     current_df.rename(columns={"0": "Current Result"}, inplace=True)
#     sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

#     # Combine the dataframes on the Metric index
#     combined_df = pd.concat([current_df, sota_df], axis=1)

#     # Filter the combined DataFrame to retain only the important metrics
#     filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

#     def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
#         results = []
#         for metric, row in filtered_combined_df.iterrows():
#             current = row["Current Result"]
#             sota = row["SOTA Result"]
#             results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
#         return "; ".join(results)

#     return format_filtered_combined_df(filtered_combined_df)


# class QlibFactorExperiment2Feedback(Experiment2Feedback):
#     def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
#         """
#         Generate feedback for the given experiment and hypothesis.

#         Args:
#             exp (QlibFactorExperiment): The experiment to generate feedback for.
#             hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
#             trace (Trace): The trace of the experiment.

#         Returns:
#             Any: The feedback generated for the given experiment and hypothesis.
#         """
#         hypothesis = exp.hypothesis
#         logger.info("Generating feedback...")
#         hypothesis_text = hypothesis.hypothesis
#         current_result = exp.result
#         tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
#         sota_result = exp.based_experiments[-1].result

#         # Process the results to filter important metrics
#         combined_result = process_results(current_result, sota_result)

#         # Generate the system prompt
#         if isinstance(self.scen, QlibQuantScenario):
#             sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
#                 scenario=self.scen.get_scenario_all_desc(action="factor")
#             )
#         else:
#             sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
#                 scenario=self.scen.get_scenario_all_desc()
#             )

#         # Generate the user prompt
#         usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
#             hypothesis_text=hypothesis_text,
#             task_details=tasks_factors,
#             combined_result=combined_result,
#         )

#         # Call the APIBackend to generate the response for hypothesis feedback
#         response = APIBackend().build_messages_and_create_chat_completion(
#             user_prompt=usr_prompt,
#             system_prompt=sys_prompt,
#             json_mode=True,
#             json_target_type=Dict[str, str | bool | int],
#         )

#         # Parse the JSON response to extract the feedback
#         response_json = json.loads(response)

#         # Extract fields from JSON response
#         observations = response_json.get("Observations", "No observations provided")
#         hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
#         new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
#         reason = response_json.get("Reasoning", "No reasoning provided")
#         decision = convert2bool(response_json.get("Replace Best Result", "no"))

#         return HypothesisFeedback(
#             observations=observations,
#             hypothesis_evaluation=hypothesis_evaluation,
#             new_hypothesis=new_hypothesis,
#             reason=reason,
#             decision=decision,
#         )


import json
import os   # <-- 新增
from pathlib import Path
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

DIRNAME = Path(__file__).absolute().resolve().parent

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]

QLIB_BUILTIN_FACTORS = {
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
    "PMOM5", "PMOM10", "PMOM20",
    "VMOM5", "OIMOM5", "OIMOM20",
}

def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Set the metric as the index
    current_df.index.name = "metric"
    sota_df.index.name = "metric"

    # Rename the value column to reflect the result type
    current_df.rename(columns={"0": "Current Result"}, inplace=True)
    sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)

    # Filter the combined DataFrame to retain only the important metrics
    filtered_combined_df = combined_df.loc[IMPORTANT_METRICS]

    def format_filtered_combined_df(filtered_combined_df: pd.DataFrame) -> str:
        results = []
        for metric, row in filtered_combined_df.iterrows():
            current = row["Current Result"]
            sota = row["SOTA Result"]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        return "; ".join(results)

    return format_filtered_combined_df(filtered_combined_df)


class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Process the results to filter important metrics
        combined_result = process_results(current_result, sota_result)

        # ====== 新增：把单因子 IC 注入 combined_result，但过滤掉 Qlib 自带因子 ======
        try:
            ws = getattr(exp, "experiment_workspace", None)
            exp_dir = getattr(ws, "workspace_path", None) if ws is not None else None

            # 兼容 Path / str
            if exp_dir is not None and hasattr(exp_dir, "__fspath__"):
                exp_dir = str(exp_dir)

            ic_path = None
            if isinstance(exp_dir, str):
                # 在 workspace 下递归寻找 single_factor_ic_train*.json
                candidate_names = [
                    "single_factor_ic_train.json",
                    "single_factor_ic_train_timeseries.json",
                ]
                for root, _, files in os.walk(exp_dir):
                    if ic_path is not None:
                        break
                    for fname in candidate_names:
                        if fname in files:
                            ic_path = os.path.join(root, fname)
                            break

            if ic_path is not None:
                with open(ic_path, "r", encoding="utf-8") as f:
                    ic_data = json.load(f)

                if isinstance(ic_data, dict):
                    filtered_ic = {
                        name: stats
                        for name, stats in ic_data.items()
                        if name not in QLIB_BUILTIN_FACTORS
                    }
                else:
                    filtered_ic = {}

                if filtered_ic:
                    ic_json_str = json.dumps(filtered_ic, ensure_ascii=False)
                    combined_result = (
                        combined_result
                        + "\n\n"
                        + "Per-factor IC (train segment) for non-built-in factors (JSON):\n"
                        + ic_json_str
                    )
        except Exception as e:
            # 注入失败不影响主流程，只简单记录错误
            logger.error(
                f"Failed to inject per-factor IC JSON into combined_result: {repr(e)}"
            )
        # ====== 新增部分结束 ======

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="factor")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        usr_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)

        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )


class QlibModelExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibModelExperiment): The experiment to generate feedback for.
            hypothesis (QlibModelHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            HypothesisFeedback: The feedback generated for the given experiment and hypothesis.
        """
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")

        # Generate the system prompt
        if isinstance(self.scen, QlibQuantScenario):
            sys_prompt = T("scenarios.qlib.prompts:model_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc(action="model")
            )
        else:
            sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
                scenario=self.scen.get_scenario_all_desc()
            )

        # Generate the user prompt
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()
        user_prompt = T("scenarios.qlib.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result.loc[IMPORTANT_METRICS] if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result.loc[IMPORTANT_METRICS] if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response)

        # Call the APIBackend to generate the response for hypothesis feedback
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
