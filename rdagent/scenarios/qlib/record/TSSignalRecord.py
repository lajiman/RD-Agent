import warnings
import logging
import pandas as pd
import numpy as np
from tqdm import trange
from pprint import pprint
from typing import Union, List, Optional, Dict

# qlib evaluation helpers
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return
from qlib.workflow.record_temp import SignalRecord, ACRecordTemp
from qlib.log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)

class TSSigAnaRecord(ACRecordTemp):
    """
    发生在Qlib模型训练后，记录训练结果的阶段.
    用于处理单标的主力连续模型的结果分析（界面 -> 时序），这样UI的展示才不会出错.
    与 RD-Agent/rdagent/scenarios/qlib/model/logging_lgbm.py 中的 LGBModelWithFeatLog 不同的是，LGBModelWithFeatLog处理单因子的IC，
    TSSigAnaRecord处理模型预测值（signal）的IC；发生的阶段也不同，前者在模型训练阶段，后者在模型训练后.
    继承普通 ACRecordTemp 类.
    """

    artifact_path = "sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, ana_long_short=False, ann_scaler=252, label_col=0, skip_existing=False):
        super().__init__(recorder=recorder, skip_existing=skip_existing)
        self.ana_long_short = ana_long_short
        self.ann_scaler = ann_scaler
        self.label_col = label_col

    # def _generate(self, label: Optional[pd.DataFrame] = None, **kwargs):
    #     """
    #     Parameters
    #     ----------
    #     label : Optional[pd.DataFrame]
    #         Label should be a dataframe.
    #     """
    #     pred = self.load("pred.pkl")
    #     if label is None:
    #         label = self.load("label.pkl")
    #     if label is None or not isinstance(label, pd.DataFrame) or label.empty:
    #         logger.warning(f"Empty label.")
    #         return
    #     ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, self.label_col])
    #     metrics = {
    #         "IC": ic.mean(),
    #         "ICIR": ic.mean() / ic.std(),
    #         "Rank IC": ric.mean(),
    #         "Rank ICIR": ric.mean() / ric.std(),
    #     }
    #     objects = {"ic.pkl": ic, "ric.pkl": ric}
    #     if self.ana_long_short:
    #         long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, self.label_col])
    #         metrics.update(
    #             {
    #                 "Long-Short Ann Return": long_short_r.mean() * self.ann_scaler,
    #                 "Long-Short Ann Sharpe": long_short_r.mean() / long_short_r.std() * self.ann_scaler**0.5,
    #                 "Long-Avg Ann Return": long_avg_r.mean() * self.ann_scaler,
    #                 "Long-Avg Ann Sharpe": long_avg_r.mean() / long_avg_r.std() * self.ann_scaler**0.5,
    #             }
    #         )
    #         objects.update(
    #             {
    #                 "long_short_r.pkl": long_short_r,
    #                 "long_avg_r.pkl": long_avg_r,
    #             }
    #         )
    #     self.recorder.log_metrics(**metrics)
    #     pprint(metrics)
    #     return objects

    # 重写 _generate 方法，支持单标的时间序列 IC 计算
    def _generate(self, label: Optional[pd.DataFrame] = None, **kwargs):
        pred = self.load("pred.pkl")
        if label is None:
            label = self.load("label.pkl")
        if label is None or not isinstance(label, pd.DataFrame) or label.empty:
            logger.warning("Empty label.")
            return

        pred_s = pred.iloc[:, 0]
        label_s = label.iloc[:, self.label_col]

        # ===== 1) 先尝试正常的“截面 IC” =====
        ic, ric = calc_ic(pred_s, label_s)
        ic_valid = ic.dropna()
        ric_valid = ric.dropna()

        # 判断截面 IC 是否“退化”（几乎没有有效截面）
        use_ts_ic = (len(ic_valid) < 2) or (len(ric_valid) < 2)

        if not use_ts_ic:
            # ===== 多标的 / 截面样本足够：沿用原来的逻辑 =====
            metrics = {
                "IC": ic_valid.mean(),
                "ICIR": ic_valid.mean() / ic_valid.std(),
                "Rank IC": ric_valid.mean(),
                "Rank ICIR": ric_valid.mean() / ric_valid.std(),
            }
            objects = {"ic.pkl": ic, "ric.pkl": ric}

            # 截面 IC 情况下，长多/多空分析仍然有意义
            if self.ana_long_short:
                long_short_r, long_avg_r = calc_long_short_return(pred_s, label_s)
                metrics.update(
                    {
                        "Long-Short Ann Return": long_short_r.mean() * self.ann_scaler,
                        "Long-Short Ann Sharpe": long_short_r.mean() / long_short_r.std()
                        * self.ann_scaler ** 0.5,
                        "Long-Avg Ann Return": long_avg_r.mean() * self.ann_scaler,
                        "Long-Avg Ann Sharpe": long_avg_r.mean() / long_avg_r.std()
                        * self.ann_scaler ** 0.5,
                    }
                )
                objects.update(
                    {
                        "long_short_r.pkl": long_short_r,
                        "long_avg_r.pkl": long_avg_r,
                    }
                )

        # 只触发这条分支
        else:
            # ===== 2) 截面退化（典型单标场景）：fallback 为时间序列 IC =====
            logger.warning(
                "Cross-sectional IC is degenerate (likely single-asset or too few instruments per date). "
                "Falling back to time-series IC over time."
            )

            # pred_ts / label_ts：按时间对齐的单标时间序列
            # 这里AI做了冗余处理，实际有效的是“pred_ts, label_ts = pred_s.align(label_s, join="inner")”，和 RD-Agent/rdagent/scenarios/qlib/model/logging_lgbm.py 的思路一致
            idx = pred_s.index
            if isinstance(idx, pd.MultiIndex) and "datetime" in idx.names:
                # 多层索引（datetime, instrument），只保留按 datetime 聚合后的单标时间序列
                pred_ts = pred_s.groupby(level="datetime").mean()
                label_ts = label_s.groupby(level="datetime").mean()
            else:
                # 普通索引：直接按 index 对齐
                pred_ts, label_ts = pred_s.align(label_s, join="inner")

            # 计算 IC / RIC
            ts_ic = float(pred_ts.corr(label_ts)) if len(pred_ts) > 1 else np.nan
            ts_ric = float(pred_ts.corr(label_ts, method="spearman")) if len(pred_ts) > 1 else np.nan

            # ICIR / Rank ICIR 在“时间序列整体一个点”的意义下不再合理，设为 NaN
            metrics = {
                "IC": ts_ic,
                "ICIR": np.nan,
                "Rank IC": ts_ric,
                "Rank ICIR": np.nan,
            }

            # 保留原始 ic/ric（尽管可能全是 NaN），额外存 ts_ic/ts_ric
            objects = {
                "ic.pkl": ic,
                "ric.pkl": ric,
                "ts_ic.pkl": ts_ic,
                "ts_ric.pkl": ts_ric,
            }

            # 单标时间序列下，“长多/多空分组”没有截面意义，这里直接跳过 long_short 分析
            # 如果你未来有专门的单标多空逻辑，可以在这里另行实现

        # 统一记录 & 返回
        self.recorder.log_metrics(**metrics)
        pprint(metrics)
        return objects

    def list(self):
        paths = ["ic.pkl", "ric.pkl"]
        if self.ana_long_short:
            paths.extend(["long_short_r.pkl", "long_avg_r.pkl"])
        return paths



if __name__ == "__main__":
    print(2)