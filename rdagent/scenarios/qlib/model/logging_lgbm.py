# src/qlib_models/logging_lgbm.py
import os
import json
import numpy as np
import pandas as pd

from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R


def _compute_timeseries_ic(X_model, y_train, label_col):
    """
    时间序列 IC
    """
    import pandas as pd

    ic_result = {}

    # 保证是 Series（如果 y_train 是 DataFrame）
    if isinstance(y_train, pd.DataFrame):
        y = y_train[label_col]
    else:
        y = y_train

    for fac in X_model.columns:
        x = X_model[fac]

        # 按 index 对齐，同时去掉NAN。比方说，一个因子需要前60天数据，那它的NAN会比label多
        pair = pd.concat([x, y], axis=1, join="inner").dropna()
        if pair.shape[0] < 2:
            continue
        
        # 计算 Spearman IC
        ic = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        if pd.isna(ic):
            continue

        ic_result[str(fac)] = {
            "ic": float(ic),
            "n_obs": int(pair.shape[0]),
        }

    return ic_result


class LGBModelWithFeatLog(LGBModel):
    """
    发生在Qlib模型训练阶段
    用于记录训练过程中特征的IC值和原始特征样本，方便debug，和将IC值注入到feedback中
    继承普通 LGBModel 基础上，额外记录：
    1. 每个因子的单因子 IC（基于 train 段，模型视角的 feature + label）
    2. 一份 train 段 raw feature 的样本（handler.fetch），仅用于排查

    所有输出都写在当前 experiment 的 local_dir（RD-Agent_workspace 内，可以在UI中看到每次实验workspace的地址），包括：
    - <exp_dir>/single_factor_ic_train.json
    - <exp_dir>/features_debug/train_features_fetch_raw_sample.csv
    """

    # 重写 fit 方法。未来如果不用 LGBModel，可以把这个类拆出来单独用
    def fit(self, dataset, **kwargs):
        handler = dataset.handler

        # ========= 1) 取 train 段定义（用于 fetch raw） =========
        df_feat_raw = None
        train_seg = None
        try:
            segs = getattr(dataset, "segments", None)
            if isinstance(segs, dict):
                train_seg = segs.get("train")
            if train_seg is not None:
                start, end = train_seg
                df_feat_raw = handler.fetch(
                    slice(pd.Timestamp(start), pd.Timestamp(end)),
                    col_set="feature",
                )
        except Exception as e:
            print(f"[LGBModelWithFeatLog] failed to fetch train features (raw): {e}")
            df_feat_raw = None

        # ========= 2) 用 dataset.prepare 拿模型视角的 feature + label =========
        X_model, y_train = None, None
        try:
            prepared = dataset.prepare("train", col_set=["feature", "label"])
            if isinstance(prepared, dict):
                X_model = prepared.get("feature")
                y_train = prepared.get("label")

            # 事实上会触发这个分支，数据如：
            # [LGBModelWithFeatLog] prepared.columns: MultiIndex([
            #             ('feature', 'CORR20_VOLGATE_price_volume_regime'),
            #             ('feature',             'KLOW_OI_meanrev_capped'),
            #             ('feature',  'OIMOM20_PMOM10_cross_stable_trend'),
            #             ('feature',       'PMOM10_VOLCAP_gated_momentum'),
            #             ('feature',                       '_DUMMY_CLOSE'),
            #             (  'label',                             'LABEL0')
            #         ], )
            # 这样数据的处理过程就明了了
            elif isinstance(prepared, pd.DataFrame):
                # 当前这种情况：一个 DataFrame，columns 是 MultiIndex('feature'/'label', 因子名)
                cols = prepared.columns
                print("[LGBModelWithFeatLog] prepared.columns:", cols)

                if isinstance(cols, pd.MultiIndex) and "feature" in cols.get_level_values(0):
                    # 拆出特征和标签
                    if ("label" in cols.get_level_values(0)):
                        X_model = prepared["feature"]
                        y_train = prepared["label"]
                    else:
                        # 只有 feature，没有 label
                        X_model = prepared["feature"]
                        y_train = None
                else:
                    # 没有 MultiIndex 或没有 'feature' 这一层，当成纯特征
                    X_model = prepared
                    y_train = None
            else:
                X_model = prepared
                y_train = None

            # 收集到的数据：
            # [LGBModelWithFeatLog] dataset.prepare('train', ['feature','label'])
            #                     CORR20_VOLGATE_price_volume_regime  ...  _DUMMY_CLOSE
            # datetime   instrument                                      ...              
            # 2003-01-02 HP_CME_SOY                           -0.085111  ...        571.50
            # 2003-01-03 HP_CME_SOY                           -0.030212  ...        568.75
            # 2003-01-06 HP_CME_SOY                            0.055399  ...        567.75
            # 2003-01-07 HP_CME_SOY                            0.100356  ...        570.50
            # 2003-01-08 HP_CME_SOY                            0.086335  ...        571.75

            # [5 rows x 5 columns]
            #                         LABEL0
            # datetime   instrument          
            # 2003-01-02 HP_CME_SOY -0.048352
            # 2003-01-03 HP_CME_SOY -0.045795
            # 2003-01-06 HP_CME_SOY -0.051709
            # 2003-01-07 HP_CME_SOY -0.042414
            # 2003-01-08 HP_CME_SOY -0.037719
            print(
                f"[LGBModelWithFeatLog] dataset.prepare('train', ['feature','label'])")
            print(X_model.head())
            print(y_train.head())
        except Exception as e:
            print(
                "[LGBModelWithFeatLog] dataset.prepare('train', ['feature','label']) failed: "
                f"{e}"
            )
            X_model, y_train = None, None

        # ========= 3) 写入 experiment 目录：raw sample + 单因子 IC =========
        try:
            rec = R.get_recorder()
            if rec is not None:
                exp_dir = rec.get_local_dir()
                os.makedirs(exp_dir, exist_ok=True)

                # ---- 3.1 保存 raw feature sample ----
                if df_feat_raw is not None:
                    debug_dir = os.path.join(exp_dir, "features_debug")
                    os.makedirs(debug_dir, exist_ok=True)

                    raw_to_save = df_feat_raw.reset_index()
                    # 这里选取5000是为了当时的debug需要：Start/end time incorrect
                    max_rows = 5000
                    if len(raw_to_save) > max_rows:
                        raw_to_save = raw_to_save.head(max_rows)

                    raw_path = os.path.join(
                        debug_dir, "train_features_fetch_raw_sample.csv"
                    )
                    raw_to_save.to_csv(raw_path, index=False)
                    print(
                        f"[LGBModelWithFeatLog] saved RAW feature sample to {raw_path}, "
                        f"shape={raw_to_save.shape}"
                    )
                else:
                    print(
                        "[LGBModelWithFeatLog] df_feat_raw is None, "
                        "skip saving RAW feature sample."
                    )

                # ---- 3.2 计算单因子 IC（时间序列 IC） ----
                if (X_model is not None) and (y_train is not None) and (not y_train.empty):
                    try:
                        if isinstance(y_train, pd.DataFrame):
                            label_col = y_train.columns[0]
                        else:
                            label_col = getattr(y_train, "name", "LABEL")

                        ic_result = _compute_timeseries_ic(X_model, y_train, label_col)

                        ic_path = os.path.join(exp_dir, "single_factor_ic_train_timeseries.json")
                        with open(ic_path, "w", encoding="utf-8") as f:
                            json.dump(ic_result, f, ensure_ascii=False, indent=2)

                        print(
                            f"[LGBModelWithFeatLog] saved TIME-SERIES single-factor IC to {ic_path} "
                            f"(factors with IC: {len(ic_result)})"
                        )
                    except Exception as e:
                        print(
                            f"[LGBModelWithFeatLog] failed to compute time-series single-factor IC: {e}"
                        )
                else:
                    print(
                        "[LGBModelWithFeatLog] skip single-factor IC: "
                        "X_model or y_train is None/empty."
                    )
        except Exception as e:
            print(f"[LGBModelWithFeatLog] failed in logging & IC part: {e}")

        # ========= 4) 正常训练 =========
        return super().fit(dataset, **kwargs)
