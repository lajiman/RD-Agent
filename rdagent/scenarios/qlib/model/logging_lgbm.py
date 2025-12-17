# # src/qlib_models/logging_lgbm.py
# import os
# import json
# import pandas as pd

# from qlib.contrib.model.gbdt import LGBModel
# from qlib.workflow import R


# class LGBModelWithFeatLog(LGBModel):
#     """
#     在普通 LGBModel 基础上，额外记录：
#     - 当前训练使用的 feature 列名（因子名）
#     每次 fit 时会写一个 features_used.json 到当前 experiment 的本地目录。
#     """

#     def fit(self, dataset, **kwargs):
#         # 1) 拿到 feature 列名
#         handler = dataset.handler

#         try:
#             # 优先用 handler.get_cols("feature")，更轻量
#             cols = handler.get_cols("feature")
#             # 有的版本返回 MultiIndex，这里统一转成字符串
#             if isinstance(cols, (list, tuple)):
#                 feature_names = [str(c) for c in cols]
#             else:
#                 # 兜底：取一段数据看 columns
#                 segs = dataset.segments
#                 train_seg = segs.get("train")
#                 if train_seg is not None:
#                     start, end = train_seg
#                     df_feat = handler.fetch(
#                         slice(pd.Timestamp(start), pd.Timestamp(end)),
#                         col_set="feature"
#                     )
#                     feature_names = [str(c) for c in df_feat.columns]
#                 else:
#                     feature_names = []
#         except Exception as e:
#             # 如果上面任一步出错，就记录空，并继续训练
#             feature_names = []
#             print(f"[LGBModelWithFeatLog] failed to get feature names: {e}")

#         # 2) 写到当前 experiment 的本地目录
#         try:
#             rec = R.get_recorder()
#             if rec is not None:
#                 local_dir = "/home/lujing/projects/RD-Agent"
#                 os.makedirs(local_dir, exist_ok=True)
#                 out_path = os.path.join(local_dir, "features_used.json")
                
#                 # 读取现有数据（如果存在），每个 loop 新增一条记录
#                 loop_records = []
#                 if os.path.exists(out_path):
#                     try:
#                         with open(out_path, "r", encoding="utf-8") as f:
#                             loop_records = json.load(f)
#                     except Exception:
#                         loop_records = []
                
#                 # 确保是列表类型
#                 if not isinstance(loop_records, list):
#                     loop_records = []
#                 loop_num = len(loop_records) + 1
#                 current_record = {
#                     "loop": loop_num,
#                     "features": feature_names
#                 }
#                 loop_records.append(current_record)
                
#                 with open(out_path, "w", encoding="utf-8") as f:
#                     json.dump(loop_records, f, ensure_ascii=False, indent=2)
#                 print(f"[LGBModelWithFeatLog] Loop {loop_num}: saved {len(feature_names)} features to {out_path}")
#             else:
#                 print("[LGBModelWithFeatLog] no recorder, skip feature logging.")
#         except Exception as e:
#             print(f"[LGBModelWithFeatLog] failed to save feature names: {e}")

#         # 3) 正常训练
#         return super().fit(dataset, **kwargs)



# # src/qlib_models/logging_lgbm.py
# import os
# import json
# import pandas as pd

# from qlib.contrib.model.gbdt import LGBModel
# from qlib.workflow import R


# class LGBModelWithFeatLog(LGBModel):
#     """
#     在普通 LGBModel 基础上，额外记录：
#     1. 当前训练使用的 feature 列名（以模型视角为准）
#     2. train 段的 raw 特征（handler.fetch）
#     3. train 段的 model 特征（dataset.prepare）
#     4. raw vs model 的对比样本（方便肉眼比较差异）

#     每次 fit 时会：
#     - 追加写入 features_used.json
#     - 在 features_debug/ 下写出：
#         loop_XXX_train_features_fetch_raw.csv
#         loop_XXX_train_features_prepare_model.csv
#         loop_XXX_train_features_compare_sample.csv
#     """

#     def fit(self, dataset, **kwargs):
#         handler = dataset.handler

#         # ========= 1) 拿到 train 段 raw feature（fetch） =========
#         df_feat_raw = None
#         train_seg = None
#         try:
#             segs = getattr(dataset, "segments", None)
#             if isinstance(segs, dict):
#                 train_seg = segs.get("train")
#             if train_seg is not None:
#                 start, end = train_seg
#                 df_feat_raw = handler.fetch(
#                     slice(pd.Timestamp(start), pd.Timestamp(end)),
#                     col_set="feature",
#                 )
#         except Exception as e:
#             print(f"[LGBModelWithFeatLog] failed to fetch train features (raw): {e}")
#             df_feat_raw = None

#         # ========= 2) 拿到 train 段 model 特征（prepare） =========
#         X_model = None
#         try:
#             prepared = dataset.prepare("train", col_set="feature")
#             # DatasetH 通常返回 dict: {"feature": DataFrame, "label": DataFrame}
#             if isinstance(prepared, dict):
#                 X_model = prepared.get("feature")
#             else:
#                 X_model = prepared
#         except Exception as e:
#             print(f"[LGBModelWithFeatLog] dataset.prepare('train','feature') failed: {e}")
#             X_model = None

#         # ========= 3) 获取 feature 名称（优先以模型视角为准） =========
#         feature_names = []
#         if X_model is not None:
#             feature_names = [str(c) for c in X_model.columns]
#         else:
#             # 如果模型视角拿不到，再退回 handler.get_cols / raw 的列名
#             try:
#                 cols = handler.get_cols("feature")
#                 if isinstance(cols, (list, tuple)):
#                     feature_names = [str(c) for c in cols]
#                 else:
#                     try:
#                         feature_names = [str(c) for c in list(cols)]
#                     except Exception:
#                         feature_names = [str(cols)]
#             except Exception as e:
#                 print(f"[LGBModelWithFeatLog] handler.get_cols('feature') failed: {e}")
#                 if df_feat_raw is not None:
#                     feature_names = [str(c) for c in df_feat_raw.columns]
#                 else:
#                     feature_names = []

#         # ========= 4) 写 features_used.json + 导出三份特征文件 =========
#         try:
#             rec = R.get_recorder()
#             if rec is not None:
#                 base_dir = "/home/lujing/projects/RD-Agent"
#                 os.makedirs(base_dir, exist_ok=True)

#                 # ---- 4.1 记录每个 loop 使用的因子名 ----
#                 json_path = os.path.join(base_dir, "features_used.json")
#                 loop_records = []
#                 if os.path.exists(json_path):
#                     try:
#                         with open(json_path, "r", encoding="utf-8") as f:
#                             loop_records = json.load(f)
#                     except Exception:
#                         loop_records = []

#                 if not isinstance(loop_records, list):
#                     loop_records = []

#                 loop_num = len(loop_records) + 1
#                 current_record = {
#                     "loop": loop_num,
#                     "features": feature_names,
#                 }
#                 loop_records.append(current_record)

#                 with open(json_path, "w", encoding="utf-8") as f:
#                     json.dump(loop_records, f, ensure_ascii=False, indent=2)

#                 print(
#                     f"[LGBModelWithFeatLog] Loop {loop_num}: "
#                     f"saved {len(feature_names)} feature names to {json_path}"
#                 )

#                 # ---- 4.2 导出 fetch(raw) & prepare(model) & 对比样本 ----
#                 debug_dir = os.path.join(base_dir, "features_debug")
#                 os.makedirs(debug_dir, exist_ok=True)

#                 # 4.2.1 raw：handler.fetch 的结果
#                 if df_feat_raw is not None:
#                     raw_to_save = df_feat_raw.reset_index()
#                     raw_path = os.path.join(
#                         debug_dir,
#                         f"loop_{loop_num:03d}_train_features_fetch_raw.csv",
#                     )
#                     raw_to_save.to_csv(raw_path, index=False)
#                     print(
#                         f"[LGBModelWithFeatLog] Loop {loop_num}: "
#                         f"saved RAW feature sample to {raw_path}, shape={raw_to_save.shape}"
#                     )
#                 else:
#                     print(
#                         "[LGBModelWithFeatLog] df_feat_raw is None, "
#                         "skip saving RAW feature values."
#                     )

#                 # 4.2.2 model：dataset.prepare 的结果
#                 if X_model is not None:
#                     model_to_save = X_model.reset_index()
#                     model_path = os.path.join(
#                         debug_dir,
#                         f"loop_{loop_num:03d}_train_features_prepare_model.csv",
#                     )
#                     model_to_save.to_csv(model_path, index=False)
#                     print(
#                         f"[LGBModelWithFeatLog] Loop {loop_num}: "
#                         f"saved MODEL feature sample to {model_path}, shape={model_to_save.shape}"
#                     )
#                 else:
#                     print(
#                         "[LGBModelWithFeatLog] X_model is None, "
#                         "skip saving MODEL feature values."
#                     )

#                 # 4.2.3 raw vs model 对比样本（方便肉眼对比差异）
#                 if (df_feat_raw is not None) and (X_model is not None):
#                     try:
#                         # 按 MultiIndex 对齐，两边都有的才比较
#                         df_compare = df_feat_raw.join(
#                             X_model,
#                             how="inner",
#                             lsuffix="_raw",
#                             rsuffix="_model",
#                         )

#                         # 先把索引变成列
#                         idx_names = list(df_compare.index.names)
#                         df_compare = df_compare.reset_index()

#                         # 按 base_name 把 *_raw 和 *_model 排在一起
#                         all_cols = list(df_compare.columns)
#                         # 索引列（一般是 instrument, datetime）
#                         index_cols = [c for c in all_cols if c in idx_names]

#                         # 特征列
#                         feat_cols = [c for c in all_cols if c not in index_cols]

#                         # 提取 base_name 顺序：先按 raw，再按 model，保持原始顺序去重
#                         raw_bases = [c[:-4] for c in feat_cols if c.endswith("_raw")]
#                         model_bases = [c[:-6] for c in feat_cols if c.endswith("_model")]

#                         base_order = []
#                         seen = set()
#                         for b in raw_bases + model_bases:
#                             if b not in seen:
#                                 seen.add(b)
#                                 base_order.append(b)

#                         # 构造新的列顺序：index 列 + 每个因子的 raw/model 成对出现
#                         new_cols = list(index_cols)
#                         for b in base_order:
#                             raw_name = f"{b}_raw"
#                             model_name = f"{b}_model"
#                             if raw_name in feat_cols:
#                                 new_cols.append(raw_name)
#                             if model_name in feat_cols:
#                                 new_cols.append(model_name)

#                         # 重新排列
#                         df_compare = df_compare[new_cols]

#                         # 为防止文件太大，这里只保留前若干行做示例
#                         max_rows = 5000
#                         if len(df_compare) > max_rows:
#                             df_compare = df_compare.head(max_rows)

#                         compare_path = os.path.join(
#                             debug_dir,
#                             f"loop_{loop_num:03d}_train_features_compare_sample.csv",
#                         )
#                         df_compare.to_csv(compare_path, index=False)
#                         print(
#                             f"[LGBModelWithFeatLog] Loop {loop_num}: "
#                             f"saved RAW vs MODEL compare sample to {compare_path}, "
#                             f"shape={df_compare.shape}"
#                         )
#                     except Exception as e:
#                         print(
#                             f"[LGBModelWithFeatLog] failed to build compare sample: {e}"
#                         )
#             else:
#                 print("[LGBModelWithFeatLog] no recorder, skip feature logging.")
#         except Exception as e:
#             print(f"[LGBModelWithFeatLog] failed to save feature logs: {e}")

#         # ========= 5) 正常训练 =========
#         return super().fit(dataset, **kwargs)


# src/qlib_models/logging_lgbm.py
import os
import json
import numpy as np
import pandas as pd

from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R


def _compute_timeseries_ic(X_model, y_train, label_col):
    """
    简单的时间序列 IC:
    对每个因子列和标签列在时间维度上计算 Spearman 相关系数
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

        # 按 index 对齐，同时去掉缺失
        pair = pd.concat([x, y], axis=1, join="inner").dropna()
        if pair.shape[0] < 2:
            continue

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
    在普通 LGBModel 基础上，额外记录：
    1. 每个因子的单因子 IC（基于 train 段，模型视角的 feature + label）
    2. 一份 train 段 raw feature 的样本（handler.fetch），仅用于排查

    所有输出都写在当前 experiment 的 local_dir（RD-Agent_workspace 内），包括：
    - <exp_dir>/single_factor_ic_train.json
    - <exp_dir>/features_debug/train_features_fetch_raw_sample.csv
    """

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

                # ---- 3.1 保存 raw feature sample（可选小样本） ----
                if df_feat_raw is not None:
                    debug_dir = os.path.join(exp_dir, "features_debug")
                    os.makedirs(debug_dir, exist_ok=True)

                    raw_to_save = df_feat_raw.reset_index()
                    # 防止太大，截一部分就够做 sanity check
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
                        # 取 label 的第一列作为目标
                        if isinstance(y_train, pd.DataFrame):
                            label_col = y_train.columns[0]
                        else:
                            # 如果是 Series，就给个名字
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
