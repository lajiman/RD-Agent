import pandas as pd
from qlib.data.dataset.loader import StaticDataLoader

class AutoFeatureStaticDataLoader(StaticDataLoader):
    """
    Drop-in replacement for StaticDataLoader:
    - reads parquet (same as StaticDataLoader)
    - automatically uses all numeric columns as features except excluded ones
    """
    '''
    为了添加新的因子时，不需要每次都手动指定 feature 列。因此实现这个 AutoFeatureStaticDataLoader 类
    逻辑是：加载数据后，自动把所有数值型列作为 feature，排除掉 drop_cols 和 label_cols 指定的列
    '''
    def __init__(self, config, drop_cols=None, label_cols=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.drop_cols = set(drop_cols or [])
        self.label_cols = set(label_cols or [])

    def load(self, instruments=None, start_time=None, end_time=None):
        df = super().load(instruments=instruments, start_time=start_time, end_time=end_time)

        # infer features: all columns except drop/label
        cols = [c for c in df.columns if c not in self.drop_cols and c not in self.label_cols]

        # keep only numeric features
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        return df[numeric_cols]
