# rdagent/scenarios/qlib/strategy/single_asset.py

from typing import DefaultDict, Dict, Union
import numpy as np
from collections import defaultdict, deque
import pandas as pd

from qlib.contrib.strategy import WeightStrategyBase
from qlib.backtest.position import Position

'''
在SignalRecord的基础上,提供了一个单标的序列的可选回测策略
虽然我们不太关注回测策略，但至少需要一个可用的策略
目前使用的是 SingleAssetSmoothQuantileStrategy ，基于分位数的平滑映射策略。目前这个策略的 return 变化，和 signal 的 IC 变化呈现一致性，说明策略是合理的
'''


class SingleAssetThresholdStrategy(WeightStrategyBase):
    """
    单资产阈值策略（适用于 HP_CME_SOY 这种单标的期货）：

    - 输入是模型预测的 signal（一般就是 <PRED> 那一列）
    - 对每个交易日、每个 instrument，根据 signal 和阈值决定仓位

    逻辑（按单标的来理解）：
      - signal > long_threshold  →  持有 long_weight（例如 1.0，代表满仓）
      - allow_short = True 且 signal < short_threshold →  持有 short_weight（例如 -1.0）
      - 其他情况 →  flat_weight（例如 0.0，空仓）

    注意：
      - 继承自 WeightStrategyBase，因此只需要实现 generate_target_weight_position，
        返回一个 {instrument: target_weight} 的字典即可。
      - trade_start_time / trade_end_time 是 Qlib 标准接口要求的参数，这里不做特别使用，
        只为了兼容接口签名。
    """

    def __init__(
        self,
        *,
        signal: Union[pd.Series, pd.DataFrame] = None,
        long_threshold: float = 0.0,
        short_threshold: float = None,
        long_weight: float = 1.0,
        short_weight: float = -1.0,
        flat_weight: float = 0.0,
        allow_short: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        signal :
            Qlib 的预测信号，可以是 Signal 对象，也可以是 DataFrame/Series。
            在 YAML 里你依然会写 `signal: <PRED>`，由 Qlib 帮你转成 Signal。
        long_threshold : float
            做多阈值，signal 大于该值时做多。
        short_threshold : float
            做空阈值（仅在 allow_short=True 时使用），signal 小于该值时做空。
        long_weight : float
            做多时的目标权重，比如 1.0 表示 100% 仓位。
        short_weight : float
            做空时的目标权重，比如 -1.0 表示 -100% 仓位。
        flat_weight : float
            空仓时权重，默认 0.0。
        allow_short : bool
            是否允许做空，如果 False 就只有多 / 空仓。
        kwargs :
            透传给 WeightStrategyBase / BaseSignalStrategy 的其他参数（一般不用管）。
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.flat_weight = flat_weight
        self.allow_short = allow_short

        # 交给 WeightStrategyBase / BaseSignalStrategy 处理 signal 等通用逻辑
        super().__init__(signal=signal, **kwargs)

    def generate_target_weight_position(
        self,
        score: Union[pd.Series, pd.DataFrame],
        current: Position,
        trade_start_time,
        trade_end_time,
    ) -> Dict[str, float]:
        """
        核心逻辑：根据当期的预测 score 生成目标权重。

        Parameters
        ----------
        score :
            当前交易步长对应的预测信号，通常是：
            - pd.Series: index 是 instrument，value 是预测分数
            - 或 pd.DataFrame: index 是 instrument，包含 'score' 列
        current :
            当前持仓 Position（这里我们不显式用它，只做“目标仓位控制”，交给 Qlib 做换仓）。
        trade_start_time :
            当前交易步长的起始时间（Qlib 内部传入，单日频下可视为当日日期）。
        trade_end_time :
            当前交易步长的结束时间（这里不使用，仅为接口兼容）。

        Returns
        -------
        Dict[str, float]
            {instrument: target_weight} 的字典，权重之和通常不超过 1（不考虑现金）。
        """

        # 统一成 Series：index=instrument, value=float score
        if isinstance(score, pd.DataFrame):
            if "score" in score.columns:
                s = score["score"]
            else:
                # 如果只有 1 列，就直接用那一列；否则退化用第一列
                if score.shape[1] == 1:
                    s = score.iloc[:, 0]
                else:
                    s = score.iloc[:, 0]
        else:
            s = score

        s = s.astype(float)

        target: Dict[str, float] = {}

        for inst, val in s.items():
            w = self.flat_weight

            # 做多逻辑
            if val > self.long_threshold:
                w = self.long_weight
            # 做空逻辑（可选）
            elif self.allow_short and self.short_threshold is not None and val < self.short_threshold:
                w = self.short_weight
            # 否则保持 flat_weight

            target[inst] = w

        return target



class SingleAssetSmoothQuantileStrategy(WeightStrategyBase):
    """
    单资产平滑分位数策略（适用于 HP_CME_SOY 等单标期货）
    发生在 Qlib 回测 / 交易阶段（WeightStrategyBase.generate_target_weight_position）：

    功能概述：
    - 步骤：
        1) 对 signal 做时间序列平滑（滚动均值），降低日度噪声 → 得到 smooth_signal_t
        2) 用过去一段历史的 smooth_signal，按分位数估计动态阈值：
              - low_quantile（例如 0.3）
              - high_quantile（例如 0.7）
        3) 将当前 smooth_signal_t 按阈值带做连续单调映射到权重：
              - 超过 high_quantile → 正权重（做多），权重随 signal 单调增
              - 低于 low_quantile → 负权重（做空，可选），权重随 signal 单调减
              - 中间区域 → 接近 0（空仓或轻仓）
           映射函数使用 tanh，保证权重平滑、连续。

    重要特性：
    - 分位数按“历史数据”动态估计，只用过去信息，不看未来（避免泄露）。
    - 支持 allow_short=True/False：
        - False：只在高分位区间做多，其余时间仓位接近 0；
        - True：高分位做多、低分位做空，中间区域接近 0。
    """

    def __init__(
        self,
        *,
        signal: Union[pd.Series, pd.DataFrame] = None,
        # ===== 分位数相关参数 =====
        high_quantile: float = 0.7,
        low_quantile: float = 0.3,
        quantile_history_window: int = 60,   # 计算分位数的回看窗口（天数）
        min_history: int = 30,               # 至少多少天历史之后才启用分位数逻辑

        # ===== 信号平滑参数 =====
        smooth_window: int = 3,              # 对原始 signal 做 rolling mean 的窗口

        # ===== 权重映射参数 =====
        max_long_weight: float = 1.0,        # 最大多头仓位（绝对值）
        max_short_weight: float = -1.0,      # 最大空头仓位（绝对值）
        tanh_scale: float = 3.0,             # tanh 的缩放系数，控制斜率

        # ===== 做空开关 =====
        allow_short: bool = False,

        **kwargs,
    ):
        """
        参数说明
        ----------
        signal :
            Qlib 的预测信号（一般在 YAML 中写 `signal: <PRED>`，由 Qlib 注入）。
        high_quantile / low_quantile :
            用于标定“高 / 低信号”的分位数，例如 0.7 / 0.3。
            注意：在单资产场景下，是基于时间维度（过去 N 日）的分布计算。
        quantile_history_window :
            计算分位数时使用的历史长度，太短会不稳定，太长会滞后。
        min_history :
            至少积累多少天历史后，才启用分位数逻辑；之前可以退化为简单 tanh 映射。
        smooth_window :
            对原始 signal 做滚动均值的窗口（越大越平滑，但反应越慢）。
        max_long_weight / max_short_weight :
            多 / 空方向的权重上限。例如 1.0 / -1.0 代表满仓。
        tanh_scale :
            控制 tanh(k * x) 中的 k，越大越接近“阶跃”，越小越平滑。
        allow_short :
            是否允许做空（即权重为负）。如果 False，则权重下界会被截为 0。
        """
        super().__init__(signal=signal, **kwargs)

        # 存参数
        self.high_quantile = high_quantile
        self.low_quantile = low_quantile
        self.quantile_history_window = quantile_history_window
        self.min_history = min_history

        self.smooth_window = smooth_window
        self.max_long_weight = max_long_weight
        self.max_short_weight = max_short_weight
        self.tanh_scale = tanh_scale
        self.allow_short = allow_short

        # 为每个 instrument 维护：
        # 1) 原始 signal 的历史（用于平滑）
        # 2) 平滑后的 signal 历史（用于计算分位数）
        self._raw_hist: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=max(self.quantile_history_window, self.smooth_window) + 10)
        )
        self._smooth_hist: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.quantile_history_window + 10)
        )

    def _smooth_signal(self, inst: str, new_val: float) -> float:
        """
        对某个 instrument 的最新 signal 做滚动平滑，返回平滑后的值。
        使用简单滚动均值。
        """
        raw_deque = self._raw_hist[inst]
        raw_deque.append(float(new_val))

        window = min(self.smooth_window, len(raw_deque))
        smooth_val = float(np.mean(list(raw_deque)[-window:]))

        self._smooth_hist[inst].append(smooth_val)

        return smooth_val

    def _get_quantile_thresholds(self, inst: str):
        """
        基于历史的平滑信号，计算高低分位数阈值。
        只用过去的数据（不含当前刚加入的那一个），避免“看未来”。
        历史不足时返回 None，让外层退化为简单映射。
        """
        smooth_hist = self._smooth_hist[inst]

        if len(smooth_hist) <= self.min_history:
            return None, None

        history = list(smooth_hist)[:-1]
        if len(history) > self.quantile_history_window:
            history = history[-self.quantile_history_window:]

        if len(history) == 0:
            return None, None

        arr = np.asarray(history, dtype=float)
        try:
            q_low = float(np.quantile(arr, self.low_quantile))
            q_high = float(np.quantile(arr, self.high_quantile))
        except Exception:
            return None, None

        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
            return None, None

        return q_low, q_high

    def _map_signal_to_weight(self, inst: str, val: float) -> float:
        """
        将当前的原始 signal 值映射为目标权重：
          1) 先做平滑，得到 smooth_val；
          2) 再用历史平滑值的高/低分位数估计阈值；
          3) 最后用一个连续单调的 tanh 函数映射到 [-1, 1] 再缩放到权重上限。
        """
        # 第一步：平滑当前 signal
        smooth_val = self._smooth_signal(inst, val)

        # 第二步：分位数阈值
        q_low, q_high = self._get_quantile_thresholds(inst)

        # 映射到权重空间的中间值 w ∈ (-1, 1)
        if q_low is None or q_high is None:
            z = self.tanh_scale * smooth_val
            w = np.tanh(z)
        else:
            mid = 0.5 * (q_low + q_high)
            scale = max(q_high - q_low, 1e-6)

            z = (smooth_val - mid) / scale
            w = np.tanh(self.tanh_scale * z)

        # 第三步：将 w ∈ (-1, 1) 映射到实际仓位
        if self.allow_short:
            # 允许做空：对称缩放到 [max_short_weight, max_long_weight]
            if w >= 0:
                weight = w * float(self.max_long_weight)
            else:
                weight = -w * float(abs(self.max_short_weight))
        else:
            # 不允许做空：负值截断为 0，只保留正方向
            w = max(0.0, w)
            weight = w * float(self.max_long_weight)

        return float(weight)

    # ================== WeightStrategyBase 要求实现的核心方法 ==================

    def generate_target_weight_position(
        self,
        score: Union[pd.Series, pd.DataFrame],
        current: Position,
        trade_start_time,
        trade_end_time,
    ) -> Dict[str, float]:
        """
        根据当前步长的预测 score 生成目标权重。

        Parameters
        ----------
        score :
            当前时点的预测信号：
              - pd.Series: index 是 instrument，value 是预测分数
              - pd.DataFrame: index 是 instrument，包含 'score' 列或单列
        current :
            当前持仓 Position（此处不显式使用，交给 Qlib 根据 target_weight 做调仓）。
        trade_start_time / trade_end_time :
            Qlib 标准接口参数，这里只用来保证签名兼容。

        Returns
        -------
        Dict[str, float]
            {instrument: target_weight} 的字典。
        """

        # 统一 score 成 Series：index=instrument, value=float score
        if isinstance(score, pd.DataFrame):
            if "score" in score.columns:
                s = score["score"]
            else:
                if score.shape[1] == 1:
                    s = score.iloc[:, 0]
                else:
                    s = score.iloc[:, 0]
        else:
            s = score

        s = s.astype(float)
        target: Dict[str, float] = {}

        for inst, val in s.items():
            w = self._map_signal_to_weight(inst, val)
            target[inst] = w

        return target