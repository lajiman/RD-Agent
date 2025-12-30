# RD-Agent/rdagent/scenarios/qlib/experiment/factor_data_template/enrich_daily_py.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd


'''
这段代码与 RD-Agent/rdagent/scenarios/qlib/experiment/factor_data_template/generate.py 配合使用，用于生成“高级因子”
代码的逻辑很简单，相对解耦。想要什么因子，就添加相应的计算逻辑，最后被 enrich_daily_pv_h5 函数调用
'''

EPS = 1e-12


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (datetime, instrument) or (instrument, datetime).")
    if "datetime" not in df.index.names or "instrument" not in df.index.names:
        raise ValueError(f"Index names must include ['datetime','instrument'], got {df.index.names}")
    return df.sort_index()


def _groupby_instrument(df: pd.DataFrame):
    return df.groupby(level="instrument", group_keys=False)


def _resi_series(x: np.ndarray) -> float:
    n = len(x)
    t = np.arange(n, dtype=float)
    coef = np.polyfit(t, x, 1)
    pred = coef[0] * t + coef[1]
    return float((x[-1] - pred[-1]) / (x[-1] + EPS))


def _rsq_series(x: np.ndarray) -> float:
    n = len(x)
    t = np.arange(n, dtype=float)
    coef = np.polyfit(t, x, 1)
    pred = coef[0] * t + coef[1]
    ss_res = np.sum((x - pred) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2) + EPS
    return float(1.0 - ss_res / ss_tot)


def _rolling_apply(g: pd.Series, window: int, fn) -> pd.Series:
    return g.rolling(window).apply(fn, raw=True)


def _enrich_df_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must contain at least:
      $open, $high, $low, $close, $volume, $open_interest
    and MultiIndex with levels: datetime, instrument
    """
    df = _ensure_sorted(df).copy()

    required = ["$open", "$high", "$low", "$close", "$volume", "$open_interest"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    gb = _groupby_instrument(df)

    # ===== Level-0 primitives =====

    # RESI10/20
    df["$RESI10"] = gb["$close"].apply(lambda s: _rolling_apply(s, 10, _resi_series))
    df["$RESI20"] = gb["$close"].apply(lambda s: _rolling_apply(s, 20, _resi_series))

    # RSQR 5/10/20/60
    for w in (5, 10, 20, 60):
        df[f"$RSQR{w}"] = gb["$close"].apply(lambda s, w=w: _rolling_apply(s, w, _rsq_series))

    # KLEN / KLOW
    df["$KLEN"] = (df["$high"] - df["$low"]) / (df["$open"] + EPS)
    df["$KLOW"] = (np.minimum(df["$open"], df["$close"]) - df["$low"]) / (df["$open"] + EPS)

    # PMOM 5/10/20/60
    for w in (5, 10, 20, 60):
        df[f"$PMOM{w}"] = gb["$close"].apply(lambda s, w=w: s / s.shift(w) - 1.0)

    # VMOM 5/10/20/60
    for w in (5, 10, 20, 60):
        df[f"$VMOM{w}"] = gb["$volume"].apply(lambda s, w=w: s / s.shift(w) - 1.0)

    # OIMOM 5/10/20/60
    for w in (5, 10, 20, 60):
        df[f"$OIMOM{w}"] = gb["$open_interest"].apply(lambda s, w=w: s / s.shift(w) - 1.0)

    # STD 5/10/20/60 (keep same convention as your YAML: Std($close,w)/$close)
    for w in (5, 10, 20, 60):
        df[f"$STD{w}"] = gb["$close"].apply(lambda s, w=w: s.rolling(w).std()) / (df["$close"] + EPS)

    # VSTD 5/20 (your YAML: Std($volume,w)/$volume)
    df["$VSTD5"] = gb["$volume"].apply(lambda s: s.rolling(5).std()) / (df["$volume"] + EPS)
    df["$VSTD20"] = gb["$volume"].apply(lambda s: s.rolling(20).std()) / (df["$volume"] + EPS)

    # WVMA 5/20/60 (keep your established definition: std/mean of |ret|*volume)
    ret_abs = gb["$close"].apply(lambda s: (s / s.shift(1) - 1.0).abs())
    act = ret_abs * df["$volume"]
    for w in (5, 20, 60):
        df[f"$WVMA{w}"] = act.groupby(level="instrument", group_keys=False).apply(
            lambda s, w=w: s.rolling(w).std() / (s.rolling(w).mean() + EPS)
        )

    # CORR 10/20/60: Corr(close, log(volume+1), w)
    logv = np.log(df["$volume"] + 1.0)
    for w in (10, 20, 60):
        df[f"$CORR{w}"] = gb["$close"].apply(lambda s, w=w: s.rolling(w).corr(logv.loc[s.index]))

    # CORD 10/20/60: Corr(close ratio, log(volume ratio + 1), w)
    pr = gb["$close"].apply(lambda s: s / s.shift(1))
    vr = gb["$volume"].apply(lambda s: np.log(s / s.shift(1) + 1.0))
    for w in (10, 20, 60):
        df[f"$CORD{w}"] = pr.groupby(level="instrument", group_keys=False).apply(
            lambda s, w=w: s.rolling(w).corr(vr.loc[s.index])
        )
    return df


def _enrich_df_level1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Level-1 primitives
    df["$OIMOM20_vol_scaled_VSTD5"] = df["$OIMOM20"] / (1.0 + df["$VSTD5"])
    df["$PMOM5_vol_scaled_STD5"] = df["$PMOM5"] / (1.0 + df["$STD5"])
    df["$CORR20_vol_scaled_WVMA5"] = df["$CORR20"] / (1.0 + df["$WVMA5"])
    # KLOW_OI_pressure
    oi = df["$open_interest"]
    oi_ma = (
        oi.groupby(level="instrument", group_keys=False)
        .apply(lambda s: s.rolling(60).mean().shift(1))
    )

    oi_rel = oi / (oi_ma + 1e-12)
    df["$KLOW_OI_pressure"] = df["$KLOW"] * oi_rel
    # VMOM5_tanh_squash
    df["$VMOM5_tanh_squash"] = np.tanh(df["$VMOM5"] / (1.0 + df["$VSTD5"]))

    # ============================================================
    # New factors (minimal changes; appended here)
    # ============================================================

    # (1) KLOW_OI_momentum_filter
    # s5 = tanh(lam * PMOM5); factor = KLOW_OI_pressure * s5
    lam = 1.0
    df["$KLOW_OI_momentum_filter"] = df["$KLOW_OI_pressure"] * np.tanh(lam * df["$PMOM5"])

    # (2) OI_accel_VSTD20_smooth_horizon_blend
    eps = 1e-8
    q_short = 0.5 * df["$OIMOM5"] + 0.5 * df["$OIMOM10"]
    q_medium = 0.5 * df["$OIMOM20"] + 0.5 * df["$OIMOM60"]
    accel_blend = q_short - q_medium
    df["$OI_accel_VSTD20_smooth_horizon_blend"] = -accel_blend / (df["$VSTD20"] + eps)

    # (3) range_vol_efficiency5
    window = 5
    close = df["$close"]
    high = df["$high"]
    low = df["$low"]
    close_shift = close.groupby(level="instrument", group_keys=False).shift(window)
    r_5 = (close - close_shift) / (close_shift + eps)

    daily_range = high - low
    avg_range_5 = (
        daily_range
        .groupby(level="instrument", group_keys=False)
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        / window
    )
    e_5 = r_5 / (avg_range_5 + eps)
    df["$range_vol_efficiency5"] = e_5 / (df["$STD5"] + eps)

    # (4) Seasonal_CORR20_WVMA5_carry_adjusted
    # requires $dayofyear in df columns
    if "$dayofyear" in df.columns:
        corr20 = df["$CORR20"].astype("float64")
        wvma5 = df["$WVMA5"].astype("float64")
        dayofyear = df["$dayofyear"].astype("float64")

        hat_C = corr20 / (1.0 + wvma5)

        mu_carry = 210.0
        w_carry = 40.0
        diff = np.abs(dayofyear - mu_carry)
        d_tilde = np.minimum(diff, 365.0 - diff)
        c_t = np.exp(- (d_tilde ** 2) / (2.0 * (w_carry ** 2)))

        df["$Seasonal_CORR20_WVMA5_carry_adjusted"] = c_t * hat_C

    # (5) Seasonal_OIMOM20_harvest_participation
    # requires $dayofyear in df columns
    if "$dayofyear" in df.columns:
        d_t = df["$dayofyear"].astype("float64")
        mu_harv = 280.0
        w_harv = 40.0
        raw_diff = np.abs(d_t - mu_harv)
        wrapped_diff = 365.0 - raw_diff
        d_tilde = np.minimum(raw_diff, wrapped_diff)
        denom = 2.0 * (w_harv ** 2)
        h_t = np.exp(-(d_tilde ** 2) / denom)
        oimom20 = df["$OIMOM20"].astype("float64")
        df["$Seasonal_OIMOM20_harvest_participation"] = h_t * oimom20

    return df


def enrich_daily_pv_h5(h5_path: str, key: str = "data") -> None:
    df = pd.read_hdf(h5_path, key=key)
    df = _enrich_df_primitives(df)
    df = _enrich_df_level1(df)
    with pd.HDFStore(h5_path, mode="a") as store:
        if f"/{key}" in store.keys():
            store.remove(key)          
        store.put(key, df, format="fixed")