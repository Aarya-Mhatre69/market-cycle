"""
Feature engineering for the price-band forecasting pipeline.

All features are derived purely from OHLCV price-action. The same patterns in volatility clustering,
momentum, or candlestick anatomy are equally valid for any equity in any
market; baking in ticker identity would overfit to in-sample instruments
and break generalisation to unseen ones.

Implementation note
-------------------
All sub-functions return a plain dict of {column_name: Series}.  The
top-level ``add_features`` collects every dict, then calls a single
``pd.concat`` to attach all columns at once.  This avoids the pandas
"highly fragmented DataFrame" PerformanceWarning that arises from
hundreds of individual ``df[col] = ...`` assignments.

Feature groups
--------------
1.  Return / momentum
2.  Candlestick anatomy  (body, wicks, gap-open)
3.  Volume / liquidity
4.  Trend / moving-average
5.  Oscillators          (RSI, Stochastic %K, Williams %R, MACD)
6.  Volatility           (ATR, realised vol, Parkinson, Garman–Klass, Bollinger)
7.  Rolling high/low / breakout
8.  Calendar
9.  Regime proxies       (vol-of-vol, return skew/kurt/autocorr, drawdown)
10. Targets              (shift(-1) — next-day high/low in log-return space)
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute all OHLCV-derived features and targets.

    ``ticker`` is used only as a grouping key so that rolling windows
    and lag operations do not bleed across instruments.  It is excluded
    from the model feature set (see ``get_feature_cols``).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV frame with columns
        ``date, open, high, low, close, volume, ticker``.
        Must be sorted by (ticker, date).
    cfg : dict, optional
        Feature config sub-dict from CONFIG["features"].
        Uses sensible defaults when None.

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features and targets.
        Rows with NaN or inf in any column are dropped.
        ``ticker`` is retained for bookkeeping / evaluation.
    """
    if cfg is None:
        cfg = _default_feature_cfg()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    g  = df.groupby("ticker", group_keys=False)

    # Collect new columns from every group as plain dicts, then concat once.
    new_cols: dict[str, pd.Series] = {}

    new_cols.update(_return_features(df, g, cfg))
    new_cols.update(_candle_features(df))
    new_cols.update(_volume_features(df, g))
    new_cols.update(_trend_features(df, g, cfg))

    # Trend features must exist before oscillators and volatility
    # (bb uses ma_20 / std_20; oscillators use rolling highs/lows).
    # Attach them now so subsequent helpers can reference them.
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    new_cols = {}

    new_cols.update(_oscillator_features(df, g, cfg))
    new_cols.update(_volatility_features(df, g, cfg))
    new_cols.update(_breakout_features(df, g))
    new_cols.update(_calendar_features(df))

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    new_cols = {}

    new_cols.update(_regime_proxy_features(df, g))
    new_cols.update(_target_columns(df, g))

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Drop rows with NaN / inf in any column
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    logger.info(
        "Feature engineering: %d → %d rows (dropped %d with NaN/inf)",
        n_before, len(df), n_before - len(df),
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the column names to use as model inputs.

    Explicitly excluded:
    - ``date``, ``ticker``     — identifiers, not signals
    - raw OHLCV                — not normalised; targets are derived from them
    - ``target_*``, ``next_*`` — look-ahead labels
    """
    _exclude = {
        "date", "ticker",
        "open", "high", "low", "close", "volume",
        "target_upper", "target_lower",
        "next_high_actual", "next_low_actual", "next_close_actual",
    }
    return [
        c for c in df.columns
        if c not in _exclude and pd.api.types.is_numeric_dtype(df[c])
    ]



def _return_features(df: pd.DataFrame, g, cfg: dict) -> dict[str, pd.Series]:
    cols: dict[str, pd.Series] = {}

    cols["log_ret_1"] = g["close"].transform(lambda s: np.log(s / s.shift(1)))

    for lag in cfg.get("ret_lags", [1, 2, 3, 5, 10, 20]):
        cols[f"ret_{lag}"] = g["close"].pct_change(lag)

    for w in cfg.get("rolling_windows", [5, 10, 20, 50]):
        cols[f"cum_ret_{w}"] = g["close"].transform(lambda s, w=w: s.pct_change(w))

    prev_close = g["close"].shift(1)
    cols["gap_open"] = (df["open"] / (prev_close + 1e-12)) - 1.0

    return cols


def _candle_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]  # noqa: E741
    cols: dict[str, pd.Series] = {}

    cols["hl_range"]  = (h - l) / (c + 1e-12)
    cols["oc_body"]   = (c - o) / (o + 1e-12)
    cols["body_size"] = (c - o).abs() / (c + 1e-12)

    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    cols["upper_wick"] = (h - body_top) / (c + 1e-12)
    cols["lower_wick"] = (body_bot - l) / (c + 1e-12)

    span = (h - l).clip(lower=1e-12)
    cols["body_to_span"] = (c - o).abs() / span
    cols["wick_ratio"]   = cols["upper_wick"] / (cols["lower_wick"] + 1e-12)

    return cols


def _volume_features(df: pd.DataFrame, g) -> dict[str, pd.Series]:
    cols: dict[str, pd.Series] = {}

    dollar_vol = df["close"] * df["volume"]
    cols["dollar_volume"] = dollar_vol
    cols["log_volume"]    = np.log1p(df["volume"])
    cols["vol_chg_1"]     = g["volume"].pct_change(1)

    vol_ma_5  = g["volume"].transform(lambda s: s.rolling(5).mean())
    vol_ma_20 = g["volume"].transform(lambda s: s.rolling(20).mean())
    cols["vol_ma_5"]    = vol_ma_5
    cols["vol_ma_20"]   = vol_ma_20
    cols["vol_std_20"]  = g["volume"].transform(lambda s: s.rolling(20).std())
    cols["vol_ratio_5"] = df["volume"] / (vol_ma_5  + 1e-12)
    cols["vol_ratio_20"]= df["volume"] / (vol_ma_20 + 1e-12)

    dv_ma  = dollar_vol.groupby(df["ticker"]).transform(lambda s: s.rolling(20).mean())
    dv_std = dollar_vol.groupby(df["ticker"]).transform(lambda s: s.rolling(20).std())
    cols["dv_zscore_20"] = (dollar_vol - dv_ma) / (dv_std + 1e-12)

    return cols


def _trend_features(df: pd.DataFrame, g, cfg: dict) -> dict[str, pd.Series]:
    windows = cfg.get("rolling_windows", [5, 10, 20, 50])
    cols: dict[str, pd.Series] = {}

    for w in windows:
        ma  = g["close"].transform(lambda s, w=w: s.rolling(w).mean())
        ema = g["close"].transform(lambda s, w=w: s.ewm(span=w, adjust=False).mean())
        std = g["close"].transform(lambda s, w=w: s.rolling(w).std())

        cols[f"ma_{w}"]       = ma
        cols[f"ema_{w}"]      = ema
        cols[f"std_{w}"]      = std
        cols[f"mom_{w}"]      = df["close"] / (ma  + 1e-12) - 1.0
        cols[f"mom_ema_{w}"]  = df["close"] / (ema + 1e-12) - 1.0
        cols[f"zscore_{w}"]   = (df["close"] - ma) / (std + 1e-12)

    if 5 in windows and 20 in windows:
        cols["ma_cross_5_20"]  = cols["ma_5"]  / (cols["ma_20"]  + 1e-12) - 1.0
    if 10 in windows and 50 in windows:
        cols["ma_cross_10_50"] = cols["ma_10"] / (cols["ma_50"] + 1e-12) - 1.0

    return cols


def _oscillator_features(df: pd.DataFrame, g, cfg: dict) -> dict[str, pd.Series]:
    """Requires ma_*/std_* columns already present in df."""
    cols: dict[str, pd.Series] = {}

    rsi_period = cfg.get("rsi_period", 14)
    cols["rsi_14"] = g["close"].transform(lambda s: _rsi(s, rsi_period))

    roll_high_14 = g["high"].transform(lambda s: s.rolling(14).max())
    roll_low_14  = g["low"].transform(lambda s: s.rolling(14).min())
    hl_range_14  = roll_high_14 - roll_low_14
    cols["roll_high_14"] = roll_high_14
    cols["roll_low_14"]  = roll_low_14
    cols["stoch_k_14"]   = (df["close"] - roll_low_14) / (hl_range_14 + 1e-12) * 100.0
    cols["williams_r_14"]= (roll_high_14 - df["close"]) / (hl_range_14 + 1e-12) * -100.0

    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    macd_norm   = (ema12 - ema26) / (df["close"] + 1e-12)
    macd_signal = macd_norm.groupby(df["ticker"]).transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    cols["macd_norm"]        = macd_norm
    cols["macd_signal_norm"] = macd_signal
    cols["macd_hist"]        = macd_norm - macd_signal

    return cols


def _volatility_features(df: pd.DataFrame, g, cfg: dict) -> dict[str, pd.Series]:
    """Requires ma_20, std_20 already present in df."""
    cols: dict[str, pd.Series] = {}

    atr_period  = cfg.get("atr_period", 14)
    bb_std_mult = cfg.get("bb_std", 2.0)
    vol_windows = cfg.get("vol_windows", [5, 10, 20])

    atr = _atr(df, atr_period)
    cols["atr_14"]  = atr
    cols["atr_pct"] = atr / (df["close"] + 1e-12)

    log_ret = df["log_ret_1"] if "log_ret_1" in df.columns else (
        g["close"].transform(lambda s: np.log(s / s.shift(1)))
    )

    rv_series: dict[int, pd.Series] = {}
    for w in vol_windows:
        rv = log_ret.groupby(df["ticker"]).transform(
            lambda s, w=w: s.rolling(w).std() * np.sqrt(252)
        )
        cols[f"rv_{w}"] = rv
        rv_series[w] = rv

    # Parkinson high-low volatility estimator
    log_hl_sq = np.log(df["high"] / (df["low"] + 1e-12)) ** 2
    for w in vol_windows:
        cols[f"parkinson_{w}"] = (
            log_hl_sq.groupby(df["ticker"]).transform(
                lambda s, w=w: (s.rolling(w).mean() / (4 * np.log(2))) ** 0.5
            ) * np.sqrt(252)
        )

    # Garman–Klass estimator (20-day)
    gk_term = (
        0.5 * np.log(df["high"] / (df["low"] + 1e-12)) ** 2
        - (2 * np.log(2) - 1) * np.log(df["close"] / (df["open"] + 1e-12)) ** 2
    )
    cols["garman_klass_20"] = (
        gk_term.groupby(df["ticker"]).transform(lambda s: s.rolling(20).mean()) ** 0.5
        * np.sqrt(252)
    )

    # Bollinger Bands — requires ma_20 and std_20 already in df
    ma_20  = df["ma_20"]
    std_20 = df["std_20"]
    bb_upper = ma_20 + bb_std_mult * std_20
    bb_lower = ma_20 - bb_std_mult * std_20
    cols["bb_upper"] = bb_upper
    cols["bb_lower"] = bb_lower
    cols["bb_pct_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-12)
    cols["bb_width"] = (bb_upper - bb_lower) / (ma_20 + 1e-12)

    # Short/long vol regime ratio
    if 5 in rv_series and 20 in rv_series:
        cols["vol_ratio_5_20"] = rv_series[5] / (rv_series[20] + 1e-12)

    return cols


def _breakout_features(df: pd.DataFrame, g) -> dict[str, pd.Series]:
    cols: dict[str, pd.Series] = {}

    for w in [5, 10, 20, 50]:
        rh = g["high"].transform(lambda s, w=w: s.rolling(w).max())
        rl = g["low"].transform(lambda s, w=w: s.rolling(w).min())
        cols[f"rolling_high_{w}"]  = rh
        cols[f"rolling_low_{w}"]   = rl
        cols[f"breakout_up_{w}"]   = df["close"] / (rh + 1e-12) - 1.0
        cols[f"breakdown_dn_{w}"]  = df["close"] / (rl + 1e-12) - 1.0
        cols[f"range_pct_{w}"]     = (rh - rl) / (df["close"] + 1e-12)

    return cols


def _calendar_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "dayofweek":     df["date"].dt.dayofweek,
        "month":         df["date"].dt.month,
        "quarter":       df["date"].dt.quarter,
        "is_month_end":  df["date"].dt.is_month_end.astype(np.int8),
        "is_month_start":df["date"].dt.is_month_start.astype(np.int8),
        "is_quarter_end":df["date"].dt.is_quarter_end.astype(np.int8),
    }


def _regime_proxy_features(df: pd.DataFrame, g) -> dict[str, pd.Series]:
    """
    Regime proxies derived purely from price/volume history.
    Requires rv_20 and log_ret_1 columns already present in df.
    """
    cols: dict[str, pd.Series] = {}

    if "rv_20" in df.columns:
        cols["vov_20"] = df["rv_20"].groupby(df["ticker"]).transform(
            lambda s: s.rolling(20).std()
        )

    log_ret = df["log_ret_1"]
    cols["ret_skew_20"] = log_ret.groupby(df["ticker"]).transform(
        lambda s: s.rolling(20).skew()
    )
    cols["ret_kurt_20"] = log_ret.groupby(df["ticker"]).transform(
        lambda s: s.rolling(20).kurt()
    )
    cols["ret_autocorr_20"] = log_ret.groupby(df["ticker"]).transform(
        lambda s: s.rolling(20).apply(
            lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= 4 else np.nan,
            raw=False,
        )
    )
    cols["drawdown_50"] = (
        df["close"] / (
            df["high"].groupby(df["ticker"]).transform(lambda s: s.rolling(50).max()) + 1e-12
        ) - 1.0
    )

    return cols


def _target_columns(df: pd.DataFrame, g) -> dict[str, pd.Series]:
    """
    Next-day high / low / close targets.  shift(-1) within each ticker.

    These are the supervised labels — they must NEVER appear in the model
    feature set.  ``get_feature_cols`` explicitly excludes them.
    """
    next_high  = g["high"].shift(-1)
    next_low   = g["low"].shift(-1)
    next_close = g["close"].shift(-1)
    log_close  = np.log(df["close"] + 1e-12)

    return {
        "target_upper":      np.log(next_high  + 1e-12) - log_close,
        "target_lower":      np.log(next_low   + 1e-12) - log_close,
        "next_high_actual":  next_high,
        "next_low_actual":   next_low,
        "next_close_actual": next_close,
    }



def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df.groupby("ticker")["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.groupby(df["ticker"]).transform(lambda s: s.rolling(period).mean())


# ---------------------------------------------------------------------------
# Default feature config
# ---------------------------------------------------------------------------

def _default_feature_cfg() -> dict:
    return {
        "rolling_windows": [5, 10, 20, 50],
        "rsi_period":      14,
        "atr_period":      14,
        "bb_period":       20,
        "bb_std":          2.0,
        "vol_windows":     [5, 10, 20],
        "ret_lags":        [1, 2, 3, 5, 10, 20],
    }
