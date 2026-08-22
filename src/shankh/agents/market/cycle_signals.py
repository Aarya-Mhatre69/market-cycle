"""
Core Cycle Signal Engine — Phase 1 (price-structure only, no external dependency).

Implements the three "skeleton" signals from the Market Cycle Agent architecture spec:
ZigZag swing structure, Schaff Trend Cycle (STC), and 50/200-day trend context.
Each signal is a pure function over an OHLCV DataFrame so it can be unit-tested and
backtested independently of any LLM or live API call.

Every signal outputs a normalized score in [-1.0, +1.0] (expansionary vs contractionary
pressure). Scores are combined by classify_core_cycle() into a phase + confidence +
transition-risk read using a trend-direction x momentum-direction matrix — not a
hand-tuned if/elif waterfall — so thresholds are few, explicit, and defensible.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Phase = Literal["ACCUMULATION", "EXPANSION", "DISTRIBUTION", "CONTRACTION"]


# ---------------------------------------------------------------------------
# 1. ZigZag swing structure
# ---------------------------------------------------------------------------

@dataclass
class ZigZagResult:
    pivots: List[Dict[str, Any]]
    structure: str
    score: float
    note: str


def compute_zigzag(df: pd.DataFrame, pct_threshold: float = 4.0) -> ZigZagResult:
    """
    Percentage-reversal ZigZag over high/low series.

    A new pivot is confirmed once price reverses by >= pct_threshold% from the last
    confirmed pivot. Threshold is deliberately >= typical daily noise (~1-2% on Nifty)
    so pivots mark genuine swings, not single-day wiggles.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    dates = df["date"].to_numpy()

    if len(df) < 10:
        return ZigZagResult([], "INSUFFICIENT_DATA", 0.0, "Fewer than 10 bars available.")

    pivots: List[Dict[str, Any]] = []
    trend: Optional[str] = None  # None until bootstrapped, then "up" or "down" since last pivot
    last_pivot_price = float(df["close"].iloc[0])
    last_pivot_idx = 0

    for i in range(1, len(df)):
        if trend is None:
            # Bootstrap: wait for the first threshold-sized move from bar 0 to establish
            # an initial direction. Using an if/elif here (not two independent ifs) is
            # deliberate — running both branches in the same iteration was the original
            # bug: the down-branch immediately overwrote whatever the up-branch had just
            # set, so last_pivot_price could never accumulate and no pivot ever confirmed.
            if highs[i] >= last_pivot_price * (1 + pct_threshold / 100.0):
                trend = "up"
                last_pivot_price = highs[i]
                last_pivot_idx = i
            elif lows[i] <= last_pivot_price * (1 - pct_threshold / 100.0):
                trend = "down"
                last_pivot_price = lows[i]
                last_pivot_idx = i
        elif trend == "up":
            if highs[i] > last_pivot_price:
                last_pivot_price = highs[i]
                last_pivot_idx = i
            elif lows[i] <= last_pivot_price * (1 - pct_threshold / 100.0):
                pivots.append({
                    "date": str(pd.Timestamp(dates[last_pivot_idx]).date()),
                    "type": "HIGH",
                    "price": round(float(last_pivot_price), 2),
                })
                trend = "down"
                last_pivot_price = lows[i]
                last_pivot_idx = i
        elif trend == "down":
            if lows[i] < last_pivot_price:
                last_pivot_price = lows[i]
                last_pivot_idx = i
            elif highs[i] >= last_pivot_price * (1 + pct_threshold / 100.0):
                pivots.append({
                    "date": str(pd.Timestamp(dates[last_pivot_idx]).date()),
                    "type": "LOW",
                    "price": round(float(last_pivot_price), 2),
                })
                trend = "up"
                last_pivot_price = highs[i]
                last_pivot_idx = i

    if len(pivots) < 4:
        return ZigZagResult(pivots, "INSUFFICIENT_SWINGS", 0.0,
                             f"Only {len(pivots)} confirmed swings in window; need >= 4.")

    last4 = pivots[-4:]
    highs_seq = [p["price"] for p in last4 if p["type"] == "HIGH"]
    lows_seq = [p["price"] for p in last4 if p["type"] == "LOW"]

    higher_highs = len(highs_seq) >= 2 and highs_seq[-1] > highs_seq[-2]
    higher_lows = len(lows_seq) >= 2 and lows_seq[-1] > lows_seq[-2]
    lower_highs = len(highs_seq) >= 2 and highs_seq[-1] < highs_seq[-2]
    lower_lows = len(lows_seq) >= 2 and lows_seq[-1] < lows_seq[-2]

    if higher_highs and higher_lows:
        structure, score = "HIGHER_HIGHS_HIGHER_LOWS", 1.0
    elif lower_highs and lower_lows:
        structure, score = "LOWER_HIGHS_LOWER_LOWS", -1.0
    elif higher_lows and lower_highs:
        structure, score = "CONTRACTING_RANGE", 0.0
    elif higher_highs and lower_lows:
        structure, score = "EXPANDING_RANGE_VOLATILE", 0.0
    else:
        structure, score = "MIXED", 0.0

    return ZigZagResult(
        pivots=last4,
        structure=structure,
        score=score,
        note=f"Last 4 confirmed swings ({pct_threshold}% reversal threshold) show {structure}.",
    )


# ---------------------------------------------------------------------------
# 2. Schaff Trend Cycle (STC)
# ---------------------------------------------------------------------------

@dataclass
class STCResult:
    value: float
    direction: str
    score: float
    note: str


def compute_stc(
    close: pd.Series,
    fast: int = 23,
    slow: int = 50,
    cycle: int = 10,
    smooth_factor: float = 0.5,
) -> STCResult:
    """
    Schaff Trend Cycle: a double-smoothed stochastic oscillator applied to MACD,
    designed to react to momentum turns faster than raw MACD while filtering more
    noise than a raw price stochastic. Standard Schaff (2008) parameters (23/50/10).
    """
    if len(close) < slow + cycle * 2:
        return STCResult(0.0, "INSUFFICIENT_DATA", 0.0,
                          f"Need >= {slow + cycle * 2} bars, have {len(close)}.")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow

    def _bounded_stochastic(series: pd.Series) -> pd.Series:
        """Stochastic of `series` over `cycle` bars, in [0, 100].

        During a strong, sustained trend the rolling range can go (near) flat while
        pinned at an extreme (e.g. every bar is a new high, so max == current == a
        fixed local max for several bars). Dividing 0/~0 there and calling it 0 would
        misreport a pinned-at-the-top reading as "collapsed to the bottom" — the
        standard fix (and what public STC implementations do) is to hold the last
        valid reading forward through the degenerate window instead.
        """
        ll = series.rolling(cycle).min()
        hh = series.rolling(cycle).max()
        span = hh - ll
        raw = (series - ll) / span * 100.0
        raw = raw.where(span > 1e-9)
        return raw.ffill().fillna(50.0).clip(0.0, 100.0)

    stoch1 = _bounded_stochastic(macd)
    pf = stoch1.ewm(alpha=smooth_factor, adjust=False).mean()
    stoch2 = _bounded_stochastic(pf)

    stc = stoch2.ewm(alpha=smooth_factor, adjust=False).mean()
    stc = stc.clip(0.0, 100.0)

    latest = float(stc.iloc[-1])
    prev5 = float(stc.iloc[-6]) if len(stc) >= 6 else float(stc.iloc[0])
    rising = latest > prev5

    if latest > 75.0:
        zone = "OVERBOUGHT_LATE_CYCLE_MOMENTUM"
    elif latest < 25.0:
        zone = "OVERSOLD_EARLY_CYCLE_MOMENTUM"
    else:
        zone = "NEUTRAL_MOMENTUM"

    # Score: distance from midline (50) direction-weighted by 5-bar slope, not just level —
    # a rising STC from 30 is expansionary even though the raw level is still low.
    level_component = (latest - 50.0) / 50.0
    slope_component = np.clip((latest - prev5) / 25.0, -1.0, 1.0)
    score = float(np.clip(0.5 * level_component + 0.5 * slope_component, -1.0, 1.0))

    return STCResult(
        value=round(latest, 2),
        direction="RISING" if rising else "FALLING",
        score=round(score, 3),
        note=f"STC={round(latest, 1)} ({zone}), {'rising' if rising else 'falling'} over last 5 bars.",
    )


# ---------------------------------------------------------------------------
# 3. Trend / MA context
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    price_vs_sma200_pct: float
    sma50_vs_sma200_pct: float
    score: float
    note: str


def compute_trend_context(close: pd.Series, short: int = 50, long: int = 200) -> TrendResult:
    """Anchors momentum readings to a slow-moving trend baseline (classic golden/death-cross context)."""
    if len(close) < long:
        return TrendResult(0.0, 0.0, 0.0, f"Need >= {long} bars, have {len(close)}.")

    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()

    last_close = float(close.iloc[-1])
    last_short = float(sma_short.iloc[-1])
    last_long = float(sma_long.iloc[-1])

    price_vs_long = (last_close - last_long) / last_long * 100.0
    short_vs_long = (last_short - last_long) / last_long * 100.0

    score = float(np.clip((price_vs_long / 10.0 + short_vs_long / 10.0) / 2.0, -1.0, 1.0))

    note = (
        f"Price is {price_vs_long:+.1f}% vs {long}DMA; "
        f"{short}DMA is {short_vs_long:+.1f}% vs {long}DMA "
        f"({'golden' if short_vs_long > 0 else 'death'}-cross side)."
    )

    return TrendResult(
        price_vs_sma200_pct=round(price_vs_long, 2),
        sma50_vs_sma200_pct=round(short_vs_long, 2),
        score=round(score, 3),
        note=note,
    )


# ---------------------------------------------------------------------------
# 4. Composite classification
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    signal: str
    tier: str
    reading: str
    score: float
    note: str = ""


@dataclass
class CycleClassification:
    cycle_phase: Phase
    cycle_confidence: float
    transition_risk: str
    transition_watch: str
    composite_score: float
    evidence: List[EvidenceItem] = field(default_factory=list)


_TIER_WEIGHTS = {"core": 0.6, "supporting": 0.3, "contextual": 0.1}


def classify_cycle(evidence: List[EvidenceItem]) -> CycleClassification:
    """
    Weighted composite classification (architecture.md section 6).

    Phase is decided by a trend-direction x momentum-direction matrix, not a
    hand-tuned threshold waterfall:
        trend up   + momentum rising  -> EXPANSION
        trend up   + momentum falling -> DISTRIBUTION   (topping / momentum divergence)
        trend down + momentum falling -> CONTRACTION
        trend down + momentum rising  -> ACCUMULATION   (basing / momentum divergence)

    "Trend" and "momentum" each come from a designated core signal (trend_context,
    stc respectively) so the matrix is reproducible from named inputs, not from the
    composite score alone.
    """
    by_signal = {e.signal: e for e in evidence}
    trend_score = by_signal.get("trend_context", EvidenceItem("trend_context", "core", "", 0.0)).score
    momentum_score = by_signal.get("stc", EvidenceItem("stc", "core", "", 0.0)).score

    weighted_sum = 0.0
    weight_total = 0.0
    for e in evidence:
        w = _TIER_WEIGHTS.get(e.tier, 0.1)
        weighted_sum += e.score * w
        weight_total += w
    composite_score = round(weighted_sum / weight_total, 3) if weight_total > 0 else 0.0

    trend_up = trend_score >= 0
    momentum_up = momentum_score >= 0

    if trend_up and momentum_up:
        phase: Phase = "EXPANSION"
        watch = "Expansion -> Distribution"
    elif trend_up and not momentum_up:
        phase = "DISTRIBUTION"
        watch = "Distribution -> Contraction"
    elif not trend_up and not momentum_up:
        phase = "CONTRACTION"
        watch = "Contraction -> Accumulation"
    else:
        phase = "ACCUMULATION"
        watch = "Accumulation -> Expansion"

    agree = sum(1 for e in evidence if (e.score >= 0) == (composite_score >= 0))
    confidence = round(agree / len(evidence), 2) if evidence else 0.0

    divergence = (trend_up and not momentum_up) or (not trend_up and momentum_up)
    magnitude = abs(trend_score - momentum_score)
    # Thresholds below (0.9 / 1.35) were calibrated against the walk-forward backtest
    # (scripts/backtest_cycle_phase.py, ~623 evaluation points over 2018-2026 Nifty
    # data), not picked a priori. The original untuned guess (0.3 / 0.6) turned out to
    # sit below the median divergence magnitude actually observed (~1.18 among
    # divergent cases), which pinned almost every divergence at "high" and left
    # "medium" essentially unused (2.6% of readings). These values are tertiles of the
    # empirical divergence-magnitude distribution, so the three risk tiers now split
    # roughly evenly among cases where trend and momentum actually disagree.
    if divergence and magnitude > 1.35:
        transition_risk = "high"
    elif divergence and magnitude > 0.9:
        transition_risk = "medium"
    else:
        transition_risk = "low"

    return CycleClassification(
        cycle_phase=phase,
        cycle_confidence=confidence,
        transition_risk=transition_risk,
        transition_watch=watch if transition_risk != "low" else "none",
        composite_score=composite_score,
        evidence=evidence,
    )
