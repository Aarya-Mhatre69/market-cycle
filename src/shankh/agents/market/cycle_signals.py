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
    # Full confirmed-pivot history (not just the last 4 used for trend structure) —
    # added for Phase 3's Harmonic Pattern detection, which needs 5 consecutive
    # pivots (X-A-B-C-D). Backward compatible: `pivots` is unchanged, this is additive.
    all_pivots: List[Dict[str, Any]] = field(default_factory=list)


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
                             f"Only {len(pivots)} confirmed swings in window; need >= 4.",
                             all_pivots=pivots)

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
        all_pivots=pivots,
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
# 4. Candlestick Recognition (accuracy-roadmap Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class CandlestickResult:
    patterns: List[Dict[str, Any]]  # [{date, pattern, direction}]
    score: float
    note: str


def compute_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> CandlestickResult:
    """
    Detects Doji, (Bullish/Bearish) Engulfing, Hammer, and Morning Star in the last
    `lookback` bars — standard body/wick-ratio rules on daily OHLC, no external state.

    Deliberately does NOT add the bearish mirror patterns (Hanging Man, Evening Star)
    that aren't in the reference indicator list — adding un-requested patterns would
    silently change what's being evaluated. This does mean the set is structurally
    biased toward flagging bullish reversals more often than bearish ones: Hammer and
    Morning Star are bullish-only by definition, Engulfing is the only fully
    bidirectional pattern here, and Doji's direction depends on the preceding local
    trend (standard candlestick theory: a doji after an uptrend reads bearish, after a
    downtrend reads bullish). This asymmetry is disclosed, not hidden — the backtest
    evaluation (scripts/evaluate_indicators.py) will show whether it matters in practice.
    """
    if len(df) < 12:
        return CandlestickResult([], 0.0, "Fewer than 12 bars available (need >= 10 for trend context + pattern bars).")

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)

    def _body(i: int) -> float:
        return abs(c[i] - o[i])

    def _range(i: int) -> float:
        return max(h[i] - l[i], 1e-9)

    def _upper_wick(i: int) -> float:
        return h[i] - max(c[i], o[i])

    def _lower_wick(i: int) -> float:
        return min(c[i], o[i]) - l[i]

    def _local_trend(i: int, window: int = 10) -> int:
        """+1 if price rose over the prior `window` bars (before bar i), -1 if it
        fell by more than 1%, else 0 (flat/insufficient context)."""
        start = max(0, i - window)
        if start >= i or c[start] <= 0:
            return 0
        pct = (c[i - 1] - c[start]) / c[start] * 100.0
        if pct > 1.0:
            return 1
        if pct < -1.0:
            return -1
        return 0

    detected: List[Dict[str, Any]] = []
    start_i = max(10, n - lookback)
    for i in range(start_i, n):
        bar_date = str(pd.Timestamp(dates[i]).date())

        if _body(i) <= 0.1 * _range(i):
            trend = _local_trend(i)
            if trend > 0:
                detected.append({"idx": i, "date": bar_date, "pattern": "DOJI", "direction": "bearish"})
            elif trend < 0:
                detected.append({"idx": i, "date": bar_date, "pattern": "DOJI", "direction": "bullish"})

        if i >= 1 and _body(i - 1) > 1e-9:
            if c[i] > o[i] and c[i - 1] < o[i - 1] and o[i] <= c[i - 1] and c[i] >= o[i - 1]:
                detected.append({"idx": i, "date": bar_date, "pattern": "BULLISH_ENGULFING", "direction": "bullish"})
            elif c[i] < o[i] and c[i - 1] > o[i - 1] and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                detected.append({"idx": i, "date": bar_date, "pattern": "BEARISH_ENGULFING", "direction": "bearish"})

        body = _body(i)
        if body > 1e-9 and _lower_wick(i) >= 2.0 * body and _upper_wick(i) <= 0.3 * body and _local_trend(i) < 0:
            detected.append({"idx": i, "date": bar_date, "pattern": "HAMMER", "direction": "bullish"})

        if i >= 2:
            c1_bearish = c[i - 2] < o[i - 2] and _body(i - 2) >= 0.6 * _range(i - 2)
            star_small = _body(i - 1) <= 0.3 * _range(i - 1)
            gapped_down = max(o[i - 1], c[i - 1]) < c[i - 2]
            c3_bullish = c[i] > o[i] and _body(i) >= 0.6 * _range(i)
            closes_into_c1 = c[i] >= (o[i - 2] + c[i - 2]) / 2.0
            if c1_bearish and star_small and gapped_down and c3_bullish and closes_into_c1:
                detected.append({"idx": i, "date": bar_date, "pattern": "MORNING_STAR", "direction": "bullish"})

    if not detected:
        return CandlestickResult([], 0.0, f"No Doji/Engulfing/Hammer/Morning Star patterns in the last {lookback} bars.")

    # Recency-weighted score: the most recent bar in the lookback window carries full
    # weight; older ones decay linearly toward a floor so a pattern from several bars
    # ago still counts but matters less than one from yesterday.
    weighted, weight_total = 0.0, 0.0
    for d in detected:
        age = (n - 1) - d["idx"]
        w = max(0.2, 1.0 - age / max(lookback, 1))
        sign = 1.0 if d["direction"] == "bullish" else -1.0
        weighted += sign * w
        weight_total += w

    score = float(np.clip(weighted / weight_total, -1.0, 1.0)) if weight_total > 0 else 0.0
    pattern_names = ", ".join(sorted({d["pattern"] for d in detected}))
    public_patterns = [{"date": d["date"], "pattern": d["pattern"], "direction": d["direction"]} for d in detected]
    return CandlestickResult(
        patterns=public_patterns,
        score=round(score, 3),
        note=f"{len(detected)} pattern(s) in last {lookback} bars: {pattern_names}.",
    )


# ---------------------------------------------------------------------------
# 5. Harmonic Patterns (accuracy-roadmap Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class HarmonicResult:
    pattern: Optional[str]
    direction: Optional[str]  # "bullish" | "bearish"
    completion_ratio: Optional[float]  # AD/XA
    score: float
    note: str


# XABCD retracement/extension ratio bands — one specific, documented convention among
# several published for harmonic trading (exact bands vary by source); recorded here
# so the rule actually used is auditable, matching the discipline the roadmap requires
# for Gann Time Cycles below. Each entry: (AB/XA range, BC/AB range, AD/XA range) —
# CD/BC isn't separately gated since AD/XA already constrains it given AB/XA and BC/AB.
_HARMONIC_PATTERNS: Dict[str, Dict[str, "tuple[float, float]"]] = {
    "GARTLEY":   {"ab_xa": (0.55, 0.68), "bc_ab": (0.38, 0.89), "ad_xa": (0.72, 0.85)},
    "BAT":       {"ab_xa": (0.35, 0.55), "bc_ab": (0.38, 0.89), "ad_xa": (0.82, 0.92)},
    "BUTTERFLY": {"ab_xa": (0.72, 0.85), "bc_ab": (0.38, 0.89), "ad_xa": (1.20, 1.70)},
    "CRAB":      {"ab_xa": (0.35, 0.68), "bc_ab": (0.38, 0.89), "ad_xa": (1.55, 1.70)},
    "SHARK":     {"ab_xa": (0.40, 0.68), "bc_ab": (1.05, 1.70), "ad_xa": (0.82, 1.20)},
}


def compute_harmonic_patterns(df: pd.DataFrame, pct_threshold: float = 4.0) -> HarmonicResult:
    """
    Checks whether the most recent 5 ZigZag pivots (X-A-B-C-D, reusing
    compute_zigzag's pivot detection so both indicators agree on what a "swing" is)
    form a completed Gartley/Bat/Butterfly/Crab/Shark pattern at D, using Fibonacci
    retracement/extension ratio bands (see _HARMONIC_PATTERNS).

    Only checks completion AT the most recent pivot — harmonic patterns are a
    reversal-zone call at the D point, not a continuously-valid score, so a pattern
    that completed several pivots ago and has since been invalidated by new price
    action correctly reports no signal (score 0.0), not a stale one.
    """
    zz = compute_zigzag(df, pct_threshold=pct_threshold)
    pivots = zz.all_pivots
    if len(pivots) < 5:
        return HarmonicResult(None, None, None, 0.0,
                               f"Only {len(pivots)} confirmed swings; need >= 5 for an XABCD pattern.")

    X, A, B, C, D = (p["price"] for p in pivots[-5:])
    xa, ab, bc = abs(A - X), abs(B - A), abs(C - B)
    # AD is the distance from A to D (NOT X to D) — "D completes at the 0.786
    # retracement of XA" means the A-to-D leg is 0.786x the XA leg, landing D at
    # (1 - 0.786) = 21.4% of XA away from X for a Gartley, not 78.6% away from X.
    ad = abs(D - A)
    if xa < 1e-9 or ab < 1e-9:
        return HarmonicResult(None, None, None, 0.0, "Degenerate swing legs (near-zero move); cannot compute ratios.")

    ab_xa, bc_ab, ad_xa = ab / xa, bc / ab, ad / xa
    # D and X are always the same pivot type (4 steps apart in a strictly-alternating
    # HIGH/LOW sequence), so direction is read off D's type, not a numeric D-vs-X
    # comparison — for a shallow completion (e.g. Gartley/Bat, AD/XA < 1) D can sit on
    # either side of X price-wise while still being the same "type" of extreme.
    last_pivot_type = pivots[-1]["type"]
    direction = "bullish" if last_pivot_type == "LOW" else "bearish"

    for name, bands in _HARMONIC_PATTERNS.items():
        if (bands["ab_xa"][0] <= ab_xa <= bands["ab_xa"][1]
                and bands["bc_ab"][0] <= bc_ab <= bands["bc_ab"][1]
                and bands["ad_xa"][0] <= ad_xa <= bands["ad_xa"][1]):
            score = 0.8 if direction == "bullish" else -0.8
            return HarmonicResult(
                pattern=name, direction=direction, completion_ratio=round(ad_xa, 3), score=score,
                note=f"{name} pattern completed at D (AD/XA={ad_xa:.3f}); "
                     f"projects a {direction} reversal from the current swing point.",
            )

    return HarmonicResult(
        None, None, round(ad_xa, 3), 0.0,
        f"No XABCD pattern match (AB/XA={ab_xa:.3f}, BC/AB={bc_ab:.3f}, AD/XA={ad_xa:.3f}).",
    )


# ---------------------------------------------------------------------------
# 6. Gann Time Cycles (accuracy-roadmap Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class GannTimeResult:
    active_cycle_days: Optional[int]
    reference_pivot_date: Optional[str]
    days_since_pivot: Optional[int]
    score: float
    note: str


# Gann "cycle" day-counts used here: even divisions of Gann's 360-degree annual circle
# (360/8, 360/4, 360*3/8, 360/2, 360*3/4, 360) plus the two most commonly cited
# non-360-derived Gann numbers (49, 144). This is ONE documented convention among
# several in circulation for "Gann time cycles" — there is no single universally
# agreed formula (flagged explicitly as contested in the accuracy roadmap, Section 14)
# — counted in calendar days from the most recent major ZigZag pivot, matching how
# Gann's own square-of-9/circle-of-360 time counting is conventionally applied.
_GANN_CYCLE_DAYS = [45, 49, 90, 135, 144, 180, 270, 360]
_GANN_WINDOW_TOLERANCE_DAYS = 3


def compute_gann_time_cycles(df: pd.DataFrame, pct_threshold: float = 4.0) -> GannTimeResult:
    """
    Time-based (not price-based) reversal-window flag: is today within
    `_GANN_WINDOW_TOLERANCE_DAYS` calendar days of one of the fixed Gann cycle
    intervals counted forward from the most recent major ZigZag pivot?

    Gann time theory predicts WHEN a reversal may occur, not which direction. This
    implementation scores a bearish reversal risk if price has risen since the
    reference pivot (an active cycle window suggests upside exhaustion) and a bullish
    reversal risk if it has fallen — the standard retail-Gann interpretation, but
    itself a debatable modeling choice, not a settled formula. Per the roadmap's own
    framing, a "this doesn't help" backtest result for this indicator is a legitimate,
    expected outcome to report honestly, not a bug to re-tune away.
    """
    zz = compute_zigzag(df, pct_threshold=pct_threshold)
    pivots = zz.all_pivots
    if not pivots:
        return GannTimeResult(None, None, None, 0.0,
                               "No confirmed ZigZag pivot available as a time-cycle reference point.")

    ref = pivots[-1]
    ref_date = pd.Timestamp(ref["date"])
    latest_date = pd.Timestamp(df["date"].iloc[-1])
    days_since = (latest_date - ref_date).days

    active_cycle = next((d for d in _GANN_CYCLE_DAYS if abs(days_since - d) <= _GANN_WINDOW_TOLERANCE_DAYS), None)
    if active_cycle is None:
        return GannTimeResult(
            None, str(ref_date.date()), days_since, 0.0,
            f"{days_since} calendar days since last major pivot ({ref_date.date()}); no Gann cycle window active.",
        )

    price_then, price_now = ref["price"], float(df["close"].iloc[-1])
    trended_up = price_now > price_then
    direction_score = -0.6 if trended_up else 0.6

    return GannTimeResult(
        active_cycle_days=active_cycle,
        reference_pivot_date=str(ref_date.date()),
        days_since_pivot=days_since,
        score=direction_score,
        note=f"{days_since}d since {ref_date.date()} pivot matches the {active_cycle}-day Gann cycle window "
             f"(+/-{_GANN_WINDOW_TOLERANCE_DAYS}d); {'upside' if trended_up else 'downside'} exhaustion risk flagged.",
    )


# ---------------------------------------------------------------------------
# 7. Composite classification
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
    # Hysteresis transparency (Phase 2) — set only by classify_cycle_stateful, when a
    # candidate phase is building but not yet confirmed. Deliberately NOT folded into
    # transition_risk/transition_watch: an earlier version tried exactly that (see the
    # comment in classify_cycle_stateful) and it measurably backfired — it inflated
    # recall against independently-labeled turns only by flagging "elevated" on 61-74%
    # of all evaluation points, while precision-vs-base-rate lift stayed flat at
    # ~0.94-0.96 (no better than the pre-Phase-2 baseline's 0.96) across every dwell-
    # progress threshold tried. That's a real, reproducible negative finding about the
    # underlying trend/momentum divergence signal's correlation with genuine turns —
    # not a bug to tune away — so transition_risk stays the plain magnitude-based read
    # and this field exists purely for live/dashboard transparency into hysteresis
    # state, without corrupting the metric the roadmap asks to be honestly reported.
    pending_phase: Optional[Phase] = None
    dwell_progress: Optional[str] = None  # e.g. "3/5" candidate confirmations


# accuracy-roadmap Phase 4 note (tier-weight recalibration, P1): these three weights
# were investigated for empirical recalibration, same discipline as the transition_risk
# magnitude thresholds. Findings, kept here so a future pass doesn't re-derive them:
#   - core/supporting/contextual (0.6/0.3/0.1) themselves cannot be empirically
#     recalibrated with currently available data — valuation/liquidity/earnings
#     (supporting tier) have no clean backtestable daily history (same limitation
#     documented throughout this module and scripts/backtest_cycle_phase.py), and
#     candlestick/harmonic/gann (contextual tier) already have a definitive backtest
#     result at their current 0.1 weight (scripts/evaluate_indicators.py: zero effect
#     on any phase decision across 626 points) — there's no live "at what weight would
#     this matter" question worth answering when 2 of the 3 were verdicted DROP and
#     the 3rd (Harmonic) explicitly should not be weighted into production yet anyway.
#   - The one internally-backtestable question — should zigzag and trend_context split
#     the directional axis's core weight 50/50, as they currently do, or unevenly —
#     WAS grid-searched (0.6/0.6 baseline vs 0.9/0.3, 0.3/0.9, 1.0/0.2, 0.2/1.0) on the
#     same calibration (<=2020) / validation (2021-2023) split used for min_dwell.
#     Result: calibration and validation DISAGREE on which direction to shift (calib
#     recall favors trend-heavy splits, validation recall favors the current 50/50 —
#     the best performer on validation), and the recall deltas involved are within
#     noise given how few independently-labeled turns exist per window (8-15 — a
#     single turn flipping hit/miss swings recall by 7-13 points). This is exactly the
#     "calibration and validation disagree sharply" signal the roadmap says means
#     don't fit to test, not a green light to pick a side. Kept at 50/50 — no change.
_TIER_WEIGHTS = {"core": 0.6, "supporting": 0.3, "contextual": 0.1}

# Axis membership for the 4-pillar vote (accuracy-roadmap Phase 2). Adapted from the
# Section 10 diagram, with one deliberate change from that diagram's literal reading:
#   Directional axis (full vote): trend_context + zigzag — genuine price trend/
#     structure. This deliberately does NOT include valuation as a full vote, unlike
#     the diagram's solid "V --> D1" line. Reason: valuation is a level/mean-reversion
#     signal, not a direction signal, and letting it fully co-vote with trend produces
#     mislabeled quadrants — e.g. a mild uptrend + rich valuation + rising momentum
#     computes to a NEGATIVE directional score, which (with momentum still up) reads
#     as ACCUMULATION ("basing after a decline"). That's backwards: there was no
#     decline, the correct read for "still trending up but fragile/expensive" is
#     DISTRIBUTION-risk, not ACCUMULATION. (See tests/unit/test_cycle_signals.py::
#     TestFourPillarVote for the worked case.) So valuation is instead a
#     tiebreaker/modifier on the directional axis, symmetric with how the diagram
#     already treats liquidity as a modifier on momentum (dotted "L -.-> D2" line).
#   Momentum axis (full vote): stc + earnings_momentum — both are rate-of-change
#     signals on the *same* underlying question (is the driving force accelerating or
#     decelerating), so earnings gets full-vote status alongside STC, unlike valuation.
#   Both modifiers (erp on directional, liquidity on momentum) only nudge their axis
#     when it is near-zero/indecisive — they break ties, they don't outvote a clear
#     trend or momentum read. This still fixes roadmap Section 3.1's core finding
#     (valuation/liquidity/earnings can now change cycle_phase, not just
#     composite_score/confidence after the phase was already fixed by 2 signals) while
#     keeping the phase taxonomy's own definitions (market_cycle.md: "DISTRIBUTION:
#     trend still rising but momentum weakening") anchored to what trend actually means.
_DIRECTIONAL_SIGNALS = {"trend_context", "zigzag"}
_MOMENTUM_SIGNALS = {"stc", "earnings_momentum"}
_DIRECTIONAL_MODIFIER_SIGNAL = "erp"
_MOMENTUM_MODIFIER_SIGNAL = "liquidity"

# Phase 3's 3 new indicators (candlestick, harmonic, gann) are reversal/momentum-
# timing signals — assigned membership on the momentum axis (not directional; they
# don't establish a "trend," they call a turn) at CONTEXTUAL tier. Membership alone
# does NOT mean they vote: classify_cycle only sums evidence items that are actually
# present in the `evidence` list passed to it, so unless a caller explicitly includes
# a "candlestick"/"harmonic"/"gann" EvidenceItem, this is a no-op — the live agent
# (cycle_tools.py) does not pass them, only scripts/evaluate_indicators.py does, for
# the ablation study. No indicator here votes in production until the roadmap's
# Section 9.1 evaluation shows it earns that weight.
_MOMENTUM_SIGNALS |= {"candlestick", "harmonic", "gann"}
_MODIFIER_TIEBREAK_EPSILON = 0.10  # only nudges an axis when it's already near-zero
_MODIFIER_TIEBREAK_WEIGHT = 0.15   # small nudge, not a vote — never flips a non-tied axis


def _axis_score(evidence: List[EvidenceItem], names: set) -> float:
    """Tier-weighted average of every evidence item whose signal name is in `names`.
    Returns 0.0 (treated as "up"/tie, matching the prior >= 0 convention) if none present."""
    items = [e for e in evidence if e.signal in names]
    if not items:
        return 0.0
    weighted_sum = sum(e.score * _TIER_WEIGHTS.get(e.tier, 0.1) for e in items)
    weight_total = sum(_TIER_WEIGHTS.get(e.tier, 0.1) for e in items)
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def _apply_tiebreak_modifier(axis_score: float, evidence: List[EvidenceItem], modifier_signal: str) -> float:
    """Nudges `axis_score` toward a modifier signal's reading, but only when the axis
    is already near zero (indecisive) — a tiebreaker, never an outright vote."""
    modifier_items = [e for e in evidence if e.signal == modifier_signal]
    if modifier_items and abs(axis_score) < _MODIFIER_TIEBREAK_EPSILON:
        return axis_score + modifier_items[0].score * _MODIFIER_TIEBREAK_WEIGHT
    return axis_score


def classify_cycle(evidence: List[EvidenceItem]) -> CycleClassification:
    """
    Weighted composite classification, 4-pillar-vote mechanism (accuracy-roadmap
    Phase 2 / Section 10). This is a single-point-in-time (stateless) read — see
    `classify_cycle_stateful` for the hysteresis-confirmed version used by the live
    agent and the backtest, which is what should actually be reported as "the" phase.

    Phase is still decided by a 2x2 direction x momentum matrix — the taxonomy itself
    (EXPANSION/DISTRIBUTION/CONTRACTION/ACCUMULATION) is preserved — but each axis is
    now a tier-weighted vote across every pillar assigned to it, not a single named
    signal read in isolation:
        directional up + momentum rising  -> EXPANSION
        directional up + momentum falling -> DISTRIBUTION   (topping / momentum divergence)
        directional down + momentum falling -> CONTRACTION
        directional down + momentum rising  -> ACCUMULATION (basing / momentum divergence)

    When only Core-tier evidence is supplied (trend_context + stc + zigzag, as in the
    price-structure-only walk-forward backtest — valuation/liquidity/earnings have no
    clean backtestable daily history, see scripts/backtest_cycle_phase.py), this
    degrades to directional=avg(trend_context, zigzag) and momentum=stc — a small but
    real change from the old trend_context-only directional read, and a much larger
    change than a no-op once valuation/liquidity/earnings evidence is actually present.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for e in evidence:
        w = _TIER_WEIGHTS.get(e.tier, 0.1)
        weighted_sum += e.score * w
        weight_total += w
    composite_score = round(weighted_sum / weight_total, 3) if weight_total > 0 else 0.0

    directional_score = _axis_score(evidence, _DIRECTIONAL_SIGNALS)
    momentum_score = _axis_score(evidence, _MOMENTUM_SIGNALS)

    directional_score = _apply_tiebreak_modifier(directional_score, evidence, _DIRECTIONAL_MODIFIER_SIGNAL)
    momentum_score = _apply_tiebreak_modifier(momentum_score, evidence, _MOMENTUM_MODIFIER_SIGNAL)

    directional_up = directional_score >= 0
    momentum_up = momentum_score >= 0

    if directional_up and momentum_up:
        phase: Phase = "EXPANSION"
        watch = "Expansion -> Distribution"
    elif directional_up and not momentum_up:
        phase = "DISTRIBUTION"
        watch = "Distribution -> Contraction"
    elif not directional_up and not momentum_up:
        phase = "CONTRACTION"
        watch = "Contraction -> Accumulation"
    else:
        phase = "ACCUMULATION"
        watch = "Accumulation -> Expansion"

    agree = sum(1 for e in evidence if (e.score >= 0) == (composite_score >= 0))
    confidence = round(agree / len(evidence), 2) if evidence else 0.0

    divergence = (directional_up and not momentum_up) or (not directional_up and momentum_up)
    magnitude = abs(directional_score - momentum_score)
    # Thresholds below (0.9 / 1.35) were calibrated against the walk-forward backtest
    # (scripts/backtest_cycle_phase.py, ~623 evaluation points over 2018-2026 Nifty
    # data) against the OLD trend_score/stc_score divergence. Carried over unchanged
    # for the new directional/momentum axis divergence, since on the Core-tier
    # backtest the two are close enough (both still core-signal-dominated) that
    # re-deriving them is Phase 4 work (tier-weight/threshold recalibration), not
    # Phase 2 — don't re-tune thresholds and the phase-vote mechanism in the same step.
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


# ---------------------------------------------------------------------------
# 5. Hysteresis / minimum-dwell confirmation (accuracy-roadmap Phase 2, point 2)
# ---------------------------------------------------------------------------

@dataclass
class HysteresisState:
    """Carries phase-confirmation state across evaluation points. Immutable — each
    call to apply_phase_hysteresis returns a new state rather than mutating this one,
    so the backtest's walk-forward loop and a live caller can each own their own copy
    without aliasing bugs."""
    confirmed_phase: Optional[Phase] = None
    candidate_phase: Optional[Phase] = None
    candidate_count: int = 0


def apply_phase_hysteresis(
    raw_phase: Phase, state: HysteresisState, min_dwell: int = 3
) -> "tuple[Phase, HysteresisState, bool]":
    """
    Minimum-dwell confirmation rule directly targeting the ~11-trading-day whipsaw
    found in the roadmap's baseline backtest (166 phase changes / 626 evaluation
    points, step=3 days => mean phase duration ~11 trading days).

    A phase change is only confirmed once the new instantaneous (raw) phase has been
    read `min_dwell` consecutive evaluation points in a row; until then, the
    previously-confirmed phase is still reported. This is deliberately a simple
    consecutive-count rule (not a magnitude threshold) so its only tunable parameter
    is a single integer, calibrated empirically (see scripts/backtest_cycle_phase.py
    --min-dwell and scripts/validate_cycle_classifier.py), not picked a priori.

    Returns (confirmed_phase, new_state, pending) where `pending` is True exactly
    when a phase change is actively building (raw_phase != confirmed_phase) but has
    not yet reached min_dwell — the caller should elevate transition_risk in this
    case, since a building-but-unconfirmed flip is by construction a leading signal
    of the phase change about to happen, not a coincidence.
    """
    if state.confirmed_phase is None:
        # Bootstrap: nothing to compare against yet, so the very first raw read is
        # confirmed immediately rather than waiting min_dwell points with no prior.
        return raw_phase, HysteresisState(confirmed_phase=raw_phase), False

    if raw_phase == state.confirmed_phase:
        # Back in line with the confirmed phase (or never left it) — cancel any
        # in-flight candidate so a single stray reading can't half-confirm later.
        return state.confirmed_phase, HysteresisState(confirmed_phase=state.confirmed_phase), False

    candidate_count = state.candidate_count + 1 if raw_phase == state.candidate_phase else 1

    if candidate_count >= min_dwell:
        return raw_phase, HysteresisState(confirmed_phase=raw_phase), True

    new_state = HysteresisState(
        confirmed_phase=state.confirmed_phase,
        candidate_phase=raw_phase,
        candidate_count=candidate_count,
    )
    return state.confirmed_phase, new_state, True


def classify_cycle_stateful(
    evidence: List[EvidenceItem], state: HysteresisState, min_dwell: int = 3
) -> "tuple[CycleClassification, HysteresisState]":
    """
    The phase reading that should actually be reported (live agent + backtest):
    the 4-pillar-vote classify_cycle() read, passed through the minimum-dwell
    hysteresis filter. min_dwell=1 recovers the un-hysteresised axis-vote read
    (every raw read confirms immediately), which is what lets Phase 2's two changes
    (axis-vote, hysteresis) be A/B tested independently with the same code path.
    """
    raw = classify_cycle(evidence)
    confirmed_phase, new_state, pending = apply_phase_hysteresis(raw.cycle_phase, state, min_dwell)

    # transition_risk/transition_watch are deliberately left as the plain, unmodified
    # raw magnitude-based read (see the CycleClassification.pending_phase docstring
    # for why an earlier attempt at folding hysteresis-pending status into this
    # measurably did not work: it inflated recall against independently-labeled turns
    # purely by widening the "elevated" base rate, without improving precision lift).
    # A pending build is instead surfaced separately, so live/dashboard consumers get
    # the transparency without the metric being silently gamed.
    is_pending_build = pending and confirmed_phase == state.confirmed_phase
    result = CycleClassification(
        cycle_phase=confirmed_phase,
        cycle_confidence=raw.cycle_confidence,
        transition_risk=raw.transition_risk,
        transition_watch=raw.transition_watch,
        composite_score=raw.composite_score,
        evidence=raw.evidence,
        pending_phase=raw.cycle_phase if is_pending_build else None,
        dwell_progress=f"{new_state.candidate_count}/{min_dwell}" if is_pending_build else None,
    )
    return result, new_state
