"""
Walk-forward phase history for any price series (index or single stock).

Lives in the package (not scripts/backtest_cycle_phase.py) so the deployed dashboard
can use it: the dashboard runs on the slim requirements-dashboard.txt (no matplotlib),
and the backtest script imports matplotlib at the top. scripts/backtest_cycle_phase.py
re-imports these two names, so every script that does
`from backtest_cycle_phase import fetch_history, run_walk_forward` keeps working
unchanged.
"""

import logging
from typing import List

import pandas as pd
import yfinance as yf

from shankh.agents.market.cycle_signals import (
    EvidenceItem,
    HysteresisState,
    classify_cycle,
    classify_cycle_stateful,
    compute_candlestick_patterns,
    compute_cmf,
    compute_gann_time_cycles,
    compute_harmonic_patterns,
    compute_obv,
    compute_stc,
    compute_trend_context,
    compute_volume_confirmed_pivot,
    compute_zigzag,
)

logger = logging.getLogger(__name__)

_MIN_WARMUP_BARS = 260  # >= 200DMA + margin


def fetch_history(start: str, ticker: str = "^NSEI") -> pd.DataFrame:
    """`ticker` defaults to the Nifty 50 index; pass e.g. 'RELIANCE.NS' to run this
    same walk-forward backtest on an individual stock instead — same mechanism,
    same no-lookahead discipline, just a different price series."""
    hist = yf.Ticker(ticker).history(start=start, interval="1d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} — check the symbol or network access.")
    df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def run_walk_forward(
    df: pd.DataFrame,
    step: int,
    min_dwell: int = 1,
    stc_level_weight: float = 0.5,
    include_context_indicators: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward Core-tier classification.

    `min_dwell` controls the Phase 2 hysteresis filter (scripts/backtest_cycle_phase.py
    --min-dwell / cycle_signals.classify_cycle_stateful): min_dwell=1 confirms every
    raw axis-vote read immediately (no hysteresis — isolates Phase 2's "axis vote"
    change from its "hysteresis" change for independent A/B comparison against the
    pre-Phase-2 baseline); min_dwell>1 requires that many consecutive raw reads in
    the same direction before a phase change is confirmed, directly targeting the
    ~11-trading-day whipsaw found in the baseline backtest.

    `include_context_indicators=False` skips the candlestick/harmonic/Gann/volume
    columns (computed and logged only for the indicator-evaluation scripts, never
    voting) — several times faster, which is what the dashboard's on-demand
    per-stock chart wants. Default True keeps the full logged schema every
    evaluation script relies on.
    """
    rows = []
    n = len(df)
    state = HysteresisState()
    for i in range(_MIN_WARMUP_BARS, n, step):
        window = df.iloc[: i + 1]

        zz = compute_zigzag(window)
        stc = compute_stc(window["close"], level_weight=stc_level_weight)
        trend = compute_trend_context(window["close"])

        evidence: List[EvidenceItem] = [
            EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
            EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
            EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}%", trend.score, trend.note),
        ]
        raw_result = classify_cycle(evidence)
        result, state = classify_cycle_stateful(evidence, state, min_dwell=min_dwell)

        row = {
            "date": window["date"].iloc[-1],
            "close": float(window["close"].iloc[-1]),
            "cycle_phase": result.cycle_phase,
            "cycle_confidence": result.cycle_confidence,
            "transition_risk": result.transition_risk,
            "composite_score": result.composite_score,
            "zigzag_structure": zz.structure,
            "stc_value": stc.value,
            # Per-signal scores (Phase 1 of the accuracy roadmap) — logged so indicator
            # comparison / ablation work (Phase 3) and the 4-pillar-vote mechanism (Phase 2)
            # can be evaluated against this same walk-forward history without re-running it.
            "zigzag_score": zz.score,
            "stc_score": stc.score,
            "trend_score": trend.score,
            "trend_price_vs_sma200_pct": trend.price_vs_sma200_pct,
            "trend_sma50_vs_sma200_pct": trend.sma50_vs_sma200_pct,
            "stc_direction": stc.direction,
            # Phase 2: raw (stateless, un-hysteresised) read vs the hysteresis-confirmed
            # phase actually reported.
            "raw_phase": raw_result.cycle_phase,
            "raw_transition_risk": raw_result.transition_risk,
            # Raw continuous input to the transition_risk bucketing, logged so a
            # threshold grid search can re-bucket post-hoc without re-running this.
            "transition_magnitude": raw_result.transition_magnitude,
            "transition_divergence": raw_result.transition_divergence,
        }

        if include_context_indicators:
            # Phase 3 + volume indicators: computed and logged for the comparison/
            # ablation study (scripts/evaluate_indicators.py,
            # scripts/volume_experiment_report.py) but NOT added to `evidence` — they
            # don't vote on the phase until the evaluation says they earn it.
            candles = compute_candlestick_patterns(window)
            harmonic = compute_harmonic_patterns(window)
            gann = compute_gann_time_cycles(window)
            obv = compute_obv(window["close"], window["volume"])
            cmf = compute_cmf(window)
            vol_confirm = compute_volume_confirmed_pivot(window, zz)
            row.update({
                "candlestick_score": candles.score,
                "candlestick_patterns": ",".join(sorted({p["pattern"] for p in candles.patterns})),
                "harmonic_pattern": harmonic.pattern or "",
                "harmonic_score": harmonic.score,
                "gann_score": gann.score,
                "gann_active_cycle_days": gann.active_cycle_days if gann.active_cycle_days is not None else "",
                "obv_score": obv.score,
                "obv_z_score": obv.z_score,
                "cmf_score": cmf.score,
                "volume_confirmation_score": vol_confirm.score,
                "volume_confirmation_ratio": vol_confirm.volume_ratio,
            })

        rows.append(row)

        if (i - _MIN_WARMUP_BARS) % (step * 100) == 0:
            logger.info("Processed %d/%d bars...", i, n)

    return pd.DataFrame(rows)
