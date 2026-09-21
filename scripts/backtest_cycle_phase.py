"""
Walk-Forward Backtest — Market Cycle Core Classifier.

Runs the Core-tier classifier (ZigZag + STC + trend context — the price-only signals
from cycle_signals.py) over real historical Nifty 50 data in a walk-forward fashion:
at each evaluation date, only data up to that date is used (no lookahead), exactly as
the live get_cycle_price_structure tool would see it on that day.

Scope note: this backtests the Core tier only. Valuation (ERP) and liquidity (M3) are
excluded because free sources (FMP/FRED) don't offer enough clean historical daily
history to backtest without a paid data plan — this is the same "insufficient_data"
honesty policy used elsewhere in this codebase, applied to the backtest itself rather
than glossing over the gap.

Usage:
    python scripts/backtest_cycle_phase.py
    python scripts/backtest_cycle_phase.py --start 2018-01-01 --step 3

Outputs (into scripts/output/):
    cycle_backtest_history.csv   — date, close, phase, confidence, transition_risk, composite_score
    cycle_backtest_chart.png     — Nifty price with phase-colored background + reference-turn markers
"""

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_MIN_WARMUP_BARS = 260  # >= 200DMA + margin

# Publicly known approximate Nifty 50 cycle turns, for illustrative reference only.
# This is NOT a rigorous labeled dataset (no formal peak/trough detection algorithm was
# used to pick these dates) — it is a sanity check: does transition_risk elevate around
# dates that are public knowledge as major turns, or does it stay flat/uninformative.
REFERENCE_TURNS = [
    ("2020-01-14", "Pre-COVID all-time high (before the Feb-Mar 2020 crash)"),
    ("2020-03-23", "COVID crash bottom"),
    ("2021-10-19", "2021 bull-market top (before the Oct 2021-Mar 2022 correction)"),
    ("2022-06-17", "2022 correction bottom (Fed-tightening selloff low)"),
    ("2024-09-27", "2024 top (before the Oct 2024-Feb 2025 correction)"),
]


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


def run_walk_forward(df: pd.DataFrame, step: int, min_dwell: int = 1) -> pd.DataFrame:
    """
    Walk-forward Core-tier classification.

    `min_dwell` controls the Phase 2 hysteresis filter (scripts/backtest_cycle_phase.py
    --min-dwell / cycle_signals.classify_cycle_stateful): min_dwell=1 confirms every
    raw axis-vote read immediately (no hysteresis — isolates Phase 2's "axis vote"
    change from its "hysteresis" change for independent A/B comparison against the
    pre-Phase-2 baseline); min_dwell>1 requires that many consecutive raw reads in
    the same direction before a phase change is confirmed, directly targeting the
    ~11-trading-day whipsaw found in the baseline backtest.
    """
    rows = []
    n = len(df)
    state = HysteresisState()
    for i in range(_MIN_WARMUP_BARS, n, step):
        window = df.iloc[: i + 1]

        zz = compute_zigzag(window)
        stc = compute_stc(window["close"])
        trend = compute_trend_context(window["close"])
        # Phase 3 (accuracy roadmap): the 3 new indicators are computed and logged
        # for the comparison/ablation study (scripts/evaluate_indicators.py) but are
        # NOT added to `evidence` here — they don't vote on the phase until the
        # evaluation says they earn it (roadmap Phase 3: "no indicator is added to
        # the core/supporting tier without backtest evidence it earns that weight").
        candles = compute_candlestick_patterns(window)
        harmonic = compute_harmonic_patterns(window)
        gann = compute_gann_time_cycles(window)
        # Volume indicators (mentor-suggested follow-up) — same "log, don't vote
        # until evaluated" treatment as the Phase 3 indicators above.
        obv = compute_obv(window["close"], window["volume"])
        cmf = compute_cmf(window)
        vol_confirm = compute_volume_confirmed_pivot(window, zz)

        evidence: List[EvidenceItem] = [
            EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
            EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
            EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}%", trend.score, trend.note),
        ]
        raw_result = classify_cycle(evidence)
        result, state = classify_cycle_stateful(evidence, state, min_dwell=min_dwell)

        rows.append({
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
            # Purely additive: does not change any column produced before this point.
            "zigzag_score": zz.score,
            "stc_score": stc.score,
            "trend_score": trend.score,
            "trend_price_vs_sma200_pct": trend.price_vs_sma200_pct,
            "trend_sma50_vs_sma200_pct": trend.sma50_vs_sma200_pct,
            "stc_direction": stc.direction,
            # Phase 2: raw (stateless, un-hysteresised) read vs the hysteresis-confirmed
            # phase actually reported — lets before/after and pending-vs-confirmed be
            # inspected directly from the CSV without re-running the backtest.
            "raw_phase": raw_result.cycle_phase,
            "raw_transition_risk": raw_result.transition_risk,
            # Phase 3: new-indicator scores, logged but not voting (see comment above).
            "candlestick_score": candles.score,
            "candlestick_patterns": ",".join(sorted({p["pattern"] for p in candles.patterns})),
            "harmonic_pattern": harmonic.pattern or "",
            "harmonic_score": harmonic.score,
            "gann_score": gann.score,
            "gann_active_cycle_days": gann.active_cycle_days if gann.active_cycle_days is not None else "",
            # Volume indicators, logged but not voting (see comment above).
            "obv_score": obv.score,
            "obv_z_score": obv.z_score,
            "cmf_score": cmf.score,
            "volume_confirmation_score": vol_confirm.score,
            "volume_confirmation_ratio": vol_confirm.volume_ratio,
        })

        if (i - _MIN_WARMUP_BARS) % (step * 100) == 0:
            logger.info("Processed %d/%d bars...", i, n)

    return pd.DataFrame(rows)


def check_reference_turns(results: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("REFERENCE-TURN SANITY CHECK (illustrative only, not a formal label set)")
    print("=" * 78)
    for date_str, label in REFERENCE_TURNS:
        target = pd.Timestamp(date_str)
        window = results[(results["date"] >= target - pd.Timedelta(days=20)) & (results["date"] <= target + pd.Timedelta(days=5))]
        if window.empty:
            print(f"\n{date_str} ({label}): no backtest coverage in this range.")
            continue
        risk_counts = window["transition_risk"].value_counts().to_dict()
        phase_at_turn = window.iloc[-1]["cycle_phase"] if not window.empty else "n/a"
        print(f"\n{date_str} ({label})")
        print(f"  transition_risk distribution in +/-20 trading days around this date: {risk_counts}")
        print(f"  classified phase at nearest evaluation point: {phase_at_turn}")


def make_chart(results: pd.DataFrame, out_path: Path, title: str = "Market Cycle Core Classifier — Walk-Forward Backtest (Nifty 50)",
                ylabel: str = "Nifty 50 Close", show_reference_turns: bool = True) -> None:
    phase_colors = {
        "EXPANSION": "#2E7D32",
        "DISTRIBUTION": "#F9A825",
        "CONTRACTION": "#C62828",
        "ACCUMULATION": "#1565C0",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(results["date"], results["close"], color="black", linewidth=1.0, zorder=3)

    for phase, color in phase_colors.items():
        mask = results["cycle_phase"] == phase
        ax.scatter(results.loc[mask, "date"], results.loc[mask, "close"], s=8, color=color, label=phase, zorder=2)

    if show_reference_turns:
        # Nifty-specific public turn dates — only meaningful as an overlay on the
        # index itself, not on an individual stock's price path.
        for date_str, label in REFERENCE_TURNS:
            target = pd.Timestamp(date_str)
            if results["date"].min() <= target <= results["date"].max():
                ax.axvline(target, color="grey", linestyle="--", alpha=0.5, zorder=1)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Chart saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01", help="History start date (YYYY-MM-DD)")
    parser.add_argument("--step", type=int, default=3, help="Evaluate every N trading days (speed/resolution tradeoff)")
    parser.add_argument(
        "--min-dwell", type=int, default=5,
        help="Phase 2 hysteresis: consecutive same-direction raw reads required to confirm a phase "
             "change. Default 5 is the calibrated, adopted value (scripts/calibrate_min_dwell.py; "
             "chosen on the 2018-2020 calibration window, confirmed on 2021-2023 validation, never "
             "re-tuned against the 2024+ test slice) — cuts mean phase duration whipsaw from ~11 to "
             "~46 trading days. Pass 1 to recover the pre-hysteresis (axis-vote-only) read.",
    )
    parser.add_argument(
        "--out-suffix", default="",
        help="Suffix appended to output filenames (e.g. '_dwell3') so A/B runs don't overwrite each other. "
             "Auto-derived from --ticker when that's set and this is left blank.",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Run this same walk-forward backtest on an individual stock instead of the Nifty 50 index, "
             "e.g. --ticker RELIANCE.NS. Same mechanism (axis-vote + hysteresis), same no-lookahead "
             "discipline, just a different price series. Reference-turn overlay is skipped (those dates "
             "are Nifty-specific, not meaningful for an individual stock).",
    )
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ticker = args.ticker or "^NSEI"
    out_suffix = args.out_suffix or (f"_{args.ticker.split('.')[0]}" if args.ticker else "")

    logger.info("Fetching daily history for %s from %s...", ticker, args.start)
    df = fetch_history(args.start, ticker=ticker)
    logger.info("Fetched %d daily bars (%s to %s).", len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date())

    logger.info("Running walk-forward classification (step=%d bars, min_dwell=%d)...", args.step, args.min_dwell)
    results = run_walk_forward(df, step=args.step, min_dwell=args.min_dwell)
    logger.info("Produced %d evaluation points.", len(results))

    csv_path = _OUTPUT_DIR / f"cycle_backtest_history{out_suffix}.csv"
    results.to_csv(csv_path, index=False)
    logger.info("History saved to %s", csv_path)

    chart_path = _OUTPUT_DIR / f"cycle_backtest_chart{out_suffix}.png"
    if args.ticker:
        make_chart(
            results, chart_path,
            title=f"Market Cycle Classifier — Walk-Forward Backtest ({args.ticker})",
            ylabel=f"{args.ticker} Close",
            show_reference_turns=False,
        )
    else:
        make_chart(results, chart_path)

    phase_dist = results["cycle_phase"].value_counts(normalize=True).round(3).to_dict()
    risk_dist = results["transition_risk"].value_counts(normalize=True).round(3).to_dict()
    n_phase_changes = (results["cycle_phase"] != results["cycle_phase"].shift(1)).sum() - 1
    mean_run_length_points = len(results) / (n_phase_changes + 1) if n_phase_changes >= 0 else float("nan")
    mean_run_length_days = mean_run_length_points * args.step

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Mean phase run length: {mean_run_length_points:.1f} evaluation points (~{mean_run_length_days:.1f} trading days)")
    print(f"Evaluation points: {len(results)}  |  Phase changes over the window: {n_phase_changes}")
    print(f"Phase distribution: {phase_dist}")
    print(f"Transition-risk distribution: {risk_dist}")

    if not args.ticker:
        check_reference_turns(results)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print(
        "This checks internal consistency and rough alignment with public-knowledge market\n"
        "turns — it is NOT a rigorous precision/recall backtest (that needs a formal peak/\n"
        "trough labeling algorithm, e.g. a fixed-lookback local-extrema detector, applied\n"
        "independently of this classifier, then scored against it). Treat this script's\n"
        "output as the first checkpoint, not the final validation."
    )


if __name__ == "__main__":
    main()
