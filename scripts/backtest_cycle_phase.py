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
    classify_cycle,
    compute_stc,
    compute_trend_context,
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


def fetch_history(start: str) -> pd.DataFrame:
    hist = yf.Ticker("^NSEI").history(start=start, interval="1d")
    if hist.empty:
        raise RuntimeError("yfinance returned no data for ^NSEI — check network access.")
    df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def run_walk_forward(df: pd.DataFrame, step: int) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i in range(_MIN_WARMUP_BARS, n, step):
        window = df.iloc[: i + 1]

        zz = compute_zigzag(window)
        stc = compute_stc(window["close"])
        trend = compute_trend_context(window["close"])

        evidence: List[EvidenceItem] = [
            EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
            EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
            EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}%", trend.score, trend.note),
        ]
        result = classify_cycle(evidence)

        rows.append({
            "date": window["date"].iloc[-1],
            "close": float(window["close"].iloc[-1]),
            "cycle_phase": result.cycle_phase,
            "cycle_confidence": result.cycle_confidence,
            "transition_risk": result.transition_risk,
            "composite_score": result.composite_score,
            "zigzag_structure": zz.structure,
            "stc_value": stc.value,
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


def make_chart(results: pd.DataFrame, out_path: Path) -> None:
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

    for date_str, label in REFERENCE_TURNS:
        target = pd.Timestamp(date_str)
        if results["date"].min() <= target <= results["date"].max():
            ax.axvline(target, color="grey", linestyle="--", alpha=0.5, zorder=1)

    ax.set_title("Market Cycle Core Classifier — Walk-Forward Backtest (Nifty 50)")
    ax.set_ylabel("Nifty 50 Close")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Chart saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01", help="History start date (YYYY-MM-DD)")
    parser.add_argument("--step", type=int, default=3, help="Evaluate every N trading days (speed/resolution tradeoff)")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Nifty 50 daily history from %s...", args.start)
    df = fetch_history(args.start)
    logger.info("Fetched %d daily bars (%s to %s).", len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date())

    logger.info("Running walk-forward classification (step=%d bars)...", args.step)
    results = run_walk_forward(df, step=args.step)
    logger.info("Produced %d evaluation points.", len(results))

    csv_path = _OUTPUT_DIR / "cycle_backtest_history.csv"
    results.to_csv(csv_path, index=False)
    logger.info("History saved to %s", csv_path)

    chart_path = _OUTPUT_DIR / "cycle_backtest_chart.png"
    make_chart(results, chart_path)

    phase_dist = results["cycle_phase"].value_counts(normalize=True).round(3).to_dict()
    risk_dist = results["transition_risk"].value_counts(normalize=True).round(3).to_dict()
    n_phase_changes = (results["cycle_phase"] != results["cycle_phase"].shift(1)).sum() - 1

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Evaluation points: {len(results)}  |  Phase changes over the window: {n_phase_changes}")
    print(f"Phase distribution: {phase_dist}")
    print(f"Transition-risk distribution: {risk_dist}")

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
