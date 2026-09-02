"""
Rigorous Validation — Market Cycle Classifier vs. Independently-Labeled Turns.

Upgrades backtest_cycle_phase.py's illustrative "does it look right near 5 dates I
picked by hand" check into an actual precision/recall scorer:

1. Labels major price turns using a LOCAL-EXTREMA + MINIMUM-RETRACEMENT algorithm that
   is intentionally NOT the classifier's own ZigZag (cycle_signals.compute_zigzag) —
   using the same algorithm to both label the ground truth and produce the prediction
   would be circular validation. This is a separate, standalone implementation.
2. For each labeled turn, checks whether the classifier's transition_risk was elevated
   (medium/high) within a lead window BEFORE the turn — i.e. did it warn ahead of time,
   not just agree in hindsight.
3. Reports recall (fraction of real turns the model gave advance warning for) and a
   precision proxy (of all elevated-risk readings, what fraction were actually near a
   real turn) against the base rate, so "the model is right 60% of the time" can be
   read against "elevated risk fires on 35% of all days anyway."

Usage:
    python scripts/validate_cycle_classifier.py
    python scripts/validate_cycle_classifier.py --start 2016-01-01 --lead-days 15
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_cycle_phase import fetch_history, run_walk_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def find_major_turns(df: pd.DataFrame, window: int = 10, min_move_pct: float = 8.0) -> pd.DataFrame:
    """
    Standalone local-extrema turn labeler — deliberately NOT cycle_signals.compute_zigzag,
    so the ground truth used to score the classifier isn't produced by the classifier's
    own swing-detection logic (that would be circular).

    A bar is a raw local max/min if it's the highest/lowest close within +/- `window`
    trading days. Raw extrema are then filtered to keep only alternating max/min pairs
    where the move from the prior kept extremum is >= min_move_pct — i.e. "major" turns,
    not every local wiggle.
    """
    close = df["close"].to_numpy()
    n = len(close)
    raw_extrema: List[Tuple[int, str]] = []

    for i in range(window, n - window):
        segment = close[i - window: i + window + 1]
        if close[i] == segment.max():
            raw_extrema.append((i, "HIGH"))
        elif close[i] == segment.min():
            raw_extrema.append((i, "LOW"))

    if not raw_extrema:
        return pd.DataFrame(columns=["date", "type", "price"])

    kept = [raw_extrema[0]]
    for idx, kind in raw_extrema[1:]:
        last_idx, last_kind = kept[-1]
        if kind == last_kind:
            # Same type as the last kept extremum — keep whichever is more extreme.
            if (kind == "HIGH" and close[idx] > close[last_idx]) or (kind == "LOW" and close[idx] < close[last_idx]):
                kept[-1] = (idx, kind)
            continue
        move_pct = abs(close[idx] - close[last_idx]) / close[last_idx] * 100.0
        if move_pct >= min_move_pct:
            kept.append((idx, kind))
        # else: too small a move to count as a "major" turn — drop it (implicit skip).

    rows = [{"date": df["date"].iloc[idx], "type": kind, "price": float(close[idx])} for idx, kind in kept]
    return pd.DataFrame(rows)


def score_lead_time_recall(results: pd.DataFrame, turns: pd.DataFrame, lead_days: int) -> dict:
    """
    For each labeled turn, was transition_risk in {medium, high} at any evaluation
    point within `lead_days` calendar days BEFORE the turn (and not after)?
    """
    hits = 0
    detail = []
    for _, turn in turns.iterrows():
        window_start = turn["date"] - pd.Timedelta(days=lead_days)
        window = results[(results["date"] >= window_start) & (results["date"] < turn["date"])]
        elevated = window[window["transition_risk"].isin(["medium", "high"])]
        hit = len(elevated) > 0
        hits += int(hit)
        detail.append({
            "turn_date": str(turn["date"].date()),
            "turn_type": turn["type"],
            "advance_warning_given": hit,
            "evaluation_points_in_lead_window": len(window),
        })

    recall = round(hits / len(turns), 3) if len(turns) > 0 else 0.0
    return {"recall": recall, "hits": hits, "total_turns": len(turns), "detail": detail}


def score_precision_vs_base_rate(results: pd.DataFrame, turns: pd.DataFrame, lead_days: int) -> dict:
    """
    Of all elevated-risk (medium/high) evaluation points, what fraction fall within
    `lead_days` of an actual labeled turn (in either direction)? Compared against the
    base rate of elevated-risk readings across the whole series, so a precision number
    is only meaningful relative to how often the model fires anyway.
    """
    turn_dates = turns["date"].to_numpy()
    elevated = results[results["transition_risk"].isin(["medium", "high"])].copy()

    def _near_a_turn(d) -> bool:
        deltas = np.abs((turn_dates - np.datetime64(d)) / np.timedelta64(1, "D"))
        return bool((deltas <= lead_days).any())

    elevated["near_turn"] = elevated["date"].apply(_near_a_turn)
    precision = round(elevated["near_turn"].mean(), 3) if len(elevated) > 0 else 0.0
    base_rate = round(len(elevated) / len(results), 3) if len(results) > 0 else 0.0

    all_dates = results["date"].copy()
    all_near_turn = all_dates.apply(_near_a_turn)
    overall_near_turn_rate = round(all_near_turn.mean(), 3)

    return {
        "precision_elevated_reading_is_near_a_real_turn": precision,
        "base_rate_of_elevated_readings_overall": base_rate,
        "share_of_all_days_that_are_near_a_turn": overall_near_turn_rate,
        "lift_over_random": round(precision / overall_near_turn_rate, 2) if overall_near_turn_rate > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--label-window", type=int, default=10, help="Bars each side for local-extrema detection")
    parser.add_argument("--min-move-pct", type=float, default=8.0, help="Minimum %% move to count as a major turn")
    parser.add_argument("--lead-days", type=int, default=15, help="Calendar days of advance warning credited")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Nifty 50 history from %s...", args.start)
    df = fetch_history(args.start)
    logger.info("Fetched %d bars (%s to %s).", len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date())

    logger.info("Running walk-forward classification...")
    results = run_walk_forward(df, step=args.step)
    logger.info("Produced %d evaluation points.", len(results))

    logger.info("Labeling major turns independently (window=%d bars, min_move=%.1f%%)...", args.label_window, args.min_move_pct)
    turns = find_major_turns(df, window=args.label_window, min_move_pct=args.min_move_pct)
    logger.info("Found %d independently-labeled major turns.", len(turns))

    turns_csv = _OUTPUT_DIR / "independent_labeled_turns.csv"
    turns.to_csv(turns_csv, index=False)

    recall_report = score_lead_time_recall(results, turns, lead_days=args.lead_days)
    precision_report = score_precision_vs_base_rate(results, turns, lead_days=args.lead_days)

    print("\n" + "=" * 78)
    print("INDEPENDENT TURN LABELS (local extrema, min %.1f%% move, NOT the classifier's own ZigZag)" % args.min_move_pct)
    print("=" * 78)
    print(turns.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"RECALL @ {args.lead_days}-DAY LEAD WINDOW")
    print("=" * 78)
    print(f"Of {recall_report['total_turns']} labeled major turns, transition_risk was elevated")
    print(f"(medium/high) at least once in the {args.lead_days} days BEFORE {recall_report['hits']} of them.")
    print(f"Recall: {recall_report['recall']:.1%}")
    print()
    for d in recall_report["detail"]:
        flag = "YES" if d["advance_warning_given"] else "NO "
        print(f"  [{flag}]  {d['turn_date']}  ({d['turn_type']:<4})  eval points in lead window: {d['evaluation_points_in_lead_window']}")

    print("\n" + "=" * 78)
    print("PRECISION vs. BASE RATE")
    print("=" * 78)
    for k, v in precision_report.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print(
        "Recall answers: \"when a major turn happened, did the model warn in advance?\"\n"
        "Precision-vs-base-rate answers: \"when the model warns, is that meaningfully more\n"
        "often near a real turn than chance?\" A lift > 1.0 means yes; a lift near 1.0 means\n"
        "the model's warnings are no more informative than its unconditional firing rate.\n"
        "This is Core-tier only (no valuation/liquidity/earnings — those don't have enough\n"
        "clean free historical daily coverage to backtest), and the turn threshold\n"
        f"(>= {args.min_move_pct}% move) and lead window ({args.lead_days} days) are both\n"
        "adjustable parameters, not fixed truths — rerun with different values before\n"
        "quoting a single number as definitive."
    )

    report_path = _OUTPUT_DIR / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Recall @ {args.lead_days}-day lead: {recall_report['recall']:.1%} ({recall_report['hits']}/{recall_report['total_turns']} turns)\n")
        for k, v in precision_report.items():
            f.write(f"{k}: {v}\n")
    logger.info("Summary written to %s", report_path)


if __name__ == "__main__":
    main()
