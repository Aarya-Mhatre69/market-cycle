"""
Calibrate the Phase 2 hysteresis min-dwell parameter — accuracy-roadmap Section 7.

Grid-searches min_dwell (consecutive raw reads required to confirm a phase change,
see cycle_signals.apply_phase_hysteresis) on the CALIBRATION window only, confirms
the choice generalizes on the VALIDATION window, and reports final numbers on the
TEST window exactly once (touch-once discipline, roadmap Section 7 / 11).

Windows (roadmap Section 7):
    Calibration: earliest available (2018-01-01, matching the existing backtest
                 default start) -> 2020-12-31   (tune min_dwell here)
    Validation:  2021-01-01 -> 2023-12-31        (confirm the choice generalizes)
    Test:        2024-01-01 -> present            (report once, do not re-tune after)

For each candidate min_dwell, walk-forward classification is run ONCE over the full
2018-present history (hysteresis state is inherently continuous/causal — a 2021
reading legitimately depends on 2018-2020 history), then sliced into the three
windows for independent scoring, reusing the exact same turn-labeling and
recall/precision functions validate_cycle_classifier.py uses for the headline
numbers, so this calibration is scored the same way the final claim will be.

Usage:
    python scripts/calibrate_min_dwell.py
    python scripts/calibrate_min_dwell.py --candidates 1,2,3,4,5,6,8,10
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_cycle_phase import fetch_history, run_walk_forward  # noqa: E402
from validate_cycle_classifier import (  # noqa: E402
    find_major_turns,
    score_lead_time_recall,
    score_precision_vs_base_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

CALIBRATION_END = "2020-12-31"
VALIDATION_END = "2023-12-31"


def _window_metrics(results: pd.DataFrame, turns: pd.DataFrame, start: str, end: str, lead_days: int) -> dict:
    r = results[(results["date"] >= start) & (results["date"] <= end)]
    t = turns[(turns["date"] >= start) & (turns["date"] <= end)]
    if r.empty or t.empty:
        return {"n_points": len(r), "n_turns": len(t), "recall": float("nan"), "lift": None, "mean_run_days": float("nan")}

    n_changes = (r["cycle_phase"] != r["cycle_phase"].shift(1)).sum()
    # shift(1) on the first row of the slice compares against NaN (out-of-window), which
    # pandas counts as a "change" — subtract 1 to correct for that boundary artifact.
    n_changes = max(n_changes - 1, 0)
    mean_run_points = len(r) / (n_changes + 1)

    recall_report = score_lead_time_recall(r, t, lead_days=lead_days)
    precision_report = score_precision_vs_base_rate(r, t, lead_days=lead_days)

    return {
        "n_points": len(r),
        "n_turns": len(t),
        "n_phase_changes": int(n_changes),
        "mean_run_points": round(mean_run_points, 2),
        "recall": recall_report["recall"],
        "hits": recall_report["hits"],
        "lift": precision_report["lift_over_random"],
        "precision": precision_report["precision_elevated_reading_is_near_a_real_turn"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--lead-days", type=int, default=15)
    parser.add_argument("--label-window", type=int, default=10)
    parser.add_argument("--min-move-pct", type=float, default=8.0)
    parser.add_argument("--candidates", default="1,2,3,4,5,6,8,10", help="Comma-separated min_dwell values to try")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [int(c) for c in args.candidates.split(",")]

    logger.info("Fetching Nifty 50 history from %s (fetched once, reused for every candidate)...", args.start)
    df = fetch_history(args.start)
    turns = find_major_turns(df, window=args.label_window, min_move_pct=args.min_move_pct)
    logger.info("Fetched %d bars, %d independently-labeled turns.", len(df), len(turns))

    today = str(pd.Timestamp.today().date())
    rows = []
    per_dwell_results = {}
    for min_dwell in candidates:
        logger.info("Running walk-forward with min_dwell=%d...", min_dwell)
        results = run_walk_forward(df, step=args.step, min_dwell=min_dwell)
        per_dwell_results[min_dwell] = results

        calib = _window_metrics(results, turns, args.start, CALIBRATION_END, args.lead_days)
        valid = _window_metrics(results, turns, CALIBRATION_END, VALIDATION_END, args.lead_days)
        test = _window_metrics(results, turns, VALIDATION_END, today, args.lead_days)

        rows.append({
            "min_dwell": min_dwell,
            "calib_mean_run_days": round(calib["mean_run_points"] * args.step, 1) if not pd.isna(calib.get("mean_run_points", float("nan"))) else None,
            "calib_recall": calib["recall"],
            "calib_lift": calib["lift"],
            "valid_mean_run_days": round(valid["mean_run_points"] * args.step, 1) if not pd.isna(valid.get("mean_run_points", float("nan"))) else None,
            "valid_recall": valid["recall"],
            "valid_lift": valid["lift"],
            "test_mean_run_days": round(test["mean_run_points"] * args.step, 1) if not pd.isna(test.get("mean_run_points", float("nan"))) else None,
            "test_recall": test["recall"],
            "test_lift": test["lift"],
        })

    grid = pd.DataFrame(rows)
    grid_path = _OUTPUT_DIR / "hysteresis_calibration_grid.csv"
    grid.to_csv(grid_path, index=False)

    print("\n" + "=" * 100)
    print(f"HYSTERESIS MIN-DWELL CALIBRATION GRID  (calibration <= {CALIBRATION_END}, "
          f"validation <= {VALIDATION_END}, test = rest)")
    print("=" * 100)
    print(grid.to_string(index=False))
    print(f"\nSaved to {grid_path}")

    print("\n" + "=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "Pick min_dwell using the CALIBRATION columns only (mean_run_days should rise well above\n"
        "the ~11-day baseline whipsaw without recall collapsing; lift should move meaningfully\n"
        "above 1.0). Then confirm the choice on VALIDATION before locking it in. TEST numbers are\n"
        "printed for completeness but must be treated as reported-once — don't re-pick min_dwell\n"
        "after looking at them; if calibration and validation disagree sharply, that's a signal to\n"
        "widen the candidate grid or reconsider the mechanism, not to fit to test."
    )


if __name__ == "__main__":
    main()
