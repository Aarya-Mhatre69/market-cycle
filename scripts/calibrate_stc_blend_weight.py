"""
Calibrate STC's internal level/slope blend weight — brutal-review audit finding #15.

compute_stc's score used to hardcode `0.5*level_component + 0.5*slope_component` —
never empirically grid-searched, unlike the directional-axis zigzag/trend_context
50/50 split (see cycle_signals.py's Phase 4 tier-weight comment), which WAS
grid-searched. Now parameterized as `level_weight` (cycle_signals.compute_stc),
this sweeps it through the exact same discipline: walk-forward re-run per candidate
(level_weight changes stc.score itself, which feeds the momentum axis and hysteresis,
so — unlike transition_risk's thresholds — this can't be re-bucketed post-hoc), same
calibration (<=2020) / validation (2021-2023) split, same
find_major_turns/score_lead_time_recall/score_precision_vs_base_rate scoring as
calibrate_min_dwell.py.

Usage:
    python scripts/calibrate_stc_blend_weight.py
    python scripts/calibrate_stc_blend_weight.py --candidates 0.2,0.3,0.4,0.5,0.6,0.7,0.8
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
_CURRENT_LEVEL_WEIGHT = 0.5


def _window_metrics(results: pd.DataFrame, turns: pd.DataFrame, start: str, end: str, lead_days: int) -> dict:
    r = results[(results["date"] >= start) & (results["date"] <= end)]
    t = turns[(turns["date"] >= start) & (turns["date"] <= end)]
    if r.empty or t.empty:
        return {"n_points": len(r), "n_turns": len(t), "recall": float("nan"), "lift": None}
    n_changes = (r["cycle_phase"] != r["cycle_phase"].shift(1)).sum()
    n_changes = max(n_changes - 1, 0)
    mean_run_points = len(r) / (n_changes + 1)
    recall_report = score_lead_time_recall(r, t, lead_days=lead_days)
    precision_report = score_precision_vs_base_rate(r, t, lead_days=lead_days)
    return {
        "n_points": len(r), "n_turns": len(t), "n_phase_changes": int(n_changes),
        "mean_run_points": round(mean_run_points, 2),
        "recall": recall_report["recall"], "hits": recall_report["hits"],
        "lift": precision_report["lift_over_random"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--min-dwell", type=int, default=5)
    parser.add_argument("--lead-days", type=int, default=15)
    parser.add_argument("--candidates", default="0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [float(c) for c in args.candidates.split(",")]
    if _CURRENT_LEVEL_WEIGHT not in candidates:
        candidates.append(_CURRENT_LEVEL_WEIGHT)
    candidates = sorted(set(candidates))

    logger.info("Fetching Nifty 50 history from %s (fetched once, reused for every candidate)...", args.start)
    df = fetch_history(args.start)
    turns = find_major_turns(df, window=10, min_move_pct=8.0)
    logger.info("Fetched %d bars, %d independently-labeled turns.", len(df), len(turns))

    rows = []
    for level_weight in candidates:
        logger.info("Running walk-forward with STC level_weight=%.1f...", level_weight)
        results = run_walk_forward(df, step=args.step, min_dwell=args.min_dwell, stc_level_weight=level_weight)
        calib = _window_metrics(results, turns, args.start, CALIBRATION_END, args.lead_days)
        valid = _window_metrics(results, turns, CALIBRATION_END, VALIDATION_END, args.lead_days)
        rows.append({
            "level_weight": level_weight,
            "is_current": level_weight == _CURRENT_LEVEL_WEIGHT,
            "calib_mean_run_days": round(calib["mean_run_points"] * args.step, 1) if not pd.isna(calib.get("mean_run_points", float("nan"))) else None,
            "calib_recall": calib["recall"], "calib_lift": calib["lift"],
            "valid_mean_run_days": round(valid["mean_run_points"] * args.step, 1) if not pd.isna(valid.get("mean_run_points", float("nan"))) else None,
            "valid_recall": valid["recall"], "valid_lift": valid["lift"],
        })

    grid = pd.DataFrame(rows)
    grid_path = _OUTPUT_DIR / "stc_blend_weight_grid.csv"
    grid.to_csv(grid_path, index=False)

    print("\n" + "=" * 100)
    print(f"STC LEVEL/SLOPE BLEND WEIGHT GRID  (calibration <= {CALIBRATION_END}, validation <= {VALIDATION_END})")
    print("=" * 100)
    print(grid.to_string(index=False))
    print(f"\nSaved to {grid_path}")
    print(f"Currently shipped: level_weight={_CURRENT_LEVEL_WEIGHT}")

    print("\n" + "=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "Pick level_weight using CALIBRATION recall/lift, confirm on VALIDATION before adopting.\n"
        "Same discipline as min_dwell/directional-axis-split calibration: if calibration and\n"
        "validation disagree on direction, or the delta is within a turn or two's worth of noise\n"
        "(~11-19 turns per window here), that means don't fit to test -- keep what's shipped."
    )


if __name__ == "__main__":
    main()
