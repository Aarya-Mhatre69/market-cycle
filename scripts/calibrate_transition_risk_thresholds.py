"""
Calibrate the transition_risk magnitude thresholds — brutal-review audit finding #11.

The (medium=0.9, high=1.35) thresholds in cycle_signals.classify_cycle were already
flagged in that function's own comments as carried over unvalidated from an older
2-signal (trend_score/stc_score) divergence mechanism, and are doubly stale now that
the trend-score redesign (percentile-based, see compute_trend_context) reshaped the
directional axis's score distribution.

Unlike min_dwell (calibrate_min_dwell.py), transition_risk's bucketing doesn't affect
phase/hysteresis state, so it doesn't need a full walk-forward re-run per candidate —
the walk-forward is run ONCE to get the raw `transition_magnitude`/`transition_divergence`
values (added to CycleClassification for exactly this purpose), then every threshold
candidate is just a fast, cheap re-bucketing of that same history. Scored with the
same find_major_turns/score_lead_time_recall/score_precision_vs_base_rate functions
validate_cycle_classifier.py uses for the headline numbers, on the same calibration
(<=2020) / validation (2021-2023) split calibrate_min_dwell.py uses.

Usage:
    python scripts/calibrate_transition_risk_thresholds.py
"""

import argparse
import itertools
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

_CURRENT_MEDIUM, _CURRENT_HIGH = 0.9, 1.35


def _bucket(results: pd.DataFrame, medium: float, high: float) -> pd.DataFrame:
    """Re-derives transition_risk from the raw magnitude/divergence columns for one
    candidate threshold pair — no re-run of the walk-forward needed."""
    out = results.copy()
    div = out["transition_divergence"]
    mag = out["transition_magnitude"]
    out["transition_risk"] = "low"
    out.loc[div & (mag > medium), "transition_risk"] = "medium"
    out.loc[div & (mag > high), "transition_risk"] = "high"
    return out


def _window_score(results: pd.DataFrame, turns: pd.DataFrame, start: str, end: str, lead_days: int) -> dict:
    r = results[(results["date"] >= start) & (results["date"] <= end)]
    t = turns[(turns["date"] >= start) & (turns["date"] <= end)]
    if r.empty or t.empty:
        return {"n_points": len(r), "n_turns": len(t), "recall": float("nan"), "lift": None}
    recall_report = score_lead_time_recall(r, t, lead_days=lead_days)
    precision_report = score_precision_vs_base_rate(r, t, lead_days=lead_days)
    return {
        "n_points": len(r), "n_turns": len(t),
        "recall": recall_report["recall"], "hits": recall_report["hits"],
        "lift": precision_report["lift_over_random"],
        "base_rate": precision_report["base_rate_of_elevated_readings_overall"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--min-dwell", type=int, default=5)
    parser.add_argument("--lead-days", type=int, default=15)
    parser.add_argument("--medium-candidates", default="0.5,0.7,0.9,1.1")
    parser.add_argument("--high-candidates", default="1.0,1.35,1.6,1.9")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Nifty 50 history from %s (walk-forward run ONCE, reused for every threshold candidate)...", args.start)
    df = fetch_history(args.start)
    results = run_walk_forward(df, step=args.step, min_dwell=args.min_dwell)
    turns = find_major_turns(df, window=10, min_move_pct=8.0)
    logger.info("Produced %d evaluation points, %d independently-labeled turns.", len(results), len(turns))
    logger.info("transition_magnitude range: %.3f - %.3f (median %.3f)",
                results["transition_magnitude"].min(), results["transition_magnitude"].max(),
                results["transition_magnitude"].median())

    medium_candidates = [float(x) for x in args.medium_candidates.split(",")]
    high_candidates = [float(x) for x in args.high_candidates.split(",")]

    rows = []
    for medium, high in itertools.product(medium_candidates, high_candidates):
        if medium >= high:
            continue
        bucketed = _bucket(results, medium, high)
        calib = _window_score(bucketed, turns, args.start, CALIBRATION_END, args.lead_days)
        valid = _window_score(bucketed, turns, CALIBRATION_END, VALIDATION_END, args.lead_days)
        rows.append({
            "medium": medium, "high": high,
            "is_current": (medium == _CURRENT_MEDIUM and high == _CURRENT_HIGH),
            "calib_recall": calib["recall"], "calib_lift": calib["lift"], "calib_base_rate": calib.get("base_rate"),
            "valid_recall": valid["recall"], "valid_lift": valid["lift"], "valid_base_rate": valid.get("base_rate"),
        })

    # Always include the currently-shipped pair even if not in the candidate grid,
    # so "did anything actually beat what's already in production" has a direct answer.
    if not any(r["is_current"] for r in rows):
        bucketed = _bucket(results, _CURRENT_MEDIUM, _CURRENT_HIGH)
        calib = _window_score(bucketed, turns, args.start, CALIBRATION_END, args.lead_days)
        valid = _window_score(bucketed, turns, CALIBRATION_END, VALIDATION_END, args.lead_days)
        rows.append({
            "medium": _CURRENT_MEDIUM, "high": _CURRENT_HIGH, "is_current": True,
            "calib_recall": calib["recall"], "calib_lift": calib["lift"], "calib_base_rate": calib.get("base_rate"),
            "valid_recall": valid["recall"], "valid_lift": valid["lift"], "valid_base_rate": valid.get("base_rate"),
        })

    grid = pd.DataFrame(rows).sort_values(["medium", "high"]).reset_index(drop=True)
    grid_path = _OUTPUT_DIR / "transition_risk_threshold_grid.csv"
    grid.to_csv(grid_path, index=False)

    print("\n" + "=" * 100)
    print(f"TRANSITION_RISK THRESHOLD GRID  (calibration <= {CALIBRATION_END}, validation <= {VALIDATION_END})")
    print("=" * 100)
    print(grid.to_string(index=False))
    print(f"\nSaved to {grid_path}")
    print(f"\nCurrently shipped: medium={_CURRENT_MEDIUM}, high={_CURRENT_HIGH}")

    print("\n" + "=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "Pick a (medium, high) pair using CALIBRATION lift/recall only, then confirm the choice\n"
        "holds on VALIDATION before adopting it — same 'calibration and validation must agree'\n"
        "discipline as min_dwell (calibrate_min_dwell.py). If no candidate clearly beats the\n"
        "currently-shipped pair on BOTH windows, that itself is the answer: keep what's shipped\n"
        "rather than fit to noise (this system has ~11-19 independently-labeled turns per window,\n"
        "small enough that a 1-2 turn swing moves recall by 7-13 points on its own)."
    )


if __name__ == "__main__":
    main()
