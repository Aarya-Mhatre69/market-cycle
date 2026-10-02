"""
Volume experiment: did adding volume signals improve the phase classifier?

The mentor suggested volume. This runs the same ablation used for every other
indicator (scripts/evaluate_indicators.py), but on five price series instead of one
(Nifty 50 + four stocks), so the conclusion does not rest on a single chart.

Variants per series: core signals only, +OBV, +CMF, +volume-confirmed pivot, all three.
Volume signals are re-injected as evidence into the current classifier (recalibrated
transition_risk threshold included) using the already-logged walk-forward scores.

Outputs (scripts/output/):
    volume_experiment_summary.csv     one row per series x variant
    volume_experiment_verdict.txt     one data-derived sentence (shown in the dashboard)
    volume_experiment_onepager.png    slide-ready summary

Usage:
    python scripts/volume_experiment_report.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_indicators import _ablation_metrics, _replay_with_signals  # noqa: E402
from validate_cycle_classifier import find_major_turns  # noqa: E402

from shankh.agents.market.cycle_history import fetch_history, run_walk_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SERIES = [
    ("Nifty 50", "^NSEI"),
    ("MARKSANS", "MARKSANS.NS"),
    ("RELIANCE", "RELIANCE.NS"),
    ("TCS", "TCS.NS"),
    ("ICICIBANK", "ICICIBANK.NS"),
]
START = "2016-01-01"
STEP = 3
LEAD_DAYS = 15
MIN_DWELL = 5
BASELINE = "Without volume"
VARIANTS = {
    BASELINE: set(),
    "+ OBV": {"obv"},
    "+ CMF": {"cmf"},
    "+ Volume-confirmed pivot": {"volume_confirmation"},
    "With all 3 volume signals": {"obv", "cmf", "volume_confirmation"},
}
ALL3 = "With all 3 volume signals"
_EPS = 1e-9


def run_series(name: str, ticker: str) -> list:
    logger.info("Running %s (%s)...", name, ticker)
    df = fetch_history(START, ticker=ticker)
    history = run_walk_forward(df, step=STEP, min_dwell=MIN_DWELL)
    turns = find_major_turns(df, window=10, min_move_pct=8.0)

    replays = {v: _replay_with_signals(history, inc, min_dwell=MIN_DWELL) for v, inc in VARIANTS.items()}
    base = replays[BASELINE]
    rows = []
    for variant, replay in replays.items():
        m = _ablation_metrics(replay, turns, LEAD_DAYS, STEP)
        rows.append({
            "series": name, "variant": variant, "n_points": len(replay), "n_turns": len(turns),
            "recall": m["recall"], "lift": m["lift"],
            "phase_changes": m["phase_changes"], "mean_run_days": m["mean_run_days"],
            "phase_differs_pct": round((replay["cycle_phase"] != base["cycle_phase"]).mean() * 100, 1),
            "risk_differs_pct": round((replay["transition_risk"] != base["transition_risk"]).mean() * 100, 1),
        })
    return rows


def build_verdict(summary: pd.DataFrame) -> str:
    base = summary[summary["variant"] == BASELINE].set_index("series")
    full = summary[summary["variant"] == ALL3].set_index("series")
    d_recall = (full["recall"] - base["recall"]).astype(float)
    better = int((d_recall > _EPS).sum())
    worse = int((d_recall < -_EPS).sum())
    same = len(d_recall) - better - worse
    avg = float(d_recall.mean())
    phase_diff = float(full["phase_differs_pct"].mean())
    detail = (f"Recall changed by {avg:+.3f} on average with all three volume signals "
              f"(better on {better}, worse on {worse}, unchanged on {same} of {len(d_recall)} series); "
              f"the reported phase differed on {phase_diff:.1f}% of days.")
    if avg >= 0.05 and better >= len(d_recall) - 1:
        head = "Volume improved warning accuracy."
    elif avg <= -0.02 or worse > better:
        head = "Volume did not improve accuracy."
    else:
        head = "No reliable improvement from volume."
    return f"{head} {detail}"


def draw_onepager(summary: pd.DataFrame, verdict: str, out_path: Path) -> None:
    base = summary[summary["variant"] == BASELINE].set_index("series")
    full = summary[summary["variant"] == ALL3].set_index("series")
    names = list(base.index)
    x = np.arange(len(names))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.suptitle("Did adding volume improve the phase classifier?", fontsize=19, fontweight="bold", y=0.99)

    for ax, col, title in [(axes[0], "recall", "Turns warned in advance (recall, higher is better)"),
                           (axes[1], "lift", "Warning quality vs chance (lift, above 1.0 is better)")]:
        ax.bar(x - w / 2, base[col].astype(float), w, label="Without volume", color="#78909C")
        ax.bar(x + w / 2, full[col].astype(float), w, label="With volume", color="#EF6C00")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if col == "lift":
            ax.axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].legend(frameon=False)

    ax = axes[2]
    ax.bar(x - w / 2, full["phase_differs_pct"].astype(float), w, label="Phase changed", color="#5C6BC0")
    ax.bar(x + w / 2, full["risk_differs_pct"].astype(float), w, label="Risk level changed", color="#26A69A")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, fontsize=10)
    ax.set_title("% of days volume changed the output", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    fig.text(0.5, 0.075, verdict, ha="center", fontsize=12.5, fontweight="bold", wrap=True)
    fig.text(0.5, 0.025,
             "Decision: volume signals are not added to the phase vote. They stay computed for reference only.   "
             "Method: same ablation as every other indicator, 5 series, 2016 onward.",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, ticker in SERIES:
        try:
            rows.extend(run_series(name, ticker))
        except Exception as exc:  # one failing series must not discard the others
            logger.warning("Skipping %s: %s", name, exc)
    summary = pd.DataFrame(rows)
    summary.to_csv(_OUTPUT_DIR / "volume_experiment_summary.csv", index=False)

    verdict = build_verdict(summary)
    (_OUTPUT_DIR / "volume_experiment_verdict.txt").write_text(verdict, encoding="utf-8")
    draw_onepager(summary, verdict, _OUTPUT_DIR / "volume_experiment_onepager.png")

    print(summary.to_string(index=False))
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
