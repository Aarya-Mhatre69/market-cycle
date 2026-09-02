"""
Event Case Study — Nifty 50 Cycle Classification Around the Hindenburg-Adani Report (Jan 2023).

Runs the Core classifier day-by-day (no lookahead) over a window bracketing the
24-Jan-2023 Hindenburg short-seller report on Adani, and plots:
  1. Nifty 50 close price with a rolling volatility band (NOT a forecast — the Cycle
     Agent does not do price forecasting; that's the separate per-stock price-band
     LightGBM pipeline in src/shankh/agents/equity/. This band is purely descriptive:
     price ± 1 rolling 10-day std, for visual context only).
  2. A phase-colored regime timeline (same 4-color scheme as the other backtest charts).
  3. cycle_confidence over time, so a confidence dip/spike around the event is visible
     if one exists — this is a genuine model output, not fabricated for the chart.

Honesty note: Hindenburg's report targeted the Adani Group specifically. Nifty 50 is a
50-stock index — Adani Enterprises + Adani Ports are a small fraction of index weight,
so do not expect the index-level chart to show anything like the ~50% single-stock
Adani crash. That contrast (large single-stock shock, muted index-level reaction) is
itself the honest, tellable story here.

Usage:
    python scripts/event_case_study_hindenburg.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_cycle_phase import fetch_history, run_walk_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

EVENT_DATE = pd.Timestamp("2023-01-24")
EVENT_LABEL = "Hindenburg report on Adani Group\n(24 Jan 2023)"

WINDOW_START = pd.Timestamp("2022-09-01")
WINDOW_END = pd.Timestamp("2023-04-30")
FETCH_START = "2021-09-01"  # extra buffer so the 200DMA/warmup is real, not truncated

PHASE_COLORS = {
    "EXPANSION": "#2E7D32",
    "DISTRIBUTION": "#C77700",
    "CONTRACTION": "#C62828",
    "ACCUMULATION": "#1565C0",
}


def build_regime_bands(results: pd.DataFrame):
    """Collapse a per-day phase series into contiguous (start, end, phase) spans for shading."""
    spans = []
    current_phase = None
    span_start = None
    for _, row in results.iterrows():
        if row["cycle_phase"] != current_phase:
            if current_phase is not None:
                spans.append((span_start, row["date"], current_phase))
            current_phase = row["cycle_phase"]
            span_start = row["date"]
    if current_phase is not None:
        spans.append((span_start, results["date"].iloc[-1], current_phase))
    return spans


def main():
    logger.info("Fetching Nifty 50 history from %s...", FETCH_START)
    df = fetch_history(FETCH_START)

    logger.info("Running daily walk-forward classification (no lookahead)...")
    results = run_walk_forward(df, step=1)

    window = results[(results["date"] >= WINDOW_START) & (results["date"] <= WINDOW_END)].reset_index(drop=True)
    if window.empty:
        raise RuntimeError("No evaluation points in the requested window.")

    window["roll_std10"] = window["close"].rolling(10, min_periods=3).std()
    window["band_upper"] = window["close"] + window["roll_std10"]
    window["band_lower"] = window["close"] - window["roll_std10"]

    spans = build_regime_bands(window)

    fig, (ax_price, ax_conf) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for start, end, phase in spans:
        ax_price.axvspan(start, end, color=PHASE_COLORS.get(phase, "#CCC"), alpha=0.15, lw=0)
        ax_conf.axvspan(start, end, color=PHASE_COLORS.get(phase, "#CCC"), alpha=0.15, lw=0)

    ax_price.fill_between(window["date"], window["band_lower"], window["band_upper"],
                           color="#4C72B0", alpha=0.25, label="±1 rolling 10-day std (descriptive, not a forecast)")
    ax_price.plot(window["date"], window["close"], color="black", linewidth=1.3, label="Nifty 50 close")

    ax_price.axvline(EVENT_DATE, color="#C62828", linestyle="--", linewidth=1.3)
    event_price = window.loc[window["date"] >= EVENT_DATE, "close"].iloc[0]
    y_min, y_max = window["band_lower"].min(), window["band_upper"].max()
    y_span = y_max - y_min
    ax_price.set_ylim(y_min - 0.05 * y_span, y_max + 0.22 * y_span)
    ax_price.annotate(
        EVENT_LABEL,
        xy=(EVENT_DATE, event_price),
        xytext=(EVENT_DATE - pd.Timedelta(days=25), y_max + 0.15 * y_span),
        fontsize=9, color="#C62828", ha="center",
        arrowprops=dict(arrowstyle="->", color="#C62828"),
    )

    ax_price.set_ylabel("Nifty 50 Close (INR)")
    ax_price.set_title(
        "Nifty 50 Cycle Classification Around the Hindenburg-Adani Report\n"
        "Phase-colored background · descriptive volatility band · daily walk-forward, no lookahead",
        fontsize=12, fontweight="bold", pad=14,
    )
    handles = [plt.Line2D([0], [0], color="black", lw=1.3, label="Nifty 50 close")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3, label=p) for p, c in PHASE_COLORS.items()]
    ax_price.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)

    ax_conf.plot(window["date"], window["cycle_confidence"], color="#333333", linewidth=1.2)
    ax_conf.axvline(EVENT_DATE, color="#C62828", linestyle="--", linewidth=1.3)
    ax_conf.set_ylabel("cycle_confidence")
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_xlabel("Date")

    ax_conf.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = _OUTPUT_DIR / "event_case_study_hindenburg.png"
    fig.savefig(out_path, dpi=150)
    logger.info("Chart saved to %s", out_path)

    event_row = window.iloc[(window["date"] - EVENT_DATE).abs().argsort()[:1]]
    print("\n" + "=" * 78)
    print("MODEL READ ON/NEAR THE EVENT DATE")
    print("=" * 78)
    print(event_row[["date", "close", "cycle_phase", "cycle_confidence", "transition_risk", "composite_score"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("HONEST CONTEXT")
    print("=" * 78)
    print(
        "Adani Enterprises + Adani Ports together are a small fraction of Nifty 50's\n"
        "float-weighted index. Adani group stocks lost roughly half their value in the\n"
        "weeks after this report; the index itself did NOT see anything close to that —\n"
        "this chart is expected to show a muted, not dramatic, reaction. That contrast\n"
        "(single-stock/sector shock vs. diversified-index resilience) is the actual\n"
        "story this chart tells, not a missed crash-call."
    )


if __name__ == "__main__":
    main()
