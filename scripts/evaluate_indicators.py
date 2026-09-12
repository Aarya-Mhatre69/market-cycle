"""
Indicator Comparison & Ablation Evaluation — accuracy-roadmap Phase 3, Section 9.1.

Evaluates all 5 reference indicators (ZigZag + STC, already Core; Candlestick
Recognition, Harmonic Patterns, Gann Time Cycles, newly implemented in
cycle_signals.py) so the "should this indicator earn a place in the model" question
is answered with numbers, not assumed because it's on the reference list.

Reuses scripts/output/cycle_backtest_history.csv (produced by backtest_cycle_phase.py,
which now logs every indicator's per-evaluation-point score walk-forward, no
lookahead) and the same independently-labeled-turns methodology
validate_cycle_classifier.py uses, so results are directly comparable to the
Phase 0/2 headline numbers.

Five things measured, per Section 9.1 of the roadmap:
    1. Solo signal test — recall / precision-lift of each indicator's OWN non-neutral
       reading against independently-labeled major turns.
    2. Forward-return analysis — 5/10/20-day forward Nifty returns bucketed by each
       indicator's signal direction (bullish/neutral/bearish).
    3. Signal frequency & noise — how often each indicator fires, and how often it
       reverses direction within N bars (a noisiness proxy).
    4. Regime association — cross-tab of each indicator's reading vs. the classifier's
       phase output.
    5. Ablation — baseline (Core-only) vs each new indicator added individually vs all
       3 combined, re-scored on recall / precision-lift / phase-persistence using the
       SAME logged per-signal scores (no re-fetch, no re-derivation of the underlying
       walk-forward indicator values — only the phase-decision layer is re-run).

Usage:
    python scripts/evaluate_indicators.py
"""

import argparse
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

from backtest_cycle_phase import fetch_history  # noqa: E402
from validate_cycle_classifier import find_major_turns  # noqa: E402

from shankh.agents.market.cycle_signals import (  # noqa: E402
    EvidenceItem,
    HysteresisState,
    classify_cycle_stateful,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_BACKTEST_CSV = _OUTPUT_DIR / "cycle_backtest_history.csv"

_INDICATORS = {
    "zigzag": {"score_col": "zigzag_score", "tier": "core", "label": "ZigZag"},
    "stc": {"score_col": "stc_score", "tier": "core", "label": "STC"},
    "candlestick": {"score_col": "candlestick_score", "tier": "contextual", "label": "Candlestick Recognition"},
    "harmonic": {"score_col": "harmonic_score", "tier": "contextual", "label": "Harmonic Patterns"},
    "gann": {"score_col": "gann_score", "tier": "contextual", "label": "Gann Time Cycles"},
}
_DIRECTION_THRESHOLD = 0.05  # |score| below this counts as "neutral" / not fired


def _direction(score: float) -> str:
    if score > _DIRECTION_THRESHOLD:
        return "bullish"
    if score < -_DIRECTION_THRESHOLD:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# 1. Solo signal test
# ---------------------------------------------------------------------------

def _solo_recall_precision(history: pd.DataFrame, turns: pd.DataFrame, score_col: str, lead_days: int) -> dict:
    fired = history[score_col].abs() > _DIRECTION_THRESHOLD
    hits = 0
    for _, turn in turns.iterrows():
        window = history[(history["date"] >= turn["date"] - pd.Timedelta(days=lead_days)) & (history["date"] < turn["date"])]
        if fired.loc[window.index].any():
            hits += 1
    recall = round(hits / len(turns), 3) if len(turns) else 0.0

    turn_dates = turns["date"].to_numpy()

    def _near_turn(d) -> bool:
        deltas = np.abs((turn_dates - np.datetime64(d)) / np.timedelta64(1, "D"))
        return bool((deltas <= lead_days).any())

    fired_rows = history[fired]
    precision = round(fired_rows["date"].apply(_near_turn).mean(), 3) if len(fired_rows) else 0.0
    base_rate = round(fired.mean(), 3)
    overall_near = round(history["date"].apply(_near_turn).mean(), 3)
    lift = round(precision / overall_near, 2) if overall_near > 0 else None

    return {"recall": recall, "hits": hits, "n_turns": len(turns), "fired_base_rate": base_rate,
            "precision": precision, "lift": lift}


# ---------------------------------------------------------------------------
# 2. Forward-return analysis
# ---------------------------------------------------------------------------

def _attach_forward_returns(history: pd.DataFrame, daily: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    daily = daily.set_index("date").sort_index()
    idx_pos = daily.index.get_indexer(history["date"], method="nearest")
    closes = daily["close"].to_numpy()
    out = history.copy()
    for h in horizons:
        fwd = np.full(len(history), np.nan)
        valid = idx_pos + h < len(closes)
        fwd[valid] = (closes[idx_pos[valid] + h] / closes[idx_pos[valid]] - 1.0) * 100.0
        out[f"fwd_ret_{h}d"] = fwd
    return out


def _forward_return_table(history: pd.DataFrame, score_col: str, horizons=(5, 10, 20)) -> pd.DataFrame:
    direction = history[score_col].apply(_direction)
    rows = []
    for d in ["bullish", "neutral", "bearish"]:
        mask = direction == d
        row = {"direction": d, "n": int(mask.sum())}
        for h in horizons:
            vals = history.loc[mask, f"fwd_ret_{h}d"].dropna()
            row[f"median_{h}d"] = round(vals.median(), 2) if len(vals) else None
            row[f"mean_{h}d"] = round(vals.mean(), 2) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Signal frequency & noise
# ---------------------------------------------------------------------------

def _frequency_and_noise(history: pd.DataFrame, score_col: str, reversal_window: int = 3) -> dict:
    direction = history[score_col].apply(_direction)
    fire_rate = round((direction != "neutral").mean(), 3)

    # Noise proxy: of the bars where direction is non-neutral, what fraction flip
    # sign (bullish<->bearish) within `reversal_window` evaluation points?
    reversals, opportunities = 0, 0
    dir_arr = direction.to_numpy()
    for i in range(len(dir_arr) - reversal_window):
        if dir_arr[i] == "neutral":
            continue
        opportunities += 1
        future = dir_arr[i + 1: i + 1 + reversal_window]
        opposite = "bearish" if dir_arr[i] == "bullish" else "bullish"
        if opposite in future:
            reversals += 1
    reversal_rate = round(reversals / opportunities, 3) if opportunities else None

    return {"fire_rate": fire_rate, "reversal_rate_within_3pts": reversal_rate}


# ---------------------------------------------------------------------------
# 4. Regime association
# ---------------------------------------------------------------------------

def _regime_association(history: pd.DataFrame, score_col: str) -> pd.DataFrame:
    direction = history[score_col].apply(_direction)
    ct = pd.crosstab(direction, history["cycle_phase"], normalize="columns").round(3)
    return ct


# ---------------------------------------------------------------------------
# 5. Ablation
# ---------------------------------------------------------------------------

def _replay_with_signals(history: pd.DataFrame, include: set, min_dwell: int = 5) -> pd.DataFrame:
    """Re-runs ONLY the phase-decision layer (axis-vote + hysteresis) over the
    already-logged, walk-forward-computed per-signal scores in `history`, with a
    chosen subset of the 3 new indicators included as contextual-tier evidence. Does
    NOT re-fetch data or re-derive indicator scores — those are already correct
    (no-lookahead) from the canonical backtest run; only the decision layer changes.
    """
    state = HysteresisState()
    rows = []
    for _, r in history.iterrows():
        evidence = [
            EvidenceItem("zigzag", "core", "", r["zigzag_score"]),
            EvidenceItem("stc", "core", "", r["stc_score"]),
            EvidenceItem("trend_context", "core", "", r["trend_score"]),
        ]
        if "candlestick" in include and abs(r["candlestick_score"]) > 1e-9:
            evidence.append(EvidenceItem("candlestick", "contextual", "", r["candlestick_score"]))
        if "harmonic" in include and abs(r["harmonic_score"]) > 1e-9:
            evidence.append(EvidenceItem("harmonic", "contextual", "", r["harmonic_score"]))
        if "gann" in include and abs(r["gann_score"]) > 1e-9:
            evidence.append(EvidenceItem("gann", "contextual", "", r["gann_score"]))

        result, state = classify_cycle_stateful(evidence, state, min_dwell=min_dwell)
        rows.append({"date": r["date"], "cycle_phase": result.cycle_phase, "transition_risk": result.transition_risk})
    return pd.DataFrame(rows)


def _ablation_metrics(variant_history: pd.DataFrame, turns: pd.DataFrame, lead_days: int, step: int) -> dict:
    n_changes = (variant_history["cycle_phase"] != variant_history["cycle_phase"].shift(1)).sum() - 1
    mean_run_days = len(variant_history) / max(n_changes + 1, 1) * step

    hits = 0
    for _, turn in turns.iterrows():
        window = variant_history[(variant_history["date"] >= turn["date"] - pd.Timedelta(days=lead_days))
                                  & (variant_history["date"] < turn["date"])]
        if window["transition_risk"].isin(["medium", "high"]).any():
            hits += 1
    recall = round(hits / len(turns), 3) if len(turns) else 0.0

    turn_dates = turns["date"].to_numpy()

    def _near_turn(d) -> bool:
        deltas = np.abs((turn_dates - np.datetime64(d)) / np.timedelta64(1, "D"))
        return bool((deltas <= lead_days).any())

    elevated = variant_history[variant_history["transition_risk"].isin(["medium", "high"])]
    precision = round(elevated["date"].apply(_near_turn).mean(), 3) if len(elevated) else 0.0
    overall_near = round(variant_history["date"].apply(_near_turn).mean(), 3)
    lift = round(precision / overall_near, 2) if overall_near > 0 else None

    return {"phase_changes": int(n_changes), "mean_run_days": round(mean_run_days, 1), "recall": recall, "lift": lift}


# ---------------------------------------------------------------------------
# Section 9.2 visuals — shipped as pre-rendered PNGs (like the existing backtest
# chart), not live in-dashboard plotting: the underlying data is expensive to
# (re)compute (a data fetch + a full walk-forward replay per indicator), so the
# dashboard displays these images rather than recomputing on every page load.
# ---------------------------------------------------------------------------

_DIR_COLORS = {"bullish": "#2E7D32", "neutral": "#888888", "bearish": "#C62828"}
_PHASE_COLORS = {"EXPANSION": "#2E7D32", "DISTRIBUTION": "#F9A825", "CONTRACTION": "#C62828", "ACCUMULATION": "#1565C0"}


def plot_forward_return_boxplots(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(_INDICATORS), figsize=(4 * len(_INDICATORS), 4.5), sharey=True)
    for ax, (key, meta) in zip(axes, _INDICATORS.items()):
        direction = history[meta["score_col"]].apply(_direction)
        data, labels, colors = [], [], []
        for d in ["bearish", "neutral", "bullish"]:
            vals = history.loc[direction == d, "fwd_ret_20d"].dropna()
            if len(vals):
                data.append(vals)
                labels.append(f"{d}\n(n={len(vals)})")
                colors.append(_DIR_COLORS[d])
        if data:
            # Not passing labels=/tick_labels= to boxplot() directly — that kwarg's
            # name changed across matplotlib versions (labels -> tick_labels, then
            # labels was removed entirely in 3.10), so set ticks manually instead;
            # this works identically on any matplotlib version.
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(meta["label"], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("20-day forward Nifty return (%)")
    fig.suptitle("Forward-Return Distribution by Signal Direction (accuracy-roadmap Section 9.2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_heatmap(history: pd.DataFrame, out_path: Path) -> None:
    phases = ["ACCUMULATION", "EXPANSION", "DISTRIBUTION", "CONTRACTION"]
    rows, row_labels = [], []
    for key, meta in _INDICATORS.items():
        direction = history[meta["score_col"]].apply(_direction)
        ct = pd.crosstab(direction, history["cycle_phase"], normalize="columns")
        for d in ["bullish", "bearish"]:
            row = [ct.loc[d, p] if d in ct.index and p in ct.columns else 0.0 for p in phases]
            rows.append(row)
            row_labels.append(f"{meta['label']} ({d})")

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    im = ax.imshow(rows, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(phases)):
            ax.text(j, i, f"{rows[i][j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Agreement rate (column-normalized)")
    ax.set_title("Regime Association — indicator reading vs. current classifier phase")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_bars(ablation_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(ablation_df))
    ax1.bar(x, ablation_df["recall"], color="#1565C0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ablation_df["variant"], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Recall @ lead window")
    ax1.set_title("Ablation: recall by variant")
    ax1.axhline(ablation_df["recall"].iloc[0], color="grey", linestyle="--", linewidth=0.8, label="baseline")
    ax1.legend(fontsize=8)

    ax2.bar(x, ablation_df["lift"], color="#6A4C93")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ablation_df["variant"], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Precision-vs-base-rate lift")
    ax2.set_title("Ablation: lift by variant")
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle="--", label="random (1.0)")
    ax2.legend(fontsize=8)

    fig.suptitle("Marginal contribution of each new indicator (accuracy-roadmap Section 9.2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_timeline(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(_INDICATORS) + 1, 1, figsize=(14, 1.1 * (len(_INDICATORS) + 1) + 1), sharex=True,
                              gridspec_kw={"height_ratios": [1] * len(_INDICATORS) + [2]})

    for ax, (key, meta) in zip(axes[:-1], _INDICATORS.items()):
        direction = history[meta["score_col"]].apply(_direction)
        colors = direction.map(_DIR_COLORS)
        ax.scatter(history["date"], [0] * len(history), c=colors, s=6, marker="s")
        ax.set_yticks([])
        ax.set_ylabel(meta["label"], rotation=0, ha="right", va="center", fontsize=8)

    phase_ax = axes[-1]
    for phase, color in _PHASE_COLORS.items():
        mask = history["cycle_phase"] == phase
        phase_ax.scatter(history.loc[mask, "date"], history.loc[mask, "close"], s=6, color=color, label=phase)
    phase_ax.set_ylabel("Nifty close")
    phase_ax.legend(loc="upper left", fontsize=7, ncol=4)

    fig.suptitle("Signal Timeline — each indicator's direction over time, vs. confirmed phase (accuracy-roadmap Section 9.2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ranked summary — sorting by raw lift alone would put Harmonic Patterns at #1
# despite only 9 total firings, which is a hypothesis, not evidence. Indicators
# below _MIN_RELIABLE_FIRES are listed but deliberately left unranked rather than
# sorted on a number too noisy to trust — same "don't fit to a too-small sample"
# discipline used for the min_dwell/tier-weight calibration elsewhere.
# ---------------------------------------------------------------------------

_MIN_RELIABLE_FIRES = 30

_VERDICTS = {
    "ZigZag": "CORE (existing) — feeds the directional axis. Solo lift isn't the right test for it (its job is classification, not standalone early warning).",
    "STC": "CORE (existing) — feeds the momentum axis. Fires on ~99.5% of points at this threshold (its score rarely sits exactly at 0), which likely inflates its solo recall without real skill — read its row with that caveat.",
    "Candlestick Recognition": "DROP from the phase vote. Noisy (40% reversal within 3 points), forward returns run backwards from expectation (bearish bucket beats bullish), zero ablation impact.",
    "Harmonic Patterns": "WATCH, don't weight into production yet. Most promising number (lift 2.44) but only 9 total firings — too small a sample to trust. Revisit once more history accumulates.",
    "Gann Time Cycles": "DROP. Statistically random (lift 0.99), zero ablation impact — matches the roadmap's own expectation that this is the least empirically-grounded of the five.",
}


def build_ranking(solo_df: pd.DataFrame, freq_df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    merged = solo_df.merge(freq_df, on="indicator")
    merged["fired_count"] = (merged["fired_base_rate"] * n_points).round().astype(int)
    merged["reliable_sample"] = merged["fired_count"] >= _MIN_RELIABLE_FIRES
    merged["verdict"] = merged["indicator"].map(_VERDICTS)

    reliable = merged[merged["reliable_sample"]].sort_values("lift", ascending=False).reset_index(drop=True)
    reliable["rank"] = reliable.index + 1
    unreliable = merged[~merged["reliable_sample"]].sort_values("lift", ascending=False).reset_index(drop=True)
    unreliable["rank"] = None

    ranking = pd.concat([reliable, unreliable], ignore_index=True)
    return ranking[["rank", "indicator", "lift", "recall", "fired_count", "reliable_sample", "verdict"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--lead-days", type=int, default=15)
    parser.add_argument("--label-window", type=int, default=10)
    parser.add_argument("--min-move-pct", type=float, default=8.0)
    parser.add_argument("--min-dwell", type=int, default=5)
    parser.add_argument("--backtest-step", type=int, default=3, help="Must match the --step used to produce cycle_backtest_history.csv")
    args = parser.parse_args()

    if not _BACKTEST_CSV.exists():
        raise SystemExit(f"{_BACKTEST_CSV} not found — run backtest_cycle_phase.py first.")

    history = pd.read_csv(_BACKTEST_CSV, parse_dates=["date"])
    logger.info("Loaded %d evaluation points from %s.", len(history), _BACKTEST_CSV)

    logger.info("Fetching daily Nifty history for forward-return computation and turn labeling...")
    daily = fetch_history(args.start)
    turns = find_major_turns(daily, window=args.label_window, min_move_pct=args.min_move_pct)
    logger.info("Found %d independently-labeled major turns.", len(turns))

    history = _attach_forward_returns(history, daily)

    print("\n" + "=" * 100)
    print("1. SOLO SIGNAL TEST — recall & precision-lift of each indicator's OWN non-neutral reading")
    print("=" * 100)
    solo_rows = []
    for key, meta in _INDICATORS.items():
        m = _solo_recall_precision(history, turns, meta["score_col"], args.lead_days)
        solo_rows.append({"indicator": meta["label"], **m})
    solo_df = pd.DataFrame(solo_rows)
    print(solo_df.to_string(index=False))
    solo_df.to_csv(_OUTPUT_DIR / "indicator_solo_signal_test.csv", index=False)

    print("\n" + "=" * 100)
    print("2. FORWARD-RETURN ANALYSIS — 5/10/20-day forward Nifty return by signal direction")
    print("=" * 100)
    fwd_frames = []
    for key, meta in _INDICATORS.items():
        t = _forward_return_table(history, meta["score_col"])
        t.insert(0, "indicator", meta["label"])
        fwd_frames.append(t)
        print(f"\n-- {meta['label']} --")
        print(t.to_string(index=False))
    pd.concat(fwd_frames, ignore_index=True).to_csv(_OUTPUT_DIR / "indicator_forward_returns.csv", index=False)

    print("\n" + "=" * 100)
    print("3. SIGNAL FREQUENCY & NOISE")
    print("=" * 100)
    freq_rows = []
    for key, meta in _INDICATORS.items():
        m = _frequency_and_noise(history, meta["score_col"])
        freq_rows.append({"indicator": meta["label"], **m})
    freq_df = pd.DataFrame(freq_rows)
    print(freq_df.to_string(index=False))
    freq_df.to_csv(_OUTPUT_DIR / "indicator_frequency_noise.csv", index=False)

    print("\n" + "=" * 100)
    print("4. REGIME ASSOCIATION — indicator reading vs. current classifier phase (column-normalized)")
    print("=" * 100)
    regime_frames = []
    for key, meta in _INDICATORS.items():
        ct = _regime_association(history, meta["score_col"])
        print(f"\n-- {meta['label']} --")
        print(ct.to_string())
        ct = ct.reset_index().rename(columns={"index": "direction"})
        ct.insert(0, "indicator", meta["label"])
        regime_frames.append(ct)
    pd.concat(regime_frames, ignore_index=True, sort=False).to_csv(_OUTPUT_DIR / "indicator_regime_association.csv", index=False)

    print("\n" + "=" * 100)
    print("5. ABLATION — baseline (Core-only) vs each new indicator vs all 3 combined")
    print("=" * 100)
    variants = {
        "baseline (Core-only: zigzag+stc+trend)": set(),
        "+candlestick": {"candlestick"},
        "+harmonic": {"harmonic"},
        "+gann": {"gann"},
        "all 3 combined": {"candlestick", "harmonic", "gann"},
    }
    ablation_rows = []
    for name, include in variants.items():
        logger.info("Replaying variant: %s", name)
        variant_history = _replay_with_signals(history, include, min_dwell=args.min_dwell)
        m = _ablation_metrics(variant_history, turns, args.lead_days, args.backtest_step)
        ablation_rows.append({"variant": name, **m})
    ablation_df = pd.DataFrame(ablation_rows)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(_OUTPUT_DIR / "indicator_ablation.csv", index=False)

    print("\n" + "=" * 100)
    print("RANKED SUMMARY — ranked by lift among reliably-sampled indicators only (>= %d firings)" % _MIN_RELIABLE_FIRES)
    print("=" * 100)
    ranking_df = build_ranking(solo_df, freq_df, len(history))
    print(ranking_df.to_string(index=False))
    ranking_df.to_csv(_OUTPUT_DIR / "indicator_ranking.csv", index=False)

    logger.info("Rendering Section 9.2 visuals...")
    plot_forward_return_boxplots(history, _OUTPUT_DIR / "indicator_forward_return_boxplots.png")
    plot_regime_heatmap(history, _OUTPUT_DIR / "indicator_regime_heatmap.png")
    plot_ablation_bars(ablation_df, _OUTPUT_DIR / "indicator_ablation_chart.png")
    plot_signal_timeline(history, _OUTPUT_DIR / "indicator_signal_timeline.png")
    logger.info("Visuals saved to %s", _OUTPUT_DIR)

    print("\n" + "=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print(
        "Solo/ablation lift > 1.0 means the indicator's non-neutral readings cluster near\n"
        "real turns more than chance; ~1.0 means no better than random (the same standard\n"
        "already applied to transition_risk in validate_cycle_classifier.py). Forward-return\n"
        "medians should differ meaningfully between bullish/bearish buckets, in the expected\n"
        "direction, for a signal to be worth its tier weight. High fire_rate + high\n"
        "reversal_rate together mean an indicator is mostly noise. This is Core-tier price\n"
        "data only (no valuation/liquidity/earnings), same limitation as the rest of this\n"
        "roadmap's backtesting."
    )


if __name__ == "__main__":
    main()
