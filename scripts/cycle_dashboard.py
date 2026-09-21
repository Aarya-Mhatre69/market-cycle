"""
Market Cycle Agent — Live Demo Dashboard.

A Streamlit app for demoing the Cycle Agent without needing the LLM layer: it calls
the deterministic tools directly (same as demo_cycle_agent.py) and lays out the live
classification, the evidence trail, the walk-forward backtest chart, and the honest
validation numbers (including the negative finding) in one screen.

Run:
    streamlit run scripts/cycle_dashboard.py

No API keys required to load — each viewer can enter their own FMP_API_KEY /
FRED_API_KEY in the sidebar to unlock live valuation and liquidity numbers; without
one (yours or a server default in .env), those sections show their
FALLBACK_BENCHMARK / insufficient_data labels honestly, same as the rest of this
system. Sidebar keys are held only in that viewer's own session — see
shankh.agents.market.cycle_tools.set_session_api_keys for why this is safe to share
with multiple concurrent viewers (e.g. a team deployment) without one person's key
leaking into another's request.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shankh.agents.market.cycle_tools import get_market_cycle_synthesis, set_session_api_keys  # noqa: E402
from shankh.agents.market.universe_cycle import _LATEST_CSV as _UNIVERSE_CSV  # noqa: E402

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PHASE_COLORS = {
    "EXPANSION": "#2E7D32",
    "DISTRIBUTION": "#C77700",
    "CONTRACTION": "#C62828",
    "ACCUMULATION": "#1565C0",
}
RISK_COLORS = {"low": "#2E7D32", "medium": "#C77700", "high": "#C62828"}

st.set_page_config(page_title="Market Cycle Agent", page_icon="🔄", layout="wide")


def render_api_key_sidebar():
    """
    Lets each viewer supply their own FMP/FRED keys instead of requiring them in the
    server's .env — important once this dashboard is deployed somewhere shared
    (a team, not just its author). Keys live only in this browser session's
    st.session_state and are applied via set_session_api_keys, a thread-local
    override (not an env var), so one viewer's key can never leak into another
    concurrent viewer's request on a shared deployment.
    """
    st.sidebar.header("API Keys")
    fmp_key = st.sidebar.text_input("FMP API Key", type="password", key="fmp_key_input",
                                     placeholder="fmp_xxxxxxxxxxxx")
    fred_key = st.sidebar.text_input("FRED API Key", type="password", key="fred_key_input",
                                      placeholder="fred_xxxxxxxxxxxx")

    def _status(user_value: str, env_var: str) -> str:
        if user_value:
            return "✅ using your key"
        if os.getenv(env_var):
            return "🔧 using server default"
        return "⚠️ not set"

    st.sidebar.caption(f"FMP: {_status(fmp_key, 'FMP_API_KEY')}")
    st.sidebar.caption(f"FRED: {_status(fred_key, 'FRED_API_KEY')}")
    st.sidebar.markdown(
        "[Get a free FMP key](https://site.financialmodelingprep.com/developer/docs) · "
        "[Get a free FRED key](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    return fmp_key, fred_key


@st.cache_data(ttl=300, show_spinner="Running synthesis (ZigZag / STC / trend / ERP / liquidity / earnings)...")
def fetch_synthesis(fmp_key: str, fred_key: str) -> dict:
    # fmp_key/fred_key are part of the cache signature on purpose: two viewers with
    # different keys (or one viewer switching keys) must never be served each
    # other's cached result.
    set_session_api_keys(fmp_api_key=fmp_key, fred_api_key=fred_key)
    raw = get_market_cycle_synthesis.invoke({})
    return json.loads(raw)


def render_header(fmp_key: str, fred_key: str):
    st.title("Market Cycle Agent — Live Dashboard")
    st.caption(
        "Deterministic tool-layer output, called directly — no LLM in this view. "
        "This is the same JSON the LLM narration layer would receive and report verbatim."
    )
    col1, col2 = st.columns(2)
    with col1:
        fmp = "configured" if (fmp_key or os.getenv("FMP_API_KEY")) else "not set → FALLBACK_BENCHMARK"
        fred = "configured" if (fred_key or os.getenv("FRED_API_KEY")) else "not set → insufficient_data"
        st.caption(f"FMP_API_KEY: **{fmp}**  |  FRED_API_KEY: **{fred}**  |  yfinance needs no key")
    with col2:
        if st.button("🔄 Refresh synthesis"):
            fetch_synthesis.clear()
            st.rerun()


def render_classification(data: dict):
    phase = data.get("cycle_phase", "N/A")
    confidence = data.get("cycle_confidence", 0.0)
    risk = data.get("transition_risk", "n/a")
    composite = data.get("composite_score", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cycle Phase", phase)
    c2.metric("Confidence", f"{confidence:.0%}")
    c3.metric("Transition Risk", risk.upper())
    c4.metric("Composite Score", f"{composite:+.3f}")

    phase_color = PHASE_COLORS.get(phase, "#888")
    risk_color = RISK_COLORS.get(risk, "#888")
    st.markdown(
        f"""<div style="display:flex; gap:10px; margin-top:-8px; margin-bottom:16px;">
        <span style="background:{phase_color}20; color:{phase_color}; border:1px solid {phase_color};
        border-radius:6px; padding:2px 10px; font-size:13px; font-weight:600;">{phase}</span>
        <span style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color};
        border-radius:6px; padding:2px 10px; font-size:13px; font-weight:600;">RISK: {risk.upper()}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    watch = data.get("transition_watch", "none")
    if watch and watch != "none":
        st.info(f"Transition watch: **{watch}**")

    pending_phase = data.get("pending_phase")
    dwell_progress = data.get("dwell_progress")
    if pending_phase:
        st.warning(
            f"**Phase change building:** {phase} -> {pending_phase} "
            f"({dwell_progress} consecutive confirmations needed). Reported phase stays "
            f"**{phase}** until confirmed — this is the Phase 2 hysteresis filter, not "
            "a 5th state; see the Methodology tab."
        )

    st.caption(f"As of: {data.get('as_of_date', 'n/a')}  |  Market: {data.get('market', 'NIFTY50')}")


def render_evidence(data: dict):
    st.subheader("Evidence Trail")
    evidence = data.get("evidence", [])
    if not evidence:
        st.warning("No evidence available.")
        return
    df = pd.DataFrame(evidence)
    df = df[["tier", "signal", "score", "reading", "note"]]
    df["score"] = df["score"].map(lambda s: f"{s:+.3f}")

    def _tier_badge(tier):
        colors = {"core": "#1565C0", "supporting": "#6A4C93", "contextual": "#888"}
        return f'<span style="color:{colors.get(tier, "#888")}; font-weight:600;">{tier.upper()}</span>'

    df_display = df.copy()
    df_display["tier"] = df_display["tier"].map(_tier_badge)
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.caption(f"Data freshness: {data.get('data_freshness', {})}")


def render_backtest():
    st.subheader("Walk-Forward Backtest (Core signals, 2018-2026 Nifty 50)")
    chart_path = _OUTPUT_DIR / "cycle_backtest_chart.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
        st.caption(
            "Generated by scripts/backtest_cycle_phase.py — Core-tier only (no lookahead). "
            "Colors match the phase badges above."
        )
    else:
        st.warning("No backtest chart found. Run: `python scripts/backtest_cycle_phase.py`")

    history_csv = _OUTPUT_DIR / "cycle_backtest_history.csv"
    if history_csv.exists():
        hist = pd.read_csv(history_csv)
        with st.expander(f"Raw backtest history ({len(hist)} evaluation points)"):
            st.dataframe(hist.tail(50), use_container_width=True)


def render_validation():
    st.subheader("Validation — Independent Precision/Recall Check")
    report_path = _OUTPUT_DIR / "validation_report.txt"
    if not report_path.exists():
        st.warning("No validation report found. Run: `python scripts/validate_cycle_classifier.py`")
        return

    report_text = report_path.read_text(encoding="utf-8")
    lines = [l for l in report_text.splitlines() if l.strip()]
    metrics = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            metrics[k.strip()] = v.strip()

    recall_line = next((l for l in lines if l.startswith("Recall")), None)
    lift = metrics.get("lift_over_random")

    if recall_line:
        st.metric("Recall @ lead window", recall_line.split(":", 1)[1].strip().split("(")[0].strip())
    if lift is not None:
        try:
            lift_val = float(lift)
        except ValueError:
            lift_val = None
        if lift_val is not None:
            if lift_val < 1.1:
                st.error(
                    f"**Lift over random base rate: {lift_val}x** — the transition_risk signal is "
                    "NOT showing meaningful advance-warning skill over its own unconditional firing "
                    "rate at this lead window. This is an honest negative finding, not hidden: the "
                    "Core-tier (2-signal) transition_risk needs more evidence — breadth, valuation "
                    "history — before it can be trusted as a leading indicator. Concurrent phase "
                    "classification (the backtest chart above) tracks price structure more convincingly "
                    "than transition_risk anticipates turns."
                )
            else:
                st.success(f"**Lift over random base rate: {lift_val}x** — meaningfully above chance.")

    with st.expander("Full validation report + methodology"):
        st.code(report_text, language="text")
        st.caption(
            "Independent turn labels use a separate local-extrema + minimum-retracement "
            "algorithm — deliberately NOT the classifier's own ZigZag — so the ground truth "
            "isn't produced by the thing being scored. See scripts/validate_cycle_classifier.py."
        )

    turns_csv = _OUTPUT_DIR / "independent_labeled_turns.csv"
    if turns_csv.exists():
        with st.expander("Independently-labeled major turns"):
            st.dataframe(pd.read_csv(turns_csv), use_container_width=True)


def render_indicators():
    st.subheader("Five-Indicator Comparison (accuracy-roadmap Phase 3, Section 9)")
    st.caption(
        "STC and ZigZag are already implemented (Core tier). Candlestick Recognition, Harmonic "
        "Patterns, and Gann Time Cycles are newly implemented here and evaluated — not assumed "
        "to help because they're on the reference list. Generated by scripts/evaluate_indicators.py."
    )

    solo_path = _OUTPUT_DIR / "indicator_solo_signal_test.csv"
    freq_path = _OUTPUT_DIR / "indicator_frequency_noise.csv"
    ranking_path = _OUTPUT_DIR / "indicator_ranking.csv"
    if not solo_path.exists():
        st.warning("No indicator evaluation found. Run: `python scripts/evaluate_indicators.py`")
        return

    st.markdown("#### Ranked summary")
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.caption(
            "Ranked by precision-vs-base-rate lift, but ONLY among indicators with >= 30 "
            "non-neutral readings (`reliable_sample`) — an indicator with fewer firings (like "
            "Harmonic Patterns' 9) can post an eye-catching lift number that's a hypothesis, not "
            "evidence, so it's shown unranked rather than sorted to the top on a too-small sample."
        )
    else:
        st.info("Ranking not found — re-run `python scripts/evaluate_indicators.py` to generate it.")

    st.markdown("#### Indicator performance table")
    solo = pd.read_csv(solo_path)
    freq = pd.read_csv(freq_path) if freq_path.exists() else None
    table = solo.merge(freq, on="indicator") if freq is not None else solo
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "lift > 1.0 = the indicator's non-neutral reading clusters near real turns more than "
        "chance; ~1.0 = no better than random. fire_rate = how often it's non-neutral; "
        "reversal_rate_within_3pts = how often it flips direction shortly after firing (noise proxy)."
    )

    st.markdown("#### Forward-return distributions")
    fwd_path = _OUTPUT_DIR / "indicator_forward_return_boxplots.png"
    if fwd_path.exists():
        st.image(str(fwd_path), use_container_width=True)

    st.markdown("#### Regime association")
    heatmap_path = _OUTPUT_DIR / "indicator_regime_heatmap.png"
    if heatmap_path.exists():
        st.image(str(heatmap_path), use_container_width=True)

    st.markdown("#### Ablation — does adding these 3 indicators actually help?")
    ablation_path = _OUTPUT_DIR / "indicator_ablation_chart.png"
    if ablation_path.exists():
        st.image(str(ablation_path), use_container_width=True)
    ablation_csv = _OUTPUT_DIR / "indicator_ablation.csv"
    if ablation_csv.exists():
        st.dataframe(pd.read_csv(ablation_csv), use_container_width=True, hide_index=True)

    st.markdown("#### Signal timeline")
    timeline_path = _OUTPUT_DIR / "indicator_signal_timeline.png"
    if timeline_path.exists():
        st.image(str(timeline_path), use_container_width=True)

    st.markdown("#### Verdict")
    st.error(
        "**Candlestick Recognition: DROP from the phase vote.** Solo lift 1.05 (no real skill), "
        "40% reversal rate within 3 points (noisy), and forward returns run BACKWARDS from "
        "expectation — its bearish-signal bucket shows higher 20-day forward returns than its "
        "bullish-signal bucket. Ablation confirms zero impact on any phase decision at contextual "
        "weight. Keep it logged/visible (it's already computed) but not weighted into the score."
    )
    st.warning(
        "**Harmonic Patterns: KEEP implemented, DO NOT weight into production yet.** The most "
        "intriguing result — solo lift 2.44, and both its bullish (n=5) and bearish (n=4) forward-"
        "return buckets move cleanly in the expected direction. But it fires on only 1.4% of "
        "evaluation points (9 total); a lift number built on single-digit counts is not "
        "trustworthy evidence, it's a hypothesis. Recommend accumulating more history before "
        "revisiting whether it earns real weight — exactly what the roadmap asked for "
        "('lower expected signal frequency... evaluate hit-rate carefully before granting weight')."
    )
    st.error(
        "**Gann Time Cycles: DROP.** Solo lift 0.99 (statistically random), no consistent forward-"
        "return direction, zero ablation impact. This matches the roadmap's own explicit "
        "expectation that Gann is 'the most contested/least empirically-grounded' of the five — "
        "a negative finding here is a legitimate, expected result, not a bug to re-tune away."
    )
    st.caption(
        "STC and ZigZag's own solo lift (1.01 / 1.00) is also ~random — but their job was never "
        "solo turn-prediction, it's feeding the phase-classification axes (Phase 2), which is a "
        "different question the ablation/backtest already answers separately. One measurement "
        "caveat: STC fires on 99.5% of points at the |score|>0.05 threshold used here (its score "
        "formula rarely lands exactly at 0), which likely inflates its solo recall without real "
        "skill — read its solo-test row with that in mind."
    )


def render_methodology():
    st.subheader("Phase 2 Mechanism Fix — Before / After")
    st.caption(
        "accuracy-roadmap Phase 2: (1) all evidence now votes on phase via a directional "
        "axis (trend + zigzag, valuation as tiebreaker) and a momentum axis (STC + earnings, "
        "liquidity as tiebreaker) instead of only trend_context + stc; (2) a minimum-dwell "
        "hysteresis filter (min_dwell=5, calibrated on 2018-2020, confirmed on 2021-2023, "
        "reported once on 2024+) requires 5 consecutive same-direction raw reads before a "
        "phase change is confirmed."
    )

    rows = [
        {"variant": "Before (2-signal, no hysteresis)", "phase_changes": 166, "mean_run_days": 11.0, "recall": "46.4%", "lift": 0.96},
        {"variant": "Axis-vote only (min_dwell=1)", "phase_changes": 165, "mean_run_days": 11.3, "recall": "46.4%", "lift": 0.92},
        {"variant": "Axis-vote + hysteresis (adopted, min_dwell=5)", "phase_changes": 40, "mean_run_days": 45.8, "recall": "46.4%", "lift": 0.92},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.success(
        "**Whipsaw fix: adopted.** Hysteresis cuts phase transitions from 166 to 40 over the "
        "same 626 evaluation points, pushing mean phase duration from ~11 to ~46 trading days "
        "(~2 months) — consistent with what 'market cycle' should mean, and calibrated without "
        "touching the 2024+ test slice."
    )
    st.error(
        "**transition_risk early-warning skill: honest negative finding, unchanged.** Recall "
        "against independently-labeled major turns stays at 46.4% and the precision-vs-base-rate "
        "lift stays at/below 1.0 (0.92) regardless of the mechanism fix — neither the axis-vote "
        "change nor hysteresis meaningfully improves it. This re-confirms the original audit's "
        "finding rather than fixing it; an earlier attempt to fold hysteresis-pending status into "
        "transition_risk was tried and reverted because it only inflated recall by widening the "
        "'elevated' base rate (37% -> 61-74%) without any real gain in precision lift."
    )
    st.caption(
        "Reproduce: `python scripts/backtest_cycle_phase.py --min-dwell {1,5}` and "
        "`python scripts/validate_cycle_classifier.py --min-dwell {1,5}`. Grid search behind "
        "min_dwell=5: `python scripts/calibrate_min_dwell.py` "
        "(scripts/output/hysteresis_calibration_grid.csv)."
    )


@st.cache_data(ttl=300)
def _load_universe_data() -> pd.DataFrame:
    return pd.read_csv(_UNIVERSE_CSV, parse_dates=["as_of_date"])


@st.cache_data(ttl=300, show_spinner="Fetching live data and classifying...")
def _fetch_live_stock(ticker: str):
    from shankh.agents.market.universe_cycle import fetch_and_classify_live
    return fetch_and_classify_live(ticker)


def render_universe_view():
    st.title("Universe Cycle Map — All NSE-Listed Stocks")
    st.caption(
        "The SAME validated mechanism used for the Nifty 50 index above (axis-vote + "
        "hysteresis, accuracy-roadmap Phases 0-5) applied to every NSE-listed stock "
        "individually. This is a cross-sectional SNAPSHOT (each stock's current phase, one "
        "point in time) — not a historical backtest per stock, which is a larger separate "
        "undertaking. Built by `scripts/build_universe_cycle_data.py`."
    )

    if not _UNIVERSE_CSV.exists():
        st.warning(
            "No universe dataset found yet. Build it with:\n\n"
            "```\npython scripts/build_universe_cycle_data.py\n```\n\n"
            "(add `--limit 200` for a quick test run first — the full NSE universe takes "
            "several minutes)."
        )
        return

    df = _load_universe_data()
    snapshot_date = df["as_of_date"].max()
    snapshot_age_days = (pd.Timestamp.now().normalize() - snapshot_date).days
    st.caption(f"{len(df)} stocks classified, as of {snapshot_date.date()} ({snapshot_age_days} days ago).")
    if snapshot_age_days > 3:
        st.warning(
            f"⚠️ This treemap/table is a **snapshot** from {snapshot_age_days} days ago — prices and phases "
            f"shown here (including 'Close') are as of {snapshot_date.date()}, not today. Rebuild it with "
            "`python scripts/build_universe_cycle_data.py`, or use **Stock detail** below and click "
            "**Get live data** for an up-to-the-minute read on any single stock."
        )

    # --- Filters ---
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        phases = st.multiselect("Phase", ["EXPANSION", "DISTRIBUTION", "CONTRACTION", "ACCUMULATION"],
                                 default=["EXPANSION", "DISTRIBUTION", "CONTRACTION", "ACCUMULATION"])
    with c2:
        sectors = st.multiselect("Sector", sorted(df["sector"].dropna().unique().tolist()), default=[])
    with c3:
        search = st.text_input("Search ticker or company name", "")

    filtered = df[df["cycle_phase"].isin(phases)]
    if sectors:
        filtered = filtered[filtered["sector"].isin(sectors)]
    if search:
        s = search.strip().upper()
        filtered = filtered[
            filtered["ticker"].str.upper().str.contains(s)
            | filtered["company_name"].str.upper().str.contains(s)
        ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks shown", len(filtered))
    for col, phase in zip([m2, m3, m4], ["EXPANSION", "DISTRIBUTION", "CONTRACTION"]):
        pct = (filtered["cycle_phase"] == phase).mean() * 100 if len(filtered) else 0
        col.metric(phase.title(), f"{pct:.0f}%")

    st.markdown("#### Sector Treemap — box size = stock count, color = cycle phase")
    if len(filtered):
        tree_df = filtered.copy()
        tree_df["sector"] = tree_df["sector"].fillna("Unknown")
        fig = px.treemap(
            tree_df, path=[px.Constant("NSE Universe"), "sector", "ticker"],
            color="cycle_phase", color_discrete_map={**PHASE_COLORS, "(?)": "#888888"},
            hover_data={"cycle_phase": True, "cycle_confidence": True, "pe_ratio": ":.1f", "close": ":.1f"},
        )
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stocks match the current filters.")

    st.markdown("#### Stock detail")
    if len(filtered):
        options = (filtered["ticker"] + " — " + filtered["company_name"]).tolist()
        picked = st.selectbox("Pick a stock for its full evidence trail", options)
        ticker = picked.split(" — ")[0]
        row = filtered[filtered["ticker"] == ticker].iloc[0]

        live_col1, live_col2 = st.columns([1, 3])
        with live_col1:
            get_live = st.button("🔄 Get live data", key=f"live_{ticker}")
        if get_live:
            live_row = _fetch_live_stock(ticker)
            if live_row is None:
                with live_col2:
                    st.error(f"Live fetch failed for {ticker} (insufficient history or data unavailable).")
            else:
                row = pd.Series(live_row)
                with live_col2:
                    st.success(f"✅ Live as of {row['as_of_date']} — this replaces the {snapshot_age_days}-day-old snapshot above for this stock only.")
        else:
            with live_col2:
                st.caption(f"Showing the {snapshot_age_days}-day-old snapshot (as of {row['as_of_date']}). Click **Get live data** for right now.")

        phase_color = PHASE_COLORS.get(row["cycle_phase"], "#888")
        risk_color = RISK_COLORS.get(row["transition_risk"], "#888")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Phase", row["cycle_phase"])
        d2.metric("Confidence", f"{row['cycle_confidence']:.0%}")
        d3.metric("Transition Risk", row["transition_risk"].upper())
        d4.metric("Close", f"₹{row['close']:.2f}")
        if pd.notna(row.get("pending_phase")) and row.get("pending_phase"):
            st.warning(f"Phase change building: {row['cycle_phase']} → {row['pending_phase']} "
                       f"({row['dwell_progress']} confirmations)")

        st.markdown(
            f"""<div style="display:flex; gap:10px; margin-bottom:10px;">
            <span style="background:{phase_color}20; color:{phase_color}; border:1px solid {phase_color};
            border-radius:6px; padding:2px 10px; font-size:13px; font-weight:600;">{row['cycle_phase']}</span>
            <span style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color};
            border-radius:6px; padding:2px 10px; font-size:13px; font-weight:600;">RISK: {row['transition_risk'].upper()}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        evidence_cols = {
            "Momentum/Trend": {"zigzag_score": row["zigzag_score"], "stc_score": row["stc_score"], "trend_score": row["trend_score"]},
            "Valuation": {"pe_ratio": row["pe_ratio"], "pe_percentile_vs_universe": row["pe_percentile_vs_universe"], "pb_ratio": row["pb_ratio"]},
            "Earnings": {"earnings_quarterly_growth_percent": row["earnings_quarterly_growth_percent"]},
            "New indicators": {"candlestick_score": row["candlestick_score"],
                               "harmonic_pattern": row["harmonic_pattern"] if pd.notna(row["harmonic_pattern"]) and row["harmonic_pattern"] != "" else "none",
                               "gann_score": row["gann_score"]},
        }
        cols = st.columns(len(evidence_cols))
        for col, (title, vals) in zip(cols, evidence_cols.items()):
            with col:
                st.caption(f"**{title}**")
                for k, v in vals.items():
                    st.text(f"{k}: {v}")

    with st.expander(f"Full table ({len(filtered)} stocks)"):
        st.dataframe(
            filtered[["ticker", "company_name", "sector", "cycle_phase", "cycle_confidence",
                      "transition_risk", "close", "pe_ratio", "pe_percentile_vs_universe"]],
            use_container_width=True, hide_index=True,
        )


def main():
    fmp_key, fred_key = render_api_key_sidebar()

    view = st.radio("View", ["Nifty 50 (Index)", "Universe (All NSE Stocks)"], horizontal=True, label_visibility="collapsed")
    st.divider()

    if view == "Universe (All NSE Stocks)":
        render_universe_view()
        return

    render_header(fmp_key, fred_key)
    st.divider()

    try:
        data = fetch_synthesis(fmp_key, fred_key)
    except Exception as exc:
        st.error(f"Synthesis failed: {exc}")
        return

    if data.get("status") == "unavailable":
        st.error(data.get("error", "Synthesis unavailable."))
        return

    render_classification(data)
    st.divider()
    render_evidence(data)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Backtest", "Validation", "Indicators", "Methodology"])
    with tab1:
        render_backtest()
    with tab2:
        render_validation()
    with tab3:
        render_indicators()
    with tab4:
        render_methodology()


if __name__ == "__main__":
    main()
