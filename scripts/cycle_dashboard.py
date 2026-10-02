"""
Market Cycle Agent dashboard.

Four pages, one question each:
    Market Map      Where is every NSE stock in its cycle, and what just changed?
    Stock Explorer  Why is this stock in this phase, and how did it get here?
    Nifty 50        What phase is the index in right now?
    Research        Why should I trust this? (method, validation, experiments)

Run:  streamlit run scripts/cycle_dashboard.py

Data keys are optional. A viewer's FMP/FRED key lives only in that browser session
(shankh.agents.market.cycle_tools.set_session_api_keys is a per-thread override, not an
env var), so on a shared deployment one person's key never reaches another's request.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shankh.agents.market.cycle_explain import explain_classification  # noqa: E402
from shankh.agents.market.cycle_history import fetch_history, run_walk_forward  # noqa: E402
from shankh.agents.market.cycle_tools import get_market_cycle_synthesis, set_session_api_keys  # noqa: E402
from shankh.agents.market.universe_cycle import (  # noqa: E402
    _LATEST_CSV as _UNIVERSE_CSV,
    _PHASE_CHANGES_CSV,
)

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

PHASES = ["EXPANSION", "DISTRIBUTION", "CONTRACTION", "ACCUMULATION"]
PHASE_COLORS = {
    "EXPANSION": "#2E7D32",
    "DISTRIBUTION": "#C77700",
    "CONTRACTION": "#C62828",
    "ACCUMULATION": "#1565C0",
}
RISKS = ["low", "medium", "high"]
PAGES = ["Market Map", "Stock Explorer", "Nifty 50", "Research"]
STALE_AFTER_DAYS = 3  # covers a normal weekend; beyond it the scheduled refresh has likely failed

st.set_page_config(page_title="Market Cycle Agent", page_icon="🔄", layout="wide")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple:
    """Keys are operator settings, so they live in a collapsed expander."""
    with st.sidebar.expander("Data keys (optional)"):
        fmp_key = st.text_input("FMP key", type="password", key="fmp_key_input")
        fred_key = st.text_input("FRED key", type="password", key="fred_key_input")
        st.caption("Without a key, valuation and liquidity fall back to labelled defaults.")
    return fmp_key, fred_key


def fmt(value, template: str, missing: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return missing
    return template.format(value)


@st.cache_data(ttl=300)
def load_universe() -> pd.DataFrame:
    return pd.read_csv(_UNIVERSE_CSV, parse_dates=["as_of_date"])


@st.cache_data(ttl=300)
def load_phase_changes() -> pd.DataFrame:
    if not _PHASE_CHANGES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(_PHASE_CHANGES_CSV, parse_dates=["date"])


@st.cache_data(ttl=300, show_spinner="Fetching live data and classifying...")
def fetch_live_stock(ticker: str, fmp_key: str, fred_key: str):
    # keys are cache arguments on purpose: two viewers with different keys must never
    # be served each other's cached result.
    set_session_api_keys(fmp_api_key=fmp_key, fred_api_key=fred_key)
    from shankh.agents.market.universe_cycle import fetch_and_classify_live
    return fetch_and_classify_live(ticker)


@st.cache_data(ttl=300, show_spinner="Running the index analysis...")
def fetch_index_synthesis(fmp_key: str, fred_key: str) -> dict:
    set_session_api_keys(fmp_api_key=fmp_key, fred_api_key=fred_key)
    return json.loads(get_market_cycle_synthesis.invoke({}))


@st.cache_data(ttl=3600, show_spinner="Building the history (about 10-30 seconds)...")
def phase_history(ticker: str, years: int, with_context: bool = False):
    """Walk-forward phase per ~3 trading days for the last `years`, plus raw prices.
    Warm-up history before the window is fetched so the first phase is already valid."""
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.DateOffset(years=years)
    prices = fetch_history((cutoff - pd.Timedelta(days=420)).strftime("%Y-%m-%d"), ticker=ticker)
    hist = run_walk_forward(prices, step=3, min_dwell=5, include_context_indicators=with_context)
    return hist[hist["date"] >= cutoff].reset_index(drop=True), prices[prices["date"] >= cutoff].reset_index(drop=True)


def phase_shapes(hist: pd.DataFrame) -> list:
    """Contiguous same-phase runs as shaded x-ranges, so the chart shows each phase once."""
    runs, start, current = [], None, None
    for date, phase in zip(hist["date"], hist["cycle_phase"]):
        if phase != current:
            if current is not None:
                runs.append((start, date, current))
            start, current = date, phase
    if current is not None:
        runs.append((start, hist["date"].iloc[-1], current))
    return runs


def add_phase_shading(fig, hist: pd.DataFrame, **kwargs) -> None:
    for x0, x1, phase in phase_shapes(hist):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_COLORS[phase], opacity=0.16, line_width=0, **kwargs)
    for phase in PHASES:  # legend swatches, one per phase
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=phase.title(),
                                 marker=dict(size=10, color=PHASE_COLORS[phase], symbol="square")),
                      **({"row": 1, "col": 1} if "row" in kwargs else {}))


def price_figure(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=hist["date"], y=hist["close"], mode="lines", name="Close",
                               line=dict(color="#90A4AE", width=2)))
    add_phase_shading(fig, hist)
    fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10), yaxis_title="Close (₹)",
                      legend=dict(orientation="h", y=1.08))
    return fig


# ---------------------------------------------------------------------------
# Page 1 - Market Map
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
    phases = c1.multiselect("Phase", PHASES, default=PHASES)
    sectors = c2.multiselect("Sector", sorted(df["sector"].dropna().unique()))
    risks = c3.multiselect("Chance of change", RISKS, default=RISKS)
    search = c4.text_input("Search ticker or company")
    with st.expander("More filters"):
        m1, m2 = st.columns(2)
        min_conf = m1.slider("Minimum confidence", 0, 100, 0, format="%d%%")
        max_pe = m2.number_input("Maximum P/E (0 = no limit)", min_value=0.0, value=0.0, step=5.0)

    out = df[df["cycle_phase"].isin(phases) & df["transition_risk"].isin(risks)]
    out = out[out["cycle_confidence"] * 100 >= min_conf]
    if sectors:
        out = out[out["sector"].isin(sectors)]
    if max_pe > 0:
        out = out[out["pe_ratio"].between(0, max_pe)]
    if search:
        s = search.strip().upper()
        out = out[out["ticker"].str.upper().str.contains(s) | out["company_name"].str.upper().str.contains(s)]
    return out


def render_market_map() -> None:
    st.header("Market Map")
    if not _UNIVERSE_CSV.exists():
        st.warning("No universe data yet. Run `python scripts/build_universe_cycle_data.py`.")
        return

    df = load_universe()
    as_of = df["as_of_date"].max()
    age = (pd.Timestamp.now().normalize() - as_of).days
    st.caption(f"Current phase of {len(df):,} NSE stocks as of {as_of.date()}. Refreshed every weekday after market close.")
    if age > STALE_AFTER_DAYS:
        st.warning(f"This data is {age} days old, so the scheduled refresh may have failed. "
                   "Use Stock Explorer for a live read on a single stock.")

    filtered = apply_filters(df)
    if filtered.empty:
        st.info("No stocks match these filters.")
        return

    cols = st.columns(4)
    for col, phase in zip(cols, PHASES):
        n = int((filtered["cycle_phase"] == phase).sum())
        col.metric(phase.title(), f"{n:,}", f"{n / len(filtered):.0%} of {len(filtered):,}", delta_color="off")

    st.subheader("Phase by sector")
    tree = filtered.assign(sector=filtered["sector"].fillna("Unknown"))
    fig = px.treemap(tree, path=[px.Constant("NSE"), "sector", "ticker"], color="cycle_phase",
                     color_discrete_map={**PHASE_COLORS, "(?)": "#888888"},
                     hover_data={"cycle_phase": True, "cycle_confidence": ":.0%", "pe_ratio": ":.1f", "close": ":.1f"})
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=520)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recently changed phase")
    feed = load_phase_changes()
    if feed.empty:
        st.caption("No changes recorded yet. This fills in as the daily refresh runs.")
    else:
        window = st.segmented_control("Window", [7, 30, 90], default=30, format_func=lambda d: f"Last {d} days",
                                      label_visibility="collapsed")
        recent = feed[feed["ticker"].isin(filtered["ticker"]) & (feed["date"] >= feed["date"].max() - pd.Timedelta(days=window or 30))]
        if recent.empty:
            st.caption("No filtered stock changed phase in this window.")
        else:
            show = recent.assign(date=recent["date"].dt.date, close=recent["close"].round(2))
            st.dataframe(show[["date", "ticker", "company_name", "sector", "from_phase", "to_phase", "close"]]
                         .rename(columns={"company_name": "company", "from_phase": "from", "to_phase": "to"}),
                         hide_index=True, width="stretch")

    st.subheader("Watch list: high chance of a phase change")
    watch = filtered[filtered["transition_risk"] == "high"].sort_values("cycle_confidence", ascending=False)
    if watch.empty:
        st.caption("No stock in the current filter is flagged high.")
    else:
        st.dataframe(watch[["ticker", "company_name", "sector", "cycle_phase", "cycle_confidence", "close"]].head(50)
                     .assign(cycle_confidence=lambda d: (d["cycle_confidence"] * 100).round(0)),
                     hide_index=True, width="stretch",
                     column_config={"cycle_confidence": st.column_config.NumberColumn("Confidence", format="%d%%")})

    st.subheader("Screener")
    table = filtered[["ticker", "company_name", "sector", "cycle_phase", "cycle_confidence",
                      "transition_risk", "close", "pe_ratio", "as_of_date"]].copy()
    table["cycle_confidence"] = (table["cycle_confidence"] * 100).round(0)
    table["as_of_date"] = table["as_of_date"].dt.date
    table = table.rename(columns={"company_name": "company", "cycle_phase": "phase", "cycle_confidence": "confidence",
                                  "transition_risk": "chance of change", "pe_ratio": "P/E", "as_of_date": "as of"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.NumberColumn(format="%d%%"),
                                "close": st.column_config.NumberColumn(format="₹%.2f"),
                                "P/E": st.column_config.NumberColumn(format="%.1f")})
    st.download_button(f"Download {len(table):,} rows (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"market_cycle_{as_of.date()}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Page 2 - Stock Explorer
# ---------------------------------------------------------------------------

def evidence_rows(row: dict) -> pd.DataFrame:
    pe = fmt(row.get("pe_ratio"), "{:.1f}")
    pct = row.get("pe_percentile_vs_universe")
    if pe != "n/a" and fmt(pct, "{}") != "n/a":
        pe = f"{pe} (higher than {pct:.0f}% of stocks)"  # percentile = share of stocks with a P/E at or below this one
    return pd.DataFrame([
        ("Trend vs 200-day average", "Core", fmt(row.get("trend_score"), "{:+.2f}")),
        ("Swing structure (ZigZag)", "Core", fmt(row.get("zigzag_score"), "{:+.2f}")),
        ("Momentum (STC)", "Core", fmt(row.get("stc_score"), "{:+.2f}")),
        ("Valuation: P/E", "Supporting", pe),
        ("Earnings growth (quarterly)", "Supporting", fmt(row.get("earnings_quarterly_growth_percent"), "{:+.1f}%")),
    ], columns=["Signal", "Weight", "Reading"])


def render_stock_explorer(fmp_key: str, fred_key: str) -> None:
    st.header("Stock Explorer")
    if not _UNIVERSE_CSV.exists():
        st.warning("No universe data yet. Run `python scripts/build_universe_cycle_data.py`.")
        return

    df = load_universe()
    labels = (df["ticker"] + "  ·  " + df["company_name"]).tolist()
    picked = st.selectbox("Stock", labels, index=None, placeholder="Search a ticker or company")
    if picked is None:
        st.caption("Pick a stock to see its phase, the reason, and its history.")
        return
    ticker = picked.split("  ·  ")[0]
    snapshot = df[df["ticker"] == ticker].iloc[0].to_dict()

    if st.button("Get live data", help="Fetch today's prices and reclassify this stock now."):
        live = fetch_live_stock(ticker, fmp_key, fred_key)
        if live is None:
            st.error(f"Live data is unavailable for {ticker}.")
            row, source = snapshot, f"Snapshot, {snapshot['as_of_date'].date()}"
        else:
            row, source = live, f"Live, {live['as_of_date']}"
    else:
        row, source = snapshot, f"Snapshot, {snapshot['as_of_date'].date()}"
    st.caption(f"{row.get('company_name', ticker)}  |  {row.get('sector', '')}  |  Source: {source}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Phase", str(row["cycle_phase"]).title())
    m2.metric("Confidence", f"{row['cycle_confidence']:.0%}")
    m3.metric("Chance of change", str(row["transition_risk"]).title())
    m4.metric("Close", f"₹{row['close']:,.2f}")

    st.info(explain_classification(row))

    st.subheader("Price history by phase")
    years = st.segmented_control("Timeline", [1, 2, 3, 5], default=2, format_func=lambda y: f"{y}y",
                                 label_visibility="collapsed") or 2
    try:
        hist, _ = phase_history(ticker, years)
        st.plotly_chart(price_figure(hist), width="stretch")
        st.caption("Built from trend, swings and momentum only. The phase above also uses valuation and earnings, so the two can differ.")
    except Exception as exc:
        st.error(f"Could not build the history: {exc}")

    with st.expander("Evidence behind this phase"):
        st.dataframe(evidence_rows(row), hide_index=True, width="stretch")
    with st.expander("Reference only (not used in the phase)"):
        pattern = row.get("harmonic_pattern")
        st.write(f"Candlestick score {fmt(row.get('candlestick_score'), '{:+.2f}')}  ·  "
                 f"Gann score {fmt(row.get('gann_score'), '{:+.2f}')}  ·  "
                 f"Harmonic pattern {pattern if isinstance(pattern, str) and pattern else 'none'}")
        st.caption("Tested against history and dropped from the vote. See Research.")


# ---------------------------------------------------------------------------
# Page 3 - Nifty 50
# ---------------------------------------------------------------------------

def index_row(data: dict) -> dict:
    scores = {e["signal"]: e["score"] for e in data.get("evidence", [])}
    return {"cycle_phase": data.get("cycle_phase"), "transition_risk": data.get("transition_risk"),
            "pending_phase": data.get("pending_phase"), "dwell_progress": data.get("dwell_progress"),
            "trend_score": scores.get("trend_context"), "zigzag_score": scores.get("zigzag"),
            "stc_score": scores.get("stc")}


def render_index(fmp_key: str, fred_key: str) -> None:
    head, action = st.columns([5, 1])
    head.header("Nifty 50")
    if action.button("Refresh"):
        fetch_index_synthesis.clear()
        st.rerun()

    try:
        data = fetch_index_synthesis(fmp_key, fred_key)
    except Exception as exc:
        st.error(f"The index analysis failed: {exc}")
        return
    if data.get("status") == "unavailable":
        st.error(data.get("error", "The index analysis is unavailable."))
        return

    st.caption(f"As of {data.get('as_of_date', 'n/a')}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Phase", str(data.get("cycle_phase", "n/a")).title())
    m2.metric("Confidence", f"{data.get('cycle_confidence', 0.0):.0%}")
    m3.metric("Chance of change", str(data.get("transition_risk", "n/a")).title())
    st.info(explain_classification(index_row(data)))

    evidence = data.get("evidence", [])
    if evidence:
        st.subheader("Evidence")
        table = pd.DataFrame(evidence)[["tier", "signal", "reading", "score"]]
        table["score"] = table["score"].map(lambda s: f"{s:+.2f}")
        st.dataframe(table.rename(columns=str.title), hide_index=True, width="stretch")
        with st.expander("What each signal means"):
            for e in evidence:
                st.markdown(f"**{e['signal']}**: {e.get('note', '')}")


# ---------------------------------------------------------------------------
# Page 4 - Research
# ---------------------------------------------------------------------------

PARAMETER_REGISTER = pd.DataFrame([
    ("Swing size", "4% reversal", "Above typical daily noise, so a swing is a real move."),
    ("Momentum (STC)", "23 / 50 / 10, blend 0.5 / 0.5", "Standard Schaff settings. The blend was tested; no value beat 0.5 on both test periods."),
    ("Trend", "Rank vs the stock's own 200-day-average history (252+ days)", "Replaced a fixed ±10% band that was maxed out for 51% of stocks."),
    ("Phase confirmation", "5 readings in a row", "Chosen on 2018-20, confirmed on 2021-23. Phases now last about 46 days instead of 11."),
    ("Chance of change", "Medium above 0.5, high above 1.35", "Medium was recalibrated from 0.9: better recall on both periods. High cannot be tested with this method, so it is unchanged."),
    ("Confidence", "Agreement with the two votes × how complete the evidence is", "Stocks with thin data can no longer look more certain than well-covered ones."),
    ("Data", "FMP first, yfinance as backup; refreshed weekdays 5:30 PM IST", "Fundamentals coverage rose from about 55% to about 100%."),
], columns=["Parameter", "Value", "Why"])


def research_how_it_works() -> None:
    diagram = _OUTPUT_DIR / "market_cycle_workflow.png"
    if diagram.exists():
        st.image(str(diagram), width="stretch")
    st.subheader("Parameters in use")
    st.dataframe(PARAMETER_REGISTER, hide_index=True, width="stretch")


def research_backtest() -> None:
    chart = _OUTPUT_DIR / "cycle_backtest_chart.png"
    if chart.exists():
        st.image(str(chart), width="stretch")
        st.caption("Nifty 50, 2018 onward. Price-based signals only, no lookahead.")
    else:
        st.warning("No backtest chart. Run `python scripts/backtest_cycle_phase.py`.")
    history = _OUTPUT_DIR / "cycle_backtest_history.csv"
    if history.exists():
        hist = pd.read_csv(history)
        with st.expander(f"Raw history ({len(hist)} points)"):
            st.dataframe(hist.tail(50), width="stretch")


def research_validation() -> None:
    report = _OUTPUT_DIR / "validation_report.txt"
    if not report.exists():
        st.warning("No validation report. Run `python scripts/validate_cycle_classifier.py`.")
        return
    text = report.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    metrics = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in lines if ":" in l)}
    recall_line = next((l for l in lines if l.startswith("Recall")), "")
    recall = recall_line.split(":", 1)[1].split("(")[0].strip() if ":" in recall_line else "n/a"
    try:
        lift = float(metrics.get("lift_over_random", ""))
    except ValueError:
        lift = None

    c1, c2 = st.columns(2)
    c1.metric("Recall", recall, help="Share of major market turns that had an elevated warning in the 15 days before.")
    c2.metric("Lift", f"{lift:.2f}" if lift is not None else "n/a", help="Warning quality vs chance. Above 1.0 is better than chance.")
    if lift is not None:
        st.write("Warnings are better than chance." if lift >= 1.1
                 else "Warnings are not clearly better than chance, so treat the chance-of-change signal as soft.")
    with st.expander("Full report"):
        st.code(text, language="text")
        st.caption("Turn dates come from a separate algorithm, not the classifier's own ZigZag, so the answer key is independent.")
    turns = _OUTPUT_DIR / "independent_labeled_turns.csv"
    if turns.exists():
        with st.expander("Labelled major turns"):
            st.dataframe(pd.read_csv(turns), width="stretch")


def research_indicators() -> None:
    ranking_path = _OUTPUT_DIR / "indicator_ranking.csv"
    if not ranking_path.exists():
        st.warning("No indicator evaluation. Run `python scripts/evaluate_indicators.py`.")
        return
    ranking = pd.read_csv(ranking_path)
    ranking["decision"] = ranking["verdict"].str.split().str[0].str.strip(".,;:")
    st.caption("Every candidate indicator went through the same tests before being allowed to vote.")
    st.dataframe(ranking[["indicator", "decision", "lift", "recall", "fired_count"]]
                 .rename(columns={"fired_count": "times fired"}), hide_index=True, width="stretch")
    st.caption("Decision: CORE votes, WATCH needs more data, DROP is excluded. Lift above 1.0 beats chance.")

    ablation = _OUTPUT_DIR / "indicator_ablation_chart.png"
    if ablation.exists():
        st.subheader("Does adding them help?")
        st.image(str(ablation), width="stretch")
    with st.expander("Detailed charts"):
        for name, title in [("indicator_forward_return_boxplots.png", "20-day forward return by signal"),
                            ("indicator_regime_heatmap.png", "Signal vs phase"),
                            ("indicator_signal_timeline.png", "Signal timeline")]:
            path = _OUTPUT_DIR / name
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), width="stretch")


def research_volume() -> None:
    st.markdown("**Question:** does adding volume signals improve the phase warnings?  \n"
                "**Method:** the same test as every other indicator, run on Nifty 50 plus four stocks, with and without volume.")
    verdict_path = _OUTPUT_DIR / "volume_experiment_verdict.txt"
    onepager = _OUTPUT_DIR / "volume_experiment_onepager.png"
    summary = _OUTPUT_DIR / "volume_experiment_summary.csv"
    if not verdict_path.exists():
        st.warning("No volume experiment found. Run `python scripts/volume_experiment_report.py`.")
        return

    st.success(f"**Result:** {verdict_path.read_text(encoding='utf-8')}")
    st.write("**Decision:** volume is not added to the phase vote. It stays computed for reference.")
    if onepager.exists():
        st.image(str(onepager), width="stretch")
    if summary.exists():
        with st.expander("Numbers behind the chart"):
            table = pd.read_csv(summary)
            st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("See the volume signals on a stock")
    labels = (load_universe()["ticker"]).tolist() if _UNIVERSE_CSV.exists() else []
    ticker = st.selectbox("Stock", labels, index=None, placeholder="Pick a ticker (first run takes about 30 seconds)",
                          key="volume_lens_ticker")
    if ticker:
        try:
            hist, prices = phase_history(ticker, 2, True)
        except Exception as exc:
            st.error(f"Could not build the view: {exc}")
            return
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Close", line=dict(color="#90A4AE", width=2)), row=1, col=1)
        add_phase_shading(fig, hist, row=1, col=1)
        fig.add_trace(go.Bar(x=prices["date"], y=prices["volume"], name="Volume", marker_color="#90A4AE", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["obv_score"], name="OBV score", line=dict(color="#5C6BC0")), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["cmf_score"], name="CMF score", line=dict(color="#26A69A")), row=3, col=1)
        fig.update_layout(height=620, margin=dict(t=10, l=10, r=10, b=10), legend=dict(orientation="h", y=1.05))
        fig.update_yaxes(title_text="Close", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=1)
        st.plotly_chart(fig, width="stretch")
        st.caption("OBV and CMF measure buying and selling pressure. They are shown for comparison and do not affect the phase.")


def research_methodology() -> None:
    st.caption("Phase 2 experiment, as measured at the time on the Nifty 50. Later recalibrations are on the Validation tab.")
    st.dataframe(pd.DataFrame([
        {"variant": "Before (two signals, no confirmation)", "phase changes": 166, "mean phase length (days)": 11.0},
        {"variant": "Axis vote only", "phase changes": 165, "mean phase length (days)": 11.3},
        {"variant": "Axis vote + 5-reading confirmation (adopted)", "phase changes": 40, "mean phase length (days)": 45.8},
    ]), hide_index=True, width="stretch")
    st.write("Confirmation cut phase flips from 166 to 40 without improving warning accuracy. "
             "An earlier attempt to fold pending changes into the warning level was reverted: it only raised recall by firing more often.")


def render_research() -> None:
    st.header("Research")
    tabs = st.tabs(["How it works", "Backtest", "Validation", "Indicators", "Volume", "Method history"])
    for tab, fn in zip(tabs, [research_how_it_works, research_backtest, research_validation,
                              research_indicators, research_volume, research_methodology]):
        with tab:
            fn()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fmp_key, fred_key = render_sidebar()
    set_session_api_keys(fmp_api_key=fmp_key, fred_api_key=fred_key)
    st.title("Market Cycle Agent")
    page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed")
    st.divider()
    # radio, not tabs: only the chosen page runs, so the slow live index analysis
    # doesn't execute while someone is browsing the Market Map.
    if page == "Market Map":
        render_market_map()
    elif page == "Stock Explorer":
        render_stock_explorer(fmp_key, fred_key)
    elif page == "Nifty 50":
        render_index(fmp_key, fred_key)
    else:
        render_research()


if __name__ == "__main__":
    main()
