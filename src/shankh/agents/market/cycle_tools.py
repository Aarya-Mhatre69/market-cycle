"""
Market Cycle Tools Module.

Computes the Cycle Agent's own price-structure "skeleton" (ZigZag / Schaff Trend Cycle /
trend context — see cycle_signals.py), Equity Risk Premium (ERP), index valuation
percentiles, and bank credit/monetary liquidity metrics, then combines them into a
single weighted-evidence Market Cycle classification (phase + confidence + transition
risk) per the Market Cycle Agent architecture spec.

Ownership note: US 10Y-2Y yield curve slope and India-US yield spread are the Macro
Agent's tools (get_indian_macro_indicators), not recomputed here, to avoid two agents
independently fetching the same series and silently disagreeing.
"""

import contextvars
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from langchain_core.tools import tool

from shankh.agents.market.cycle_signals import (
    CycleClassification,
    EvidenceItem,
    HysteresisState,
    classify_cycle,
    classify_cycle_stateful,
    compute_stc,
    compute_trend_context,
    compute_zigzag,
)

logger = logging.getLogger(__name__)

# Per-session API key override — lets a UI (e.g. the Streamlit sidebar) supply keys
# at runtime instead of requiring FMP_API_KEY/FRED_API_KEY in the server's own .env.
# Deliberately NOT `os.environ[...] = value`: this module can be shared by multiple
# concurrent users (e.g. a dashboard deployed for a team), and mutating process-wide
# env vars would leak one viewer's key into another viewer's concurrent request.
# contextvars.ContextVar isolates by thread/async-task instead — correct for
# Streamlit, which runs each browser session's script execution on its own thread.
_fmp_key_override: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("fmp_key_override", default=None)
_fred_key_override: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("fred_key_override", default=None)


def set_session_api_keys(fmp_api_key: Optional[str] = None, fred_api_key: Optional[str] = None) -> None:
    """
    Called once per script run by a UI (e.g. cycle_dashboard.py's sidebar) to supply
    API keys for the current session/thread only. Passing None for a key leaves any
    existing override for it unchanged; pass "" explicitly to clear an override and
    fall back to the server's own environment variable.
    """
    if fmp_api_key is not None:
        _fmp_key_override.set(fmp_api_key or None)
    if fred_api_key is not None:
        _fred_key_override.set(fred_api_key or None)


def _get_fmp_api_key() -> Optional[str]:
    return _fmp_key_override.get() or os.getenv("FMP_API_KEY")


def _get_fred_api_key() -> Optional[str]:
    return _fred_key_override.get() or os.getenv("FRED_API_KEY")

# Phase 2 hysteresis (accuracy-roadmap): min_dwell=5 is the calibrated, adopted value
# (see scripts/calibrate_min_dwell.py — chosen on the 2018-2020 calibration window,
# confirmed on 2021-2023 validation, never re-tuned against the 2024+ test slice).
_HYSTERESIS_MIN_DWELL = 5

# Live phase-confirmation state persists across calls to get_market_cycle_synthesis
# (roadmap Phase 2 implementation note: "needs a clear owner decision on where state
# lives, e.g. in the backtest loop vs. a persisted last-known-phase for live use").
# A single local JSON file is the smallest defensible choice for a single-process
# deployment at this stage — data/ is already the repo's convention for local,
# git-ignored state (see breadth_tools.py's data/universe/ cache).
_STATE_DIR = Path(__file__).resolve().parents[4] / "data"
_STATE_FILE = _STATE_DIR / "cycle_phase_state.json"


def _load_hysteresis_state() -> "tuple[HysteresisState, Optional[str]]":
    if _STATE_FILE.exists():
        try:
            payload = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            state = HysteresisState(
                confirmed_phase=payload.get("confirmed_phase"),
                candidate_phase=payload.get("candidate_phase"),
                candidate_count=payload.get("candidate_count", 0),
            )
            return state, payload.get("as_of_date")
        except Exception as exc:
            logger.warning("Failed to load persisted cycle phase state (%s); starting fresh.", exc)
    return HysteresisState(), None


def _save_hysteresis_state(state: HysteresisState, as_of_date: str) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "confirmed_phase": state.confirmed_phase,
        "candidate_phase": state.candidate_phase,
        "candidate_count": state.candidate_count,
        "as_of_date": as_of_date,
    }
    _STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _classify_with_persisted_hysteresis(evidence: List[EvidenceItem], as_of_date: str) -> CycleClassification:
    """
    Applies the hysteresis-confirmed classification using state persisted across
    process/tool-call boundaries, advancing the dwell count at most once per distinct
    `as_of_date` — calling this twice for the same trading day (e.g. a dashboard
    refresh) must not double-count that day's evidence toward phase confirmation.
    """
    state, last_date = _load_hysteresis_state()

    if as_of_date != "unavailable" and as_of_date != last_date:
        classification, new_state = classify_cycle_stateful(evidence, state, min_dwell=_HYSTERESIS_MIN_DWELL)
        _save_hysteresis_state(new_state, as_of_date)
        return classification

    # Same trading day as the last recorded update (or date unavailable) — report the
    # already-confirmed phase without re-running hysteresis a second time for today.
    raw = classify_cycle(evidence)
    return CycleClassification(
        cycle_phase=state.confirmed_phase or raw.cycle_phase,
        cycle_confidence=raw.cycle_confidence,
        transition_risk=raw.transition_risk,
        transition_watch=raw.transition_watch,
        composite_score=raw.composite_score,
        evidence=raw.evidence,
        pending_phase=state.candidate_phase,
        dwell_progress=f"{state.candidate_count}/{_HYSTERESIS_MIN_DWELL}" if state.candidate_phase else None,
    )

_HTTP_TIMEOUT = 10.0
_INDEX_HISTORY_DAYS = 420  # >= 200DMA warmup + lookback margin


def _fetch_index_ohlcv(days: int = _INDEX_HISTORY_DAYS) -> Optional[pd.DataFrame]:
    """Fetch daily Nifty 50 OHLCV. Tries FMP (existing key) first, falls back to yfinance."""
    fmp_api_key = _get_fmp_api_key()

    if fmp_api_key:
        try:
            url = (
                "https://financialmodelingprep.com/api/v3/historical-price-full/%5ENSEI"
                f"?timeseries={days}&apikey={fmp_api_key}"
            )
            res = requests.get(url, timeout=_HTTP_TIMEOUT).json()
            rows = res.get("historical", []) if isinstance(res, dict) else []
            if rows:
                df = pd.DataFrame(rows)[["date", "open", "high", "low", "close", "volume"]]
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                if len(df) >= 200:
                    return df
        except Exception as exc:
            logger.warning("FMP index OHLCV fetch failed: %s. Falling back to yfinance.", exc)

    try:
        hist = yf.Ticker("^NSEI").history(period=f"{days}d", interval="1d")
        if not hist.empty:
            df = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) >= 200:
                return df
    except Exception as exc:
        logger.warning("yfinance index OHLCV fallback failed: %s", exc)

    return None


@tool
def get_cycle_price_structure() -> str:
    """
    Compute the Cycle Agent's own price-structure signals: ZigZag swing structure,
    Schaff Trend Cycle (STC) momentum, and 50/200-day trend context on the Nifty 50 index.

    This is the "Core" tier — it works with zero dependency on other agents and forms
    the skeleton of the cycle classification.

    Returns:
        JSON string with per-signal readings/scores and a Core-only phase classification.
    """
    df = _fetch_index_ohlcv()
    if df is None:
        return json.dumps(
            {"error": "Could not retrieve Nifty 50 index OHLCV from FMP or yfinance.", "status": "unavailable"},
            indent=2,
        )

    zz = compute_zigzag(df)
    stc = compute_stc(df["close"])
    trend = compute_trend_context(df["close"])

    evidence = [
        EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
        EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
        EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}% vs 200DMA", trend.score, trend.note),
    ]
    # Deliberately the raw (stateless, un-hysteresised) read — this tool is documented
    # as a component-level detail breakdown, not the source of truth for the reported
    # phase (get_market_cycle_synthesis is, via the persisted hysteresis filter). Field
    # is named accordingly so it isn't mistaken for the confirmed live phase, which can
    # legitimately differ from this raw read while a phase change is pending.
    classification = classify_cycle(evidence)

    result = {
        "as_of_date": str(df["date"].iloc[-1].date()),
        "core_signals": {
            "zigzag": {"structure": zz.structure, "score": zz.score, "note": zz.note, "recent_swings": zz.pivots},
            "stc": {"value": stc.value, "direction": stc.direction, "score": stc.score, "note": stc.note},
            "trend_context": {
                "price_vs_sma200_pct": trend.price_vs_sma200_pct,
                "sma50_vs_sma200_pct": trend.sma50_vs_sma200_pct,
                "score": trend.score,
                "note": trend.note,
            },
        },
        "core_only_cycle_phase_raw": classification.cycle_phase,
        "core_only_confidence": classification.cycle_confidence,
        "core_only_composite_score": classification.composite_score,
        "note": "This is the raw, un-hysteresised instantaneous read from Core signals only. "
                "It can legitimately differ from get_market_cycle_synthesis's cycle_phase, which is "
                "the hysteresis-confirmed source of truth.",
    }
    return json.dumps(result, indent=2)


def _compute_pe_percentile(current_pe: float, historical_pe: Optional[List[float]]) -> Any:
    """
    Real rolling percentile: what fraction of `historical_pe` is <= `current_pe`.
    Pure function, unit-tested independently of any data fetch — see
    tests/unit/test_cycle_tools.py::test_compute_pe_percentile_*.

    Returns "insufficient_data" (not a guess) if fewer than 252 historical points
    (~1 trading year — a genuine "10-year percentile" needs far more, but this is the
    floor below which a percentile claim isn't defensible at all) are available.
    """
    if not historical_pe or len(historical_pe) < 252:
        return "insufficient_data"
    arr = np.array(historical_pe, dtype=float)
    return round(float((arr <= current_pe).mean() * 100.0), 1)


def _fetch_historical_pe_series() -> Optional[List[float]]:
    """
    Attempts to fetch ~10 years of historical Nifty 50 P/E ratio. No free, reliably
    machine-readable INDEX-level (as opposed to per-company) historical P/E API was
    found/validated for this integration — FMP's ratios/ratios-ttm endpoints are
    documented for individual companies, not indices, and this repo's FMP_API_KEY
    additionally returns 401 Invalid API KEY on every endpoint tested as of this
    writing (a separate, pre-existing issue — see get_market_cycle_metrics's comment).
    Returns None (honest "no data", not a synthetic estimate) until a real source is
    wired in. NSE India's public historical P/E/P/B/dividend-yield archive is the
    most likely real path forward but needs its own build-vs-buy evaluation (same
    category of decision already flagged for bank credit growth) before being added
    here — deliberately not scraped in as a fragile, undocumented dependency.
    """
    return None


@tool
def get_market_cycle_metrics() -> str:
    """
    Fetch market cycle valuation metrics: Equity Risk Premium (ERP) and index valuation
    percentiles for Nifty 50 (P/E, P/B, Dividend Yield vs 10-year history).

    Returns:
        JSON string containing valuation percentiles, ERP spread, and per-field data
        source labels (LIVE vs FALLBACK_BENCHMARK) so downstream consumers know which
        figures are real-time and which are static placeholders pending a data source.
    """
    fmp_api_key = _get_fmp_api_key()

    pe_ratio, pe_source = 22.40, "FALLBACK_BENCHMARK"
    pb_ratio, pb_source = 3.85, "FALLBACK_BENCHMARK"
    div_yield, div_source = 1.25, "FALLBACK_BENCHMARK"
    india_10y_yield, yield_source = 7.02, "FALLBACK_BENCHMARK"

    if fmp_api_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/%5ENSEI?apikey={fmp_api_key}"
            res = requests.get(url, timeout=_HTTP_TIMEOUT).json()
            if isinstance(res, list) and len(res) > 0:
                pe_val = res[0].get("pe")
                if pe_val and float(pe_val) > 0.0:
                    pe_ratio, pe_source = round(float(pe_val), 2), "LIVE_FMP"
        except Exception as exc:
            logger.warning("FMP Nifty quote API failed: %s. Using benchmark proxy.", exc)

        try:
            ratios_url = (
                f"https://financialmodelingprep.com/api/v3/ratios-ttm/%5ENSEI?apikey={fmp_api_key}"
            )
            res = requests.get(ratios_url, timeout=_HTTP_TIMEOUT).json()
            if isinstance(res, list) and len(res) > 0:
                pb_val = res[0].get("priceToBookRatioTTM")
                dy_val = res[0].get("dividendYieldTTM")
                if pb_val and float(pb_val) > 0.0:
                    pb_ratio, pb_source = round(float(pb_val), 2), "LIVE_FMP"
                if dy_val is not None:
                    div_yield, div_source = round(float(dy_val) * 100.0, 2), "LIVE_FMP"
        except Exception as exc:
            logger.warning("FMP ratios-ttm API failed: %s. Using benchmark proxy for P/B and yield.", exc)

    fred_api_key = _get_fred_api_key()
    if fred_api_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_api_key)
            in_series = fred.get_series("INDIRLTLT01STM")
            if not in_series.empty:
                india_10y_yield, yield_source = round(float(in_series.dropna().iloc[-1]), 2), "LIVE_FRED"
        except Exception as exc:
            logger.warning("FRED India 10Y fetch failed: %s", exc)

    earnings_yield = round((1.0 / pe_ratio) * 100.0, 2)
    erp_percent = round(earnings_yield - india_10y_yield, 2)

    # Phase 4 (accuracy roadmap, P1): this used to be 4 hardcoded thresholds
    # (>23->82, >21->68, <18->25, else 50) despite the prompt and README both calling
    # it a "10-Year Historical Percentile" — a spec/implementation mismatch flagged in
    # the audit (Section 3.2). Fixed to a real rolling percentile against actual
    # historical Nifty P/E, computed by _compute_pe_percentile. No free API endpoint
    # for historical INDEX-level P/E was found/validated in this environment (FMP's
    # ratios endpoints are per-company; the FMP_API_KEY configured here also currently
    # returns 401 Invalid API KEY on every endpoint tested, live index OHLCV included —
    # a separate, pre-existing key issue worth checking independently of this fix).
    # Rather than keep a fabricated-looking percentile, this now honestly reports
    # insufficient_data when no real historical P/E series is available — matching the
    # same policy already used for bank credit growth in get_liquidity_and_credit_cycle.
    pe_percentile = _compute_pe_percentile(pe_ratio, _fetch_historical_pe_series())

    erp_score = float(max(-1.0, min(1.0, erp_percent / 2.0)))

    result = {
        "cycle_phase": None,  # populated by get_market_cycle_synthesis, not this tool
        "index_valuations": {
            "nifty_pe_ratio": pe_ratio,
            "nifty_pe_data_source": pe_source,
            "nifty_pb_ratio": pb_ratio,
            "nifty_pb_data_source": pb_source,
            "dividend_yield_percent": div_yield,
            "dividend_yield_data_source": div_source,
            "pe_10y_historical_percentile": pe_percentile,
        },
        "equity_risk_premium": {
            "index_earnings_yield_percent": earnings_yield,
            "india_10y_gsec_yield_percent": india_10y_yield,
            "india_10y_gsec_data_source": yield_source,
            "equity_risk_premium_spread_percent": erp_percent,
            "erp_score": round(erp_score, 3),
            "valuation_attractiveness": (
                "ATTRACTIVE" if erp_percent >= 0.5 else "EXPENSIVE_ELEVATED_RISK"
            ),
        },
    }

    return json.dumps(result, indent=2)


@tool
def get_liquidity_and_credit_cycle() -> str:
    """
    Fetch systemic money supply growth and central bank liquidity stance.

    Bank credit growth YoY % has no clean free India data source (RBI fortnightly
    releases are not machine-readable via a stable free API); it is reported as
    'insufficient_data' rather than a fabricated number, matching the same policy
    used for the microcap-forensics governance rules elsewhere in this codebase.

    Returns:
        JSON string containing M3 money supply growth YoY %, credit cycle status
        (or insufficient_data), and RBI liquidity stance.
    """
    fred_api_key = _get_fred_api_key()
    m3_growth_yoy: Optional[float] = None
    m3_source = "UNAVAILABLE"

    if fred_api_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_api_key)
            m3_series = fred.get_series("MYAGM3INM189N")
            clean_m3 = m3_series.dropna()
            if len(clean_m3) >= 13:
                latest = float(clean_m3.iloc[-1])
                prev = float(clean_m3.iloc[-13])
                m3_growth_yoy = round(((latest - prev) / prev) * 100.0, 2)
                m3_source = "LIVE_FRED"
        except Exception as exc:
            logger.warning("FRED M3 series failed: %s", exc)

    if m3_growth_yoy is not None:
        liquidity_score = float(max(-1.0, min(1.0, (m3_growth_yoy - 9.0) / 6.0)))
        credit_cycle_status = "EXPANSIONARY_LIQUIDITY" if m3_growth_yoy > 9.0 else "TIGHT_LIQUIDITY"
    else:
        liquidity_score = 0.0
        credit_cycle_status = "insufficient_data"

    result = {
        "bank_credit_growth_yoy_percent": "insufficient_data",
        "bank_credit_growth_note": "No stable free India bank-credit-growth API is wired; needs a build-vs-buy decision.",
        "m3_money_supply_growth_yoy_percent": m3_growth_yoy if m3_growth_yoy is not None else "insufficient_data",
        "m3_data_source": m3_source,
        "credit_cycle_status": credit_cycle_status,
        "liquidity_score": round(liquidity_score, 3),
        "rbi_monetary_policy_stance": "WITHDRAWAL_OF_ACCOMMODATION",
        "rbi_stance_data_source": "FALLBACK_BENCHMARK",
    }

    return json.dumps(result, indent=2)


_EARNINGS_BASKET = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
]


@tool
def get_index_earnings_momentum() -> str:
    """
    Fetch aggregate YoY quarterly earnings growth across a representative large-cap
    basket, as a proxy for index-level earnings momentum (the 4th cycle pillar
    alongside valuation, momentum, and liquidity).

    No index-level earnings-revision API is available for free; this uses yfinance's
    per-stock quarterly earnings growth averaged across a fixed basket of 15 large-cap
    names spanning financials, IT, energy, autos, and consumer — no API key required.

    Returns:
        JSON string with mean YoY earnings growth %, basket coverage, and a score.
        Reports insufficient_data if fewer than 5 of the 15 basket names return data.
    """
    growth_rates: List[float] = []

    for ticker_symbol in _EARNINGS_BASKET:
        try:
            info = yf.Ticker(ticker_symbol).get_info()
            growth = info.get("earningsQuarterlyGrowth")
            if growth is not None and isinstance(growth, (int, float)):
                growth_rates.append(float(growth))
        except Exception as exc:
            logger.warning("yfinance earnings growth fetch failed for %s: %s", ticker_symbol, exc)

    coverage = len(growth_rates)
    if coverage < 5:
        return json.dumps(
            {
                "earnings_momentum_yoy_percent": "insufficient_data",
                "basket_coverage": f"{coverage}/{len(_EARNINGS_BASKET)}",
                "note": "Fewer than 5 of 15 basket names returned earnings data.",
            },
            indent=2,
        )

    mean_growth_pct = round(float(np.mean(growth_rates)) * 100.0, 2)
    earnings_score = float(max(-1.0, min(1.0, mean_growth_pct / 20.0)))

    result = {
        "earnings_momentum_yoy_percent": mean_growth_pct,
        "basket_coverage": f"{coverage}/{len(_EARNINGS_BASKET)}",
        "earnings_score": round(earnings_score, 3),
        "data_source": "LIVE_YFINANCE_BASKET_PROXY",
        "note": "Equal-weighted average of quarterly YoY earnings growth across a 15-name large-cap basket, not a true float-weighted index figure.",
    }
    return json.dumps(result, indent=2)


@tool
def get_market_cycle_synthesis() -> str:
    """
    Combine Core price-structure signals (ZigZag, STC, trend context), valuation (ERP),
    liquidity (M3 growth), and earnings momentum into a single weighted Market Cycle
    classification — the four pillars (valuation, momentum, liquidity, earnings)
    required by the project brief.

    This is the primary tool for answering "what cycle phase is the market in" — it
    performs the deterministic composite scoring itself so the classification is
    reproducible and backtestable, rather than left to free-form LLM judgment.

    Returns:
        JSON string with cycle_phase, cycle_confidence, transition_risk,
        composite_score, and a structured evidence array citing every signal used.
    """
    price_structure_raw = get_market_cycle_metrics.invoke({})
    liquidity_raw = get_liquidity_and_credit_cycle.invoke({})
    earnings_raw = get_index_earnings_momentum.invoke({})
    valuation = json.loads(price_structure_raw)
    liquidity = json.loads(liquidity_raw)
    earnings = json.loads(earnings_raw)

    df = _fetch_index_ohlcv()
    evidence: List[EvidenceItem] = []

    if df is not None:
        zz = compute_zigzag(df)
        stc = compute_stc(df["close"])
        trend = compute_trend_context(df["close"])
        evidence.extend([
            EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
            EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
            EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}% vs 200DMA", trend.score, trend.note),
        ])
        as_of_date = str(df["date"].iloc[-1].date())
    else:
        as_of_date = "unavailable"
        logger.warning("Index OHLCV unavailable; synthesis proceeding on Supporting tier only (low confidence).")

    erp = valuation["equity_risk_premium"]
    evidence.append(EvidenceItem(
        "erp", "supporting",
        f"{erp['equity_risk_premium_spread_percent']}% ({erp['valuation_attractiveness']})",
        erp["erp_score"],
        "Earnings Yield minus India 10Y G-Sec Yield.",
    ))

    if liquidity["credit_cycle_status"] != "insufficient_data":
        evidence.append(EvidenceItem(
            "liquidity", "supporting",
            f"M3 {liquidity['m3_money_supply_growth_yoy_percent']}% YoY ({liquidity['credit_cycle_status']})",
            liquidity["liquidity_score"],
            "M3 money supply growth YoY vs 9% expansionary threshold.",
        ))

    if earnings.get("earnings_momentum_yoy_percent") != "insufficient_data":
        evidence.append(EvidenceItem(
            "earnings_momentum", "supporting",
            f"{earnings['earnings_momentum_yoy_percent']}% YoY ({earnings['basket_coverage']} basket coverage)",
            earnings["earnings_score"],
            "Equal-weighted average YoY quarterly earnings growth across a 15-name large-cap basket proxy.",
        ))

    if not evidence:
        return json.dumps({"error": "No signals available to classify cycle.", "status": "unavailable"}, indent=2)

    # Phase 2 (accuracy roadmap): the reported cycle_phase is hysteresis-confirmed using
    # state persisted across calls (data/cycle_phase_state.json), not a raw per-call read
    # — this is what actually fixes the whipsaw problem for the live agent, not just the
    # backtest. See _classify_with_persisted_hysteresis and cycle_signals.classify_cycle_stateful.
    classification = _classify_with_persisted_hysteresis(evidence, as_of_date)

    result = {
        "as_of_date": as_of_date,
        "market": "NIFTY50",
        "cycle_phase": classification.cycle_phase,
        "cycle_confidence": classification.cycle_confidence,
        "transition_risk": classification.transition_risk,
        "transition_watch": classification.transition_watch,
        "composite_score": classification.composite_score,
        "pending_phase": classification.pending_phase,
        "dwell_progress": classification.dwell_progress,
        "evidence": [
            {"signal": e.signal, "tier": e.tier, "reading": e.reading, "score": e.score, "note": e.note}
            for e in classification.evidence
        ],
        "data_freshness": {
            "price_data": as_of_date,
            "valuation": erp.get("india_10y_gsec_data_source", "unknown"),
            "liquidity": liquidity.get("m3_data_source", "unknown"),
            "earnings": earnings.get("data_source", "insufficient_data"),
        },
    }
    return json.dumps(result, indent=2)
