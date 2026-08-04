"""
src/shankh/ml/macro/tool.py

LangChain Tool for live Stock Clustering + Forensic Anomaly inference.

Flow
----
1.  Accept tickers from the agent (required).
2.  Pad with a minimal set of liquid NSE anchors only if needed to satisfy
    the model's minimum sample count — anchors are never shown in output.
3.  Fetch OHLCV from yfinance for all tickers.
4.  Build per-ticker factor features (return, volatility, momentum, trend, beta).
5.  Load trained KMeans scaler + model and IsolationForest model from disk.
6.  Run inference on the padded set.
7.  Return results filtered to only the requested tickers.
"""

import json
import logging
import os
import requests
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from shankh.agents.macro.config import CONFIG
from shankh.agents.macro.inference import run_inference

logger = logging.getLogger(__name__)

_HISTORY_DAYS = 120
# Request timeout for production network calls (in seconds)
_HTTP_TIMEOUT = 10.0
# Minimal liquid anchors used ONLY as silent padding when the agent provides
# fewer tickers than the model minimum. Sourced from the trained universe.
_PADDING_ANCHORS = [
    "HDFCBANK.NS", "INFY.NS", "AXISBANK.NS", "HINDUNILVR.NS", "DRREDDY.NS",
    "GAIL.NS", "ITC.NS", "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS",
    "AUROPHARMA.NS", "HINDALCO.NS", "EICHERMOT.NS", "FEDERALBNK.NS", "IOC.NS",
]


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith(".NS") else t + ".NS"


def _fetch_ohlcv(tickers: list[str], days: int = _HISTORY_DAYS) -> pd.DataFrame:
    frames = []
    for tkr in tickers:
        try:
            raw = yf.download(tkr, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)
            if raw.empty:
                logger.warning("No data for %s — skipping.", tkr)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = tkr
            df = df.dropna(subset=["open", "high", "low", "close"])
            df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) >= 60:
                frames.append(df)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", tkr, exc)

    if not frames:
        raise RuntimeError("Could not fetch OHLCV data for any of the requested tickers.")

    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@tool
def get_stock_clusters(tickers: str) -> str:
    """
    Run live stock clustering and forensic anomaly detection for any NSE stock(s).

    Fetches recent OHLCV data from yfinance for the specified tickers, computes
    per-ticker factor features (annualised return, volatility, Sharpe, beta,
    RSI, ATR, max drawdown, trend distance), loads the trained KMeans and
    IsolationForest models, and returns cluster assignments and forensic flags.

    Args:
        tickers: Comma-separated NSE tickers, e.g. 'TCS.NS' or
                 'INFY,TCS,RELIANCE,HDFCBANK'. The .NS suffix is optional.

    Returns:
        JSON with:
          - kmeans_factor_clusters: {ticker: cluster_id}
          - flagged_forensic_anomalies: [tickers with unusual factor profiles]
          - note: padding info if anchors were added to satisfy model minimum
    """
    if not tickers or not tickers.strip():
        return json.dumps({"error": "tickers is required. Provide at least one NSE ticker, e.g. 'TCS.NS'."}, indent=2)

    requested = [_normalize(t) for t in tickers.split(",") if t.strip()]
    if not requested:
        return json.dumps({"error": "No valid tickers parsed. Use comma-separated NSE symbols."}, indent=2)

    min_required = CONFIG["data"]["min_tickers"]

    # Pad silently with anchors if below model minimum — exclude already requested
    padding = [t for t in _PADDING_ANCHORS if t not in requested]
    shortfall = max(0, min_required - len(requested))
    fetch_list = requested + padding[:shortfall]
    padded = shortfall > 0

    try:
        df_raw = _fetch_ohlcv(fetch_list, days=_HISTORY_DAYS)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    n_fetched = df_raw["ticker"].nunique()
    if n_fetched < min_required:
        return json.dumps({
            "error": (
                f"Only {n_fetched} tickers returned sufficient data after padding "
                f"(need {min_required}). Try providing more tickers."
            )
        }, indent=2)

    try:
        result = run_inference(df_raw)
    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "action_required": "Train the clustering model first: uv run src/shankh/ml/macro/train_cluster.py",
        }, indent=2)
    except Exception as exc:
        logger.exception("Stock clustering inference failed")
        return json.dumps({"error": f"Inference failed: {exc}"}, indent=2)

    # Filter to requested tickers and enrich with peer context.
    # Peers are derived entirely from the live inference — no file lookup,
    # works identically in dev and prod.
    all_clusters: dict = result.get("kmeans_factor_clusters", {})
    all_anomalies: list = result.get("flagged_forensic_anomalies", [])

    requested_clusters = {t: v for t, v in all_clusters.items() if t in requested}
    requested_anomalies = [t for t in all_anomalies if t in requested]

    # Group requested tickers by cluster so each ticker can see its peers
    cluster_to_members: dict[int, list[str]] = {}
    for t, cid in requested_clusters.items():
        cluster_to_members.setdefault(cid, []).append(t)

    ticker_analysis = {}
    for t, cid in requested_clusters.items():
        peers = [p for p in cluster_to_members[cid] if p != t]
        ticker_analysis[t] = {
            "cluster_id": cid,
            "peers_in_same_cluster": peers,
            "is_forensic_anomaly": t in requested_anomalies,
        }

    if padded:
        logger.info(
            "Clustering padded: %d anchor(s) added for model minimum of %d. "
            "Results filtered to: %s", shortfall, min_required, requested
        )

    return json.dumps({
        "ticker_analysis": ticker_analysis,
        "flagged_forensic_anomalies": requested_anomalies,
    }, indent=2, default=str)








# ---------------------------------------------------------------------------
# Tool 1: FII / DII Institutional Flows (India Specific)
# ---------------------------------------------------------------------------
@tool
def get_fii_dii_flows() -> str:
    """
    Fetch the latest Foreign Institutional Investors (FII / FPI) and
    Domestic Institutional Investors (DII) daily trading activity in India (in ₹ Crore).

    Returns:
        JSON string containing date, FII net buy/sell, DII net buy/sell, and market sentiment.
    """
    # Endpoint backed by NSE daily participant data
    url = "https://fii-dii.mrchartist.workers.dev/api/data"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        fii_net = data.get("fii_net", 0)
        dii_net = data.get("dii_net", 0)
        total_net = fii_net + dii_net

        result = {
            "date": data.get("date"),
            "fii_net_cash_cr": fii_net,
            "dii_net_cash_cr": dii_net,
            "institutional_net_combined_cr": total_net,
            "market_bias": "BULLISH" if total_net > 0 else "BEARISH",
            "source": "NSE Public Feed",
        }
        return json.dumps(result, indent=2)

    except Exception as exc:
        logger.error("Failed to fetch FII/DII data: %s", exc)
        return json.dumps(
            {
                "error": f"Failed to retrieve FII/DII flows: {exc}",
                "status": "unavailable",
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tool 2: Indian Macro Indicators (Rates, CPI Inflation, 10Y Yield)
# ---------------------------------------------------------------------------
@tool
def get_indian_macro_indicators() -> str:
    """
    Fetch key Indian macroeconomic indicators: RBI Repo Rate, CPI Inflation Rate (YoY),
    and India 10-Year Government Securities (G-Sec) Benchmark Yield.

    Returns:
        JSON string containing RBI Repo Rate, CPI Inflation %, and 10Y Yield.
    """
    fred_api_key = os.getenv("FRED_API_KEY")
    result = {}

    if fred_api_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_api_key)

            # India CPI Index (INDCPIALLMINMEI)
            cpi_series = fred.get_series("INDCPIALLMINMEI")
            if not cpi_series.empty:
                latest_cpi = cpi_series.iloc[-1]
                prev_year_cpi = (
                    cpi_series.iloc[-13] if len(cpi_series) >= 13 else None
                )
                cpi_yoy = (
                    round(((latest_cpi - prev_year_cpi) / prev_year_cpi) * 100, 2)
                    if prev_year_cpi
                    else None
                )
                result["india_cpi_index"] = latest_cpi
                result["india_cpi_yoy_percent"] = cpi_yoy

            # India 10-Year Long-Term Government Bond Yield
            yield_series = fred.get_series("INDIRLTLT01STM")
            if not yield_series.empty:
                result["india_10y_gsec_yield"] = round(yield_series.iloc[-1], 2)

        except Exception as exc:
            logger.warning("FRED API fetch failed: %s", exc)

    # Fallback / Market-implied indicators if FRED is unavailable or incomplete
    if "india_10y_gsec_yield" not in result:
        try:
            yield_ticker = yf.Ticker("IN10Y=X").history(period="1d")
            if not yield_ticker.empty:
                result["india_10y_gsec_yield"] = round(
                    yield_ticker["Close"].iloc[-1], 2
                )
        except Exception:
            result["india_10y_gsec_yield"] = "N/A"

    # Current RBI Policy Rates (Updated as of latest MPC decision)
    result["rbi_repo_rate_percent"] = 6.50
    result["rbi_reverse_repo_rate_percent"] = 3.35
    result["rbi_standing_deposit_facility_percent"] = 6.25

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 3: Forex & Commodities (USD/INR, Brent, WTI, Gold) via FMP
# ---------------------------------------------------------------------------
@tool
def get_forex_and_commodities() -> str:
    """
    Fetch live market prices for USD/INR exchange rate, Brent Crude Oil, WTI Crude Oil,
    and Gold prices, which strongly impact Indian equities and inflation.

    Returns:
        JSON string with USDINR spot rate, Brent Crude ($/bbl), WTI Crude ($/bbl), and Gold ($/oz).
    """
    fmp_api_key = os.getenv("FMP_API_KEY")

    # Primary: Financial Modeling Prep (FMP)
    if fmp_api_key:
        try:
            # Query FX and Commodities quotes from FMP
            fx_url = f"https://financialmodelingprep.com/api/v3/quote/USDINR?apikey={fmp_api_key}"
            comm_url = f"https://financialmodelingprep.com/api/v3/quote/CLUSD,BZUSD,XAUUSD?apikey={fmp_api_key}"

            fx_res = requests.get(fx_url, timeout=_HTTP_TIMEOUT).json()
            comm_res = requests.get(comm_url, timeout=_HTTP_TIMEOUT).json()

            quotes = {}
            if fx_res and isinstance(fx_res, list):
                quotes["USDINR"] = fx_res[0].get("price")

            if comm_res and isinstance(comm_res, list):
                for item in comm_res:
                    symbol = item.get("symbol")
                    price = item.get("price")
                    if symbol == "CLUSD":
                        quotes["WTI_Crude_USD"] = price
                    elif symbol == "BZUSD":
                        quotes["Brent_Crude_USD"] = price
                    elif symbol == "XAUUSD":
                        quotes["Gold_USD"] = price

            if quotes:
                quotes["provider"] = "Financial Modeling Prep"
                return json.dumps(quotes, indent=2)

        except Exception as exc:
            logger.warning("FMP API fetch failed: %s. Falling back to yfinance.", exc)

    # Fallback: yfinance (Reliable & free backup)
    try:
        usdinr = yf.Ticker("USDINR=X").history(period="1d")["Close"].iloc[-1]
        brent = yf.Ticker("BZ=F").history(period="1d")["Close"].iloc[-1]
        wti = yf.Ticker("CL=F").history(period="1d")["Close"].iloc[-1]
        gold = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]

        return json.dumps(
            {
                "USDINR": round(float(usdinr), 2),
                "Brent_Crude_USD": round(float(brent), 2),
                "WTI_Crude_USD": round(float(wti), 2),
                "Gold_USD": round(float(gold), 2),
                "provider": "yfinance (fallback)",
            },
            indent=2,
        )

    except Exception as exc:
        logger.error("Failed to fetch Forex/Commodity data: %s", exc)
        return json.dumps({"error": f"Failed to retrieve data: {exc}"}, indent=2)


# ---------------------------------------------------------------------------
# Tool 4: Economic Calendar (FMP API)
# ---------------------------------------------------------------------------
@tool
def get_economic_calendar(country_code: str = "IN") -> str:
    """
    Fetch upcoming economic calendar events (e.g., Inflation releases, GDP figures,
    Interest Rate Decisions, Trade Balance) using Financial Modeling Prep (FMP).

    Args:
        country_code: Two-letter ISO country code, e.g. 'IN' for India, 'US' for USA.

    Returns:
        JSON string listing upcoming/recent economic events and market expectations.
    """
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        return json.dumps(
            {
                "error": "FMP_API_KEY environment variable is not set.",
                "status": "unconfigured",
            },
            indent=2,
        )

    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={fmp_api_key}"

    try:
        response = requests.get(url, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        events = response.json()

        # Filter events for specified country (default 'IN')
        filtered_events = [
            {
                "event": e.get("event"),
                "date": e.get("date"),
                "country": e.get("country"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "previous": e.get("previous"),
                "impact": e.get("impact"),
            }
            for e in events
            if e.get("country") == country_code.upper()
        ]

        # If no India-specific events found in immediate window, return top global events
        if not filtered_events and country_code.upper() == "IN":
            filtered_events = [
                {
                    "event": e.get("event"),
                    "date": e.get("date"),
                    "country": e.get("country"),
                    "actual": e.get("actual"),
                    "estimate": e.get("estimate"),
                    "impact": e.get("impact"),
                }
                for e in events[:10]  # Top global events
            ]

        return json.dumps(
            {
                "country_code": country_code.upper(),
                "events_count": len(filtered_events),
                "events": filtered_events[:15],  # Limit payload size for LLM context
            },
            indent=2,
        )

    except Exception as exc:
        logger.error("Failed to fetch economic calendar from FMP: %s", exc)
        return json.dumps(
            {"error": f"Failed to retrieve economic calendar: {exc}"}, indent=2
        )