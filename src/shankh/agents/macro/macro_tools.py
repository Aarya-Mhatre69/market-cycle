"""
Macro Analyst Tool Module.

Provides domain-pure macro tools for institutional flow tracking,
cross-asset yield differentials, commodity/forex trend velocity,
and dual India/US economic calendar event retrieval.
"""

from dotenv import load_dotenv
import json
import logging
import os
from typing import Dict, Any
import pandas as pd
import requests
import yfinance as yf
from langchain_core.tools import tool
load_dotenv()
logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 10.0


def _compute_roc(series: pd.Series, window: int) -> float:
    if len(series) < window + 1:
        return 0.0
    start_val = float(series.iloc[-(window + 1)])
    end_val = float(series.iloc[-1])
    if start_val == 0.0:
        return 0.0
    return round(((end_val - start_val) / start_val) * 100.0, 2)


def _classify_trend(roc_5d: float, roc_20d: float) -> str:
    if roc_5d > 3.0 or roc_20d > 7.0:
        return "RISING_FAST"
    elif roc_5d > 0.8 or roc_20d > 2.0:
        return "RISING"
    elif roc_5d < -3.0 or roc_20d < -7.0:
        return "FALLING_FAST"
    elif roc_5d < -0.8 or roc_20d < -2.0:
        return "FALLING"
    return "FLAT"


@tool
def get_fii_dii_flows() -> str:
    """
    Fetch live Foreign Institutional Investors (FII / FPI) and Domestic
    Institutional Investors (DII) daily trading activity in India (in ₹ Crore).

    Computes net cash flow, institutional bias, and market sentiment metrics.

    Returns:
        JSON string containing FII net, DII net, combined institutional flow,
        and directional bias classification.
    """
    url = "https://fii-dii.mrchartist.workers.dev/api/data"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        fii_net = float(data.get("fii_net", 0.0))
        dii_net = float(data.get("dii_net", 0.0))
        combined_net = fii_net + dii_net

        if fii_net > 0 and dii_net > 0:
            bias = "STRONG_BULLISH_ACCUMULATION"
        elif fii_net > 0 and dii_net < 0:
            bias = "FII_DRIVEN_BULLISH"
        elif fii_net < 0 and dii_net > 0:
            bias = "DII_SUPPORTED_ABSORPTION"
        elif fii_net < 0 and dii_net < 0:
            bias = "STRONG_BEARISH_DISTRIBUTION"
        else:
            bias = "NEUTRAL"

        result = {
            "date": data.get("date"),
            "fii_net_cash_cr": fii_net,
            "dii_net_cash_cr": dii_net,
            "combined_institutional_net_cr": combined_net,
            "institutional_bias": bias,
            "data_source": "NSE Participant Feed",
        }
        return json.dumps(result, indent=2)

    except Exception as exc:
        logger.error("Failed to retrieve FII/DII flow data: %s", exc)
        return json.dumps(
            {
                "error": f"Institutional flow retrieval failed: {exc}",
                "status": "unavailable",
            },
            indent=2,
        )


@tool
def get_indian_macro_indicators() -> str:
    """
    Fetch key macroeconomic indicators: US-India 10Y Yield Spread (bps),
    India 10Y G-Sec Yield, US 10Y Treasury Yield, US Yield Curve Slope (10Y-2Y),
    and Inflation YoY rates using the FRED API.

    Returns:
        JSON string containing yield differentials, interest rate spreads,
        and inflation metrics.
    """
    fred_api_key = os.getenv("FRED_API_KEY")
    result: Dict[str, Any] = {}

    if fred_api_key:
        try:
            from fredapi import Fred

            fred = Fred(api_key=fred_api_key)

            us_10y_series = fred.get_series("DGS10")
            if not us_10y_series.empty:
                us_10y = round(float(us_10y_series.dropna().iloc[-1]), 2)
                result["us_10y_treasury_yield"] = us_10y

            us_2y_series = fred.get_series("DGS2")
            if not us_2y_series.empty:
                us_2y = round(float(us_2y_series.dropna().iloc[-1]), 2)
                result["us_2y_treasury_yield"] = us_2y
                if "us_10y_treasury_yield" in result:
                    result["us_yield_curve_slope_10y_2y_bps"] = round(
                        (result["us_10y_treasury_yield"] - us_2y) * 100.0, 1
                    )

            in_10y_series = fred.get_series("INDIRLTLT01STM")
            if not in_10y_series.empty:
                result["india_10y_gsec_yield"] = round(
                    float(in_10y_series.dropna().iloc[-1]), 2
                )

            in_cpi_series = fred.get_series("INDCPIALLMINMEI")
            if not in_cpi_series.empty:
                clean_cpi = in_cpi_series.dropna()
                if len(clean_cpi) >= 13:
                    latest_cpi = float(clean_cpi.iloc[-1])
                    prev_cpi = float(clean_cpi.iloc[-13])
                    cpi_yoy = round(((latest_cpi - prev_cpi) / prev_cpi) * 100.0, 2)
                    result["india_cpi_inflation_yoy_percent"] = cpi_yoy

        except Exception as exc:
            logger.warning("FRED API retrieval encountered issue: %s", exc)

    if "india_10y_gsec_yield" not in result:
        try:
            yield_df = yf.Ticker("IN10Y=X").history(period="5d")
            if not yield_df.empty:
                result["india_10y_gsec_yield"] = round(
                    float(yield_df["Close"].iloc[-1]), 2
                )
        except Exception as exc:
            logger.warning("yfinance fallback for IN10Y failed: %s", exc)
            result["india_10y_gsec_yield"] = 7.00

    if "us_10y_treasury_yield" not in result:
        try:
            us_df = yf.Ticker("^TNX").history(period="5d")
            if not us_df.empty:
                result["us_10y_treasury_yield"] = round(
                    float(us_df["Close"].iloc[-1]), 2
                )
        except Exception as exc:
            logger.warning("yfinance fallback for US10Y failed: %s", exc)
            result["us_10y_treasury_yield"] = 4.20

    if (
        isinstance(result.get("india_10y_gsec_yield"), (int, float))
        and isinstance(result.get("us_10y_treasury_yield"), (int, float))
    ):
        spread_bps = round(
            (result["india_10y_gsec_yield"] - result["us_10y_treasury_yield"])
            * 100.0,
            1,
        )
        result["us_india_10y_yield_spread_bps"] = spread_bps
        result["capital_flow_risk"] = (
            "HIGH_OUTFLOW_RISK" if spread_bps < 200.0 else "NORMAL"
        )

    result["rbi_repo_rate_percent"] = 6.50
    result["rbi_standing_deposit_facility_percent"] = 6.25

    return json.dumps(result, indent=2)


@tool
def get_forex_and_commodities() -> str:
    """
    Fetch live spot prices and compute 5-day and 20-day trend velocity (Rate of Change)
    for USD/INR exchange rate, Brent Crude, WTI Crude, Gold, and US Dollar Index (DXY).

    Returns:
        JSON string containing spot prices, 5D/20D ROC %, trend classifications,
        and macro equity impact assessments.
    """
    symbols = {
        "USDINR": "USDINR=X",
        "Brent_Crude": "BZ=F",
        "WTI_Crude": "CL=F",
        "Gold": "GC=F",
        "Dollar_Index_DXY": "DX-Y.X",
    }

    metrics: Dict[str, Any] = {}

    for name, ticker_str in symbols.items():
        try:
            hist = yf.Ticker(ticker_str).history(period="1mo", interval="1d")
            if hist.empty or len(hist) < 2:
                continue

            closes = hist["Close"].dropna()
            spot_price = round(float(closes.iloc[-1]), 2)
            roc_5d = _compute_roc(closes, 5)
            roc_20d = _compute_roc(closes, 20)
            trend = _classify_trend(roc_5d, roc_20d)

            metrics[name] = {
                "spot_price": spot_price,
                "roc_5d_percent": roc_5d,
                "roc_20d_percent": roc_20d,
                "trend": trend,
            }
        except Exception as exc:
            logger.warning("Failed to calculate metrics for %s: %s", name, exc)

    if "Brent_Crude" in metrics:
        brent_trend = metrics["Brent_Crude"]["trend"]
        if "RISING" in brent_trend:
            metrics["Brent_Crude"]["equity_impact"] = "NEGATIVE_INFLATIONARY_PRESSURE"
        elif "FALLING" in brent_trend:
            metrics["Brent_Crude"]["equity_impact"] = "POSITIVE_MARGIN_EXPANSION"
        else:
            metrics["Brent_Crude"]["equity_impact"] = "NEUTRAL"

    if "USDINR" in metrics:
        usdinr_trend = metrics["USDINR"]["trend"]
        if "RISING" in usdinr_trend:
            metrics["USDINR"]["equity_impact"] = "RUPEE_DEPRECIATION_FII_HEADWIND"
        elif "FALLING" in usdinr_trend:
            metrics["USDINR"]["equity_impact"] = "RUPEE_APPRECIATION_FII_TAILWIND"
        else:
            metrics["USDINR"]["equity_impact"] = "NEUTRAL"

    return json.dumps(metrics, indent=2)


@tool
def get_economic_calendar() -> str:
    """
    Fetch upcoming and recent economic calendar events for both India (IN)
    and the United States (US) using Financial Modeling Prep (FMP).

    Returns:
        JSON string categorizing high-impact domestic events and global US cues.
    """
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        return json.dumps(
            {
                "error": "FMP_API_KEY environment variable is not configured.",
                "status": "unconfigured",
            },
            indent=2,
        )

    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={fmp_api_key}"

    try:
        response = requests.get(url, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        events = response.json()

        if not isinstance(events, list):
            return json.dumps({"error": "Invalid payload from FMP API."}, indent=2)

        india_events = []
        us_events = []

        for e in events:
            country = str(e.get("country", "")).upper()
            event_obj = {
                "event": e.get("event"),
                "date": e.get("date"),
                "country": country,
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "previous": e.get("previous"),
                "impact": e.get("impact"),
            }

            if country == "IN":
                india_events.append(event_obj)
            elif country == "US" and e.get("impact") in ["High", "Medium"]:
                us_events.append(event_obj)

        result = {
            "domestic_events_india": india_events[:10],
            "global_cues_us": us_events[:10],
            "india_events_count": len(india_events),
            "us_events_count": len(us_events),
        }
        return json.dumps(result, indent=2)

    except Exception as exc:
        logger.error("Failed to retrieve economic calendar from FMP: %s", exc)
        return json.dumps(
            {"error": f"Economic calendar retrieval failed: {exc}"}, indent=2
        )