# from dotenv import load_dotenv
# from fredapi import Fred
# import os
# load_dotenv()
# # Initialize with your free FRED API key
# fred = Fred(api_key=os.environ["FRED_API_KEY"])

# # 1. Fetch India 10-Year Government Bond Yields
# india_10y_yield = fred.get_series('INDIRLTLT01STM')
# print("Latest India 10Y Yield:", india_10y_yield.iloc[-1])

# # 2. Fetch India CPI Index and compute YoY Inflation %
# india_cpi = fred.get_series('INDCPIALLMINMEI')
# cpi_latest = india_cpi.iloc[-1]
# cpi_1yr_ago = india_cpi.iloc[-13] # 12 months prior
# yoy_cpi_inflation = ((cpi_latest - cpi_1yr_ago) / cpi_1yr_ago) * 100

# print(f"Latest India CPI Index: {cpi_latest}")
# print(f"Calculated YoY CPI Inflation: {yoy_cpi_inflation:.2f}%")
"""
Live Integration Tests for Macro Tools (No Mocks).

Execute with:
    pytest -s tests/test_macro_tools.py
"""

from dotenv import load_dotenv
import json
import os
import pytest

from shankh.agents.macro.tools import (
    get_economic_calendar,
    get_fii_dii_flows,
    get_forex_and_commodities,
    get_indian_macro_indicators,
)
load_dotenv

def test_get_fii_dii_flows_live():
    """Live test for FII / DII Institutional Flows from NSE feed."""
    raw_response = get_fii_dii_flows.invoke({})
    print("\n[LIVE RESPONSE - FII/DII]:\n", raw_response)

    data = json.loads(raw_response)

    # Ensure no network failure / exception wrapper returned
    assert "error" not in data, f"FII/DII API returned error: {data.get('error')}"

    # Structural Assertions
    assert "fii_net_cash_cr" in data
    assert "dii_net_cash_cr" in data
    assert "institutional_net_combined_cr" in data
    assert "market_bias" in data
    assert data["market_bias"] in ["BULLISH", "BEARISH"]

    # Type & Value Assertions
    assert isinstance(data["fii_net_cash_cr"], (int, float))
    assert isinstance(data["dii_net_cash_cr"], (int, float))


def test_get_forex_and_commodities_live():
    """Live test for USDINR, Crude Oil, and Gold (FMP or yfinance fallback)."""
    raw_response = get_forex_and_commodities.invoke({})
    print("\n[LIVE RESPONSE - Forex & Commodities]:\n", raw_response)

    data = json.loads(raw_response)

    # Ensure no network failure
    assert "error" not in data, f"Forex/Commodity API returned error: {data.get('error')}"

    # Structural Assertions
    assert "USDINR" in data
    assert "Brent_Crude_USD" in data
    assert "WTI_Crude_USD" in data
    assert "Gold_USD" in data
    assert "provider" in data

    # Sanity Range Assertions for Live Market Data
    assert 60.0 < float(data["USDINR"]) < 120.0, f"Unrealistic USDINR: {data['USDINR']}"
    assert float(data["Brent_Crude_USD"]) > 10.0, "Brent Crude price abnormal"
    assert float(data["WTI_Crude_USD"]) > 10.0, "WTI Crude price abnormal"
    assert float(data["Gold_USD"]) > 500.0, "Gold price abnormal"


@pytest.mark.skipif(
    not os.getenv("FRED_API_KEY"),
    reason="Skipping FRED live API test because FRED_API_KEY is not set in environment."
)
def test_get_indian_macro_indicators_live():
    """Live test for FRED API (RBI Repo Rate, Indian CPI, 10Y Yield)."""
    raw_response = get_indian_macro_indicators.invoke({})
    print("\n[LIVE RESPONSE - Indian Macro Indicators]:\n", raw_response)

    data = json.loads(raw_response)

    assert "rbi_repo_rate_percent" in data
    assert "india_10y_gsec_yield" in data

    # RBI Repo Rate Sanity
    assert isinstance(data["rbi_repo_rate_percent"], (int, float))
    assert 2.0 <= data["rbi_repo_rate_percent"] <= 12.0

    # Yield Sanity
    yield_val = data["india_10y_gsec_yield"]
    if yield_val != "N/A":
        assert 3.0 <= float(yield_val) <= 15.0, f"Unrealistic 10Y Yield: {yield_val}"


@pytest.mark.skipif(
    not os.getenv("FMP_API_KEY"),
    reason="Skipping FMP live API test because FMP_API_KEY is not set in environment."
)
def test_get_economic_calendar_live():
    """Live test for Financial Modeling Prep (FMP) Economic Calendar."""
    raw_response = get_economic_calendar.invoke({"country_code":"IN"})
    print("\n[LIVE RESPONSE - Economic Calendar]:\n", raw_response)

    data = json.loads(raw_response)

    # Ensure no API key error or network error
    assert "error" not in data, f"FMP Economic Calendar error: {data.get('error')}"

    # Structural Assertions
    assert "events_count" in data
    assert "events" in data
    assert isinstance(data["events"], list)

    # If events are returned, check the first event structure
    if data["events_count"] > 0:
        event = data["events"][0]
        assert "event" in event
        assert "date" in event
        assert "country" in event