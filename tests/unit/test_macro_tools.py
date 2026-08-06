"""
Unit Test Suite for Macro Analyst Tools.

Validates output payload schemas, key data types, numeric boundary constraints,
and error handling across all domain-pure macro tools.
"""

import json
from shankh.agents.macro.macro_tools import (
    get_fii_dii_flows,
    get_indian_macro_indicators,
    get_forex_and_commodities,
    get_economic_calendar,
)


class TestMacroToolsUnit:
    """Unit tests validating tool payload structure and contract invariants."""

    def test_get_fii_dii_flows_schema_and_types(self):
        """Verify FII/DII tool returns valid JSON with required financial metrics."""
        raw_output = get_fii_dii_flows.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a valid JSON object"
        if "error" in data:
            assert data.get("status") == "unavailable"
        else:
            assert "fii_net_cash_cr" in data
            assert "dii_net_cash_cr" in data
            assert "combined_institutional_net_cr" in data
            assert "institutional_bias" in data
            assert isinstance(data["fii_net_cash_cr"], (int, float))
            assert isinstance(data["dii_net_cash_cr"], (int, float))
            assert data["institutional_bias"] in [
                "STRONG_BULLISH_ACCUMULATION",
                "FII_DRIVEN_BULLISH",
                "DII_SUPPORTED_ABSORPTION",
                "STRONG_BEARISH_DISTRIBUTION",
                "NEUTRAL",
            ]

    def test_get_indian_macro_indicators_yield_spread(self):
        """Verify macro indicators return yield differentials and policy rates."""
        raw_output = get_indian_macro_indicators.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a valid JSON object"
        assert "rbi_repo_rate_percent" in data
        assert data["rbi_repo_rate_percent"] == 6.50
        assert "india_10y_gsec_yield" in data

        if "us_10y_treasury_yield" in data and "india_10y_gsec_yield" in data:
            if isinstance(data["india_10y_gsec_yield"], (int, float)) and isinstance(
                data["us_10y_treasury_yield"], (int, float)
            ):
                assert "us_india_10y_yield_spread_bps" in data
                expected_spread = round(
                    (data["india_10y_gsec_yield"] - data["us_10y_treasury_yield"])
                    * 100.0,
                    1,
                )
                assert abs(data["us_india_10y_yield_spread_bps"] - expected_spread) < 0.2

    def test_get_forex_and_commodities_velocity_metrics(self):
        """Verify forex and commodities tool returns spot prices and 5D/20D ROC %."""
        raw_output = get_forex_and_commodities.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a valid JSON object"
        expected_assets = ["USDINR", "Brent_Crude", "WTI_Crude", "Gold", "Dollar_Index_DXY"]

        for asset in expected_assets:
            if asset in data:
                asset_data = data[asset]
                assert "spot_price" in asset_data
                assert "roc_5d_percent" in asset_data
                assert "roc_20d_percent" in asset_data
                assert "trend" in asset_data
                assert isinstance(asset_data["spot_price"], (int, float))
                assert asset_data["spot_price"] > 0.0

    def test_get_economic_calendar_dual_country_filter(self):
        """Verify economic calendar returns separated India and US event lists."""
        raw_output = get_economic_calendar.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a valid JSON object"
        if "error" in data:
            assert data.get("status") == "unconfigured"
        else:
            assert "domestic_events_india" in data
            assert "global_cues_us" in data
            assert isinstance(data["domestic_events_india"], list)
            assert isinstance(data["global_cues_us"], list)