"""
Unit Test Suite for Market Cycle Tools.

Validates Equity Risk Premium (ERP) formula calculations, valuation percentiles,
JSON schema integrity, and credit cycle status payloads.
"""

import json
import pytest
from shankh.agents.market.cycle_tools import (
    get_market_cycle_metrics,
    get_liquidity_and_credit_cycle,
)


class TestMarketCycleToolsUnit:
    """Unit tests validating market cycle tool payloads and invariants."""

    def test_get_market_cycle_metrics_payload(self):
        """Verify cycle metrics tool returns valid JSON with ERP and valuation percentiles."""
        raw_output = get_market_cycle_metrics.invoke({})
        data = json.loads(raw_output)

        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        assert "cycle_phase" in data
        assert "index_valuations" in data
        assert "equity_risk_premium" in data
        assert "yield_curve_and_rates" in data

        val = data["index_valuations"]
        assert "nifty_pe_ratio" in val
        assert "pe_10y_historical_percentile" in val
        assert val["nifty_pe_ratio"] > 0.0

        erp = data["equity_risk_premium"]
        assert "index_earnings_yield_percent" in erp
        assert "india_10y_gsec_yield_percent" in erp
        assert "equity_risk_premium_spread_percent" in erp

        expected_erp = round(
            erp["index_earnings_yield_percent"] - erp["india_10y_gsec_yield_percent"], 2
        )
        assert abs(erp["equity_risk_premium_spread_percent"] - expected_erp) < 0.1

    def test_get_liquidity_and_credit_cycle_payload(self):
        """Verify credit cycle tool returns valid credit growth and monetary stance."""
        raw_output = get_liquidity_and_credit_cycle.invoke({})
        data = json.loads(raw_output)

        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        assert "bank_credit_growth_yoy_percent" in data
        assert "m3_money_supply_growth_yoy_percent" in data
        assert "credit_cycle_status" in data
        assert "rbi_monetary_policy_stance" in data
        assert isinstance(data["bank_credit_growth_yoy_percent"], (int, float))