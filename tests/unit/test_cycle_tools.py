"""
Unit Test Suite for Market Cycle Tools.

Validates Equity Risk Premium (ERP) formula calculations, valuation percentiles,
JSON schema integrity, credit cycle payloads, and the weighted synthesis classification.
"""

import json
import pytest
from shankh.agents.market.cycle_tools import (
    _compute_pe_percentile,
    get_market_cycle_metrics,
    get_liquidity_and_credit_cycle,
    get_index_earnings_momentum,
    get_market_cycle_synthesis,
)


class TestMarketCycleToolsUnit:
    """Unit tests validating market cycle tool payloads and invariants."""

    def test_get_market_cycle_metrics_payload(self):
        """Verify cycle metrics tool returns valid JSON with ERP and valuation percentiles."""
        raw_output = get_market_cycle_metrics.invoke({})
        data = json.loads(raw_output)

        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        assert "index_valuations" in data
        assert "equity_risk_premium" in data

        val = data["index_valuations"]
        assert "nifty_pe_ratio" in val
        assert "nifty_pe_data_source" in val
        assert "pe_10y_historical_percentile" in val
        assert val["nifty_pe_ratio"] > 0.0
        assert val["nifty_pe_data_source"] in ("LIVE_FMP", "FALLBACK_BENCHMARK")

        erp = data["equity_risk_premium"]
        assert "index_earnings_yield_percent" in erp
        assert "india_10y_gsec_yield_percent" in erp
        assert "equity_risk_premium_spread_percent" in erp
        assert -1.0 <= erp["erp_score"] <= 1.0

        expected_erp = round(
            erp["index_earnings_yield_percent"] - erp["india_10y_gsec_yield_percent"], 2
        )
        assert abs(erp["equity_risk_premium_spread_percent"] - expected_erp) < 0.1

    def test_get_liquidity_and_credit_cycle_payload(self):
        """Verify credit cycle tool returns valid liquidity payload without fabricating bank credit growth."""
        raw_output = get_liquidity_and_credit_cycle.invoke({})
        data = json.loads(raw_output)

        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        assert data["bank_credit_growth_yoy_percent"] == "insufficient_data", (
            "No live bank-credit-growth source is wired; must not report a fabricated figure."
        )
        assert "m3_money_supply_growth_yoy_percent" in data
        assert "credit_cycle_status" in data
        assert "rbi_monetary_policy_stance" in data

    def test_get_index_earnings_momentum_payload(self):
        """Verify earnings-momentum tool never fabricates a figure below the coverage floor."""
        raw_output = get_index_earnings_momentum.invoke({})
        data = json.loads(raw_output)

        if data["earnings_momentum_yoy_percent"] == "insufficient_data":
            pytest.skip("Fewer than 5/15 basket tickers returned data in this environment (no network or yfinance rate-limited).")

        assert "basket_coverage" in data
        assert -1.0 <= data["earnings_score"] <= 1.0

    def test_get_market_cycle_synthesis_schema(self):
        """Verify the synthesis tool returns the full architecture-spec output schema."""
        raw_output = get_market_cycle_synthesis.invoke({})
        data = json.loads(raw_output)

        if data.get("status") == "unavailable":
            pytest.skip("No network access to fetch index OHLCV in this environment.")

        assert data["cycle_phase"] in (
            "ACCUMULATION", "EXPANSION", "DISTRIBUTION", "CONTRACTION",
        )
        assert 0.0 <= data["cycle_confidence"] <= 1.0
        assert data["transition_risk"] in ("low", "medium", "high")
        assert -1.0 <= data["composite_score"] <= 1.0
        assert isinstance(data["evidence"], list) and len(data["evidence"]) > 0
        for item in data["evidence"]:
            assert {"signal", "tier", "reading", "score", "note"} <= set(item.keys())
            assert item["tier"] in ("core", "supporting", "contextual")


class TestComputePePercentile:
    """accuracy-roadmap Phase 4: real rolling P/E percentile, replacing the 4
    hardcoded threshold buckets that silently mismatched the prompt/README's
    '10-Year Historical Percentile' claim."""

    def test_insufficient_history_reports_honestly(self):
        assert _compute_pe_percentile(22.0, None) == "insufficient_data"
        assert _compute_pe_percentile(22.0, [20.0] * 100) == "insufficient_data"

    def test_current_value_at_known_percentile(self):
        history = list(range(1, 253))  # 1..252, uniform
        # current_pe=126 -> exactly half the history (1..126) is <= 126 -> 50th percentile.
        result = _compute_pe_percentile(126, [float(x) for x in history])
        assert 49.0 <= result <= 51.0

    def test_extreme_high_pe_reads_near_100th_percentile(self):
        history = [float(x) for x in range(1, 253)]
        result = _compute_pe_percentile(1000.0, history)
        assert result == 100.0

    def test_extreme_low_pe_reads_near_0th_percentile(self):
        history = [float(x) for x in range(1, 253)]
        result = _compute_pe_percentile(0.0, history)
        assert result == 0.0
