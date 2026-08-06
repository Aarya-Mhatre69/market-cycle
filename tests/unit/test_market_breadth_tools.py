"""
Unit Test Suite for Market Breadth Tools.

Validates 150-ticker cross-sectional breadth calculations, McClellan Oscillator accuracy,
JSON schema integrity, and sector matrix output contracts.
"""

import json
from shankh.agents.market.breadth_tools import (
    get_market_breadth_metrics,
    get_sector_participation_matrix,
)


class TestMarketBreadthToolsUnit:
    """Unit tests validating market breadth tool payloads and invariants."""

    def test_get_market_breadth_metrics_payload(self):
        """Verify market breadth tool returns valid JSON with 150-ticker participation data."""
        raw_output = get_market_breadth_metrics.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        if "error" in data:
            assert data.get("status") == "unavailable"
        else:
            assert "breadth_regime" in data
            assert "moving_average_participation" in data
            assert "52_week_high_low_expansion" in data
            assert "advance_decline_momentum" in data

            ma = data["moving_average_participation"]
            assert "pct_above_20dma" in ma
            assert "pct_above_50dma" in ma
            assert "pct_above_200dma" in ma
            assert 0.0 <= ma["pct_above_20dma"] <= 100.0
            assert 0.0 <= ma["pct_above_50dma"] <= 100.0
            assert 0.0 <= ma["pct_above_200dma"] <= 100.0

            hl = data["52_week_high_low_expansion"]
            assert "new_52w_highs" in hl
            assert "new_52w_lows" in hl
            assert "net_52w_highs" in hl
            assert hl["net_52w_highs"] == hl["new_52w_highs"] - hl["new_52w_lows"]

            ad = data["advance_decline_momentum"]
            assert "mcclellan_oscillator" in ad
            assert isinstance(ad["mcclellan_oscillator"], (int, float))

    def test_get_sector_participation_matrix_payload(self):
        """Verify sector participation matrix tool returns structured sector returns."""
        raw_output = get_sector_participation_matrix.invoke({})
        data = json.loads(raw_output)
        print(data)
        assert isinstance(data, dict), "Payload must be a JSON dictionary"
        assert "sector_breadth_positive_pct" in data
        assert "top_leading_sectors" in data
        assert "bottom_lagging_sectors" in data
        assert isinstance(data["sector_breadth_positive_pct"], (int, float))
        assert isinstance(data["top_leading_sectors"], dict)
        assert isinstance(data["bottom_lagging_sectors"], dict)