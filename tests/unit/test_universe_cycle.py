"""
Unit tests for universe_cycle.classify_one_stock — the pure per-stock classification
logic (no network I/O), using synthetic OHLCV. Mirrors the fixture style already used
in tests/unit/test_cycle_signals.py.
"""

import numpy as np
import pandas as pd

from shankh.agents.market.universe_cycle import classify_one_stock


def _make_series(prices: np.ndarray, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(prices))
    high = prices * 1.005
    low = prices * 0.995
    return pd.DataFrame({
        "date": dates, "open": prices, "high": high, "low": low, "close": prices,
        "volume": np.full(len(prices), 1_000_000),
    })


def _uptrend(n=300, start=100.0, drift=0.15, noise=0.4, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wobble = 3.0 * np.sin(t / 15.0)
    steps = drift + rng.normal(0, noise, n)
    return start + np.cumsum(steps) + wobble


class TestClassifyOneStock:
    def test_uptrend_stock_with_full_fundamentals_classifies_and_returns_expected_shape(self):
        ohlcv = _make_series(_uptrend(n=300))
        fundamentals = {
            "company_name": "Test Corp", "sector": "Technology",
            "pe_ratio": 22.0, "pb_ratio": 4.0, "dividend_yield_percent": 1.2,
            "earnings_quarterly_growth": 0.15,
        }
        row = classify_one_stock(
            "TEST.NS", ohlcv, fundamentals,
            liquidity_score=0.2, india_10y_yield=7.0, pe_percentile_vs_universe=55.0,
            prior_state=None,
        )
        assert row["ticker"] == "TEST.NS"
        assert row["cycle_phase"] in ("EXPANSION", "DISTRIBUTION", "CONTRACTION", "ACCUMULATION")
        assert 0.0 <= row["cycle_confidence"] <= 1.0
        assert row["transition_risk"] in ("low", "medium", "high")
        assert row["pe_ratio"] == 22.0
        assert row["pe_percentile_vs_universe"] == 55.0
        assert "_new_state" in row
        assert row["_new_state"]["confirmed_phase"] == row["cycle_phase"]

    def test_missing_fundamentals_does_not_crash_and_omits_valuation_evidence(self):
        ohlcv = _make_series(_uptrend(n=300))
        row = classify_one_stock(
            "NOFUNDA.NS", ohlcv, None,
            liquidity_score=None, india_10y_yield=7.0, pe_percentile_vs_universe=None,
            prior_state=None,
        )
        assert row["pe_ratio"] is None
        assert row["pe_percentile_vs_universe"] is None
        assert row["cycle_phase"] in ("EXPANSION", "DISTRIBUTION", "CONTRACTION", "ACCUMULATION")

    def test_prior_state_is_carried_forward_through_hysteresis(self):
        """A stock whose raw read disagrees with its prior confirmed phase must not
        flip immediately — same min_dwell=5 hysteresis guarantee as the index. This
        fixture (_uptrend) is the same one test_cycle_signals.py's ZigZag/trend-context
        tests already assert scores positive for, so its raw read is reliably
        EXPANSION-leaning, not CONTRACTION."""
        ohlcv = _make_series(_uptrend(n=300))
        prior_state = {"confirmed_phase": "CONTRACTION", "candidate_phase": None, "candidate_count": 0}
        row = classify_one_stock(
            "CARRY.NS", ohlcv, None,
            liquidity_score=None, india_10y_yield=7.0, pe_percentile_vs_universe=None,
            prior_state=prior_state,
        )
        assert row["cycle_phase"] == "CONTRACTION", "Prior phase must still be reported — one read can't confirm a flip."
        assert row["pending_phase"] is not None
        assert row["dwell_progress"] == "1/5"
