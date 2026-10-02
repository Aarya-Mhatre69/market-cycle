"""
Unit tests for universe_cycle.classify_one_stock — the pure per-stock classification
logic (no network I/O), using synthetic OHLCV. Mirrors the fixture style already used
in tests/unit/test_cycle_signals.py.
"""

import numpy as np
import pandas as pd

from shankh.agents.market.universe_cycle import (
    append_phase_changes, classify_one_stock, detect_phase_change, yoy_net_income_growth,
)


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


def _row(phase="DISTRIBUTION", date="2026-10-01", ticker="AAA.NS"):
    return {"as_of_date": date, "ticker": ticker, "company_name": "AAA Ltd", "sector": "Industrials",
            "cycle_phase": phase, "close": 123.456, "cycle_confidence": 0.6}


class TestPhaseChangeFeed:
    def test_flip_in_confirmed_phase_is_an_event(self):
        event = detect_phase_change({"confirmed_phase": "EXPANSION"}, _row("DISTRIBUTION"))
        assert event["from_phase"] == "EXPANSION" and event["to_phase"] == "DISTRIBUTION"
        assert event["close"] == 123.46

    def test_same_phase_or_first_sighting_is_not_an_event(self):
        assert detect_phase_change({"confirmed_phase": "DISTRIBUTION"}, _row("DISTRIBUTION")) is None
        assert detect_phase_change(None, _row("DISTRIBUTION")) is None
        assert detect_phase_change({}, _row("DISTRIBUTION")) is None

    def test_rerun_on_the_same_day_does_not_double_count(self, tmp_path):
        path = tmp_path / "phase_changes.csv"
        event = detect_phase_change({"confirmed_phase": "EXPANSION"}, _row("DISTRIBUTION"))
        append_phase_changes([event], path=path)
        assert append_phase_changes([event], path=path) == 1

    def test_feed_is_trimmed_to_the_keep_window(self, tmp_path):
        path = tmp_path / "phase_changes.csv"
        old = detect_phase_change({"confirmed_phase": "EXPANSION"}, _row("DISTRIBUTION", date="2026-01-01", ticker="OLD.NS"))
        new = detect_phase_change({"confirmed_phase": "EXPANSION"}, _row("CONTRACTION", date="2026-10-01", ticker="NEW.NS"))
        n = append_phase_changes([old, new], path=path, keep_days=90)
        assert n == 1
        assert pd.read_csv(path)["ticker"].tolist() == ["NEW.NS"]

    def test_no_events_leaves_a_header_only_file(self, tmp_path):
        path = tmp_path / "phase_changes.csv"
        assert append_phase_changes([], path=path) == 0
        assert path.exists() and len(pd.read_csv(path)) == 0


def _quarters(*incomes, dates=("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30")):
    return [{"date": d, "netIncome": ni} for d, ni in zip(dates, incomes)]


class TestYoyNetIncomeGrowth:
    def test_matches_live_marksans_and_reliance_figures(self):
        # Real FMP values checked on 2026-10-02. MARKSANS: +169.5% YoY. RELIANCE: -22.4% YoY,
        # even though the quarter-over-quarter figure FMP's growth endpoint returns is +23%.
        assert round(yoy_net_income_growth(_quarters(1571.67, 1481.29, 1132.02, 982.5, 583.17)), 3) == 1.695
        assert round(yoy_net_income_growth(_quarters(209.46, 169.71, 186.45, 181.65, 269.94)), 3) == -0.224

    def test_too_few_quarters_or_zero_base_is_none(self):
        assert yoy_net_income_growth(_quarters(1, 2, 3, 4)) is None
        assert yoy_net_income_growth(_quarters(5, 4, 3, 2, 0)) is None

    def test_a_missing_quarter_is_not_silently_compared(self):
        gapped = _quarters(5, 4, 3, 2, 1, dates=("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-03-31"))
        assert yoy_net_income_growth(gapped) is None

    def test_loss_to_profit_reads_as_growth(self):
        assert yoy_net_income_growth(_quarters(50, 0, 0, 0, -50)) == 2.0

    def test_malformed_rows_are_none(self):
        assert yoy_net_income_growth([{"date": "x"}] * 5) is None
