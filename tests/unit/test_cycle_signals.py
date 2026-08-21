"""
Unit Test Suite for the Core Cycle Signal Engine (cycle_signals.py).

Uses synthetic OHLCV series with known shapes (steady uptrend, steady downtrend,
topping pattern, bottoming pattern) so ZigZag / STC / trend-context / classify_cycle
behavior can be validated deterministically, with no network dependency. This is the
suite to run before trusting the classifier against real historical cycle turns.
"""

import numpy as np
import pandas as pd
import pytest

from shankh.agents.market.cycle_signals import (
    EvidenceItem,
    classify_cycle,
    compute_stc,
    compute_trend_context,
    compute_zigzag,
)


def _make_series(prices: np.ndarray, start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(prices))
    high = prices * 1.005
    low = prices * 0.995
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": high,
        "low": low,
        "close": prices,
        "volume": np.full(len(prices), 1_000_000),
    })


def _uptrend(n=300, start=100.0, drift=0.15, noise=0.4, seed=1):
    """Drift + noise + a slow sine wobble, so genuine multi-day pullbacks and momentum
    cycles exist (a pure linear drift never triggers a ZigZag reversal or an STC cycle —
    real index prices always have this kind of wobble on top of trend)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wobble = 3.0 * np.sin(t / 15.0)
    steps = drift + rng.normal(0, noise, n)
    return start + np.cumsum(steps) + wobble


def _downtrend(n=300, start=200.0, drift=-0.15, noise=0.4, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wobble = 3.0 * np.sin(t / 15.0)
    steps = drift + rng.normal(0, noise, n)
    return start + np.cumsum(steps) + wobble


class TestZigZag:
    def test_uptrend_produces_higher_highs_higher_lows(self):
        df = _make_series(_uptrend())
        result = compute_zigzag(df, pct_threshold=1.5)
        assert result.structure != "INSUFFICIENT_SWINGS", "Wobble+drift fixture must produce >= 4 swings."
        # A clean synthetic uptrend must score non-negative at worst.
        assert result.score >= 0.0

    def test_downtrend_produces_lower_highs_lower_lows(self):
        df = _make_series(_downtrend())
        result = compute_zigzag(df, pct_threshold=1.5)
        assert result.structure != "INSUFFICIENT_SWINGS"
        assert result.score <= 0.0

    def test_insufficient_data_returns_neutral(self):
        df = _make_series(_uptrend(n=5))
        result = compute_zigzag(df)
        assert result.structure == "INSUFFICIENT_DATA"
        assert result.score == 0.0


class TestSTC:
    def test_sustained_uptrend_gives_positive_score(self):
        """Accelerating (convex) uptrend keeps MACD monotonically rising, which pins the
        double-stochastic at its upper extreme for an extended stretch — the degenerate
        flat-range case the ffill-hold fix in compute_stc is specifically there for."""
        t = np.arange(250, dtype=float)
        rng = np.random.default_rng(3)
        prices = 100.0 + 0.05 * t + 0.0025 * t**2 + rng.normal(0, 0.3, 250)
        close = pd.Series(prices)
        result = compute_stc(close)
        assert result.score > 0.0
        assert 0.0 <= result.value <= 100.0

    def test_sustained_downtrend_gives_negative_score(self):
        t = np.arange(250, dtype=float)
        rng = np.random.default_rng(4)
        prices = 300.0 - 0.05 * t - 0.0025 * t**2 + rng.normal(0, 0.3, 250)
        close = pd.Series(prices)
        result = compute_stc(close)
        assert result.score < 0.0

    def test_pinned_extreme_holds_forward_instead_of_collapsing(self):
        """Regression test for the ffill-on-degenerate-range fix: a strictly monotonic
        run (zero rolling range once the window is inside the run) must report a high
        reading, not snap to 0 from a 0/~0 division."""
        close = pd.Series(np.linspace(100.0, 200.0, 120))
        result = compute_stc(close)
        assert result.value > 60.0, "Strong monotonic uptrend must not collapse to a near-zero STC reading."

    def test_insufficient_data_flagged(self):
        close = pd.Series(_uptrend(n=20))
        result = compute_stc(close)
        assert result.direction == "INSUFFICIENT_DATA"


class TestTrendContext:
    def test_price_above_rising_mas_scores_positive(self):
        close = pd.Series(_uptrend(n=250))
        result = compute_trend_context(close)
        assert result.score > 0.0
        assert result.price_vs_sma200_pct > 0.0

    def test_price_below_falling_mas_scores_negative(self):
        close = pd.Series(_downtrend(n=250, start=300.0))
        result = compute_trend_context(close)
        assert result.score < 0.0


class TestClassifyCycle:
    def test_all_bullish_signals_classify_expansion(self):
        evidence = [
            EvidenceItem("zigzag", "core", "HH/HL", 0.8),
            EvidenceItem("stc", "core", "72 (RISING)", 0.7),
            EvidenceItem("trend_context", "core", "+8% vs 200DMA", 0.6),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "EXPANSION"
        assert result.transition_risk == "low"
        assert result.cycle_confidence == 1.0

    def test_all_bearish_signals_classify_contraction(self):
        evidence = [
            EvidenceItem("zigzag", "core", "LH/LL", -0.8),
            EvidenceItem("stc", "core", "18 (FALLING)", -0.6),
            EvidenceItem("trend_context", "core", "-9% vs 200DMA", -0.7),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "CONTRACTION"
        assert result.transition_risk == "low"

    def test_rising_trend_falling_momentum_flags_distribution_with_transition_risk(self):
        """The exact divergence example from architecture.md: price still rising while
        momentum weakens should read as Distribution risk, not Expansion."""
        evidence = [
            EvidenceItem("trend_context", "core", "+5% vs 200DMA", 0.5),
            EvidenceItem("stc", "core", "35 (FALLING)", -0.5),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "DISTRIBUTION"
        assert result.transition_risk in ("medium", "high")
        assert result.transition_watch == "Distribution -> Contraction"

    def test_falling_trend_rising_momentum_flags_accumulation(self):
        evidence = [
            EvidenceItem("trend_context", "core", "-5% vs 200DMA", -0.4),
            EvidenceItem("stc", "core", "40 (RISING)", 0.4),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "ACCUMULATION"

    def test_supporting_tier_is_downweighted_relative_to_core(self):
        """A single strongly-bullish Supporting-tier signal should not flip a
        conflicting Core-tier read on its own — Core must dominate the composite."""
        evidence = [
            EvidenceItem("trend_context", "core", "-6% vs 200DMA", -0.9),
            EvidenceItem("stc", "core", "20 (FALLING)", -0.8),
            EvidenceItem("erp", "supporting", "3.0% (ATTRACTIVE)", 1.0),
        ]
        result = classify_cycle(evidence)
        assert result.composite_score < 0.0
        assert result.cycle_phase == "CONTRACTION"
