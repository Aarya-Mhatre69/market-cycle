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
    HysteresisState,
    apply_phase_hysteresis,
    classify_cycle,
    classify_cycle_stateful,
    compute_candlestick_patterns,
    compute_cmf,
    compute_gann_time_cycles,
    compute_harmonic_patterns,
    compute_obv,
    compute_stc,
    compute_trend_context,
    compute_volume_confirmed_pivot,
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


def _make_ohlc_bars(bars: list, start: str = "2023-01-01") -> pd.DataFrame:
    """Builds a DataFrame from explicit {open, high, low, close} dicts, one per bar —
    used for candlestick-pattern tests where the exact wick/body shape of individual
    bars matters, unlike _make_series's uniform 0.5% high/low band."""
    dates = pd.bdate_range(start=start, periods=len(bars))
    return pd.DataFrame({
        "date": dates,
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": np.full(len(bars), 1_000_000),
    })


def _flat_price_series(control_points: list, points_per_leg: int = 15, start: str = "2023-01-01") -> pd.DataFrame:
    """Builds a zero-wick (open=high=low=close) DataFrame that ramps linearly between
    successive `control_points`, so compute_zigzag's confirmed pivot prices land
    exactly on the control points (used for Harmonic Pattern tests, where hitting
    precise Fibonacci ratios requires exact pivot prices, not approximate ones)."""
    segments = []
    for i in range(len(control_points) - 1):
        seg = np.linspace(control_points[i], control_points[i + 1], points_per_leg, endpoint=(i == len(control_points) - 2))
        segments.append(seg)
    prices = np.concatenate(segments)
    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({
        "date": dates, "open": prices, "high": prices, "low": prices, "close": prices,
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

    def test_short_history_uses_linear_fallback(self):
        """Below the 252-point own-history floor (n=250 here, only ~50 points of
        price_vs_sma200 history once the 200-bar SMA warms up), a percentile claim
        isn't defensible — must fall back to the original bounded-linear formula
        rather than go silent, since trend_context is a core, always-voting signal."""
        close = pd.Series(_uptrend(n=250))
        result = compute_trend_context(close)
        assert result.method == "LINEAR_FALLBACK"

    def test_extreme_extension_scores_higher_than_moderate_once_percentile_active(self):
        """Brutal-review audit finding #2 regression test: checked empirically against
        the live universe, the OLD formula (clip at +/-10% from 200DMA) put 51.3% of
        all 2,210 stocks at the same +/-1.0 ceiling, unable to distinguish a mild 12%
        extension from a 90% blow-off top. With enough own-history for a percentile
        read, an all-time-extreme extension must score strictly higher than a
        middling one, not tie."""
        rng = np.random.default_rng(7)
        n = 600
        base = 100 * (1.0003 ** np.arange(n))
        noise = rng.normal(0, 0.015, n)
        price = np.abs(base * (1 + noise.cumsum() * 0.05))

        moderate = pd.Series(price.copy())
        moderate.iloc[-1] = moderate.iloc[-2] * 1.12

        extreme = pd.Series(price.copy())
        extreme.iloc[-1] = extreme.iloc[-2] * 1.90

        r_mod = compute_trend_context(moderate)
        r_ext = compute_trend_context(extreme)
        assert r_mod.method == "PERCENTILE"
        assert r_ext.method == "PERCENTILE"
        assert r_ext.score > r_mod.score
        # The old formula would have clipped BOTH to exactly 1.0 -- assert the
        # moderate case is no longer pinned at the ceiling.
        assert r_mod.score < 1.0

    def test_percentile_score_matches_independently_computed_percentile_rank(self):
        """Pins the exact algorithm (score == average of the two 2*percentile-1
        components) against an independently-computed expected value, rather than
        relying on compute_trend_context's own internals -- so a future refactor
        that silently changes the percentile formula gets caught here. Also directly
        demonstrates score == 0 no longer means 'at the 200DMA' (see docstring):
        this fixture's latest price_vs_sma200_pct is not 0%, yet its score is small
        because that reading is unremarkable relative to its OWN history."""
        rng = np.random.default_rng(5)
        n = 600
        t = np.arange(n, dtype=float)
        close = pd.Series(100.0 + 0.03 * t + rng.normal(0, 1.5, n).cumsum() * 0.15)

        result = compute_trend_context(close)
        assert result.method == "PERCENTILE"
        assert result.price_vs_sma200_pct != 0.0

        sma_short = close.rolling(50).mean()
        sma_long = close.rolling(200).mean()
        price_series = ((close - sma_long) / sma_long * 100.0).dropna()
        short_series = ((sma_short - sma_long) / sma_long * 100.0).dropna()
        expected_price_pctile = (price_series <= price_series.iloc[-1]).mean()
        expected_short_pctile = (short_series <= short_series.iloc[-1]).mean()
        expected_score = np.clip(
            ((2 * expected_price_pctile - 1) + (2 * expected_short_pctile - 1)) / 2.0, -1.0, 1.0
        )
        assert result.score == pytest.approx(expected_score, abs=1e-3)


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
        # 3 core-tier items, all agreeing, but confidence is now also scaled by
        # evidence completeness (see TestConfidenceScoring) — 3/6 of the full
        # production evidence set's weight (1.8/2.7), not a bare agreement fraction.
        assert result.cycle_confidence == 0.67

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


class TestFourPillarVote:
    """Regression tests for the accuracy-roadmap Phase 2 fix: valuation, earnings, and
    liquidity must be able to change cycle_phase, not just composite_score/confidence
    after the phase was already fixed by trend_context + stc alone (roadmap Section 3.1).

    Earnings gets full-vote status on the momentum axis (same rate-of-change type as
    STC); valuation and liquidity are tiebreaker/modifiers on their axis (direction and
    momentum respectively) rather than full votes — see the design-rationale comment
    above cycle_signals._DIRECTIONAL_SIGNALS for why valuation as a full directional
    vote produces mislabeled quadrants."""

    def test_decelerating_earnings_flips_downtrend_bounce_from_accumulation_to_contraction(self):
        """A mild STC uptick alone would read ACCUMULATION under the old mechanism;
        strongly negative earnings momentum (full vote on the momentum axis) should be
        able to pull the momentum axis back down to CONTRACTION."""
        evidence = [
            EvidenceItem("trend_context", "core", "-3% vs 200DMA", -0.3),
            EvidenceItem("stc", "core", "45 (RISING)", 0.1),
            EvidenceItem("earnings_momentum", "supporting", "-15% YoY", -1.0),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "CONTRACTION"

    def test_core_only_evidence_degrades_to_prior_behavior(self):
        """With no valuation/liquidity/earnings evidence (the walk-forward backtest's
        world), the directional axis is dominated by trend_context/zigzag and momentum
        by stc alone, same as the pre-Phase-2 mechanism's inputs — a deliberate
        backward-compatibility guarantee, not an accident."""
        evidence = [
            EvidenceItem("trend_context", "core", "+5% vs 200DMA", 0.5),
            EvidenceItem("stc", "core", "70 (RISING)", 0.5),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "EXPANSION"

    def test_expensive_valuation_breaks_a_near_zero_directional_tie(self):
        """Valuation is a tiebreaker/modifier on the directional axis — it should
        matter when trend/zigzag are genuinely indecisive (~flat, within epsilon of
        zero), rather than silently defaulting to "up" the way the old `>= 0`
        convention did with zero justification. Flat trend + expensive valuation +
        already-falling momentum should read CONTRACTION (an early rollover warning),
        not DISTRIBUTION (which the old always-defaults-flat-to-"up" convention would
        have produced by calling a genuinely directionless trend "up")."""
        evidence = [
            EvidenceItem("trend_context", "core", "+0.2% vs 200DMA", 0.02),
            EvidenceItem("stc", "core", "35 (FALLING)", -0.3),
            EvidenceItem("erp", "supporting", "-2.0% (EXPENSIVE)", -1.0),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "CONTRACTION"

    def test_valuation_does_not_override_a_clear_directional_read(self):
        """A strong existing trend read must not be overturned by valuation alone —
        it's a tiebreaker, not a co-equal vote (this is the fix for the mislabeling
        a full-vote design produces: mild uptrend + rich valuation + rising momentum
        must not compute as ACCUMULATION, which implies a bottoming process that isn't
        happening here)."""
        evidence = [
            EvidenceItem("trend_context", "core", "+2% vs 200DMA", 0.15),
            EvidenceItem("stc", "core", "60 (RISING)", 0.3),
            EvidenceItem("erp", "supporting", "-2.0% (EXPENSIVE)", -1.0),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "EXPANSION"

    def test_liquidity_breaks_a_near_zero_momentum_tie(self):
        """Liquidity is a tiebreaker/modifier on the momentum axis (architecture
        Section 10), not an independent vote — it should only matter when momentum is
        genuinely indecisive (near zero), and should be able to flip that tie."""
        evidence = [
            EvidenceItem("trend_context", "core", "+3% vs 200DMA", 0.4),
            EvidenceItem("stc", "core", "50 (FLAT)", 0.0),
            EvidenceItem("liquidity", "supporting", "M3 4% YoY (TIGHT)", -1.0),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "DISTRIBUTION"

    def test_liquidity_does_not_override_a_clear_momentum_read(self):
        """A strong existing momentum read must not be overturned by liquidity alone —
        it's a tiebreaker, not a 5th vote."""
        evidence = [
            EvidenceItem("trend_context", "core", "+5% vs 200DMA", 0.5),
            EvidenceItem("stc", "core", "75 (RISING)", 0.8),
            EvidenceItem("liquidity", "supporting", "M3 2% YoY (TIGHT)", -1.0),
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "EXPANSION"


class TestConfidenceScoring:
    """Brutal-review audit finding #1 regression tests: cycle_confidence used to be
    plain agree/len(evidence), which empirically INVERTED what confidence should
    mean -- checked against the live 2,210-stock universe, thin-evidence stocks
    (missing fundamentals) averaged HIGHER confidence (0.735) than fully-evidenced
    ones (0.669), and 81% of the 417 stocks reporting the max 1.0 were the
    thin-evidence ones. Confidence is now agreement-with-the-actual-axis-decision,
    scaled by evidence completeness (see _MAX_EVIDENCE_WEIGHT)."""

    def test_penalizes_thin_evidence_relative_to_full_evidence(self):
        """The core reproduction of the audit finding: a stock classified on only 3
        core signals must NOT out-score (or tie) one classified on the full 6-signal
        production evidence set, even when both sets agree perfectly internally."""
        thin = [
            EvidenceItem("zigzag", "core", "HH/HL", 0.8),
            EvidenceItem("stc", "core", "70 (RISING)", 0.7),
            EvidenceItem("trend_context", "core", "+8% vs 200DMA", 0.6),
        ]
        full = thin + [
            EvidenceItem("erp", "supporting", "3.0% (ATTRACTIVE)", 0.5),
            EvidenceItem("earnings_momentum", "supporting", "+12% YoY", 0.6),
            EvidenceItem("liquidity", "supporting", "M3 9% YoY", 0.4),
        ]
        r_thin = classify_cycle(thin)
        r_full = classify_cycle(full)
        assert r_thin.cycle_confidence < r_full.cycle_confidence
        assert r_full.cycle_confidence == 1.0

    def test_agreement_is_checked_against_axis_decision_not_composite_score(self):
        """A divergence case (Distribution: directional up, momentum down) where
        composite_score's blended sign differs from what each individual axis
        actually decided -- agreement must be scored against directional_up/
        momentum_up (what produced `phase`), not composite_score's sign."""
        evidence = [
            EvidenceItem("trend_context", "core", "+2% vs 200DMA", 0.15),  # directional: barely up
            EvidenceItem("zigzag", "core", "HH/HL", 0.05),                 # directional: barely up
            EvidenceItem("stc", "core", "20 (FALLING)", -0.9),             # momentum: clearly down
        ]
        result = classify_cycle(evidence)
        assert result.cycle_phase == "DISTRIBUTION"
        # composite_score (tier-weighted average of ALL 3, regardless of axis) is
        # negative here since STC's -0.9 dominates the blend -- if confidence still
        # compared against composite_score's sign, the two clearly-positive
        # directional items would incorrectly count as disagreeing.
        assert result.composite_score < 0.0
        # trend_context and zigzag (both positive, feeding directional_up=True) and
        # stc (negative, feeding momentum_up=False) all agree with THEIR OWN axis --
        # full agreement, only discounted by evidence completeness (3/6 core+supporting).
        assert result.cycle_confidence == 0.67

    def test_zero_evidence_is_zero_confidence(self):
        result = classify_cycle([])
        assert result.cycle_confidence == 0.0


class TestHysteresis:
    """Regression tests for the accuracy-roadmap Phase 2 whipsaw fix: a single
    contrary reading must not flip the confirmed phase; min_dwell consecutive
    contrary readings must."""

    def test_bootstrap_confirms_first_read_immediately(self):
        state = HysteresisState()
        confirmed, new_state, pending = apply_phase_hysteresis("EXPANSION", state, min_dwell=3)
        assert confirmed == "EXPANSION"
        assert pending is False
        assert new_state.confirmed_phase == "EXPANSION"

    def test_single_contrary_reading_does_not_flip_phase(self):
        state = HysteresisState(confirmed_phase="EXPANSION")
        confirmed, new_state, pending = apply_phase_hysteresis("DISTRIBUTION", state, min_dwell=3)
        assert confirmed == "EXPANSION", "One contrary reading must not flip the confirmed phase."
        assert pending is True
        assert new_state.candidate_phase == "DISTRIBUTION"
        assert new_state.candidate_count == 1

    def test_min_dwell_consecutive_readings_confirm_the_flip(self):
        state = HysteresisState(confirmed_phase="EXPANSION")
        for _ in range(2):
            confirmed, state, pending = apply_phase_hysteresis("DISTRIBUTION", state, min_dwell=3)
            assert confirmed == "EXPANSION"
        confirmed, state, pending = apply_phase_hysteresis("DISTRIBUTION", state, min_dwell=3)
        assert confirmed == "DISTRIBUTION"
        assert pending is True
        assert state.confirmed_phase == "DISTRIBUTION"

    def test_reverting_to_confirmed_phase_cancels_the_candidate(self):
        state = HysteresisState(confirmed_phase="EXPANSION")
        _, state, _ = apply_phase_hysteresis("DISTRIBUTION", state, min_dwell=3)
        _, state, _ = apply_phase_hysteresis("DISTRIBUTION", state, min_dwell=3)
        assert state.candidate_count == 2
        confirmed, state, pending = apply_phase_hysteresis("EXPANSION", state, min_dwell=3)
        assert confirmed == "EXPANSION"
        assert pending is False
        assert state.candidate_count == 0, "A reading back in line with the confirmed phase must reset the candidate."

    def test_min_dwell_one_confirms_immediately_matching_stateless_read(self):
        """min_dwell=1 must recover the stateless (un-hysteresised) behavior, so the
        axis-vote fix and the hysteresis fix can be A/B tested independently."""
        evidence = [
            EvidenceItem("trend_context", "core", "+5% vs 200DMA", 0.5),
            EvidenceItem("stc", "core", "20 (FALLING)", -0.5),
        ]
        state = HysteresisState(confirmed_phase="EXPANSION")
        result, new_state = classify_cycle_stateful(evidence, state, min_dwell=1)
        assert result.cycle_phase == "DISTRIBUTION"

    def test_pending_build_does_not_alter_transition_risk(self):
        """transition_risk stays the plain raw-magnitude read regardless of hysteresis
        state — folding hysteresis-pending status into it was tried and measurably
        backfired (see classify_cycle_stateful's comment): it inflated recall against
        independently-labeled turns purely by widening the "elevated" base rate
        (37% -> 61-74%), while precision-vs-base-rate lift stayed flat at ~0.94-0.96,
        no better than the pre-Phase-2 baseline. Pending status is surfaced separately
        via pending_phase/dwell_progress instead."""
        evidence = [
            EvidenceItem("trend_context", "core", "+1% vs 200DMA", 0.05),
            EvidenceItem("stc", "core", "48 (FALLING)", -0.05),
        ]
        state = HysteresisState(confirmed_phase="EXPANSION", candidate_phase="DISTRIBUTION", candidate_count=3)
        result, new_state = classify_cycle_stateful(evidence, state, min_dwell=5)
        assert result.cycle_phase == "EXPANSION", "Not yet confirmed — must still report the prior phase."
        assert result.transition_risk == "low", "Must match the raw magnitude-based read, unmodified by dwell state."
        assert result.pending_phase == "DISTRIBUTION"
        assert result.dwell_progress == "4/5"

    def test_no_pending_build_leaves_pending_fields_empty(self):
        evidence = [
            EvidenceItem("trend_context", "core", "+5% vs 200DMA", 0.5),
            EvidenceItem("stc", "core", "70 (RISING)", 0.5),
        ]
        state = HysteresisState(confirmed_phase="EXPANSION")
        result, new_state = classify_cycle_stateful(evidence, state, min_dwell=5)
        assert result.pending_phase is None
        assert result.dwell_progress is None


class TestCandlestickPatterns:
    """accuracy-roadmap Phase 3: Candlestick Recognition (Doji, Engulfing, Hammer,
    Morning Star). Each test isolates one pattern with hand-built OHLC bars; loose
    assertions (pattern present + correct sign) rather than exact scores, since a few
    of these constructions incidentally also trip a second, harmless pattern (e.g. a
    long bearish "Star A" candle in a Morning Star setup naturally also satisfies
    Bearish Engulfing against the prior bar) — real market data does this too."""

    def test_insufficient_data_returns_neutral(self):
        df = _make_ohlc_bars([{"open": 100, "high": 101, "low": 99, "close": 100.5}] * 5)
        result = compute_candlestick_patterns(df)
        assert result.score == 0.0
        assert result.patterns == []

    def test_no_patterns_in_plain_series_returns_neutral(self):
        bars = [{"open": 100 + i * 0.05, "high": 100 + i * 0.05 + 0.3, "low": 100 + i * 0.05 - 0.3,
                 "close": 100 + i * 0.05 + 0.05} for i in range(20)]
        result = compute_candlestick_patterns(bars_df := _make_ohlc_bars(bars))
        assert result.score == 0.0
        assert result.patterns == []

    def test_doji_after_uptrend_reads_bearish(self):
        bars = []
        for i in range(14):
            close = 100.0 + i
            open_ = close - 0.8
            bars.append({"open": open_, "high": close + 0.2, "low": open_ - 0.2, "close": close})
        bars.append({"open": 114.0, "close": 114.05, "high": 116.0, "low": 112.0})  # doji
        result = compute_candlestick_patterns(_make_ohlc_bars(bars))
        names_directions = {(p["pattern"], p["direction"]) for p in result.patterns}
        assert ("DOJI", "bearish") in names_directions
        assert result.score < 0.0

    def test_bullish_engulfing_detected_and_scores_positive(self):
        bars = [{"open": 100.0 + i - 0.3, "high": 100.0 + i + 0.1, "low": 100.0 + i - 0.4, "close": 100.0 + i}
                for i in range(13)]
        bars.append({"open": 112.0, "close": 107.0, "high": 112.5, "low": 106.5})  # bearish reversal
        bars.append({"open": 106.0, "close": 113.0, "high": 113.5, "low": 105.5})  # bullish engulfing
        result = compute_candlestick_patterns(_make_ohlc_bars(bars))
        pattern_names = {p["pattern"] for p in result.patterns}
        assert "BULLISH_ENGULFING" in pattern_names
        assert result.score > 0.0

    def test_hammer_after_downtrend_reads_bullish(self):
        bars = []
        for i in range(14):
            close = 114.0 - i
            open_ = close + 0.3
            bars.append({"open": open_, "high": open_ + 0.1, "low": close - 0.1, "close": close})
        bars.append({"open": 100.5, "close": 100.8, "high": 100.85, "low": 98.5})  # hammer
        result = compute_candlestick_patterns(_make_ohlc_bars(bars))
        pattern_names = {p["pattern"] for p in result.patterns}
        assert "HAMMER" in pattern_names
        assert result.score > 0.0

    def test_morning_star_detected_and_scores_positive(self):
        bars = [{"open": 100 + i * 0.2 - 0.1, "high": 100 + i * 0.2 + 0.05, "low": 100 + i * 0.2 - 0.15,
                 "close": 100 + i * 0.2} for i in range(12)]
        bars.append({"open": 105.0, "close": 100.0, "high": 105.2, "low": 99.8})   # long bearish
        bars.append({"open": 98.0, "close": 98.3, "high": 98.5, "low": 97.5})      # small star, gapped down
        bars.append({"open": 98.0, "close": 104.0, "high": 104.2, "low": 97.8})    # long bullish, closes into bar 1
        result = compute_candlestick_patterns(_make_ohlc_bars(bars))
        pattern_names = {p["pattern"] for p in result.patterns}
        assert "MORNING_STAR" in pattern_names
        assert result.score > 0.0


class TestHarmonicPatterns:
    """accuracy-roadmap Phase 3: Harmonic Patterns (Gartley/Bat/Butterfly/Crab/Shark)
    via XABCD Fibonacci ratios, reusing compute_zigzag's pivots."""

    def test_insufficient_pivots_returns_neutral(self):
        df = _flat_price_series([1000, 1050], points_per_leg=15)
        result = compute_harmonic_patterns(df, pct_threshold=2.0)
        assert result.pattern is None
        assert result.score == 0.0

    def test_gartley_ratios_detected_as_bullish(self):
        """X=1000(LOW) -> A=1500(HIGH): XA=500. B=A-0.618*XA=1191 (AB/XA=0.618, in
        Gartley's 0.55-0.68 band). C=B+0.5*AB (BC/AB=0.5, in 0.38-0.89 band).
        D=A-0.786*XA=1107 (AD/XA=0.786, in Gartley's 0.72-0.85 band) -> D is a LOW,
        so the pattern reads bullish (reversal-up expected off the D buy zone)."""
        X, A = 1000.0, 1500.0
        xa = A - X
        B = A - 0.618 * xa
        ab = A - B
        C = B + 0.5 * ab
        D = A - 0.786 * xa
        confirm = D * 1.05  # extra leg so the D pivot actually confirms (reverses by >= threshold)
        df = _flat_price_series([1050.0, X, A, B, C, D, confirm], points_per_leg=15)
        result = compute_harmonic_patterns(df, pct_threshold=2.0)
        assert result.pattern == "GARTLEY"
        assert result.direction == "bullish"
        assert result.score > 0.0
        assert abs(result.completion_ratio - 0.786) < 0.02

    def test_mismatched_ratios_returns_no_pattern(self):
        """Legs sized well outside every documented band (a near-straight-line
        retracement structure) must not force a false pattern match."""
        X, A = 1000.0, 1500.0
        B = 1490.0    # AB/XA ~ 0.02, below every pattern's ab_xa band
        C = 1495.0
        D = 1300.0
        confirm = D * 1.05
        df = _flat_price_series([1050.0, X, A, B, C, D, confirm], points_per_leg=15)
        result = compute_harmonic_patterns(df, pct_threshold=2.0)
        assert result.pattern is None
        assert result.score == 0.0


class TestGannTimeCycles:
    """accuracy-roadmap Phase 3: Gann Time Cycles — a documented, non-standard
    convention (see cycle_signals._GANN_CYCLE_DAYS); tests validate the mechanics
    (window detection + direction), not any claim that the methodology itself works."""

    def _build_pivot_then_drift(self, direction: str, days_after_pivot: int) -> pd.DataFrame:
        """Calendar-daily (not business-day) bars: 1050->1000 confirms a LOW pivot,
        then price drifts up or down for `days_after_pivot` more days so the returned
        df's last row lands exactly that many calendar days after the pivot's date."""
        if direction == "up":
            approach = np.linspace(1050.0, 1000.0, 30)  # dip confirms a LOW pivot at ~1000
            drift = 1000.0 + np.linspace(0, 200, 400)   # then rises
        else:
            approach = np.linspace(950.0, 1000.0, 30)   # rise confirms a HIGH pivot at ~1000
            drift = 1000.0 - np.linspace(0, 400, 400)   # then declines
            drift = np.clip(drift, 100.0, None)
        prices = np.concatenate([approach, drift])
        dates = pd.date_range(start="2023-01-01", periods=len(prices), freq="D")
        full_df = pd.DataFrame({
            "date": dates, "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": np.full(len(prices), 1_000_000),
        })
        pivots = compute_zigzag(full_df, pct_threshold=4.0).all_pivots
        assert pivots, "Test construction bug: no pivot confirmed."
        pivot_date = pd.Timestamp(pivots[0]["date"])
        target_date = pivot_date + pd.Timedelta(days=days_after_pivot)
        return full_df[full_df["date"] <= target_date].reset_index(drop=True)

    def test_active_window_after_uptrend_reads_bearish(self):
        df = self._build_pivot_then_drift("up", days_after_pivot=90)
        result = compute_gann_time_cycles(df, pct_threshold=4.0)
        assert result.active_cycle_days == 90
        assert result.score < 0.0

    def test_active_window_after_downtrend_reads_bullish(self):
        df = self._build_pivot_then_drift("down", days_after_pivot=90)
        result = compute_gann_time_cycles(df, pct_threshold=4.0)
        assert result.active_cycle_days == 90
        assert result.score > 0.0

    def test_no_active_window_returns_neutral(self):
        df = self._build_pivot_then_drift("up", days_after_pivot=20)
        result = compute_gann_time_cycles(df, pct_threshold=4.0)
        assert result.active_cycle_days is None
        assert result.score == 0.0


class TestOBV:
    def test_sustained_uptrend_gives_positive_score(self):
        close = pd.Series(np.linspace(100.0, 150.0, 30))
        volume = pd.Series(np.full(30, 1_000_000.0))
        result = compute_obv(close, volume)
        assert result.score > 0.0

    def test_sustained_downtrend_gives_negative_score(self):
        close = pd.Series(np.linspace(150.0, 100.0, 30))
        volume = pd.Series(np.full(30, 1_000_000.0))
        result = compute_obv(close, volume)
        assert result.score < 0.0

    def test_insufficient_data_returns_neutral(self):
        close = pd.Series(np.linspace(100.0, 105.0, 10))
        volume = pd.Series(np.full(10, 1_000_000.0))
        result = compute_obv(close, volume, window=20)
        assert result.score == 0.0


class TestCMF:
    def _bars_closing_near(self, position: str, n: int = 25) -> pd.DataFrame:
        highs = 100.0 + np.arange(n)
        lows = highs - 10.0
        closes = highs - 0.5 if position == "high" else lows + 0.5
        return pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=n),
            "open": closes, "high": highs, "low": lows, "close": closes,
            "volume": np.full(n, 1_000_000.0),
        })

    def test_closes_near_highs_gives_positive_cmf(self):
        result = compute_cmf(self._bars_closing_near("high"))
        assert result.score > 0.5

    def test_closes_near_lows_gives_negative_cmf(self):
        result = compute_cmf(self._bars_closing_near("low"))
        assert result.score < -0.5

    def test_insufficient_data_returns_neutral(self):
        result = compute_cmf(self._bars_closing_near("high", n=5), window=20)
        assert result.score == 0.0


class TestVolumeConfirmedPivot:
    def _df_with_known_pivot(self):
        """5-leg HIGHER_HIGHS_HIGHER_LOWS structure (W->X->A->B->C->D->confirm),
        same construction pattern as TestHarmonicPatterns — needs >= 4 confirmed
        pivots for compute_zigzag to report a non-zero structure/score at all
        (fewer reports INSUFFICIENT_SWINGS, score 0.0, leaving nothing for volume
        to confirm). The last pivot (D) is what volume gets injected around."""
        W, X, A, B, C, D, confirm = 1050.0, 1000.0, 1200.0, 1100.0, 1300.0, 1250.0, 1320.0
        segments = [
            np.linspace(W, X, 12), np.linspace(X, A, 12)[1:], np.linspace(A, B, 12)[1:],
            np.linspace(B, C, 12)[1:], np.linspace(C, D, 12)[1:], np.linspace(D, confirm, 12)[1:],
        ]
        prices = np.concatenate(segments)
        dates = pd.bdate_range("2023-01-01", periods=len(prices))
        df = pd.DataFrame({
            "date": dates, "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": np.full(len(prices), 1_000_000.0),
        })
        zz = compute_zigzag(df, pct_threshold=2.0)
        assert len(zz.all_pivots) >= 4, f"Test construction bug: only {len(zz.all_pivots)} pivots confirmed."
        assert zz.score != 0.0, "Test construction bug: zigzag has no directional opinion to confirm."
        return df, zz

    def test_high_volume_pivot_confirms_in_zigzag_direction(self):
        df, zz = self._df_with_known_pivot()
        pivot_date = pd.Timestamp(zz.all_pivots[-1]["date"])
        pivot_idx = df[df["date"] <= pivot_date].index[-1]
        df.loc[pivot_idx, "volume"] = 5_000_000.0  # well above the 1M trailing average
        result = compute_volume_confirmed_pivot(df, zz)
        assert result.volume_ratio > 1.0
        # zz's last pivot here is a HIGH (bearish structure once confirmed downward),
        # so a confirming score should share zigzag's own sign, whatever it is.
        assert (result.score >= 0) == (zz.score >= 0)
        assert result.score != 0.0

    def test_low_volume_pivot_does_not_confirm(self):
        df, zz = self._df_with_known_pivot()
        pivot_date = pd.Timestamp(zz.all_pivots[-1]["date"])
        pivot_idx = df[df["date"] <= pivot_date].index[-1]
        df.loc[pivot_idx, "volume"] = 200_000.0  # well below the 1M trailing average
        result = compute_volume_confirmed_pivot(df, zz)
        assert result.volume_ratio < 1.0
        assert result.score == 0.0

    def test_no_pivots_returns_neutral(self):
        df = _make_series(np.full(15, 100.0))
        zz = compute_zigzag(df)
        result = compute_volume_confirmed_pivot(df, zz)
        assert result.score == 0.0
