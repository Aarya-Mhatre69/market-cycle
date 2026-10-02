import math

from shankh.agents.market.cycle_explain import explain_classification


def _row(**overrides):
    row = {
        "cycle_phase": "DISTRIBUTION", "transition_risk": "low",
        "trend_score": 0.94, "zigzag_score": 0.0, "stc_score": -0.503,
        "pending_phase": None, "dwell_progress": None,
    }
    row.update(overrides)
    return row


def test_distribution_explanation_matches_the_marksans_case():
    text = explain_classification(_row())
    assert text.startswith("Distribution: the trend is still up but momentum is fading")
    assert "relative to its own history, price is far above its long-term average" in text.lower()
    assert "no clear direction" in text
    assert "STC -0.50" in text


def test_phase_is_the_source_of_truth_for_the_headline():
    # Even if scores look contradictory, the headline follows the reported phase.
    text = explain_classification(_row(cycle_phase="ACCUMULATION", trend_score=0.9, stc_score=-0.9))
    assert text.startswith("Accumulation: the trend is down but momentum is recovering")


def test_risk_and_pending_sentences():
    text = explain_classification(_row(transition_risk="medium", pending_phase="EXPANSION", dwell_progress="2/5"))
    assert "Chance of a phase change soon: medium." in text
    assert "A move to Expansion is building (2/5 confirmations)." in text


def test_low_risk_says_no_change_signalled():
    assert "No phase change is signalled." in explain_classification(_row(transition_risk="low"))


def test_missing_scores_are_skipped_not_crashed():
    text = explain_classification({"cycle_phase": "EXPANSION", "trend_score": math.nan, "stc_score": None})
    assert text == "Expansion: the trend is up and momentum is rising."


def test_unknown_phase():
    assert explain_classification({"cycle_phase": "???"}) == "No classification available."


def test_low_risk_with_a_pending_change_does_not_claim_no_change():
    text = explain_classification(_row(transition_risk="low", pending_phase="EXPANSION", dwell_progress="2/5"))
    assert "No phase change is signalled." not in text
    assert "A move to Expansion is building" in text


def test_momentum_wording_follows_the_sign():
    assert "slightly negative (STC -0.17)" in explain_classification(_row(stc_score=-0.17))
    assert "slightly positive (STC +0.10)" in explain_classification(_row(stc_score=0.10))
