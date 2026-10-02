"""
Plain-English reason for a stock's phase.

Deterministic and descriptive only: the reported phase is the source of truth for
"trend up/down" and "momentum rising/falling"; the stored scores are used only to
describe the evidence behind it. It therefore cannot disagree with the classifier,
even when a tie-breaking modifier (valuation, liquidity) nudged an axis.
"""

import math
from typing import Mapping, Optional

_PHASE_MEANING = {
    "EXPANSION": "the trend is up and momentum is rising",
    "DISTRIBUTION": "the trend is still up but momentum is fading, which can mark a top",
    "CONTRACTION": "the trend is down and momentum is weak",
    "ACCUMULATION": "the trend is down but momentum is recovering, which can mark a bottom",
}


def _num(row: Mapping, key: str) -> Optional[float]:
    value = row.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _trend_clause(score: float) -> str:
    # The trend score is a percentile of the stock's OWN distance-from-200-day-average
    # history, so it says "high or low versus its usual", not "above or below the average".
    # The wording must keep that qualifier or it misstates what the score means.
    if score >= 0.5:
        return "relative to its own history, price is far above its long-term average"
    if score >= 0:
        return "relative to its own history, price is above its long-term average"
    if score > -0.5:
        return "relative to its own history, price is below its long-term average"
    return "relative to its own history, price is far below its long-term average"


def _swing_clause(score: float) -> str:
    if score > 0:
        return "recent swings make higher highs and higher lows"
    if score < 0:
        return "recent swings make lower highs and lower lows"
    return "recent swings show no clear direction"


def _momentum_clause(score: float) -> str:
    # Wording follows the sign, because the sign is what decides rising vs falling:
    # a mildly negative score must not read as "flat" under a "momentum is fading" headline.
    if score >= 0.3:
        return "momentum is strengthening"
    if score >= 0:
        return "momentum is slightly positive"
    if score > -0.3:
        return "momentum is slightly negative"
    return "momentum is fading"


def explain_classification(row: Mapping) -> str:
    phase = str(row.get("cycle_phase", "")).upper()
    meaning = _PHASE_MEANING.get(phase)
    if meaning is None:
        return "No classification available."

    parts = [f"{phase.title()}: {meaning}."]

    trend, swing, stc = _num(row, "trend_score"), _num(row, "zigzag_score"), _num(row, "stc_score")
    evidence = []
    if trend is not None:
        evidence.append(_trend_clause(trend))
    if swing is not None:
        evidence.append(_swing_clause(swing))
    if stc is not None:
        evidence.append(f"{_momentum_clause(stc)} (STC {stc:+.2f})")
    if evidence:
        parts.append(evidence[0].capitalize() + "".join(f", {e}" for e in evidence[1:]) + ".")

    risk = str(row.get("transition_risk", "")).lower()
    pending = row.get("pending_phase")
    has_pending = isinstance(pending, str) and bool(pending)
    tail = []
    if risk in ("medium", "high"):
        tail.append(f"Chance of a phase change soon: {risk}.")
    elif risk == "low" and not has_pending:
        tail.append("No phase change is signalled.")
    if has_pending:
        progress = row.get("dwell_progress")
        suffix = f" ({progress} confirmations)" if isinstance(progress, str) and progress else ""
        tail.append(f"A move to {pending.title()} is building{suffix}.")
    if tail:
        parts.append(" ".join(tail))

    return " ".join(parts)
