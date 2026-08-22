"""
Standalone Market Cycle Agent Demo — no OPENAI_API_KEY required.

Calls the deterministic Python tools directly (bypassing the LangGraph/LLM wrapper),
so this runs even with zero API keys configured — useful to prove the underlying
system works before worrying about whether an LLM key is available for the full
narrated "Executive Market Cycle Brief" agent experience.

FMP_API_KEY / FRED_API_KEY are optional: if unset, valuation/liquidity fields report
FALLBACK_BENCHMARK / insufficient_data honestly instead of failing. Price structure
(ZigZag/STC/trend) and the earnings-momentum basket work with zero keys (yfinance only).

Usage:
    python scripts/demo_cycle_agent.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shankh.agents.market.cycle_tools import (
    get_cycle_price_structure,
    get_market_cycle_metrics,
    get_liquidity_and_credit_cycle,
    get_index_earnings_momentum,
    get_market_cycle_synthesis,
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    import os

    section("ENVIRONMENT CHECK")
    for key in ("FMP_API_KEY", "FRED_API_KEY"):
        status = "SET" if os.getenv(key) else "NOT SET (fallback/insufficient_data paths will be used)"
        print(f"  {key}: {status}")
    print("  OPENAI_API_KEY is NOT required for this script — it calls the tools directly, no LLM.")

    section("1. CORE PRICE STRUCTURE (ZigZag / STC / Trend) — zero API keys needed")
    print(get_cycle_price_structure.invoke({}))

    section("2. VALUATION & ERP")
    print(get_market_cycle_metrics.invoke({}))

    section("3. LIQUIDITY")
    print(get_liquidity_and_credit_cycle.invoke({}))

    section("4. EARNINGS MOMENTUM (basket proxy)")
    print(get_index_earnings_momentum.invoke({}))

    section("5. FULL SYNTHESIS — the primary output")
    synthesis_raw = get_market_cycle_synthesis.invoke({})
    synthesis = json.loads(synthesis_raw)
    print(synthesis_raw)

    section("EXECUTIVE SUMMARY (what the LLM layer would narrate)")
    print(f"  Cycle Phase:      {synthesis.get('cycle_phase')}")
    print(f"  Confidence:       {synthesis.get('cycle_confidence')}")
    print(f"  Transition Risk:  {synthesis.get('transition_risk')} ({synthesis.get('transition_watch')})")
    print(f"  Composite Score:  {synthesis.get('composite_score')}")
    print("  Evidence:")
    for item in synthesis.get("evidence", []):
        print(f"    - [{item['tier']:>10}] {item['signal']:<18} score={item['score']:+.2f}  {item['reading']}")


if __name__ == "__main__":
    main()
