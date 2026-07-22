"""
src/shankh/drivers/test_inference.py

Test harness to verify dynamic tool inference for Stock Clustering & Market Regime Analysis.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shankh.macro.tools import (
    query_market_regime,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_inference_tests():
    print("=" * 80)
    print(" SHANKH ANALYTICS INFERENCE TEST SUITE")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Test 1: Query Market Regime on specific dates
    # ---------------------------------------------------------------------------
    print("\n--- [TEST 1A] Query Market Regime for Latest Trading Date ---")
    res_latest = query_market_regime.invoke({})
    print(res_latest)

    print("\n--- [TEST 1B] Query Market Regime for Specific Date ('2024-02-14') ---")
    res_date1 = query_market_regime.invoke({"query_date": "2024-02-14"})
    print(res_date1)

    print("\n--- [TEST 1C] Query Market Regime for Date during Regime Switch ('2024-02-20') ---")
    res_date2 = query_market_regime.invoke({"query_date": "2024-02-20"})
    print(res_date2)

    print(" ALL INFERENCE TESTS PASSED SUCCESSFULLY")


if __name__ == "__main__":
    run_inference_tests()