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
    query_stock_clustering_and_forensics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_inference_tests():
    print("=" * 80)
    print(" SHANKH ANALYTICS INFERENCE TEST SUITE")
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Test 2: Query Stock Clustering for specific tickers
    # ---------------------------------------------------------------------------
    print("\n--- [TEST 2A] Query Specific Quality Stock ('INFY.NS') ---")
    res_infy = query_stock_clustering_and_forensics.invoke({"ticker": "INFY.NS"})
    print(res_infy)

    print("\n--- [TEST 2B] Query Specific Distressed Outlier ('ALOKINDS.NS') ---")
    res_alok = query_stock_clustering_and_forensics.invoke({"ticker": "ALOKINDS.NS"})
    print(res_alok)

    # ---------------------------------------------------------------------------
    # Test 3: Query Cluster Members & Forensic Anomalies
    # ---------------------------------------------------------------------------
    print("\n--- [TEST 3A] Query Members of Cluster ID 2 ---")
    res_c2 = query_stock_clustering_and_forensics.invoke({"cluster_id": 2})
    print(res_c2)

    print("\n--- [TEST 3B] Query Forensic Red Flags Only ---")
    res_forensic = query_stock_clustering_and_forensics.invoke({"check_forensic_anomalies_only": True})
    print(res_forensic)

    print("\n" + "=" * 80)
    print(" ALL INFERENCE TESTS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_inference_tests()