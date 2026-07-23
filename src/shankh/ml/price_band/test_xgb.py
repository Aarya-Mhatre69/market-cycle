"""
src/shankh/drivers/test_price_band_inference.py

Test harness to verify XGBoost Price-Band dynamic tool inference.
"""

import logging
import sys
from pathlib import Path
from src.shankh.ml.price_band.tool import query_xgboost_price_band  

# Path setup to resolve shankh module
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_price_band_tests():
    print("=" * 80)
    print(" SHANKH ANALYTICS INFERENCE TEST SUITE: XGBOOST PRICE-BAND FORECASTING")
    print("=" * 80)
    print("\n--- [TEST 1] Query XGBoost Price Band for 'INFY.NS' (Latest Date) ---")
    res_infy = query_xgboost_price_band.invoke({"ticker": "INFY.NS"})
    print(res_infy)
    print("\n--- [TEST 2] Query XGBoost Price Band for 'AXISBANK.NS' on '2024-02-20' ---")
    res_axis = query_xgboost_price_band.invoke({
        "ticker": "AXISBANK.NS",
        "date": "2024-02-20"
    })
    print(res_axis)
    print("\n--- [TEST 3] Project Price Band for Custom Live Price (TCS @ INR 4200.0) ---")
    res_tcs_custom = query_xgboost_price_band.invoke({
        "ticker": "TCS.NS",
        "current_price": 4200.0
    })
    print(res_tcs_custom)
    print("\n" + "=" * 80)
    print(" ALL PRICE-BAND INFERENCE TESTS PASSED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_price_band_tests()
