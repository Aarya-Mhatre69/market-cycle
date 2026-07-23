"""
LangChain/LangGraph tool for market regime inference.
"""

import json
from langchain_core.tools import tool
from src.shankh.ml.market.inference import run_inference
from src.shankh.ml.market.data_loader import load_data
from src.shankh.ml.market.config import CONFIG

@tool
def get_market_regime(min_overlap_days: int = 500) -> str:
    """
    Fetch the current market regime using the HMM model.
    """
    try:
        df = load_data(
            data_dir=CONFIG["data"]["data_dir"],
            min_rows=CONFIG["data"]["min_rows"],
            min_tickers=CONFIG["data"]["min_tickers"],
            max_tickers=CONFIG["data"]["max_tickers"],
            min_overlap_days=min_overlap_days,
            required_cols=CONFIG["data"]["required_cols"]
        )
        
        results = run_inference(df)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error running regime inference: {str(e)}"
