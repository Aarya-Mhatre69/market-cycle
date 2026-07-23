"""
LangChain/LangGraph tool for stock clustering inference.
"""

import json
from langchain_core.tools import tool
from src.shankh.ml.macro.inference import run_inference
from src.shankh.ml.macro.data_loader import load_data
from src.shankh.ml.macro.config import CONFIG

@tool
def get_stock_clusters(min_overlap_days: int = 500) -> str:
    """
    Fetch stock clusters and anomaly predictions for the universe of stocks.
    Useful for identifying groupings of stocks and anomalies.
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
        return f"Error running clustering inference: {str(e)}"
