"""
Builds the cross-sectional Universe Cycle dataset — one current cycle-phase read per
NSE-listed stock, using the same axis-vote + hysteresis mechanism validated for the
Nifty 50 index (accuracy-roadmap Phases 0-5), applied individually to every ticker.

This is a batch/offline build, not a live per-request tool: fetching + classifying
2,500+ tickers takes several minutes (OHLCV batch download ~9s/50 tickers, threaded
fundamentals ~60ms/ticker), so results are cached to data/universe_cycle/latest.csv
and the dashboard reads that cache rather than recomputing on every page load.

Usage:
    python scripts/build_universe_cycle_data.py                 # full NSE universe
    python scripts/build_universe_cycle_data.py --limit 200      # smaller test run
    python scripts/build_universe_cycle_data.py --limit 100 --workers 10
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shankh.agents.market.cycle_tools import (  # noqa: E402
    get_liquidity_and_credit_cycle,
    get_market_cycle_metrics,
)
from shankh.agents.market.universe_cycle import (  # noqa: E402
    _DATA_DIR,
    _LATEST_CSV,
    batch_fetch_fundamentals,
    batch_fetch_ohlcv,
    classify_one_stock,
    fetch_nse_universe,
    _load_state_store,
    _save_state_store,
)

import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tickers (for testing).")
    parser.add_argument("--workers", type=int, default=8, help="Threads for fundamentals fetch (see universe_cycle._FUNDAMENTALS_MAX_WORKERS).")
    parser.add_argument("--batch-size", type=int, default=75, help="Tickers per yf.download batch.")
    args = parser.parse_args()

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching NSE equity universe list...")
    tickers = fetch_nse_universe()
    if args.limit:
        tickers = tickers[: args.limit]
    logger.info("Universe: %d tickers.", len(tickers))

    logger.info("Fetching shared macro readings (liquidity, India 10Y yield)...")
    liquidity_raw = json.loads(get_liquidity_and_credit_cycle.invoke({}))
    valuation_raw = json.loads(get_market_cycle_metrics.invoke({}))
    liquidity_score = None
    if liquidity_raw.get("credit_cycle_status") != "insufficient_data":
        liquidity_score = liquidity_raw["liquidity_score"]
    india_10y_yield = valuation_raw["equity_risk_premium"]["india_10y_gsec_yield_percent"]
    logger.info("liquidity_score=%s, india_10y_yield=%s (source=%s)",
                liquidity_score, india_10y_yield,
                valuation_raw["equity_risk_premium"]["india_10y_gsec_data_source"])

    logger.info("Batch-downloading OHLCV (this is the slow step for a full-universe run)...")
    ohlcv_by_ticker = batch_fetch_ohlcv(tickers, batch_size=args.batch_size)
    logger.info("OHLCV usable for %d/%d tickers.", len(ohlcv_by_ticker), len(tickers))

    usable_tickers = list(ohlcv_by_ticker.keys())
    logger.info("Fetching fundamentals for %d tickers (threaded)...", len(usable_tickers))
    fundamentals_by_ticker = batch_fetch_fundamentals(usable_tickers, max_workers=args.workers)
    logger.info("Fundamentals usable for %d/%d tickers.", len(fundamentals_by_ticker), len(usable_tickers))

    # Cross-sectional P/E percentile: this stock's P/E vs. every other stock's P/E,
    # right now — NOT a historical time-series percentile (see universe_cycle.py's
    # module docstring for why these are kept as two clearly-different metrics).
    # isinstance guard is defense-in-depth: _fetch_one_fundamentals already coerces
    # pe_ratio to float-or-None, but a crash here at full-universe scale (a string
    # slipped through before that fix) is expensive to re-discover, so this stays
    # even though it should now be unreachable.
    valid_pes = sorted(
        f["pe_ratio"] for f in fundamentals_by_ticker.values()
        if isinstance(f.get("pe_ratio"), (int, float)) and f["pe_ratio"] > 0
    )
    logger.info("Cross-sectional P/E percentile base: %d tickers with a valid, positive P/E.", len(valid_pes))

    def _pe_percentile(pe):
        if not pe or pe <= 0 or not valid_pes:
            return None
        import numpy as np
        return round(float((np.array(valid_pes) <= pe).mean() * 100.0), 1)

    state_store = _load_state_store()

    rows = []
    for ticker in usable_tickers:
        ohlcv = ohlcv_by_ticker[ticker]
        fundamentals = fundamentals_by_ticker.get(ticker)
        pe_pctile = _pe_percentile(fundamentals.get("pe_ratio") if fundamentals else None)
        prior_state = state_store.get(ticker)
        try:
            row = classify_one_stock(
                ticker, ohlcv, fundamentals, liquidity_score, india_10y_yield, pe_pctile, prior_state,
            )
        except Exception as exc:
            logger.warning("Classification failed for %s: %s", ticker, exc)
            continue
        state_store[ticker] = row.pop("_new_state")
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(_LATEST_CSV, index=False)
    _save_state_store(state_store)
    logger.info("Saved %d classified tickers to %s", len(result_df), _LATEST_CSV)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Universe requested: {len(tickers)}  |  OHLCV usable: {len(ohlcv_by_ticker)}  |  "
          f"Fundamentals usable: {len(fundamentals_by_ticker)}  |  Classified: {len(result_df)}")
    if len(result_df):
        print("\nPhase distribution:")
        print(result_df["cycle_phase"].value_counts(normalize=True).round(3).to_string())
        print("\nSector coverage (top 10):")
        print(result_df["sector"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
