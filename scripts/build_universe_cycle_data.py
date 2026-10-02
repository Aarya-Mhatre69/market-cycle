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
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shankh.agents.market.cycle_tools import (  # noqa: E402
    _get_fmp_api_key,
    get_liquidity_and_credit_cycle,
    get_market_cycle_metrics,
)
from shankh.agents.market.universe_cycle import (  # noqa: E402
    _DATA_DIR,
    _LATEST_CSV,
    append_phase_changes,
    detect_phase_change,
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

# Brutal-review audit finding #9: the classification loop below used to only write
# latest.csv/hysteresis_state.json once, after EVERY ticker finished — a crash or
# the scheduled CI job's 30-minute timeout partway through lost the whole run's
# already-fetched, already-classified data with nothing durable to show for it.
# Checkpointing here doesn't protect the earlier OHLCV/fundamentals fetch stages
# (batch_fetch_ohlcv/batch_fetch_fundamentals, the slower, network-bound, more
# failure-prone part of a run) — those still need to fully return before this loop
# starts — but it does mean a failure DURING this loop leaves a snapshot at most
# _CHECKPOINT_INTERVAL tickers stale instead of the entire run's work discarded.
_CHECKPOINT_INTERVAL = 300


def _write_checkpoint(rows: list, state_store: dict, events: list) -> None:
    """Atomic write (temp file + os.replace) so a crash mid-write never leaves a
    truncated/corrupt latest.csv for the dashboard to read. Phase-change events are
    saved with the state store they were derived from: if they were not, a crash after
    the state advanced would make those flips undetectable on the next run."""
    tmp_path = _LATEST_CSV.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, _LATEST_CSV)
    _save_state_store(state_store)
    append_phase_changes(events)


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
    fmp_api_key = _get_fmp_api_key()
    logger.info("Fetching fundamentals for %d tickers (threaded, source=%s)...",
                len(usable_tickers), "FMP+yfinance fallback" if fmp_api_key else "yfinance only")
    fundamentals_by_ticker = batch_fetch_fundamentals(usable_tickers, max_workers=args.workers, fmp_api_key=fmp_api_key)
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
    events = []
    for i, ticker in enumerate(usable_tickers, 1):
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
        event = detect_phase_change(prior_state, row)
        if event:
            events.append(event)
        state_store[ticker] = row.pop("_new_state")
        rows.append(row)

        if i % _CHECKPOINT_INTERVAL == 0:
            _write_checkpoint(rows, state_store, events)
            logger.info("Checkpoint: %d/%d tickers classified, saved to %s.", i, len(usable_tickers), _LATEST_CSV)

    result_df = pd.DataFrame(rows)
    _write_checkpoint(rows, state_store, events)
    logger.info("Saved %d classified tickers to %s (%d phase changes this run)", len(result_df), _LATEST_CSV, len(events))

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
