"""
Universe-Wide Market Cycle Scoring — accuracy-roadmap Phase 6 (universe expansion),
started ahead of the roadmap's own sequencing at explicit user request, after Phases
0-5's accuracy work was validated on the Nifty 50 index.

Applies the SAME validated classification mechanism (cycle_signals.classify_cycle /
classify_cycle_stateful — axis-vote + hysteresis, accuracy-roadmap Phase 2) to every
NSE-listed equity individually, instead of only the Nifty 50 index. This is a
cross-sectional SNAPSHOT view (each stock's CURRENT phase, one point in time) —
deliberately not a full historical walk-forward backtest per stock, which would be a
much larger undertaking (2,500+ tickers x hundreds of evaluation points each, plus its
own independent-turn-labeling validation methodology per stock). That remains a
natural next step, not attempted here.

Data sources, all free / no paid API required:
- Ticker universe: nsepython.nse_eq_symbols() — NSE's own equity symbol list. NSE's
  archive CSV endpoints (e.g. archives.nseindia.com) return 403/404 to direct
  requests (confirmed while building this) — nsepython handles the session/header
  dance NSE's site requires, so it's used here rather than a hand-rolled scraper.
- OHLCV: yfinance batch download (yf.download with group_by='ticker'), not one
  Ticker().history() call per stock — ~50-100x fewer HTTP round trips.
- Fundamentals (P/E, P/B, dividend yield, earnings growth, sector): yfinance
  per-ticker .info, threaded. This sidesteps the FMP_API_KEY issue found during
  Phase 4 (it returns 401 Invalid API KEY on every endpoint in this environment) —
  individual-company fundamentals are available directly from yfinance without FMP.
- Liquidity (M3 growth) and India 10Y G-Sec yield: same macro-level values used for
  the index (cycle_tools.get_liquidity_and_credit_cycle / get_market_cycle_metrics),
  shared across every stock — liquidity conditions are economically a market-wide
  condition, not a per-stock one, so this is not a limitation to work around.

Valuation percentile is CROSS-SECTIONAL here (this stock's P/E vs. every other
stock's P/E, right now), not the index's TIME-SERIES percentile (this stock's P/E vs.
its own history) — these are genuinely different metrics answering different
questions, and are named accordingly (`pe_percentile_vs_universe`, not
`pe_10y_historical_percentile`) so they are never confused with each other, after
Phase 4 already found and fixed exactly that kind of naming mismatch for the index.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from shankh.agents.market.cycle_signals import (
    EvidenceItem,
    HysteresisState,
    classify_cycle_stateful,
    compute_candlestick_patterns,
    compute_gann_time_cycles,
    compute_harmonic_patterns,
    compute_stc,
    compute_trend_context,
    compute_zigzag,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "data" / "universe_cycle"
_LATEST_CSV = _DATA_DIR / "latest.csv"
_STATE_FILE = _DATA_DIR / "hysteresis_state.json"

_HYSTERESIS_MIN_DWELL = 5  # same calibrated value as the index (scripts/calibrate_min_dwell.py)
_MIN_HISTORY_BARS = 220     # >= 200DMA + small margin; stocks with less are skipped, not faked
_OHLCV_PERIOD = "2y"
_OHLCV_BATCH_SIZE = 75
_FUNDAMENTALS_MAX_WORKERS = 8  # lowered from an initial 15 after a full-universe run
# showed Yahoo's crumb-based auth starting to 401 under sustained concurrent load
# past ~1,000 tickers (fine up to a 300-ticker sample) — see _fetch_one_fundamentals.


def fetch_nse_universe() -> List[str]:
    """Full NSE-listed equity symbol list (yfinance-format, '.NS' suffix)."""
    import nsepython  # local import: only needed for this batch-build path

    symbols = nsepython.nse_eq_symbols()
    return [f"{s}.NS" for s in symbols]


def batch_fetch_ohlcv(tickers: List[str], period: str = _OHLCV_PERIOD,
                       batch_size: int = _OHLCV_BATCH_SIZE) -> Dict[str, pd.DataFrame]:
    """Batched OHLCV download (yf.download, not per-ticker Ticker().history() calls —
    ~50-100x fewer HTTP round trips, tested at ~9s per 50 tickers)."""
    result: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i: i + batch_size]
        try:
            data = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                                threads=True, progress=False, auto_adjust=False)
        except Exception as exc:
            logger.warning("Batch OHLCV download failed for chunk starting %s: %s", chunk[0], exc)
            continue

        for t in chunk:
            try:
                sub = data[t] if len(chunk) > 1 else data
                sub = sub.dropna(subset=["Close"])
                if len(sub) < _MIN_HISTORY_BARS:
                    continue
                df = sub.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df.columns = ["date", "open", "high", "low", "close", "volume"]
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                result[t] = df.sort_values("date").reset_index(drop=True)
            except Exception:
                continue
        logger.info("OHLCV batch %d-%d: %d/%d tickers usable so far.",
                    i, i + len(chunk), len(result), i + len(chunk))
    return result


def _coerce_float(value: Any) -> Optional[float]:
    """yfinance's .info occasionally returns a numeric field as a string (or other
    non-numeric junk) for certain tickers — found by running this at full-universe
    scale, not visible in small samples. Never let a bad type crash the pipeline;
    treat un-coercible values the same as missing data."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fetch_one_fundamentals(ticker: str, retries: int = 2, retry_delay: float = 1.5) -> Optional[Dict[str, Any]]:
    """Retries on failure — at full-universe scale (~2000+ tickers under sustained
    concurrent load), Yahoo's session/crumb auth starts intermittently returning
    401 Invalid Crumb (a known yfinance/Yahoo rate-limiting behavior, not specific to
    any ticker); a short backoff and retry recovers most of these rather than losing
    the ticker outright."""
    import time

    last_exc = None
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).get_info()
            return {
                "ticker": ticker,
                "company_name": info.get("shortName") or info.get("longName") or ticker,
                "sector": info.get("sector") or "Unknown",
                "pe_ratio": _coerce_float(info.get("trailingPE")),
                "pb_ratio": _coerce_float(info.get("priceToBook")),
                "dividend_yield_percent": _coerce_float(info.get("dividendYield")) or 0.0,
                "earnings_quarterly_growth": _coerce_float(info.get("earningsQuarterlyGrowth")),
                "current_price": _coerce_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            }
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_delay)
    logger.debug("Fundamentals fetch failed for %s after %d attempts: %s", ticker, retries + 1, last_exc)
    return None


def batch_fetch_fundamentals(tickers: List[str], max_workers: int = _FUNDAMENTALS_MAX_WORKERS) -> Dict[str, dict]:
    """Threaded per-ticker .info fetch. Tested clean (0 errors) up to a 300-ticker
    sample; at full-universe scale (~2000+ tickers), Yahoo's crumb-based auth starts
    intermittently 401'ing under sustained concurrent load — _fetch_one_fundamentals
    retries each ticker up to twice with a backoff before giving up on it, which
    recovers most of these. A conservative default worker count (see
    _FUNDAMENTALS_MAX_WORKERS) trades some throughput for fewer 401s in the first
    place, rather than relying on retries alone."""
    result: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one_fundamentals, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                result[r["ticker"]] = r
            if i % 200 == 0:
                logger.info("Fundamentals: %d/%d tickers processed (%d usable).", i, len(tickers), len(result))
    return result


def _load_state_store() -> Dict[str, dict]:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load universe hysteresis state (%s); starting fresh.", exc)
    return {}


def _save_state_store(store: Dict[str, dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def classify_one_stock(
    ticker: str,
    ohlcv: pd.DataFrame,
    fundamentals: Optional[dict],
    liquidity_score: Optional[float],
    india_10y_yield: Optional[float],
    pe_percentile_vs_universe: Optional[float],
    prior_state: Optional[dict],
) -> dict:
    """Builds the same EvidenceItem set the index uses (cycle_tools.get_market_cycle_synthesis),
    per stock, and runs it through the identical hysteresis-confirmed classifier."""
    zz = compute_zigzag(ohlcv)
    stc = compute_stc(ohlcv["close"])
    trend = compute_trend_context(ohlcv["close"])
    candles = compute_candlestick_patterns(ohlcv)
    harmonic = compute_harmonic_patterns(ohlcv)
    gann = compute_gann_time_cycles(ohlcv)

    evidence: List[EvidenceItem] = [
        EvidenceItem("zigzag", "core", zz.structure, zz.score, zz.note),
        EvidenceItem("stc", "core", f"{stc.value} ({stc.direction})", stc.score, stc.note),
        EvidenceItem("trend_context", "core", f"{trend.price_vs_sma200_pct:+.1f}% vs 200DMA", trend.score, trend.note),
    ]
    if abs(candles.score) > 1e-9:
        evidence.append(EvidenceItem("candlestick", "contextual", candles.note, candles.score, candles.note))
    if harmonic.pattern:
        evidence.append(EvidenceItem("harmonic", "contextual", harmonic.note, harmonic.score, harmonic.note))
    if abs(gann.score) > 1e-9:
        evidence.append(EvidenceItem("gann", "contextual", gann.note, gann.score, gann.note))

    pe_ratio = fundamentals.get("pe_ratio") if fundamentals else None
    if pe_ratio and pe_ratio > 0:
        earnings_yield = (1.0 / pe_ratio) * 100.0
        if india_10y_yield is not None:
            erp_percent = earnings_yield - india_10y_yield
            erp_score = float(np.clip(erp_percent / 2.0, -1.0, 1.0))
            evidence.append(EvidenceItem("erp", "supporting", f"{erp_percent:+.2f}%", erp_score,
                                          "Earnings Yield minus India 10Y G-Sec Yield."))

    earnings_growth = fundamentals.get("earnings_quarterly_growth") if fundamentals else None
    if earnings_growth is not None:
        growth_pct = earnings_growth * 100.0
        earnings_score = float(np.clip(growth_pct / 20.0, -1.0, 1.0))
        evidence.append(EvidenceItem("earnings_momentum", "supporting", f"{growth_pct:+.1f}% YoY", earnings_score,
                                      "yfinance quarterly earnings growth for this ticker."))

    if liquidity_score is not None:
        evidence.append(EvidenceItem("liquidity", "supporting", "shared macro M3 reading", liquidity_score,
                                      "Same market-wide M3 growth reading used for the index."))

    state = HysteresisState(
        confirmed_phase=prior_state.get("confirmed_phase") if prior_state else None,
        candidate_phase=prior_state.get("candidate_phase") if prior_state else None,
        candidate_count=prior_state.get("candidate_count", 0) if prior_state else 0,
    )
    result, new_state = classify_cycle_stateful(evidence, state, min_dwell=_HYSTERESIS_MIN_DWELL)

    return {
        "ticker": ticker,
        "company_name": (fundamentals or {}).get("company_name", ticker),
        "sector": (fundamentals or {}).get("sector", "Unknown"),
        "as_of_date": str(ohlcv["date"].iloc[-1].date()),
        "close": float(ohlcv["close"].iloc[-1]),
        "cycle_phase": result.cycle_phase,
        "cycle_confidence": result.cycle_confidence,
        "transition_risk": result.transition_risk,
        "pending_phase": result.pending_phase,
        "dwell_progress": result.dwell_progress,
        "composite_score": result.composite_score,
        "zigzag_score": zz.score,
        "stc_score": stc.score,
        "trend_score": trend.score,
        "candlestick_score": candles.score,
        "harmonic_pattern": harmonic.pattern or "",
        "gann_score": gann.score,
        "pe_ratio": pe_ratio,
        "pe_percentile_vs_universe": pe_percentile_vs_universe,
        "pb_ratio": (fundamentals or {}).get("pb_ratio"),
        "dividend_yield_percent": (fundamentals or {}).get("dividend_yield_percent"),
        "earnings_quarterly_growth_percent": (earnings_growth * 100.0) if earnings_growth is not None else None,
        "_new_state": {
            "confirmed_phase": new_state.confirmed_phase,
            "candidate_phase": new_state.candidate_phase,
            "candidate_count": new_state.candidate_count,
        },
    }
