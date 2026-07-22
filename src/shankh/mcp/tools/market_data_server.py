"""
shankh/mcp/market_data_server.py
Mock MCP server exposing market-data tools to the financial advisor agent.
"""

import json
import random
import sys
from datetime import date, timedelta

from fastmcp import FastMCP

mcp = FastMCP(
    name="shankh-market-data",
    instructions=(
        "Provides mock Indian equity market data: indices, macro indicators, "
        "sector performance, stock quotes, and regime signals."
    ),
)

_RNG = random.Random(42)


def _jitter(base: float, pct: float = 0.02) -> float:
    return round(base * (1 + _RNG.uniform(-pct, pct)), 4)


def _today() -> str:
    return date.today().isoformat()


def market_snapshot() -> dict:
    """Broad Indian market indices, India VIX, and advance/decline for today."""
    return {
        "date": _today(),
        "indices": {
            "NIFTY_50": {
                "level": _jitter(25500),
                "change_pct": _jitter(0.42, 0.5),
            },
            "SENSEX": {
                "level": _jitter(83500),
                "change_pct": _jitter(0.38, 0.5),
            },
            "BANK_NIFTY": {
                "level": _jitter(57500),
                "change_pct": _jitter(0.61, 0.6),
            },
            "FINNIFTY": {
                "level": _jitter(26500),
                "change_pct": _jitter(0.29, 0.5),
            },
            "MIDCPNIFTY": {
                "level": _jitter(14300),
                "change_pct": _jitter(0.47, 0.6),
            },
        },
        "india_vix": round(_jitter(13.8, 0.2), 2),
        "adv_decline": {
            "advances": int(_jitter(1450, 0.15)),
            "declines": int(_jitter(950, 0.15)),
            "unchanged": int(_jitter(120, 0.20)),
        },
        "top_movers": {
            "gainers": ["RELIANCE", "HDFCBANK", "ICICIBANK"],
            "losers": ["TCS", "INFY", "LT"],
        },
        "note": "Mock Indian market data feed.",
    }

def macro_indicators() -> dict:
    """RBI rates, G-sec yield, USD/INR, crude, CPI, IIP, FII/DII flows."""
    return {
        "date": _today(),
        "rates": {
            "rbi_repo_rate_pct":    6.50,
            "gsec_10yr_yield_pct":  _jitter(7.08, 0.02),
            "overnight_call_pct":   _jitter(6.52, 0.01),
        },
        "fx": {
            "usd_inr":               _jitter(83.65, 0.005),
            "usd_inr_1m_change_pct": _jitter(-0.12, 2.0),
        },
        "commodities": {
            "brent_crude_usd":   _jitter(78.40, 0.03),
            "gold_inr_per_10g":  _jitter(72500, 0.02),
        },
        "macro_prints": {
            "cpi_yoy_pct":  _jitter(4.83, 0.05),
            "iip_yoy_pct":  _jitter(5.20, 0.10),
            "gdp_q_yoy_pct": 7.00,
        },
        "flows_30d_cr_inr": {
            "fii_net": _jitter(4200,  0.20),
            "dii_net": _jitter(12500, 0.15),
        },
        "note": "Mock indicators.",
    }


def sector_performance(period: str = "1M") -> dict:
    """Sector returns, relative strength vs NIFTY, and % above 20-DMA."""
    valid = {"1D", "1W", "1M", "3M", "6M", "1Y"}
    if period not in valid:
        return {"error": f"Invalid period '{period}'."}

    base_returns = {
        "IT": 3.2, "BANK": 1.8, "AUTO": 4.1, "PHARMA": 2.7, "FMCG": 0.9,
        "METAL": -1.2, "ENERGY": 2.0, "REALTY": 5.8, "INFRA": 3.5, "MEDIA": -0.4,
    }
    nifty_ret = _jitter(2.1, 0.3)
    sectors   = {}
    for s, base in base_returns.items():
        ret = _jitter(base, 0.15)
        sectors[s] = {
            "return_pct":         round(ret, 2),
            "relative_to_nifty":  round(ret - nifty_ret, 2),
            "pct_above_20dma":    round(_jitter(62, 0.20), 1),
            "adv_decline_ratio":  round(_jitter(1.8, 0.30), 2),
        }
    return {
        "date": _today(),
        "period": period,
        "nifty50_return_pct": round(nifty_ret, 2),
        "sectors": sectors,
        "note": "Mock sector metrics.",
    }


def stock_quote(symbol: str) -> dict:
    """LTP, day change, 52-week range, P/E, market cap, promoter holding."""
    symbol = symbol.upper().strip()
    _bases = {
        "TCS":      {"ltp": 3800, "pe": 28.5, "mcap_cr": 138000, "promoter_pct": 72.4},
        "RELIANCE": {"ltp": 2950, "pe": 22.1, "mcap_cr": 198000, "promoter_pct": 50.3},
        "INFY":     {"ltp": 1620, "pe": 24.3, "mcap_cr":  67000, "promoter_pct": 14.9},
        "HDFCBANK": {"ltp": 1780, "pe": 18.7, "mcap_cr": 134000, "promoter_pct": 25.6},
        "WIPRO":    {"ltp":  545, "pe": 20.1, "mcap_cr":  28000, "promoter_pct": 72.9},
        "ITC":      {"ltp":  465, "pe": 26.8, "mcap_cr":  58000, "promoter_pct":  0.0},
    }
    base      = _bases.get(symbol, {"ltp": _jitter(1000, 0.5), "pe": _jitter(22, 0.2),
                                     "mcap_cr": _jitter(5000, 0.5), "promoter_pct": _jitter(45, 0.1)})
    ltp       = _jitter(base["ltp"], 0.015)
    prev      = ltp / (1 + _jitter(0.005, 1.0))
    return {
        "symbol": symbol, "exchange": "NSE", "date": _today(),
        "ltp": round(ltp, 2), "prev_close": round(prev, 2),
        "change_pct": round((ltp / prev - 1) * 100, 2),
        "52w_high": round(ltp * _jitter(1.32, 0.05), 2),
        "52w_low":  round(ltp * _jitter(0.72, 0.05), 2),
        "volume":   int(_jitter(3_500_000, 0.40)),
        "fundamentals": {
            "pe_ratio":      round(_jitter(base["pe"], 0.05), 1),
            "market_cap_cr": int(_jitter(base["mcap_cr"], 0.03)),
            "promoter_pct":  round(base["promoter_pct"], 1),
        },
        "note": "Mock fundamentals.",
    }


def market_regime_signal() -> dict:
    """Latest regime classification with confidence and contributing signals."""
    regimes = ["risk-on", "risk-off", "range-bound",
               "liquidity-driven", "sector-rotation", "high-volatility"]
    weights = [0.30, 0.15, 0.20, 0.15, 0.15, 0.05]
    regime  = _RNG.choices(regimes, weights=weights)[0]
    return {
        "date":   _today(),
        "regime": regime,
        "confidence": round(_jitter(0.72, 0.10), 3),
        "contributing_signals": {
            "macro_score":    round(_jitter(0.65, 0.15), 3),
            "breadth_score":  round(_jitter(0.70, 0.15), 3),
            "cycle_label":    _RNG.choice(["expansion", "contraction", "recovery", "slowdown"]),
            "vix_percentile": round(_jitter(35, 0.30), 1),
        },
        "last_regime_change": (date.today() - timedelta(days=int(_jitter(12, 0.5)))).isoformat(),
        "note": "Mock signal.",
    }


@mcp.tool()
def get_market_snapshot() -> str:
    """Broad snapshot of US equity market indices, VIX, and A/D ratio."""
    return json.dumps(market_snapshot(), indent=2)


@mcp.tool()
def get_macro_indicators() -> str:
    """Key Indian macro indicators: rates, FX, crude, CPI, IIP, FII/DII flows."""
    return json.dumps(macro_indicators(), indent=2)


@mcp.tool()
def get_sector_performance(period: str = "1M") -> str:
    """Sector-level returns, relative strength vs NIFTY, and % above 20-DMA."""
    return json.dumps(sector_performance(period), indent=2)


@mcp.tool()
def get_stock_quote(symbol: str) -> str:
    """Real-time quote and basic fundamentals for a single NSE stock."""
    return json.dumps(stock_quote(symbol), indent=2)


@mcp.tool()
def get_market_regime_signal() -> str:
    """Latest market regime classification with confidence and contributing signals."""
    return json.dumps(market_regime_signal(), indent=2)


if __name__ == "__main__":
    transport = "stdio"
    port      = 8765
    args      = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
        if arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])

    if transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")