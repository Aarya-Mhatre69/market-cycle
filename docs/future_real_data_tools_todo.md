# Future Real Data Tools TODO

The mock market-data tools and mock market-data MCP server were removed. Reintroduce these capabilities only when they are backed by real data sources, clear update semantics, and tests.

## Market Data Feed

- Implement a production `fetch_market_snapshot()` backed by a real market data vendor or exchange-approved feed.
- Include NIFTY/SENSEX/BANK NIFTY levels, India VIX, advance/decline, turnover, source, and timestamp metadata.
- Add stale-data detection and explicit market-holiday handling.
- Add unit tests with recorded fixtures and opt-in live integration tests.

## Macro Data Feed

- Implement a production `fetch_macro_indicators()` backed by authoritative sources such as RBI, MOSPI, exchange FII/DII reports, commodity/FX providers, or a licensed aggregator.
- Track repo rate, CPI/WPI, bond yields, USDINR, crude, FII/DII flows, and release dates.
- Add source attribution and last-updated timestamps for every field.

## Sector Data Feed

- Implement a production `fetch_sector_performance(period)` using real sector/index constituents and live or end-of-day prices.
- Define supported periods and benchmark methodology.
- Include validation for partial trading days and missing constituents.

## Stock Quote Feed

- Implement a production `fetch_stock_quote(symbol)` with real quote, volume, corporate-action-adjusted 52-week range, and fundamental snapshot.
- Normalize NSE/BSE ticker formats.
- Separate live quotes from delayed/end-of-day data in the response.

## Market Signal Service

- Replace the removed mock `fetch_market_signal()` with a real service that either wraps `query_market_regime()` artifacts or a production online inference pipeline.
- Include model version, artifact timestamp, input data window, confidence, and failure modes.

## MCP Server

- Rebuild the market-data MCP server only after real data tools exist.
- Keep the server as a thin transport layer over production tool functions.
- Add authentication/configuration for vendor credentials.
- Add health checks that verify source connectivity without returning synthetic values.
