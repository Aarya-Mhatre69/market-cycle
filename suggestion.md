## 1) Financial Advisor Agent

**Task:** supervisor / router / final response composer.

**Tools:**

* `web_search` — fetch current news or event context when the question depends on recent information.
* `query_market_regime()` — pull market regime output when the user asks about market state.
* `query_xgboost_price_band()` — pull stock forecast output when the user asks about a company/stock.
* `query_stock_peers()` — get similar stocks and cluster peers.
* `query_forensic_red_flags()` — get anomaly / red-flag summaries.

---

## 2) Macro Analyst Agent

**Task:** explain macro events and macro context in plain language.

**Tools:**

* `web_search` — search for RBI, inflation, rates, currency, crude, flows, or global event context.

**No real macro model exists yet**, so this agent is mainly search + reasoning.

---

## 3) Market Analyst Agent

**Task:** analyze market-wide condition: regime, breadth, volatility, rotation.

**Tools:**

* `query_market_regime()` — return the trained regime classification and related metrics.
* `web_search` — add current news context around the market move.
* `score_market_breadth()` — if this is only mock, keep it internal or remove from the exposed tool set.
* `classify_volatility_regime()` — same: only expose if it has real logic.
* `identify_sector_rotation()` — same: only expose if it has real logic.

If only `query_market_regime()` is real, then the other three should stay hidden or be treated as placeholders.

---

## 4) Company Analyst Agent

**Task:** analyze one stock/company.

**Tools:**

* `query_xgboost_price_band()` — get the forecast band for the stock.
* `query_stock_peers()` — compare the stock with similar names / cluster peers.
* `query_forensic_red_flags()` — check anomaly / red-flag status.
* `web_search` — fetch recent company news, results, filings, or commentary.

---

## Tool ownership summary

| Agent             | Real tools it should own                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| Financial Advisor | `web_search`, routes to others                                                                  |
| Macro Analyst     | `web_search`                                                                                    |
| Market Analyst    | `query_market_regime()`, `web_search`                                                           |
| Company Analyst   | `query_xgboost_price_band()`, `query_stock_peers()`, `query_forensic_red_flags()`, `web_search` |

---

## Tools I would keep out for now if they are mock

* `fetch_market_snapshot()`
* `fetch_macro_indicators()`
* `fetch_sector_performance()`
* `fetch_stock_quote()`
* `fetch_regime_signal()`
