## 1. Supervisor / Financial Advisor

 
### Tools

* **Web search**: for fresh news, events, policy changes, earnings headlines.
* **Market snapshot MCP**: broad market state.
* **Macro indicators MCP**: RBI, inflation, yields, oil, USD/INR, flows.
* **Sector performance MCP**: rotation and breadth.
* **Stock quote MCP**: single-name context.
* **Regime signal MCP**: quick regime label.
* **Macro analyst sub-agent**
* **Regime analyst sub-agent**

### Data sources

* NSE / BSE market data
* RBI / MOSPI / trading economics style macro feeds
* FII/DII flow data
* Reuters / business news / company announcements
* Internal regime outputs

### Responsibility

* Decide which tool or sub-agent to call
* Merge outputs into one research response
* Keep the answer structured: data → interpretation → caveats

### Should not have

* Raw database write access
* Trading execution APIs
* Portfolio management APIs
* Heavy feature engineering or model-training tools

---

## 2. Macro Analyst

This agent should be focused on **economy, liquidity, rates, inflation, global spillovers**.

### Tools

* **Macro indicators MCP**
* **Web search**
* **Economic calendar API**
* **Bond/yield curve API**
* **Currency API**
* **Commodity API** for crude, gold, industrial metals
* **Central bank / policy document fetchers**
* **News summarization tools**
* Optional: **country comparison / global macro screener**

### Data sources

* RBI repo rate, CPI, WPI, IIP, credit growth
* G-sec yields and yield curve
* USD/INR
* Brent crude
* FII/DII flows
* Fed / ECB / BOJ / global rate decisions
* IMF / World Bank / OECD reports
* Macro news and policy releases

### Outputs it should produce

* Growth regime
* Inflation regime
* Liquidity regime
* Rates / bond interpretation
* Risk-on vs risk-off view
* Sector implications

### Should not have

* Intraday microstructure data
* Trade execution
* Individual stock fundamentals unless needed for macro context

---

## 3. Regime Analyst

This agent should classify the **current market state**.

### Tools

* **Market snapshot MCP**
* **Sector performance MCP**
* **Breadth data MCP** if available
* **Volatility data MCP**
* **Macro indicators MCP**
* **Web search** for context around shocks or policy events
* **change-point / regime model API**
* **HMM / clustering inference tool**

### Data sources

* Index levels and returns
* India VIX
* Advance/decline breadth
* % above key moving averages
* Sector leadership
* FII/DII flows
* Rates, oil, USD/INR
* Event calendar and major headlines

### Typical regime labels

* Bull trend
* Bear trend
* Sideways / range-bound
* High volatility
* Low volatility
* Risk-on
* Risk-off
* Defensive rotation
* Liquidity-driven rally

### What it should answer

* What regime are we in?
* How confident is the classification?
* What evidence supports that?
* What regime changes are emerging?

### Should not have

* Unbounded stock-picking tools
* Order management
* Portfolio optimization unless explicitly added

---

## 4. Market Data MCP Server

This is not really an “agent”; it is a **tool provider**.

### Good MCP functions

* `market_snapshot()`
* `macro_indicators()`
* `sector_performance(period)`
* `stock_quote(symbol)`
* `market_regime_signal()`
* `breadth_snapshot()`
* `volatility_snapshot()`
* `index_history(symbol, timeframe)`
* `fundamental_snapshot(symbol)`

### Data sources behind MCP

* Exchange data
* Broker or vendor market feeds
* Macro data providers
* Internal computed indicators
* Cached daily snapshots

### Why MCP is useful

* Standard interface
* Reusable by multiple agents
* Keeps data access separate from reasoning
* Easier to test and mock

---

## 5. Web Search Tool

This should be a shared external-research tool.

### Use for

* Breaking news
* RBI announcements
* Earnings surprises
* Regulatory changes
* Corporate actions
* Analyst commentary
* Geopolitical events affecting markets

### Best practice

* Give it to the supervisor
* Optionally allow macro/regime agents to call it
* Do not let every sub-agent freely browse without need

---

## 6. Optional specialized agents you may add later

### A. Stock Research Agent

Tools:

* stock quote
* company fundamentals
* earnings transcripts
* analyst reports
* filings / announcements
* sector comparison

Purpose:

* single-name deep dive

---

### B. Earnings Agent

Tools:

* quarterly results API
* transcript fetcher
* guidance parser
* estimate revision tracker

Purpose:

* earnings quality, surprise, margins, guidance

---

### C. Portfolio Risk Agent

Tools:

* exposure calculator
* correlation matrix
* drawdown tracker
* factor exposure API
* scenario stress test

Purpose:

* portfolio-level risk, not market narrative

---

### D. Event Impact Agent

Tools:

* news search
* event calendar
* historical event-response lookup

Purpose:

* estimate impact of RBI, CPI, budget, Fed, election, oil shock

---

## Practical split recommendation

If you are building this cleanly, use:

### Financial Advisor

* web search
* all MCP read-only market tools
* macro analyst sub-agent
* regime analyst sub-agent

### Macro Analyst

* macro APIs
* bond/currency/commodity feeds
* web search
* no trading tools

### Regime Analyst

* market snapshot
* breadth
* volatility
* sector rotation
* macro indicators
* optional statistical regime model

### MCP layer

* pure data access only
* no reasoning
* no prompt logic

---

## Simple rule

* **Supervisor** decides.
* **Specialists** analyze.
* **MCP/APIs** fetch data.
* **Models** infer regime or forecast.
* **Nothing should mix fetching, reasoning, and execution in one place** unless the system is very small.
