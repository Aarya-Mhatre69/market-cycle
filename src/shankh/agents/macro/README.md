# Shankh Macro Analyst Agent
`src/shankh/agents/macro/`

The **Shankh Macro Analyst Agent** is a specialized, domain-pure quantitative research engine designed for top-down Indian macroeconomic synthesis, institutional liquidity tracking, cross-asset yield analysis, and global market cue integration.

---

## 1. Executive Summary & Capabilities

### What It Does
* **Top-Down Macro Synthesis:** Evaluates domestic growth drivers, monetary policy stances, inflation dynamics, and currency movements.
* **Institutional Liquidity Tracking:** Monitors Foreign Institutional Investor (FII/FPI) and Domestic Institutional Investor (DII) cash net flows and directional market bias.
* **Cross-Asset Yield Spread Engine:** Dynamically calculates the **US-India 10Y Yield Differential** ($\text{IN10Y} - \text{US10Y}$) and the US $10\text{Y}-2\text{Y}$ Yield Curve Slope to flag capital flight risks.
* **Commodity & Forex Velocity Analysis:** Analyzes 5-day and 20-day Rate of Change (ROC %) and trend directional momentum for Brent Crude, WTI Crude, Gold, USD/INR, and the US Dollar Index (DXY).
* **Dual-Country Economic Calendar Integration:** Isolates high-impact upcoming economic releases from both **India (IN)** and the **United States (US)**.

---

## 2. When to Use the Macro Agent

| Use Case / Workflow | Agent Application |
| :--- | :--- |
| **Morning Market Briefs** | Synthesizing overnight US Fed cues, Crude Oil velocity, and FII flows before domestic market open. |
| **Asset Allocation Meetings** | Evaluating capital flight risks based on US-India yield spread compression ($<200 \text{ bps}$). |
| **RBI / FOMC Policy Reviews** | Analyzing interest rate trajectory transmission to domestic equity profit margins. |
| **Portfolio Risk Audits** | Assessing macro headwind risks (e.g., surging Brent Crude + depreciating Rupee). |

---

## 3. What It Can Answer vs. What It Cannot Answer

### What It Can Answer
* *"What is the current US-India 10Y yield spread and what does it imply for capital flows?"*
* *"How is recent Brent Crude oil price velocity impacting domestic inflation expectations?"*
* *"What are FII and DII net cash positions over the latest trading sessions?"*
* *"What high-impact domestic and US economic calendar events are scheduled this week?"*
* *"Synthesize a complete Executive Macro Brief with current regime classification."*

### What It CANNOT Answer (Strict Scope Boundaries)
* ❌ **No Single-Stock Advice:** Never answers questions about individual equity tickers (e.g., *"Should I buy Reliance or Infosys?"*).
* ❌ **No Single-Company Financials:** Does not analyze single-stock earnings, margins, or balance sheets.
* ❌ **No Direct Trade Execution / Unsupervised Advice:** Operating strictly as a human-supervised research assistant, it never issues guaranteed buy/sell price targets.

---

## 4. Tools & Data Sources Matrix

The Macro Agent operates using 5 specialized tools defined in `macro_tools.py`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MACRO TOOL & DATA SOURCE MATRIX                           │
├───────────────────────────────┬──────────────────────────────────┬─────────────────────┤
│ Tool Name                     │ Primary External Data Source     │ Extracted Metrics   │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ `get_fii_dii_flows`           │ NSE Participant Feeds            │ FII Cash Net (₹ Cr),│
│                               │                                  │ DII Cash Net (₹ Cr),│
│                               │                                  │ Combined Flow, Bias │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ `get_indian_macro_indicators` │ FRED API (`DGS10`, `DGS2`,       │ US-India 10Y Spread,│
│                               │ `INDIRLTLT01STM`, `INDCPI...`)   │ US Yield Curve Slope│
│                               │                                  │ RBI Repo Rate (6.5%)│
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ `get_forex_and_commodities`   │ FMP API & `yfinance`             │ USD/INR, Brent, WTI,│
│                               │                                  │ Gold, DXY Spot +    │
│                               │                                  │ 5D/20D ROC %        │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ `get_economic_calendar`       │ Financial Modeling Prep (FMP)    │ High-Impact IN & US │
│                               │ Economic Calendar API            │ Catalyst Releases   │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ `search_web`                  │ Tavily Search API                │ Qualitative RBI/Fed │
│                               │                                  │ MPC Statements &    │
│                               │                                  │ Geopolitical News   │
└───────────────────────────────┴──────────────────────────────────┴─────────────────────┘
```

---

## 5. Machine Learning Architecture (Current vs. Planned)

### Current ML Implementation (`src/shankh/agents/macro/ml/`)
* **Baseline Regularized Regressors:** Linear/Ridge models fitted on monthly FRED macroeconomic time-series ($N \approx 240$ monthly rows over 20 years).
* **Role:** Estimates baseline trend expectations for inflation and yield series.

### Planned Future ML Architecture
* **MIDAS Nowcasting Engine (Mixed-Data Sampling):**  
  Bridges daily high-frequency financial market velocity (Brent Crude 5D/20D ROC, USD/INR, US 10Y Treasuries) with lagged monthly macro targets (CPI Inflation, GDP Growth).
* **Dynamic Factor Models (DFM):**  
  Extracts unobserved common macro factors from high-density cross-asset time-series to update real-time GDP and inflation probabilities daily.

---

## 6. Output Standard & Compliance

The Macro Agent formats every output as a standardized **Executive Macro Brief** with six mandatory Markdown sections:

1. `### 1. Macro Regime & Assessment` (Regime: `BULLISH_EXPANSION`, `BEARISH_CONTRACTION`, `STAGFLATIONARY_PRESSURE`, `HIGH_VOLATILITY_RISK_OFF`, or `CONSOLIDATING_NEUTRAL`)
2. `### 2. Institutional Flows & Liquidity`
3. `### 3. Fixed Income & Cross-Asset Spreads`
4. `### 4. Forex & Commodity Velocity`
5. `### 5. Economic Calendar & Global Cues`
6. `### 6. Key Macro Risk Factors`

Every numerical figure is sourced strictly from tool execution payloads with mandatory non-advisory compliance disclaimers appended.