# Shankh Market Specialist Agents
`src/shankh/agents/market/`

The **Shankh Market Specialist Suite** consists of two decoupled, highly specialized quantitative agents: the **Market Breadth Specialist Agent** and the **Market Cycle Specialist Agent**. Together, they provide institutional-grade analysis of short-to-medium-term market participation and long-horizon macro valuation positioning.

---

## 1. Executive Overview & Architecture Split

```
                                  MARKET DOMAIN
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ 1. MARKET BREADTH SPECIALIST AGENT      │   │ 2. MARKET CYCLE SPECIALIST AGENT        │
│ (src/shankh/agents/market/breadth_*)   │   │ (src/shankh/agents/market/cycle_*)     │
├───────────────────────────────┬─────────┤   ├───────────────────────────────┬─────────┤
│ Moving Average Coverage       │ % > 20/ │   │ Nifty 50 Index Multiples      │ P/E, P/B│
│ (% > 20, 50, 200 DMA)         │ 50/200  │   │ 10-Yr Historical Percentiles  │ Div Yld │
├───────────────────────────────┼─────────┤   ├───────────────────────────────┼─────────┤
│ Advance-Decline Momentum      │ McClellan│   │ Fed Model Equity Risk Premium  │ ERP %   │
│ & Net 52W Highs/Lows          │ Oscill. │   │ Spread vs India 10Y Yield     │ Spread  │
├───────────────────────────────┼─────────┤   ├───────────────────────────────┼─────────┤
│ Sector Leadership Matrix      │ FMP API │   │ Monetary & Credit Cycle       │ Bank Cr.│
│ (11 Industry Sectors)         │ Sectors │   │ Growth YoY & M3 Expansion     │ Growth  │
└───────────────────────────────┴─────────┘   └───────────────────────────────┴─────────┘
```

---

## 2. When to Use Which Agent

| Workflow / Decision Need | Agent to Invoke | Primary Signal Evaluated |
| :--- | :--- | :--- |
| **Detecting Distribution Under the Surface** | **Market Breadth Agent** | Index making new highs while % stocks > 50 DMA declines (Breadth Divergence). |
| **Evaluating Short-Term Buying/Selling Volume**| **Market Breadth Agent** | McClellan Oscillator score and Volume Breadth log ratio. |
| **Sector Rotation & Leadership Audits** | **Market Breadth Agent** | Sector positive participation rate and Top 3 vs Bottom 3 sector returns. |
| **Strategic Asset Allocation (Equity vs Debt)**| **Market Cycle Agent** | Fed Model Equity Risk Premium ($\text{ERP} = \frac{1}{\text{P/E}} - \text{IN10Y Yield}$). |
| **Assessing Valuation Stretch / Overbought Risk**| **Market Cycle Agent** | Nifty 50 P/E historical percentile rank over 10-year rolling window. |
| **Macro Credit Expansion / Tightening Audits**| **Market Cycle Agent** | Bank Credit Growth YoY %, M3 Money Supply, and RBI Liquidity Stance. |

---

## 3. What They Can Answer vs. What They Cannot Answer

### What They Can Answer
* **Market Breadth Agent:**
  * *"Is the current market rally backed by broad 150-ticker participation or narrow mega-cap concentration?"*
  * *"What is the current McClellan Oscillator score and advance-decline momentum?"*
  * *"Which sectors are leading vs lagging in the current market rotation?"*
* **Market Cycle Agent:**
  * *"What is the current Nifty 50 P/E ratio and where does it sit on a 10-year historical percentile basis?"*
  * *"What is the Equity Risk Premium (ERP) spread and are equities attractive relative to G-Sec debt?"*
  * *"Is systemic bank credit growth expanding or contracting, and what is the current market cycle phase?"*

### What They CANNOT Answer (Strict Scope Boundaries)
* ❌ **No Single-Stock Advice:** Neither agent evaluates single-stock tickers or gives stock tips (e.g., *"Should I buy Reliance?"*).
* ❌ **No Single-Stock Technicals:** They evaluate cross-sectional universe breadth and index valuations, not individual candlestick charts.
* ❌ **No Unsupervised Trade Execution:** Outputs are structured exclusively for human-in-the-loop analyst review.

---

## 4. Tools & Data Sources Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          MARKET AGENTS TOOL & DATA SOURCE MATRIX                       │
├───────────────────────────────┬──────────────────────────────────┬─────────────────────┤
│ Agent / Tool Name             │ Primary Data Source              │ Key Metrics         │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ **Market Breadth Agent**      │ Local 20-Yr 150-Ticker Parquet   │ % > 20/50/200 DMA,  │
│ `get_market_breadth_metrics`  │ Dataset (`data/universe/`)       │ Net 52W Highs/Lows, │
│                               │ Sub-second execution             │ McClellan Oscillator│
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ **Market Breadth Agent**      │ Financial Modeling Prep (FMP)    │ Sector positive %   │
│ `get_sector_participation`    │ `/v3/sector-performance`         │ Top/Bottom Sectors  │
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ **Market Cycle Agent**        │ FMP API & Nifty Indices          │ Nifty P/E, P/B,     │
│ `get_market_cycle_metrics`    │ Historical Archives              │ 10Y P/E Percentile, │
│                               │ FRED API (`INDIRLTLT01STM`)      │ ERP Spread, US Slope│
├───────────────────────────────┼──────────────────────────────────┼─────────────────────┤
│ **Market Cycle Agent**        │ FRED API (`MYB599INM189S`)       │ Bank Credit Growth, │
│ `get_liquidity_credit_cycle`  │ & RBI Public Releases            │ M3 Supply Growth %, │
│                               │                                  │ Liquidity Stance    │
└───────────────────────────────┴──────────────────────────────────┴─────────────────────┘
```

---

## 5. Machine Learning Architecture (Current vs. Refactored Target)

### Current Implementation (`src/shankh/agents/market/ml/`)
* **3-State Gaussian Hidden Markov Model (`GaussianHMM`):** Fits on 7 technical features (`mkt_return_20d`, `mkt_volatility`, `parkinson_volatility`, `composite_breadth`, `ad_index`, `volume_breadth_ratio`, `correlation_density`).
* **Limitation:** Gaussian HMM assumes unconstrained multivariate normal distributions, which violates bounded percentage inputs (`composite_breadth` $0$–$100\%$, `ad_index` $-1$ to $+1$).

### Refactored Target Architecture
1. **Two-Stage Market Regime Model:**
   * **Stage 1 (LightGBM Quantile Tree Classifier):** Learns non-linear threshold boundaries on non-Gaussian bounded inputs (% > 50/200 DMA, Net Highs, VIX).
   * **Stage 2 (Constrained Markov Transition Filter):** Enforces sticky state inertia ($P(i|i) \ge 0.95$) to eliminate daily state whipsaws.
2. **Multi-Factor Composite Cycle Distance Matrix:**
   * Computes Euclidean distance between current normalized factor Z-scores ($Z_{\text{Valuation}}$, $Z_{\text{ERP}}$, $Z_{\text{Credit}}$) and historical cycle phase centroids (`EARLY_EXPANSION`, `MID_CYCLE_NEUTRAL`, `MID_CYCLE_PEAK`, `LATE_VALUATION_BUBBLE`, `LIQUIDITY_CONTRACTION`, `RECESSIONARY_TROUGH`).

---

## 6. Output Standards & Compliance

* **Market Breadth Agent Output:** Generates a formal **Executive Market Breadth Brief** featuring Breadth Regime classification (`BROAD_BULLISH_EXPANSION`, `NARROW_BEAR_MARKET_RALLY`, `BREADTH_DIVERGENCE_WARNING`, etc.), DMA coverage, McClellan momentum, and key breadth risk signals.
* **Market Cycle Agent Output:** Generates a formal **Executive Market Cycle Brief** featuring Market Cycle Phase classification (`EARLY_EXPANSION`, `LATE_VALUATION_BUBBLE`, etc.), Nifty 10Y P/E percentile rank, Equity Risk Premium analysis, and strategic asset allocation risk signals.

All numerical figures are strictly grounded in tool JSON payloads with mandatory non-advisory compliance disclaimers appended.