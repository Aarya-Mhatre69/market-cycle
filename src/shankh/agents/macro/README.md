# 🔍 Stock Clustering & Forensic Anomaly Detection Model Notes

## 1. Model Overview & Purpose

The **Stock Clustering and Forensic Anomaly Pipeline** provides unsupervised portfolio risk analytics and outlier screening across an equity trading universe.

Unlike prediction models that output a single target price or market mood, this pipeline analyzes **multi-asset interactions** to perform two distinct functions:

1. **Portfolio Diversification**: Groups stocks into distinct risk personalities and price co-movement structures so portfolios are not over-concentrated in identical risk profiles.
2. **Forensic Outlier Screening**: Detects structural anomalies (e.g., price manipulation, illiquidity traps, extreme drawdown risk) before stocks are added to a trading strategy.

---

# 2. Pipeline Architecture

The pipeline consists of **three complementary unsupervised models**. Each solves a different mathematical problem while answering a different portfolio management question.

```text
                               STOCK UNIVERSE
                                     │
                    Historical Prices & Risk Features
                                     │
                      Feature Engineering + Scaling
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────────┐
│ AGGLOMERATIVE      │     │ K-MEANS            │     │ ISOLATION FOREST       │
├────────────────────┤     ├────────────────────┤     ├────────────────────────┤
│ Price Co-Movement  │     │ Risk Personality   │     │ Anomaly Detection      │
│                    │     │                    │     │                        │
│ "Who moves         │     │ "Who acts          │     │ "Who is                │
│  together?"        │     │  alike?"           │     │  dangerous?"           │
└────────────────────┘     └────────────────────┘     └────────────────────────┘
```

---

## A. Agglomerative Hierarchical Clustering (Price Co-Movement)

**Purpose**

Groups stocks according to **daily price co-movement** (timing and synchronicity).

**Input**

An $M \times M$ pairwise **Correlation Distance Matrix**

$$
d=\sqrt{2(1-\rho)}
$$

where $\rho$ is the Pearson correlation between stock returns.

**Linkage Method**

- Average linkage
- Precomputed distance matrix

**Why it is needed**

Stocks belonging to the same sector or ETF basket frequently move together. Even if two companies are fundamentally different, highly correlated price movement can create hidden concentration risk within a portfolio.

---

## B. K-Means Clustering (Statistical Risk Personality)

**Purpose**

Groups stocks according to their long-term statistical characteristics rather than daily synchronization.

**Input**

A scaled

$$
N \times 9
$$

multi-factor feature matrix using **RobustScaler**.

**Default Number of Clusters**

$$
K = 4
$$

**Why it is needed**

Two stocks may never move together on the same day yet still possess nearly identical:

- volatility
- beta
- drawdown characteristics
- momentum profile
- return distribution

K-Means identifies these shared "risk personalities" to improve portfolio diversification.

---

## C. Isolation Forest (Forensic Anomaly Detection)

**Purpose**

Acts as a **sanity filter** by identifying statistically abnormal stocks.

**Input**

The same scaled

$$
N \times 9
$$

feature matrix used by K-Means.

**Contamination Rate**

```text
0.10
```

meaning approximately the most abnormal **10%** of observations are flagged.

**Why it is needed**

Unlike clustering algorithms, Isolation Forest is designed specifically for anomaly detection.

K-Means is forced to assign **every stock** to some cluster—even fraudulent, illiquid, or manipulated securities.

Isolation Forest instead isolates these unusual observations using random decision trees and flags them for manual review.

Predictions are:

- `1` → Normal
- `-1` → Anomaly

---

# 3. Input Features & Preprocessing

## Scaling Strategy — RobustScaler

Traditional Z-score normalization (`StandardScaler`) uses the mean and standard deviation, both of which are highly sensitive to extreme market outliers.

Instead, this pipeline uses **RobustScaler**, which scales using the median and interquartile range (IQR).

$$
x_{\text{scaled}}
=
\frac{x-\mathrm{Median}(x)}
{\mathrm{IQR}(x)}
$$

This produces more stable feature distributions in the presence of extreme returns.

---

## Multi-Factor Features (`factor_df`)

| Feature | Mathematical Formula | Category | Description |
|----------|----------------------|----------|-------------|
| **ann_return** | $\text{mean}(r_t)\times252$ | Return Profile | Annualized log return. |
| **ann_vol** | $\text{std}(r_t)\times\sqrt{252}$ | Volatility | Annualized volatility. |
| **sharpe** | $\frac{\text{ann\_return}-0.065}{\text{ann\_vol}+10^{-6}}$ | Risk-Adjusted | Sharpe ratio using a 6.5% risk-free rate. |
| **skewness** | $\text{skew}(r_t)$ | Tail Risk | Measures asymmetry of returns. |
| **max_drawdown** | $\min\left(\frac{\text{Peak}_t-P_t}{\text{Peak}_t}\right)$ | Tail Risk | Largest historical peak-to-trough decline. |
| **beta** | $\frac{\operatorname{Cov}(r_{\text{stock}},r_{\text{market}})}{\operatorname{Var}(r_{\text{market}})}$ | Market Sensitivity | Sensitivity to the equal-weight market portfolio. |
| **dist_20dma** | $\frac{P_t-\text{SMA}_{20}}{\text{SMA}_{20}}$ | Trend | Distance from the 20-day moving average. |
| **rsi_14** | $100-\frac{100}{1+\text{RS}_{14}}$ | Momentum | Relative Strength Index. |
| **atr_pct** | $\frac{\text{ATR}_{14}}{P_t}$ | Volatility | ATR expressed as a percentage of price. |

---

# 4. Model Outputs

The inference pipeline (`run_inference()`) returns a JSON object similar to:

```json
{
  "kmeans_factor_clusters": {
    "RELIANCE.NS": 0,
    "TCS.NS": 2,
    "INFY.NS": 2,
    "PENNY_STOCK.NS": 1
  },
  "flagged_forensic_anomalies": [
    "PENNY_STOCK.NS"
  ]
}
```

### `kmeans_factor_clusters`

Maps every stock to its assigned behavioral cluster.

Example:

```text
Cluster 0 → Defensive
Cluster 1 → High Volatility
Cluster 2 → Growth
Cluster 3 → Cyclical
```

(The numeric labels themselves have no intrinsic meaning.)

---

### `flagged_forensic_anomalies`

Contains the list of securities classified as anomalies by Isolation Forest.

These should be manually investigated before inclusion in any portfolio.

---

# 5. Unsupervised Evaluation Metrics

Because no ground-truth labels exist, cluster quality is evaluated using geometric separation metrics.

| Metric | Mathematical Definition | Goal | Interpretation |
|---------|-------------------------|------|----------------|
| **Silhouette Score** | $S=\frac{b-a}{\max(a,b)}$ | Maximize | Measures how well each point fits within its own cluster relative to neighboring clusters. Values above 0.25 generally indicate meaningful structure. |
| **Davies–Bouldin Index** | $DB=\frac{1}{K}\sum\max\left(\frac{\sigma_i+\sigma_j}{d(c_i,c_j)}\right)$ | Minimize | Ratio of within-cluster spread to between-cluster separation. Lower values indicate better clustering. |
| **Calinski–Harabasz Index** | $\frac{\text{Between-Cluster Variance}}{\text{Within-Cluster Variance}}$ | Maximize | Higher values indicate compact and well-separated clusters. |

---

# 6. Why Three Different Models?

Each algorithm answers a fundamentally different investment question.

| Portfolio Question | Model | Mathematical Input | Benefit |
|--------------------|-------|--------------------|----------|
| **Which stocks move together every day?** | Agglomerative Clustering | Correlation Distance Matrix | Prevents hidden correlation concentration. |
| **Which stocks have similar long-term risk characteristics?** | K-Means | Scaled Multi-Factor Feature Matrix | Improves diversification across risk profiles. |
| **Which stocks appear statistically abnormal?** | Isolation Forest | Scaled Multi-Factor Feature Matrix | Detects potentially manipulated, illiquid, or structurally unusual securities. |

---

# 7. End-to-End Workflow

```text
                      Historical OHLCV Data
                               │
                               ▼
                    Feature Engineering Pipeline
                               │
                               ▼
                     RobustScaler Transformation
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
 Agglomerative          K-Means Clustering    Isolation Forest
   Clustering              (Risk Types)      (Anomaly Detection)
        │                      │                      │
        └──────────────┬───────┴──────────────┬───────┘
                       ▼                      ▼
             Cluster Assignments     Anomaly Flags
                       │
                       ▼
          Portfolio Construction & Risk Review
```

---

# 8. Summary

The pipeline combines three complementary unsupervised learning techniques:

- **Agglomerative Clustering** discovers stocks that exhibit similar price movement patterns.
- **K-Means Clustering** groups stocks by long-term statistical risk characteristics.
- **Isolation Forest** identifies structurally abnormal securities that warrant further investigation.

Together, these models provide a more robust foundation for portfolio construction by simultaneously addressing:

- correlation risk,
- factor diversification,
- and forensic anomaly detection.