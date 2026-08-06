# 🏛️ Market Regime Analysis Model Documentation

## 1. Executive Summary

The **Market Regime Analysis Pipeline** is an unsupervised machine learning system designed to detect macro market environments (regimes) from multi-asset OHLCV market data.

It fits a **3-State Gaussian Hidden Markov Model (GaussianHMM)** on systemic market features (Volatility, Breadth, and Correlation Density). The discovered states are deterministically sorted and mapped to three actionable market regimes:
1. **Risk-On**: Low volatility, high market breadth, low correlation density.
2. **Range-Bound**: Moderate volatility, neutral breadth, steady correlation.
3. **Risk-Off**: High volatility, collapsed market breadth, high correlation density.

---

## 2. Input Features

The model aggregates a daily cross-sectional panel of assets into 3 normalized macro features:

| Feature Name | Formula / Derivation | Range | ML Rationale |
| :--- | :--- | :--- | :--- |
| `mkt_volatility` | $\text{std}\left(\bar{r}_t\right)_{20\text{d}} \times \sqrt{252} \times 100$ | $[0, \infty)$ | Annualized volatility of universe mean returns. Primary metric for regime sorting. |
| `breadth_pct_above_20dma` | $\frac{\sum \mathbb{I}(P_{m,t} > \text{SMA}_{20})}{M} \times 100$ | $[0, 100\%]$ | Market participation rate. |
| `correlation_density` | $\text{mean}\left(\text{triu}(\mathbf{R}_{20\times 20})\right)$ | $[-1.0, 1.0]$ | Average pairwise stock return correlation across the universe over a 20-day window. |

> **Pre-processing Note**: Features are standardized using `sklearn.preprocessing.StandardScaler` prior to fitting the HMM.

---

## 3. Model Architecture & Training

* **Algorithm**: `hmmlearn.hmm.GaussianHMM`
* **Hidden States ($N$)**: 3
* **Covariance Type**: `full` (learns full $3 \times 3$ feature covariance per state)
* **Optimization**: Expectation-Maximization (EM / Baum-Welch Algorithm)

### Deterministic State Alignment Logic
Unsupervised state labels ($0, 1, 2$) are sorted by their mean annualized market volatility ($\mu_{\text{vol}}$) in ascending order:
* **State with Lowest $\mu_{\text{vol}}$** $\rightarrow$ `"risk-on"`
* **State with Median $\mu_{\text{vol}}$** $\rightarrow$ `"range-bound"`
* **State with Highest $\mu_{\text{vol}}$** $\rightarrow$ `"risk-off"`

---

## 4. Model Outputs

1. **Active Regime Label (`regime_label`)**:
   * `"risk-on"`, `"range-bound"`, or `"risk-off"`
2. **State Discrete Sequence (`regime_state`)**:
   * Integer state assigned to each historical trading day.
3. **State Posterior Probabilities (`predict_proba`)**:
   * Probability distribution across all 3 regimes for any given trading date:
     $$[P(\text{Risk-On}), P(\text{Range-Bound}), P(\text{Risk-Off})], \quad \sum P_i = 1.0$$

---

## 5. Evaluation Metrics

| Metric Category | Metric Name | Definition / Formula | Interpretation |
| :--- | :--- | :--- | :--- |
| **Model Fit** | `log_likelihood` | $\log P(\mathbf{X} \mid \boldsymbol{\lambda})$ | Total log-likelihood score of observations under the HMM. Used for Optuna hyperparameter optimization (higher is better). |
| **State Dynamics** | `transition_matrix` | $A_{i,j} = P(S_t = j \mid S_{t-1} = i)$ | $3 \times 3$ matrix of regime transition probabilities. High diagonal values ($A_{i,i} > 0.90$) indicate stable, sticky regimes. |
| **Regime Persistence** | `state_duration_days` | $D_i = \frac{1}{1 - A_{i,i}}$ | Expected consecutive trading days spent in regime $i$ before transitioning. |

---

## 6. Pipeline Artifacts

Artifacts stored in `models/regime/artifacts/`:

* `regime_model.joblib`: Binary fitted `GaussianHMM` model object.
* `regime_scaler.joblib`: Binary fitted `StandardScaler` object.
* `regime_metadata.json`: Contains state mappings, historical state mean statistics, transition metrics, and current active regime state.
* `historical_regimes.csv`: Historical daily time series of features, assigned state IDs, and regime labels.