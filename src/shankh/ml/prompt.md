
---

# Prompt

You are refactoring the existing `macro` (Stock Clustering) and `market` (Market Regime Analysis) modules.

The `price_band` module is the reference architecture and coding standard for the repository.

**Do not invent a new project structure.**
**Mirror the architecture, coding style, file organization, and modularity of `src/shankh/ml/price_band`.**

Current structure:

```text
src/shankh/ml/

price_band/
    config.py
    data_loader.py
    evaluation.py
    features.py
    inference.py
    model.py
    train_xgb.py
    tune_xgb.py
    validation.py
    tool.py
```

Refactor both

```
src/shankh/ml/macro
src/shankh/ml/market
```

to follow this exact architecture.

---

## General Requirements

Do **not** rewrite the underlying ML algorithms unless required.

Preserve existing mathematical logic.

The objective is to make both modules production-quality and consistent with `price_band`.

Keep naming conventions identical wherever possible.

---

# Expected Directory Structure

### Stock Clustering

```
macro/

    config.py

    data_loader.py

    features.py

    validation.py

    model.py

    evaluation.py

    inference.py

    tool.py

    train_cluster.py

    tune_cluster.py

    test_inference.py

    README.md

    __init__.py
```

---

### Market Regime

```
market/

    config.py

    data_loader.py

    features.py

    validation.py

    model.py

    evaluation.py

    inference.py

    tool.py

    train_regime.py

    tune_regime.py

    test_inference.py

    README.md

    __init__.py
```

---

# File Responsibilities

## config.py

Contain only

* hyperparameters
* feature configuration
* model configuration
* paths
* random seed
* experiment settings

No executable logic.

---

## data_loader.py

Responsible only for

* loading market data
* loading macroeconomic data
* loading cached datasets
* preprocessing timestamps
* schema validation
* train/test split helpers

No feature engineering.

---

## features.py

Responsible only for feature engineering.

Split into reusable functions such as

```
build_returns()

build_volatility()

build_trend()

build_volume()

build_momentum()

build_market_features()

build_macro_features()

build_cross_sectional_features()

build_cluster_features()

build_regime_features()
```

No model training.

---

## validation.py

Responsible only for

* missing values
* NaNs
* infinities
* duplicate rows
* schema validation
* feature validation
* inference input validation

---

## model.py

Contains only model classes.

For clustering include implementations such as

* KMeans
* HDBSCAN
* Gaussian Mixture
* Hierarchical
* Spectral

For regime models include

* Hidden Markov Models
* Gaussian HMM
* Bayesian models
* Change Point Detection
* Any existing regime classifier

Every model should expose a consistent interface

```
fit()

predict()

predict_proba()      # when supported

save()

load()
```

No plotting.

No CLI.

---

## evaluation.py

Contains all evaluation logic.

### Clustering

* Silhouette Score
* Davies-Bouldin
* Calinski-Harabasz
* Cluster distribution
* Cluster stability
* Inertia
* BIC/AIC (where applicable)

### Regime

* Transition Matrix
* Regime Persistence
* State Duration
* Accuracy
* F1
* Precision
* Recall
* Confusion Matrix
* Log Likelihood

---

## inference.py

Responsible only for inference.

Workflow

```
load artifacts

validate input

build features

load scaler

predict

return structured result
```

Training code should never appear here.

---

## tool.py

Expose LangChain/LangGraph tools exactly like the existing `price_band.tool`.

Should only wrap inference.

No training logic.

---

## train_cluster.py / train_regime.py

Pipeline entrypoint.

Responsibilities

```
load data

validate

feature engineering

train model

evaluate

save artifacts

log metrics
```

No duplicated logic.

Should call reusable functions from other modules.

---

## tune_cluster.py / tune_regime.py

Responsible only for hyperparameter optimization.

Support existing tuning framework used in `price_band`.

No duplicated training logic.

---

## README.md

Document

* pipeline
* features
* training
* inference
* artifacts
* expected inputs
* outputs

---

# Refactoring Rules

Follow the same coding conventions as `price_band`.

Move duplicated code into reusable functions.

Avoid large monolithic scripts.

Every file should have one clear responsibility.

Keep type hints.

Keep docstrings.

Replace print statements with logging.

Do not change public APIs unless necessary.

---

# End-to-End Training Flow

The final training scripts should execute

```
Load Data
    ↓
Validate
    ↓
Engineer Features
    ↓
Train Model
    ↓
Tune (optional)
    ↓
Evaluate
    ↓
Save Model
    ↓
Save Metrics
    ↓
Ready for Inference
```

---

# Goal

After the refactor, `macro`, `market`, and `price_band` should feel like three implementations of the same ML framework, with identical module organization, naming conventions, coding style, and lifecycle. A developer familiar with `price_band` should immediately understand and navigate the `macro` and `market` modules.
