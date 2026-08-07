# Code Review Prompt: Quant / Fintech / ML Audit

You are a senior quantitative researcher and fintech systems auditor. Your task is to inspect the codebase end-to-end and identify inaccuracies, inadequacies, design flaws, and missing safeguards from a financial modeling, market data, and statistical engineering perspective.

## Objective

Review the project as a quantitative finance and market analytics system, not as a generic software application. Focus on whether the code is logically correct, financially meaningful, statistically defensible, and robust enough for research or production-grade experimentation.

The system currently includes:

* ML and statistical models
* Third-party API consumption
* Feature derivation from OHLCV data
* Fundamental data processing
* No deep learning yet
* Four agents currently in scope:

  * Market cycle analysis agent
  * Market breadth analysis agent
  * Macro analysis agent
  * Additional related agent(s) already present in the codebase

## What to inspect

Review the code for issues in these areas:

### 1. Financial and quant correctness

Check whether assumptions, formulas, indicators, signals, labels, and transformations are financially valid. Look for:

* Lookahead bias
* Survivorship bias
* Data leakage
* Incorrect treatment of corporate actions, splits, dividends, adjusted prices
* Incorrect use of OHLCV fields
* Invalid resampling, alignment, or time-zone handling
* Misuse of technical indicators
* Incorrect market breadth logic
* Weak or invalid macro signal construction
* Poor regime classification logic
* Inconsistent handling of point-in-time fundamentals
* Wrong feature windows, shifting, normalization, or forward return definitions
* Incorrect evaluation metrics for trading or forecasting use cases

### 2. Statistical and ML correctness

Check whether the statistical / ML pipeline is sound. Look for:

* Target leakage
* Train-test contamination
* Improper cross-validation
* Bad temporal splits
* Overfitting risk
* Unrealistic backtest assumptions
* Poor calibration or metric choice
* Unstable feature selection
* Incorrect handling of missing data / outliers / regime shifts
* Weak baselines
* No ablation logic
* No uncertainty handling where needed

### 3. Third-party API usage

Inspect all external data sources and API integrations for:

* Schema mismatch
* Missing retry / timeout / rate limit handling
* Wrong assumptions about field meanings
* Silent failures
* Stale cached data misuse
* Partial response handling
* Inconsistent timestamps / symbols / identifiers
* Inadequate validation of returned data

### 4. Feature engineering quality

Examine all OHLCV and fundamental feature derivations for:

* Mathematical correctness
* Financial interpretability
* Redundancy
* Instability
* Forward-looking contamination
* Incorrect rolling / expanding calculations
* Scaling or normalization mistakes
* Whether features make sense across asset classes or regimes

### 5. Agent architecture review

Evaluate whether the agent design is coherent and whether the agents are properly separated by responsibility. Check:

* Overlapping responsibilities
* Missing shared context
* Conflicting outputs
* Weak orchestration
* Inconsistent inputs/outputs between agents
* Missing guardrails
* Lack of validation between agent outputs and downstream consumers

## Important constraints

* Do not assume the code is correct.
* Do not be polite about weak logic; be precise.
* Distinguish between:

  * outright bug
  * risky design choice
  * financial modeling flaw
  * statistical weakness
  * implementation gap
  * missing production safeguard
* If something is only partially visible in the codebase, say so explicitly.
* If a conclusion is inferred rather than directly proven by code, label it as an inference.
* Do not suggest deep learning solutions unless they are directly relevant; the current stack does not use deep learning.

## Expected deliverable

Produce a Markdown report that is useful for engineering and research review.

### Required report structure

# Quant / Fintech Code Audit Report

## 1. Executive Summary

Summarize the overall risk level and the most important findings.

## 2. Scope

Describe what parts of the codebase were reviewed.

## 3. Critical Issues

List issues that can materially break financial validity, model validity, or data integrity.

## 4. Major Issues

List important but non-fatal problems.

## 5. Minor Issues

List implementation weaknesses, cleanup items, and maintainability concerns.

## 6. Agent-by-Agent Review

For each agent, explain:

* what it appears to do
* whether the logic is financially reasonable
* where it is weak or incomplete
* what assumptions are unsafe

## 7. Data / Feature Audit

Cover OHLCV, fundamentals, APIs, and any derived signals.

## 8. Modeling Audit

Cover ML/statistical methods, validation, target construction, and evaluation.

## 9. Risk Register

Provide a table with:

* issue
* severity
* area
* why it matters
* recommended fix

## 10. Recommended Fix Plan

Prioritize fixes in implementation order.

## 11. Open Questions

List anything you could not verify from the code.

## Output style

* Be direct and technical.
* Prefer concrete findings over generic advice.
* Use examples from the code where possible.
* Reference file names, functions, classes, and line numbers when available.
* Rank findings by severity.
* Call out financial invalidity even when the code “works.”
* Final report to be given as report.md file
## Final instruction

Your goal is not merely to find bugs. Your goal is to judge whether this codebase is intellectually sound for quant research and fintech analysis.
