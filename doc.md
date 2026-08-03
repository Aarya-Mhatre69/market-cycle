📋 Document Outline
1. Project Snapshot
2. How Interns Should Use This Learning Document
3. Common Foundation for All Agentic Projects
4. Project-Specific Goal
5. Workstreams
6. Domain Concepts to Learn
7. Suggested Architecture
Suggested 8-Week Learning and Build Plan
Key Resources
Prior Internal Learning Material to Review
Starter Assignments
Evaluation Criteria
Important Guardrails
The contents of this document are confidential and subject to the terms of a Non-Disclosure Agreement (NDA).

Learning Material for BE1: Financial Assistant Agentic Workflows

Shankh / Decuple Internship 2026-27

1. Project Snapshot
Project Code

BE1

Project Name

Financial Assistant Agentic Workflows

Target Intern Group

BE interns with Python, AI/ML, finance and backend interest

Expected Outcome

A working prototype, learning notes, reproducible code, demo video, and handover document.

2. How Interns Should Use This Learning Document
Use the first week to understand the domain, terminology and toolchain before writing production-style code.
Maintain a learning journal with resource links, notes, experiments, bugs and open questions.
Build in small increments: notebook or script, then API, then workflow, then UI or demo.
Every output must be reproducible from a GitHub repository with a README, sample inputs, screenshots and known limitations.
Do not use client data or live credentials in experiments. Use synthetic data, public data or sanitized samples only.
3. Common Foundation for All Agentic Projects
Agent basics: learn planning, tool use, memory, handoffs, observability and human approval loops.
RAG basics: learn how documents are chunked, embedded, retrieved and grounded before an LLM generates an answer.
Evaluation basics: learn to measure accuracy, hallucination rate, latency, cost and completion success rate.
Security basics: never expose API keys, OAuth tokens, private documents or personal information in prompts, logs or repositories.
4. Project-Specific Goal
Build a financial assistant platform that can run existing and new analytical workflows, improve forecast accuracy, explain market regimes, and support specialized agents for macro, market breadth, market cycle, microfish and microcap forensic analysis. The project is a research and prototype system, not an unsupervised advisory engine.

5. Workstreams
Price band forecast workflow: re-run the existing ML workflow, understand the baseline, evaluate accuracy and improve features, model choice and validation.
Last-year workflow improvement: study last year’s predictive analytics work, reproduce it, identify gaps and create a cleaner v2 pipeline.
Market regime analysis: create macro, market breadth and market cycle indicators that classify market environment as trending, range-bound, volatile, risk-on or risk-off.
Macro analyst agent: summarize rates, inflation, currency, crude, bond yield, FII/DII flows and global market cues into a daily/weekly market context note.
Market breadth agent: track advance-decline, new highs/lows, sector participation, volume participation and index breadth.
Market cycle agent: classify where the market may be in the cycle using valuation, momentum, liquidity and earnings indicators.
Microfish: build a small-stock discovery assistant that looks for early signals such as improving fundamentals, liquidity expansion and institutional interest while avoiding hype.
Microcap forensic agent: screen microcap companies for red flags such as auditor churn, pledge, receivable build-up, related-party transactions, cash flow mismatch, sudden promoter changes and abnormal price-volume behavior.
6. Domain Concepts to Learn
Equity price forecasting, walk-forward validation, leakage prevention and time-series cross-validation.
Technical indicators: moving averages, RSI, MACD, ATR, volatility, momentum, relative strength and volume signals.
Fundamental indicators: revenue growth, margins, ROE, debt, cash conversion, free cash flow, valuation and earnings revisions.
Macro indicators: inflation, policy rates, bond yields, currency, crude, liquidity, FII/DII flows and global risk appetite.
Market breadth: advance-decline, percent above moving average, sector leadership, new highs/lows and volume participation.
Forensic finance basics: financial statement quality, cash flow vs profit, auditor notes, pledge, corporate governance and disclosure quality.
7. Suggested Architecture
Data layer: NSE bhavcopy, Nifty indices, FMP, broker data, news/sentiment, internal CSV/parquet lake.
Feature layer: technical, fundamental, macro, breadth, sentiment and quality features with timestamped feature store.
Model layer: baseline models first, then LightGBM, random forest, logistic regression, Prophet or deep learning only where justified.
Agent layer: planner agent, data fetcher tools, analyst agents, verifier agent and report generator agent.
Memory layer: store model runs, assumptions, user preferences, past mistakes, evaluation metrics and reusable workflow notes.
Output layer: forecast report, regime dashboard, red-flag report and daily market context summary.
Suggested 8-Week Learning and Build Plan
Weeks

Learning Focus

Build Output

1

Python, pandas, stock market basics, existing project review

Repository setup and baseline data notebook

2

Time-series validation, leakage, metrics

Reproduced last-year workflow with documented baseline accuracy

3

Feature engineering and model comparison

Improved price band forecast pipeline with validation report

4

Macro, breadth and market cycle indicators

Market regime classifier v1

5

RAG and long-term memory for finance agents

Macro analyst and breadth analyst agents

6

Microcap forensic screening

Red-flag scoring notebook with example companies

7

Agent orchestration, observability and guardrails

Integrated workflow with verifier and report generator

8

Demo and documentation

End-to-end demo, README and handover video

Key Resources
Resource

Link

Why it matters

OpenAI Agents SDK

Open resource

Agent orchestration, handoffs, tool calling, guardrails, sessions and tracing.

LangGraph memory concepts

Open resource

Short-term and long-term memory design for agentic systems.

LangGraph long-term memory guide

Open resource

Persistent memory stores, namespaces and retrieval patterns.

n8n AI agents

Open resource

Low-code agent workflows with tools, memory and app integrations.

n8n AI workflow tutorial

Open resource

How to combine AI agent nodes with traditional workflow steps.

CrewAI documentation

Open resource

Multi-agent crews, flows, memory, knowledge, tools and observability.

CrewAI memory

Open resource

Unified memory class and adaptive recall for crews, agents and flows.

LlamaIndex documentation

Open resource

RAG, document indexing, workflows and data connectors for LLM applications.

Zerodha Kite Connect Python docs

Open resource

Portfolio, holdings, order, historical candle and streaming market data workflows.

Kite Connect product page

Open resource

Broker API capabilities and commercial constraints.

Financial Modeling Prep API docs

Open resource

Historical prices, fundamentals, analyst estimates, ratios and market news.

NSE historical reports

Open resource

Capital market archives, daily reports and breadth-related raw material.

NSE derivative market reports

Open resource

FO archives, contract-wise reports and derivatives market reference.

Nifty Indices historical data

Open resource

Index history, P/E, P/B, dividend yield and total return index data.

vectorbt documentation

Open resource

Fast quantitative research and vectorized strategy testing in Python.

backtesting.py documentation

Open resource

Beginner-friendly event-style backtesting and strategy optimization.

Prior Internal Learning Material to Review
Learning Material for Financial Advisor Agentic Workflow
Learning Material for Marketer Agentic Workflow
Learning Material for AI Engine - Options Trading Strategy
Agents Learning Material for Interns
Learning Material for AR Indic Chatbot
Learnings for Shankh Predictive Analytics and Indian Languages ChatBot
Starter Assignments
Reproduce one old predictive analytics notebook and document every assumption.
Create a data dictionary for all fields used in the forecast workflow.
Build a simple regime classifier using Nifty 50 trend, volatility and breadth.
Create a microcap red-flag checklist and test it on five public companies using only public information.
Evaluation Criteria
Clarity of problem understanding and domain assumptions.
Quality of data ingestion, cleaning, validation and documentation.
Correctness of agent workflow design, including tools, memory, guardrails and fallback behavior.
Evidence-based outputs with citations, logs, screenshots or evaluation tables.
Demo quality: realistic scenario, clear user journey and measurable improvement over baseline.
Code quality: modular design, README, requirements file, sample config and no hard-coded secrets.
Important Guardrails
Do not present model output as investment advice. This is a research prototype for supervised review.
Forensic outputs must say “red flag indicators” or “needs review”, not “fraud”.
Every forecast must show uncertainty, validation period and known weaknesses.
No live trading or order placement is permitted from this project.