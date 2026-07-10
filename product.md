*Problem Statement : Financial Assistant Agentic Workflows* 
> What it is about 
BE1 will build and improve AI-assisted financial research workflows for Shankh. The team will focus on machine learning workflows, price-band forecasting, previous-year workflow improvement, market regime analysis, MiroFish-style swarm simulation experiments, and a microcap forensic research agent. The goal is not to create a blind trading bot, but to create a supervised research assistant that can collect data, run workflows, evaluate signals, explain assumptions, and generate investment-research-ready outputs for human review. 
> Why it is interesting 
This project sits at the intersection of AI, quant research, market intelligence, and wealth advisory. Interns will learn how financial agents combine data perception, reasoning, signal generation, and controlled execution. It also gives exposure to practical investment research problems such as regime shifts, breadth deterioration, liquidity risk, microcap governance issues, and model accuracy improvement. 
> Major Workstreams 

Workstream 

Core activities 

Data / tools 

Expected output 

Price Band Forecast ML Workflows 

Run existing ML workflows for price-band forecasting. Evaluate accuracy, error bands, feature importance, leakage risk, and market-condition dependency. 

Python, pandas, scikit-learn, XGBoost/LightGBM, FMP, NSE bhavcopy, yfinance only as fallback, notebooks 

Improved forecast pipeline, baseline vs improved metrics, prediction report, feature documentation. 

Improve Last Year's Workflows 

Review prior internship workflows. Identify broken assumptions, stale APIs, missing tests, poor prompts, weak evaluation, and hard-coded logic. Refactor into reusable modules. 

GitHub, Python, n8n, LangGraph/CrewAI where useful, Docker, pytest 

Cleaned and reusable workflow library with README, test cases, and demo notebooks. 

Market Regime Analysis 

Create agents for Macro Analyst, Market Breadth, and Market Cycle. Classify market environment as risk-on, risk-off, range-bound, liquidity-driven, sector-rotation-led, or high-volatility. 

Index data, sector indices, FII/DII data, rates, crude, USD/INR, VIX, breadth indicators, HMM/clustering 

Regime dashboard, regime classification logic, monthly/weekly regime notes, explainable signal cards. 

MiroFish Experiment 

Experiment with swarm/simulation-style forecasting for market narratives, policy events, earnings events, or macro shocks. Compare with simpler methods so the team learns where swarm simulation helps and where it does not. 

MiroFish or similar multi-agent simulation, news inputs, policy/event seeds, synthetic personas, scenario prompts 

Prototype scenario simulator, sample event reports, limitations note, recommendation on whether to integrate further. 

Microcap Specialist Agent: Forensic 

Build a forensic screening agent for microcap research. Focus on governance, promoter pledge, related-party transactions, auditor issues, liquidity, unusual price/volume behavior, corporate announcements, and financial statement red flags. 

Annual reports, exchange filings, Screener/Tijori-style data, NSE/BSE announcements, FMP, document extraction, RAG 

Microcap forensic checklist, risk scoring prototype, evidence-backed report template, red-flag database. 

> Suggested Tech & Tools 
Language: Python as the primary language; SQL for data storage and querying. 

Agent frameworks: LangGraph for controlled stateful workflows; CrewAI for role-based multi-agent experiments; n8n for visual orchestration and scheduled runs. 

ML stack: pandas, NumPy, scikit-learn, XGBoost/LightGBM, statsmodels, PyTorch only if needed. 

Data sources: NSE/BSE files, FMP APIs, Zerodha Kite where permitted, public filings, sector/index data, macro indicators, and curated news/event feeds. 

Storage and retrieval: PostgreSQL, Chroma/FAISS, object storage for raw files, and strict versioning of datasets. 

> Expected Final Deliverables 
1. One working demo for price-band forecasting and accuracy improvement. 
2. One market regime dashboard or notebook with regime explanations. 
3. One MiroFish-style event simulation prototype and findings document. 
4. One microcap forensic agent that generates evidence-backed research reports. 
5. Architecture diagram, setup guide, evaluation metrics, and final presentation.