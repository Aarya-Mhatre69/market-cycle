# Shankh

**AI-powered financial research assistant for Indian equity markets.**

Shankh is a multi-agent research system that combines a **Deep Agents** supervisor, three specialized analyst sub-agents, and a set of decoupled **ML pipelines** (market regime detection, price-band forecasting, stock clustering & forensic screening). Trained models are exposed to the LLM layer through lightweight **MCP servers**, so data fetching stays separate from agent reasoning.

The system is designed as a **supervised research assistant** — it collects data, runs ML workflows, explains its assumptions, and produces investment-research-ready output for human review. It does **not** execute trades or give personalized financial advice.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [ML Pipelines](#ml-pipelines)
- [MCP Servers](#mcp-servers)
- [Agents](#agents)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Running Tests](#running-tests)
- [Docker & Deployment](#docker--deployment)
- [Project Documentation](#project-documentation)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Highlights

- **Supervisor / sub-agent orchestration** built with `deepagents.create_deep_agent` and LangGraph checkpoints.
- **Three analyst sub-agents**: Macro Analyst, Market Analyst, and Company Analyst, each with its own system prompt and tool set.
- **Three independent ML pipelines**:
  - **Market Regime** — 3-state Gaussian HMM classifying *risk-on / range-bound / risk-off*.
  - **Price Band** — LightGBM quantile regression predicting next-day high/low bands for NSE stocks.
  - **Stock Clustering & Forensics** — K-Means + Agglomerative clustering and Isolation Forest anomaly screening.
- **Decoupled ML via MCP** — each trained model runs behind its own Streamable-HTTP MCP server (ports 8001–8003); the agent fetches tools dynamically at runtime.
- **Live data** — web search via Tavily; market data fetched on demand from yfinance (no stale pre-computed files).
- **Web API + MCP** — both a FastAPI HTTP API (`/ask`, `/health`) and an MCP server expose the advisor.
- **Next.js chat frontend** with connection badge, streaming-aware timeout handling, and thread-based memory.
- **Dockerized** — single `uv`-based image plus `docker-compose` for the optional Postgres checkpointer.

---

## Architecture

```text
                 ┌────────────────────────────┐
                 │       Chat Frontend       │
                 │       (Next.js 16)        │
                 └────────────┬───────────────┘
                              │ POST /ask  (FastAPI)
                 ┌────────────▼───────────────┐
                 │     Financial Advisor      │
                 │   supervisor (Deep Agent)  │
                 └───┬────────┬────────┬──────┘
              routes │        │        │
       ┌─────────────▼┐ ┌─────▼─────┐ ┌▼─────────────────┐
       │ macro-analyst│ │market-    │ │ company-analyst  │
       │ sub-agent    │ │analyst    │ │ sub-agent        │
       └──────┬───────┘ │sub-agent  │ └─────┬────────────┘
              │         └─────┬─────┘       │
              │  shared tools ─┼─────────────┤
              │   search_web (Tavily)        │
              │                              │
              └──────────┬───────────────────┘
                         │ MCP over Streamable HTTP
        ┌────────────────┼────────────────┬──────┐
        ▼                ▼                ▼      │
 ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
 │  Macro MCP  │  │ Market MCP  │  │  Price Band │ │  web search
 │  :8001      │  │ :8002       │  │  MCP :8003  │ │  (Tavily)
 │ Clustering+ │  │ Regime HMM  │  │ LGBM quantile│ │
 │ Forensics   │  │             │  │ regression  │ │
 └─────────────┘  └─────────────┘  └─────────────┘ │
        │               │                │          │
        ▼               ▼                ▼          ▼
   models/clustering  models/regime  models/price_band
            └─────────── data/universe (OHLCV CSVs) ──┘
```

**Design rule** (from `docs/design.md`): the **supervisor decides**, the **specialists analyze**, the **MCP servers / APIs fetch data**, and the **ML models infer regime or forecast**. Nothing mixes fetching, reasoning, and execution in one place.

---

## Repository Structure

```text
shankh/
├── main.py                        # FastAPI server for the Financial Advisor (/health, /ask)
├── Dockerfile                     # uv-based production image
├── supervisord.conf               # Runs backend + frontend together in a container
├── pyproject.toml                 # Project metadata, dependencies, pytest config
├── uv.lock                        # Reproducible dependency lock (committed)
├── requirements.txt               # Flat dependency list (for pip users)
│
├── src/shankh/
│   ├── agents/
│   │   ├── financial_advisor.py   # Supervisor: Deep Agent + MCP client wiring
│   │   ├── agent.py               # Generic OpenAI agent factory (create_agent + middleware)
│   │   └── shared_tools.py        # Tavily web-search tool + tool filtering helpers
│   ├── mcp/
│   │   ├── server.py              # FastMCP server exposing the whole advisor
│   │   └── tools/
│   │       ├── macro.py           # MCP server :8001 — clustering + forensics
│   │       ├── market.py          # MCP server :8002 — HMM market regime
│   │       └── company.py         # MCP server :8003 — price-band forecasting
│   ├── ml/
│   │   ├── macro/                 # Stock clustering & forensic anomaly pipeline
│   │   ├── market/                # Market regime (HMM) pipeline
│   │   └── company/               # Price-band (LightGBM quantile) pipeline
│   └── utils.py                   # Prompt loading, response text extraction, model resolver
│
├── config/prompts/
│   ├── financial_advisor.md       # Supervisor system prompt
│   ├── macro_analyst.md           # Macro sub-agent prompt
│   ├── market_analyst.md          # Market sub-agent prompt
│   └── company_analyst.md         # Company sub-agent prompt
│
├── models/                        # Trained artifacts (committed so Docker includes them)
│   ├── clustering/                #   scaler + kmeans + isolation forest
│   ├── regime/                    #   scaler + HMM + metadata
│   └── price_band/lightgbm/       #   upper/lower boosters + feature list
│
├── data/universe/                 # Raw per-ticker OHLCV CSVs (git-ignored)
├── scripts/backtest_q1_2024.py    # Backtest of price-band models on Q1 2024
├── tests/
│   ├── unit/                      # MCP server + ML tool tests
│   └── integration/               # Live Financial Advisor tests
│
├── frontend/                      # Next.js 16 chat application
│   ├── app/page.tsx
│   ├── components/ChatAdvisorUI.tsx
│   ├── hooks/useFinancialAdvisor.tsx
│   └── lib/advisor-client.ts      # Typed API client for /ask and /health
│
├── docker/docker-compose.yml      # Local Postgres 16 for LangGraph checkpointing
├── docs/                          # Design notes, product brief, deployment guide
├── n8n/main.json                  # n8n workflow (optional visual orchestration)
└── DEPLOY.md                      # Step-by-step Azure Container Apps + Vercel deploy
```

---

## Prerequisites

- **Python 3.11+** (`.python-version` pins 3.11; the Docker image uses 3.13)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) **or** `pip`
- **Node.js 20+** and `npm` for the frontend
- API keys (see [Environment Variables](#environment-variables))
- Docker (optional — for Postgres / container deployment)

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd shankh
```

### 2. Create a virtual environment

Using `uv` (recommended):

```bash
uv venv
uv sync
```

> `uv sync` installs everything from `uv.lock` including dev/test dependencies.

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the pattern below into a `.env` file (the app auto-loads `.env` via `python-dotenv`):

```bash
# LLM provider (required)
OPENAI_API_KEY=sk-...
# Fallback providers (used by utils.resolve_model)
GOOGLE_API_KEY=...
CEREBRAS_API_KEY=...
MISTRALAI_API_KEY=...

# Web search (required for live news / web tools)
TAVILY_API_KEY=tvly-...

# Optional: Postgres URL for persistent conversation memory
# (falls back to in-memory when unset)
DATABASE_URL=postgresql://postgres:postgrespassword@localhost:5433/langgraph_db

# Optional: MCP server URLs (defaults shown)
MACRO_MCP_URL=http://localhost:8001/mcp
MARKET_MCP_URL=http://localhost:8002/mcp
PRICE_BAND_MCP_URL=http://localhost:8003/mcp

# Server host/port (FastAPI, defaults: 0.0.0.0 / 8000)
HOST=0.0.0.0
PORT=8000
```

### 4. Install the frontend

```bash
cd frontend
npm install
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Backend LLM (default model `gpt-4o`). |
| `TAVILY_API_KEY` | Yes¹ | Web-search tool for current news / macro events. |
| `GOOGLE_API_KEY` | No | Fallback LLM (`gemini-2.5-flash`) in `resolve_model`. |
| `CEREBRAS_API_KEY` | No | Optional `gpt-oss-120b` provider. |
| `MISTRALAI_API_KEY` | No | Optional `mistral-large-latest` provider. |
| `DATABASE_URL` | No | LangGraph `PostgresSaver` checkpointer. Unset → `MemorySaver`. |
| `MACRO_MCP_URL` | No | Macro MCP endpoint (default `http://localhost:8001/mcp`). |
| `MARKET_MCP_URL` | No | Market MCP endpoint (default `http://localhost:8002/mcp`). |
| `PRICE_BAND_MCP_URL` | No | Price-band MCP endpoint (default `http://localhost:8003/mcp`). |
| `HOST` / `PORT` | No | FastAPI bind settings (defaults `0.0.0.0` / `8000`). |
| `MCP_HOST` / `MCP_PORT` | No | MCP advisor server bind settings (defaults `0.0.0.0` / `8000`). |
| `NEXT_PUBLIC_SHANKH_API_URL` | Yes² | Frontend-only; base URL of the FastAPI server. |

¹ Without Tavily the web-search tool is skipped automatically (`filter_tools`).  
² The frontend throws at startup if this is missing — set it to `http://localhost:8000` in dev.

---

## Quick Start

### Option A — Run everything locally

```bash
# 1. Terminal 1 — Macro MCP server (clustering + forensics)
uv run python src/shankh/mcp/tools/macro.py        # port 8001

# 2. Terminal 2 — Market MCP server (regime HMM)
uv run python src/shankh/mcp/tools/market.py       # port 8002

# 3. Terminal 3 — Company MCP server (price bands)
uv run python src/shankh/mcp/tools/company.py      # port 8003

# 4. Terminal 4 — FastAPI backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Terminal 5 — Frontend (http://localhost:3000)
cd frontend && npm run dev
```

The MCP servers fail fast at startup if trained artifacts are missing — see [ML Pipelines](#ml-pipelines) to train first.

### Option B — With Docker Compose (backend + Postgres)

```bash
docker compose -f docker/docker-compose.yml up -d   # Postgres on :5433
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Smoke test the API

```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"Shankh Financial Advisor"}

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the current market regime for Indian equities?","thread_id":"demo"}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## ML Pipelines

Each pipeline follows the same structure under `src/shankh/ml/<module>/`:

| File | Purpose |
|---|---|
| `config.py` | Central config (paths, features, model hyperparams, artifacts). |
| `data_loader.py` | Reads per-ticker OHLCV CSVs from `data/universe/`. |
| `features.py` | Feature engineering shared by train/inference. |
| `model.py` | Model classes / builders. |
| `train*.py` / `tune*.py` | Training and Optuna hyperparameter-optimization entrypoints. |
| `inference.py` | Loads artifacts and runs predictions. |
| `validation.py` | Feature/data validation guards. |
| `evaluation.py` | Metrics and report persistence. |
| `tool.py` | LangChain tool wrapper for live inference (fetches live data via yfinance). |
| `README.md` | Module-specific deep documentation. |

### 1. Market Regime — `src/shankh/ml/market/`

Unsupervised **3-state Gaussian HMM** (`hmmlearn`) on cross-sectional market features to label the Indian equity market as **risk-on**, **range-bound**, or **risk-off**.

**Features**: annualized market volatility, breadth (% of universe above 20-DMA), and correlation density (mean pairwise return correlation). Features are standardized with `StandardScaler`; hidden states are deterministically sorted by volatility to produce stable, interpretable labels.

**Train / tune / infer:**

```bash
uv run python src/shankh/ml/market/train_regime.py   # end-to-end training
uv run python src/shankh/ml/market/tune_regime.py    # Optuna HPO (max log-likelihood)
uv run python src/shankh/ml/market/inference.py      # predict current regime
```

**Artifacts** (`models/regime/artifacts/`): `regime_model.joblib`, `regime_scaler.joblib`, `regime_metadata.json`, `historical_regimes.csv`.

### 2. Price-Band Forecasting — `src/shankh/ml/company/`

Predicts the **next trading day's high/low price band** for NSE stocks using two **LightGBM quantile-regression** boosters (pinball loss) — one at the 78th percentile for the high, one at the 22nd percentile for the low, calibrated for ~68% coverage.

**Features**: rolling windows (5/10/20/50), RSI-14, ATR-14, Bollinger Bands, volatility windows, and return lags (1–20).

**Training** uses date-aligned **walk-forward cross-validation** (5 folds, 1-day purging gap to prevent lookahead, ~5-year warmup).

```bash
uv run python -m shankh.ml.company.train            # full pipeline (train + eval + plots)
uv run python -m shankh.ml.company.tune             # Optuna HPO
uv run python -m shankh.ml.company.train --plot-only --ticker INFY.NS   # plot saved predictions
uv run python scripts/backtest_q1_2024.py           # backtest trained models on Q1 2024
```

**Artifacts** (`models/price_band/lightgbm/artifacts/`): `upper_model.txt`, `lower_model.txt`, `feature_cols.pkl`, `eval_report.json`, `predictions.parquet`, `feature_importance_*.csv`.

Post-processing guarantees (in `inference.py`): band ordering (`low ≤ high`), strictly positive prices, and a minimum band width of 0.1% of close.

### 3. Stock Clustering & Forensics — `src/shankh/ml/macro/`

Three complementary unsupervised models over a stock universe:

| Model | Answers | Math |
|---|---|---|
| **Agglomerative clustering** | "Who moves together?" | Correlation distance `d = √(2(1−ρ))`, average linkage |
| **K-Means clustering** | "Who has the same risk personality?" | 9-factor matrix, `RobustScaler`, K=4 |
| **Isolation Forest** | "Who is statistically abnormal?" | Contamination 0.10 → forensic flags |

**Factors**: annualized return, volatility, Sharpe (6.5% risk-free), skewness, max drawdown, beta, distance from 20-DMA, RSI-14, ATR%.

```bash
uv run python src/shankh/ml/macro/train_cluster.py  # end-to-end training
uv run python src/shankh/ml/macro/tune_cluster.py   # Optuna HPO (max silhouette)
uv run python src/shankh/ml/macro/inference.py      # run inference
```

**Artifacts** (`models/clustering/artifacts/`): `cluster_scaler.joblib`, `kmeans_cluster_model.joblib`, `isolation_forest_model.joblib`, `cluster_results.json`.

---

## MCP Servers

Each trained pipeline is exposed as a lightweight **FastMCP** server using Streamable HTTP. The advisor connects to all three at runtime and dynamically loads their tools — no agent code change needed when a model is retrained.

| Server | Module | Port | Tool | Returns |
|---|---|---|---|---|
| Macro | `src/shankh/mcp/tools/macro.py` | 8001 | `get_stock_clusters` | K-Means cluster id + peers, forensic anomaly flags |
| Market | `src/shankh/mcp/tools/market.py` | 8002 | `get_market_regime` | regime label, volatility, breadth, correlation density, recent history |
| Company | `src/shankh/mcp/tools/company.py` | 8003 | `query_gbm_price_band` | next-day predicted high/low, band width %, quantiles |

All three **eagerly load models into RAM at startup** (zero-latency inference) and **fail fast** with a clear message if artifacts are missing. Live data is fetched from yfinance on each request.

A fourth MCP server (`src/shankh/mcp/server.py`) exposes the whole Financial Advisor as a single `ask_financial_advisor(question, thread_id)` tool — useful for wiring Shankh into other MCP clients.

---

## Agents

### Supervisor — Financial Advisor

Built in `src/shankh/agents/financial_advisor.py` via `deepagents.create_deep_agent`:

- **Model**: OpenAI (`gpt-4o`, temperature 0) — configurable via `model_name`.
- **Checkpointer**: `PostgresSaver` when `DATABASE_URL` is set, otherwise `MemorySaver` (per-thread conversation memory).
- **Tools**: `search_web`, `get_market_regime`, `query_gbm_price_band`, `get_stock_clusters`.
- **Sub-agents**:

| Sub-agent | Focus | Tools |
|---|---|---|
| `macro-analyst` | RBI policy, inflation, rates, FX, crude, FII/DII, global macro | `search_web`, `get_stock_clusters` |
| `market-analyst` | Regime, volatility, breadth, sector rotation, cycles | `get_market_regime` |
| `company-analyst` | Stock-level research, price bands, peers, forensics | `search_web`, `query_gbm_price_band`, `get_stock_clusters` |

System prompts live in `config/prompts/` and follow a strict **data → analysis → implications → caveats** output structure.

### Standalone agent — `src/shankh/agents/agent.py`

A generic `create_agent` factory using OpenAI with:

- **Tool-error middleware** (`handle_tool_errors`) — converts a failing tool call into a `ToolMessage` so the model retries instead of crashing.
- **In-memory checkpointer** by default.

Run its interactive REPL:

```bash
uv run python -m shankh.agents.agent
```

---

## API Reference

### FastAPI backend (`main.py`)

| Method | Path | Body / Params | Description |
|---|---|---|---|
| GET | `/health` | — | Liveness check. |
| GET | `/` | — | Service info (points to `/docs`). |
| POST | `/ask` | `{ "question": str, "thread_id": str = "default" }` | Ask the Financial Advisor. Returns `{ "response": str }` (Markdown). |

Error codes: `503` if the advisor isn't initialized, `500` on agent failure (with detail).

### MCP advisor server (`src/shankh/mcp/server.py`)

| Tool | Args | Returns |
|---|---|---|
| `ask_financial_advisor` | `question: str`, `thread_id: str = "default"` | Markdown research response |

---

## Frontend

A Next.js 16 (React 19, Tailwind v4, shadcn-style components) chat interface in `frontend/`.

- **Entry**: `frontend/app/page.tsx` → `components/ChatAdvisorUI.tsx` (client component).
- **State**: `hooks/useFinancialAdvisor.tsx` manages messages, thread ids, and connection state.
- **API client**: `lib/advisor-client.ts` normalizes every failure mode (network, timeout, 503, 500, malformed JSON) into a single `AdvisorApiError` so the UI only branches on one type.
- **Environment**: `NEXT_PUBLIC_SHANKH_API_URL` (required — the app refuses to start without it).

```bash
cd frontend
npm run dev        # development  → http://localhost:3000
npm run build && npm run start   # production
npm run lint
```

---

## Running Tests

Pytest is configured in `pyproject.toml` (`pythonpath = ["src"]`, `testpaths = ["tests"]`, async auto mode).

```bash
uv run pytest -v          # unit tests
uv run pytest tests/integration -v   # live advisor tests (needs API keys + MCP servers)
```

> Integration tests call the real model and web-search tools, so they require running MCP servers and valid keys. Unit tests target the MCP servers and ML tools in isolation.

---

## Docker & Deployment

### Docker image

The `Dockerfile` is `uv`-based (`uv run uvicorn main:app`), bakes in `main.py`, `src/`, `config/`, and `models/`, and runs as a non-root user on port 8000.

### Local Postgres (optional)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Spins up `postgres:16-alpine` on port **5433** (mapped), database `langgraph_db`, user `postgres` / password `postgrespassword`. Set `DATABASE_URL=postgresql://postgres:postgrespassword@localhost:5433/langgraph_db` for persistent conversation checkpoints.

### Production deployment

See **[DEPLOY.md](DEPLOY.md)** for a complete walkthrough:

- **Backend** → Azure Container Apps (`az acr build` + `az containerapp create`, secrets via env-var references).
- **Frontend** → Vercel with `NEXT_PUBLIC_SHANKH_API_URL` pointing at the backend FQDN.
- `supervisord.conf` runs backend (`uvicorn`, :8000) and frontend (`node server.js`, :3000) together in a single container when preferred.

---

## Project Documentation

| File | Contents |
|---|---|
| [DEPLOY.md](DEPLOY.md) | Azure Container Apps + Vercel deployment guide. |
| [docs/design.md](docs/design.md) | Agent/tool/ML architecture decisions and boundaries. |
| [docs/product.md](docs/product.md) | Product brief: workstreams, tools, and deliverables. |
| [docs/future_real_data_tools_todo.md](docs/future_real_data_tools_todo.md) | Roadmap for replacing yfinance with production market-data feeds. |
| `src/shankh/ml/{macro,market,company}/README.md` | Deep math/model documentation per pipeline. |
