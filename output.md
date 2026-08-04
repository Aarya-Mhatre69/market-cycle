---

# 1. Entity Table

| Entity             | Description                  | Primary Key    |
| ------------------ | ---------------------------- | -------------- |
| Company            | Listed company               | company_id     |
| Stock              | Tradable equity              | ticker         |
| Index              | Market index                 | index_id       |
| Sector             | Economic sector              | sector_id      |
| Industry           | Industry classification      | industry_id    |
| Country            | Nation                       | country_code   |
| Exchange           | Trading exchange             | exchange_id    |
| Currency           | Currency                     | currency_code  |
| FinancialStatement | Quarterly/Annual report      | statement_id   |
| FinancialMetric    | Revenue, EPS, EBITDA etc     | metric_id      |
| MacroIndicator     | CPI, GDP, Repo Rate, PMI     | indicator_id   |
| MacroObservation   | One value of macro indicator | observation_id |
| EconomicEvent      | RBI Meeting, FOMC etc        | event_id       |
| NewsArticle        | News document                | news_id        |
| NewsSource         | Bloomberg, Reuters           | source_id      |
| ResearchReport     | Broker report                | report_id      |
| Analyst            | Human analyst                | analyst_id     |
| Institution        | Bank, Fund, Brokerage        | institution_id |
| ETF                | ETF instrument               | etf_id         |
| MutualFund         | Mutual fund                  | mf_id          |
| Holding            | Portfolio holding            | holding_id     |
| FIIFlow            | Foreign investment flow      | fii_id         |
| DIIFlow            | Domestic investment flow     | dii_id         |
| Commodity          | Gold, Oil etc                | commodity_id   |
| Bond               | Government bond              | bond_id        |
| YieldCurve         | Treasury curve               | curve_id       |
| Option             | Derivative contract          | option_id      |
| Future             | Futures contract             | future_id      |
| Strategy           | Trading strategy             | strategy_id    |
| Signal             | Buy/Sell/Hold                | signal_id      |
| MLModel            | Forecasting model            | model_id       |
| Embedding          | Vector embedding             | embedding_id   |
| Document           | Raw text                     | document_id    |
| Chunk              | Document chunk               | chunk_id       |
| KnowledgeNode      | Generic graph node           | node_id        |
| User               | Platform user                | user_id        |
| Portfolio          | Investment portfolio         | portfolio_id   |

---

# 2. Attribute Table

| Entity           | Important Attributes                    |
| ---------------- | --------------------------------------- |
| Company          | name, ticker, isin, market_cap, sector  |
| Stock            | open high low close volume adj_close    |
| MacroIndicator   | name, frequency, unit, country          |
| MacroObservation | timestamp, value                        |
| NewsArticle      | title, summary, text, url, published_at |
| ResearchReport   | title, recommendation, target_price     |
| FinancialMetric  | metric_name, value, period              |
| Prediction       | horizon, confidence, predicted_value    |
| Signal           | type, probability, timestamp            |
| Strategy         | rules, timeframe, asset_class           |
| MLModel          | architecture, version, metrics          |
| Embedding        | vector_dimension, model_name            |
| Chunk            | text, embedding_id                      |
| Portfolio        | name, owner, value                      |
| Holding          | quantity, avg_price                     |

---

# 3. Relationship Table

| Source             | Relationship    | Target             | Cardinality |
| ------------------ | --------------- | ------------------ | ----------- |
| Company            | listed_on       | Exchange           | N:1         |
| Company            | belongs_to      | Sector             | N:1         |
| Sector             | contains        | Industry           | 1:N         |
| Company            | issues          | Stock              | 1:N         |
| Stock              | trades_in       | Currency           | N:1         |
| Company            | publishes       | FinancialStatement | 1:N         |
| FinancialStatement | contains        | FinancialMetric    | 1:N         |
| MacroIndicator     | has_observation | MacroObservation   | 1:N         |
| NewsArticle        | mentions        | Company            | N:M         |
| NewsArticle        | discusses       | MacroIndicator     | N:M         |
| NewsArticle        | cites           | Institution        | N:M         |
| ResearchReport     | covers          | Company            | N:M         |
| Analyst            | authors         | ResearchReport     | 1:N         |
| Institution        | employs         | Analyst            | 1:N         |
| Institution        | owns            | Holding            | 1:N         |
| Holding            | references      | Stock              | N:1         |
| Portfolio          | contains        | Holding            | 1:N         |
| User               | owns            | Portfolio          | 1:N         |
| User               | follows         | Company            | N:M         |
| Watchlist          | contains        | Company            | N:M         |
| Strategy           | generates       | Signal             | 1:N         |
| Signal             | recommends      | Stock              | N:1         |
| MLModel            | creates         | Prediction         | 1:N         |
| Prediction         | predicts        | Stock              | N:1         |
| Prediction         | uses            | MacroIndicator     | N:M         |
| Prediction         | uses            | FinancialMetric    | N:M         |
| Document           | split_into      | Chunk              | 1:N         |
| Chunk              | embedded_as     | Embedding          | 1:1         |
| Chunk              | references      | Company            | N:M         |
| Commodity          | influences      | Company            | N:M         |
| YieldCurve         | affects         | Sector             | N:M         |
| EconomicEvent      | impacts         | Market             | N:M         |
| FIIFlow            | affects         | Index              | N:M         |
| DIIFlow            | affects         | Index              | N:M         |

---

# 4. Ontology

```
Thing
│
├── FinancialEntity
│   ├── Company
│   ├── Stock
│   ├── ETF
│   ├── MutualFund
│   ├── Bond
│   ├── Option
│   ├── Future
│   └── Index
│
├── Organization
│   ├── Institution
│   ├── Exchange
│   └── Regulator
│
├── EconomicEntity
│   ├── MacroIndicator
│   ├── Commodity
│   ├── Currency
│   ├── YieldCurve
│   └── EconomicEvent
│
├── Information
│   ├── NewsArticle
│   ├── ResearchReport
│   ├── FinancialStatement
│   ├── Document
│   ├── Chunk
│   └── Embedding
│
├── Intelligence
│   ├── Prediction
│   ├── Signal
│   ├── Strategy
│   ├── MLModel
│   └── Agent
│
├── UserDomain
│   ├── User
│   ├── Portfolio
│   ├── Holding
│   ├── Watchlist
│   └── Alert
│
└── Geography
    ├── Country
    ├── Sector
    └── Industry
```

---

# 5. Knowledge Graph

```text
                      Country
                          │
             publishes macro indicators
                          │
                          ▼
                  MacroIndicator
                          │
                    hasObservation
                          │
                          ▼
                 MacroObservation
                          │
                      influences
                          │
                          ▼
Commodity ─────► Company ◄──────── NewsArticle
                   │  ▲                ▲
           belongs_to │          mentions
                   │  │
                Sector │
                   │   │
                   ▼   │
               Industry│
                   │   │
                   ▼   │
                 Stock ◄──────────── Strategy
                   │                     │
              predicted_by              generates
                   ▲                     │
              Prediction ◄────────── MLModel
                   ▲
                   │
              uses features
       ┌───────────┴─────────────┐
       ▼                         ▼
FinancialMetric         MacroObservation

Document
   │
split_into
   ▼
Chunk
   │
embedded_as
   ▼
Embedding

User
 │
owns
 ▼
Portfolio
 │
contains
 ▼
Holding
 │
references
 ▼
Stock

Agent ──uses──► Tool ──retrieves──► News / Macro / Financials
```

---

# 6. Ontology Constraints

| Rule                                               | Constraint |
| -------------------------------------------------- | ---------- |
| Company must belong to exactly one Sector          | Mandatory  |
| Stock must reference one Company                   | Mandatory  |
| MacroObservation must belong to one MacroIndicator | Mandatory  |
| Prediction must reference one MLModel              | Mandatory  |
| Holding cannot exist without Portfolio             | Mandatory  |
| Embedding cannot exist without Chunk               | Mandatory  |
| Chunk belongs to one Document                      | Mandatory  |
| FinancialMetric belongs to one FinancialStatement  | Mandatory  |
| NewsArticle has one Source                         | Mandatory  |
| Signal generated by one Strategy                   | Mandatory  |

---

# 7. Knowledge Graph Validation Report

## Structural Validation

| Check                    | Status |
| ------------------------ | ------ |
| Duplicate entities       | PASS   |
| Missing primary keys     | PASS   |
| Circular ownership       | PASS   |
| Dangling relationships   | PASS   |
| Consistent cardinalities | PASS   |

---

## Graph Quality Metrics

| Metric                    | Value                                      |
| ------------------------- | ------------------------------------------ |
| Entity Types              | 36                                         |
| Relationship Types        | 34                                         |
| Core Classes              | 7                                          |
| Average Node Degree       | High                                       |
| Graph Connectivity        | Strongly Connected (through Company/Stock) |
| Suitable for RDF/OWL      | Yes                                        |
| Suitable for Neo4j        | Yes                                        |
| Suitable for GraphRAG     | Yes                                        |
| Suitable for Agent Memory | Yes                                        |
| Extensible                | High                                       |
