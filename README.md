# ✈️ Aviation MRO Supply Chain Analytics

> End-to-end network analysis, ML demand forecasting, inventory optimisation,
> and disruption risk simulation for aviation MRO spare-parts supply chains.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen?style=flat-square&logo=streamlit)](https://YOUR_STREAMLIT_URL)
[![GitHub Pages](https://img.shields.io/badge/Portfolio%20Site-Live-blue?style=flat-square&logo=github)](https://YOUR_USERNAME.github.io/aviation-supply-chain)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-94%20passed-brightgreen?style=flat-square&logo=pytest)](tests/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.x-orange?style=flat-square)](https://networkx.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Problem Statement

Aviation's MRO supply chain faces $11B+ disruption costs in 2025–2026 — aircraft
delivery backlogs hit a record 17,000 jets. Yet most MRO operations still rely on
static Excel-based inventory management with no network visibility or predictive
analytics. This project builds a full end-to-end analytics solution:

**Hypothesis:** Supply disruptions concentrate at a small number of high-betweenness-
centrality supplier nodes. Buffering these nodes reduces AOG risk >30% while increasing
inventory cost <15%.

---

## 📊 Key Results

| Finding | Value |
|---------|-------|
| 🕸 Network bottleneck impact | Removing 5 suppliers disrupts **39% of supply edges** + increases path length **45.3%** |
| 📈 XGBoost forecast MAPE | **2.05%** vs ARIMA baseline 13.11% (**84.4% improvement**) |
| 📦 Inventory saving/year | **$4.37M (24%)** vs naive ordering policy across 196 SKUs |
| 🛣️ LP route saving | **$13.1M (60.7%)** vs current historical routing policy |
| 🏷️ Supplier risk accuracy | **92% OOB** (Random Forest) — surfaces GE Aviation as hidden high-risk |
| ⚠️ Longest supply path | **194 days** (Rolls-Royce → Qantas via Precision Castparts) |

---

## 🌐 Live Demos

| Link | Description |
|------|-------------|
| [🚀 Streamlit Dashboard](https://YOUR_STREAMLIT_URL) | 5-page interactive analytics app |
| [🕸 Interactive Network](https://YOUR_USERNAME.github.io/aviation-supply-chain/network.html) | 32-node Pyvis supply chain graph |
| [🌍 Portfolio Site](https://YOUR_USERNAME.github.io/aviation-supply-chain) | Full project landing page |
| [📄 PDF Report](https://YOUR_USERNAME.github.io/aviation-supply-chain/assets/Aviation_SC_Report.pdf) | Executive summary report |

---

## 🗄 Data Sources

| Dataset | Source | Used for |
|---------|--------|----------|
| Aerospace Supply Chain Performance & Forecasting | [Kaggle](https://www.kaggle.com/datasets/robertocarlost/aerospace-supply-chain-performance-and-forecasting) | Supplier KPIs, lead times, OTD% |
| BTS Airline On-Time Performance | [bts.gov](https://www.transtats.bts.gov/) | MRO demand signals from delays |
| Global Aviation Safety 1970–2024 | [Kaggle](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses) | AOG incident correlation |
| Synthetic MRO Inventory | [supplychainpy](https://github.com/KevinFasusi/supplychainpy) | 500 SKUs × 24 months baseline |
| OpenSky Network API | [opensky-network.org](https://opensky-network.org/) | Aircraft utilisation signals |
| World Bank LPI | [worldbank.org](https://lpi.worldbank.org/) | Country logistics risk scores |

---

## 🔬 Methodology

### Phase 3 — Network Modelling
Built a 32-node directed supply chain graph (NetworkX + Pyvis) spanning 5 tiers:
OEM → Tier-1 → Tier-2 → MRO → Airline. Computed betweenness centrality, PageRank,
composite risk scores, and ran cascading failure simulations.

### Phase 4 — ML & Optimisation
- **Demand Forecasting:** ARIMA(2,1,2) baseline vs XGBoost with 16 engineered features
  (lag demand, rolling stats, seasonality, part criticality, cost). XGBoost: 2.05% MAPE.
- **Inventory EOQ:** Safety stock sized by criticality tier (Z=2.576 for AOG-critical,
  Z=1.645 for standard). Reorder point = demand during lead time + safety stock.
- **Route LP:** PuLP transportation problem minimising total logistics cost subject to
  supply, demand, and lead-time SLA constraints. CBC solver: optimal in <0.01s.
- **Supplier Risk:** 6-factor weighted rule-based score + Random Forest classifier (OOB=92%).

---

## 🗂 Project Structure

```
aviation-supply-chain/
├── .github/workflows/
│   ├── deploy.yml          # Auto-run notebooks + deploy GitHub Pages on push to main
│   └── lint.yml            # Black + isort + flake8 on push to dev/main
├── data/
│   ├── raw/                # Downloaded datasets (gitignored if >50MB)
│   ├── processed/          # Cleaned CSVs + parquet
│   └── synthetic/          # Generated MRO inventory data (500 SKUs × 24 months)
├── notebooks/
│   ├── 01_EDA_Aviation_Data.ipynb
│   ├── 02_Network_Construction.ipynb
│   ├── 03_Demand_Forecasting.ipynb
│   ├── 04_Inventory_Optimisation.ipynb
│   ├── 05_Disruption_Analysis.ipynb
│   └── 06_Final_Report.ipynb
├── src/
│   ├── data/
│   │   ├── ingest.py       # Dataset download + synthetic MRO generator
│   │   └── clean.py        # ETL cleaning + validation
│   ├── network/
│   │   ├── build_graph.py  # 32-node supply chain DiGraph
│   │   ├── metrics.py      # Centrality, disruption simulation, critical paths
│   │   └── visualise.py    # Pyvis dark-theme interactive export
│   └── models/
│       ├── forecasting.py  # ARIMA + XGBoost demand models
│       ├── inventory.py    # EOQ + safety stock + reorder point
│       ├── optimisation.py # PuLP LP transportation problem
│       └── risk_score.py   # Weighted rule score + Random Forest
├── app/
│   ├── streamlit_app.py    # Entry point: router + shared sidebar
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Network_Graph.py
│       ├── 3_Demand_Forecast.py
│       ├── 4_Inventory.py
│       └── 5_Risk_Routing.py
├── docs/                   # GitHub Pages source
│   ├── index.html          # Portfolio landing page
│   ├── network.html        # Interactive Pyvis network graph
│   └── assets/
├── tests/
│   ├── test_graph.py       # 62 graph tests (construction, metrics, simulation)
│   ├── test_inventory.py   # 24 inventory tests (EOQ, safety stock, optimisation)
│   └── test_forecasting.py # 21 forecasting tests (features, XGBoost, comparison)
├── requirements.txt
├── environment.yml
└── Makefile
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aviation-supply-chain.git
cd aviation-supply-chain

# 2. Set up environment
conda env create -f environment.yml
conda activate aviation-sc

# 3. Generate synthetic data (no API keys needed)
make data

# 4. Launch the Streamlit dashboard
make run
# → Opens at http://localhost:8501

# 5. Run unit tests
make test
# → 94 tests, 0 failures

# 6. Export notebooks to HTML (for GitHub Pages)
make notebooks
```

---

## 🔧 Tech Stack

| Layer | Tools |
|-------|-------|
| Network Analysis | NetworkX 3.x, Pyvis |
| ML / Forecasting | XGBoost 2.0, statsmodels (ARIMA), scikit-learn |
| Optimisation | PuLP (CBC solver) |
| Data | pandas, NumPy, SciPy, supplychainpy |
| Visualisation | Plotly, Streamlit |
| Deployment | GitHub Actions, GitHub Pages, Streamlit Cloud |
| Testing | pytest 9.x, pytest-cov |

---

## 👤 Author

**Srivatsa** · Aviation Supply Chain Analytics Portfolio Project · 2025–2026

[🔗 LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [🌍 Portfolio](https://YOUR_USERNAME.github.io/aviation-supply-chain) · [📧 Email](mailto:your@email.com)

---

*Tags: `aviation` `supply-chain` `mro` `network-analysis` `xgboost` `linear-programming` `streamlit` `python` `data-science` `portfolio`*
