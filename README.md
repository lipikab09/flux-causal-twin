# 🧬 FLUX-CAUSAL-TWIN

**Structural Causal Model (SCM), Double Machine Learning (DML), and Pearl `do`-Calculus Counterfactual Digital Twin for Cloud Infrastructure Optimization.**

[![CI Pipeline](https://github.com/lipikab09/flux-causal-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/lipikab09/flux-causal-twin/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-38bdf8)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%20%2B%20Plotly-34d399)
![FastAPI](https://img.shields.io/badge/API-FastAPI-a78bfa)
![Causal AI](https://img.shields.io/badge/Causal%20Inference-Pearl%20do--Calculus%20%7C%20DML-f59e0b)

---

## 🚀 Project Overview

Standard predictive ML models fail under infrastructure interventions because they conflate **observational correlation** with **causal effect**. For example, increasing database connection pool sizes (`db_pool_size`) often correlates positively with `p99_latency_ms` in production logs because autoscalers expand connection pools during high-load traffic bursts—an unobserved confounding bias.

**FLUX-CAUSAL-TWIN** replaces naive correlation modeling with a **Counterfactual Digital Twin** built on three causal inference pillars:

1. **Structural Causal Modeling (SCM):** Encodes a 9-node, 13-edge Directed Acyclic Graph (DAG) capturing mechanistic microservice telemetry (`traffic_rps`, `db_pool_size`, `cache_hit_rate`, `cpu_util`, `db_lock_wait_ms`, `gc_pause_ms`, `p99_latency_ms`, `infra_cost_hr`, `sla_breach_prob`) using non-linear Gradient Boosted structural equations.
2. **Chernozhukov Double Machine Learning (DML):** Computes orthogonalized residual-on-residual Average Treatment Effects ($\theta$) with 95% confidence intervals to isolate true causal impact from latent traffic burstiness confounding.
3. **Pearl's `do(X = x)` Graph Surgery & Bootstrapped Counterfactuals:** Mutilates incoming causal edges to intervened variables, propagates downstream structural mechanisms in topological order, and resamples empirical structural residuals to produce calibrated `P10`–`P90` uncertainty intervals.
4. **Pareto Policy Optimizer:** Searches the counterfactual intervention space across connection pool and cache hit configurations to minimize P99 latency and SLA breach risk (`>145 ms`) under strict hourly cloud infrastructure budgets.

---

## 🏗️ System Architecture & Causal DAG

```text
  traffic_rps ──────────┬───────────────────────► cpu_util ───────► gc_pause_ms
       │                │                            │                   │
       │                ▼                            ▼                   ▼
       │         cache_hit_rate ◄── db_pool_size ──► infra_cost_hr   p99_latency_ms ──► sla_breach_prob
       │                │                │                               ▲
       │                ▼                ▼                               │
       └────────► db_lock_wait_ms ◄──────┴───────────────────────────────┘


📂 Repository File Structure

       flux-causal-twin/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow running automated pytest checks
├── dashboard/
│   └── app.py                  # Interactive Streamlit + Plotly dark-glass causal console
├── src/
│   ├── __init__.py             # Package metadata and version export
│   ├── api.py                  # Production FastAPI inference & intervention REST endpoints
│   └── scm_engine.py           # StructuralCausalTwin engine (DAG, DML estimator, do-calculus, Pareto search)
├── tests/
│   └── test_scm.py             # Unit tests verifying DAG acyclicity, DML signs, and counterfactual shifts
├── .gitignore                  # Excludes virtual environments, caches, and build artifacts
├── pyproject.toml              # Pytest configuration and root pythonpath resolution
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # Project documentation and quickstart guide