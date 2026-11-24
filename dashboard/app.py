import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from src.scm_engine import StructuralCausalTwin

st.set_page_config(
    page_title="FLUX-CAUSAL-TWIN | Counterfactual Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 12% 10%, #131c31 0%, #090d16 55%, #06080f 100%);
        color: #e2e8f0;
    }
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(56, 189, 248, 0.10));
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8, #34d399, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 16px 18px;
        backdrop-filter: blur(10px);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-delta-good {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #34d399;
    }
    .kpi-delta-bad {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #fb7185;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_twin():
    twin = StructuralCausalTwin(seed=42)
    df = twin.generate_observational_data(n_samples=2500)
    twin.fit(df)
    return twin, df


twin, df = load_twin()

st.sidebar.markdown("## 🎛️ Pearl `do(X = x)` Console")
st.sidebar.caption("Simulate surgical infrastructure interventions by mutilating incoming DAG edges.")

scenario = st.sidebar.selectbox(
    "Baseline Telemetry Regime",
    ["Flash Sale Traffic Spike (High Load)", "Normal Production Steady-State", "Cache Cold-Start Degradation"],
)

if scenario == "Flash Sale Traffic Spike (High Load)":
    base_rps, base_pool, base_cache = 1480.0, 32.0, 71.0
elif scenario == "Cache Cold-Start Degradation":
    base_rps, base_pool, base_cache = 1050.0, 40.0, 56.0
else:
    base_rps, base_pool, base_cache = 820.0, 48.0, 86.0

baseline = {
    "traffic_rps": base_rps,
    "db_pool_size": base_pool,
    "cache_hit_rate": base_cache,
    "cpu_util": min(96.0, 18.0 + 0.036 * base_rps + 0.000004 * (base_rps ** 2)),
    "db_lock_wait_ms": max(2.0, 42.0 - 0.38 * base_pool - 0.42 * base_cache + 0.008 * base_rps),
    "gc_pause_ms": 24.0,
    "p99_latency_ms": 168.0,
    "infra_cost_hr": 12.0 + 0.34 * base_pool + 0.26 * 65.0,
    "sla_breach_prob": 0.72,
}
baseline_sim = twin.intervene(
    baseline,
    {"traffic_rps": base_rps, "db_pool_size": base_pool, "cache_hit_rate": base_cache},
    n_mc=250,
)
for k, v in baseline_sim.items():
    baseline[k] = v["mean"]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Active Interventions")
do_pool = st.sidebar.checkbox("Intervene: `do(db_pool_size)`", value=True)
pool_val = st.sidebar.slider("Target DB Connection Pool", 15, 110, 72, disabled=not do_pool)

do_cache = st.sidebar.checkbox("Intervene: `do(cache_hit_rate)`", value=True)
cache_val = st.sidebar.slider("Target Redis Cache Hit (%)", 45.0, 98.5, 92.5, step=0.5, disabled=not do_cache)

do_rps = st.sidebar.checkbox("Stress Test: `do(traffic_rps)`", value=False)
rps_val = st.sidebar.slider("Injected Traffic (RPS)", 200, 2400, int(base_rps), step=50, disabled=not do_rps)

interventions = {}
if do_pool:
    interventions["db_pool_size"] = float(pool_val)
if do_cache:
    interventions["cache_hit_rate"] = float(cache_val)
if do_rps:
    interventions["traffic_rps"] = float(rps_val)

cf = twin.intervene(baseline, interventions, n_mc=350)

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">FLUX-CAUSAL-TWIN // Counterfactual Digital Twin</div>
        <div class="hero-sub">
            Structural Causal Modeling (SCM) • Chernozhukov Double Machine Learning (DML) • Pearl do-Calculus Inference Engine
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

def render_kpi(col, label, val_str, delta_val, unit="", invert_good=True):
    is_good = (delta_val <= 0) if invert_good else (delta_val >= 0)
    cls = "kpi-delta-good" if is_good else "kpi-delta-bad"
    sign = "+" if delta_val > 0 else ""
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val_str}</div>
            <div class="{cls}">{sign}{delta_val:.2f}{unit} vs baseline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_kpi(
    c1,
    "Counterfactual P99 Latency",
    f"{cf['p99_latency_ms']['mean']:.1f} ms",
    cf["p99_latency_ms"]["delta"],
    " ms",
    invert_good=True,
)
render_kpi(
    c2,
    "SLA Breach Risk (>145ms)",
    f"{cf['sla_breach_prob']['mean']*100:.1f}%",
    cf["sla_breach_prob"]["delta"] * 100.0,
    "%",
    invert_good=True,
)
render_kpi(
    c3,
    "DB Lock Contention",
    f"{cf['db_lock_wait_ms']['mean']:.1f} ms",
    cf["db_lock_wait_ms"]["delta"],
    " ms",
    invert_good=True,
)
render_kpi(
    c4,
    "Cluster Burn Rate",
    f"${cf['infra_cost_hr']['mean']:.2f}/hr",
    cf["infra_cost_hr"]["delta"],
    "/hr",
    invert_good=True,
)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["🕸️ Causal DAG & Mutilation Topology", "📊 Counterfactual Distributions & DML", "🎯 Pareto Policy Optimizer"]
)

with tab1:
    col_dag, col_info = st.columns([2.1, 1.0])
    with col_dag:
        pos = {
            "traffic_rps": (0.0, 2.2),
            "db_pool_size": (0.0, 0.2),
            "cache_hit_rate": (1.4, 0.8),
            "cpu_util": (1.4, 2.6),
            "db_lock_wait_ms": (2.8, 0.3),
            "gc_pause_ms": (2.8, 2.5),
            "p99_latency_ms": (4.2, 1.5),
            "infra_cost_hr": (2.8, -0.7),
            "sla_breach_prob": (5.5, 1.5),
        }

        edge_x, edge_y = [], []
        mutilated_x, mutilated_y = [], []
        for u, v in twin.graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            if v in interventions:
                mutilated_x.extend([x0, x1, None])
                mutilated_y.extend([y0, y1, None])
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        fig_dag = go.Figure()
        fig_dag.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2.2, color="rgba(56, 189, 248, 0.55)"),
                hoverinfo="none",
                mode="lines",
                name="Active Causal Mechanism",
            )
        )
        if mutilated_x:
            fig_dag.add_trace(
                go.Scatter(
                    x=mutilated_x,
                    y=mutilated_y,
                    line=dict(width=2.0, color="rgba(251, 113, 133, 0.55)", dash="dot"),
                    hoverinfo="none",
                    mode="lines",
                    name="Mutilated by do(X=x)",
                )
            )

        node_x, node_y, node_text, node_color, node_hover = [], [], [], [], []
        for node in twin.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            val = cf[node]["mean"]
            delta = cf[node]["delta"]
            node_text.append(f"<b>{node}</b><br>{val:.2f}")
            if node in interventions:
                node_color.append("#f59e0b")
            elif node in ("p99_latency_ms", "sla_breach_prob"):
                node_color.append("#34d399")
            else:
                node_color.append("#38bdf8")
            node_hover.append(
                f"Node: {node}<br>Counterfactual Mean: {val:.3f}<br>Delta vs Baseline: {delta:+.3f}"
            )

        fig_dag.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="top center",
                hovertext=node_hover,
                hoverinfo="text",
                marker=dict(
                    size=26,
                    color=node_color,
                    line=dict(width=2, color="#0f172a"),
                ),
                name="SCM Variables",
            )
        )
        fig_dag.update_layout(
            title="Structural Causal Graph (Amber = Intervened Node, Dotted = Severed Edge)",
            showlegend=True,
            plot_bgcolor="rgba(15,23,42,0.45)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 6.2]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 3.3]),
            height=450,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig_dag, use_container_width=True)

    with col_info:
        st.markdown("### 🔬 Graph Surgery Log")
        if interventions:
            for k, v in interventions.items():
                incoming = list(twin.graph.predecessors(k))
                st.info(
                    f"**`do({k} = {v})`**\n\n"
                    f"Severed incoming confounders: `{incoming if incoming else 'Root Exogenous'}`"
                )
        else:
            st.warning("No active `do()` interventions. Showing observational baseline.")

        st.markdown("### 📐 DML Orthogonal ATEs")
        ate_rows = []
        for (src, dst), stats in twin.ate_estimates.items():
            ate_rows.append(
                {
                    "Treatment → Outcome": f"{src} → {dst}",
                    "ATE (θ)": stats["ate"],
                    "95% CI": f"[{stats['ci_low']}, {stats['ci_high']}]",
                }
            )
        st.dataframe(pd.DataFrame(ate_rows), use_container_width=True, hide_index=True)

with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        metrics = ["p99_latency_ms", "db_lock_wait_ms", "gc_pause_ms", "infra_cost_hr"]
        comp_df = pd.DataFrame(
            {
                "Metric": metrics * 2,
                "Regime": ["Observational Baseline"] * 4 + ["Counterfactual do(X=x)"] * 4,
                "Value": [baseline[m] for m in metrics] + [cf[m]["mean"] for m in metrics],
            }
        )
        fig_bar = px.bar(
            comp_df,
            x="Metric",
            y="Value",
            color="Regime",
            barmode="group",
            color_discrete_map={
                "Observational Baseline": "#64748b",
                "Counterfactual do(X=x)": "#34d399",
            },
            title="Baseline vs. Counterfactual Twin Shift",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(15,23,42,0.45)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        fig_scatter = px.scatter(
            df.sample(600, random_state=42),
            x="cache_hit_rate",
            y="p99_latency_ms",
            color="db_pool_size",
            color_continuous_scale="Tealgrn",
            title="Observational Telemetry Manifold (Confounded by Traffic)",
            opacity=0.78,
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[cf["cache_hit_rate"]["mean"]],
                y=[cf["p99_latency_ms"]["mean"]],
                mode="markers+text",
                text=["Counterfactual Twin"],
                textposition="top center",
                marker=dict(size=16, color="#f59e0b", symbol="star"),
                name="do(X=x) State",
            )
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(15,23,42,0.45)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=380,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### 🎯 Automated Counterfactual Policy Search (Pareto Frontier)")
    budget = st.slider("Max Hourly Infrastructure Budget ($/hr)", 30.0, 95.0, 62.0, step=2.0)
    pareto_policies = twin.find_pareto_optimal_policy(baseline, max_budget_hr=budget)
    if pareto_policies:
        p_df = pd.DataFrame(pareto_policies)
        st.dataframe(p_df, use_container_width=True, hide_index=True)
        best = pareto_policies[0]
        st.success(
            f"**Recommended Surgical Policy:** Set `db_pool_size = {best['db_pool_size']}` and "
            f"`cache_hit_rate = {best['cache_hit_rate']}%` → Projected P99 Latency: "
            f"**{best['p99_latency_ms']} ms** (SLA Breach Risk: **{best['sla_breach_pct']}%**) at **${best['infra_cost_hr']}/hr**."
        )
    else:
        st.error("No feasible policy under selected budget ceiling. Increase hourly budget.")
