import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CausalEdge:
    source: str
    target: str
    mechanism: str
    true_ate: float
    estimated_ate: float = 0.0
    std_err: float = 0.0


class StructuralCausalTwin:
    """
    Structural Causal Model (SCM) and Double Machine Learning (DML) Counterfactual Twin
    for high-stakes microservice & cloud infrastructure latency/cost optimization.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.graph = nx.DiGraph()
        self.models: Dict[str, GradientBoostingRegressor] = {}
        self.noise_residuals: Dict[str, np.ndarray] = {}
        self.ate_estimates: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._build_dag()

    def _build_dag(self):
        edges = [
            ("traffic_rps", "cpu_util", "Non-linear saturation", 0.042),
            ("traffic_rps", "cache_hit_rate", "Working-set eviction", -0.018),
            ("traffic_rps", "p99_latency_ms", "Queueing delay", 0.055),
            ("db_pool_size", "cache_hit_rate", "Connection warmth", 0.25),
            ("db_pool_size", "db_lock_wait_ms", "Contention reduction", -0.45),
            ("db_pool_size", "infra_cost_hr", "Provisioning cost", 0.35),
            ("cache_hit_rate", "db_lock_wait_ms", "Read offloading", -0.60),
            ("cache_hit_rate", "p99_latency_ms", "Fast-path memory hit", -1.40),
            ("cpu_util", "gc_pause_ms", "Heap pressure", 0.48),
            ("cpu_util", "infra_cost_hr", "Autoscaling compute", 0.28),
            ("db_lock_wait_ms", "p99_latency_ms", "Lock tail amplification", 1.85),
            ("gc_pause_ms", "p99_latency_ms", "Stop-the-world stalls", 1.35),
            ("p99_latency_ms", "sla_breach_prob", "Sigmoid SLA threshold", 0.015),
        ]
        for u, v, mech, ate in edges:
            self.graph.add_edge(u, v, mechanism=mech, true_ate=ate)

    def generate_observational_data(self, n_samples: int = 2500) -> pd.DataFrame:
        """Generates realistic telemetry honoring structural equations + unobserved confounding."""
        rng = np.random.default_rng(self.seed)
        u_burst = rng.normal(0, 1, n_samples)

        traffic_rps = np.clip(rng.gamma(shape=5.0, scale=180.0, size=n_samples) + 90 * u_burst, 150, 2500)
        db_pool_size = np.clip(rng.normal(45, 12, size=n_samples) + 4 * u_burst, 10, 120)

        cache_hit_rate = np.clip(
            86.0 - 0.014 * traffic_rps + 0.22 * db_pool_size - 1.8 * u_burst + rng.normal(0, 2.5, n_samples),
            35.0,
            99.5,
        )
        cpu_util = np.clip(
            18.0 + 0.036 * traffic_rps + 0.000004 * (traffic_rps ** 2) + 2.5 * u_burst + rng.normal(0, 3.2, n_samples),
            5.0,
            99.8,
        )
        db_lock_wait_ms = np.clip(
            42.0 - 0.38 * db_pool_size - 0.42 * cache_hit_rate + 0.008 * traffic_rps + rng.normal(0, 2.8, n_samples),
            1.0,
            120.0,
        )
        gc_pause_ms = np.clip(
            4.0 + 0.35 * cpu_util + 0.0045 * np.maximum(0, cpu_util - 65.0) ** 2 + rng.normal(0, 2.0, n_samples),
            1.0,
            150.0,
        )
        p99_latency_ms = np.clip(
            28.0
            + 0.048 * traffic_rps
            - 1.15 * cache_hit_rate
            + 1.75 * db_lock_wait_ms
            + 1.30 * gc_pause_ms
            + rng.normal(0, 5.5, n_samples),
            12.0,
            650.0,
        )
        infra_cost_hr = np.clip(
            12.0 + 0.34 * db_pool_size + 0.26 * cpu_util + rng.normal(0, 1.2, n_samples),
            10.0,
            150.0,
        )
        logits = (p99_latency_ms - 145.0) / 18.0
        sla_breach_prob = 1.0 / (1.0 + np.exp(-logits))

        return pd.DataFrame(
            {
                "traffic_rps": traffic_rps,
                "db_pool_size": db_pool_size,
                "cache_hit_rate": cache_hit_rate,
                "cpu_util": cpu_util,
                "db_lock_wait_ms": db_lock_wait_ms,
                "gc_pause_ms": gc_pause_ms,
                "p99_latency_ms": p99_latency_ms,
                "infra_cost_hr": infra_cost_hr,
                "sla_breach_prob": sla_breach_prob,
            }
        )

    def fit(self, df: pd.DataFrame) -> None:
        """Fits structural equations in topological order and computes DML causal effects."""
        topo_nodes = list(nx.topological_sort(self.graph))
        for node in topo_nodes:
            parents = list(self.graph.predecessors(node))
            if not parents:
                continue
            X = df[parents].values
            y = df[node].values
            model = GradientBoostingRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.06, random_state=self.seed
            )
            model.fit(X, y)
            self.models[node] = model
            self.noise_residuals[node] = y - model.predict(X)

        self._estimate_dml_effects(df)

    def _estimate_dml_effects(self, df: pd.DataFrame) -> None:
        """Chernozhukov Double Machine Learning (Orthogonalized Residual-on-Residual ATE)."""
        key_treatments = [
            ("db_pool_size", "p99_latency_ms", ["traffic_rps"]),
            ("db_pool_size", "infra_cost_hr", ["traffic_rps"]),
            ("cache_hit_rate", "p99_latency_ms", ["traffic_rps", "db_pool_size"]),
            ("cpu_util", "p99_latency_ms", ["traffic_rps", "db_pool_size", "cache_hit_rate"]),
            ("db_lock_wait_ms", "p99_latency_ms", ["traffic_rps", "db_pool_size", "cache_hit_rate"]),
        ]
        for T_col, Y_col, W_cols in key_treatments:
            W = df[W_cols].values
            T = df[T_col].values
            Y = df[Y_col].values

            m_T = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=self.seed).fit(W, T)
            m_Y = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=self.seed).fit(W, Y)

            T_res = T - m_T.predict(W)
            Y_res = Y - m_Y.predict(W)

            theta = float(np.sum(T_res * Y_res) / np.sum(T_res ** 2))
            residuals = Y_res - theta * T_res
            se = float(np.sqrt(np.mean(residuals ** 2) / np.sum(T_res ** 2)))
            self.ate_estimates[(T_col, Y_col)] = {
                "ate": round(theta, 4),
                "std_err": round(se, 4),
                "ci_low": round(theta - 1.96 * se, 4),
                "ci_high": round(theta + 1.96 * se, 4),
            }

    def intervene(
        self,
        baseline: Dict[str, float],
        interventions: Dict[str, float],
        n_mc: int = 300,
    ) -> Dict[str, Dict[str, float]]:
        """
        Pearl's do-calculus operator do(X = x):
        Mutilates incoming edges to intervened variables and propagates downstream
        with bootstrapped structural noise for calibrated uncertainty intervals.
        """
        rng = np.random.default_rng(self.seed + 7)
        topo_nodes = list(nx.topological_sort(self.graph))
        samples: Dict[str, np.ndarray] = {}

        for node in topo_nodes:
            if node in interventions:
                samples[node] = np.full(n_mc, float(interventions[node]))
            elif node not in self.models:
                samples[node] = np.full(n_mc, float(baseline[node]))
            else:
                parents = list(self.graph.predecessors(node))
                X_in = np.column_stack([samples[p] for p in parents])
                noise = rng.choice(self.noise_residuals[node], size=n_mc, replace=True)
                pred = self.models[node].predict(X_in) + noise
                if node == "cache_hit_rate":
                    pred = np.clip(pred, 20.0, 99.9)
                elif node == "cpu_util":
                    pred = np.clip(pred, 2.0, 100.0)
                elif node == "sla_breach_prob":
                    pred = 1.0 / (1.0 + np.exp(-(samples["p99_latency_ms"] - 145.0) / 18.0))
                else:
                    pred = np.maximum(pred, 0.5)
                samples[node] = pred

        summary = {}
        for node, vals in samples.items():
            summary[node] = {
                "mean": round(float(np.mean(vals)), 3),
                "p10": round(float(np.percentile(vals, 10)), 3),
                "p90": round(float(np.percentile(vals, 90)), 3),
                "delta": round(float(np.mean(vals) - baseline[node]), 3),
            }
        return summary

    def find_pareto_optimal_policy(
        self, baseline: Dict[str, float], max_budget_hr: float = 65.0
    ) -> List[Dict[str, float]]:
        """Searches counterfactual intervention space for Pareto-optimal SLA vs Cost configs."""
        candidates = []
        for pool in [25, 40, 55, 70, 85]:
            for cache in [75.0, 85.0, 92.0, 96.0]:
                out = self.intervene(
                    baseline,
                    {"db_pool_size": pool, "cache_hit_rate": cache},
                    n_mc=120,
                )
                cost = out["infra_cost_hr"]["mean"]
                lat = out["p99_latency_ms"]["mean"]
                breach = out["sla_breach_prob"]["mean"] * 100.0
                if cost <= max_budget_hr:
                    candidates.append(
                        {
                            "db_pool_size": pool,
                            "cache_hit_rate": cache,
                            "p99_latency_ms": round(lat, 2),
                            "infra_cost_hr": round(cost, 2),
                            "sla_breach_pct": round(breach, 2),
                        }
                    )
        candidates.sort(key=lambda x: (x["p99_latency_ms"], x["infra_cost_hr"]))
        return candidates[:5]
