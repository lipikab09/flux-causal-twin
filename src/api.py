from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict
from src.scm_engine import StructuralCausalTwin

app = FastAPI(
    title="FLUX-CAUSAL-TWIN API",
    description="Structural Causal Model & Counterfactual Digital Twin Inference Service",
    version="1.0.0",
)

twin = StructuralCausalTwin(seed=42)
telemetry_df = twin.generate_observational_data(n_samples=2500)
twin.fit(telemetry_df)


class InterventionRequest(BaseModel):
    baseline: Dict[str, float] = Field(
        default={
            "traffic_rps": 1250.0,
            "db_pool_size": 35.0,
            "cache_hit_rate": 74.0,
            "cpu_util": 68.0,
            "db_lock_wait_ms": 22.0,
            "gc_pause_ms": 28.0,
            "p99_latency_ms": 158.0,
            "infra_cost_hr": 42.0,
            "sla_breach_prob": 0.67,
        }
    )
    interventions: Dict[str, float] = Field(default={"db_pool_size": 70.0, "cache_hit_rate": 92.0})


@app.get("/health")
def health():
    return {
        "status": "online",
        "nodes": len(twin.graph.nodes),
        "edges": len(twin.graph.edges),
        "dml_pairs": len(twin.ate_estimates),
    }


@app.get("/causal-effects")
def get_causal_effects():
    return {f"{k[0]} -> {k[1]}": v for k, v in twin.ate_estimates.items()}


@app.post("/intervene")
def run_intervention(req: InterventionRequest):
    result = twin.intervene(req.baseline, req.interventions, n_mc=300)
    return {"do_calculus_counterfactual": result}
