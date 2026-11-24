import pytest
from src.scm_engine import StructuralCausalTwin


def test_dag_is_acyclic():
    twin = StructuralCausalTwin(seed=42)
    assert len(twin.graph.nodes) == 9
    assert len(twin.graph.edges) == 13


def test_dml_and_do_calculus_intervention():
    twin = StructuralCausalTwin(seed=42)
    df = twin.generate_observational_data(n_samples=1000)
    twin.fit(df)

    assert ("db_pool_size", "p99_latency_ms") in twin.ate_estimates
    assert twin.ate_estimates[("cache_hit_rate", "p99_latency_ms")]["ate"] < 0

    baseline = {col: float(df[col].median()) for col in df.columns}
    cf_high_cache = twin.intervene(baseline, {"cache_hit_rate": 96.0, "db_pool_size": 80.0}, n_mc=150)
    cf_low_cache = twin.intervene(baseline, {"cache_hit_rate": 50.0, "db_pool_size": 20.0}, n_mc=150)

    assert cf_high_cache["p99_latency_ms"]["mean"] < cf_low_cache["p99_latency_ms"]["mean"]
