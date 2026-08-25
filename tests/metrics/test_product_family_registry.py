from __future__ import annotations

from gpu_fault.policy import (
    GpuFaultPolicyEngine,
    catalog_product_family,
    load_xid_policy,
)
from gpu_fault.policy.models import CatalogProductFamily


def test_default_product_families_come_from_catalog_yaml() -> None:
    policy = load_xid_policy()
    assert catalog_product_family("NVIDIA H200", policy.product_families) == "H100"
    assert catalog_product_family("GB300", policy.product_families) == "GB200"


def test_new_product_prefix_requires_only_catalog_data() -> None:
    families = [CatalogProductFamily(family="Z100", modelPrefixes=["Z"])]
    assert catalog_product_family("NVIDIA Z500", families) == "Z100"


def test_unknown_product_is_counted_for_metrics(caplog) -> None:
    engine = GpuFaultPolicyEngine()
    with caplog.at_level("ERROR"):
        assert engine.observe_product("ZX900") is None
        assert engine.observe_product("ZX900") is None
    assert engine.unknown_product_counts() == {"ZX900": 2}
    assert caplog.text.count("does not match any catalog") == 1
