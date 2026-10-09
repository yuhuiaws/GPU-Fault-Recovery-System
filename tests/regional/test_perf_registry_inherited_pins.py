"""Synthetic perf clusters register with the Runtime Profile pins of their
provider's production registration (``regional_capacity_pins``): the seeded
Agents are copies of a live Agent of that provider, and
``FleetRegistry.cluster_policy`` holds a cluster's Agents to its own
registration's pins -- the integrated suite on the eks-base mixed site held 96
of 128 workflows on ``runtime profile mismatch: expected eks-v1, got
hyperpod-v1 ... PIN_AHEAD_OF_FLEET`` because perf-cap-* carried none."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.regional_registry import (
    configured_regional_registrations,
    regional_registry_config_sha256,
)
from scripts.perf import regional_capacity_pins as pins
from scripts.perf import regional_capacity_registry as registry
from tests._builders import build_store
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
SEED = lazy_script_module(ROOT / "scripts/perf/seed_regional_action_workflows.py")
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
HYPERPOD_DIGEST = "e" * 64

# The live Secret's two production entries, neutralised: the EKS cluster is the
# site's base family (no pins, explicit provider); the HyperPod entry carries
# its own profile and digest and no ``provider`` key (the registry default).
EKS_ENTRY: dict[str, Any] = {
    "cluster_id": "gpu-eks-a",
    "region": "us-west-2",
    "hyperpod_cluster_name": "",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/gpu-eks-a",
    "provider": "eks-managed",
    "token": "eks-token-" + "e" * 32,
    "allowed_namespaces": ["gpu-fault-system", "training"],
    "agent_endpoint_allowed_cidrs": ["10.0.32.0/20"],
}
HYPERPOD_ENTRY: dict[str, Any] = {
    "cluster_id": "hp-a",
    "region": "us-west-2",
    "hyperpod_cluster_name": "hp-a",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/eks-hp-a",
    "runtime_profile_version": "hyperpod-v1",
    "agent_config_digest": HYPERPOD_DIGEST,
    "token": "hp-token-" + "h" * 32,
    "allowed_namespaces": ["gpu-fault-system", "training"],
    "agent_endpoint_allowed_cidrs": ["10.90.7.0/24"],
}
HYPERPOD_PINS = {
    "runtime_profile_version": "hyperpod-v1",
    "agent_config_digest": HYPERPOD_DIGEST,
}


def _validated(entry: dict[str, Any]) -> RegionalClusterRegistration:
    item = dict(entry)
    token = item.pop("token")
    item["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    return RegionalClusterRegistration(**item)


def test_same_provider_pins_are_inherited() -> None:
    inheritance = pins.inherited_pins([EKS_ENTRY, HYPERPOD_ENTRY], "hyperpod")

    assert inheritance == {
        "provider": "hyperpod",
        "pins": HYPERPOD_PINS,
        "source_cluster_ids": ["hp-a"],
    }
    entries = registry.perf_cluster_entries(
        2,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        provider=inheritance["provider"],
        pins=inheritance["pins"],
    )
    for entry in entries:
        assert entry["provider"] == "hyperpod"
        assert entry["runtime_profile_version"] == "hyperpod-v1"
        assert entry["agent_config_digest"] == HYPERPOD_DIGEST
        assert entry["synthetic"] is True
        registration = _validated(entry)
        assert registration.runtime_profile_version == "hyperpod-v1"
        assert registration.agent_config_digest == HYPERPOD_DIGEST


def test_base_provider_entry_inherits_no_pins() -> None:
    inheritance = pins.inherited_pins([EKS_ENTRY, HYPERPOD_ENTRY], "eks-managed")

    assert inheritance == {
        "provider": "eks-managed",
        "pins": {},
        "source_cluster_ids": ["gpu-eks-a"],
    }
    (entry,) = registry.perf_cluster_entries(
        1,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        provider=inheritance["provider"],
        pins=inheritance["pins"],
    )
    assert entry["provider"] == "eks-managed"
    assert "runtime_profile_version" not in entry
    assert "agent_config_digest" not in entry
    registration = _validated(entry)
    assert registration.runtime_profile_version == ""
    assert registration.agent_config_digest == ""


def test_a_missing_provider_entry_refuses() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"no production registration of provider hyperpod .*eks-managed",
    ):
        pins.inherited_pins([EKS_ENTRY], "hyperpod")
    with pytest.raises(RuntimeError, match="production registrations: none"):
        pins.inherited_pins([], "hyperpod")


def test_disagreeing_same_provider_pins_refuse() -> None:
    other = {**HYPERPOD_ENTRY, "cluster_id": "hp-b", "agent_config_digest": "f" * 64}
    with pytest.raises(RuntimeError, match="different Runtime Profile pins") as info:
        pins.inherited_pins([EKS_ENTRY, HYPERPOD_ENTRY, other], "hyperpod")
    assert "hp-a" in str(info.value) and "hp-b" in str(info.value)


def test_default_provider_entries_and_the_default_target_agree() -> None:
    """An entry without ``provider`` and an unset selector are both hyperpod."""

    assert pins.registration_provider({"cluster_id": "x"}) == "hyperpod"
    assert pins.registration_provider(EKS_ENTRY) == "eks-managed"
    assert pins.synthetic_registration_provider() == "hyperpod"
    assert pins.SYNTHETIC_PROVIDER_ENV == SEED.SYNTHETIC_PROVIDER_ENV


def test_an_unknown_synthetic_provider_refuses(monkeypatch) -> None:
    monkeypatch.setenv(pins.SYNTHETIC_PROVIDER_ENV, "openstack")
    with pytest.raises(RuntimeError, match="not a cluster provider"):
        pins.synthetic_registration_provider()


def test_unknown_pin_fields_are_refused() -> None:
    with pytest.raises(RuntimeError, match="limited to"):
        registry.synthetic_cluster_entry(
            "perf-cap-000",
            run_id="run-a",
            expires_at=NOW,
            pins={"required_agent_version": "1.0"},
        )


def test_entry_shape_without_provider_or_pins_is_unchanged() -> None:
    """The AUTH-007/008 runner's second logical cluster keeps its shape."""

    entry = registry.synthetic_cluster_entry(
        "auth-logical-b", run_id="run-a", expires_at=NOW
    )
    assert "provider" not in entry
    assert "runtime_profile_version" not in entry
    assert "agent_config_digest" not in entry


def _register_with_fakes(
    tmp_path: Path, monkeypatch, baseline: list[dict[str, Any]]
) -> tuple[list[list[dict[str, Any]]], list[tuple[str, dict[str, bytes]]]]:
    from scripts.perf import regional_capacity_data

    written: list[list[dict[str, Any]]] = []
    token_secrets: list[tuple[str, dict[str, bytes]]] = []
    monkeypatch.setattr(registry, "validate_registry_target", lambda **_k: "isolated")
    monkeypatch.setattr(registry, "validate_notification_safety", lambda: None)
    monkeypatch.setattr(
        registry, "validate_alertmanager_drill_route", lambda: {"stub": True}
    )
    monkeypatch.setattr(registry, "cleanup_registry_residuals", lambda **_k: 0)
    monkeypatch.setattr(registry, "load_registry", lambda: [dict(e) for e in baseline])
    monkeypatch.setattr(registry, "verify_registry_alignment", lambda **_k: {})
    monkeypatch.setattr(registry, "capture_secret_baseline", lambda *_a: None)
    monkeypatch.setattr(regional_capacity_data, "invoke", lambda *a, **k: {"total": 0})
    monkeypatch.setattr(
        registry, "write_registry", lambda values, **_k: written.append(list(values))
    )
    monkeypatch.setattr(
        registry, "sync_dataplane_connection_secret", lambda: {"changed": False}
    )

    def upsert_secret(name, files, *, run_id):
        token_secrets.append((name, files))
        return {"run_id": run_id, "uid": "token-uid", "resource_version": "1"}

    monkeypatch.setattr(registry, "upsert_secret", upsert_secret)
    return written, token_secrets


def test_register_publishes_inherited_pins_and_records_them(
    tmp_path: Path, monkeypatch
) -> None:
    written, _tokens = _register_with_fakes(
        tmp_path, monkeypatch, [EKS_ENTRY, HYPERPOD_ENTRY]
    )

    registry.register(
        2,
        tmp_path,
        run_id="run-a",
        # Wall clock: the registry's load-time hygiene drops expired synthetic
        # entries, and the alignment inputs are checked on the published list.
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        allow_live_registry=False,
        live_registry_confirmation=None,
    )

    (published,) = written
    assert published[:2] == [EKS_ENTRY, HYPERPOD_ENTRY], "production entries untouched"
    for entry in published[2:]:
        assert entry["provider"] == "hyperpod"
        assert entry["runtime_profile_version"] == "hyperpod-v1"
        assert entry["agent_config_digest"] == HYPERPOD_DIGEST
    intent = json.loads((tmp_path / "registry-registration-intent.json").read_text())
    assert intent["provider"] == "hyperpod"
    assert intent["inherited_pins"] == HYPERPOD_PINS
    assert intent["pins_source_cluster_ids"] == ["hp-a"]
    assert intent["cluster_ids"] == ["perf-cap-000", "perf-cap-001"]
    # The alignment gate's own inputs accept the published shape.
    registrations = configured_regional_registrations(published)
    assert [item.cluster_id for item in registrations[2:]] == intent["cluster_ids"]
    assert all(
        item.runtime_profile_version == "hyperpod-v1" for item in registrations[2:]
    ), "the published synthetic registrations carry the inherited profile"
    assert len(regional_registry_config_sha256(registrations)) == 64


def test_register_as_the_base_family_records_no_pins(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(pins.SYNTHETIC_PROVIDER_ENV, "eks-managed")
    written, _tokens = _register_with_fakes(
        tmp_path, monkeypatch, [EKS_ENTRY, HYPERPOD_ENTRY]
    )

    registry.register(
        1,
        tmp_path,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        allow_live_registry=False,
        live_registry_confirmation=None,
    )

    (published,) = written
    assert published[-1]["provider"] == "eks-managed"
    assert "runtime_profile_version" not in published[-1]
    intent = json.loads((tmp_path / "registry-registration-intent.json").read_text())
    assert intent["provider"] == "eks-managed"
    assert intent["inherited_pins"] == {}
    assert intent["pins_source_cluster_ids"] == ["gpu-eks-a"]


def test_register_refuses_before_any_write_without_a_provider_entry(
    tmp_path: Path, monkeypatch
) -> None:
    written, tokens = _register_with_fakes(tmp_path, monkeypatch, [EKS_ENTRY])

    with pytest.raises(RuntimeError, match="no production registration of provider"):
        registry.register(
            1,
            tmp_path,
            run_id="run-a",
            expires_at=NOW + timedelta(hours=1),
            allow_live_registry=False,
            live_registry_confirmation=None,
        )

    assert written == [] and tokens == [], "refusal precedes every write"
    assert not (tmp_path / "registry-registration-intent.json").exists(), (
        "a refused registration grants no cleanup ownership"
    )


def test_cleanup_removes_pinned_synthetic_entries_and_keeps_production(
    monkeypatch,
) -> None:
    owned = registry.perf_cluster_entries(
        1,
        run_id="run-a",
        expires_at=NOW + timedelta(hours=1),
        provider="hyperpod",
        pins=HYPERPOD_PINS,
    )[0]
    entries = [dict(EKS_ENTRY), dict(HYPERPOD_ENTRY), owned]
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(registry, "load_registry", lambda: list(entries))

    def write(values, **kwargs):
        writes.append(kwargs)
        entries[:] = values

    monkeypatch.setattr(registry, "write_registry", write)

    removed = registry.cleanup_registry_residuals(
        scope="isolated", artifacts=None, phase="postflight", force=True, run_id="run-a"
    )

    assert removed == 1
    assert entries == [EKS_ENTRY, HYPERPOD_ENTRY]
    assert writes[0]["expected_entries"][-1] == owned


def _synthetic_registration(index: int, **pinned: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=f"perf-cap-{index:03d}",
        region="us-west-2",
        hyperpod_cluster_name=f"perf-cap-{index:03d}",
        eks_cluster_arn=f"arn:aws:eks:us-west-2:000000000000:cluster/perf-cap-{index:03d}",
        token_sha256="a" * 64,
        synthetic=True,
        synthetic_run_id="run-a",
        synthetic_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["127.0.0.1/32"],
        **pinned,
    )


def test_seed_records_the_registered_pins_and_refuses_a_differing_template() -> None:
    context = ApplicationContext(store=build_store())
    for index in range(2):
        context.store.save_regional_cluster(
            _synthetic_registration(index, **HYPERPOD_PINS)
        )

    assert SEED.registered_pins_facts(
        context,
        clusters=2,
        runtime_profile_version="hyperpod-v1",
        config_digest=HYPERPOD_DIGEST,
    ) == {
        "registered_runtime_profile_version": "hyperpod-v1",
        "registered_agent_config_digest": HYPERPOD_DIGEST,
    }
    with pytest.raises(RuntimeError, match="hold every destructive step") as info:
        SEED.registered_pins_facts(
            context,
            clusters=2,
            runtime_profile_version="eks-v1",
            config_digest="f" * 64,
        )
    assert "hyperpod-v1 (registered) vs eks-v1 (Agent template)" in str(info.value)
    with pytest.raises(RuntimeError, match="perf-cap-002 is not registered"):
        SEED.registered_pins_facts(
            context,
            clusters=3,
            runtime_profile_version="hyperpod-v1",
            config_digest=HYPERPOD_DIGEST,
        )


def test_seed_accepts_any_template_for_unpinned_registrations() -> None:
    """The base family inherits the site pins; the registration checks nothing."""

    context = ApplicationContext(store=build_store())
    context.store.save_regional_cluster(_synthetic_registration(0))

    assert SEED.registered_pins_facts(
        context, clusters=1, runtime_profile_version="eks-v1", config_digest="f" * 64
    ) == {
        "registered_runtime_profile_version": "",
        "registered_agent_config_digest": "",
    }
