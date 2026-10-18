"""HA-011 owned-resource boundary: the refusals around admission and cleanup.

Each case drives the public ``IsolatedResources`` surface against the fake CPU
Kubernetes boundary and checks that a drifted admission, a missing receipt or
an unknown capture stage is refused with the documented proof error, and that
cleanup tolerates an owned resource that is already gone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha011_manifests as manifests
from scripts.e2e.regional import ha011_resources as resources
from tests.regional._cov95_ha011_support import (
    IMAGE_ID,
    INTENT,
    deadline,
    install_kubernetes,
    settings_at,
)
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


def prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, Any, Any, resources.IsolatedResources, list[dict[str, Any]]]:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    api = resources.CpuKubernetes(settings)
    identity = api.identity()
    lifetime = resources.IsolatedResources(
        api,
        intent_sha256=INTENT,
        deadline=deadline(),
        cpu_node_identity=identity["cpu_node"],
    )
    objects = manifests.manifests(
        settings,
        runtime_image=identity["runtime_image"],
        bundle="public-fake-bundle",
        password="public-fake-private-database-password",
        cpu_node=identity["cpu_node"],
    )
    return settings, fake, api, lifetime, objects


def test_verify_pod_refuses_a_foreign_service_account() -> None:
    observed = {
        "kind": "Pod",
        "metadata": {},
        "spec": {"serviceAccount": "elevated", "containers": []},
    }
    with pytest.raises(resources.ProofError, match="different service account"):
        resources.verify_pod(
            observed, {"metadata": {}}, node_name="cpu", scheduled=False
        )


def test_priority_class_cannot_be_created_before_its_namespace_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, _objects = prepare(monkeypatch, tmp_path)
    with pytest.raises(resources.ProofError, match="namespace owner is unproven"):
        lifetime.create(manifests.priority_class_manifest(settings))
    assert fake.created == [], "a refused PriorityClass must never reach the API"


def test_creation_without_an_acknowledged_uid_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, _fake, api, lifetime, _objects = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "call", lambda *a, **k: "   ")
    with pytest.raises(resources.ProofError, match="returned no UID"):
        lifetime.create(lifetime.namespace)
    assert lifetime.receipts == {}, "no UID means no ownership receipt"


def test_network_policy_spec_drift_during_admission_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)

    def drift(args: list[str]) -> None:
        if args[:2] == ["get", "NetworkPolicy"]:
            fake.object("NetworkPolicy", "loopback-only")["spec"]["unexpected"] = True

    fake.before = drift
    with pytest.raises(resources.ProofError, match="network isolation policy changed"):
        lifetime.start(objects)
    assert "NetworkPolicy/loopback-only" in lifetime.receipts, (
        "the acknowledged identity survives the failed validation"
    )
    assert lifetime.pod is None, "admission must stop before the probe Pod"


def test_start_refuses_an_unapproved_priority_class_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    for resource in objects:
        if resource["kind"] == "PriorityClass":
            resource["value"] = resource["value"] + 1
    with pytest.raises(resources.ProofError, match="unapproved PriorityClass intent"):
        lifetime.start(objects)
    assert "PriorityClass" not in {item["kind"] for item in fake.created}, (
        "a refused PriorityClass intent must never reach the API"
    )


def test_arm_and_scheduling_gate_require_an_owned_probe_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, _objects = prepare(monkeypatch, tmp_path)
    with pytest.raises(resources.ProofError, match="no owned probe Pod"):
        lifetime.release_scheduling_gate()
    monkeypatch.setattr(lifetime, "release_scheduling_gate", lambda: None)
    with pytest.raises(resources.ProofError, match="no owned probe Pod"):
        lifetime.arm(IMAGE_ID)
    assert fake.armed is False, "arming without a Pod must not reach the exec"


def test_capture_failure_rejects_an_unknown_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, _fake, _api, lifetime, _objects = prepare(monkeypatch, tmp_path)
    with pytest.raises(resources.ProofError, match="unknown failure capture stage"):
        lifetime.capture_failure({}, stage="after-arm")
    assert lifetime.failure_details is None


def test_collect_fails_when_the_isolated_database_container_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    lifetime.arm(IMAGE_ID)
    pod = fake.object("Pod", resources.POD_NAME)
    for status in pod["status"]["containerStatuses"]:
        if status["name"] == "postgres":
            status["state"] = {"waiting": {"reason": "CrashLoopBackOff"}}
    with pytest.raises(resources.ProofError, match="isolated database did not survive"):
        lifetime.collect(IMAGE_ID)


def test_cleanup_skips_owned_resources_that_are_already_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    del fake.objects["ConfigMap", settings.isolated_namespace, "probe-source"]
    cleanup = lifetime.cleanup()
    assert cleanup["namespace_absent"] is True
    assert "ConfigMap/probe-source" in cleanup["resource_uids"], (
        "the receipt of a vanished resource is still reported"
    )
    assert fake.deleted is True


def test_cleanup_requires_a_priority_class_resource_version_to_delete_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    priority = fake.object("PriorityClass", settings.priority_class_name)
    del priority["metadata"]["resourceVersion"]
    with pytest.raises(resources.ProofError, match="requires a resource version"):
        lifetime.cleanup()
    assert ("PriorityClass", None, settings.priority_class_name) in fake.objects, (
        "an unversioned PriorityClass must not be deleted unconditionally"
    )
