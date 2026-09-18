from __future__ import annotations

import base64
import copy
import io
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_manifests as manifests
from scripts.e2e.regional.ha011_cpu_nodes import cpu_node
from scripts.e2e.regional.ha011_resources import CpuKubernetes
from tests.regional._cov95_ha011_support import (
    INTENT,
    POD_UID,
    RUN_ID,
    RUNTIME_IMAGE,
    install_kubernetes,
    passing_probe,
    settings_at,
)
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


def test_bundle_contains_only_the_self_contained_probe_and_no_runtime_override(
    tmp_path: Path,
) -> None:
    root = Path(manifests.__file__).parent
    first = manifests.source_bundle(root)
    assert first == manifests.source_bundle(root), (
        "bundle identity must be stable across planning and execution"
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(first))) as bundle:
        assert all(name.startswith("scripts/") for name in bundle.namelist()), (
            "the bundle must not shadow the production gpu_fault package"
        )
        assert len(bundle.namelist()) == 8, (
            "all four probe sources and package initializers are required"
        )
    settings = settings_at(tmp_path)
    generated = manifests.manifests(
        settings,
        runtime_image=RUNTIME_IMAGE,
        bundle=first,
        password="public-fake",
        cpu_node={"name": "cpu-node", "hostname": "cpu-host"},
    )
    pod = generated[-1]
    assert pod["spec"]["automountServiceAccountToken"] is False, (
        "no production service-account token may be mounted"
    )
    assert pod["spec"]["preemptionPolicy"] == "Never", (
        "acceptance must never preempt business Pods"
    )
    assert pod["spec"]["affinity"]["nodeAffinity"], "placement must bind one CPU node"
    assert len(pod["spec"]["containers"]) == 2, (
        "the only containers are runtime and private PostgreSQL"
    )
    database = next(
        item for item in pod["spec"]["containers"] if item["name"] == "postgres"
    )
    environment = {item["name"]: item.get("value") for item in database["env"]}
    assert "PGHOST" not in environment
    assert "unix_socket_directories=/var/run/postgresql,/tmp" in database["args"]
    assert {"name": "postgres-socket", "mountPath": "/var/run/postgresql"} in database[
        "volumeMounts"
    ]
    assert {"name": "postgres-socket", "emptyDir": {"sizeLimit": "1Mi"}} in pod["spec"][
        "volumes"
    ]
    runtime = next(
        item for item in pod["spec"]["containers"] if item["name"] == "runtime"
    )
    assert "PGHOST" not in {item["name"] for item in runtime["env"]}
    assert all(
        item["mountPath"] != "/var/run/postgresql" for item in runtime["volumeMounts"]
    ), (
        'test_bundle_contains_only_the_self_contained_probe_and_no_runtime_override: expected all( item["mountPath"] != "/var/run/postgresql" for ...'
    )
    assert all(
        container["securityContext"]["privileged"] is False
        for container in pod["spec"]["containers"]
    ), "both admitted containers must remain unprivileged"
    assert contracts.digest(generated) != contracts.digest([]), (
        "the resource intent must bind the complete manifest set"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_context", ""),
        ("cluster_id", ""),
        ("region", ""),
        ("namespace", "bad/namespace"),
        ("isolation_id", "not-a-run"),
        ("postgres_image", "postgres:16"),
        ("postgres_image", "postgres:15-bookworm@sha256:" + "c" * 64),
        ("predecessor_case", ""),
    ],
)
def test_settings_refuse_ambiguous_boundaries(
    tmp_path: Path, field: str, value: str
) -> None:
    settings = settings_at(tmp_path)
    with pytest.raises(contracts.ProofError):
        replace(settings, **{field: value})


def test_settings_refuse_missing_config_and_business_namespace_reuse(
    tmp_path: Path,
) -> None:
    settings = settings_at(tmp_path)
    with pytest.raises(contracts.ProofError, match="kubeconfig"):
        replace(settings, cpu_kubeconfig=tmp_path / "missing")
    with pytest.raises(contracts.ProofError, match="business namespace"):
        replace(settings, namespace=settings.isolated_namespace)
    assert settings.environment()["CPU_EKS_CONTEXT"] == "cpu-fixture", (
        "CPU context must be in approval identity"
    )


def test_evidence_requires_all_invariants_and_explicit_mock_scope_is_not_deployed() -> (
    None
):
    proof = passing_probe()
    assert (
        contracts.evidence_errors(
            proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
        )
        == []
    ), "complete bound deployed evidence should satisfy the pure verdict"
    proof["validation_scope"] = "local-worker-lease-simulation"
    assert contracts.evidence_errors(
        proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
    ), "local fake worker receipts cannot be relabelled as deployed proof"


@pytest.mark.parametrize(
    "field",
    [
        "case_id",
        "isolation_id",
        "pod_uid",
        "arm_intent_sha256",
        "postgres_major",
        "business_worker_targeted",
        "cpu_saturation_tested",
    ],
)
def test_evidence_identity_fields_are_mandatory(field: str) -> None:
    proof = passing_probe()
    proof.pop(field)
    assert contracts.evidence_errors(
        proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
    ), f"missing {field} must fail the deployed boundary contract"


@pytest.mark.parametrize(
    "field",
    [
        "same_durable_work",
        "fence_changed",
        "early_claim_count",
        "old_exitcode",
        "late_completion_refused",
        "replacement_live_before_late",
        "replacement_live_after_late",
        "backlog_at_crash",
        "completed_count",
        "final_depth",
        "owned_processes_stopped",
    ],
)
def test_each_role_must_prove_every_takeover_invariant(field: str) -> None:
    proof = passing_probe()
    proof["roles"]["spool"].pop(field)
    assert contracts.evidence_errors(
        proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
    ), f"spool takeover cannot pass without {field}"


@pytest.mark.parametrize(
    "value", [None, -1, 0, True, float("nan"), float("inf"), 61, "0.1"]
)
def test_busy_cpu_evidence_is_bounded_finite_and_numeric(value: object) -> None:
    proof = passing_probe()
    proof["roles"]["processor"]["old_cpu_seconds"] = value
    assert contracts.evidence_errors(
        proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
    ), "unmeasured or malformed CPU time cannot prove a busy worker"


@pytest.mark.parametrize(
    "change",
    [
        "roles",
        "role",
        "owners",
        "owner-type",
        "owner-pid",
        "same-owner",
        "digest",
        "extra",
    ],
)
def test_unknown_or_malformed_evidence_is_not_persistable(change: str) -> None:
    proof = passing_probe()
    role = proof["roles"]["spool"]
    if change == "roles":
        proof["roles"] = []
    elif change == "role":
        proof["roles"]["spool"] = None
    elif change == "owners":
        role["owners"] = None
    elif change == "owner-type":
        role["owners"][0] = 12
    elif change == "owner-pid":
        role["owners"][0] = POD_UID + ":business"
    elif change == "same-owner":
        role["owners"][1] = role["owners"][0]
    elif change == "digest":
        role["work_sha256"] = "missing"
    else:
        role["unapproved-field"] = "must-not-be-persisted"
    assert contracts.evidence_errors(
        proof, isolation_id=RUN_ID, pod_uid=POD_UID, intent_sha256=INTENT
    ), "ambiguous identity or unapproved evidence data must fail closed"


@pytest.mark.parametrize(
    "actual,expected",
    [
        ({}, {"a": 1}),
        ([], {"a": 1}),
        ({"a": []}, {"a": [1]}),
        ({"a": {}}, {"a": []}),
        ({"a": 2}, {"a": 1}),
    ],
)
def test_subset_checker_rejects_missing_or_changed_fields(
    actual: object, expected: object
) -> None:
    with pytest.raises(contracts.ProofError):
        contracts.require_subset(actual, expected)


@pytest.mark.parametrize("change", ["label", "uid", "missing-uid", "deleting"])
def test_resource_identity_is_uid_fenced(change: str) -> None:
    value = {"metadata": {"uid": "expected", "labels": {contracts.LABEL: RUN_ID}}}
    if change == "label":
        value["metadata"]["labels"][contracts.LABEL] = "foreign"
    elif change == "uid":
        value["metadata"]["uid"] = "replacement"
    elif change == "missing-uid":
        value["metadata"].pop("uid")
    else:
        value["metadata"]["deletionTimestamp"] = "in-progress"
    with pytest.raises(contracts.ProofError):
        contracts.require_owned(value, RUN_ID, "expected")


@pytest.mark.parametrize(
    "change",
    [
        "gpu",
        "gpu-label",
        "unknown-family",
        "unknown-provider",
        "not-ready",
        "taint",
        "cordon",
        "uid",
        "linux",
        "capacity",
        "invalid-cpu",
        "small-cpu",
        "invalid-gpu",
        "future-lease",
        "stale-lease",
        "naive-lease",
        "lease-owner",
        "node-owner",
    ],
)
def test_cpu_placement_requires_fresh_non_gpu_inventory(
    monkeypatch, tmp_path: Path, change: str
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    node = fake.object("Node", "cpu-node")
    lease = fake.object("Lease", "cpu-node")
    if change == "gpu":
        node["status"]["capacity"]["nvidia.com/gpu"] = "1"
    elif change == "gpu-label":
        node["metadata"]["labels"]["nvidia.com/gpu.product"] = "accelerator"
    elif change == "unknown-family":
        node["metadata"]["labels"]["node.kubernetes.io/instance-type"] = "p5.48xlarge"
    elif change == "unknown-provider":
        node["spec"]["providerID"] = "unknown://host"
    elif change == "not-ready":
        node["status"]["conditions"][0]["status"] = "False"
    elif change == "taint":
        node["spec"]["taints"] = [{"effect": "NoSchedule"}]
    elif change == "cordon":
        node["spec"]["unschedulable"] = True
    elif change == "uid":
        node["metadata"]["uid"] = ""
    elif change == "linux":
        node["metadata"]["labels"]["kubernetes.io/os"] = "windows"
    elif change == "capacity":
        node["status"].pop("capacity")
    elif change == "invalid-cpu":
        node["status"]["capacity"]["cpu"] = "unknown"
    elif change == "small-cpu":
        node["status"]["allocatable"]["cpu"] = "2000m"
    elif change == "invalid-gpu":
        node["status"]["capacity"]["nvidia.com/gpu"] = "unknown"
    elif change == "future-lease":
        lease["spec"]["renewTime"] = "2099-01-01T00:00:00+00:00"
    elif change == "stale-lease":
        lease["spec"]["renewTime"] = "2000-01-01T00:00:00+00:00"
    elif change == "naive-lease":
        lease["spec"]["renewTime"] = "2026-01-01T00:00:00"
    elif change == "lease-owner":
        lease["spec"]["holderIdentity"] = "other"
    else:
        lease["metadata"]["ownerReferences"] = [{"uid": "old-node"}]
    with pytest.raises(contracts.ProofError):
        cpu_node(CpuKubernetes(settings), "cpu-node")
    assert not fake.created, "CPU placement failures must happen before any mutation"


def test_zero_accelerator_inventory_and_valid_cpu_lease_can_bind(
    monkeypatch, tmp_path: Path
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    fake.object("Node", "cpu-node")["status"]["capacity"]["nvidia.com/gpu"] = "0"
    bound = cpu_node(CpuKubernetes(settings), "cpu-node")
    assert bound["uid"] == "cpu-node-uid", (
        "placement must retain the exact observed node UID"
    )
    changed = copy.deepcopy(bound)
    changed["uid"] = "replacement"
    assert contracts.digest(bound) != contracts.digest(changed), (
        "node replacement must invalidate planned identity"
    )


@pytest.mark.parametrize(
    "value", [None, "", "sha256:", "containerd://sha256:short", "image:tag"]
)
def test_running_image_identity_requires_a_complete_digest(value: object) -> None:
    with pytest.raises(contracts.ProofError, match="digest"):
        contracts.image_digest(value)


def test_cpu_region_must_match_operator_approval(monkeypatch, tmp_path: Path) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    fake.object("Node", "cpu-node")["metadata"]["labels"][
        "topology.kubernetes.io/region"
    ] = "us-east-1"
    with pytest.raises(contracts.ProofError):
        cpu_node(CpuKubernetes(settings), "cpu-node")
    assert not fake.created, "a different Region must be rejected before placement"
