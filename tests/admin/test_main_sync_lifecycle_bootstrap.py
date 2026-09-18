"""Committed-main target completion without weakening local identity barriers."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
)
from gpu_fault.admin.bootstrap_site import (
    INITIAL_DEPLOY_TARGET,
    bind_initial_deploy_target,
)
from tests.admin._bootstrap_support import _cluster
from tests.admin._cov95_join_support import target
from tests.admin._cov95_removal_support import RemovalTransport


def site_document(
    cpu: ClusterIdentity, *gpu_clusters: ClusterIdentity
) -> dict[str, Any]:
    return {
        "spec": {
            "cpu": {"eksArn": cpu.eks_arn, "hyperpodClusterName": cpu.hyperpod_name},
            "clusters": [
                {
                    "eksClusterArn": item.eks_arn,
                    "hyperpodClusterName": item.hyperpod_name,
                }
                for item in gpu_clusters
            ],
        }
    }


def pending_state(tmp_path: Path, *gpu_clusters: ClusterIdentity) -> BootstrapState:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="test-site")
    bind_initial_deploy_target(state, None, _cluster(), gpu_clusters)
    return state


def test_fulfilled_initial_target_accepts_growth_and_then_settles(
    tmp_path: Path,
) -> None:
    cpu, gpu_a, gpu_b = _cluster(), target("a"), target("b")
    state = pending_state(tmp_path, gpu_a)
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="persisted checkpoint"):
        bind_initial_deploy_target(state, None, cpu, [gpu_b])
    assert state.path.read_bytes() == before, "a rejected resume rewrote its target"

    assert bind_initial_deploy_target(
        state, site_document(cpu, gpu_a), cpu, [gpu_a, gpu_b]
    ) == [gpu_a], "bootstrap must leave the new member to the supervised join"
    stored = state.result(INITIAL_DEPLOY_TARGET)
    assert stored["status"] == "PENDING", "growth must retain the pending GPU target"
    assert [item["eks_arn"] for item in stored["gpu_clusters"]] == [
        gpu_a.eks_arn,
        gpu_b.eks_arn,
    ], "the new checkpoint lost the requested order or membership"

    state = BootstrapState(state.path, site_id="test-site")
    assert bind_initial_deploy_target(
        state, site_document(cpu, gpu_a, gpu_b), cpu, [gpu_a, gpu_b]
    ) == [gpu_a, gpu_b], "the completed site must reconcile both managed clusters"
    assert state.result(INITIAL_DEPLOY_TARGET)["status"] == "COMPLETE", (
        "a fully managed target must settle"
    )


@pytest.mark.parametrize("requested", [("a",), ("a", "c"), ("a", "b", "c"), ("b", "a")])
def test_partial_multi_target_site_remains_bound_to_every_original_target(
    tmp_path: Path, requested: tuple[str, ...]
) -> None:
    cpu, gpu_a, gpu_b = _cluster(), target("a"), target("b")
    state = pending_state(tmp_path, gpu_a, gpu_b)
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="persisted checkpoint"):
        bind_initial_deploy_target(
            state,
            site_document(cpu, gpu_a),
            cpu,
            [target(suffix) for suffix in requested],
        )
    assert state.path.read_bytes() == before, (
        "a partial site released or reordered an unfinished multi-target commitment"
    )


def test_fulfilled_target_allows_the_same_provider_arn_aliases(tmp_path: Path) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    assert bind_initial_deploy_target(
        state,
        site_document(cpu, gpu),
        replace(cpu, input_arn=cpu.hyperpod_arn),
        [replace(gpu, input_arn=gpu.hyperpod_arn)],
    ) == [replace(gpu, input_arn=gpu.hyperpod_arn)], (
        "EKS and HyperPod input aliases must not change the resolved identity"
    )
    assert state.result(INITIAL_DEPLOY_TARGET)["status"] == "COMPLETE", (
        "aliases of the same resolved target must settle"
    )


@pytest.mark.parametrize("status", ["PENDING", "COMPLETE"])
@pytest.mark.parametrize("field", ["eks_arn", "hyperpod_arn", "hyperpod_name"])
def test_a_fulfilled_site_cannot_rebind_the_checkpoint_cpu(
    tmp_path: Path, field: str, status: str
) -> None:
    gpu = target("a")
    state = pending_state(tmp_path, gpu)
    checkpoint = state.result(INITIAL_DEPLOY_TARGET)
    checkpoint["status"] = status
    state.record(INITIAL_DEPLOY_TARGET, checkpoint)
    cpu = replace(_cluster(), **{field: getattr(_cluster(), field) + "-other"})
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint"):
        bind_initial_deploy_target(state, site_document(cpu, gpu), cpu, [gpu])
    assert state.path.read_bytes() == before, "CPU drift rewrote the bootstrap target"


def test_fulfilled_pending_target_rejects_a_rebound_gpu_hyperpod(
    tmp_path: Path,
) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint"):
        bind_initial_deploy_target(
            state,
            site_document(cpu, gpu),
            cpu,
            [replace(gpu, hyperpod_arn=gpu.hyperpod_arn + "-replacement")],
        )
    assert state.path.read_bytes() == before, (
        "same EKS/name with another HyperPod incarnation released the target"
    )


def test_a_completed_target_with_a_missing_site_cannot_start_over(
    tmp_path: Path,
) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    bind_initial_deploy_target(state, site_document(cpu, gpu), cpu, [gpu])
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint"):
        bind_initial_deploy_target(state, None, cpu, [gpu])
    assert state.path.read_bytes() == before, (
        "missing site discarded a completed target"
    )


@pytest.mark.parametrize("resources", [None, [], "invalid"])
def test_malformed_resource_container_cannot_discard_the_checkpoint(
    tmp_path: Path, resources: object
) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    state.value["resources"] = resources
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint"):
        bind_initial_deploy_target(state, site_document(cpu, gpu), cpu, [gpu])
    assert state.path.read_bytes() == before, (
        "malformed bootstrap resources silently reset the initial target"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "null",
        "non-object",
        "missing-cpu",
        "invalid-cpu",
        "missing-gpus",
        "invalid-gpus",
        "invalid-gpu",
        "missing-gpu-arn",
        "invalid-input-alias",
        "duplicate-gpu",
        "unknown-schema",
        "boolean-schema",
        "unknown-status",
    ],
)
@pytest.mark.parametrize("status", ["PENDING", "COMPLETE"])
def test_malformed_checkpoint_never_uses_site_completion_as_a_bypass(
    tmp_path: Path, corruption: str, status: str
) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    checkpoint = deepcopy(state.result(INITIAL_DEPLOY_TARGET))
    checkpoint["status"] = status
    if corruption == "null":
        checkpoint = None
    elif corruption == "non-object":
        checkpoint = []
    elif corruption == "missing-cpu":
        checkpoint.pop("cpu")
    elif corruption == "invalid-cpu":
        checkpoint["cpu"] = {}
    elif corruption == "missing-gpus":
        checkpoint.pop("gpu_clusters")
    elif corruption == "invalid-gpus":
        checkpoint["gpu_clusters"] = {}
    elif corruption == "invalid-gpu":
        checkpoint["gpu_clusters"] = [None]
    elif corruption == "missing-gpu-arn":
        checkpoint["gpu_clusters"][0].pop("hyperpod_arn")
    elif corruption == "invalid-input-alias":
        checkpoint["gpu_clusters"][0]["input_arn"] = target("b").input_arn
    elif corruption == "duplicate-gpu":
        checkpoint["gpu_clusters"] *= 2
    elif corruption == "unknown-schema":
        checkpoint["schema_version"] = 2
    elif corruption == "boolean-schema":
        checkpoint["schema_version"] = True
    else:
        checkpoint["status"] = "UNKNOWN"
    state.record(INITIAL_DEPLOY_TARGET, checkpoint)
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint"):
        bind_initial_deploy_target(state, site_document(cpu, gpu), cpu, [gpu])
    assert state.path.read_bytes() == before, "malformed checkpoint was overwritten"


@pytest.mark.parametrize(
    "corruption", ["foreign-cpu", "non-object-gpu", "duplicate-gpu", "omitted-member"]
)
def test_settling_a_target_preserves_existing_site_validation(
    tmp_path: Path, corruption: str
) -> None:
    cpu, gpu = _cluster(), target("a")
    state = pending_state(tmp_path, gpu)
    document = site_document(cpu, gpu)
    if corruption == "foreign-cpu":
        document["spec"]["cpu"]["eksArn"] = target("b").eks_arn
    elif corruption == "non-object-gpu":
        document["spec"]["clusters"].append(None)
    elif corruption == "duplicate-gpu":
        document["spec"]["clusters"] *= 2
    else:
        document = site_document(cpu, gpu, target("b"))
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="cluster identity"):
        bind_initial_deploy_target(state, document, cpu, [gpu])
    assert state.path.read_bytes() == before, "invalid site released the checkpoint"


@pytest.mark.parametrize("key", ["hp-gpu-a", "gpu-a"])
def test_a_name_only_removed_cluster_row_does_not_release_the_target(
    tmp_path: Path, key: str
) -> None:
    state = pending_state(tmp_path, target("a"))
    state.value["removed_clusters"] = {
        key: {"removed_at": "2026-09-15T00:00:00+00:00", "vpc_id": "vpc-gpu-a"}
    }
    state.record(INITIAL_DEPLOY_TARGET, state.result(INITIAL_DEPLOY_TARGET))
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="checkpoint|removal"):
        bind_initial_deploy_target(state, site_document(_cluster()), _cluster(), [])
    assert state.path.read_bytes() == before, (
        "a name-only removal record cannot prove which provider cluster was detached"
    )


@pytest.fixture
def removed_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RemovalTransport, BootstrapState]:
    transport = RemovalTransport(tmp_path, monkeypatch)
    state = pending_state(tmp_path, target("a"))
    assert transport.remove()["phase"] == "COMPLETED", (
        "the fixture must produce a complete identity-bound removal journal"
    )
    return transport, BootstrapState(state.path, site_id="test-site")


def test_completed_identity_bound_removal_allows_cpu_only_deploy(
    removed_target: tuple[RemovalTransport, BootstrapState],
) -> None:
    transport, state = removed_target
    document = yaml.safe_load(transport.path.read_text(encoding="utf-8"))
    events = list(transport.events)

    assert bind_initial_deploy_target(state, document, _cluster(), []) == [], (
        "a completed removal must release its exact initial GPU commitment"
    )
    checkpoint = state.result(INITIAL_DEPLOY_TARGET)
    assert checkpoint["status"] == "COMPLETE" and checkpoint["gpu_clusters"] == [], (
        "the CPU-only managed site must become the completed target"
    )
    assert transport.events == events, "settling a local checkpoint ran a command"


def test_removing_one_original_target_does_not_release_another_pending_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = RemovalTransport(tmp_path, monkeypatch)
    state = pending_state(tmp_path, target("a"), target("b"))
    assert transport.remove()["phase"] == "COMPLETED", (
        "the fixture must complete only the managed target's removal"
    )
    state = BootstrapState(state.path, site_id="test-site")
    document = yaml.safe_load(transport.path.read_text(encoding="utf-8"))
    before = state.path.read_bytes()

    with pytest.raises(BootstrapError, match="persisted checkpoint"):
        bind_initial_deploy_target(state, document, _cluster(), [])
    assert state.path.read_bytes() == before, (
        "one completed removal released a different unfulfilled GPU target"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "hyperpod-arn",
        "eks-arn",
        "cpu-arn",
        "target-name",
        "site-binding",
        "partial",
        "supervision-lost",
        "incomplete-barriers",
        "legacy",
    ],
)
def test_removal_completion_requires_the_full_original_identity_and_barriers(
    removed_target: tuple[RemovalTransport, BootstrapState], corruption: str
) -> None:
    transport, state = removed_target
    path = transport.directory / "remove-cluster/gpu-a/state.json"
    journal = json.loads(path.read_text(encoding="utf-8"))
    provider = journal["evidence"]["DISCOVERED"]["provider_identity"]
    if corruption in {"hyperpod-arn", "eks-arn", "cpu-arn"}:
        field = {
            "hyperpod-arn": "hyperpod_arn",
            "eks-arn": "eks_arn",
            "cpu-arn": "cpu_eks_arn",
        }[corruption]
        provider[field] += "-replacement"
    elif corruption == "target-name":
        journal["target"]["hyperpod_cluster_name"] = "other"
        journal["evidence"]["DISCOVERED"]["target"] = dict(journal["target"])
    elif corruption == "site-binding":
        journal["remaining_site_sha256"] = "0" * 64
    elif corruption == "partial":
        journal["phase"] = "SITE_UPDATED"
        journal["completed_steps"] = [
            step
            for step in journal["completed_steps"]
            if step not in {"RELEASE_STATE_UPDATED", "VERIFIED"}
        ]
    elif corruption == "supervision-lost":
        journal["phase"] = "SUPERVISION_LOST"
    elif corruption == "incomplete-barriers":
        journal["completed_steps"].remove("KUBERNETES_QUIESCED")
    else:
        journal["schema_version"] = 1
    path.write_text(json.dumps(journal), encoding="utf-8")
    before = state.path.read_bytes()
    document = yaml.safe_load(transport.path.read_text(encoding="utf-8"))

    with pytest.raises(BootstrapError, match="checkpoint|removal|remove-cluster"):
        bind_initial_deploy_target(state, document, _cluster(), [])
    assert state.path.read_bytes() == before, (
        "foreign or incomplete removal evidence released the initial target"
    )
