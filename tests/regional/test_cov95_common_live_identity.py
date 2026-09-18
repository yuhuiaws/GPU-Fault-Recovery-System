from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import regional_live_fixture as live
from tests.regional._cov95_common_live import LiveModel, node


def test_runtime_identity_captures_both_planes_and_checks_the_observed_template(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    identity = fixture.runtime_identity()
    assert identity["release_state"]["release_id"] == "unit-release"
    assert live.runtime_identity_errors(identity) == []
    for plane, names in live.RUNTIME_IDENTITY_DEPLOYMENTS.items():
        assert set(identity["deployments"][plane]) == set(names)
        for value, document in zip(
            identity["deployments"][plane].values(),
            fixture.documents[(plane, "deployment", "")]["items"],
        ):
            template = document["spec"]["template"]
            assert (
                value["template_sha256"]
                == hashlib.sha256(
                    json.dumps(template, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            )
            assert value["images"] == ["example/app:1", "example/init:1"]
    evidence = tmp_path / "runtime.json"
    assert (
        fixture.verify_runtime_identity(
            identity, evidence_path=evidence, stage="before"
        )
        == identity
    )
    assert json.loads(evidence.read_text()) == identity


@pytest.mark.parametrize(
    "document,problem",
    [
        ({}, "not valid JSON"),
        ({"data": {"state.json": "{"}}, "not valid JSON"),
        ({"data": {"state.json": "[]"}}, "not an object"),
    ],
)
def test_runtime_identity_rejects_unreadable_release_state(
    tmp_path: Path, document: Any, problem: str
) -> None:
    fixture = LiveModel(tmp_path)
    fixture.documents[("cpu", "configmap", "gpu-fault-regional-release-state")] = (
        document
    )
    with pytest.raises(live.RegionalFixtureError, match=problem):
        fixture.runtime_identity()
    assert len(fixture.calls) == 1


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_runtime_identity_refuses_incomplete_deployment_inventory(
    tmp_path: Path, plane: str
) -> None:
    fixture = LiveModel(tmp_path)
    fixture.documents[(plane, "deployment", "")]["items"].pop()
    with pytest.raises(
        live.RegionalFixtureError, match=f"{plane} runtime deployments are missing"
    ):
        fixture.runtime_identity()


@pytest.mark.parametrize("fault", ["template", "rollback", "not-ready"])
def test_identity_verification_rejects_drift_or_consistently_unsafe_baselines(
    tmp_path: Path, fault: str
) -> None:
    fixture = LiveModel(tmp_path)
    expected = fixture.runtime_identity()
    if fault == "template":
        fixture.documents[("cpu", "deployment", "")]["items"][0]["spec"]["template"][
            "spec"
        ]["containers"][0]["env"] = [{"name": "EXAMPLE", "value": "changed"}]
        problem = "drifted"
    elif fault == "rollback":
        fixture.documents[("cpu", "configmap", "gpu-fault-regional-release-state")][
            "data"
        ]["state.json"] = json.dumps(
            {"release_id": "unit-release", "phase": "rollback-restoring"}
        )
        expected = fixture.runtime_identity()
        problem = "unsafe.*rollback"
    else:
        fixture.documents[("gpu", "deployment", "")]["items"][0]["status"][
            "readyReplicas"
        ] = 0
        expected = fixture.runtime_identity()
        problem = "unsafe.*not fully rolled out"
    path = tmp_path / "observed.json"
    with pytest.raises(live.RegionalFixtureError, match=problem):
        fixture.verify_runtime_identity(expected, evidence_path=path, stage="resume")
    observed = json.loads(path.read_text())
    assert observed == fixture.runtime_identity()


@pytest.mark.parametrize("fault", ["plane", "mixed-namespace"])
def test_kubectl_scope_errors_stop_before_the_transport(
    tmp_path: Path, fault: str
) -> None:
    fixture = LiveModel(tmp_path)
    with pytest.raises(ValueError, match="unknown Kubernetes plane|mutually exclusive"):
        fixture.kubectl(
            "wrong" if fault == "plane" else "gpu",
            "get",
            "pod",
            all_namespaces=True,
            namespace="other" if fault == "mixed-namespace" else None,
        )
    assert fixture.calls == []


def test_node_snapshots_preserve_ownership_but_not_installer_annotations(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    captured = fixture.node_snapshot("node-a")
    assert captured["uid"] == "node-a-uid"
    assert captured["boot_id"] == "boot-new"
    assert captured["ready"] == "True"
    assert captured["ownership_annotations"] == {"gpu-fault.io/owner": "unit"}
    assert [item["name"] for item in fixture.gpu_nodes()] == ["node-a"]
    assert fixture.node_metadata("node-a")["product"] == "example-product"
    item = fixture.documents[("gpu", "node", "node-a")]
    item["metadata"]["labels"] = {"node.kubernetes.io/instance-type": "fallback"}
    assert fixture.node_metadata("node-a")["product"] == "fallback"
    item["metadata"]["labels"] = {}
    assert fixture.node_metadata("node-a")["product"] is None


def test_workload_reads_keep_user_workloads_and_filter_only_known_system_work(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    items = []
    for name, namespace, phase, gpus in [
        ("training", "user", "Running", 2),
        ("pending", "user", "Pending", 1),
        ("done", "user", "Succeeded", 4),
        ("system", "kube-system", "Running", 0),
        ("runtime", "test-namespace", "Running", 0),
        ("local-gpu", "test-namespace", "Running", 1),
        ("cpu-work", "user", "Running", 0),
    ]:
        items.append(
            {
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "nodeName": "node-a",
                    "containers": [{"resources": {"limits": {"nvidia.com/gpu": gpus}}}],
                },
                "status": {"phase": phase},
            }
        )
    fixture.documents[("gpu", "pod", "")] = {"items": items}
    assert [item["name"] for item in fixture.gpu_workloads()] == [
        "local-gpu",
        "pending",
        "training",
    ]
    assert "--all-namespaces" in fixture.calls[-1][0]
    running = [item for item in items if item["status"]["phase"] == "Running"]
    fixture.documents[("gpu", "pod", "")] = {"items": running}
    assert [item["name"] for item in fixture.business_workloads("node-a")] == [
        "training",
        "local-gpu",
        "cpu-work",
    ]
    assert "spec.nodeName=node-a,status.phase=Running" in fixture.calls[-1][0]


def test_cpu_blast_snapshot_tracks_owned_jobs_and_eviction_events(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    value = node("cpu-a", 0)
    value["metadata"]["labels"]["gpu-fault.io/role"] = "control"
    value["spec"]["taints"] = [
        {"key": "example", "value": "held", "effect": "NoSchedule"}
    ]
    fixture.documents[("cpu", "node", "")] = {"items": [value]}
    fixture.documents[("cpu", "job", "")] = {
        "items": [
            {
                "metadata": {
                    "namespace": "system",
                    "name": "owned",
                    "labels": {"gpu-fault.io/run": "unit"},
                }
            },
            {"metadata": {"namespace": "user", "name": "other", "labels": {}}},
        ]
    }
    fixture.documents[("cpu", "event", "")] = {
        "items": [
            {
                "metadata": {"uid": "e1"},
                "reason": "Evicted",
                "involvedObject": {"name": "pod-a"},
            },
            {"metadata": {"uid": "e2"}, "reason": "Scheduled"},
        ]
    }
    snapshot = fixture.cpu_blast_snapshot()
    assert snapshot["nodes"]["cpu-a"]["taints"] == [("example", "held", "NoSchedule")]
    assert snapshot["nodes"]["cpu-a"]["labels"] == {"gpu-fault.io/role": "control"}
    assert snapshot["gpu_fault_jobs"] == [("system", "owned")]
    assert snapshot["eviction_events"] == [("e1", "Evicted", "pod-a")]
