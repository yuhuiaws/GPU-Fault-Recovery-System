from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import (
    aurora_binding,
    ha_cleanup,
    ha_evidence,
    ha_plan_preflight,
    ha_probe_resources,
    ha_store_probe,
    regional_commands,
)
from scripts.e2e.regional import ha009_observation as observation
from tests.regional._aurora_binding_support import CLUSTER, NAMESPACE, FakeAurora
from tests.regional._cov95_ha001_harness import Clock
from tests.regional.test_ha_resource_guards import ProbeApi, pod_manifest


@pytest.mark.parametrize("mode", ["timeout", "converged"])
def test_owned_resource_deletion_waits_for_absence_or_fails_at_its_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    api = ProbeApi()
    clock = Clock()
    monkeypatch.setattr(ha_probe_resources, "time", clock)
    deleting = False

    def command(args: list[str], body: str | None) -> str:
        nonlocal deleting
        if args[0] == "delete":
            deleting = True
            return ""
        if deleting and mode == "converged" and clock.now >= 1:
            api.items.clear()
        return api.command(args, body)

    resources = ha_probe_resources.OwnedProbeResources(
        tmp_path / "receipt.json", command
    )
    resources.create(pod_manifest())
    if mode == "timeout":
        with pytest.raises(RuntimeError, match="did not converge"):
            resources.delete("Pod", "probe", timeout_seconds=1)
        assert len(api.items) == 1
    else:
        resources.delete("Pod", "probe", timeout_seconds=2)
        assert resources.records["Pod/probe"]["deleted"] is True
    assert clock.now == 1


@pytest.mark.parametrize(
    "document", ["[]", "{}", '{"metadata":{"name":"other","uid":"x"}}']
)
def test_resource_reads_require_complete_matching_identity(
    tmp_path: Path, document: str
) -> None:
    resources = ha_probe_resources.OwnedProbeResources(
        tmp_path / "receipt.json", lambda *a: document
    )
    with pytest.raises(RuntimeError, match="identity read is incomplete"):
        resources.read("Pod", "probe")
    with pytest.raises(RuntimeError, match="no creation receipt"):
        resources.owned("Pod", "probe")


def test_missing_create_observation_retains_intent_without_inventing_uid(
    tmp_path: Path,
) -> None:
    resources = ha_probe_resources.OwnedProbeResources(
        tmp_path / "receipt.json", lambda *a: ""
    )
    with pytest.raises(RuntimeError, match="not observed"):
        resources.create(pod_manifest())
    assert resources.records["Pod/probe"]["uid"] is None
    resources.delete("Pod", "probe")
    resources.delete("Pod", "unowned")
    with pytest.raises(RuntimeError, match="receipt already exists"):
        ha_probe_resources.OwnedProbeResources(resources.path, lambda *a: "")


def test_cleanup_errors_are_collected_but_supervision_loss_is_not_swallowed() -> None:
    result: dict[str, Any] = {"verdict": "PASS"}

    def failed() -> None:
        raise OSError("unit failure")

    assert ha_cleanup.attempt_cleanup(result, "owned cleanup", failed) is False
    assert result["cleanup_errors"] == ["owned cleanup: OSError: unit failure"]
    assert ha_cleanup.attempt_cleanup(result, "next", lambda: None) is True

    def lost() -> None:
        raise ProcessSupervisionLost("unit")

    assert ha_cleanup.run_cleanup(result, lost) is None
    assert result["supervision_lost"] is True
    assert result["verdict"] == "FAIL"


def test_residual_preflight_keeps_failed_and_unknown_counts_distinct() -> None:
    def unreadable() -> dict:
        raise OSError("unit")

    result = ha_plan_preflight.residual_preflight(
        unreadable, lambda: {"count": False}, lambda: {"count": 0}
    )
    assert result["errors"] == [
        "database: preflight read failed",
        "registry: residual count is missing or nonzero",
    ]
    assert result["database"] == {"error_type": "OSError"}
    assert result["kubernetes"] == {"count": 0}
    with pytest.raises(RuntimeError, match="maintenance window"):
        ha_plan_preflight.require_window(datetime(2099, 1, 1))


def test_store_probe_rejects_absent_ready_cpu_and_nonobject_response() -> None:
    with pytest.raises(RuntimeError, match="Ready CPU"):
        ha_store_probe.cpu_store_probe(lambda *a, **kw: '{"items":[]}', "")
    from tests.regional._cov95_ha001_harness import pod

    def control(*args: str, **kwargs: Any) -> str:
        return (
            json.dumps({"items": [pod("unit", "unit")]}) if args[0] == "get" else "[]"
        )

    with pytest.raises(ValueError, match="not an object"):
        ha_store_probe.cpu_store_probe(control, "")


def test_evidence_requires_release_and_cluster_and_supports_first_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = {"release_id": "", "cluster_id": "unit"}
    monkeypatch.setattr(ha_evidence, "settings_from_arguments", lambda a: a)
    monkeypatch.setattr(
        ha_evidence,
        "RegionalLiveFixture",
        lambda _: SimpleNamespace(evidence_identity=lambda: identity),
    )
    args = SimpleNamespace(run_dir=tmp_path, release_id="unit", cluster_id="unit")
    with pytest.raises(RuntimeError, match="release and cluster identity"):
        ha_evidence.chain_preflight(args, "unit-case")
    identity["release_id"] = "unit"
    monkeypatch.setattr(ha_evidence, "predecessor_path", lambda *a: (None, None))
    assert ha_evidence.chain_preflight(args, "unit-case")["errors"] == []
    assert (
        ha_evidence.isolated_chain(args, "unit-case")["predecessor"]["verdict"]
        == "NOT_REQUIRED"
    )


@pytest.mark.parametrize("output", ["", "invalid-json", "[]"])
def test_pod_rotation_observation_rejects_invalid_protocol(output: str) -> None:
    with pytest.raises(ValueError, match="JSON|object"):
        observation.pod_observation(
            lambda *a, **kw: output, "unit", 8081, python="unit-python"
        )


def test_pool_metric_parser_does_not_invent_values_from_bad_samples() -> None:
    name = observation.POOL_METRIC_NAMES[0]
    with pytest.raises(ValueError, match="invalid"):
        observation.parse_pool_metrics(f"{name} invalid\nother 10\n{name} 3\n")
    assert observation.parse_pool_metrics(f"other 10\n{name} 3\n") == {name: 3.0}, (
        "a valid pool sample must remain observable without malformed evidence"
    )
    result = observation.observation_errors(
        {"unit"},
        {"digest": "expected", "pods": {"unit": "wrong"}},
        {"samples": {"unit": []}, "auth_failures_in_logs": {"unit": 1}},
    )
    assert len(result) == 3
    assert any("did not catch up" in item for item in result), result
    assert any("not observed" in item for item in result), result
    assert any("authentication failure count" in item for item in result), result


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "generation",
        "zero",
        "invalid-replicas",
        "baseline-pods",
        "final-pods",
        "pod-uid",
        "restart",
        "unready",
        "rollout",
    ],
)
def test_rotation_steady_state_rejects_partial_or_changed_populations(
    fault: str,
) -> None:
    before = {
        "unit": {
            "uid": "deployment-uid",
            "generation": 1,
            "replicas": 1,
            "pods": {"pod": {"uid": "pod-uid", "restarts": 0, "ready": True}},
            "ready": 1,
            "updated": 1,
            "available": 1,
        }
    }
    current = copy.deepcopy(before)
    if fault == "missing":
        current.clear()
    elif fault == "generation":
        current["unit"]["generation"] = 2
    elif fault in {"zero", "invalid-replicas"}:
        before["unit"]["replicas"] = current["unit"]["replicas"] = (
            0 if fault == "zero" else True
        )
    elif fault == "baseline-pods":
        before["unit"]["pods"] = {}
    elif fault == "final-pods":
        current["unit"]["pods"] = {}
    elif fault == "pod-uid":
        current["unit"]["pods"]["pod"]["uid"] = "replacement"
    elif fault == "restart":
        current["unit"]["pods"]["pod"]["restarts"] = 1
    elif fault == "unready":
        current["unit"]["pods"]["pod"]["ready"] = False
    else:
        current["unit"]["updated"] = 0
    assert observation.steady_deployments(before, current, ["unit"]), (
        f"{fault} must violate zero-rollout observation"
    )


def test_regional_binding_bridges_real_metadata_transport_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeAurora()
    calls = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv[:3])
        result = api.aws(argv[1], argv[2], *argv[3:-4])
        return subprocess.CompletedProcess(argv, 0, json.dumps(result), "")

    def kubectl(plane: str, *args: str, **kwargs: Any) -> str:
        assert plane == "cpu"
        return api.control(*args, **kwargs)

    monkeypatch.setattr(regional_commands, "run_fixture_command", run)
    regional = SimpleNamespace(
        settings=SimpleNamespace(region="us-east-1", namespace=NAMESPACE),
        kubectl=kubectl,
    )
    guard = aurora_binding.regional_binding(regional, CLUSTER)
    proof = guard.read()
    assert (
        proof["identity"]["database"]["cluster_resource_id"] == "cluster-unit-resource"
    )
    assert calls, "binding must inspect real metadata through the fake AWS transport"
    api.failure = ProcessSupervisionLost("unit")
    with pytest.raises(ProcessSupervisionLost, match="supervision was lost"):
        api.guard.refresh_job("unit-job", {})


def test_kms_bound_refresher_requires_and_proves_the_exact_decrypt_permission() -> None:
    api = FakeAurora()
    key = "arn:aws:kms:us-east-1:111122223333:key/unit-key"
    api.cluster["MasterUserSecret"]["KmsKeyId"] = key
    api.policy["Statement"].append(
        {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": key}
    )
    proof = api.guard.read()
    assert proof["identity"]["database"]["kms_key_arn"] == key
    assert api.guard.read(proof, refreshing=True)["identity"] == proof["identity"]
