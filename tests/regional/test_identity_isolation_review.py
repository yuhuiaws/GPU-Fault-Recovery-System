from __future__ import annotations

import copy
import io
import json
import ssl
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from scripts.e2e.regional import identity_acceptance_iso as iso
from scripts.e2e.regional.identity_acceptance_common import (
    IdentityAcceptanceError,
    IdentityCaseFailure,
)
from tests._builders import build_context
from tests.regional._regional_support import TOKEN_A, registration


def execute_probe(
    monkeypatch: pytest.MonkeyPatch, script: str, *arguments: str
) -> dict[str, Any]:
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "argv", ["isolated-probe", *arguments])
        output = io.StringIO()
        with redirect_stdout(output):
            exec(script, {})
        return json.loads(output.getvalue())


def namespace_harness(
    monkeypatch: pytest.MonkeyPatch, *, namespaces: list[str] | None = None
) -> tuple[Any, Any, Any, list[str]]:
    context = build_context()
    member = registration("cluster-a", TOKEN_A).model_copy(
        update={
            "allowed_namespaces": namespaces if namespaces is not None else ["training"]
        }
    )
    context.store.save_regional_cluster(member)
    monkeypatch.setattr(ApplicationContext, "from_environment", lambda: context)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", "training")
    calls: list[str] = []

    def cpu_python(script: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["attempts"] == 1
        calls.append("cpu")
        return execute_probe(monkeypatch, script, *args)

    def executor_python(script: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["attempts"] == 1
        calls.append("executor")
        return execute_probe(monkeypatch, script, *args)

    regional = SimpleNamespace(cpu_python=cpu_python, executor_python=executor_python)
    site = SimpleNamespace(
        namespace="training",
        registry=lambda: [member.model_dump(mode="json")],
        regional=lambda target: regional,
    )
    return site, SimpleNamespace(cluster_id="cluster-a"), context.store, calls


def test_namespace_guards_use_deployed_logic_without_a_production_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, store, calls = namespace_harness(monkeypatch)
    before = store.get_regional_cluster("cluster-a").model_dump(mode="json")
    result = iso.run_iso005(site, target, case_dir=tmp_path)
    assert result["verdict"] == "PASS", result
    assert calls == ["cpu", "cpu", "executor"]
    assert store.list_remote_commands() == [], "a probe command reached the real Store"
    assert store.get_regional_cluster("cluster-a").model_dump(mode="json") == before
    assert result["cleanup"] == {"resources_created": False, "registry_modified": False}
    assert result["checks"]["allowed_namespace_reaches_adapter_selection"] is True
    saved = json.loads((tmp_path / "iso005-details.json").read_text())
    assert saved["fixture_scope"] == "isolated deployed-code guards"


def test_wrong_control_plane_denial_stops_before_the_executor_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, store, calls = namespace_harness(monkeypatch)
    monkeypatch.setattr(
        RegionalRemoteWorkflowAdapter,
        "execute",
        lambda *args: WorkflowStepOutcome.failed("unrelated refusal"),
    )
    with pytest.raises(IdentityCaseFailure, match="did not refuse") as caught:
        iso.run_iso005(site, target, case_dir=tmp_path)
    assert calls == ["cpu"]
    assert caught.value.details["verdict"] == "FAIL"
    assert store.list_remote_commands() == []
    assert (
        json.loads((tmp_path / "iso005-details.json").read_text())["verdict"] == "FAIL"
    )


@pytest.mark.parametrize("namespaces", [[], [iso.FORBIDDEN_NAMESPACE]])
def test_invalid_namespace_preconditions_do_not_start_any_probe(
    namespaces: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, store, calls = namespace_harness(monkeypatch, namespaces=namespaces)
    with pytest.raises(IdentityAcceptanceError):
        iso.run_iso005(site, target, case_dir=tmp_path)
    assert calls == []
    assert store.list_remote_commands() == []


@pytest.mark.parametrize("allowed", ["", "training," + iso.FORBIDDEN_NAMESPACE])
def test_executor_namespace_drift_cannot_be_hidden_by_control_plane_rejection(
    allowed: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, store, calls = namespace_harness(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", allowed)
    result = iso.run_iso005(site, target, case_dir=tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["checks"]["executor_rejects_local_allowlist"] is False
    assert store.list_remote_commands() == []


def test_executor_probe_refuses_a_pod_bound_to_another_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, store, calls = namespace_harness(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-b")
    with pytest.raises(IdentityCaseFailure, match="Pod cluster differs"):
        iso.run_iso005(site, target, case_dir=tmp_path)
    assert calls == ["cpu", "cpu", "executor"]
    assert store.list_remote_commands() == []


def test_namespace_case_does_not_restore_over_a_concurrent_registry_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, target, _store, _calls = namespace_harness(monkeypatch)
    original = site.registry()
    changed = copy.deepcopy(original)
    changed[0]["allowed_namespaces"] = ["other"]
    snapshots = iter([original, changed])
    site.registry = lambda: next(snapshots)
    result = iso.run_iso005(site, target, case_dir=tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["checks"]["registry_unchanged"] is False
    assert changed[0]["allowed_namespaces"] == ["other"]


def test_fleet_probe_invokes_the_real_local_guards_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", TOKEN_A)
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", "/unused-public-ca")
    monkeypatch.setattr(
        ssl, "create_default_context", lambda **kwargs: SimpleNamespace()
    )
    result = execute_probe(monkeypatch, iso.FLEET_CROSS_CLUSTER_PROBE, "cluster-b")
    assert result["request_count"] == 0
    assert result["list_agents"]["error"] == "cannot read agents for another cluster"
    assert (
        result["revoke_agent"]["error"]
        == "cannot transition an agent in another cluster"
    )


@pytest.mark.parametrize("defect", ["none", "state", "request", "reason"])
def test_fleet_case_requires_local_denial_and_unchanged_secondary(defect: str) -> None:
    before = {"agents": [{"node_id": "b", "generation": 1}]}
    after = copy.deepcopy(before)
    if defect == "state":
        after["agents"][0]["generation"] = 2
    snapshots = iter([before, after])
    result = {
        "list_agents": {
            "rejected": True,
            "error": "cannot read agents for another cluster",
        },
        "revoke_agent": {
            "rejected": True,
            "error": "cannot transition an agent in another cluster",
        },
        "request_count": 1 if defect == "request" else 0,
    }
    if defect == "reason":
        result["revoke_agent"]["error"] = "HTTP failure"
    site = SimpleNamespace(
        regional=lambda target: SimpleNamespace(
            cpu_python=lambda *args: next(snapshots)
        ),
        any_executor_pod=lambda target: "executor",
        pod_json=lambda *args: result,
    )
    outcome = iso.run_iso003(
        site, SimpleNamespace(cluster_id="a"), SimpleNamespace(cluster_id="b")
    )
    assert outcome["verdict"] == ("PASS" if defect == "none" else "FAIL")


@pytest.mark.parametrize("defect", ["none", "denial", "reason", "ready", "status"])
def test_spare_health_case_judges_both_the_negative_and_positive_control(
    defect: str,
) -> None:
    results = {
        "a": {
            "status": 200,
            "body": {"ready": False, "reasons": [iso.MISSING_AGENT_REASON]},
        },
        "b": {
            "status": 403,
            "body": {
                "detail": "authenticated cluster does not match all payload cluster_id values"
            },
        },
    }
    if defect == "denial":
        results["b"]["body"]["detail"] = "wrong token"
    elif defect == "reason":
        results["a"]["body"]["reasons"] = ["unhealthy pool"]
    elif defect == "ready":
        results["a"]["body"]["ready"] = True
    elif defect == "status":
        results["a"]["status"] = 503
    site = SimpleNamespace(
        any_executor_pod=lambda target: "executor",
        pod_json=lambda *args: {"results": results},
    )
    outcome = iso.run_iso004(
        site, SimpleNamespace(cluster_id="a"), SimpleNamespace(cluster_id="b")
    )
    assert outcome["verdict"] == ("PASS" if defect == "none" else "FAIL")
