"""Selected Collector case dispatch, preflight and cleanup ownership."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_acceptance as ordinary
from scripts.e2e.regional import run_collector_destructive as destructive
from tests.regional._cov95_collect_net import (  # noqa: F401
    StopLoop,
    no_external_effects,
)
from tests.regional._cov95_collect_scope import INSTANCE, RebootScopeReads

ROLE = "arn:aws:iam::123456789012:role/role-fixture"
HYPERPOD = "hyperpod-fixture"

CASES = [
    (module, case_id)
    for module in (ordinary, destructive)
    for case_id in module.CASE_IDS
]


@pytest.fixture
def dispatch(tmp_path: Path, monkeypatch: Any) -> Any:
    kubeconfig = tmp_path / "fixture-kubeconfig"
    kubeconfig.touch()
    harness = SimpleNamespace(
        calls=[],
        problem="",
        predecessor=True,
        focused=0,
        released_errors=[],
        released_workflows=[],
        kwargs={},
    )
    regional_settings = SimpleNamespace(
        gpu_kubeconfig=kubeconfig,
        gpu_context="gpu-a",
        namespace="fixture",
        cluster_id="cluster-a",
        region="us-west-2",
        environment=lambda: {"CLUSTER": "cluster-a"},
    )
    harness.regional_settings = regional_settings
    # COLLECT-004 binds its reboot scope at preflight through read-only AWS and
    # Kubernetes queries; answer them with one consistent synthetic identity.
    scope_reads = RebootScopeReads(
        node="node-a",
        hyperpod_cluster=HYPERPOD,
        executor_role_arn=ROLE,
        cluster_id="cluster-a",
        namespace="fixture",
        region="us-west-2",
    )
    harness.scope_reads = scope_reads

    class Regional:
        settings = regional_settings

        def run(self, command: list[str], **kwargs: Any) -> Any:
            return scope_reads.run(command, **kwargs)

        def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
            return scope_reads.kubectl(plane, *arguments, **kwargs)

        def node_snapshot(self, node: str) -> dict[str, Any]:
            return {
                "name": node,
                "uid": node + "-uid",
                "ready": "False" if harness.problem == "ready" else "True",
                "ownership_annotations": {"owner": "foreign"}
                if harness.problem == "ownership"
                else {},
                "unschedulable": harness.problem == "cordon",
                "taints": [],
            }

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "agent": {
                    "lifecycle_state": "INACTIVE"
                    if harness.problem == "agent"
                    else "ACTIVE",
                    "generation": 1,
                },
                "profile": {"profile_version": "profile-a"},
            }

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "release-a", "cluster_id": "cluster-a"}

        def business_workloads(self, node: str) -> list[Any]:
            return ["workload"] if harness.problem == "workload" else []

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            return (
                {"changed": True}
                if harness.problem == "blast"
                and any(call[0] == "handler" for call in harness.calls)
                else {}
            )

    regional = Regional()

    class Factory:
        def __new__(cls, settings: Any) -> Regional:
            return regional

        @staticmethod
        def run(command: list[str], **kwargs: Any) -> Any:
            harness.calls.append(("focused", command, kwargs))
            return subprocess.CompletedProcess(
                command, harness.focused, "fixture stdout", "fixture stderr"
            )

    class Collector:
        def __init__(self, instance: Any, **kwargs: Any) -> None:
            self.node = kwargs["node"]
            self.regional = instance
            harness.calls.append(("collector", self.node, kwargs))

        def create(self) -> None:
            harness.calls.append(("create", self.node))
            if harness.problem == "create":
                raise RuntimeError("create failed")

        def cleanup(self) -> dict[str, bool]:
            harness.calls.append(("collector-cleanup", self.node))
            if harness.problem == "collector-cleanup":
                raise RuntimeError("collector cleanup failed")
            return {"pod": harness.problem == "residual"}

    class Host:
        def __init__(self, settings: Any) -> None:
            harness.calls.append(("host", settings))

        def create(self) -> None:
            harness.calls.append(("host-create",))
            if harness.problem == "host-create":
                raise RuntimeError("host create failed")

        def cleanup(self) -> dict[str, bool]:
            harness.calls.append(("host-cleanup",))
            if harness.problem == "host-cleanup":
                raise RuntimeError("host cleanup failed")
            return {"pod": harness.problem == "residual"}

    class Cleanup:
        def finish(self, **kwargs: Any) -> dict[str, Any]:
            harness.calls.append(("restore", kwargs))
            return {
                "errors": harness.released_errors,
                "restore_workflows": harness.released_workflows,
            }

    for module in (ordinary, destructive):
        monkeypatch.setattr(module, "RegionalLiveFixture", Factory)
        monkeypatch.setattr(module, "CollectorAcceptanceFixture", Collector)
        monkeypatch.setattr(module, "CaseCleanup", Cleanup)
        monkeypatch.setattr(
            module,
            "predecessor_evidence",
            lambda *a, **k: {"valid": harness.predecessor},
        )
        monkeypatch.setattr(module, "reusable_focused_tests", lambda path: None)
        monkeypatch.setattr(
            module,
            "record_focused_tests",
            lambda details, record: details.update(focused_tests=record),
        )
        monkeypatch.setattr(
            module, "settings_from_arguments", lambda args: regional_settings
        )
        for case_id in module.CASE_IDS:
            name = "run_collect" + case_id.rsplit("-", 1)[1]

            def handler(*args: Any, name: str = name, **kwargs: Any) -> dict[str, Any]:
                harness.calls.append(("handler", name, args, kwargs))
                if harness.problem == "handler":
                    raise RuntimeError("handler failed")
                if harness.problem == "abort":
                    raise StopLoop()
                return {"verdict": "PASS", "errors": []}

            monkeypatch.setattr(module, name, handler)
    monkeypatch.setattr(destructive, "HostProbeFixture", Host)
    return harness


def settings_for(harness: Any, module: Any, case_id: str, root: Path) -> Any:
    if module is ordinary:
        return ordinary.Settings(
            harness.regional_settings,
            case_id,
            ("node-a", "node-b") if case_id.endswith("011") else ("node-a",),
            "example",
            root / "predecessor.json",
        )
    return destructive.Settings(
        harness.regional_settings,
        case_id,
        "node-a",
        None,
        "example",
        HYPERPOD,
        ROLE,
        None,
        root / "predecessor.json",
    )


@pytest.mark.parametrize("module,case_id", CASES, ids=[case for _, case in CASES])
def test_selected_case_dispatches_exact_handler_and_releases_owned_resources(
    dispatch: Any, tmp_path: Path, module: Any, case_id: str
) -> None:
    settings = settings_for(dispatch, module, case_id, tmp_path)
    assert (
        module.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 0
    )
    handlers = [call for call in dispatch.calls if call[0] == "handler"]
    assert len(handlers) == 1
    assert handlers[0][1] == "run_collect" + case_id.rsplit("-", 1)[1]
    assert any(call[0] == "restore" for call in dispatch.calls), (
        "registered containment cleanup must run"
    )
    created = [call[1] for call in dispatch.calls if call[0] == "collector"]
    cleaned = [call[1] for call in dispatch.calls if call[0] == "collector-cleanup"]
    assert cleaned == created
    document = json.loads(
        (tmp_path / "cases" / case_id / f"{case_id}.json").read_text()
    )
    assert document["verdict"] == "PASS"
    assert document["cluster_id"] == "cluster-a"
    if module is destructive:
        required = case_id.endswith(("008", "013", "014"))
        assert any(call[0] == "host-create" for call in dispatch.calls) is required
        assert any(call[0] == "host-cleanup" for call in dispatch.calls) is required
    if case_id.endswith("004"):
        preflight = json.loads(
            (tmp_path / "cases" / case_id / "preflight.json").read_text()
        )
        scope = preflight["reboot_scope"]
        assert (
            scope["node_uid"],
            scope["instance_id"],
            scope["executor_role_arn"],
            scope["cluster_name"],
        ) == ("node-a-uid", INSTANCE, ROLE, HYPERPOD), (
            "COLLECT-004 preflight must bind the fixture node to its provider scope"
        )
        aws_verbs = {call[2] for call in dispatch.scope_reads.calls if call[0] == "aws"}
        assert aws_verbs and all(
            verb.startswith(("describe-", "list-")) for verb in aws_verbs
        ), f"scope preflight must only issue read-only AWS verbs, saw {aws_verbs}"


@pytest.mark.parametrize("module", [ordinary, destructive])
@pytest.mark.parametrize(
    "problem",
    ["create", "handler", "blast", "collector-cleanup", "residual", "restore"],
)
def test_case_failure_always_reaches_all_registered_cleanup(
    dispatch: Any, tmp_path: Path, module: Any, problem: str
) -> None:
    case_id = (
        "GF-REGIONAL-COLLECT-005" if module is ordinary else "GF-REGIONAL-COLLECT-013"
    )
    settings = settings_for(dispatch, module, case_id, tmp_path)
    dispatch.problem = problem
    if problem == "restore":
        dispatch.released_errors = ["restore unconfirmed"]
        dispatch.released_workflows = [[{"status": "FAILED"}]]
    assert (
        module.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 1
    )
    document = json.loads(
        (tmp_path / "cases" / case_id / f"{case_id}.json").read_text()
    )
    assert document["verdict"] == "FAIL"
    assert any(call[0] == "collector-cleanup" for call in dispatch.calls), (
        "case failure must not suppress collector cleanup"
    )
    if problem == "restore":
        assert document["cleanup_restore_workflows"] == [[{"status": "FAILED"}]]
        assert document["cleanup_errors"] == ["restore unconfirmed"]


@pytest.mark.parametrize("problem", ["host-create", "host-cleanup"])
def test_reset_host_failure_does_not_suppress_collector_cleanup(
    dispatch: Any, tmp_path: Path, problem: str
) -> None:
    case_id = "GF-REGIONAL-COLLECT-013"
    dispatch.problem = problem
    assert (
        destructive.execute_case(
            settings_for(dispatch, destructive, case_id, tmp_path),
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        == 1
    )
    assert any(call[0] == "host-cleanup" for call in dispatch.calls), (
        "failed reset host must still be offered cleanup"
    )
    assert any(call[0] == "collector-cleanup" for call in dispatch.calls), (
        "reset cleanup failure must not suppress collector cleanup"
    )


@pytest.mark.parametrize("module", [ordinary, destructive])
def test_abort_still_runs_owned_resource_cleanup(
    dispatch: Any, tmp_path: Path, module: Any
) -> None:
    case_id = (
        "GF-REGIONAL-COLLECT-005" if module is ordinary else "GF-REGIONAL-COLLECT-013"
    )
    dispatch.problem = "abort"
    with pytest.raises(StopLoop):
        module.execute_case(
            settings_for(dispatch, module, case_id, tmp_path),
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
    assert any(call[0] == "restore" for call in dispatch.calls), (
        "abort must restore registered containment"
    )
    assert any(call[0] == "collector-cleanup" for call in dispatch.calls), (
        "abort must release owned collector resources"
    )


@pytest.mark.parametrize("module", [ordinary, destructive])
@pytest.mark.parametrize(
    "problem", ["predecessor", "ready", "ownership", "agent", "workload", "focused"]
)
def test_preflight_failure_stops_before_fixture_construction(
    dispatch: Any, tmp_path: Path, module: Any, problem: str
) -> None:
    case_id = (
        "GF-REGIONAL-COLLECT-005" if module is ordinary else "GF-REGIONAL-COLLECT-013"
    )
    settings = settings_for(dispatch, module, case_id, tmp_path)
    dispatch.problem = problem
    dispatch.predecessor = problem != "predecessor"
    dispatch.focused = int(problem == "focused")
    preflight = module.read_only_preflight(settings, tmp_path)
    assert len(preflight["errors"]) == 1
    with pytest.raises(RuntimeError, match="preflight"):
        module.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
    assert not any(call[0] in {"collector", "host"} for call in dispatch.calls), (
        "invalid admission cannot create a fixture"
    )


def test_destructive_provider_identity_and_ordinary_cordon_are_required(
    dispatch: Any, tmp_path: Path
) -> None:
    settings = settings_for(dispatch, destructive, "GF-REGIONAL-COLLECT-004", tmp_path)
    result = destructive.read_only_preflight(
        replace(settings, hyperpod_cluster=""), tmp_path
    )
    assert any("HyperPod" in error for error in result["errors"]), result
    dispatch.problem = "cordon"
    result = ordinary.read_only_preflight(
        settings_for(dispatch, ordinary, "GF-REGIONAL-COLLECT-005", tmp_path), tmp_path
    )
    assert any("schedulable" in error for error in result["errors"]), result


@pytest.mark.parametrize("module,case_id", CASES, ids=[case for _, case in CASES])
def test_configure_and_plan_roundtrip_preserve_selected_case(
    dispatch: Any, tmp_path: Path, module: Any, case_id: str
) -> None:
    argv = [
        "--run-dir",
        str(tmp_path),
        "--case",
        case_id,
        "--node",
        "node-a",
        "--host-probe-image",
        "example",
    ]
    if case_id.endswith("011"):
        argv += ["--node", "node-b"]
    if module is destructive:
        argv += [
            "--site-file",
            str(tmp_path / "site"),
            "--predecessor-evidence",
            str(tmp_path / "previous"),
            "--hyperpod-cluster",
            HYPERPOD,
            "--executor-role-arn",
            ROLE,
        ]
    settings = module.configure(module.parser().parse_args(argv))
    assert settings.confirmation.endswith("_EXECUTE"), (
        "configured case must keep its explicit execution confirmation"
    )
    assert settings.environment()["GPU_FAULT_COLLECT_CASE"] == case_id
    details = module.plan_details(
        settings, module.read_only_preflight(settings, tmp_path)
    )
    assert details["case_id"] == case_id
    assert details["preflight_identity"]["release_id"] == "release-a"
    assert details["preflight"]["errors"] == [], (
        "a clean fixture must plan without preflight errors"
    )


@pytest.mark.parametrize("tolerance", [float("nan"), -0.1, 1.1])
def test_invalid_debounce_configuration_is_refused(
    dispatch: Any, tmp_path: Path, tolerance: float
) -> None:
    args = destructive.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--case",
            "GF-REGIONAL-COLLECT-004",
            "--node",
            "node-a",
            "--host-probe-image",
            "example",
        ]
    )
    args.debounce_tolerance = tolerance
    with pytest.raises(RuntimeError, match="tolerance"):
        destructive.configure(args)


def test_invalid_node_count_does_not_silently_pick_a_target(
    dispatch: Any, tmp_path: Path
) -> None:
    args = ordinary.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--case",
            "GF-REGIONAL-COLLECT-011",
            "--node",
            "node-a",
            "--node",
            "node-a",
            "--host-probe-image",
            "example",
        ]
    )
    with pytest.raises(RuntimeError, match="distinct"):
        ordinary.configure(args)
