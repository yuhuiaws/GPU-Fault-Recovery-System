from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkloadState
from gpu_fault.node_agent.ledger import canonical_digest
from gpu_fault.nvidia_logs import FabricManagerLogEvent, NvidiaLogNormalizer
from gpu_fault.policy import GpuFaultPolicyEngine, SxidLinkScope
from scripts.e2e.regional import collector_negative_evidence as negative
from scripts.e2e.regional import collector_sxid_evidence as evidence
from scripts.e2e.regional import run_collector_acceptance as acceptance
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests._builders import active_workflow_executor, execute_workflow
from tests.regional._collector_reset_support import reset_documents


def sxid_state(
    sxid: int,
    classification: str,
    *,
    scope: str = "UNKNOWN",
    complete_inventory: bool = False,
    node: str = "node-a",
) -> dict[str, Any]:
    """Actual parser/compiler and simulated containment, never a node actuator."""

    source = FabricManagerLogEvent(
        cluster_id="cluster-a",
        node_id=node,
        record_id=f"component-{sxid}-{scope}",
        observed_at=datetime.now(timezone.utc),
        message=(
            f"nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): {sxid}, "
            f"{classification}, Link 12"
        ),
        source="component://collector-sxid-contract",
        product="H200",
        workload_state=WorkloadState.IDLE,
        runtime_profile_version="simulated-v1",
    )
    event = NvidiaLogNormalizer().normalize_fabric_manager(source).sxid_events[0]
    event = event.model_copy(
        update={
            "link_scope": SxidLinkScope(scope),
            "link_scope_source": "NVIDIA_PRODUCT_INVARIANT"
            if scope == "ACCESS"
            else None,
            "switch_id": "nvidia-nvswitch0" if scope == "ACCESS" else None,
            "fabric_partition": "component-fabric" if complete_inventory else None,
            "participating_gpu_uuids": ["GPU-a", "GPU-b"] if complete_inventory else [],
        }
    )
    context = ApplicationContext()
    decision = context.policy.evaluate_sxid(event)
    incident, workflow = context.orchestrator.ingest(event, decision)
    assert workflow is not None
    if workflow.status.value == "SAFETY_PENDING":
        adapter = SimpleNamespace(
            supports=lambda step: step.execution_owner == "simulated-runtime",
            execute=lambda step_context: WorkflowStepOutcome.succeeded(
                operation_id=step_context.idempotency_key
            ),
        )
        executor = active_workflow_executor(
            context.store, [adapter], [step.operation for step in workflow.safety_steps]
        )
        execute_workflow(
            executor, workflow.request_id, expected_fencing_token=workflow.fencing_token
        )
        workflow = context.store.get_workflow(workflow.request_id)
    decision = decision.model_copy(
        update={
            "incident_id": incident.incident_id,
            "workflow_request_id": workflow.request_id,
        }
    )
    return {
        "fabric_events": [event.model_dump(mode="json")],
        "decisions": [decision.model_dump(mode="json")],
        "incidents": [incident.model_dump(mode="json")],
        "workflows": [workflow.model_dump(mode="json")],
        "commands": [],
    }


@pytest.mark.parametrize("sxid", [11001, 12001, 24007])
@pytest.mark.parametrize("scope", ["ACCESS", "UNKNOWN"])
def test_code_specific_scope_variants_use_parser_and_policy(
    sxid: int, scope: str
) -> None:
    state = sxid_state(sxid, "Fatal", scope=scope)
    assert negative.scope_variant_errors(state, sxid=sxid, scope=scope) == []
    assert state["fabric_events"][0]["sxid"] == sxid
    assert state["decisions"][0]["action"] is None
    assert state["workflows"][0]["status"] == "BLOCKED"
    assert {step["operation"] for step in state["workflows"][0]["step_executions"]} <= {
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
    }
    assert not state["commands"]


@pytest.mark.parametrize(
    "sxid,classification", [(10003, "Fatal"), (19084, "Non-fatal")]
)
def test_nonfatal_code_specific_full_reset_is_component_proof_not_execution(
    sxid: int, classification: str
) -> None:
    state = sxid_state(sxid, classification, complete_inventory=True)
    assert (
        evidence.full_reset_variant_errors(
            state, sxid=sxid, classification=classification
        )
        == []
    )
    assert state["decisions"][0]["disposition"] == "EXECUTABLE"
    assert state["workflows"][0]["step_executions"] == []
    assert state["commands"] == []


@pytest.mark.parametrize(
    "sxid,classification", [(10003, "Non-fatal"), (19084, "Fatal")]
)
def test_severity_conflict_cannot_inherit_an_executable_full_reset(
    sxid: int, classification: str
) -> None:
    event = sxid_state(sxid, classification)["fabric_events"][0]
    from gpu_fault.policy import SxidEvent

    decision = GpuFaultPolicyEngine().evaluate_sxid(SxidEvent.model_validate(event))
    assert decision.disposition.value == "BLOCKED_MISSING_EVIDENCE"
    assert decision.action is None
    assert any("severity conflict" in reason for reason in decision.reasons), (
        "the blocked decision must identify the SXID severity conflict"
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("event", "sxid", 10003),
        ("event", "classification", "FATAL"),
        ("event", "classification_source", "USER"),
        ("decision", "event_id", "other"),
        ("decision", "event_type", "XID"),
        ("decision", "official_action", "IGNORE"),
        ("decision", "disposition", "MONITOR_ONLY"),
        ("decision", "action", "RESET_GPU"),
        ("decision", "workflow_request_id", "other"),
        ("decision", "incident_id", "other"),
        ("step", "parameters", {"sxid": 10003}),
    ],
)
def test_full_reset_variant_rejects_other_codes_and_unbound_receipts(
    section: str, key: str, value: Any
) -> None:
    state = sxid_state(19084, "Non-fatal", complete_inventory=True)
    step = next(
        step
        for step in state["workflows"][0]["official_steps"]
        if step["operation"] == "RESET_ALL_GPUS_NVSWITCHES"
    )
    target = {
        "event": state["fabric_events"][0],
        "decision": state["decisions"][0],
        "step": step,
    }[section]
    target[key] = value
    assert evidence.full_reset_variant_errors(
        state, sxid=19084, classification="Non-fatal"
    ), "full-reset proof must reject mismatched code, scope, or receipt bindings"


def test_full_reset_variant_requires_complete_unique_selection() -> None:
    state = sxid_state(19084, "Non-fatal", complete_inventory=True)
    assert evidence.full_reset_variant_errors(
        {}, sxid=19084, classification="Non-fatal"
    ), "an empty state must not prove a full-reset variant"
    assert evidence.full_reset_variant_errors(
        state, sxid=99999, classification="Non-fatal"
    ), "full-reset evidence must not authorize an unsupported SXID"
    state["workflows"].append(deepcopy(state["workflows"][0]))
    assert evidence.full_reset_variant_errors(
        state, sxid=19084, classification="Non-fatal"
    ), "duplicate workflow candidates must not satisfy unique full-reset selection"


def healthy_node() -> dict[str, Any]:
    return {
        "uid": "node-uid",
        "ready": "True",
        "ownership_annotations": {},
        "unschedulable": False,
        "taints": [],
    }


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "empty",
        "unknown",
        "duplicate",
        "missing",
        "changed",
        "boot",
        "ready",
        "uid",
        "hold",
        "cordon",
        "taint",
    ],
)
def test_full_reset_stability_demands_complete_same_inventory_and_clean_node(
    problem: str | None,
) -> None:
    baseline = {
        "boot_id": "boot-a",
        "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
    }
    current = deepcopy(baseline)
    node = healthy_node()
    if problem == "empty":
        current["gpu_inventory"] = []
    elif problem == "unknown":
        current["gpu_inventory"][0] = {}
    elif problem == "duplicate":
        current["gpu_inventory"][1]["uuid"] = "GPU-a"
    elif problem == "missing":
        current["gpu_inventory"].pop()
    elif problem == "changed":
        current["gpu_inventory"][1]["uuid"] = "GPU-c"
    elif problem == "boot":
        current["boot_id"] = "boot-b"
    elif problem == "ready":
        node["ready"] = "False"
    elif problem == "uid":
        node["uid"] = ""
    elif problem == "hold":
        node["ownership_annotations"] = {"workflow": "other"}
    elif problem == "cordon":
        node["unschedulable"] = True
    elif problem == "taint":
        node["taints"] = [{"key": "gpu-fault.io/quarantined"}]
    assert bool(evidence.stable_inventory_errors(baseline, current, node)) == (
        problem is not None
    )


class ScopeFixture:
    def __init__(
        self, node: str, scope: str, calls: list, problem: str | None = None
    ) -> None:
        self.node = node
        self.scope = scope
        self.calls = calls
        self.problem = problem
        self.sxid = 11001
        self.last_state: dict[str, Any] = {}
        self.regional = SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            business_workloads=lambda node: ["foreign"] if problem == "busy" else [],
            cpu_python=self.premise,
        )

    def premise(
        self,
        script: str,
        cluster: str,
        node: str,
        bdf: str,
        product: str,
        switch: str,
        sxid: str,
    ) -> dict:
        self.calls.append(("premise", node, int(sxid)))
        assert script == negative.SXID_PREMISE_PROBE
        assert bdf != "0000:01:00.0"
        return {
            "cluster_id": cluster,
            "node_id": node,
            "workload_state": "IDLE",
            "link_scope": "TRUNK" if self.problem == "premise" else self.scope,
            "link_scope_source": "NVIDIA_PRODUCT_INVARIANT",
            "participating_gpu_uuids": [],
        }

    def snapshot(self) -> dict:
        return {
            "boot_id": "boot-a",
            "gpu_inventory": [{"uuid": "GPU-a", "pci_bdf": "0000:01:00.0"}],
            "services": {},
        }

    def execute(self, command: str, *args: str, **kwargs: Any) -> dict:
        if command == "gpu-identity":
            return {"product": "H200"}
        assert command == "append-sxid"
        self.sxid = int(args[args.index("--sxid") + 1])
        self.calls.append(("append", self.node, self.sxid))
        return {}

    def wait_marker(self, marker: str, **kwargs: Any) -> dict:
        state = sxid_state(self.sxid, "Fatal", scope=self.scope, node=self.node)
        state["seed_marker"] = marker
        if self.problem == "receipt":
            state["fabric_events"][0]["sxid"] = 19084
        if self.problem == "workflow":
            state["workflows"] = []
        if self.problem == "status":
            state["workflows"][0]["status"] = "SUCCEEDED"
        self.last_state = deepcopy(state)
        return state

    def store_snapshot(self, marker: str) -> dict:
        assert self.last_state.get("seed_marker") == marker
        return deepcopy(self.last_state)

    def restore_incidents(self, state: dict, **kwargs: Any) -> list:
        self.calls.append(("restore", self.node, self.sxid))
        return []


@pytest.mark.parametrize(
    "problem", [None, "receipt", "workflow", "status", "busy", "premise"]
)
def test_collect011_runs_every_required_code_and_stops_before_later_variant(
    tmp_path: Path, problem: str | None
) -> None:
    calls: list = []
    fixtures = [
        ScopeFixture("node-a", "ACCESS", calls, problem),
        ScopeFixture("node-b", "UNKNOWN", calls),
    ]
    if problem in {"busy", "premise"}:
        with pytest.raises(RegionalFixtureError):
            acceptance.run_collect011(fixtures, tmp_path, 1, "simulated-v1")
        assert not any(row[0] == "append" for row in calls), (
            "an unproven scope premise must stop before any SXID append"
        )
        return
    result = acceptance.run_collect011(fixtures, tmp_path, 1, "simulated-v1")
    appended = [row for row in calls if row[0] == "append"]
    assert result["verdict"] == ("PASS" if problem is None else "FAIL")
    if problem is None:
        assert appended == [
            ("append", node, sxid)
            for sxid in (11001, 12001, 24007)
            for node in ("node-a", "node-b")
        ]
        assert len(result["variants"]) == 6
        assert all(
            row["physical_fault_injected"] is False for row in result["variants"]
        ), "synthetic scope variants must not claim physical fault injection"
        for index, row in enumerate(calls):
            if row[0] == "append":
                assert calls[index - 1][0] == "premise"
                assert calls[index + 1] == ("restore", row[1], row[2])
    else:
        assert len(appended) == 1
        assert len(result["required_variants"]) == 6


def test_scope_variant_allowlist_cannot_be_extended_by_a_direct_caller(
    tmp_path: Path,
) -> None:
    with pytest.raises(RegionalFixtureError, match="not allowlisted"):
        destructive.run_full_reset_variant(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            tmp_path,
            1,
            "profile-a",
            baseline={},
            audit_before={},
            sxid=99999,
            classification="Non-fatal",
            cleanup=acceptance.CaseCleanup(),
        )
    with pytest.raises(RegionalFixtureError):
        acceptance.run_collect011([], tmp_path, 1, "profile-a")
    with pytest.raises(RegionalFixtureError):
        acceptance.run_scope_negative_variant(
            SimpleNamespace(),
            tmp_path,
            "marker",
            "profile-a",
            sxid=99999,
            scope="ACCESS",
            cleanup=acceptance.CaseCleanup(),
        )
    with pytest.raises(RegionalFixtureError):
        acceptance.run_scope_negative_variant(
            SimpleNamespace(),
            tmp_path,
            "marker",
            "profile-a",
            sxid=11001,
            scope="TRUNK",
            cleanup=acceptance.CaseCleanup(),
        )


@pytest.mark.parametrize(
    "sxid,classification", [(10003, "Fatal"), (19084, "Non-fatal")]
)
def test_full_reset_variant_uses_exact_code_and_its_own_physical_audit(
    tmp_path: Path, sxid: int, classification: str
) -> None:
    before, after, reset_state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    after["quiesce_states"] = []
    state = sxid_state(sxid, classification, complete_inventory=True)
    state["workflows"] = [reset_state["workflow"]]
    state["workflows"][0]["official_steps"][0]["parameters"] = {"sxid": sxid}
    after["ledger"][0]["parameters_digest"] = canonical_digest({"sxid": sxid})
    state["incidents"] = [reset_state["incident"]]
    state["decisions"][0].update(
        workflow_request_id=reset_state["workflow"]["request_id"],
        incident_id=reset_state["incident"]["incident_id"],
    )
    calls = []

    def execute(command: str, *args: str, **kwargs: Any) -> dict:
        calls.append((command, args))
        if command == "reset-audit":
            return deepcopy(after)
        if command == "snapshot":
            return deepcopy(after)
        return {}

    host = SimpleNamespace(host_script="/owned/probe.py", execute=execute)
    collector = SimpleNamespace(
        execute=execute,
        wait_marker=lambda *a, **k: state,
        restore_incidents=lambda *a, **k: [],
    )
    result = destructive.run_full_reset_variant(
        SimpleNamespace(node="node-a"),
        host,
        collector,
        tmp_path,
        1,
        "simulated-v1",
        baseline=before,
        audit_before=before,
        sxid=sxid,
        classification=classification,
        cleanup=acceptance.CaseCleanup(),
    )
    assert result["verdict"] == "PASS", result["errors"]
    append = next(args for command, args in calls if command == "append-sxid")
    assert append[append.index("--sxid") + 1] == str(sxid)
    assert append[append.index("--classification") + 1] == classification
    assert any(command == "stop-reset-sampler" for command, args in calls), (
        "each full-reset variant must stop its own reset sampler"
    )
    assert (tmp_path / "reset-audit-after.json").is_file(), (
        "each full-reset variant must retain its post-action audit"
    )


@pytest.mark.parametrize("problem", [None, "inventory", "node-uid", "workload"])
def test_c014_waits_for_cooldown_and_three_clean_inventory_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    baseline = {"boot_id": "boot-a", "gpu_inventory": [{"uuid": "GPU-a"}]}
    reads = []
    sleeps = []
    monkeypatch.setattr(destructive.time, "sleep", sleeps.append)

    def snapshot(*a: Any, **k: Any) -> dict:
        reads.append("snapshot")
        current = deepcopy(baseline)
        if problem == "inventory":
            current["gpu_inventory"] = []
        return current

    def node(_node: str) -> dict:
        value = healthy_node()
        if problem == "node-uid" and len(reads) > 1:
            value["uid"] = "replacement"
        return value

    regional = SimpleNamespace(
        node_snapshot=node,
        business_workloads=lambda node: ["foreign"] if problem == "workload" else [],
    )
    if problem:
        with pytest.raises(RegionalFixtureError):
            destructive.wait_full_reset_stability(
                regional,
                SimpleNamespace(node="node-a"),
                SimpleNamespace(execute=snapshot),
                baseline,
            )
    else:
        records = destructive.wait_full_reset_stability(
            regional,
            SimpleNamespace(node="node-a"),
            SimpleNamespace(execute=snapshot),
            baseline,
        )
        assert len(records) == 3
        assert sleeps == [60, 5, 5]


@pytest.mark.parametrize("failed_code", [None, 10003, 19084])
def test_collect014_requires_both_variants_but_failure_stops_the_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_code: int | None
) -> None:
    before, _after, _state = reset_documents("RESET_ALL_GPUS_NVSWITCHES")
    calls = []
    host = SimpleNamespace(execute=lambda *a, **k: before)
    collector = SimpleNamespace(
        execute=lambda *a, **k: before,
        wait_marker=lambda *a, **k: {
            "workflows": [
                {
                    "status": "BLOCKED",
                    "blocked_reasons": [f"SXID 10003 {destructive.FAIL_CLOSED_REASON}"],
                    "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
                }
            ]
        },
    )
    regional = SimpleNamespace(
        cpu_python=lambda *a, **k: {"present": False, "legacy_observed_at": []}
    )
    monkeypatch.setattr(
        destructive, "post_fabric_event", lambda *a: calls.append("negative")
    )
    monkeypatch.setattr(
        destructive,
        "wait_full_reset_stability",
        lambda *a: calls.append("stability") or [{"host": before}],
    )

    def variant(
        *args: Any, sxid: int, classification: str, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append((sxid, classification))
        return {
            "verdict": "FAIL" if sxid == failed_code else "PASS",
            "errors": ["physical proof failed"] if sxid == failed_code else [],
            "marker": f"marker-{sxid}",
            "state": {"sxid": sxid},
            "restore_workflows": [],
        }

    monkeypatch.setattr(destructive, "run_full_reset_variant", variant)
    result = destructive.run_collect014(
        SimpleNamespace(
            node="node-a",
            case_id="GF-REGIONAL-COLLECT-014",
            regional=SimpleNamespace(cluster_id="cluster-a"),
        ),
        regional,
        host,
        collector,
        tmp_path,
        1,
        "profile-a",
        cleanup=Mock(restore=Mock(return_value=[])),
    )
    assert result["required_positive_sxids"] == [10003, 19084]
    assert result["verdict"] == ("PASS" if failed_code is None else "FAIL")
    assert calls[:2] == ["negative", (10003, "Fatal")]
    if failed_code == 10003:
        assert len(calls) == 2
    else:
        assert calls[2:] == ["stability", (19084, "Non-fatal")]


def test_sxid_plan_binds_all_live_obligations_without_equating_component_proof() -> (
    None
):
    preflight = {
        "predecessor": {},
        "release_id": "release-a",
        "nodes": [{"uid": "uid-a"}, {"uid": "uid-b"}],
        "node": {"uid": "uid-a"},
        "store": {},
        "focused_tests": {},
    }
    first = acceptance.plan_details(
        SimpleNamespace(case_id="GF-REGIONAL-COLLECT-011", nodes=("node-a", "node-b")),
        preflight,
    )["sxid_proof_obligations"]
    assert {row["sxid"] for row in first["live_log_variants"]} == {11001, 12001, 24007}
    assert len(first["live_log_variants"]) == 6
    second = destructive.plan_details(
        SimpleNamespace(case_id="GF-REGIONAL-COLLECT-014", node="node-a", xid=109),
        preflight,
    )["sxid_proof_obligations"]
    assert second["live_log_positive_variants"] == [
        {"sxid": 10003, "classification": "Fatal"},
        {"sxid": 19084, "classification": "Non-fatal"},
    ]
    assert second["physical_reset_cycles"] == 2
    assert first["component_tests_are_physical_evidence"] is False
    assert second["component_tests_are_physical_evidence"] is False
