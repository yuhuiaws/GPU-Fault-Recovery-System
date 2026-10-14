"""Contract tests for GF-REGIONAL-DESTR-020.

Every verdict is judged against synthetic evidence documents -- the store
snapshot of the alias workflow, its remote commands, the escalation chain and
the fleet's scheduling state -- once on the intended run and once per way the
run can be wrong. Nothing here touches a cluster; the runner's live phases only
call these functions.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr020_verdicts as verdicts
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as destr020
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata
from tests.regional._site_topology import site_topology_leaks

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "destr020-abc123-a1"
ALIAS = verdicts.alias_for_run(RUN_ID)
REFERENCE = "node-a"
INCIDENT = "inc-alias"
WORKFLOW = "workflow-alias"
SUPPORT_WORKFLOW = f"workflow-support-after-{WORKFLOW}"
SUPPORT_INCIDENT = f"inc-support-after-{WORKFLOW}"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_destr001() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr020.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-non-destructive",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-020",
        predecessor=destr020.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "DESTR020_", prefix
    assert destr020.CONFIRMATION.startswith(prefix), destr020.CONFIRMATION
    assert destr020.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-001", (
        "the real RESET_GPU path must already be proven so an isolation failure "
        "is attributable to identity"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-DESTR-020", verdicts.CASE_ID


def test_the_alias_is_deterministic_per_run_and_never_a_real_name() -> None:
    again = verdicts.alias_for_run(RUN_ID)
    other = verdicts.alias_for_run("destr020-abc123-a2")
    assert again == ALIAS, "the alias must be stable across --plan and --execute"
    assert other != ALIAS, "a second attempt must inject a fresh alias"
    assert ALIAS.startswith(verdicts.ALIAS_PREFIX), ALIAS
    assert ALIAS == ALIAS.lower() and " " not in ALIAS, "alias must be a DNS label"
    assert (
        verdicts.alias_errors(
            ALIAS,
            gpu_node_names=[REFERENCE, "node-b"],
            kubernetes_node_present=False,
            agent=None,
            agents=[{"node_id": REFERENCE}],
        )
        == []
    ), "an alias that resolves to nothing is what the case needs"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"gpu_node_names": [ALIAS]}, "is a GPU node"),
        ({"kubernetes_node_present": True}, "Kubernetes API knows"),
        ({"agent": {"lifecycle_state": "ACTIVE"}}, "Node Agent is registered"),
        ({"agents": [{"node_id": ALIAS}]}, "fleet registry lists"),
    ],
)
def test_an_alias_that_resolves_to_anything_is_refused(
    overrides: dict[str, Any], fragment: str
) -> None:
    arguments: dict[str, Any] = {
        "gpu_node_names": [REFERENCE],
        "kubernetes_node_present": False,
        "agent": None,
        "agents": [],
    }
    arguments.update(overrides)
    errors = verdicts.alias_errors(ALIAS, **arguments)
    assert any(fragment in item for item in errors), _text(errors)


def test_a_bare_alias_without_the_prefix_is_refused() -> None:
    errors = verdicts.alias_errors(
        "node-c",
        gpu_node_names=[],
        kubernetes_node_present=False,
        agent=None,
        agents=[],
    )
    assert any("prefix" in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _preflight(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "alias": ALIAS,
        "alias_facts": {
            "gpu_node_names": [REFERENCE],
            "kubernetes_node_present": False,
            "agent": None,
            "agents": [{"node_id": REFERENCE}],
        },
        "reference_node": {"ready": "True", "ownership_annotations": {}},
        "reference_agent": {
            "lifecycle_state": "ACTIVE",
            "runtime_profile_version": "regional-hyperpod-abc123",
        },
        "reference_profile": {"warnings": []},
        "reference_workloads": [],
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "alias_event": None,
        "tests_passed": True,
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def test_the_intended_preflight_passes() -> None:
    assert _preflight() == [], _text(_preflight())


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"reference_node": {"ready": "False", "ownership_annotations": {}}}, "Ready"),
        (
            {
                "reference_node": {
                    "ready": "True",
                    "ownership_annotations": {"gpu-fault.io/incident-id": "x"},
                }
            },
            "ownership",
        ),
        ({"reference_workloads": [{"name": "job"}]}, "non-system"),
        ({"reference_agent": {"lifecycle_state": "REVOKED"}}, "ACTIVE"),
        ({"queue": {"depth": 2}}, "queue is not empty"),
        ({"remote_commands": {"open_by_cluster": {"c": 1}}}, "remote command"),
        ({"alias_event": {"event_id": "old"}}, "earlier event"),
        ({"tests_passed": False}, "regression"),
    ],
)
def test_preflight_refuses_each_unsafe_precondition(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _preflight(**overrides)
    assert any(fragment in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Injection and policy
# --------------------------------------------------------------------------- #
def test_injection_requires_an_accepted_receipt() -> None:
    good = {"accepted": {"processor_request_id": "req-1"}, "receipt": {"status": 200}}
    assert verdicts.injection_errors(good) == [], "a 200 receipt is the contract"
    errors = verdicts.injection_errors({"accepted": {}, "receipt": {"status": 202}})
    assert len(errors) == 2, _text(errors)


def _decision(**overrides: Any) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "disposition": verdicts.EXPECTED_DISPOSITION,
        "safety_action": verdicts.EXPECTED_SAFETY_ACTION,
        "action": None,
    }
    decision.update(overrides)
    return decision


def test_the_unknown_xid_resolves_to_the_site_safety_quarantine() -> None:
    errors = verdicts.decision_errors(_decision(), {"xid": verdicts.INJECTED_XID})
    assert errors == [], _text(errors)


@pytest.mark.parametrize(
    ("decision", "event", "fragment"),
    [
        (_decision(disposition="EXECUTABLE"), {"xid": 999}, "disposition"),
        (_decision(safety_action=None), {"xid": 999}, "safety_action"),
        (_decision(action="RESET_GPU"), {"xid": 999}, "executable action"),
        (None, {"xid": 999}, "no policy decision"),
        (_decision(), {"xid": 46}, "not XID 999"),
    ],
)
def test_a_decision_that_could_act_on_the_alias_is_refused(
    decision: dict[str, Any] | None, event: dict[str, Any], fragment: str
) -> None:
    errors = verdicts.decision_errors(decision, event)
    assert any(fragment in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Workflow and remote command
# --------------------------------------------------------------------------- #
def _state() -> dict[str, Any]:
    return {
        "workflow": {
            "request_id": WORKFLOW,
            "status": "FAILED",
            "safety_only": True,
            "safety_steps": [
                {"operation": name, "step_index": index}
                for index, name in enumerate(verdicts.SAFETY_STEPS)
            ],
            "official_steps": [{"operation": "FREEZE_EVIDENCE", "step_index": 0}],
            "completed_operations": ["FREEZE_EVIDENCE"],
            "step_executions": [
                {
                    "step_index": 0,
                    "operation": "FREEZE_EVIDENCE",
                    "status": "SUCCEEDED",
                },
                {
                    "step_index": 1,
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "FAILED",
                    "error": f"node {ALIAS} is absent; isolation cannot be observed",
                },
            ],
        },
        "incident": {
            "incident_id": INCIDENT,
            "node_ids": [ALIAS],
            "state": "ESCALATED",
        },
    }


def test_the_intended_workflow_fails_closed_at_the_isolation() -> None:
    errors = verdicts.workflow_errors(_state(), alias=ALIAS)
    assert errors == [], _text(errors)


def test_the_old_memory_based_isolation_is_a_failure() -> None:
    state = _state()
    state["workflow"]["step_executions"][1]["status"] = "SUCCEEDED"
    state["workflow"]["completed_operations"].append("MARK_UNSCHEDULABLE")
    state["workflow"]["status"] = "SUCCEEDED"
    errors = verdicts.workflow_errors(state, alias=ALIAS)
    assert any("remembered, not observed" in item for item in errors), _text(errors)
    assert any("completed_operations" in item for item in errors), _text(errors)
    assert any("not FAILED" in item for item in errors), _text(errors)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda s: s["workflow"]["step_executions"].append(
                {"step_index": 2, "operation": "QUARANTINE", "status": "SUCCEEDED"}
            ),
            "QUARANTINE ran",
        ),
        (
            lambda s: s["workflow"]["step_executions"].append(
                {"step_index": 3, "operation": "RESET_GPU", "status": "WAITING"}
            ),
            "physical operations",
        ),
        (lambda s: s["workflow"]["step_executions"].pop(1), "never executed"),
        (lambda s: s["incident"].__setitem__("node_ids", [ALIAS, REFERENCE]), "covers"),
        (lambda s: s["incident"].__setitem__("state", "QUARANTINED"), "ESCALATED"),
        (lambda s: s["workflow"]["safety_steps"].pop(), "safety steps"),
        (lambda s: s.__setitem__("workflow", {}), "no workflow"),
    ],
)
def test_each_way_the_alias_workflow_can_go_wrong_is_named(
    mutate: Any, fragment: str
) -> None:
    state = _state()
    mutate(state)
    errors = verdicts.workflow_errors(state, alias=ALIAS)
    assert any(fragment in item for item in errors), _text(errors)


def _command(**overrides: Any) -> dict[str, Any]:
    command: dict[str, Any] = {
        "command_id": f"{WORKFLOW}/1/MARK_UNSCHEDULABLE",
        "operation": "MARK_UNSCHEDULABLE",
        "status": "FAILED",
        "error": f"node {ALIAS} is absent; isolation cannot be observed",
        "result_details": {"safety_rejection": True, "absent": True, "node_id": ALIAS},
    }
    command.update(overrides)
    return command


def test_the_isolation_command_carries_the_safety_rejection() -> None:
    errors = verdicts.remote_command_errors([_command()], alias=ALIAS)
    assert errors == [], _text(errors)


@pytest.mark.parametrize(
    ("commands", "fragment"),
    [
        ([_command(status="SUCCEEDED")], "not FAILED"),
        (
            [_command(result_details={"absent": True, "node_id": ALIAS})],
            "safety_rejection",
        ),
        (
            [_command(result_details={"safety_rejection": True, "node_id": ALIAS})],
            "absent",
        ),
        (
            [_command(result_details={"safety_rejection": True, "absent": True})],
            "names None",
        ),
        ([_command(error="executor-internal-error")], "does not say"),
        ([_command(), _command(command_id="dup")], "expected one"),
        (
            [_command(), _command(operation="QUIESCE_GPU_SERVICES", status="WAITING")],
            "other operations",
        ),
        ([], "expected one"),
    ],
)
def test_each_way_the_isolation_command_can_be_wrong_is_named(
    commands: list[dict[str, Any]], fragment: str
) -> None:
    errors = verdicts.remote_command_errors(commands, alias=ALIAS)
    assert any(fragment in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #
def _chain(**overrides: Any) -> dict[str, Any]:
    chain: dict[str, Any] = {
        "incident": {
            "incident_id": SUPPORT_INCIDENT,
            "effective_action": verdicts.SUPPORT_ESCALATION_ACTION,
            "reasons": [
                f"{verdicts.SUPPORT_ESCALATION_STAGE} remediation failed; "
                "failed operation or validation: MARK_UNSCHEDULABLE"
            ],
            "node_ids": [ALIAS],
        },
        "workflow": {
            "request_id": SUPPORT_WORKFLOW,
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": name} for name in verdicts.SUPPORT_ESCALATION_OPERATIONS
            ],
            "step_executions": [
                {"operation": "FREEZE_EVIDENCE", "status": "SUCCEEDED"},
                {"operation": "ESCALATE_SUPPORT", "status": "SUCCEEDED"},
            ],
        },
        "second_order_incident": None,
        "second_order_workflow": None,
    }
    chain.update(overrides)
    return chain


def test_the_support_escalation_constants_follow_the_bounded_chain() -> None:
    assert verdicts.SUPPORT_ESCALATION_STAGE == "containment_refused", (
        "a safety-rejected isolation is classified containment_refused, "
        "not containment_or_release"
    )
    assert verdicts.SUPPORT_ESCALATION_OPERATIONS == (
        "FREEZE_EVIDENCE",
        "ESCALATE_SUPPORT",
    ), "a refused containment must not plan the isolation again"
    assert verdicts.MAX_ESCALATION_CHAIN_DEPTH == 1, "support terminates the chain"


def test_one_succeeded_support_handoff_without_isolation_is_the_contract() -> None:
    errors = verdicts.escalation_errors(_chain(), alias=ALIAS)
    assert errors == [], _text(errors)
    assert verdicts.escalation_chain_depth(_chain()) == 1, "one support pair"


def test_a_missing_chain_has_depth_zero() -> None:
    assert verdicts.escalation_chain_depth({}) == 0, "no escalation at all"
    assert (
        verdicts.escalation_chain_depth(
            _chain(second_order_incident={"incident_id": "x"})
        )
        == 2
    ), "a second-order record deepens the chain"


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "second_order_workflow": {
                "request_id": f"workflow-support-after-{SUPPORT_WORKFLOW}"
            }
        },
        {
            "second_order_incident": {
                "incident_id": f"inc-support-after-{SUPPORT_WORKFLOW}"
            }
        },
    ],
)
def test_a_second_order_escalation_is_the_headline_failure(
    overrides: dict[str, Any],
) -> None:
    chain = _chain(**overrides)
    errors = verdicts.escalation_errors(chain, alias=ALIAS)
    assert any("second-order" in item and "depth 2" in item for item in errors), _text(
        errors
    )
    assert verdicts.escalation_chain_depth(chain) == 2, "depth counts the pair"


def test_the_old_replanned_isolation_is_a_failure() -> None:
    chain = _chain()
    chain["incident"]["reasons"] = [
        "containment_or_release remediation failed; "
        "failed operation or validation: MARK_UNSCHEDULABLE"
    ]
    chain["workflow"]["official_steps"] = [
        {"operation": name}
        for name in (
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "QUARANTINE",
            "ESCALATE_SUPPORT",
        )
    ]
    chain["workflow"]["step_executions"] = [
        {"operation": "FREEZE_EVIDENCE", "status": "SUCCEEDED"},
        {"operation": "MARK_UNSCHEDULABLE", "status": "FAILED"},
    ]
    chain["workflow"]["status"] = "FAILED"
    errors = verdicts.escalation_errors(chain, alias=ALIAS)
    assert any("containment_refused" in item for item in errors), _text(errors)
    assert any("steps are" in item for item in errors), _text(errors)
    assert any("planned an isolation" in item for item in errors), _text(errors)
    assert any("not SUCCEEDED" in item for item in errors), _text(errors)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda c: c.__setitem__("incident", None), "no support escalation"),
        (lambda c: c.__setitem__("workflow", None), "no support escalation"),
        (
            lambda c: c["incident"].__setitem__("effective_action", "REBOOT_NODE"),
            "action is not",
        ),
        (
            lambda c: c["incident"].__setitem__(
                "reasons", ["reset remediation failed"]
            ),
            "containment_refused",
        ),
        (lambda c: c["incident"].__setitem__("reasons", []), "containment_refused"),
        (lambda c: c["incident"].__setitem__("node_ids", [REFERENCE]), "covers"),
        (
            lambda c: c["workflow"]["official_steps"].append(
                {"operation": "RESTART_NODE"}
            ),
            "steps are",
        ),
        (
            lambda c: c["workflow"]["official_steps"].insert(
                1, {"operation": "MARK_UNSCHEDULABLE"}
            ),
            "steps are",
        ),
        (
            lambda c: c["workflow"]["step_executions"].insert(
                1, {"operation": "MARK_UNSCHEDULABLE", "status": "FAILED"}
            ),
            "planned an isolation",
        ),
        (
            lambda c: c["workflow"]["step_executions"].insert(
                1, {"operation": "QUARANTINE", "status": "SUCCEEDED"}
            ),
            "planned an isolation",
        ),
        (
            lambda c: c["workflow"]["step_executions"].append(
                {"operation": "RESTART_NODE", "status": "WAITING"}
            ),
            "physical operation",
        ),
        (lambda c: c["workflow"].__setitem__("status", "FAILED"), "not SUCCEEDED"),
        (lambda c: c["workflow"].__setitem__("status", "RUNNING"), "not SUCCEEDED"),
        (lambda c: c["workflow"].__setitem__("status", "BLOCKED"), "not SUCCEEDED"),
    ],
)
def test_each_way_the_escalation_can_be_wrong_is_named(
    mutate: Any, fragment: str
) -> None:
    chain = _chain()
    mutate(chain)
    errors = verdicts.escalation_errors(chain, alias=ALIAS)
    assert any(fragment in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Fleet and leftovers
# --------------------------------------------------------------------------- #
def _fleet() -> dict[str, dict[str, Any]]:
    return {
        REFERENCE: {"unschedulable": False, "taints": [], "ownership_annotations": {}},
        "node-b": {
            "unschedulable": True,
            "taints": [{"key": "gpu-fault.io/quarantined"}],
            "ownership_annotations": {"gpu-fault.io/incident-id": "inc-old"},
        },
    }


def test_an_unchanged_fleet_passes_even_with_a_pre_existing_quarantine() -> None:
    errors = verdicts.fleet_errors(_fleet(), copy.deepcopy(_fleet()), alias=ALIAS)
    assert errors == [], _text(errors)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda f: f[REFERENCE].__setitem__("unschedulable", True), "unschedulable"),
        (
            lambda f: f[REFERENCE].__setitem__("taints", [{"key": "gpu-fault.io/q"}]),
            "taints",
        ),
        (
            lambda f: f[REFERENCE].__setitem__(
                "ownership_annotations", {"gpu-fault.io/incident-id": INCIDENT}
            ),
            "ownership_annotations",
        ),
        (lambda f: f.__setitem__(ALIAS, _fleet()[REFERENCE]), "alias"),
        (lambda f: f.pop("node-b"), "node set changed"),
    ],
)
def test_any_real_node_that_moved_fails_the_case(mutate: Any, fragment: str) -> None:
    after = copy.deepcopy(_fleet())
    mutate(after)
    errors = verdicts.fleet_errors(_fleet(), after, alias=ALIAS)
    assert any(fragment in item for item in errors), _text(errors)


def test_leftover_records_name_both_incidents_and_workflows() -> None:
    records = verdicts.leftover_records(_state(), _chain())
    assert records == {
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
        "support_incident_id": SUPPORT_INCIDENT,
        "support_workflow_request_id": SUPPORT_WORKFLOW,
    }, records


# --------------------------------------------------------------------------- #
# Runner guard
# --------------------------------------------------------------------------- #
def test_the_xid_payload_targets_the_alias_only() -> None:
    settings = argparse.Namespace()
    regional = argparse.Namespace(cluster_id="cluster-a")
    settings.regional = regional
    payload = destr020.xid_payload(
        settings,  # type: ignore[arg-type]
        alias=ALIAS,
        marker="m-1",
        product="H200",
        runtime_profile_version="regional-hyperpod-abc123",
        observed_at=verdicts.parse_time("2026-01-01T00:00:00Z"),  # type: ignore[arg-type]
    )
    assert payload["node_id"] == ALIAS, payload
    assert payload["workload_state"] == "IDLE", payload
    assert payload["affected_workload_ids"] == [], payload
    assert f": {verdicts.INJECTED_XID}," in payload["message"], payload["message"]
    assert "m-1" in payload["message"] and payload["record_id"] == "m-1", payload
    assert payload["evidence_ref"].startswith(f"api-replay://{destr020.CASE_ID}/"), (
        payload["evidence_ref"]
    )


def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / destr020.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"alias": "other"}}}),
        encoding="utf-8",
    )
    preflight = {
        "release_id": "rel-1",
        "alias": ALIAS,
        "reference_node": {"name": REFERENCE, "uid": "uid-1"},
        "store": {"agent": {"runtime_profile_version": "v1"}},
        "alias_facts": {"gpu_node_names": [REFERENCE]},
    }
    with pytest.raises(Exception, match="plan drifted"):
        destr020.verify_plan_identity(case_dir, preflight)


def test_the_run_identity_pins_the_alias_to_run_dir_and_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "acceptance-2026"
    first = destr020.run_identity(run_dir, 1)
    assert first.startswith("destr020-") and first.endswith("-a1"), first
    assert verdicts.alias_for_run(first) != verdicts.alias_for_run(
        destr020.run_identity(run_dir, 2)
    ), "attempt 2 must not reuse attempt 1's alias"


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr020.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    assert plan.chain_watch_seconds == verdicts.CHAIN_WATCH_SECONDS, (
        plan.chain_watch_seconds
    )
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr020.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--reference-node",
            REFERENCE,
        ]
    )
    assert execute.execute is True and execute.confirm == destr020.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def test_configure_refuses_an_unbounded_chain_watch(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    arguments = destr020.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "ctx",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--reference-node",
            REFERENCE,
            "--chain-watch-seconds",
            "5",
        ]
    )
    with pytest.raises(Exception, match="chain watch seconds"):
        destr020.configure(arguments)


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr020.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_the_runner_is_executable_with_a_shebang_and_no_site_topology() -> None:
    path = ROOT / "scripts/e2e/regional/run_destr020_identity_mismatch_isolation.py"
    mode = path.stat().st_mode & 0o777
    assert mode & 0o111 == 0o111, f"{path.name} is {oct(mode)}, not executable"
    source = path.read_text(encoding="utf-8")
    assert source.splitlines()[0] == "#!/usr/bin/env python3", source.splitlines()[0]
    assert not site_topology_leaks(source), site_topology_leaks(source)


# --------------------------------------------------------------------------- #
# Cleanup: the alias's Agent record is the case's own residue
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("result", "errors"),
    [
        pytest.param({"found": False}, [], id="never-registered"),
        pytest.param({"found": True, "lifecycle_state": "REVOKED"}, [], id="revoked"),
        pytest.param(
            {"found": True, "lifecycle_state": "ACTIVE"},
            ["the alias Agent record is 'ACTIVE' after the cleanup, not REVOKED"],
            id="still-active",
        ),
    ],
)
def test_the_alias_must_end_without_a_live_agent(
    result: dict[str, Any], errors: list[str]
) -> None:
    assert verdicts.alias_agent_cleanup_errors(result) == errors


def _alias_agent(lifecycle: str) -> Any:
    from datetime import datetime, timedelta, timezone

    from gpu_fault.fleet import AgentLifecycleState, AgentRecord

    seen = datetime.now(timezone.utc) - timedelta(hours=1)
    return AgentRecord(
        cluster_id="cluster-a",
        node_id=ALIAS,
        endpoint="https://10.0.0.7:9099",
        agent_version="2026.10.14",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="regional-v1",
        config_digest="b" * 64,
        allowed_operations=[],
        node_instance_id=None,
        agent_incarnation_id=f"inc-{ALIAS}",
        first_seen_at=seen,
        last_seen_at=seen,
        lease_expires_at=seen + timedelta(minutes=2),
        generation=3,
        lifecycle_state=AgentLifecycleState(lifecycle),
        transition_id="earlier/drain" if lifecycle == "DRAINING" else None,
    )


def _cleanup_in_pod(
    monkeypatch: pytest.MonkeyPatch, store: Any, registry: Any
) -> dict[str, Any]:
    import io
    import sys

    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store, fleet_registry=registry)
    output = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            ApplicationContext, "from_environment", classmethod(lambda cls: context)
        )
        scoped.setattr(
            sys,
            "argv",
            [
                "<pod>",
                "cluster-a",
                ALIAS,
                verdicts.alias_cleanup_transition(RUN_ID),
                verdicts.alias_cleanup_reason(RUN_ID, ALIAS),
            ],
        )
        scoped.setattr(sys, "stdout", output)
        exec(compile(destr020.ALIAS_AGENT_CLEANUP, "<pod>", "exec"), {})
    return dict(json.loads(output.getvalue()))


@pytest.mark.parametrize("lifecycle", ["ACTIVE", "DRAINING", "REVOKED", None])
def test_the_cleanup_revokes_the_alias_agent_through_the_fleet_registry(
    monkeypatch: pytest.MonkeyPatch, lifecycle: str | None
) -> None:
    from gpu_fault.fleet import FleetRegistry
    from tests._builders import build_store

    store = build_store()
    registry = FleetRegistry(store, "s" * 32)
    if lifecycle is not None:
        store.save_agent(_alias_agent(lifecycle))
        if lifecycle == "REVOKED":
            store.save_agent(
                _alias_agent("REVOKED").model_copy(update={"transition_id": "x"})
            )

    result = _cleanup_in_pod(monkeypatch, store, registry)

    assert verdicts.alias_agent_cleanup_errors(result) == []
    assert result["found"] is (lifecycle is not None)
    assert result["revoked_by_case"] is (lifecycle in {"ACTIVE", "DRAINING"})
    if lifecycle in {"ACTIVE", "DRAINING"}:
        record = store.get_agent("cluster-a", ALIAS)
        assert record.lifecycle_state.value == "REVOKED"
        # A drain someone else started is finished under its own transition.
        expected = (
            verdicts.alias_cleanup_transition(RUN_ID)
            if lifecycle == "ACTIVE"
            else "earlier/drain"
        )
        assert record.transition_id == expected, "audited as the case's cleanup"
        if lifecycle == "ACTIVE":
            assert destr020.CASE_ID in str(record.transition_reason), (
                record.transition_reason
            )


def test_a_cleanup_that_cannot_revoke_the_alias_agent_fails_the_case(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    calls: list[tuple[str, ...]] = []

    def cpu_python(script: str, *arguments: str, **_options: Any) -> dict[str, Any]:
        calls.append(arguments)
        return {"found": True, "lifecycle_state": "ACTIVE", "revoked_by_case": False}

    run = SimpleNamespace(
        regional=SimpleNamespace(cpu_python=cpu_python),
        settings=SimpleNamespace(regional=SimpleNamespace(cluster_id="cluster-a")),
        alias=ALIAS,
        run_id=RUN_ID,
        case_dir=tmp_path,
    )

    with pytest.raises(destr020.RegionalFixtureError, match="not REVOKED"):
        destr020.retire_alias_agent(run)

    assert calls == [
        (
            "cluster-a",
            ALIAS,
            f"{destr020.CASE_ID}/{RUN_ID}/cleanup",
            verdicts.alias_cleanup_reason(RUN_ID, ALIAS),
        )
    ]
    evidence = json.loads((tmp_path / "alias-agent-cleanup.json").read_text())
    assert evidence["lifecycle_state"] == "ACTIVE", "the failed read-back is kept"
