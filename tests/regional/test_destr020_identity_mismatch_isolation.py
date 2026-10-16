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


def test_the_run_identity_differs_between_run_dirs_with_the_same_last_token(
    tmp_path: Path,
) -> None:
    # A later round's run directory ends in the same token as the earlier one;
    # the earlier round's alias event must not block the later preflight.
    earlier = tmp_path / "regional-acceptance-round-a-eks"
    later = tmp_path / "regional-acceptance-round-b-eks"
    assert destr020.run_identity(later, 1) == destr020.run_identity(later, 1)
    assert verdicts.alias_for_run(
        destr020.run_identity(earlier, 1)
    ) != verdicts.alias_for_run(destr020.run_identity(later, 1))


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


# --------------------------------------------------------------------------- #
# Cleanup: the alias's departure, through the administrator's own exit
# --------------------------------------------------------------------------- #
def test_the_departed_alias_exit_is_recorded_under_the_case_and_run(
    tmp_path: Path,
) -> None:
    from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN
    from gpu_fault.admin.workflow_reconcile_departed_agents import ACCEPT_REASON_LIMIT

    run_id = destr020.run_identity(
        tmp_path / "regional-acceptance-20261016T0000Z-r9-eks", 1
    )
    reference = verdicts.departed_alias_reference(run_id)
    reason = verdicts.departed_alias_reason(run_id, verdicts.alias_for_run(run_id))

    assert reference == f"{destr020.CASE_ID}/{run_id}/departed-alias", reference
    assert REFERENCE_PATTERN.fullmatch(reference), "the CLI must accept it"
    assert len(reason) <= ACCEPT_REASON_LIMIT, len(reason)
    assert destr020.CASE_ID in reason and run_id in reason, reason
    assert verdicts.alias_for_run(run_id) in reason, reason


def test_the_departed_budget_follows_the_products_guard_and_cadence() -> None:
    from gpu_fault.departed_node_cleanup import CLEANUP_INTERVAL_SECONDS
    from gpu_fault.departed_nodes import DEPARTED_AGENT_GUARD_AGE

    assert verdicts.DEPARTED_GUARD_SECONDS == (
        DEPARTED_AGENT_GUARD_AGE.total_seconds()
    ), "the exit retires an Agent only past the product's guard"
    assert verdicts.DEPARTED_CLEANUP_INTERVAL_SECONDS == CLEANUP_INTERVAL_SECONDS
    assert verdicts.DEPARTED_EXIT_BUDGET_SECONDS == (
        verdicts.DEPARTED_GUARD_SECONDS + verdicts.DEPARTED_MARGIN_SECONDS
    ), "bounded: the guard plus three cleanup cycles"
    assert verdicts.DEPARTED_READBACK_SECONDS >= 2 * CLEANUP_INTERVAL_SECONDS


def test_the_lock_refusal_the_case_retries_is_the_products_own(tmp_path: Path) -> None:
    from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock

    with site_operation_lock(tmp_path, wait=False):
        with pytest.raises(SiteOperationBusy) as busy:
            with site_operation_lock(tmp_path, wait=False):
                pass

    assert verdicts.ADMIN_LOCK_BUSY in str(busy.value), str(busy.value)


def _exit_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "cluster_id": "cluster-a",
        "node_id": ALIAS,
        "retire": False,
        "kubernetes_reports": {"open": True, "never_reported": True},
        "kubernetes_reports_read": {
            "chain_started_at": "2026-10-16T18:00:00+00:00",
            "opens_at": None,
        },
        "eligible": False,
        "reasons": [],
    }
    item.update(overrides)
    return item


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, None, id="revoked-or-no-record-nothing-to-wait-on"),
        pytest.param(
            {
                "retire": True,
                "last_seen_at": "2026-10-16T21:05:00Z",
                "lease_expires_at": "2026-10-16T21:07:00+00:00",
            },
            "2026-10-16T21:15:00+00:00",
            id="agent-guard",
        ),
        pytest.param(
            {
                "retire": True,
                "last_seen_at": "2026-10-16T21:05:00+00:00",
                "lease_expires_at": "2026-10-16T21:20:00+00:00",
            },
            "2026-10-16T21:20:00+00:00",
            id="agent-lease",
        ),
        pytest.param(
            {
                "kubernetes_reports": {"open": False, "never_reported": True},
                "kubernetes_reports_read": {
                    "chain_started_at": "2026-10-16T21:08:00+00:00"
                },
            },
            "2026-10-16T21:18:00+00:00",
            id="short-report-chain",
        ),
        pytest.param(
            {
                "kubernetes_reports": {"open": False, "never_reported": False},
                "kubernetes_reports_read": {
                    "chain_started_at": "2026-10-16T18:00:00+00:00",
                    "opens_at": "2026-10-17T21:00:00+00:00",
                },
            },
            "2026-10-17T21:00:00+00:00",
            id="listed-node-24-hours",
        ),
    ],
)
def test_the_wait_is_read_from_the_dry_runs_own_timestamps(
    overrides: dict[str, Any], expected: str | None
) -> None:
    from datetime import timedelta

    opens = verdicts.departed_exit_opens_at(
        _exit_item(**overrides), guard=timedelta(seconds=600)
    )

    assert opens == verdicts.parse_time(expected), opens


@pytest.mark.parametrize(
    ("readback", "fragment"),
    [
        pytest.param({"ledger": None}, None, id="gone"),
        pytest.param({"ledger": {"status": "CLEANED"}}, None, id="cleaned"),
        pytest.param(
            {
                "ledger": {
                    "status": "REFUSED",
                    "blocker_classes": ["provider_identity_unknown"],
                    "blockers": ["provider membership unknown: ..."],
                }
            },
            "still lists",
            id="refused",
        ),
        pytest.param(
            {"ledger": {"status": "PENDING", "blockers": []}},
            "no blocker class",
            id="pending",
        ),
    ],
)
def test_the_alias_must_end_without_a_departed_node_ledger_line(
    readback: dict[str, Any], fragment: str | None
) -> None:
    errors = verdicts.departed_ledger_errors(readback, alias=ALIAS)

    assert (errors == []) is (fragment is None), errors
    assert fragment is None or fragment in _text(errors), errors


@pytest.mark.parametrize(
    ("attempt", "fragment"),
    [
        pytest.param(
            {
                "returncode": 0,
                "result": {"accepted_identities": [f"cluster-a/{ALIAS}"]},
            },
            None,
            id="accepted",
        ),
        pytest.param(
            {"returncode": 2, "result": None, "refusal": "exit 2: refuses ineligible"},
            "refuses ineligible",
            id="refused",
        ),
        pytest.param(
            {
                "returncode": 1,
                "result": {
                    "failed_agents": [f"cluster-a/{ALIAS}"],
                    "failures": {f"cluster-a/{ALIAS}": "ValueError: plan again"},
                },
            },
            "plan again",
            id="failed-in-the-pod",
        ),
        pytest.param(
            {"returncode": 0, "result": {"accepted_identities": []}},
            "recorded no acceptance",
            id="nothing-accepted",
        ),
    ],
)
def test_an_apply_must_record_the_acceptance_of_the_alias(
    attempt: dict[str, Any], fragment: str | None
) -> None:
    errors = verdicts.departed_apply_errors(attempt, key=f"cluster-a/{ALIAS}")

    assert (errors == []) is (fragment is None), errors
    assert fragment is None or fragment in _text(errors), errors


def test_an_eligible_item_must_name_the_cases_own_cluster() -> None:
    assert verdicts.departed_plan_errors(_exit_item(), cluster_id="cluster-a") == []
    errors = verdicts.departed_plan_errors(_exit_item(), cluster_id="cluster-b")
    assert any("not the case's cluster-b" in item for item in errors), errors


def test_the_admin_cli_is_read_in_its_own_words() -> None:
    printed = 'gpu-fault-admin: note\n{\n  "items": [],\n  "mode": "plan"\n}\n'
    stderr = (
        "gpu-fault-admin: retire-departed-agents refuses ineligible Agents: x\n"
        "gpu-fault-admin: full output in /state/logs/mutating/admin.log\n"
    )

    assert verdicts.admin_document(printed) == {"items": [], "mode": "plan"}
    assert verdicts.admin_document("no document") is None
    assert verdicts.admin_document("{ truncated") is None
    assert verdicts.admin_refusal(2, "", stderr) == (
        "exit 2: retire-departed-agents refuses ineligible Agents: x"
    )
    assert verdicts.admin_refusal(1, "", "") == "exit 1: no output"


def test_configure_binds_the_administrator_state_directory(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    base = [
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
    ]
    state = tmp_path / "state"
    with_state = [*base, "--admin-state-dir", str(state)]

    with pytest.raises(Exception, match="--admin-state-dir is required"):
        destr020.configure(destr020.parser().parse_args(base))
    with pytest.raises(Exception, match="holds no site.yaml"):
        destr020.configure(destr020.parser().parse_args(with_state))
    state.mkdir()
    (state / "site.yaml").write_text("site: test\n", encoding="utf-8")
    settings = destr020.configure(destr020.parser().parse_args(with_state))

    assert settings.admin_state_dir == state.resolve(), settings


def _pod(
    monkeypatch: pytest.MonkeyPatch,
    store: Any,
    script: str,
    argv: list[str],
    *,
    registry: Any = None,
    stdin: str = "",
) -> dict[str, Any]:
    """Run a CPU Pod script on an in-memory control plane."""

    import io
    import sys

    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store, fleet_registry=registry)
    output = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            ApplicationContext, "from_environment", classmethod(lambda cls: context)
        )
        scoped.setattr(sys, "argv", ["<pod>", *argv])
        scoped.setattr(sys, "stdin", io.StringIO(stdin))
        scoped.setattr(sys, "stdout", output)
        exec(compile(script, "<pod>", "exec"), {})
    return dict(json.loads(output.getvalue()))


def _alias_residue(store: Any, *, revoked_agent: bool) -> Any:
    """What the alias XID leaves, as the live round-9 run left it: the alias's
    ESCALATED incident (FAILED isolation) and its support pair (SUCCEEDED), no
    Agent record -- or the one the case's cleanup revoked a minute ago, well
    inside the 10-minute guard, which binds only an Agent the exit retires."""

    from gpu_fault.fleet import AgentLifecycleState
    from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
    from tests._builders import fault_incident, workflow_request, workflow_step
    from tests.departed_nodes._support import (
        EKS_CLUSTER,
        EKS_PRESENT,
        agent,
        ago,
        registration,
        world,
    )

    plane = world(store)
    store.save_regional_cluster(registration(EKS_CLUSTER, provider="eks-managed"))
    if revoked_agent:
        store.save_agent(
            agent(
                EKS_CLUSTER,
                ALIAS,
                instance_id=None,
                last_seen=ago(minutes=1),
                lifecycle=AgentLifecycleState.REVOKED,
            )
        )
    for incident_id, request_id, status in (
        (INCIDENT, WORKFLOW, WorkflowStatus.FAILED),
        (SUPPORT_INCIDENT, SUPPORT_WORKFLOW, WorkflowStatus.SUCCEEDED),
    ):
        store.save_workflow(
            workflow_request(
                request_id,
                incident_id,
                status=status,
                official_steps=[
                    workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=[ALIAS])
                ],
                created_at=ago(minutes=5),
                updated_at=ago(minutes=3),
            )
        )
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{incident_id}",
                cluster_id=EKS_CLUSTER,
                node_ids=[ALIAS],
                state=IncidentState.ESCALATED,
                workflow_request_id=request_id,
            )
        )
    plane.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    return plane


OPERATOR = "arn:aws:sts::123456789012:assumed-role/Admin/destr020"


def _deploy_host_cli(
    monkeypatch: pytest.MonkeyPatch, plane: Any, state_dir: Path
) -> tuple[list[list[str]], Any]:
    """The site's ``gpu-fault-admin``, in process: the real CLI parser and the
    real ``workflow-reconcile --retire-departed-agents`` path, whose in-Pod
    script runs on the in-memory control plane and whose GPU node read sees
    one real node, never the alias."""

    import subprocess
    from types import SimpleNamespace

    from gpu_fault.admin import cli, operator_identity
    from gpu_fault.admin import workflow_reconcile as reconcile
    from gpu_fault.admin import workflow_reconcile_departed_agents as departed
    from gpu_fault.admin.bootstrap_common import BootstrapError
    from scripts.e2e.regional.admin_cli import admin_command
    from tests.departed_nodes._support import EKS_CLUSTER, EKS_PRESENT

    site = SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [
                {
                    "cluster_id": EKS_CLUSTER,
                    "context": "gpu-eks-context",
                    "provider": "eks-managed",
                    "region": "us-west-2",
                }
            ],
        },
        environment={"KUBECONFIG": str(state_dir / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )
    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: OPERATOR
    )
    monkeypatch.setattr(
        departed,
        "run_pod_script",
        lambda _site, payload: _pod(
            monkeypatch,
            plane.store,
            departed.AGENT_SCRIPT,
            [],
            registry=plane.registry,
            stdin=json.dumps(payload),
        ),
    )

    def gpu_nodes(command: list[str], **_options: Any) -> SimpleNamespace:
        assert command[0] == "kubectl", "an unknown identity is never put to AWS"
        node = {
            "metadata": {"name": EKS_PRESENT, "annotations": {}},
            "spec": {"unschedulable": False, "taints": []},
        }
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": [node]}), stderr=""
        )

    monkeypatch.setattr(reconcile, "run_command", gpu_nodes)
    commands: list[list[str]] = []

    def run(command: list[str], **_options: Any) -> subprocess.CompletedProcess[str]:
        prefix = admin_command(state_dir)
        assert command[: len(prefix)] == prefix, command
        commands.append(command[len(prefix) :])
        arguments = cli.parser().parse_args(command[len(prefix) :])
        try:
            result = departed.retire_departed_agents_command(
                arguments, site=site, state_dir=state_dir
            )
        except BootstrapError as exc:
            return subprocess.CompletedProcess(
                command, 2, "", f"gpu-fault-admin: {exc}\n"
            )
        code = 1 if result.get("failed_agents") else 0
        return subprocess.CompletedProcess(
            command, code, json.dumps(result, indent=2, sort_keys=True), ""
        )

    return commands, run


class _CleanupCycles:
    """The case's clock; each sleep is the departed-node cleanup's next cycle."""

    def __init__(self, plane: Any) -> None:
        self.plane = plane
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds
        self.plane.cleanup.run_once()


@pytest.mark.parametrize("revoked_agent", [False, True])
def test_the_case_takes_the_real_exit_and_the_cleanup_ends_the_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revoked_agent: bool
) -> None:
    """The live shape, offline: the real departed-node cleanup lists the alias,
    the case runs the real ``workflow-reconcile`` exit through the real CLI
    parser and in-Pod script, and the next real cycle ends the line CLEANED."""

    from datetime import datetime, timezone
    from types import SimpleNamespace

    from gpu_fault.departed_node_cleanup_actions import accept_identity_command
    from gpu_fault.departed_nodes import DepartedNodeCleanupStatus
    from gpu_fault.models import IncidentState
    from tests._builders import build_store
    from tests.departed_nodes._support import EKS_CLUSTER

    store = build_store()
    plane = _alias_residue(store, revoked_agent=revoked_agent)
    [line] = plane.cleanup.run_once()
    assert line.blocker_classes == ["provider_identity_unknown"], line.blockers
    assert line.admin_commands == [accept_identity_command(ALIAS)], line
    state_dir = tmp_path / "state"
    commands, deploy_host_cli = _deploy_host_cli(monkeypatch, plane, state_dir)
    monkeypatch.setattr(destr020, "time", _CleanupCycles(plane))
    run = SimpleNamespace(
        settings=SimpleNamespace(
            admin_state_dir=state_dir, regional=SimpleNamespace(cluster_id=EKS_CLUSTER)
        ),
        regional=SimpleNamespace(
            run=deploy_host_cli,
            cpu_python=lambda script, *argv: _pod(
                monkeypatch, store, script, list(argv)
            ),
        ),
        alias=ALIAS,
        run_id=RUN_ID,
        case_dir=tmp_path,
        injected_at=datetime.now(timezone.utc),
    )

    evidence = destr020.accept_alias_departure(
        run,  # type: ignore[arg-type]
        {
            "found": revoked_agent,
            "lifecycle_state": "REVOKED" if revoked_agent else None,
        },
    )

    reason = verdicts.departed_alias_reason(RUN_ID, ALIAS)
    reference = verdicts.departed_alias_reference(RUN_ID)
    assert evidence["outcome"] == "accepted", evidence.get("error")
    assert commands == [
        destr020.departed_exit_argv(state_dir, ALIAS, reason, reference=None),
        destr020.departed_exit_argv(state_dir, ALIAS, reason, reference=reference),
    ], commands
    [acceptance] = store.list_departed_node_identity_acceptances(EKS_CLUSTER)
    assert acceptance.change_reference == reference, acceptance
    assert acceptance.reason == reason and acceptance.actor == OPERATOR, acceptance
    [after] = store.list_departed_node_cleanups(EKS_CLUSTER)
    assert after.status is DepartedNodeCleanupStatus.CLEANED, after.blockers
    for incident_id in (INCIDENT, SUPPORT_INCIDENT):
        state = store.get_incident(incident_id).state
        assert state is IncidentState.RECOVERED, (incident_id, state)
    assert [item["status"] for item in evidence["readback"]] == ["REFUSED", "CLEANED"]
    assert evidence["ledger_before"]["ledger"]["status"] == "REFUSED", evidence
    assert evidence["ledger_after"]["acceptance"]["change_reference"] == reference
    [_dry_run, apply] = evidence["attempts"]
    assert apply["result"]["retired_agents"] == [], "a revoked record stays as it is"
    assert plane.provider.calls == [], "nothing is ever put to the provider"
    saved = json.loads((tmp_path / destr020.DEPARTED_EVIDENCE).read_text())
    assert saved["outcome"] == "accepted" and saved["reference"] == reference, saved


def test_the_ledger_read_back_reads_the_alias_line_and_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timezone

    from gpu_fault.departed_nodes import (
        DepartedNodeCleanupRecord,
        DepartedNodeCleanupStatus,
        DepartedNodeIdentityAcceptance,
    )
    from tests._builders import build_store

    store = build_store()
    now = datetime.now(timezone.utc)
    assert _pod(
        monkeypatch, store, destr020.ALIAS_DEPARTED_LEDGER, ["cluster-a", ALIAS]
    ) == {"acceptance": None, "ledger": None}
    for node in (ALIAS, "node-b"):
        store.save_departed_node_cleanup(
            DepartedNodeCleanupRecord(
                cluster_id="cluster-a",
                node_id=node,
                status=DepartedNodeCleanupStatus.REFUSED,
                blockers=["provider membership unknown"],
                blocker_classes=["provider_identity_unknown"],
                first_seen_at=now,
                updated_at=now,
            )
        )
    store.save_departed_node_identity_acceptance(
        DepartedNodeIdentityAcceptance(
            cluster_id="cluster-a",
            node_id=ALIAS,
            change_reference="CHG-1",
            actor="ops",
            reason="alias",
            accepted_at=now,
        )
    )

    result = _pod(
        monkeypatch, store, destr020.ALIAS_DEPARTED_LEDGER, ["cluster-a", ALIAS]
    )

    assert result["ledger"]["status"] == "REFUSED", result
    assert result["ledger"]["blocker_classes"] == ["provider_identity_unknown"]
    assert "evidence" not in result["ledger"], "only the verdict fields travel"
    assert result["acceptance"]["change_reference"] == "CHG-1", result


def test_an_agent_item_without_timestamps_gives_nothing_to_wait_on() -> None:
    from datetime import timedelta

    item = _exit_item(retire=True, kubernetes_reports_read={})

    assert verdicts.departed_exit_opens_at(item, guard=timedelta(minutes=10)) is None
    assert verdicts.departed_alias_item({"items": ["junk"]}, ALIAS) is None
    assert verdicts.departed_alias_item(None, ALIAS) is None
