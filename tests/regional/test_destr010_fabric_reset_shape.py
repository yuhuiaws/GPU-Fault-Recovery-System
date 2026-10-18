"""DESTR-010 on a Hopper NVSwitch node: the RESTART_FM chain is the full reset.

On a node whose NVLink fabric the control plane grades (batch 128) a solo XID 45
no longer restarts Fabric Manager alone -- a restart loses the fabric state --
but runs NVIDIA's H100-and-later procedure: cordon, quiesce (Fabric Manager
stopped once), one reset of every GPU and NVSwitch, restore (Fabric Manager
started once), both validations, scheduling back. The case keeps its intent
for both shapes: one Fabric Manager down/up, owned by the Node Agent, no
residue, an idempotent replay and no provider mutation. The preflight reads the
shape from the control plane's own grading and the plan pins it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr010_observe_only as observe
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from scripts.e2e.regional import xid45_opt_in
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

NODE = "node-a"
GENERATION = 3
GPUS = [f"GPU-{index}" for index in range(8)]
GRADED = {
    "available": True,
    "graded_gpus": GPUS,
    "states": dict.fromkeys(GPUS, 3.0),
    "product": "NVIDIA H200",
}
OWNERS = {
    "FREEZE_EVIDENCE": "gpu-fault-control-plane",
    "MARK_UNSCHEDULABLE": "gpu-fault-kubernetes-adapter",
    "RESTORE_SCHEDULING": "gpu-fault-kubernetes-adapter",
    "VALIDATE_GPU": "gpu-fault-validation-adapter",
    "VALIDATE_FABRIC": "gpu-fault-validation-adapter",
}
RESTORE_COMMAND_ID = f"wf-45/5/RESTORE_GPU_SERVICES/{NODE}/agent-{GENERATION}"


def _step(operation: str) -> dict[str, Any]:
    step: dict[str, Any] = {
        "operation": operation,
        "execution_owner": OWNERS.get(operation, "gpu-fault-node-agent"),
        "gpu_uuids": GPUS,
        "parameters": {},
    }
    if operation == "RESET_ALL_GPUS_NVSWITCHES":
        step["parameters"] = {
            "fabric_partition": f"cluster-a/{NODE}/local-nvswitch",
            "xid": 45,
        }
    return step


def _command(index: int, operation: str, **extra: Any) -> dict[str, Any]:
    return {
        "command_id": f"cmd-{index}",
        "status": "SUCCEEDED",
        "step_index": index,
        "idempotency_key": f"wf-45/{index}/{operation}",
        "step": {"operation": operation},
        "batched_steps": [],
        "lease_token_sha256": "a" * 64,
        "lease_token_length": 27,
        "result_details": {},
        **extra,
    }


def _fabric_state() -> dict[str, Any]:
    """A terminal, contract-shaped store read for the graded node's chain."""

    return {
        "release_id": "release-a",
        "agent": {"generation": GENERATION},
        "event": {"xid": 45, "evidence_ref": "kmsg://node-a/boot-1/7"},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "request_id": "wf-45",
            "status": "SUCCEEDED",
            "official_action": "RESTART_FM",
            "official_steps": [_step(item) for item in shapes.FABRIC_RESET_STEPS],
            "completed_operations": list(shapes.FABRIC_RESET_STEPS),
        },
        "commands": [
            _command(1, "MARK_UNSCHEDULABLE"),
            _command(
                2,
                "QUIESCE_GPU_SERVICES",
                batched_steps=[
                    {
                        "step_index": 3,
                        "step": {"operation": "VERIFY_NO_GPU_CLIENTS"},
                        "idempotency_key": "wf-45/3/VERIFY_NO_GPU_CLIENTS",
                    }
                ],
            ),
            _command(4, "RESET_ALL_GPUS_NVSWITCHES"),
            _command(5, "RESTORE_GPU_SERVICES"),
            _command(8, "RESTORE_SCHEDULING"),
        ],
        "notifications": [
            {
                "notification": {
                    "notification_id": "n-fabric",
                    "category": "GENERAL",
                    "subject": "[NOTICE][GPU/NVSwitch reset automatically] "
                    "cluster-a: node-a",
                },
                "result": {"status": "SENT", "provider_message_id": "m-1"},
            }
        ],
        "evidence": [{"record_id": "evidence-a"}],
    }


def _ledger_row(command_id: str, operation: str, attempt: int = 1) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "operation": operation,
        "attempt": attempt,
        "state": "SUCCEEDED",
        "completed_at": "t",
        "started_at": "t",
    }


CHAIN_LEDGER = [
    _ledger_row(f"wf-45/2/QUIESCE_GPU_SERVICES/{NODE}/agent-3", "QUIESCE_GPU_SERVICES"),
    _ledger_row(
        f"wf-45/3/VERIFY_NO_GPU_CLIENTS/{NODE}/attempt-1/agent-3",
        "VERIFY_NO_GPU_CLIENTS",
    ),
    _ledger_row(
        f"wf-45/4/RESET_ALL_GPUS_NVSWITCHES/{NODE}/agent-3", "RESET_ALL_GPUS_NVSWITCHES"
    ),
    _ledger_row(RESTORE_COMMAND_ID, "RESTORE_GPU_SERVICES"),
]


def _host(
    *, main_pid: str, invocation: str, ledger: list, started: int, stopped: int
) -> dict[str, Any]:
    return {
        "kmsg_writable": True,
        "compute_clients": [],
        "fabric_manager": {
            "ActiveState": "active",
            "MainPID": main_pid,
            "InvocationID": invocation,
        },
        "gpu_pci_bdf": "0000:01:00.0",
        "ledger": ledger,
        "gpu_fault_timers": [],
        "journal": {"started_count": started, "stopped_count": stopped},
    }


BASELINE = _host(main_pid="100", invocation="inv-1", ledger=[], started=0, stopped=0)
AFTER = _host(
    main_pid="200", invocation="inv-2", ledger=CHAIN_LEDGER, started=1, stopped=1
)


def _preflight(grading: Any = GRADED) -> dict[str, Any]:
    return {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "node": {
            "uid": "uid-a",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        },
        "store": {
            "agent": {"generation": GENERATION},
            "profile": {"profile_version": "profile-a"},
        },
        "fabric": {"companion_window_seconds": 60, "fabric_grading": grading},
        "focused_tests": {"passed": True, "focused_tests_reused": True},
    }


# --- the shape -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grading", "shape"),
    [
        (GRADED, shapes.FABRIC_RESET),
        ({**GRADED, "graded_gpus": []}, shapes.FABRIC_MANAGER_RESTART),
        ({"available": False, "graded_gpus": GPUS}, shapes.FABRIC_MANAGER_RESTART),
        (None, shapes.FABRIC_MANAGER_RESTART),
    ],
    ids=["graded", "not-graded", "control-plane-predates-grading", "probe-silent"],
)
def test_the_shape_is_the_control_plane_s_own_grading(grading, shape) -> None:
    preflight = _preflight(grading)

    assert destr010.run_shape(preflight)[0] == shape
    identity = destr010.preflight_identity(preflight)
    assert identity["workflow_shape"] == shape, "the plan pins the shape"
    details = destr010.plan_details(
        destr010.Settings(regional=None, node=NODE, host_probe_image="i"),  # type: ignore[arg-type]
        preflight,
    )
    assert details["workflow_shape"] == shape
    assert details["risk"] == (
        "destructive" if shape == shapes.FABRIC_RESET else "live-service-action"
    )


def test_the_probe_grades_with_the_installed_control_plane() -> None:
    assert "graded_fabric_gpus" in destr010.FABRIC_PROBE
    assert '"fabric_grading": fabric_grading(store, cluster_id, node_id)' in (
        destr010.FABRIC_PROBE
    )
    compile(destr010.FABRIC_PROBE, "fabric-probe", "exec")
    compile(shapes.FABRIC_GRADING_PROBE, "grading-probe", "exec")


def test_a_graded_node_needs_the_chain_s_node_actions_not_the_restart() -> None:
    state = {
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(shapes.FABRIC_RESET_NODE_STEPS),
        },
        "profile": {
            "capabilities": [
                {
                    "capability": name,
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                    "adapter": "node-action",
                }
                for name in ("gpuReset", "fabricReset")
            ]
        },
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
    }
    node = {"ready": "True", "unschedulable": False, "taints": []}
    fabric = {"fabric_grading": GRADED}

    assert destr010.validate_preflight(state, fabric, node, []) == []
    state["agent"]["allowed_operations"] = ["RESTART_FABRIC_MANAGER"]
    state["profile"]["capabilities"].pop()
    errors = destr010.validate_preflight(state, fabric, node, [])
    assert "target Node Agent does not allow RESET_ALL_GPUS_NVSWITCHES" in errors
    assert "runtime profile has no fabricReset capability" in errors
    assert not any("fabricManagerRestart" in item for item in errors), errors


# --- the workflow verdict --------------------------------------------------------


def test_the_full_reset_chain_passes() -> None:
    state = _fabric_state()

    assert destr010.workflow_errors(state, shapes.FABRIC_RESET, GPUS) == []
    assert (
        destr010.terminal_errors(
            state, {"fabric_grading": GRADED}, "fabric-reset", GPUS
        )
        == []
    )
    assert destr010.replay_target(state, shapes.FABRIC_RESET)["step_index"] == 5, (
        "the replay targets the RESTORE that started Fabric Manager"
    )


def test_the_two_step_restart_is_no_pass_on_a_graded_node() -> None:
    state = _fabric_state()
    state["workflow"]["official_steps"] = [
        _step("FREEZE_EVIDENCE"),
        _step("RESTART_FABRIC_MANAGER"),
    ]
    state["workflow"]["completed_operations"] = [
        "FREEZE_EVIDENCE",
        "RESTART_FABRIC_MANAGER",
    ]

    errors = destr010.workflow_errors(state, shapes.FABRIC_RESET, GPUS)

    assert any("not the full GPU + NVSwitch reset chain" in item for item in errors), (
        errors
    )
    assert any(
        "forbidden operations ['RESTART_FABRIC_MANAGER']" in item for item in errors
    ), errors


@pytest.mark.parametrize(
    "defect",
    [
        "partial-inventory",
        "second-reset-command",
        "fm-restart-notification",
        "failed-command",
        "owner",
    ],
)
def test_every_drift_from_the_chain_contract_fails(defect: str) -> None:
    state = _fabric_state()
    if defect == "partial-inventory":
        state["workflow"]["official_steps"][4]["gpu_uuids"] = GPUS[:7]
    elif defect == "second-reset-command":
        state["commands"].append(_command(4, "RESET_ALL_GPUS_NVSWITCHES"))
    elif defect == "fm-restart-notification":
        state["notifications"][0]["notification"]["subject"] = (
            "[NOTICE][GPU Fabric Manager restarted automatically] cluster-a"
        )
    elif defect == "failed-command":
        state["commands"][0]["status"] = "FAILED"
    else:
        state["workflow"]["official_steps"][4]["execution_owner"] = "someone-else"

    assert destr010.workflow_errors(state, shapes.FABRIC_RESET, GPUS), defect


def test_a_gpu_that_did_not_register_after_the_reset_fails() -> None:
    grading = {**GRADED, "states": {**GRADED["states"], "GPU-3": 2.0}}

    errors = destr010.terminal_errors(
        _fabric_state(), {"fabric_grading": grading}, shapes.FABRIC_RESET, GPUS
    )

    assert errors == [
        "GPUs whose NVLink fabric state is not Success after the reset: GPU-3"
    ]


# --- the host verdict ------------------------------------------------------------


def test_one_stop_one_start_and_one_reset_on_the_host_pass() -> None:
    errors = destr010.host_errors(
        BASELINE, AFTER, AFTER, RESTORE_COMMAND_ID, shape=shapes.FABRIC_RESET
    )

    assert errors == []


@pytest.mark.parametrize(
    "defect",
    [
        "no-stop",
        "two-resets",
        "fabric-manager-restart-row",
        "restore-missing",
        "replay",
    ],
)
def test_host_drift_from_one_window_fails(defect: str) -> None:
    after = json.loads(json.dumps(AFTER))
    replay = after
    if defect == "no-stop":
        after["journal"]["stopped_count"] = 0
    elif defect == "two-resets":
        after["ledger"].append(
            _ledger_row(CHAIN_LEDGER[2]["command_id"], "RESET_ALL_GPUS_NVSWITCHES", 2)
        )
    elif defect == "fabric-manager-restart-row":
        after["ledger"].append(
            _ledger_row("fm/node-a/agent-3", "RESTART_FABRIC_MANAGER")
        )
    elif defect == "restore-missing":
        after["ledger"] = CHAIN_LEDGER[:3]
    else:
        replay = {
            **after,
            "fabric_manager": {**after["fabric_manager"], "MainPID": "300"},
        }

    assert destr010.host_errors(
        BASELINE, after, replay, RESTORE_COMMAND_ID, shape=shapes.FABRIC_RESET
    ), defect


# --- execute_case end to end -----------------------------------------------------


def _settings(tmp_path: Path) -> destr010.Settings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    regional = RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context="gpu-context",
        namespace="gpu-fault-system",
        cluster_id="cluster-a",
        region="us-west-2",
    )
    return destr010.Settings(regional=regional, node=NODE, host_probe_image="i")


def test_execute_case_judges_the_graded_node_by_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replays: list[dict[str, Any]] = []

    class FakeHost:
        def __init__(self, _settings: Any) -> None:
            self.snapshots = 0

        def create(self) -> None:
            pass

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            if arguments[0] == "snapshot":
                self.snapshots += 1
                return json.loads(
                    json.dumps(BASELINE if len(arguments) == 1 else AFTER)
                )
            return {"command": arguments[0]}

        def cleanup(self) -> dict[str, bool]:
            return {}

    class FakeRegional(RegionalLiveFixture):
        def __init__(self, live_settings: RegionalLiveSettings) -> None:
            self.settings = live_settings

        def wait_for_workflow(self, **_kwargs: Any) -> dict[str, Any]:
            return _fabric_state()

        def cpu_python(self, script: str, *arguments: str, **_kwargs: Any) -> Any:
            return {
                "companion_window_seconds": 60,
                "evidence": [{"record_id": "evidence-a"}],
                "fabric_grading": GRADED,
            }

        def executor_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
            replays.append(json.loads(arguments[0]))
            return {"status": "SUCCEEDED"}

        def store_snapshot(self, **_kwargs: Any) -> dict[str, Any]:
            return _fabric_state()

        def node_snapshot(self, node: str) -> dict[str, Any]:
            return dict(_preflight()["node"])

        def business_workloads(self, node: str) -> list[dict[str, str]]:
            return []

        def provider_events(self, *_args: Any) -> list[dict[str, str]]:
            return []

        @staticmethod
        def provider_events_provisional(_ended_at: datetime) -> bool:
            return True

    preflight = _preflight()
    monkeypatch.setattr(destr010, "read_only_preflight", lambda *_a, **_k: preflight)
    monkeypatch.setattr(destr010, "RegionalLiveFixture", FakeRegional)
    monkeypatch.setattr(destr010, "HostProbeFixture", FakeHost)
    run_dir = tmp_path / "run-DESTR010"
    case_dir = run_dir / "cases" / destr010.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps(
            {"details": {"preflight_identity": destr010.preflight_identity(preflight)}}
        ),
        encoding="utf-8",
    )

    exit_code = destr010.execute_case(
        _settings(tmp_path), run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    evidence = json.loads(
        (case_dir / f"{destr010.CASE_ID}.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0, evidence
    assert evidence["verdict"] == "PASS", evidence
    assert evidence["workflow_shape"] == shapes.FABRIC_RESET
    assert evidence["node_action_command_id"] == RESTORE_COMMAND_ID
    assert [item["step"]["operation"] for item in replays] == ["RESTORE_GPU_SERVICES"]


# --- batch 131: the observe-only default, then the opt-in chain --------------------

OBSERVE_GRADED = {**GRADED, "observe_only_default": True}
SWITCH = xid45_opt_in.SWITCH


def _observe_preflight(opted_in: bool | None = False) -> dict[str, Any]:
    preflight = _preflight(OBSERVE_GRADED)
    preflight["xid45_switch"] = {"opted_in": opted_in, "replicas": []}
    return preflight


def _observe_state() -> dict[str, Any]:
    """A contract-shaped store read for the observe-only XID 45."""

    return {
        "event": {"xid": 45, "evidence_ref": "kmsg://node-a/boot-1/3"},
        "decision": {
            "official_action": "RESTART_FM",
            "disposition": "MONITOR_ONLY",
            "action": "NO_ACTION",
            "marker": {"active": False},
        },
        "incident": {
            "incident_id": "inc-observe",
            "state": "RECOVERED",
            "workflow_request_id": None,
            "policy_reference": shapes.XID45_EVIDENCE_REFERENCE,
            "reasons": [
                "NVIDIA WORKFLOW_XID_45 classifies this as solo and requires "
                "Fabric Manager restart",
                "observe-only: a solo XID 45 on a graded NVSwitch node (8 GPUs "
                "report an NVLink fabric state; ...)",
                f"to run the catalog's RESTART_FM on such nodes, set {SWITCH}=true",
            ],
        },
        "workflow": None,
        "commands": [],
        "remote_commands": {"open_by_cluster": {}},
        "notifications": [
            {
                "notification": {
                    "notification_id": "n-xid45",
                    "category": "FAULT_DETECTED",
                    "subject": "[GPU FAULT][XID 45 investigatory action] "
                    "cluster-a node-a",
                },
                "result": {"status": "SKIPPED"},
            }
        ],
    }


@pytest.mark.parametrize(
    ("grading", "opted_in", "phases"),
    [
        (OBSERVE_GRADED, False, [shapes.OBSERVE_ONLY, shapes.FABRIC_RESET]),
        (OBSERVE_GRADED, True, [shapes.FABRIC_RESET]),
        (GRADED, False, [shapes.FABRIC_RESET]),
        ({**OBSERVE_GRADED, "graded_gpus": []}, False, [shapes.FABRIC_MANAGER_RESTART]),
    ],
    ids=["default-observe-only", "site-opted-in", "pre-batch-131", "not-graded"],
)
def test_the_phases_follow_the_default_and_the_switch(
    grading, opted_in, phases
) -> None:
    preflight = _preflight(grading)
    preflight["xid45_switch"] = {"opted_in": opted_in, "replicas": []}

    assert destr010.run_phases(preflight) == phases
    assert destr010.run_shape(preflight)[0] == phases[-1], "the chain is proved last"
    assert destr010.opt_in_window(preflight) is (phases[0] == shapes.OBSERVE_ONLY)
    identity = destr010.preflight_identity(preflight)
    assert identity["workflow_phases"] == phases, "the plan pins the phases"
    details = destr010.plan_details(
        destr010.Settings(regional=None, node=NODE, host_probe_image="i"),  # type: ignore[arg-type]
        preflight,
    )
    assert details["xid45_opt_in_window"] is destr010.opt_in_window(preflight)
    if destr010.opt_in_window(preflight):
        assert SWITCH in details["mutation"], details["mutation"]
        assert details["rollback"]["runner_finally_closes_the_xid45_opt_in_window"]
        assert any("observe-only" in item for item in details["stop_conditions"]), (
            "a default that did not hold stops the run"
        )


def test_a_half_rolled_control_worker_refuses_the_plan() -> None:
    state = {
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(shapes.FABRIC_RESET_NODE_STEPS),
        },
        "profile": {
            "capabilities": [
                {
                    "capability": name,
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                    "adapter": "node-action",
                }
                for name in ("gpuReset", "fabricReset")
            ]
        },
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
    }
    node = {"ready": "True", "unschedulable": False, "taints": []}
    fabric = {"fabric_grading": OBSERVE_GRADED}

    assert (
        destr010.validate_preflight(state, fabric, node, [], xid45={"opted_in": False})
        == []
    ), "the default: the case opens the window itself"
    errors = destr010.validate_preflight(
        state, fabric, node, [], xid45={"opted_in": None}
    )
    assert any(SWITCH in item for item in errors), errors


def test_the_observe_only_default_passes_its_contract() -> None:
    state = _observe_state()

    assert shapes.observe_only_errors(state, GPUS) == []
    assert observe.notification_errors(state) == []
    assert shapes.observe_only_host_errors(BASELINE, BASELINE) == []


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("executable", "not MONITOR_ONLY"),
        ("workflow", "a workflow was planned"),
        ("command", "remote commands were created"),
        ("reference", "does not cite NVIDIA's GPU debug guidelines"),
        ("why", "does not say why"),
        ("opt-in", "does not say how to opt in"),
        ("official", "does not record the catalog's RESTART_FM"),
        ("active-marker", "marker is active"),
    ],
)
def test_every_drift_from_the_observe_only_contract_fails(defect, message) -> None:
    state = _observe_state()
    if defect == "executable":
        state["decision"]["disposition"] = "EXECUTABLE"
    elif defect == "workflow":
        state["workflow"] = {"request_id": "wf-45"}
    elif defect == "command":
        state["commands"] = [_command(1, "RESTART_FABRIC_MANAGER")]
    elif defect == "reference":
        state["incident"]["policy_reference"] = None
    elif defect == "why":
        state["incident"]["reasons"] = state["incident"]["reasons"][::2]
    elif defect == "opt-in":
        state["incident"]["reasons"] = state["incident"]["reasons"][:2]
    elif defect == "official":
        state["decision"]["official_action"] = "WORKFLOW_XID_45"
    else:
        state["decision"]["marker"]["active"] = True

    errors = shapes.observe_only_errors(state, GPUS)

    assert any(message in item for item in errors), (defect, errors)


@pytest.mark.parametrize(
    "change",
    [
        {"fabric_manager": {**BASELINE["fabric_manager"], "MainPID": "300"}},
        {"journal": {"started_count": 1, "stopped_count": 1}},
        {"ledger": CHAIN_LEDGER[:1]},
        {"gpu_fault_timers": ["gpu-fault-quiesce-x.timer"]},
    ],
    ids=["fabric-manager-restarted", "journal", "ledger", "timer"],
)
def test_a_node_the_observe_only_xid_touched_fails(change) -> None:
    assert shapes.observe_only_host_errors(BASELINE, {**BASELINE, **change}), change


def test_a_completion_notice_beside_the_investigatory_one_fails() -> None:
    state = _observe_state()
    state["notifications"].append(_fabric_state()["notifications"][0])

    assert observe.notification_errors(state), "a node action was reported"


class _ObserveRegional(RegionalLiveFixture):
    """The control plane of a graded node under the default, then opted in."""

    def __init__(self, live_settings: RegionalLiveSettings, events: list) -> None:
        self.settings = live_settings
        self.events = events

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        marker = str(kwargs.get("marker") or "")
        if marker.startswith("destr010-observe-"):
            self.events.append("observe.store")
            return _observe_state()
        return _fabric_state()

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append("chain.wait")
        return _fabric_state()

    def cpu_python(self, script: str, *arguments: str, **_kwargs: Any) -> Any:
        return {
            "companion_window_seconds": 0,
            "evidence": [{"record_id": "evidence-a"}],
            "fabric_grading": OBSERVE_GRADED,
            "recent_xid_events": [],
        }

    def executor_python(self, script: str, *arguments: str, **kwargs: Any) -> Any:
        return {"status": "SUCCEEDED"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return dict(_preflight()["node"])

    def business_workloads(self, node: str) -> list[dict[str, str]]:
        return []

    def provider_events(self, *_args: Any) -> list[dict[str, str]]:
        return []

    @staticmethod
    def provider_events_provisional(_ended_at: datetime) -> bool:
        return True


def _run_observe_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, observe_state: Any = None
) -> tuple[int, dict[str, Any], list[str]]:
    events: list[str] = []

    class FakeHost:
        """The node: untouched by the observe-only XID, reset by the chain's."""

        def __init__(self, _settings: Any) -> None:
            self.phase = "baseline"

        def create(self) -> None:
            pass

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            if arguments[0] == "write-xid45":
                self.phase = "observe" if "observe" in arguments[2] else "chain"
                events.append(f"inject:{self.phase}")
            if arguments[0] != "snapshot":
                return {"command": arguments[0]}
            after = len(arguments) > 1 and self.phase == "chain"
            return json.loads(json.dumps(AFTER if after else BASELINE))

        def cleanup(self) -> dict[str, bool]:
            return {}

    preflight = _observe_preflight()
    monkeypatch.setattr(destr010, "read_only_preflight", lambda *_a, **_k: preflight)
    monkeypatch.setattr(
        destr010, "RegionalLiveFixture", lambda live: _ObserveRegional(live, events)
    )
    monkeypatch.setattr(destr010, "HostProbeFixture", FakeHost)
    monkeypatch.setattr(observe, "SETTLE_SECONDS", 0)
    monkeypatch.setattr(
        observe,
        "wait_outside_companion_window",
        lambda injected_at, seconds: events.append(f"clear:{seconds}"),
    )
    if observe_state is not None:
        monkeypatch.setattr(
            _ObserveRegional,
            "store_snapshot",
            lambda self, **kwargs: (
                observe_state
                if str(kwargs.get("marker") or "").startswith("destr010-observe-")
                else _fabric_state()
            ),
        )

    def opened(_regional: Any, _case_dir: Path) -> dict[str, Any]:
        events.append("window.open")
        return {"state": "OPEN", "assignments": {SWITCH: xid45_opt_in.OPTED_IN}}

    def closed(_regional: Any, _case_dir: Path) -> dict[str, Any] | None:
        events.append("window.close")
        return {"state": "CLOSED"} if "window.open" in events else None

    monkeypatch.setattr(destr010.xid45_opt_in, "open_window", opened)
    monkeypatch.setattr(destr010.xid45_opt_in, "close_window", closed)
    run_dir = tmp_path / "run-DESTR010"
    case_dir = run_dir / "cases" / destr010.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps(
            {"details": {"preflight_identity": destr010.preflight_identity(preflight)}}
        ),
        encoding="utf-8",
    )
    code = destr010.execute_case(
        _settings(tmp_path), run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    evidence = json.loads(
        (case_dir / f"{destr010.CASE_ID}.json").read_text(encoding="utf-8")
    )
    return code, evidence, events


def test_execute_case_proves_the_default_then_the_opt_in_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, evidence, events = _run_observe_case(tmp_path, monkeypatch)

    assert code == 0 and evidence["verdict"] == "PASS", evidence
    assert evidence["workflow_phases"] == [shapes.OBSERVE_ONLY, shapes.FABRIC_RESET]
    assert evidence["observe_only"]["errors"] == [], evidence["observe_only"]
    assert evidence["observe_only"]["policy_reference"] == (
        shapes.XID45_EVIDENCE_REFERENCE
    )
    assert evidence["opt_in_window_closed"] == "CLOSED"
    assert events.index("window.open") < events.index("chain.wait"), (
        "the default is proved before the window opens and the chain runs in it"
    )
    assert events[-1] == "window.close", "the window closes in the finally"
    assert "clear:60" in events, "the chain waits out the first XID's window"
    assert evidence["node_action_command_id"] == RESTORE_COMMAND_ID


def test_a_default_that_did_not_hold_opens_no_window_and_runs_no_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = _observe_state()
    broken["decision"]["disposition"] = "EXECUTABLE"
    broken["workflow"] = {"request_id": "wf-45"}

    code, evidence, events = _run_observe_case(
        tmp_path, monkeypatch, observe_state=broken
    )

    assert code == 1 and evidence["verdict"] == "FAIL", evidence
    assert any("observe-only:" in item for item in evidence["errors"]), evidence
    assert "window.open" not in events and "chain.wait" not in events, events
    assert "opt_in_window_closed" not in evidence, "nothing was opened to close"
