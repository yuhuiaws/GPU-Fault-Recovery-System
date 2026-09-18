"""Stateful external boundaries for COLLECT016/017/021 orchestration."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from scripts.e2e.regional import collect017_training, collect021_passive
from scripts.e2e.regional import run_collect016_training_recovery as c016
from scripts.e2e.regional import run_collect017_efa_plugin as c017
from scripts.e2e.regional import run_collect021_late_xid_after_pod_death as c021
from tests.regional._collect021_passive_support import completed_passive_state
from tests.regional._cov95_collect_net import Clock


def training_attempt_observation(
    workload: Any, *, attempt_id: str, phase: WorkloadPhase, killed: bool = False
) -> dict[str, Any]:
    return AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        job_id=workload.settings.job_id,
        attempt_id=attempt_id,
        workload_phase=phase,
        observed_at=datetime.now(timezone.utc),
        runtime_profile_version="profile-a",
        expected_critical_ranks=3,
        containers=(
            []
            if phase is WorkloadPhase.STOPPED
            else [
                ContainerObservation(
                    pod_uid=row["uid"],
                    pod_name=f"worker-{index}",
                    container_name="trainer",
                    role="worker",
                    rank=index,
                    node_id=row["node"],
                    gpu_count=8,
                    terminated=killed and index == 0,
                    exit_code=1 if killed and index == 0 else None,
                )
                for index, row in enumerate(workload.source["pods"])
            ]
        ),
    ).model_dump(mode="json")


def training_cpu_read(
    harness: Any, note: Any, bundle: Any, script: str, *args: str
) -> dict[str, Any]:
    note("cpu-read", *args)
    if script == collect021_passive.PASSIVE_PROBE:
        workload = harness.workloads[-1]
        pods = [
            {**row, "uid": row["uid"] + "-new", "attempt_id": workload.new_attempt}
            for row in workload.source["pods"]
        ]
        state = completed_passive_state(
            cluster_id=args[0],
            job_id=args[1],
            attempt_id=args[2],
            new_attempt_id=workload.new_attempt,
            pods=pods,
        )
        if not args[3]:
            state["observations"] = []
        return state
    if len(args) == 1:
        return {
            "remote_commands": [
                {
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "result_details": {
                        "node_results": {
                            "node-a": {
                                "rebound_pci_bdfs": ["0000:af:00.0"],
                                "already_bound": False,
                                "driver_bound_count": 1
                                if harness.problem == "section-error"
                                else 2,
                            }
                        }
                    },
                }
            ]
        }
    harness.plan_reads += 1
    operation = harness.current_operation
    harness.planning_seen[operation] = harness.planning_seen.get(operation, 0) + 1
    if operation != "REMEDIATE_EFA_DRIVER":
        if harness.problem == "planning-timeout" or (
            harness.problem == "planning-delay"
            and harness.planning_seen[operation] == 1
        ):
            return {}
    harness.efa_unbound = False
    return {"matches": [bundle()]}


def bind_local_contract_transport(
    module: Any, harness: Any, monkeypatch: Any, note: Any
) -> None:
    from tests.test_acceptance_receipt_alignment import IDENTITY, complete_report
    from tools.pytest_result_identity import parse_pytest_receipt

    def reported_pytest(command: list[str], **kwargs: Any) -> Any:
        note("focused-tests", command)
        selectors = [item for item in command if item.startswith("tests/")]
        payload = complete_report(selectors)
        payload["session"]["exitstatus"] = harness.focused_status
        if harness.focused_status:
            payload["records"][selectors[0]]["status"] = "FAIL"
            payload["records"][selectors[0]]["phases"]["call"] = "failed"
        receipt = parse_pytest_receipt(
            payload,
            root=module.ROOT,
            expected_identity=IDENTITY,
            require_session=True,
            expected_exitstatus=harness.focused_status,
        )
        return (
            subprocess.CompletedProcess(
                command, harness.focused_status, "fixture output", ""
            ),
            receipt,
            None,
        )

    monkeypatch.setattr(module, "run_reported_pytest", reported_pytest)


def make_training_regional(
    regional_settings: Any, harness: Any, note: Any, nodes: Any, bundle: Any
) -> Any:
    class Regional:
        settings = regional_settings

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "release-a", "cluster_id": "cluster-a"}

        def gpu_nodes(self) -> list[dict[str, Any]]:
            return nodes()

        def gpu_workloads(self) -> list[Any]:
            return ["workload"] if harness.problem == "existing-workload" else []

        def business_workloads(self, node: str) -> list[Any]:
            return self.gpu_workloads()

        def node_snapshot(self, node: str) -> dict[str, Any]:
            note("node-snapshot", harness.killed, harness.restarted)
            return {
                "name": node,
                "uid": "uid-a",
                "boot_id": "boot-a",
                "ready": "False" if harness.problem == "not-ready" else "True",
                "unschedulable": (
                    harness.problem == "passive-cordon"
                    and harness.killed
                    and not harness.restarted
                ),
                "taints": [],
                "ownership_annotations": {"owner": "other"}
                if harness.problem == "owned-node"
                else {},
                "gpu_allocatable": 8,
            }

        def cpu_blast_snapshot(self) -> dict[str, Any]:
            harness.blast_reads += 1
            return (
                {"changed": True}
                if harness.problem == "blast" and harness.blast_reads > 1
                else {}
            )

        def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
            note("kube-read", plane, args)
            assert args[:2] == ("get", "node")
            return json.dumps(
                {"status": {"allocatable": {"vpc.amazonaws.com/efa": "2"}}}
            )

        def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
            return training_cpu_read(harness, note, bundle, script, *args)

        def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
            if not kwargs:
                note("incident-read")
                return {"profile": {"profile_version": "profile-a"}}
            job_id = kwargs.get("job_id", "")
            if kwargs.get("marker"):
                return {
                    "event": {
                        "xid": c021.LATE_XID,
                        "workload_state": "IDLE",
                        "affected_workload_ids": [],
                    },
                    "decision": {
                        "disposition": "MONITOR_ONLY",
                        "action": "NO_ACTION",
                        "official_action": "WRONG"
                        if harness.problem == "section-error"
                        else "RESTART_APP",
                        "reasons": ["no managed application to restart"],
                        "workflow_request_id": None,
                    },
                    "incident": {"state": "RECOVERED", "workflow_request_id": None},
                    "workflow": None,
                    "commands": [],
                }
            if kwargs.get("attempt_id"):
                workload = harness.workloads[-1]
                return {
                    "observations": [
                        {
                            "cluster_id": "cluster-a",
                            "job_id": job_id,
                            "attempt_id": kwargs["attempt_id"],
                            "workload_phase": "FAILED",
                            "containers": [
                                {
                                    "node_id": "node-a",
                                    "pod_uid": workload.source["pods"][0]["uid"],
                                    "terminated": True,
                                    "exit_code": 1,
                                    "gpu_uuids": ["GPU-a"],
                                }
                            ],
                        }
                    ]
                }
            if (
                harness.module is c016
                and harness.workloads
                and harness.workloads[-1].deleted
            ):
                workload = harness.workloads[-1]
                assert job_id == workload.settings.job_id
                return {
                    "observations": [
                        training_attempt_observation(
                            workload, attempt_id=attempt_id, phase=WorkloadPhase.STOPPED
                        )
                        for attempt_id in (
                            workload.settings.attempt_id,
                            workload.new_attempt,
                        )
                    ]
                }
            if harness.module is c021 and not job_id:
                note("node-attempt-read")
                workload = harness.workloads[-1]
                return {
                    "observations": [
                        training_attempt_observation(
                            workload,
                            attempt_id=workload.settings.attempt_id,
                            phase=(
                                WorkloadPhase.FAILED
                                if harness.killed
                                else WorkloadPhase.RUNNING
                            ),
                            killed=harness.killed,
                        )
                    ]
                }
            return {
                "restart_budget": {"restart_count": 1},
                "observations": [
                    {
                        "job_id": job_id,
                        "attempt_id": harness.workloads[-1].new_attempt,
                        "workload_phase": "RUNNING",
                    }
                ],
            }

        def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
            note("workflow-wait", kwargs)
            if "-b-" in kwargs["marker"]:
                return {
                    "workflow": {
                        "status": "FAILED",
                        "completed_operations": [],
                        "step_executions": [
                            {"details": {"reason": "RESTART_BUDGET_EXHAUSTED"}}
                        ],
                    },
                    "commands": [],
                    "restart_budget": {"restart_count": 1},
                }
            return {
                "incident": {"workload_identity_source": "SOLE_ACTIVE_ATTEMPT_ON_NODE"},
                "workflow": {
                    "official_steps": [
                        {
                            "operation": "RESTART_WORKLOAD",
                            "parameters": {"job_id": kwargs["job_id"]},
                        }
                    ]
                },
            }

    return Regional


def make_training_workload_collectors(
    regional: Any, harness: Any, note: Any, efa: Any
) -> Any:
    class Workload:
        def __init__(self, instance: Any, settings: Any) -> None:
            assert instance is regional
            self.settings = settings
            self.deleted = False
            self.progress = 0
            self.new_attempt = settings.attempt_id + "-replacement"
            self.restart_state: dict[str, Any] | None = None
            self.source = {
                "pods": [
                    {
                        "name": f"trainer-{index}",
                        "uid": settings.job_id + f"-source-{index}",
                        "node": f"node-{chr(97 + index)}",
                        "attempt_id": settings.attempt_id,
                        "phase": "Running",
                        "ready": True,
                    }
                    for index in range(3)
                ],
                "heartbeat_logs": {
                    f"trainer-{index}": "HEARTBEAT step=0 all_reduce=300"
                    for index in range(3)
                },
            }
            harness.workloads.append(self)

        def submit(self) -> None:
            note("submit")

        def wait_running(self, **kwargs: Any) -> dict[str, Any]:
            note("running")
            return deepcopy(self.source)

        def authorize_restart(self, state: dict[str, Any]) -> None:
            note("authorize-restart", state)
            self.restart_state = state

        def wait_restarted(self, uids: set[str], **kwargs: Any) -> dict[str, Any]:
            assert self.restart_state is not None
            note("restarted")
            harness.restarted = True
            return {
                "pods": [
                    {**row, "uid": row["uid"] + "-new", "attempt_id": self.new_attempt}
                    for row in self.source["pods"]
                ]
            }

        def snapshot(self) -> dict[str, Any]:
            note("workload-snapshot")
            value = deepcopy(self.source)
            self.progress += 1
            value["heartbeat_logs"] = {
                f"trainer-{index}": (f"HEARTBEAT step={self.progress} all_reduce=300")
                for index in range(3)
            }
            if harness.problem == "pod-change":
                value["pods"][0]["uid"] = "changed"
            return value

        def delete(self) -> None:
            note("workload-delete")
            self.deleted = True

    class Collector:
        def __init__(self, instance: Any, **kwargs: Any) -> None:
            self.regional = regional
            self.node = kwargs["node"]
            harness.collectors.append(self)

        def create(self) -> None:
            note("collector-create")

        def execute(self, *args: str, **kwargs: Any) -> dict[str, Any]:
            note(args[0], *args[1:])
            if args[0] == "unbind-efa":
                harness.efa_unbound = True
                harness.current_operation = "REMEDIATE_EFA_DRIVER"
            if args[0] == "restore-efa":
                harness.efa_unbound = False
                return {"already_bound": True, "timer_fired": False, "bound": True}
            if args[0] == "kill-workload":
                harness.killed = True
                return {"killed": [{"pid": 123}]}
            return {}

        def snapshot(self) -> dict[str, Any]:
            return {
                "gpu_inventory": [{"uuid": "GPU-a", "pci_bdf": "0000:af:00.0"}],
                "efa_inventory": efa(),
            }

        def efa_inventory(self) -> dict[str, Any]:
            return efa()

        def store_snapshot(self, marker: str) -> dict[str, Any]:
            note("seed-read", marker)
            return {
                "seed_marker": marker,
                "events": [{"event_id": marker}],
                "incidents": [],
                "workflows": [],
                "commands": [],
            }

        def restore_incidents(self, state: Any, **kwargs: Any) -> list[Any]:
            note("incident-restore")
            return []

        def cleanup(self) -> dict[str, bool]:
            note("collector-cleanup")
            return {"pod": harness.problem == "probe-residual"}

    return Workload, Collector


def make_training_auxiliary(regional: Any, harness: Any, note: Any) -> Any:
    class ResetHost:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def create(self) -> None:
            note("reset-probe-create")

        def cleanup(self) -> dict[str, bool]:
            note("reset-probe-cleanup")
            return {"pod": False}

    class Prewarm:
        def __init__(self, instance: Any, **kwargs: Any) -> None:
            assert instance is regional

        def create(self, candidates: list[str]) -> None:
            note("prewarm-create", candidates)

        def cleanup(self) -> dict[str, bool]:
            note("prewarm-cleanup")
            return {"pod": harness.problem == "prewarm-residual"}

    class Plugin:
        def __init__(self, instance: Any, **kwargs: Any) -> None:
            self.operation = (
                "RESTART_GPU_DEVICE_PLUGIN"
                if kwargs["resource"] == "nvidia.com/gpu"
                else "RESTART_EFA_DEVICE_PLUGIN"
            )
            harness.plugins.append(self)

        def discover(self) -> dict[str, Any]:
            note("plugin-discover")
            return {"name": "plugin-fixture", "namespace": "fixture"}

        def exclude_node(self) -> None:
            harness.current_operation = self.operation
            note("plugin-exclude")

        def restore(self) -> None:
            note("plugin-restore")

        def wait_allocatable(self, expected: int) -> dict[str, Any]:
            note("allocatable", expected)
            return {"allocatable": expected}

    return ResetHost, Prewarm, Plugin


@pytest.fixture(
    params=(c016, c017, c021), ids=("collect016", "collect017", "collect021")
)
def training_case(request: Any, tmp_path: Path, monkeypatch: Any) -> Any:
    module = request.param
    clock = Clock()
    kubeconfig = tmp_path / "fixture-kubeconfig"
    kubeconfig.touch()
    site = tmp_path / "fixture-site"
    site.touch()
    regional_settings = SimpleNamespace(
        gpu_kubeconfig=kubeconfig,
        gpu_context="gpu-a",
        namespace="fixture",
        cluster_id="cluster-a",
        environment=lambda: {"CLUSTER": "cluster-a"},
    )
    values = {
        "regional": regional_settings,
        "site_file": site,
        "host_probe_image": "example@sha256:" + "a" * 64,
        "predecessor_path": tmp_path / "predecessor.json",
    }
    if module is c017:
        values["node"] = "node-a"
    settings = module.Settings(**values)
    harness = SimpleNamespace(
        module=module,
        settings=settings,
        clock=clock,
        root=tmp_path,
        fail_at="",
        problem="",
        calls=[],
        workloads=[],
        collectors=[],
        plugins=[],
        current_operation="REMEDIATE_EFA_DRIVER",
        efa_unbound=False,
        killed=False,
        restarted=False,
        plan_reads=0,
        blast_reads=0,
        bundles=0,
        predecessor=True,
        focused_status=0,
        candidate_count=3,
        planning_seen={},
    )

    def note(name: str, *args: Any) -> None:
        harness.calls.append((name, args))
        if harness.fail_at == name:
            raise RuntimeError(name + " fixture failure")

    def nodes() -> list[dict[str, Any]]:
        return [
            {
                "name": f"node-{chr(97 + index)}",
                "uid": f"uid-{index}",
                "ready": "True",
                "unschedulable": False,
                "taints": [],
            }
            for index in range(harness.candidate_count)
        ]

    def efa() -> dict[str, Any]:
        bdfs = ["0000:af:00.0", "0000:bf:00.0"]
        if harness.efa_unbound:
            bdfs = bdfs[1:]
        return {
            "discovered_count": len(bdfs),
            "active_count": len(bdfs),
            "devices": [{"pci_bdf": bdf, "driver": "efa"} for bdf in bdfs],
        }

    def bundle() -> dict[str, Any]:
        harness.bundles += 1
        driver = harness.current_operation == "REMEDIATE_EFA_DRIVER"
        steps = (
            c017.EFA_REMEDIATION_STEPS
            if driver
            else c017.GPU_PLUGIN_STEPS
            if harness.current_operation == "RESTART_GPU_DEVICE_PLUGIN"
            else c017.EFA_PLUGIN_STEPS
        )
        if (
            harness.problem == "restart-workload"
            and harness.current_operation == "RESTART_EFA_DEVICE_PLUGIN"
        ):
            steps = (*steps, "RESTART_WORKLOAD")
        return {
            "workflow": {
                "request_id": "" if harness.problem == "request-id" else "workflow-a",
                "incident_id": "incident-a",
                "status": "SUCCEEDED",
                "official_action": harness.current_operation,
                "official_steps": [{"operation": operation} for operation in steps],
                "step_executions": [
                    {
                        "operation": operation,
                        "status": "SUCCEEDED",
                        "adapter_operation_id": "remote/command-a",
                    }
                    for operation in steps
                ],
            },
            "incident": {
                "incident_id": "incident-a",
                "cluster_id": "cluster-a",
                "node_ids": ["node-a"],
                "job_id": (
                    harness.workloads[-1].settings.job_id if harness.workloads else None
                ),
                "attempt_id": (
                    harness.workloads[-1].settings.attempt_id
                    if harness.workloads
                    else None
                ),
                "state": "ACTION_PENDING"
                if harness.problem == "incident-delay" and harness.bundles == 1
                else "RECOVERED",
                "effective_action": harness.current_operation,
                "reasons": ["driver is not bound"],
            },
        }

    Regional = make_training_regional(regional_settings, harness, note, nodes, bundle)

    regional = Regional()
    harness.regional = regional

    class Factory:
        def __new__(cls, settings: Any) -> Regional:
            return regional

        @staticmethod
        def run(command: list[str], **kwargs: Any) -> Any:
            note("focused-tests", command)
            return subprocess.CompletedProcess(
                command, harness.focused_status, "fixture output", ""
            )

    Workload, Collector = make_training_workload_collectors(
        regional, harness, note, efa
    )

    ResetHost, Prewarm, Plugin = make_training_auxiliary(regional, harness, note)

    monkeypatch.setattr(module, "RegionalLiveFixture", Factory)
    monkeypatch.setattr(
        module, "predecessor_evidence", lambda *a, **k: {"valid": harness.predecessor}
    )
    monkeypatch.setattr(module, "ManagedWorkloadFixture", Workload)
    monkeypatch.setattr(module, "CollectorAcceptanceFixture", Collector)
    monkeypatch.setattr(module, "ImagePrewarmFixture", Prewarm)
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(module.base, "time", clock)
    monkeypatch.setattr(collect017_training, "time", clock)
    monkeypatch.setattr(collect021_passive, "time", clock)
    if module is c016:
        bind_local_contract_transport(module, harness, monkeypatch, note)
    if module is c017:
        monkeypatch.setattr(module.base, "DevicePluginFixture", Plugin)
    else:
        monkeypatch.setattr(
            module.workload_case,
            "wait_observation",
            lambda regional, settings, **kwargs: {
                "cluster_id": settings.regional.cluster_id,
                "job_id": settings.job_id,
                "attempt_id": settings.attempt_id,
                "workload_phase": "RUNNING",
                "observed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if module is c016:
        monkeypatch.setattr(module, "HostProbeFixture", ResetHost)
        monkeypatch.setattr(
            module.workload_case,
            "workflow_errors",
            lambda *a, **k: ["bad restart"]
            if harness.problem == "section-error"
            else [],
        )

        def reset(*args: Any, **kwargs: Any) -> Any:
            note("reset-section")
            assert kwargs["expected_steps"] == c016.WORKLOAD_RESET_STEPS
            state = {"workflow": {"request_id": "unit-reset-workflow"}}
            observed = kwargs["observe_post_restart_workload"](state)
            return {**state, "post_restart_workload": observed}, []

        monkeypatch.setattr(module.base, "run_single_reset", reset)
    return harness
