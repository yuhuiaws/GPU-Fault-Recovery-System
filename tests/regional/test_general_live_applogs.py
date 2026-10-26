"""GF-GENERAL-LIVE-017 application error logs: category-module verdict tests.

The journal variants run the kernel-replay FakeProbe; the CRI attribution
variants run a fake managed job, the CRI Pod-log FakeProbe and scripted
``APPLOG_PROBE`` records (attempt observations, delivered NODE_LOGS entries,
the incident the product minted for the entry and its workflow with the
attribution parameters on the bundle step); ``missing_flight_recorder``'s
context (``mfr_ctx``) runs 011's fakes without a flight recorder, its verdict
tests are ``test_general_live_applogs_hung``.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_applogs as mod
from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional.general_live_applog_fixture import (
    CUDA_OOM_LINE,
    PARTIAL_HEAD,
    PARTIAL_TAIL,
)
from scripts.e2e.regional.general_live_fixture import (
    GeneralLiveSettings,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import coverage_rows
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)
from tests.regional.test_general_live_nccl import (
    PEER,
    FakeNcclJob,
    Traffic,
    efa_probe,
    hung_records,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"
POD_UID = "0f3b7c2e-4d5a-4b6c-8d9e-0a1b2c3d4e5f"
NEW_UID = "9e8d7c6b-5a49-4837-a6b5-c4d3e2f1a0b9"
PID = 4242
NEW_PID = 5151
STAMP = "2026-10-24T00:00:30.000000000Z"
RUN_FILE = "gfrs-live-abcdef0123456789.log"
PATH = f"/var/log/pods/gpu-fault-system_pod-0_{POD_UID}/pytorch/{RUN_FILE}"
OLD_PATH = PATH
ROTATED = PATH + ".1"


def _spec(variant: str) -> Any:
    return next(item for item in mod.VARIANTS if item.variant == variant)


# --------------------------------------------------------------------------- #
# journal replay variants
# --------------------------------------------------------------------------- #
def _host_script() -> dict[str, object]:
    return {
        "install-records": {"records": 1},
        "open-window": {"after": {"ActiveState": "active"}},
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
    }


def _incident(
    action: str, *, incident_id: str = "inc-log-1", workflow: str = "wf-1"
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "event_id": "log-gpu-eks-cluster-node-abc",
        "event_type": "NODE_HEALTH",
        "policy_source": "SITE_APPLICATION_LOG",
        "effective_action": action,
        "state": "ACTION_PENDING",
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": workflow,
        "node_ids": [NODE],
    }


def _workflow(
    ops: list[str], *, incident_id: str = "inc-log-1", request_id: str = "wf-1"
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "incident_id": incident_id,
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [{"operation": op, "parameters": {}} for op in ops],
        "step_executions": [],
    }


def _activity(
    incidents: list[dict[str, object]], workflows: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "incidents": incidents,
        "workflows": workflows,
        "markers": [],
        "notifications": [],
        "batches": [],
        "metric_latest": [],
        "collector_statuses": [],
    }


def _node(isolated: bool) -> dict[str, object]:
    return {
        "name": NODE,
        "ready": "True",
        "unschedulable": isolated,
        "taints": [],
        "ownership_annotations": {},
    }


def _ctx(variant: str, *, activity: dict[str, object], isolated: bool, tmp_path: Path):
    spec = _spec(variant)
    fake = FakeProbe(_host_script())
    regional = FakeRegional(
        cpu_python=store_script({"activity": activity}), node_snapshot=_node(isolated)
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": fake},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def test_cgroup_oom_installs_the_kernel_oom_kill_line(tmp_path: Path) -> None:
    import base64
    import json

    activity = _activity(
        [_incident("RUN_DIAGNOSTICS")],
        [_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    ctx = _ctx("cgroup_oom", activity=activity, isolated=False, tmp_path=tmp_path)
    mod.cgroup_oom(ctx)
    fake = ctx.probes[f"kernel-replay-journal@{NODE}"]
    install = next(call for call in fake.calls if call[0] == "install-records")
    [record] = json.loads(base64.b64decode(install[install.index("--b64") + 1]))
    assert record["message"] == mod.CGROUP_OOM_MESSAGE, record
    assert record["message"].startswith(
        "Memory cgroup out of memory: Killed process "
    ), "the reviewed record is the kernel's oom_kill_process line, not a paraphrase"
    assert not record["message"].startswith("kernel:"), (
        "a journal MESSAGE never carries the syslog identifier"
    )
    assert record["transport"] == "kernel", record


def test_cgroup_oom_line_is_graded_diagnostic_by_the_product() -> None:
    from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
    from gpu_fault.models import RecoveryAction, Severity
    from tests._builders import build_store

    now = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    [finding] = NodeHealthPolicy(build_store()).evaluate_logs(
        NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=now,
            entries=[
                NodeLogEntry(
                    entry_id="gfrs-replay-run-0-boot",
                    source="dmesg",
                    observed_at=now,
                    message=mod.CGROUP_OOM_MESSAGE,
                    fields={"_TRANSPORT": "kernel"},
                )
            ],
        )
    )
    assert (finding.severity, finding.recommended_action, finding.policy_source) == (
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "SITE_APPLICATION_LOG",
    ), "the harness and the product agree on the reviewed OOM line"


def test_cgroup_oom_is_a_diagnostic(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("RUN_DIAGNOSTICS")],
        [_workflow(["FREEZE_EVIDENCE", "VALIDATE_HOST"])],
    )
    ctx = _ctx("cgroup_oom", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.cgroup_oom(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_cgroup_oom_fails_if_it_stops_a_workload(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("RUN_DIAGNOSTICS")],
        [_workflow(["FREEZE_EVIDENCE", "STOP_WORKLOADS"])],
    )
    ctx = _ctx("cgroup_oom", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.cgroup_oom(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("stopped a workload" in error for error in result["errors"]), result[
        "errors"
    ]


def test_nccl_error_is_evidence_only(tmp_path: Path) -> None:
    activity = _activity([_incident("NO_ACTION")], [_workflow(["FREEZE_EVIDENCE"])])
    ctx = _ctx("nccl_error", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.nccl_error(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_nccl_error_fails_on_a_diagnostic_workflow(tmp_path: Path) -> None:
    activity = _activity(
        [_incident("NO_ACTION")],
        [_workflow(["FREEZE_EVIDENCE", "COLLECT_DIAGNOSTIC_BUNDLE"])],
    )
    ctx = _ctx("nccl_error", activity=activity, isolated=False, tmp_path=tmp_path)
    result = mod.nccl_error(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("diagnostic workflow" in error for error in result["errors"]), result[
        "errors"
    ]


def test_distributed_error_fails_if_it_isolates(tmp_path: Path) -> None:
    activity = _activity([_incident("QUARANTINE")], [_workflow(["QUARANTINE"])])
    ctx = _ctx("distributed_error", activity=activity, isolated=True, tmp_path=tmp_path)
    result = mod.distributed_error(ctx)
    assert result["verdict"] == "FAIL", result


# --------------------------------------------------------------------------- #
# CRI attribution variants: fakes
# --------------------------------------------------------------------------- #
class FakeApplogJob:
    """The managed attribution target: one Pod, a product restart on demand."""

    instances: list["FakeApplogJob"] = []
    #: ``pod_detail`` phases per Pod name; the fail_once job's restart Pod.
    replacement: dict[str, Any] | None = None
    alive_value = True

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        mode: str = "trainer",
        restart_budget: int = 0,
        gpu_count: int = 8,
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.mode = mode
        self.restart_budget = restart_budget
        self.attempt_id = f"{name}-a001"
        self.container = "pytorch"
        self.namespace = "gpu-fault-system"
        self.node = NODE
        self.annotations: list[tuple[str, int]] = []
        self.exits: list[str] = []
        self.started = False
        self.deleted = False
        FakeApplogJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        self.started = True
        return {}

    def pods(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "pod-0",
                "uid": POD_UID,
                "node": NODE,
                "phase": "Running",
                "ready": True,
                "attempt_id": self.attempt_id,
            }
        ]

    def _replaced(self) -> dict[str, Any] | None:
        if self.replacement is None:
            return None
        value = dict(self.replacement)
        value["attempt_id"] = str(value["attempt_id"]).replace("JOB", self.name)
        return value

    def pod_detail(self, pod_name: str) -> dict[str, Any]:
        replacement = self._replaced()
        if replacement is not None and pod_name == replacement["name"]:
            return replacement
        return {
            "name": pod_name,
            "uid": POD_UID,
            "namespace": self.namespace,
            "node": NODE,
            "phase": "Running",
            "attempt_id": self.attempt_id,
            "annotations": {},
            "container": {
                "name": "pytorch",
                "container_id": "ab" * 32,
                "running": True,
            },
        }

    def wait_replaced(
        self, old_uids: set[str], *, timeout_seconds: float, name: str
    ) -> list[dict[str, Any]]:
        replacement = self._replaced()
        if replacement is None:
            return self.pods()
        return [
            *self.pods(),
            {
                "name": replacement["name"],
                "uid": replacement["uid"],
                "node": NODE,
                "phase": replacement["phase"],
                "ready": True,
                "attempt_id": replacement["attempt_id"],
            },
        ]

    def annotate_host_pid(self, pod_name: str, pid: int) -> str:
        self.annotations.append((pod_name, pid))
        return "annotated"

    def request_exit(self, pod_name: str) -> str:
        self.exits.append(pod_name)
        return ""

    def alive(self) -> dict[str, Any]:
        return {"alive": self.alive_value}

    def identity(self) -> dict[str, Any]:
        return {"job_id": self.name, "attempt_id": self.attempt_id, "mode": self.mode}

    def delete(self) -> None:
        self.deleted = True


def _replacement() -> dict[str, Any]:
    return {
        "name": "pod-1",
        "uid": NEW_UID,
        "namespace": "gpu-fault-system",
        "node": NODE,
        "phase": "Running",
        "attempt_id": "JOB-a001-r-11223344",
        "annotations": {},
        "container": {"name": "pytorch", "container_id": "cd" * 32, "running": True},
    }


def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the managed jobs for the fakes and reset their class state.

    Plain so ``test_general_live_applogs_hung`` can run it from its own autouse
    fixture: a fixture defined here is not visible from another module.
    """

    FakeApplogJob.instances = []
    FakeApplogJob.replacement = None
    FakeApplogJob.alive_value = True
    monkeypatch.setattr(mod, "ManagedApplogJob", FakeApplogJob)
    monkeypatch.setattr(mod, "restore_incidents", lambda ctx, incidents, reason: [])
    FakeNcclJob.instances = []
    FakeNcclJob.traffic = Traffic()
    FakeNcclJob.alive_value = True
    FakeNcclJob.events = None
    monkeypatch.setattr(nc, "ManagedNcclJob", FakeNcclJob)


@pytest.fixture(autouse=True)
def _patch_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jobs(monkeypatch)


def _logs_probe(
    *, statuses: list[dict[str, Any]] | None = None, pid: int = PID
) -> FakeProbe:
    tracked = {"collector": {"tracked": {PATH: {"offset": 0}}, "pending": {}}}
    return FakeProbe(
        {
            "open-window": {
                "after": {"ActiveState": "active"},
                "proof_scope": "live-replay",
            },
            "inspect": {"process": {"init_pid": pid}},
            "create-log": {"file": {"path": PATH}},
            "status": statuses or [tracked],
            "write": {"stamps": [STAMP], "file": {"path": PATH}},
            "rotate": {"rotated": {"path": ROTATED}, "current": {"path": PATH}},
            "restart-collector": {"after": {"ActiveState": "active"}},
            "close-window": {"clean": True},
        }
    )


def _container(
    *,
    pod_uid: str = POD_UID,
    host_pid: int | None = PID,
    terminated: bool = False,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "pod_uid": pod_uid,
        "pod_name": "pod-0",
        "container_name": "pytorch",
        "container_id": "containerd://" + "ab" * 32,
        "host_pid": host_pid,
        "node_id": NODE,
        "terminated": terminated,
        "exit_code": exit_code,
    }


def _attempt(
    attempt_id: str,
    *,
    containers: list[dict[str, Any]],
    phase: str = "RUNNING",
    observed_at: str = "2026-10-24T00:00:25+00:00",
) -> dict[str, Any]:
    return {
        "job_id": "JOB",
        "attempt_id": attempt_id,
        "workload_phase": phase,
        "observed_at": observed_at,
        "started_at": "2026-10-24T00:00:00+00:00",
        "containers": containers,
    }


def _entry(
    *,
    message: str = CUDA_OOM_LINE,
    path: str = PATH,
    pod_uid: str = POD_UID,
    entry_id: str = "e" * 64,
    observed_at: str = "2026-10-24T00:00:30+00:00",
) -> dict[str, Any]:
    return {
        "record_id": "raw-1",
        "batch_id": f"logs-{NODE}-1",
        "collected_at": "2026-10-24T00:00:40+00:00",
        "entry_id": entry_id,
        "source": "training-log",
        "observed_at": observed_at,
        "message": message,
        "fields": {
            "path": path,
            "pod_uid": pod_uid,
            "container_name": "pytorch",
            "timestamp_source": "cri",
            "stream": "stdout",
        },
    }


def _parameters(
    *,
    attribution: str = "OBSERVED_PROCESS",
    process_state: str = "LIVE",
    pid: int | None = PID,
    capture: bool = True,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "attribution": attribution,
        "process_state": process_state,
        "log_source": "training-log",
        "capture_process_state": capture,
        "diagnostic_reason": "APPLICATION_ERROR",
    }
    if pid is not None:
        value["target_pids_by_node"] = {NODE: [pid]}
    return value


#: The archive and digest a Node Agent names in its bundle receipt.
BUNDLE_ARCHIVE = (
    "s3://control-records/node-diagnostics/{node}/gpu-diagnostic-e87b32e8.tar.gz"
)
BUNDLE_SHA256 = "70c4b0b3327483e8c26f91d86f6a6ffbac3a2fdb6ad39232b31a8e7a71eb5312"


def _bundle_receipt(
    node: str, *, capture: bool = True, failed_captures: int = 0
) -> dict[str, Any]:
    """What one Node Agent returns for COLLECT_DIAGNOSTIC_BUNDLE (``diagnostics.py``).

    The live HyperPod receipt of 2026-10-26 (release 37db33f254a1): 25 captures,
    none failed -- the eight node-wide captures plus the targeted PID's compute
    row, ``ps``/``ss``, three py-spy dumps, ``/proc`` status/cmdline and three
    rounds of ``/proc`` stack/wchan + strace.
    """

    return {
        "evidence_ref": BUNDLE_ARCHIVE.format(node=node),
        "manifest_summary": {
            "capture_count": 25,
            "capture_process_state": capture,
            "diagnostic_reason": "APPLICATION_ERROR",
            "failed_capture_count": failed_captures,
        },
        "node_action_command_id": f"wf-log/1/COLLECT_DIAGNOSTIC_BUNDLE/{node}/agent-26",
        "sha256": BUNDLE_SHA256,
        "size_bytes": 5777487,
    }


def _bundle_details(
    *,
    capture: bool = True,
    node: str | None = NODE,
    failure: str | None = None,
    failed_captures: int = 0,
) -> dict[str, Any]:
    """The bundle step's ``details`` as ``step_execution._finalize`` records them.

    A node-agent step keeps every agent's receipt under ``node_results[<node>]``
    and hoists nothing; a node that answered nothing usable is under
    ``node_failures`` with its cause (``failure``) and has no receipt.
    """

    return {
        "failed_nodes": [NODE] if failure else [],
        "node_failures": {NODE: failure} if failure else {},
        "node_results": (
            {
                node: _bundle_receipt(
                    node, capture=capture, failed_captures=failed_captures
                )
            }
            if node
            else {}
        ),
        "node_waiting": {},
        "waiting_nodes": [],
    }


def _records(
    *,
    attempts: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    action: str | None = "RUN_DIAGNOSTICS",
    incident_attempt: str | None = "JOB-a001",
    ops: tuple[str, ...] = (
        "FREEZE_EVIDENCE",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "VALIDATE_HOST",
        "ESCALATE_SUPPORT",
    ),
    parameters: dict[str, Any] | None = None,
    bundle_status: str = "SUCCEEDED",
    bundle_capture: bool = True,
    bundle_node: str | None = NODE,
    bundle_node_failure: str | None = None,
    bundle_failed_captures: int = 0,
    workflow_status: str = "SUCCEEDED",
    graded_entry: int = 0,
) -> dict[str, Any]:
    """Scripted ``APPLOG_PROBE`` output: the incident is minted for one entry."""

    incidents: list[dict[str, Any]] = []
    workflows: dict[str, Any] = {}
    if action is not None and entries:
        entry = entries[graded_entry]
        event_id = mod.log_event_id(CLUSTER, NODE, str(entry["entry_id"]))
        incidents.append(
            {
                "incident_id": f"inc-{event_id}",
                "event_id": event_id,
                "event_type": "NODE_HEALTH",
                "policy_source": "SITE_APPLICATION_LOG",
                "effective_action": action,
                "state": "ESCALATED" if action == "RUN_DIAGNOSTICS" else "RECOVERED",
                "job_id": "JOB" if incident_attempt else None,
                "attempt_id": incident_attempt,
                "workflow_request_id": "wf-log",
                "created_at": "2026-10-24T00:00:45+00:00",
                "node_ids": [NODE],
            }
        )
        steps = []
        for op in ops:
            params = (
                dict(parameters or _parameters())
                if op == "COLLECT_DIAGNOSTIC_BUNDLE"
                else {}
            )
            steps.append({"operation": op, "parameters": params})
        executions = [
            {
                "step_index": index,
                "operation": op,
                "status": bundle_status
                if op == "COLLECT_DIAGNOSTIC_BUNDLE"
                else "SUCCEEDED",
                "details": (
                    _bundle_details(
                        capture=bundle_capture,
                        node=bundle_node,
                        failure=bundle_node_failure,
                        failed_captures=bundle_failed_captures,
                    )
                    if op == "COLLECT_DIAGNOSTIC_BUNDLE"
                    else {}
                ),
                "updated_at": "2026-10-24T00:01:00+00:00",
            }
            for index, op in enumerate(ops)
        ]
        workflows["wf-log"] = {
            "request_id": "wf-log",
            "incident_id": f"inc-{event_id}",
            "status": workflow_status,
            "created_at": "2026-10-24T00:00:45+00:00",
            "official_steps": steps,
            "step_executions": executions,
        }
    return {
        "since": "2026-10-24T00:00:00+00:00",
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "job_id": "JOB",
        "attempts": attempts,
        "entries": entries,
        "incidents": incidents,
        "workflows": workflows,
    }


def _cri_ctx(
    variant: str,
    tmp_path: Path,
    *,
    applog: Any,
    probe: FakeProbe | None = None,
    isolated: bool = False,
) -> Any:
    spec = _spec(variant)
    (tmp_path / "site.yaml").write_text("site: test\n")
    base = make_settings(spec, node=NODE)
    settings = GeneralLiveSettings(
        **{**base.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    regional = FakeRegional(
        cpu_python=store_script({"applog": applog, "activity": _activity([], [])}),
        node_snapshot=_node(isolated),
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"cri-pod-logs@{NODE}": probe or _logs_probe()},
        settings=settings,
        window_minutes=120,
    )
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=5 * next(ticks))
    return ctx


def _job_name() -> str:
    return FakeApplogJob.instances[-1].name


def _rename(records: dict[str, Any], job: str) -> dict[str, Any]:
    """Point the scripted records at the fake job's real (derived) name."""

    text = (
        str(records).replace("'JOB-a001", f"'{job}-a001").replace("'JOB'", f"'{job}'")
    )
    return eval(text)  # noqa: S307 - test fixture round-trip of its own repr


class _Applog:
    """An ``applog`` responder that learns the job name from the first call."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.calls = 0

    def __call__(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        job = arguments[3]
        index = min(self.calls, len(self.docs) - 1)
        self.calls += 1
        return _rename(self.docs[index], job)


# --------------------------------------------------------------------------- #
# live_pid
# --------------------------------------------------------------------------- #
def _live_records(**overrides: Any) -> dict[str, Any]:
    return _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[_entry()],
        **overrides,
    )


def test_live_pid_passes_when_the_live_process_is_targeted(tmp_path: Path) -> None:
    ctx = _cri_ctx("live_pid", tmp_path, applog=_Applog([_live_records()]))
    result = mod.live_pid(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeApplogJob.instances[-1]
    assert job.mode == "trainer" and job.annotations == [("pod-0", PID)]
    assert result["host_pid"] == PID and result["proof_scope"] == "live-replay"
    probe = ctx.probes[f"cri-pod-logs@{NODE}"]
    names = [call[0] for call in probe.calls]  # type: ignore[attr-defined]
    assert names[:3] == ["open-window", "inspect", "create-log"]
    assert "write" in names and names.index("status") < names.index("write"), (
        "the file is tracked by the collector before the line is written"
    )
    assert ctx.cleanup.pending()[-1].startswith("restore-inc-log-"), (
        "the diagnostic incident is restored in cleanup"
    )


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"parameters": _parameters(process_state="EXITED")}, "process_state 'EXITED'"),
        ({"parameters": _parameters(pid=99)}, "target pids"),
        (
            {"parameters": _parameters(attribution="UNIQUE_MANAGED_ATTEMPT")},
            "attribution",
        ),
        ({"incident_attempt": None}, "incident attributed to"),
        (
            {"ops": ("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT")},
            "no COLLECT_DIAGNOSTIC_BUNDLE",
        ),
        ({"bundle_status": "FAILED"}, "did not succeed"),
        ({"bundle_capture": False}, "did not run the live process capture"),
        (
            {
                "bundle_node": None,
                "bundle_node_failure": "ObservationIncomplete: RESULT_SIZE_LIMIT",
            },
            f"recorded no result for {NODE}: ObservationIncomplete: RESULT_SIZE_LIMIT",
        ),
        ({"bundle_node": PEER}, f"recorded no result for {NODE}"),
        (
            {"ops": ("FREEZE_EVIDENCE", "STOP_WORKLOADS", "COLLECT_DIAGNOSTIC_BUNDLE")},
            "stopped the attempt",
        ),
    ],
)
def test_live_pid_fails_on_wrong_attribution_or_handling(
    tmp_path: Path, overrides: dict[str, Any], needle: str
) -> None:
    ctx = _cri_ctx("live_pid", tmp_path, applog=_Applog([_live_records(**overrides)]))
    result = mod.live_pid(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(needle in error for error in result["errors"]), result["errors"]


def test_live_pid_fails_when_the_entry_names_another_pod_or_time(
    tmp_path: Path,
) -> None:
    records = _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[_entry(pod_uid=NEW_UID, observed_at="2026-10-24T00:00:35+00:00")],
    )
    result = mod.live_pid(_cri_ctx("live_pid", tmp_path, applog=_Applog([records])))
    assert result["verdict"] == "FAIL"
    assert any("entry pod_uid" in error for error in result["errors"]), result["errors"]
    assert any("not the CRI stamp" in error for error in result["errors"]), result[
        "errors"
    ]


def test_live_pid_is_inconclusive_without_a_delivered_line(tmp_path: Path) -> None:
    records = _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[],
        action=None,
    )
    with pytest.raises(VariantInconclusive, match="delivered no record"):
        mod.live_pid(_cri_ctx("live_pid", tmp_path, applog=_Applog([records])))


def test_live_pid_is_inconclusive_when_the_watcher_never_sees_the_pid(
    tmp_path: Path,
) -> None:
    records = _records(
        attempts=[_attempt("JOB-a001", containers=[_container(host_pid=None)])],
        entries=[_entry()],
    )
    with pytest.raises(VariantInconclusive, match="host PID"):
        mod.live_pid(_cri_ctx("live_pid", tmp_path, applog=_Applog([records])))


def test_live_pid_result_carries_the_agents_bundle_receipt(tmp_path: Path) -> None:
    records = _live_records(bundle_failed_captures=2)
    result = mod.live_pid(_cri_ctx("live_pid", tmp_path, applog=_Applog([records])))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["diagnostic_bundle"] == _bundle_receipt(NODE, failed_captures=2), (
        "the receipt the Node Agent returned travels with the verdict"
    )
    assert any(
        note.startswith(f"2 of 25 bundle captures on {NODE} failed")
        for note in result["notes"]
    ), result["notes"]


# --------------------------------------------------------------------------- #
# the bundle oracle over the step record
# --------------------------------------------------------------------------- #
RUN_DIAGNOSTICS = {"effective_action": "RUN_DIAGNOSTICS"}


def _bundle_execution(details: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_index": 1,
        "operation": "COLLECT_DIAGNOSTIC_BUNDLE",
        "status": "SUCCEEDED",
        "details": details,
        "updated_at": "2026-10-24T00:01:00+00:00",
    }


def _bundle_workflow(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_steps": [
            {"operation": "COLLECT_DIAGNOSTIC_BUNDLE", "parameters": _parameters()}
        ],
        "step_executions": [execution],
    }


def test_bundle_oracle_reads_the_receipt_under_node_results() -> None:
    """The live shape (HyperPod, 2026-10-26): the receipt is per node, never hoisted."""

    execution = _bundle_execution(_bundle_details())
    assert mod.bundle_node_result(execution, NODE) == _bundle_receipt(NODE)
    assert mod.bundle_node_result(execution, PEER) is None, (
        "another node's receipt is not this node's"
    )
    assert (
        mod.diagnostic_chain_errors(
            RUN_DIAGNOSTICS, _bundle_workflow(execution), expect_bundle=True, node=NODE
        )
        == []
    )


def test_bundle_oracle_rejects_a_summary_the_product_never_hoists() -> None:
    """The key the oracle used to read; the live FAIL of 2026-10-26 was this."""

    hoisted = _bundle_execution({"manifest_summary": {"capture_process_state": True}})
    assert mod.diagnostic_chain_errors(
        RUN_DIAGNOSTICS, _bundle_workflow(hoisted), expect_bundle=True, node=NODE
    ) == [f"the bundle step recorded no result for {NODE}"]


@pytest.mark.parametrize(
    "details, expected",
    [
        (
            _bundle_details(
                node=None, failure="ObservationIncomplete: RESULT_SIZE_LIMIT"
            ),
            f"the bundle step recorded no result for {NODE}: "
            "ObservationIncomplete: RESULT_SIZE_LIMIT",
        ),
        (
            {"node_results": {}, "skipped": "hung_target_exited"},
            f"the bundle step recorded no result for {NODE}: hung_target_exited",
        ),
        (
            _bundle_details(capture=False),
            f"the bundle on {NODE} did not run the live process capture: "
            f"manifest_summary {_bundle_receipt(NODE, capture=False)['manifest_summary']!r}",
        ),
    ],
)
def test_bundle_oracle_names_the_missing_receipt_or_capture(
    details: dict[str, Any], expected: str
) -> None:
    errors = mod.diagnostic_chain_errors(
        RUN_DIAGNOSTICS,
        _bundle_workflow(_bundle_execution(details)),
        expect_bundle=True,
        node=NODE,
    )
    assert errors == [expected]


def test_bundle_oracle_needs_the_node_only_when_a_bundle_is_expected() -> None:
    workflow = _bundle_workflow(_bundle_execution(_bundle_details()))
    with pytest.raises(ValueError, match="node"):
        mod.diagnostic_chain_errors(RUN_DIAGNOSTICS, workflow, expect_bundle=True)
    errors = mod.diagnostic_chain_errors(RUN_DIAGNOSTICS, workflow, expect_bundle=False)
    assert errors[:1] == [
        "an exited process was still scheduled for sampling: ['COLLECT_DIAGNOSTIC_BUNDLE']"
    ], "an exited PID's chain is judged without a node"


# --------------------------------------------------------------------------- #
# exited_pid
# --------------------------------------------------------------------------- #
def _exited_docs(**overrides: Any) -> list[dict[str, Any]]:
    ops = overrides.pop("ops", ("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"))
    live = _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[],
        action=None,
    )
    exited = _records(
        attempts=[
            _attempt(
                "JOB-a001",
                containers=[_container(terminated=True, exit_code=0)],
                phase="SUCCEEDED",
                observed_at="2026-10-24T00:00:35+00:00",
            )
        ],
        entries=[_entry()],
        ops=ops,
        **overrides,
    )
    return [live, exited]


def test_exited_pid_passes_without_any_stack_capture(tmp_path: Path) -> None:
    ctx = _cri_ctx("exited_pid", tmp_path, applog=_Applog(_exited_docs()))
    result = mod.exited_pid(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeApplogJob.instances[-1]
    assert (
        job.mode == "hold"
        and job.exits == ["pod-0"]
        and job.annotations == [("pod-0", PID)]
    )
    assert result["exit_code"] == 0


def test_exited_pid_fails_when_a_bundle_targets_the_dead_pid(tmp_path: Path) -> None:
    docs = _exited_docs(
        ops=("FREEZE_EVIDENCE", "COLLECT_DIAGNOSTIC_BUNDLE", "VALIDATE_HOST"),
        parameters=_parameters(process_state="EXITED", pid=PID, capture=True),
    )
    result = mod.exited_pid(_cri_ctx("exited_pid", tmp_path, applog=_Applog(docs)))
    assert result["verdict"] == "FAIL"
    assert any("scheduled for sampling" in error for error in result["errors"]), result[
        "errors"
    ]
    assert any("stack capture" in error for error in result["errors"]), result["errors"]


def test_exited_pid_fails_when_unattributed_inside_the_freshness_window(
    tmp_path: Path,
) -> None:
    result = mod.exited_pid(
        _cri_ctx(
            "exited_pid", tmp_path, applog=_Applog(_exited_docs(incident_attempt=None))
        )
    )
    assert result["verdict"] == "FAIL"
    assert any(
        "incident attributed to None/None" in error for error in result["errors"]
    ), "an unattributed line inside the freshness window is FAIL"


def test_exited_pid_is_inconclusive_when_the_observation_aged_out(
    tmp_path: Path,
) -> None:
    live, exited = _exited_docs(incident_attempt=None)
    exited["attempts"][0]["observed_at"] = (
        "2026-10-23T23:50:00+00:00"  # 10 min before collection
    )
    with pytest.raises(VariantInconclusive, match="aged past"):
        mod.exited_pid(_cri_ctx("exited_pid", tmp_path, applog=_Applog([live, exited])))


# --------------------------------------------------------------------------- #
# old_pod
# --------------------------------------------------------------------------- #
def _old_pod_records(
    *, incident_attempt: str | None, **overrides: Any
) -> dict[str, Any]:
    return _records(
        attempts=[
            _attempt(
                "JOB-a001",
                containers=[_container(terminated=True, exit_code=1)],
                phase="FAILED",
                observed_at="2026-10-23T23:55:00+00:00",
            ),
            _attempt(
                "JOB-a001-r-11223344",
                containers=[_container(pod_uid=NEW_UID, host_pid=NEW_PID)],
            ),
        ],
        entries=[_entry(path=OLD_PATH, pod_uid=POD_UID)],
        incident_attempt=incident_attempt,
        **overrides,
    )


def test_old_pod_passes_when_the_line_stays_unresolved(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt=None,
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeApplogJob.instances[-1]
    assert (job.mode, job.restart_budget) == ("fail_once", 1)
    assert job.annotations == [("pod-1", NEW_PID)], "the new Pod declares its PID"
    assert result["attributed_attempt"] is None and result["old_pod_uid"] == POD_UID
    probe = ctx.probes[f"cri-pod-logs@{NODE}"]
    opened = next(call for call in probe.calls if call[0] == "open-window")  # type: ignore[attr-defined]
    assert opened[opened.index("--initial-tail-bytes") + 1] == str(
        mod.OLD_POD_INITIAL_TAIL
    )
    created = next(call for call in probe.calls if call[0] == "create-log")  # type: ignore[attr-defined]
    assert (
        "--allow-create-dir" in created
        and created[created.index("--pod-uid") + 1] == POD_UID
    )


def test_old_pod_passes_when_the_line_resolves_to_the_prior_attempt(
    tmp_path: Path,
) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt="JOB-a001",
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_old_pod_fails_when_the_new_attempt_is_blamed(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt="JOB-a001-r-11223344", parameters=_parameters(pid=NEW_PID)
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "FAIL"
    errors = result["errors"]
    assert any("attributed to the new attempt" in error for error in errors), errors
    assert any(f"PID {NEW_PID} was targeted" in error for error in errors), errors


def test_old_pod_is_inconclusive_without_a_restart(tmp_path: Path) -> None:
    records = _old_pod_records(incident_attempt=None)
    with pytest.raises(VariantInconclusive, match="did not restart"):
        mod.old_pod(_cri_ctx("old_pod", tmp_path, applog=_Applog([records])))


def test_old_pod_retries_a_file_kubelet_removed(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt=None,
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    probe = _logs_probe(pid=NEW_PID)
    # The first round's file is gone before the collector visits it (kubelet's
    # GC): nothing is ever delivered from it. The second round's file is read.
    probe.script["create-log"] = [
        {"file": {"path": OLD_PATH + ".lost"}},
        {"file": {"path": OLD_PATH}},
    ]
    ctx = _cri_ctx("old_pod", tmp_path, applog=_Applog([records]), probe=probe)
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    names = [call[0] for call in probe.calls]
    assert names.count("create-log") == 2 and names.count("write") == 2


# --------------------------------------------------------------------------- #
# cri_partial_rotation
# --------------------------------------------------------------------------- #
def _stitched_records(**overrides: Any) -> dict[str, Any]:
    ops = overrides.pop("ops", ("FREEZE_EVIDENCE",))
    action = overrides.pop("action", "NO_ACTION")
    return _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[_entry(message=PARTIAL_HEAD + PARTIAL_TAIL, path=ROTATED)],
        action=action,
        ops=ops,
        **overrides,
    )


def _partial_probe() -> FakeProbe:
    tracked = {"collector": {"tracked": {PATH: {"offset": 0}}, "pending": {}}}
    pending = {
        "collector": {
            "tracked": {PATH: {"offset": 60}},
            "pending": {
                PATH: {"cri": {"stdout": {"start": 0, "message": PARTIAL_HEAD}}}
            },
        }
    }
    return _logs_probe(statuses=[tracked, pending])


def test_cri_partial_rotation_passes_on_one_stitched_record(tmp_path: Path) -> None:
    ctx = _cri_ctx(
        "cri_partial_rotation",
        tmp_path,
        applog=_Applog([_stitched_records()]),
        probe=_partial_probe(),
    )
    result = mod.cri_partial_rotation(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    probe = ctx.probes[f"cri-pod-logs@{NODE}"]
    names = [call[0] for call in probe.calls]  # type: ignore[attr-defined]
    head = names.index("write")
    assert names[head + 1 :].index("status") >= 0
    assert names.index("rotate") < names.index("restart-collector") < len(names) - 1
    writes = [call for call in probe.calls if call[0] == "write"]  # type: ignore[attr-defined]
    assert writes[1][writes[1].index("--target") + 1] == "rotated"
    assert sorted(result["paths"]) == sorted([PATH, ROTATED])


def test_cri_partial_rotation_fails_when_fragments_are_delivered_apart(
    tmp_path: Path,
) -> None:
    records = _records(
        attempts=[_attempt("JOB-a001", containers=[_container()])],
        entries=[
            _entry(message=PARTIAL_HEAD, entry_id="1" * 64),
            _entry(message=PARTIAL_TAIL, path=ROTATED, entry_id="2" * 64),
        ],
        action=None,
    )
    ctx = _cri_ctx(
        "cri_partial_rotation",
        tmp_path,
        applog=_Applog([records]),
        probe=_partial_probe(),
    )
    result = mod.cri_partial_rotation(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "fragments were delivered on their own" in error for error in result["errors"]
    ), "a fragment alone is a stitching failure"
    assert any("exactly one" in error for error in result["errors"]), (
        "one stitched record is expected"
    )


def test_cri_partial_rotation_fails_on_a_diagnostic_or_wrong_time(
    tmp_path: Path,
) -> None:
    records = _stitched_records(
        ops=("FREEZE_EVIDENCE", "COLLECT_DIAGNOSTIC_BUNDLE"), action="RUN_DIAGNOSTICS"
    )
    records["entries"][0]["observed_at"] = "2026-10-24T00:00:36+00:00"
    ctx = _cri_ctx(
        "cri_partial_rotation",
        tmp_path,
        applog=_Applog([records]),
        probe=_partial_probe(),
    )
    result = mod.cri_partial_rotation(ctx)
    assert result["verdict"] == "FAIL"
    errors = result["errors"]
    assert any("ran a diagnostic workflow" in error for error in errors), errors
    assert any("not the P fragment's stamp" in error for error in errors), errors
    assert any("NO_ACTION" in error for error in errors), errors


def test_cri_partial_rotation_fails_when_the_fragment_was_never_held(
    tmp_path: Path,
) -> None:
    tracked = {"collector": {"tracked": {PATH: {"offset": 0}}, "pending": {}}}
    ctx = _cri_ctx(
        "cri_partial_rotation",
        tmp_path,
        applog=_Applog([_stitched_records()]),
        probe=_logs_probe(statuses=[tracked]),
    )
    result = mod.cri_partial_rotation(ctx)
    assert result["verdict"] == "FAIL"
    assert any(
        "never recorded the P fragment" in error for error in result["errors"]
    ), "the pending fragment in the collector's state is the oracle"


# --------------------------------------------------------------------------- #
# missing_flight_recorder (011's fakes, no flight recorder)
# --------------------------------------------------------------------------- #
def mfr_ctx(
    tmp_path: Path,
    *,
    ranks: list[dict[str, Any]],
    parameters: dict[str, Any] | None,
    bundle: dict[str, Any] | None = None,
):
    """``parameters`` go on the COLLECT_HUNG_TRIAGE steps, ``bundle`` on the
    COLLECT_DIAGNOSTIC_BUNDLE step (by default the conservative binding an
    UNDETERMINED decision leaves -- the live HyperPod record); ``parameters=None``
    leaves every step bare."""

    spec = _spec("missing_flight_recorder")
    base = make_settings(spec, node=NODE, peer=PEER)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = GeneralLiveSettings(
        **{**base.__dict__, "site_file": tmp_path / "site.yaml"}
    )

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        if script == nc.HUNG_PROBE:
            records = hung_records(arguments[-1], ranks=ranks)
            if parameters is not None:
                for step in records["workflows"]["wf"]["official_steps"]:
                    if step["operation"] == "COLLECT_HUNG_TRIAGE":
                        step["parameters"] = dict(parameters)
                    elif step["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE":
                        step["parameters"] = dict(
                            CONSERVATIVE_BUNDLE if bundle is None else bundle
                        )
            return records
        return _activity([], [])

    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=settings,
    )
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]
    return ctx


#: The finding's parameters as the hung DAG copies them onto its triage steps.
HUNG_PARAMETERS = {
    "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
    "detection_basis": "ATTEMPT_SCOPED_RDMA_IDLE",
    "capture_process_state": True,
}
#: The decision a stall without a flight recorder leaves (live HyperPod,
#: 2026-10-26): no FR timing, every rank alike, nothing to target.
UNDETERMINED_DECISION = {
    "classification": "UNDETERMINED",
    "confidence": "NONE",
    "reason": (
        "no communication root cause identified; inspect process observations "
        "and coverage"
    ),
    "culprit_ranks": [],
    "control_ranks": [],
    "candidate_count": 0,
    "candidate_process_count": 0,
    "targeted_capture_eligible": False,
    "temporal_verified": False,
    "flight_recorder_unavailable": True,
    "flight_recorder_missing_count": 16,
    "observed_rank_count": 16,
    "undetermined_nodes": [],
    "budget_exhausted": True,
}
#: The bundle step as ``bind_hung_bundle`` leaves it for that decision.
CONSERVATIVE_BUNDLE = {
    **HUNG_PARAMETERS,
    "capture_process_state": False,
    "hung_triage_decision": UNDETERMINED_DECISION,
    "hung_triage_target_pending": False,
    "target_pids_by_node": {},
    "max_processes": 0,
    "disable_strace": True,
    "operator_escalation_required": True,
    "bundle_scope": "bounded_triage_nodes",
}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_variants_match_the_coverage_row() -> None:
    expected = set(coverage_rows()["GF-GENERAL-LIVE-017"])
    assert {item.variant for item in mod.VARIANTS} == expected, (
        "017 registers every coverage variant"
    )
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == expected, "every 017 variant is executable"
    cri = {"live_pid", "exited_pid", "old_pod", "cri_partial_rotation"}
    for item in mod.VARIANTS:
        if item.variant in cri:
            assert item.approvals == ("ENABLE_NODE_LOG_COLLECTOR",), item.variant
            assert item.proof_mode == "live-replay" and item.lanes == ("eks", "hp")
            assert item.preflight is not None
            missing = GeneralLiveSettings(
                **{**make_settings(item).__dict__, "site_file": None}
            )
            assert item.preflight(missing) == ["a managed test job needs --site-file"]
    mfr = _spec("missing_flight_recorder")
    assert mfr.proof_mode == "native" and mfr.approvals == ()
    assert mfr.preflight is not None
    assert mfr.preflight(make_settings(mfr, node=NODE)) == [
        "a managed test job needs --site-file",
        "a two-node NCCL job needs --node-b",
    ]
