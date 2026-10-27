"""GF-GENERAL-LIVE-029 ``worker_restart``: the watcher replaced inside the
restart transaction, timed from the source failure.

The source-failure watch over a fake Pod/PyTorchJob API (the three end
signals, cadence, no store read, the live Running-to-absent shape, the
fixture-failure message), the settle bound against the product's constants,
the transaction timeline and the delete classification over scripted records,
the UID-fenced delete's exact request, the replacement wait and the graders;
then the variant end to end over a fake job, a fake inventory that ends the
source and flips the watcher on the fenced delete, and a store whose records
date the transaction around the delete -- every landing, both publish orders
and the INCONCLUSIVE rules. The last tests pin the approval wording and the
product facts the design rests on.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.completion_controller import KubernetesCompletionController
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
)
from gpu_fault.execution.dispatcher_wakeups import WAKING_REMOTE_COMMAND_STATUSES
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional import general_live_attempts as att
from scripts.e2e.regional import general_live_network_isolation as iso
from scripts.e2e.regional import general_live_watcher_restart as wr
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_registry import APPROVALS
from scripts.e2e.regional.general_live_store import ATTEMPT_PROBE
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
WATCHER_NODE = "ip-10-0-40-9.us-west-2.compute.internal"
T0 = datetime(2026, 10, 24, tzinfo=timezone.utc)
JOB = "gl29wr-deadbeef-1"
ATTEMPT = f"{JOB}-a001"
COPY = f"{ATTEMPT}-r-0a1b2c3d"
RUNNING_JOB = "Created,Running"
FAILED_JOB = "Created,Failed(PyTorchJobFailed)"


def _seconds(value: float) -> timedelta:
    return timedelta(seconds=value)


def _workflow(
    request_id: str,
    op: str,
    statuses: list[str],
    status: str = "SUCCEEDED",
    *,
    claimed_at: datetime | None = None,
    updated_at: datetime | None = None,
    settled_at: datetime | None = None,
) -> dict[str, Any]:
    events = []
    if claimed_at is not None:
        events.append({"kind": "CLAIM", "at": claimed_at.isoformat()})
        events.append(
            {
                "kind": "STEP_ATTEMPT",
                "at": (claimed_at + timedelta(milliseconds=400)).isoformat(),
                "operation": op,
                "status": statuses[0] if statuses else None,
            }
        )
    return {
        "request_id": request_id,
        "status": status,
        "official_steps": [{"operation": op, "parameters": {}}],
        "step_executions": [
            {
                "step_index": 0,
                "operation": op,
                "status": item,
                "details": {},
                **({"updated_at": updated_at.isoformat()} if updated_at else {}),
            }
            for item in statuses
        ],
        "events": events,
        **({"updated_at": settled_at.isoformat()} if settled_at else {}),
    }


def _entry(
    *,
    restart: list[str] | None,
    recovery_status: str = "SUCCEEDED",
    decision: str = "PLAN_CREATED",
    stops: int = 1,
    recoveries: int = 1,
    ended_at: datetime | None = None,
    published_at: datetime | None = None,
    decided_at: datetime | None = None,
    claimed_at: datetime | None = None,
    closed_at: datetime | None = None,
    settled_at: datetime | None = None,
) -> dict[str, Any]:
    containment = _workflow("wf-c", "STOP_WORKLOADS", ["SUCCEEDED"] * stops)
    recovery = []
    if restart is not None:
        for index in range(recoveries):
            recovery.append(
                {
                    "incident": {
                        "incident_id": f"inc-r{index}",
                        "job_id": JOB,
                        "state": "RECOVERED",
                        **(
                            {"created_at": decided_at.isoformat()} if decided_at else {}
                        ),
                    },
                    "workflow": _workflow(
                        f"wf-r{index}",
                        "RESTART_WORKLOAD",
                        restart,
                        recovery_status,
                        claimed_at=claimed_at,
                        updated_at=closed_at,
                        settled_at=settled_at,
                    ),
                }
            )
    return {
        "event": {
            "terminal_status": "FAILED",
            **({"ended_at": ended_at.isoformat()} if ended_at else {}),
        },
        "decision": {"status": decision},
        "containment_incident": {
            "incident_id": "inc-c",
            "job_id": JOB,
            "state": "RECOVERED",
            **({"created_at": published_at.isoformat()} if published_at else {}),
        },
        "containment_workflows": [containment],
        "recovery": recovery,
    }


def _records(
    entry: dict[str, Any], *, count: int = 1, reservations: list[str] | None = None
) -> dict[str, Any]:
    if reservations is None:
        reservations = ["wf-r0/0/RESTART_WORKLOAD"] * count
    return {
        "attempts": {ATTEMPT: entry},
        "restart_budget": {
            "restart_count": count,
            "reservation_ids": reservations,
            "budget": 1,
        },
    }


def _dated(
    anchor: datetime, offsets: dict[str, float], **changes: Any
) -> dict[str, Any]:
    """Records whose transaction markers sit at ``offsets`` seconds from ``anchor``."""

    entry = _entry(
        restart=changes.pop("restart", ["WAITING", "SUCCEEDED"]),
        ended_at=anchor + _seconds(offsets["ended"]),
        published_at=anchor + _seconds(offsets["published"]),
        decided_at=anchor + _seconds(offsets["decided"]),
        claimed_at=anchor + _seconds(offsets["claimed"]),
        closed_at=anchor + _seconds(offsets["closed"]),
        settled_at=anchor + _seconds(offsets["settled"]),
    )
    return _records(entry, **changes)


#: Transaction markers relative to the delete request's first clock reading
#: (the bracket is [0, 1] s under the fake clock), one set per landing.
LANDINGS: dict[str, dict[str, float]] = {
    wr.BEFORE_PUBLISH: {
        "ended": -2,
        "published": 4,
        "decided": 8,
        "claimed": 9,
        "closed": 12,
        "settled": 12,
    },
    wr.BETWEEN_PUBLISH_AND_RESTART: {
        "ended": -4,
        "published": -3,
        "decided": -1,
        "claimed": 2,
        "closed": 5,
        "settled": 5,
    },
    wr.AFTER_RESTART_BEFORE_SETTLE: {
        "ended": -12,
        "published": -10,
        "decided": -8,
        "claimed": -7,
        "closed": -2,
        "settled": 6,
    },
    wr.AFTER_SETTLE: {
        "ended": -32,
        "published": -30,
        "decided": -25,
        "claimed": -24,
        "closed": -20,
        "settled": -15,
    },
}
INSIDE = [
    wr.BEFORE_PUBLISH,
    wr.BETWEEN_PUBLISH_AND_RESTART,
    wr.AFTER_RESTART_BEFORE_SETTLE,
]


def _spec():
    return next(spec for spec in att.VARIANTS if spec.variant == "worker_restart")


def _context(tmp_path: Path, regional: Any):
    spec = _spec()
    settings = make_settings(spec, node=NODE)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(settings)(
        **{**settings.__dict__, "site_file": tmp_path / "site.yaml"}
    )
    return make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )


# --------------------------------------------------------------------------- #
# the fake GPU cluster: the job's Pods and PyTorchJob, the watcher inventory
# --------------------------------------------------------------------------- #
def _pod(uid: str, *, ready: bool = True, rv: str = "100") -> dict[str, Any]:
    return {
        "metadata": {
            "name": f"gpu-fault-completion-watcher-{uid}",
            "uid": uid,
            "resourceVersion": rv,
        },
        "spec": {"nodeName": WATCHER_NODE, "containers": [{"name": "watcher"}]},
        "status": {
            "phase": "Running",
            "podIP": "10.0.40.77",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [
                {
                    "name": "watcher",
                    "ready": ready,
                    "started": True,
                    "state": {"running": {}},
                }
            ],
        },
    }


def _labels(job_id: str, attempt_id: str) -> dict[str, str]:
    return {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": job_id,
        "gpu-fault.io/attempt-id": attempt_id,
    }


def _pod_item(
    job_id: str, attempt_id: str, phase: str, *, exit_code: int | None = None
) -> dict[str, Any]:
    """A Pod as ``kubectl get pod,pytorchjob -o json`` lists it (``kind`` set)."""

    suffix = "master-0" if attempt_id == f"{job_id}-a001" else "r-master-0"
    state: dict[str, Any] = (
        {
            "terminated": {
                "exitCode": exit_code,
                "reason": "Error" if exit_code else "Completed",
                "finishedAt": "2026-10-24T00:00:30Z",
            }
        }
        if exit_code is not None
        else ({"running": {}} if phase == "Running" else {"waiting": {}})
    )
    return {
        "kind": "Pod",
        "metadata": {
            "name": f"gpu-fault-single-node-{suffix}",
            "uid": f"pod-{attempt_id}",
            "labels": _labels(job_id, attempt_id),
        },
        "spec": {"nodeName": NODE},
        "status": {
            "phase": phase,
            "containerStatuses": [
                {"name": "pytorch", "ready": phase == "Running", "state": state}
            ],
        },
    }


def _conditions(spec: str) -> list[dict[str, Any]]:
    """``"Created,Failed(PyTorchJobFailed)"`` as training-operator conditions."""

    conditions = []
    for part in spec.split(","):
        kind, _, reason = part.partition("(")
        conditions.append(
            {
                "type": kind,
                "status": "True",
                # The operator stamps a reason on every condition; the vectors
                # spell it only where a test reads it, so the rendered state
                # stays ``Created,Failed(PyTorchJobFailed)``.
                "reason": reason.rstrip(")") or None,
                "message": f"PyTorchJob is {kind.lower()}.",
                "lastUpdateTime": "2026-10-24T00:00:31Z",
                "lastTransitionTime": "2026-10-24T00:00:31Z",
            }
        )
    return conditions


def _job_item(job_id: str, attempt_id: str, conditions: str) -> dict[str, Any]:
    """The PyTorchJob as the same listing shows it."""

    copy = attempt_id != f"{job_id}-a001"
    return {
        "kind": "PyTorchJob",
        "metadata": {
            "name": "gpu-fault-single-node-warm-spare-25496919f6"
            + ("-r-e5a41535" if copy else ""),
            "uid": f"job-{attempt_id}",
            "labels": _labels(job_id, attempt_id),
        },
        "status": {
            "conditions": _conditions(conditions),
            "replicaStatuses": {
                "Master": {"failed": 1} if "Failed" in conditions else {"active": 1}
            },
        },
    }


def _frame(
    pod: str | None,
    job: str | None,
    *,
    exit_code: int | None = None,
    copy: bool = False,
) -> dict[str, Any]:
    """What one read of the job's objects shows: the source Pod's phase (None =
    not listed), the source PyTorchJob's held conditions (None = not listed),
    the source container's exit code, and whether the restart copy is listed."""

    return {"pod": pod, "job": job, "exit_code": exit_code, "copy": copy}


#: The source as the API shows it on successive reads; the last frame repeats.
DEFAULT_FRAMES = [
    _frame("Pending", "Created"),
    _frame("Running", RUNNING_JOB),
    _frame("Failed", FAILED_JOB, exit_code=1, copy=True),
]
#: HyperPod, 2026-10-27 12:40: Running, then gone -- the containment's STOP
#: deleted the Pod before a read could list its terminated state; the job's
#: Failed condition is what remains.
LIVE_SHAPE = [
    _frame("Running", RUNNING_JOB),
    _frame("Running", RUNNING_JOB),
    _frame(None, FAILED_JOB, copy=True),
]
RUNNING_THEN_ABSENT_NO_JOB = [_frame("Running", RUNNING_JOB), _frame(None, None)]
RUNNING_FAILED_ABSENT = [
    _frame("Running", RUNNING_JOB),
    _frame("Failed", FAILED_JOB, exit_code=1),
    _frame(None, FAILED_JOB, copy=True),
]
NEVER_RUNNING_JOB_FAILED = [_frame("Pending", "Created"), _frame(None, FAILED_JOB)]
NEVER_ENDS = [_frame("Pending", "Created"), _frame("Running", RUNNING_JOB)]


class Cluster:
    """A fake GPU cluster: the source ends as its frames say, the watcher
    inventory flips on the fenced delete."""

    def __init__(self) -> None:
        self.pods = [_pod("uid-old")]
        self.deletes: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.polls_before_ready = 1
        self.replacement_appears = True
        self.on_delete: Any = None
        self.frames = list(DEFAULT_FRAMES)
        self.source_reads = 0
        self.source_reads_at_delete: int | None = None

    def source_items(self, job_id: str) -> list[dict[str, Any]]:
        """The job's Pods and PyTorchJobs as the API lists them for ``job_id``
        (the variant's own run identity, not a constant)."""

        self.source_reads += 1
        attempt_id = f"{job_id}-a001"
        frame = self.frames[min(self.source_reads, len(self.frames)) - 1]
        items: list[dict[str, Any]] = []
        if frame["pod"] is not None:
            items.append(
                _pod_item(
                    job_id, attempt_id, frame["pod"], exit_code=frame["exit_code"]
                )
            )
        if frame["job"] is not None:
            items.append(_job_item(job_id, attempt_id, frame["job"]))
        if frame["copy"]:
            copy = f"{attempt_id}-r-0a1b2c3d"
            items.append(_pod_item(job_id, copy, "Running"))
            items.append(_job_item(job_id, copy, RUNNING_JOB))
        return items

    def kubectl(
        self, plane: str, arguments: tuple[str, ...], options: dict[str, Any]
    ) -> str:
        if arguments[:1] == ("get",):
            selector = arguments[3] if len(arguments) > 3 else ""
            if selector.startswith("gpu-fault.io/job-id="):
                assert arguments == (
                    "get",
                    "pod,pytorchjob",
                    "-l",
                    selector,
                    "-o",
                    "json",
                )
                job_id = selector.split("=", 1)[1]
                return json.dumps({"items": self.source_items(job_id)})
            assert arguments[:2] == ("get", "pod"), arguments
            assert selector == f"app={iso.WATCHER_APP}", selector
            if self.polls_before_ready > 0 and any(
                p["metadata"]["uid"] == "uid-new" for p in self.pods
            ):
                self.polls_before_ready -= 1
                return json.dumps({"items": [_pod("uid-new", ready=False)]})
            return json.dumps({"items": self.pods})
        if arguments[:2] == ("delete", "--raw"):
            body = json.loads(options["input_text"])
            self.deletes.append((arguments, body))
            assert body["preconditions"]["uid"] == self.pods[0]["metadata"]["uid"], (
                "fenced on the live UID"
            )
            self.source_reads_at_delete = self.source_reads
            self.pods = [_pod("uid-new", rv="200")] if self.replacement_appears else []
            if self.on_delete is not None:
                self.on_delete()
            return ""
        raise AssertionError(f"unexpected kubectl {arguments}")


class Regional(FakeRegional):
    def __init__(self, cluster: Cluster, **options: Any) -> None:
        super().__init__(**options)
        self.cluster = cluster

    def kubectl(self, plane: str, *arguments: str, **options: Any) -> str:
        return self.cluster.kubectl(plane, arguments, options)


# --------------------------------------------------------------------------- #
# the settle bound and the source-failure watch
# --------------------------------------------------------------------------- #
def test_settle_bound_is_derived_from_the_product_constants() -> None:
    executor = ProductionExecutorConfig(
        enabled=False, executor_id="pin", allowed_operations=frozenset()
    )
    bound = wr.settle_bound()
    assert bound == {
        "replacement_timeout_seconds": float(wr.REPLACEMENT_TIMEOUT_SECONDS),
        "stop_workloads_cap_seconds": float(
            executor.step_waiting_limit(WorkflowOperation.STOP_WORKLOADS)
        ),
        "restart_workload_cap_seconds": float(
            executor.step_waiting_limit(WorkflowOperation.RESTART_WORKLOAD)
        ),
        "dispatcher_polls_seconds": 2
        * WorkflowDispatcherConfig(enabled=False).poll_interval_seconds,
        "watcher_poll_seconds": wr.WATCHER_POLL_SECONDS,
        "margin_seconds": wr.SETTLE_MARGIN_SECONDS,
        "total_seconds": 1935.0,
    }
    # The watcher poll the harness names is the controller's own default.
    parameters = inspect.signature(KubernetesCompletionController.__init__).parameters
    assert parameters["poll_interval_seconds"].default == wr.WATCHER_POLL_SECONDS
    assert 0 <= wr.DELETE_OFFSET_SECONDS <= 2, "the delete follows the failure at once"
    assert wr.SOURCE_POLL_SECONDS == 1.0 and wr.SOURCE_START_POLL_SECONDS == 5.0
    assert bound["total_seconds"] <= _spec().max_minutes * 60 - 900, (
        "the settle bound fits the variant's window beside the job start and the "
        "watcher-restored wait"
    )
    # The restart copy outlives a warm replacement start and still ends inside
    # the watcher-restored wait that starts when the replacement is Ready.
    assert 60 < wr.COPY_RUN_SECONDS < att.WATCHER_RESTORED_WAIT_SECONDS


def test_source_objects_reads_the_attempt_pods_and_job_in_one_listing() -> None:
    cluster = Cluster()
    cluster.frames = [DEFAULT_FRAMES[-1]]
    objects = wr.source_objects(Regional(cluster), JOB, ATTEMPT)
    assert [pod["name"] for pod in objects["pods"]] == [
        "gpu-fault-single-node-master-0"
    ], "the restart copy carries another attempt id and is not the source"
    (pod,) = objects["pods"]
    assert pod["ended"] is True and pod["phase"] == "Failed"
    assert pod["containers"] == [
        {
            "name": "pytorch",
            "terminated": True,
            "exit_code": 1,
            "reason": "Error",
            "finished_at": "2026-10-24T00:00:30Z",
        }
    ]
    job = objects["job"]
    assert job["name"] == "gpu-fault-single-node-warm-spare-25496919f6"
    assert job["ended"] is True
    assert [item["type"] for item in job["conditions"]] == ["Created", "Failed"]
    assert job["conditions"][1]["reason"] == "PyTorchJobFailed"
    assert job["conditions"][1]["last_transition_time"] == "2026-10-24T00:00:31Z"
    assert job["replica_statuses"] == {"Master": {"failed": 1}}
    assert wr.source_pods(Regional(cluster), JOB, ATTEMPT) == objects["pods"]
    cluster.frames = [_frame("Running", RUNNING_JOB)]
    running = wr.source_objects(Regional(cluster), JOB, ATTEMPT)
    assert running["pods"][0]["ended"] is False and running["job"]["ended"] is False
    cluster.frames = [_frame(None, None)]
    assert wr.source_objects(Regional(cluster), JOB, ATTEMPT) == {
        "pods": [],
        "job": None,
    }


@pytest.mark.parametrize(
    ("frames", "trigger", "signals", "timeline", "running_seen"),
    [
        (
            LIVE_SHAPE,
            wr.JOB_ENDED,
            [wr.JOB_ENDED, wr.POD_ABSENT],
            [
                ("Running", RUNNING_JOB, ["pod", "job"]),
                ("absent", FAILED_JOB, ["pod", "job"]),
            ],
            True,
        ),
        (
            RUNNING_THEN_ABSENT_NO_JOB,
            wr.POD_ABSENT,
            [wr.POD_ABSENT],
            [
                ("Running", RUNNING_JOB, ["pod", "job"]),
                ("absent", "absent", ["pod", "job"]),
            ],
            True,
        ),
        (
            RUNNING_FAILED_ABSENT,
            wr.POD_TERMINATED,
            [wr.POD_TERMINATED, wr.JOB_ENDED],
            [
                ("Running", RUNNING_JOB, ["pod", "job"]),
                ("Failed(exit 1)", FAILED_JOB, ["pod", "job"]),
            ],
            True,
        ),
        (
            NEVER_RUNNING_JOB_FAILED,
            wr.JOB_ENDED,
            [wr.JOB_ENDED],
            [
                ("Pending", "Created", ["pod", "job"]),
                ("absent", FAILED_JOB, ["pod", "job"]),
            ],
            False,
        ),
    ],
    ids=[
        "running-absent-live",
        "running-absent-no-job",
        "running-failed-absent",
        "never-running-job-failed",
    ],
)
def test_wait_source_failure_takes_the_earliest_end_signal(
    tmp_path: Path,
    frames: list[dict[str, Any]],
    trigger: str,
    signals: list[str],
    timeline: list[tuple[str, str, list[str]]],
    running_seen: bool,
) -> None:
    cluster = Cluster()
    cluster.frames = frames
    ctx = _context(tmp_path, Regional(cluster))
    failure = wr.wait_source_failure(ctx, job_id=JOB, attempt_id=ATTEMPT)
    assert failure["trigger"] == trigger
    assert failure["signals"] == signals
    ending = next(
        index
        for index, frame in enumerate(frames, start=1)
        if frame["exit_code"] is not None
        or (frame["job"] or "").find("Failed") >= 0
        or (frame["pod"] is None and index > 1)
    )
    assert failure["polls"] == ending, "the delete follows the first ending read"
    assert (failure["last_running_at"] is not None) == running_seen
    assert failure["pod_state"] == timeline[-1][0]
    assert failure["job_state"] == timeline[-1][1]
    written = json.loads(
        (ctx.evidence.case_dir / "source-failure-timeline.json").read_text()
    )
    assert [
        (item["pod"], item["job"], item["changed"]) for item in written["entries"]
    ] == timeline, "every transition of either side, with its source"
    if frames[ending - 1]["pod"] is None:
        assert failure["pods"] == [], "nothing listed: the Pod is gone"
    else:
        assert failure["pods"][0]["containers"][0]["exit_code"] == 1
    if frames[ending - 1]["job"] is not None:
        assert failure["job"]["ended"] is True
        assert failure["job"]["conditions"][-1]["reason"] == "PyTorchJobFailed"
    else:
        assert failure["job"] is None


def test_wait_source_failure_polls_every_five_seconds_until_a_pod_is_listed(
    tmp_path: Path,
) -> None:
    cluster = Cluster()
    cluster.frames = [
        _frame(None, None),
        _frame("Pending", "Created"),
        _frame("Running", RUNNING_JOB),
        _frame("Running", RUNNING_JOB),
        _frame("Failed", FAILED_JOB, exit_code=1),
    ]
    ctx = _context(tmp_path, Regional(cluster))
    sleeps: list[float] = []
    ctx.sleep = sleeps.append
    failure = wr.wait_source_failure(ctx, job_id=JOB, attempt_id=ATTEMPT)
    assert sleeps == [
        wr.SOURCE_START_POLL_SECONDS,
        wr.SOURCE_POLL_SECONDS,
        wr.SOURCE_POLL_SECONDS,
        wr.SOURCE_POLL_SECONDS,
    ], "five seconds while nothing is listed, one second from the first Pod on"
    assert failure["polls"] == 5 and failure["waited_seconds"] == 5.0
    assert failure["trigger"] == wr.POD_TERMINATED
    written = json.loads(
        (ctx.evidence.case_dir / "source-failure-timeline.json").read_text()
    )
    assert [item["pod"] for item in written["entries"]] == [
        "absent",
        "Pending",
        "Running",
        "Failed(exit 1)",
    ], "only transitions are written; the repeated Running read is not"
    assert written["entries"][0]["changed"] == ["pod", "job"]


def test_wait_source_failure_is_a_fixture_failure_naming_both_last_states(
    tmp_path: Path,
) -> None:
    cluster = Cluster()
    cluster.frames = NEVER_ENDS
    ctx = _context(tmp_path, Regional(cluster))
    with pytest.raises(RegionalFixtureError) as raised:
        wr.wait_source_failure(ctx, job_id=JOB, attempt_id=ATTEMPT, timeout_seconds=30)
    assert str(raised.value) == (
        f"the source attempt {ATTEMPT} did not end within 30 s (last Pod state "
        f"Running; last job condition {RUNNING_JOB})"
    )
    assert cluster.deletes == []


# --------------------------------------------------------------------------- #
# dating the transaction and the delete
# --------------------------------------------------------------------------- #
def test_transaction_timeline_dates_the_markers_from_the_records() -> None:
    anchor = T0 + _seconds(100)
    offsets = LANDINGS[wr.BETWEEN_PUBLISH_AND_RESTART]
    records = _dated(anchor, offsets)
    records["restart_budget"]["updated_at"] = (anchor + _seconds(2)).isoformat()
    assert wr.transaction_timeline(records, ATTEMPT) == {
        "source_ended_at": (anchor + _seconds(-4)).isoformat(),
        "published_at": (anchor + _seconds(-3)).isoformat(),
        "decided_at": (anchor + _seconds(-1)).isoformat(),
        "reserved_at": (anchor + _seconds(2)).isoformat(),
        "dispatched_at": (anchor + _seconds(2.4)).isoformat(),
        "restart_closed_at": (anchor + _seconds(5)).isoformat(),
        "settled_at": (anchor + _seconds(5)).isoformat(),
        "claim_open_seconds": 3.0,
        "transaction_seconds": 8.0,
        "ledger_updated_at": (anchor + _seconds(2)).isoformat(),
    }
    # A recovery still running dates no settle and no restart end.
    open_records = _records(
        _entry(
            restart=["WAITING"],
            recovery_status="RUNNING",
            published_at=anchor,
            claimed_at=anchor + _seconds(3),
            settled_at=anchor + _seconds(4),
        )
    )
    timeline = wr.transaction_timeline(open_records, ATTEMPT)
    assert (timeline["restart_closed_at"], timeline["settled_at"]) == (None, None)
    assert timeline["reserved_at"] == (anchor + _seconds(3)).isoformat()
    undated = wr.transaction_timeline(_records(_entry(restart=["SUCCEEDED"])), ATTEMPT)
    assert all(value is None for value in undated.values()), undated
    assert wr.transaction_timeline({"attempts": {}}, ATTEMPT)["published_at"] is None


def _bracket(
    anchor: datetime, before: float = 0.0, after: float = 1.0
) -> dict[str, str]:
    return {
        "deleted_at": (anchor + _seconds(before)).isoformat(),
        "deletion_returned_at": (anchor + _seconds(after)).isoformat(),
    }


@pytest.mark.parametrize("landed", list(LANDINGS))
def test_classify_delete_places_the_bracket_in_the_transaction(landed: str) -> None:
    anchor = T0 + _seconds(100)
    timeline = wr.transaction_timeline(_dated(anchor, LANDINGS[landed]), ATTEMPT)
    landing = wr.classify_delete(timeline, _bracket(anchor))
    assert landing["landed"] == landed
    assert landing["after_publish_seconds"] == -LANDINGS[landed]["published"]
    assert landing["before_settle_seconds"] == LANDINGS[landed]["settled"] - 1.0
    assert landing["detail"].startswith(
        f"the delete request ran {anchor.isoformat()} -> "
        f"{(anchor + _seconds(1)).isoformat()}, "
    ), landing["detail"]
    assert f"settled_at {timeline['settled_at']}" in landing["detail"]
    assert (wr.landing_inconclusive_reason(landing) is None) == (
        landed != wr.AFTER_SETTLE
    )


def test_classify_delete_is_conservative_at_the_settle_and_without_dates() -> None:
    anchor = T0 + _seconds(100)
    offsets = {**LANDINGS[wr.AFTER_SETTLE], "settled": 0.5}
    timeline = wr.transaction_timeline(_dated(anchor, offsets), ATTEMPT)
    straddle = wr.classify_delete(timeline, _bracket(anchor))
    assert straddle["landed"] == wr.AFTER_SETTLE, (
        "a bracket straddling the settle may have acted on a finished transaction"
    )
    reason = wr.landing_inconclusive_reason(straddle)
    assert reason is not None and reason.startswith(
        "the watcher was replaced after the restart transaction had settled: "
    )
    # Before the publish on the single stamp of a document without a bracket.
    single = wr.classify_delete(
        wr.transaction_timeline(_dated(anchor, LANDINGS[wr.BEFORE_PUBLISH]), ATTEMPT),
        {"deleted_at": anchor.isoformat()},
    )
    assert single["landed"] == wr.BEFORE_PUBLISH
    assert single["deletion_returned_at"] == anchor.isoformat()
    undated = wr.classify_delete(
        wr.transaction_timeline(_records(_entry(restart=["SUCCEEDED"])), ATTEMPT),
        _bracket(anchor),
    )
    assert undated["landed"] == wr.UNDATED
    assert "do not date the transaction's publish" in undated["detail"]
    assert wr.landing_inconclusive_reason(undated) is None, "the graders judge"
    no_document = wr.classify_delete(timeline, {})
    assert no_document["landed"] == wr.UNDATED
    assert no_document["detail"].startswith("the restart document does not date"), (
        no_document["detail"]
    )


# --------------------------------------------------------------------------- #
# the fenced delete and the replacement
# --------------------------------------------------------------------------- #
def test_delete_watcher_pod_sends_a_uid_and_version_fenced_raw_delete() -> None:
    cluster = Cluster()
    regional = Regional(cluster)
    pod = iso.watcher_pod(regional)
    wr.delete_watcher_pod(regional, pod)
    arguments, body = cluster.deletes[-1]
    assert arguments == (
        "delete",
        "--raw",
        f"/api/v1/namespaces/gpu-fault-system/pods/{pod['name']}",
        "-f",
        "-",
    )
    assert body == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": "uid-old", "resourceVersion": "100"},
        "propagationPolicy": "Foreground",
        "gracePeriodSeconds": 0,
    }
    with pytest.raises(RegionalFixtureError, match="UID/resourceVersion is unknown"):
        wr.delete_watcher_pod(
            regional, {"name": "x", "uid": "", "resource_version": ""}
        )


def test_restart_watcher_waits_for_a_ready_pod_with_a_new_uid(tmp_path: Path) -> None:
    cluster = Cluster()
    cluster.polls_before_ready = 2
    ctx = _context(tmp_path, Regional(cluster))
    document = wr.restart_watcher(ctx, source_failure={"polls": 3})
    assert (
        document["deleted"]["uid"] == "uid-old"
        and document["replacement"]["uid"] == "uid-new"
    )
    assert document["replacement"]["ready"] is True
    deleted_at = datetime.fromisoformat(document["deleted_at"])
    returned_at = datetime.fromisoformat(document["deletion_returned_at"])
    assert deleted_at < returned_at < datetime.fromisoformat(document["ready_at"]), (
        "the delete request is bracketed by two clock readings before the "
        "replacement wait"
    )
    assert document["source_failure"] == {"polls": 3}
    assert document["delete_offset_seconds"] == wr.DELETE_OFFSET_SECONDS
    assert wr.replacement_errors(document) == []
    assert (ctx.evidence.case_dir / "watcher-restart.json").is_file(), (
        "the restart document is evidence"
    )
    assert wr.replacement_errors(
        {"deleted": {"uid": "a"}, "replacement": {"uid": "a", "ready": True}}
    ) == ["the watcher Pod was not replaced by a new UID"]
    assert wr.replacement_errors(
        {"deleted": {"uid": "a"}, "replacement": {"uid": "b", "ready": False}}
    ) == ["the replacement watcher Pod is not Ready"]


def test_wait_replacement_gives_up_after_the_timeout(tmp_path: Path) -> None:
    cluster = Cluster()
    ctx = _context(tmp_path, Regional(cluster))
    with pytest.raises(RegionalFixtureError, match="no Ready replacement"):
        wr.wait_replacement(ctx, "uid-old", timeout_seconds=30)


def test_recovery_settled_needs_every_recovery_workflow_terminal() -> None:
    assert (
        wr.recovery_settled(_records(_entry(restart=None), count=0), ATTEMPT) is False
    )
    assert (
        wr.recovery_settled(
            _records(_entry(restart=["WAITING"], recovery_status="RUNNING")), ATTEMPT
        )
        is False
    )
    assert wr.recovery_settled(_records(_entry(restart=["SUCCEEDED"])), ATTEMPT) is True
    assert (
        wr.recovery_settled(
            _records(_entry(restart=["FAILED"], recovery_status="FAILED")), ATTEMPT
        )
        is True
    )


# --------------------------------------------------------------------------- #
# graders
# --------------------------------------------------------------------------- #
def test_worker_restart_errors_pin_one_restart_and_the_kept_reservation() -> None:
    good = _entry(restart=["WAITING", "SUCCEEDED"])
    assert wr.worker_restart_errors(good, _records(good)) == []
    lost = _records(good, count=0, reservations=[])
    assert (
        "the restart reservation was lost (ledger shows no consumed restart)"
        in wr.worker_restart_errors(good, lost)
    )
    doubled = _records(
        good,
        count=2,
        reservations=["wf-r0/0/RESTART_WORKLOAD", "wf-r0/safety/0/RESTART_WORKLOAD"],
    )
    errors = wr.worker_restart_errors(good, doubled)
    assert any("double decrement" in item for item in errors), errors
    foreign = _records(good, reservations=["wf-other/0/RESTART_WORKLOAD"])
    errors = wr.worker_restart_errors(good, foreign)
    assert any("does not belong to the recovery workflow" in item for item in errors), (
        errors
    )
    twice = _entry(restart=["SUCCEEDED", "SUCCEEDED"])
    assert (
        "RESTART_WORKLOAD succeeded 2 times, expected once"
        in wr.worker_restart_errors(twice, _records(twice))
    )
    two_recoveries = _entry(restart=["SUCCEEDED"], recoveries=2)
    assert "expected one recovery workflow, found 2" in wr.worker_restart_errors(
        two_recoveries, _records(two_recoveries)
    )
    no_action = _entry(restart=["SUCCEEDED"], decision="NO_ACTION")
    errors = wr.worker_restart_errors(no_action, _records(no_action))
    assert any("expected a budgeted restart" in item for item in errors), errors
    failed = _entry(restart=["FAILED"], recovery_status="FAILED")
    errors = wr.worker_restart_errors(failed, _records(failed))
    assert "RESTART_WORKLOAD succeeded 0 times, expected once" in errors
    assert "recovery workflow wf-r0 ended FAILED" in errors
    assert "no restart-budget ledger row for the job" in wr.worker_restart_errors(
        good, {"attempts": {ATTEMPT: good}, "restart_budget": None}
    )


def test_attempt_once_errors_need_exactly_one_restart_attempt() -> None:
    source = {"attempt_id": ATTEMPT, "job_id": JOB, "workload_phase": "FAILED"}
    restart = {"attempt_id": COPY, "job_id": JOB, "workload_phase": "RUNNING"}
    assert (
        wr.attempt_once_errors([source, restart], job_id=JOB, source_attempt_id=ATTEMPT)
        == []
    )
    assert wr.attempt_once_errors([source], job_id=JOB, source_attempt_id=ATTEMPT) == [
        "no restart attempt of the job was observed by the watcher"
    ]
    assert wr.attempt_once_errors([restart], job_id=JOB, source_attempt_id=ATTEMPT) == [
        f"the watcher never published an observation of {ATTEMPT}"
    ]
    third = {
        "attempt_id": f"{ATTEMPT}-r-ffffffff",
        "job_id": JOB,
        "workload_phase": "RUNNING",
    }
    several = wr.attempt_once_errors(
        [source, restart, third], job_id=JOB, source_attempt_id=ATTEMPT
    )
    assert several[0].startswith("2 restart attempts were observed"), several
    other = {"attempt_id": "other-a001", "job_id": "other", "workload_phase": "RUNNING"}
    assert (
        wr.attempt_once_errors(
            [source, restart, other], job_id=JOB, source_attempt_id=ATTEMPT
        )
        == []
    )


def test_attempt_once_passes_when_the_replacement_published_the_source_terminal() -> (
    None
):
    """BEFORE_PUBLISH: the deleted watcher never published the source; the
    replacement publishes its terminal from the outbox after the delete and
    later observes the copy -- the same two attempt ids, one restart."""

    deleted_at = T0 + _seconds(100)
    observations = [
        {
            "attempt_id": ATTEMPT,
            "job_id": JOB,
            "workload_phase": "FAILED",
            "observed_at": (deleted_at + _seconds(6)).isoformat(),
            "watcher_instance": "w2",
        },
        {
            "attempt_id": COPY,
            "job_id": JOB,
            "workload_phase": "RUNNING",
            "observed_at": (deleted_at + _seconds(40)).isoformat(),
            "watcher_instance": "w2",
        },
    ]
    assert (
        wr.attempt_once_errors(observations, job_id=JOB, source_attempt_id=ATTEMPT)
        == []
    )
    # And the other order: the deleted watcher published the source before it went.
    observations[0]["observed_at"] = (deleted_at - _seconds(3)).isoformat()
    observations[0]["watcher_instance"] = "w1"
    assert (
        wr.attempt_once_errors(observations, job_id=JOB, source_attempt_id=ATTEMPT)
        == []
    )


# --------------------------------------------------------------------------- #
# the variant end to end over fakes
# --------------------------------------------------------------------------- #
class FakeFailJob:
    instances: list["FakeFailJob"] = []

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        mode: Any,
        restart_budget: int = 1,
        gpu_count: int = 8,
        ranks: int = 1,
        copy_run_seconds: int | None = None,
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.restart_budget = restart_budget
        self.copy_run_seconds = copy_run_seconds
        self.fixture = type("F", (), {"delete": staticmethod(lambda: None)})()
        self.prewarm = type("P", (), {"cleanup": staticmethod(lambda: None)})()
        FakeFailJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def delete(self) -> None:
        pass


class Store:
    """The completion records, dated around the delete.

    ``anchor`` is the delete request's first clock reading (set by the fenced
    delete); the transaction markers sit at ``offsets`` seconds from it, so
    the same scripted store yields every landing. A read before the delete
    is counted: the variant must not read the store before it deletes.
    """

    def __init__(self) -> None:
        self.anchor: datetime | None = None
        self.offsets = dict(LANDINGS[wr.BETWEEN_PUBLISH_AND_RESTART])
        self.reads_before_delete = 0
        self.restart_statuses = ["WAITING", "SUCCEEDED"]
        self.count = 1
        self.reservations: list[str] | None = None
        self.restart_observed = 1
        # The source observation's publisher: the deleted watcher, before the
        # delete (negative), or the replacement, after it (positive).
        self.source_observed_offset = -3.0
        self.notifications = [
            {
                "incident_id": "inc-c",
                "category": "containment",
                "deduplication_key": "k1",
                "created_at": T0.isoformat(),
                "status": "SENT",
            },
            {
                "incident_id": "inc-r0",
                "category": "restart",
                "deduplication_key": "k2",
                "created_at": T0.isoformat(),
                "status": "SENT",
            },
        ]

    def respond(self, script: str, *arguments: str) -> dict[str, Any]:
        job = arguments[1]
        attempt = f"{job}-a001"
        if script == ATTEMPT_PROBE:
            if self.anchor is None:
                self.reads_before_delete += 1
                return {
                    "attempts": {attempt: _entry(restart=None)},
                    "restart_budget": None,
                }
            records = _dated(
                self.anchor,
                self.offsets,
                restart=self.restart_statuses,
                count=self.count,
                reservations=self.reservations,
            )
            return {**records, "attempts": {attempt: records["attempts"][ATTEMPT]}}
        if script == iso.ISOLATION_RECORDS_PROBE:
            anchor = self.anchor or T0
            observations = [
                {
                    "attempt_id": attempt,
                    "job_id": job,
                    "workload_phase": "FAILED",
                    "observed_at": (
                        anchor + _seconds(self.source_observed_offset)
                    ).isoformat(),
                    "watcher_instance": "w1"
                    if self.source_observed_offset < 0
                    else "w2",
                    "containers": 1,
                }
            ]
            observations += [
                {
                    "attempt_id": f"{attempt}-r-{index:08x}",
                    "job_id": job,
                    "workload_phase": "RUNNING",
                    "observed_at": (anchor + _seconds(40)).isoformat(),
                    "watcher_instance": "w2",
                    "containers": 1,
                }
                for index in range(self.restart_observed)
            ]
            return {
                "heartbeat": {
                    "cluster_id": "gpu-eks-cluster",
                    "observed_at": (anchor + timedelta(days=1)).isoformat(),
                    "watched_pods": 1,
                    "watched_attempts": 1,
                    "watcher_instance": "w2",
                },
                "observations": observations,
                "notifications": self.notifications if arguments[3] else [],
                "agent": None,
            }
        raise AssertionError("unexpected store script")


def _variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Store, cluster: Cluster
):
    FakeFailJob.instances = []
    monkeypatch.setattr(att, "ManagedFailJob", FakeFailJob)
    ctx = _context(tmp_path, Regional(cluster, cpu_python=store.respond))
    readings: list[datetime] = []
    base_clock = ctx.clock

    def clock() -> datetime:
        value = base_clock()
        readings.append(value)
        return value

    ctx.clock = clock

    def on_delete() -> None:
        # The reading the runner took just before the delete request.
        store.anchor = readings[-1]

    cluster.on_delete = on_delete
    original = cluster.kubectl

    def guarded(plane: str, arguments: tuple[str, ...], options: dict[str, Any]) -> str:
        if arguments[:1] == ("get",):
            # The job is every managed-job variant's first mutation
            # (test_general_live_managed_job_start); the watcher comes after.
            assert FakeFailJob.instances, "the watcher was looked up before the job"
        return original(plane, arguments, options)

    cluster.kubectl = guarded  # type: ignore[method-assign]
    return ctx


def test_a_missing_watcher_after_the_job_started_is_a_fixture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    cluster.pods = []
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(RegionalFixtureError, match="exactly one Ready"):
        ctx.spec.execute(ctx)
    (job,) = FakeFailJob.instances
    assert ctx.cleanup.pending() == [f"job-{job.name}"], "the job's undo is registered"
    assert cluster.deletes == [], "nothing is deleted without a watcher"
    assert cluster.source_reads == 0, "the source is not watched without a watcher"


@pytest.mark.parametrize("source_observed_offset", [-3.0, 6.0], ids=["w1", "w2"])
@pytest.mark.parametrize("landed", INSIDE)
def test_worker_restart_passes_for_every_landing_inside_the_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    landed: str,
    source_observed_offset: float,
) -> None:
    store, cluster = Store(), Cluster()
    store.offsets = dict(LANDINGS[landed])
    store.source_observed_offset = source_observed_offset
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    spec = ctx.spec
    assert (
        spec.approvals == ("CONTROL_PLANE_RESTART",)
        and spec.flags == "X"
        and spec.risk == "H"
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["delete_landed"] == landed
    assert store.reads_before_delete == 0, "no store read before the delete"
    assert cluster.source_reads_at_delete == len(cluster.frames), (
        "the delete follows the first read that showed the source ended"
    )
    assert len(cluster.deletes) == 1
    assert result["watcher"] == {
        "deleted": "uid-old",
        "replacement": "uid-new",
        "node": WATCHER_NODE,
    }
    assert set(result["stages"]) == {
        "watcher_replaced",
        "single_restart",
        "attempt_once",
        "notifications",
        "watcher_restored",
    }
    timing = result["delete_timing"]
    assert timing["after_publish_seconds"] == -store.offsets["published"]
    assert (
        result["transaction"]["published_at"]
        == (store.anchor + _seconds(store.offsets["published"])).isoformat()
    )
    assert any(f"the delete landed {landed}: " in note for note in result["notes"]), (
        result["notes"]
    )
    job = FakeFailJob.instances[-1]
    assert job.mode == "exit_once:1" and job.restart_budget == 1
    assert job.copy_run_seconds == wr.COPY_RUN_SECONDS, (
        "the restart copy runs COPY_RUN_SECONDS, then exits 0"
    )
    assert ctx.cleanup.pending() == [f"job-{job.name}"]
    for name in ("source-failure-timeline", "watcher-restart", "delete-timing"):
        assert (ctx.evidence.case_dir / f"{name}.json").is_file(), name
    document = json.loads((ctx.evidence.case_dir / "delete-timing.json").read_text())
    assert document["landing"] == timing
    assert document["settle_bound"]["total_seconds"] == 1935.0
    assert document["source_failure"]["trigger"] == wr.POD_TERMINATED
    assert document["source_failure"]["signals"] == [wr.POD_TERMINATED, wr.JOB_ENDED]
    assert document["source_failure"]["pods"][0]["containers"][0]["exit_code"] == 1
    assert set(document["stages"]) == set(result["stages"])


def test_worker_restart_passes_on_the_live_running_to_absent_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HyperPod, 2026-10-27 12:40: the source Pod went from Running straight to
    absent and the job's Failed condition was what remained; the delete fires
    on that read instead of waiting 900 s for a terminated state."""

    store, cluster = Store(), Cluster()
    cluster.frames = LIVE_SHAPE
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert cluster.source_reads_at_delete == len(LIVE_SHAPE)
    failure = json.loads((ctx.evidence.case_dir / "delete-timing.json").read_text())[
        "source_failure"
    ]
    assert failure["trigger"] == wr.JOB_ENDED
    assert failure["signals"] == [wr.JOB_ENDED, wr.POD_ABSENT]
    assert failure["pod_state"] == "absent" and failure["job_state"] == FAILED_JOB
    assert failure["last_running_at"] is not None
    timeline = json.loads(
        (ctx.evidence.case_dir / "source-failure-timeline.json").read_text()
    )
    assert [(item["pod"], item["job"]) for item in timeline["entries"]] == [
        ("Running", RUNNING_JOB),
        ("absent", FAILED_JOB),
    ]


def test_worker_restart_is_inconclusive_when_the_delete_landed_after_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    store.offsets = dict(LANDINGS[wr.AFTER_SETTLE])
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(VariantInconclusive) as raised:
        ctx.spec.execute(ctx)
    detail = str(raised.value)
    assert detail.startswith(
        "the watcher was replaced after the restart transaction had settled: "
    ), detail
    assert (
        "15.0 s after the publish" not in detail
        and "30.0 s after the publish" in detail
    )
    assert len(cluster.deletes) == 1, "the delete did go out at the source failure"
    document = json.loads((ctx.evidence.case_dir / "delete-timing.json").read_text())
    assert document["landing"]["landed"] == wr.AFTER_SETTLE
    assert document["stages"]["single_restart"] == [], (
        "the oracle itself was clean; only the precondition failed"
    )


def test_worker_restart_is_a_fixture_failure_without_a_ready_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    cluster.replacement_appears = False
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(RegionalFixtureError, match="no Ready replacement"):
        ctx.spec.execute(ctx)
    assert len(cluster.deletes) == 1
    assert store.reads_before_delete == 0, "no store read before the delete"


def test_worker_restart_is_a_fixture_failure_when_the_source_never_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    cluster.frames = NEVER_ENDS
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(RegionalFixtureError) as raised:
        ctx.spec.execute(ctx)
    assert str(raised.value).endswith(
        f"did not end within 900 s (last Pod state Running; last job condition "
        f"{RUNNING_JOB})"
    ), str(raised.value)
    assert cluster.deletes == [], "no watcher Pod is deleted without a source failure"
    assert store.reads_before_delete == 0


@pytest.mark.parametrize(
    ("change", "stage", "needle"),
    [
        ({"count": 0, "reservations": []}, "single_restart", "reservation was lost"),
        (
            {
                "count": 2,
                "reservations": [
                    "wf-r0/0/RESTART_WORKLOAD",
                    "wf-r0/1/RESTART_WORKLOAD",
                ],
            },
            "single_restart",
            "double decrement",
        ),
        (
            {"restart_statuses": ["SUCCEEDED", "SUCCEEDED"]},
            "single_restart",
            "succeeded 2 times",
        ),
        ({"restart_observed": 2}, "attempt_once", "2 restart attempts"),
        ({"restart_observed": 0}, "attempt_once", "no restart attempt"),
    ],
    ids=["lost", "doubled", "restart-twice", "two-attempts", "no-attempt"],
)
def test_worker_restart_fails_on_a_lost_or_doubled_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, Any],
    stage: str,
    needle: str,
) -> None:
    store, cluster = Store(), Cluster()
    for key, value in change.items():
        setattr(store, key, value)
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert any(needle in item for item in result["stages"][stage]), result["stages"]


def test_worker_restart_fails_on_duplicate_notifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    store.notifications = store.notifications + store.notifications[:1]
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    result = ctx.spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert result["stages"]["notifications"] == [
        "notification k1 was recorded 2 times",
        "2 containment notifications for incident inc-c",
    ]


# --------------------------------------------------------------------------- #
# the wording and the product facts the design rests on
# --------------------------------------------------------------------------- #
def test_control_plane_restart_approval_and_spec_name_the_transaction() -> None:
    meaning = APPROVALS["CONTROL_PLANE_RESTART"]
    assert "inside the restart transaction" in meaning, meaning
    assert "open restart claim" not in meaning, meaning
    assert "the Pod gone after Running, or the PyTorchJob's Failed condition" in meaning
    spec = _spec()
    assert "first API observation of the source's end" in spec.injection
    assert "the Pod gone after Running, or the PyTorchJob's Failed condition" in (
        spec.injection
    )
    assert "no store read" in spec.injection
    assert "a delete dated after the settle is INCONCLUSIVE" in spec.expected


def test_open_claim_is_one_control_loop_round_trip_in_the_product() -> None:
    """The variant times its delete from the source failure instead of an
    "open claim" because of these product facts; a change to any of them is
    the cue to revisit.

    The data plane's RESTART_WORKLOAD returns SUCCEEDED in the call that
    creates the restart copy, without reading a Pod of it (pinned by the
    product's own tests/execution/test_restart_safety.py
    ``test_terminal_pytorch_restart_creates_retry_object``: ``core_api`` is an
    unused double). The dispatcher is woken by the command's terminal status,
    so the step is recorded SUCCEEDED within the round trip rather than at the
    next poll; the poll is the fallback the settle bound counts.
    """

    assert WAKING_REMOTE_COMMAND_STATUSES == frozenset({"SUCCEEDED", "FAILED"})
    assert WorkflowDispatcherConfig(enabled=False).poll_interval_seconds == 5.0
    cap = ProductionExecutorConfig(
        enabled=False, executor_id="pin", allowed_operations=frozenset()
    ).step_waiting_limit(WorkflowOperation.RESTART_WORKLOAD)
    assert cap == 600, "the RESTART_WORKLOAD waiting cap the settle bound counts"
    assert wr.TERMINAL_STEPS == frozenset({"SUCCEEDED", "FAILED"})
    assert wr.LANDINGS == (
        wr.BEFORE_PUBLISH,
        wr.BETWEEN_PUBLISH_AND_RESTART,
        wr.AFTER_RESTART_BEFORE_SETTLE,
        wr.AFTER_SETTLE,
        wr.UNDATED,
    )
    assert wr.SOURCE_END_SIGNALS == (wr.POD_TERMINATED, wr.JOB_ENDED, wr.POD_ABSENT)
