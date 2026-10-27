"""GF-GENERAL-LIVE-029 ``worker_restart``: the watcher replaced mid-claim.

The claim-state reader, the open-claim wait (open, closed-first, never), the
product-dated claim window and the delete judged against it, the UID-fenced
delete's exact request, the replacement wait, and the graders over scripted
completion records; then the variant end to end over a fake job, a fake
watcher inventory that flips on the delete, and a phase-driven store. The last
test pins the product facts the window reasoning rests on.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

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
from scripts.e2e.regional.general_live_store import ATTEMPT_PROBE
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
WATCHER_NODE = "ip-10-0-40-9.us-west-2.compute.internal"
T0 = datetime(2026, 10, 24, tzinfo=timezone.utc)
JOB = "gl29wr-deadbeef-1"
ATTEMPT = f"{JOB}-a001"
#: The claim as the product dates it in the scripted records: reserved at the
#: recovery workflow's CLAIM, closed when the terminal RESTART_WORKLOAD record
#: was written. The fake clock of ``make_context`` starts at T0 and advances a
#: second per reading, so the default window brackets every reading of a test.
CLAIMED_AT = T0 - timedelta(hours=1)
CLOSED_AT = T0 + timedelta(hours=1)


def _workflow(
    request_id: str,
    op: str,
    statuses: list[str],
    status: str = "SUCCEEDED",
    *,
    claimed_at: datetime | None = None,
    updated_at: datetime | None = None,
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
    }


def _entry(
    *,
    restart: list[str] | None,
    recovery_status: str = "SUCCEEDED",
    decision: str = "PLAN_CREATED",
    stops: int = 1,
    recoveries: int = 1,
    claimed_at: datetime | None = None,
    updated_at: datetime | None = None,
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
                    },
                    "workflow": _workflow(
                        f"wf-r{index}",
                        "RESTART_WORKLOAD",
                        restart,
                        recovery_status,
                        claimed_at=claimed_at,
                        updated_at=updated_at,
                    ),
                }
            )
    return {
        "event": {"terminal_status": "FAILED"},
        "decision": {"status": decision},
        "containment_incident": {
            "incident_id": "inc-c",
            "job_id": JOB,
            "state": "RECOVERED",
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


# --------------------------------------------------------------------------- #
# claim state and waits
# --------------------------------------------------------------------------- #
def test_claim_state_reads_the_reservation_and_the_restart_execution() -> None:
    assert wr.claim_state(_records(_entry(restart=None), count=0), ATTEMPT) == wr.NONE
    assert wr.claim_state(_records(_entry(restart=[])), ATTEMPT) == wr.OPEN
    assert (
        wr.claim_state(
            _records(_entry(restart=["WAITING", "WAITING"], recovery_status="RUNNING")),
            ATTEMPT,
        )
        == wr.OPEN
    )
    assert (
        wr.claim_state(_records(_entry(restart=["WAITING", "SUCCEEDED"])), ATTEMPT)
        == wr.CLOSED
    )
    assert wr.claim_state(_records(_entry(restart=["FAILED"])), ATTEMPT) == wr.CLOSED
    assert wr.claim_state({"attempts": {}}, ATTEMPT) == wr.NONE


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


def test_wait_open_claim_returns_the_first_open_read_and_refuses_a_closed_one(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, FakeRegional())
    sequence = iter(
        [
            _records(_entry(restart=None), count=0),
            _records(_entry(restart=["WAITING"], recovery_status="RUNNING")),
            _records(_entry(restart=["WAITING", "SUCCEEDED"])),
        ]
    )
    opened = wr.wait_open_claim(
        ctx, lambda: next(sequence), ATTEMPT, timeout_seconds=600
    )
    assert wr.claim_state(opened, ATTEMPT) == wr.OPEN
    timeline = json.loads(
        (ctx.evidence.case_dir / "open-claim-timeline.json").read_text()
    )
    assert [item["accepted"] for item in timeline["entries"]] == [False, True], (
        "the wait writes its timeline: one NONE read, then the OPEN one"
    )
    assert not (ctx.evidence.case_dir / "open-claim-window.json").is_file(), (
        "an open claim is not a miss; the window is dated after the records settle"
    )

    closed = _records(_entry(restart=["SUCCEEDED"]))
    with pytest.raises(VariantInconclusive, match="closed before the watcher"):
        wr.wait_open_claim(ctx, lambda: closed, ATTEMPT, timeout_seconds=600)

    never = _records(_entry(restart=None), count=0)
    with pytest.raises(VariantInconclusive, match="no open restart claim appeared"):
        wr.wait_open_claim(ctx, lambda: never, ATTEMPT, timeout_seconds=30)


def test_wait_open_claim_dates_the_window_it_missed_from_the_product_records(
    tmp_path: Path,
) -> None:
    """Closed at the first read: the detail says how long the claim was open and
    how long a records read takes, and the window is written as evidence."""

    ctx = _context(tmp_path, FakeRegional())
    claimed = T0 + timedelta(seconds=30)
    closed = _records(
        _entry(
            restart=["SUCCEEDED"],
            claimed_at=claimed,
            updated_at=claimed + timedelta(seconds=2, milliseconds=400),
        )
    )
    with pytest.raises(VariantInconclusive) as raised:
        wr.wait_open_claim(ctx, lambda: closed, ATTEMPT, timeout_seconds=600)
    detail = str(raised.value)
    assert "closed before the watcher could be replaced" in detail
    assert (
        f"date the claim open from {claimed.isoformat()} to "
        f"{(claimed + timedelta(seconds=2, milliseconds=400)).isoformat()}, 2.4 s"
    ) in detail, detail
    # The fake clock moves one second per reading: a read "takes" 1 s.
    assert f"a records read takes up to 1.0 s plus {wr.CLAIM_POLL_SECONDS} s" in detail
    window = json.loads((ctx.evidence.case_dir / "open-claim-window.json").read_text())
    assert window["open_seconds"] == 2.4
    assert window["read_seconds"] == [1.0]
    assert window["first_read_state"] == wr.CLOSED

    undated = _records(_entry(restart=["SUCCEEDED"]))
    with pytest.raises(VariantInconclusive, match="records do not date the claim"):
        wr.wait_open_claim(ctx, lambda: undated, ATTEMPT, timeout_seconds=600)


def test_claim_window_dates_the_claim_from_the_recovery_workflow_records() -> None:
    claimed = T0 + timedelta(seconds=30)
    closed = claimed + timedelta(seconds=1, milliseconds=750)
    records = _records(
        _entry(restart=["SUCCEEDED"], claimed_at=claimed, updated_at=closed)
    )
    records["restart_budget"]["updated_at"] = claimed.isoformat()
    assert wr.claim_window(records, ATTEMPT) == {
        "reserved_at": claimed.isoformat(),
        "dispatched_at": (claimed + timedelta(milliseconds=400)).isoformat(),
        "closed_at": closed.isoformat(),
        "open_seconds": 1.75,
        "ledger_updated_at": claimed.isoformat(),
    }
    # Still WAITING: the claim has not closed; a workflow without events is
    # not dated at all; an attempt without records neither.
    waiting = wr.claim_window(
        _records(
            _entry(restart=["WAITING"], recovery_status="RUNNING", claimed_at=claimed)
        ),
        ATTEMPT,
    )
    assert (waiting["reserved_at"], waiting["closed_at"], waiting["open_seconds"]) == (
        claimed.isoformat(),
        None,
        None,
    )
    undated = wr.claim_window(_records(_entry(restart=["SUCCEEDED"])), ATTEMPT)
    assert undated["reserved_at"] is None and undated["open_seconds"] is None
    assert wr.claim_window({"attempts": {}}, ATTEMPT)["closed_at"] is None


def test_delete_timing_errors_judge_the_delete_bracket_against_the_window() -> None:
    claimed = T0 + timedelta(seconds=30)
    window = wr.claim_window(
        _records(
            _entry(
                restart=["SUCCEEDED"],
                claimed_at=claimed,
                updated_at=claimed + timedelta(seconds=3),
            )
        ),
        ATTEMPT,
    )

    def document(before: float, after: float) -> dict[str, str]:
        return {
            "deleted_at": (claimed + timedelta(seconds=before)).isoformat(),
            "deletion_returned_at": (claimed + timedelta(seconds=after)).isoformat(),
        }

    assert wr.delete_timing_errors(window, document(0.5, 1.5)) == []
    (late,) = wr.delete_timing_errors(window, document(7.0, 8.0))
    assert "4.0 s after the claim closed" in late and "3.0 s" in late, late
    (early,) = wr.delete_timing_errors(window, document(-5.0, -4.0))
    assert "4.0 s before the claim opened" in early, early
    (straddle,) = wr.delete_timing_errors(window, document(2.5, 3.5))
    assert "straddles the window's edge" in straddle, straddle
    assert wr.delete_timing_errors(
        wr.claim_window(_records(_entry(restart=["SUCCEEDED"])), ATTEMPT),
        document(0.5, 1.5),
    ) == [
        "the product's records do not date the claim (no CLAIM event on the "
        "recovery workflow or no terminal RESTART_WORKLOAD record)"
    ]
    assert wr.delete_timing_errors(window, {}) == [
        "the restart document does not date the delete"
    ]
    # A document from before the bracket existed is judged on its one stamp.
    assert (
        wr.delete_timing_errors(
            window, {"deleted_at": document(1.0, 1.0)["deleted_at"]}
        )
        == []
    )


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
# the fenced delete and the replacement
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


class Cluster:
    """A fake GPU cluster: the watcher inventory flips on the fenced delete."""

    def __init__(self) -> None:
        self.pods = [_pod("uid-old")]
        self.deletes: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.polls_before_ready = 1
        self.exec_calls: list[tuple[str, ...]] = []
        self.on_delete: Any = None

    def kubectl(
        self, plane: str, arguments: tuple[str, ...], options: dict[str, Any]
    ) -> str:
        if arguments[:2] == ("get", "pod"):
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
            self.pods = [_pod("uid-new", rv="200")]
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
    document = wr.restart_watcher(ctx)
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
    restart = {
        "attempt_id": f"{ATTEMPT}-r-0a1b2c3d",
        "job_id": JOB,
        "workload_phase": "RUNNING",
    }
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
    ) -> None:
        self.name = name
        self.attempt_id = f"{name}-a001"
        self.mode = mode
        self.restart_budget = restart_budget
        self.fixture = type("F", (), {"delete": staticmethod(lambda: None)})()
        self.prewarm = type("P", (), {"cleanup": staticmethod(lambda: None)})()
        FakeFailJob.instances.append(self)

    def start(self) -> dict[str, Any]:
        return {}

    def delete(self) -> None:
        pass


class Store:
    """The completion records by phase: none -> open -> (delete) -> closed."""

    def __init__(self) -> None:
        self.phase = "none"
        self.reads_before_open = 1
        self.restart_statuses = ["WAITING", "SUCCEEDED"]
        self.count = 1
        self.reservations: list[str] | None = None
        self.restart_observed = 1
        # The product-dated claim window in the settled records.
        self.claimed_at: datetime | None = CLAIMED_AT
        self.closed_at: datetime | None = CLOSED_AT
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
        self.deleted_in_phase: str | None = None

    def respond(self, script: str, *arguments: str) -> dict[str, Any]:
        job = arguments[1]
        attempt = f"{job}-a001"
        if script == ATTEMPT_PROBE:
            if self.phase == "none":
                if self.reads_before_open == 0:
                    self.phase = "open"
                else:
                    self.reads_before_open -= 1
                    entry = _entry(restart=None)
                    return {"attempts": {attempt: entry}, "restart_budget": None}
            if self.phase == "open":
                entry = _entry(restart=["WAITING"], recovery_status="RUNNING")
                return {
                    "attempts": {attempt: entry},
                    "restart_budget": {
                        "restart_count": 1,
                        "reservation_ids": ["wf-r0/0/RESTART_WORKLOAD"],
                        "budget": 1,
                    },
                }
            entry = _entry(
                restart=self.restart_statuses,
                claimed_at=self.claimed_at,
                updated_at=self.closed_at,
            )
            records = _records(entry, count=self.count, reservations=self.reservations)
            return {**records, "attempts": {attempt: entry}}
        if script == iso.ISOLATION_RECORDS_PROBE:
            observations = [
                {
                    "attempt_id": attempt,
                    "job_id": job,
                    "workload_phase": "FAILED",
                    "observed_at": T0.isoformat(),
                    "containers": 1,
                }
            ]
            observations += [
                {
                    "attempt_id": f"{attempt}-r-{index:08x}",
                    "job_id": job,
                    "workload_phase": "RUNNING",
                    "observed_at": T0.isoformat(),
                    "containers": 1,
                }
                for index in range(self.restart_observed)
            ]
            return {
                "heartbeat": {
                    "cluster_id": "gpu-eks-cluster",
                    "observed_at": (T0 + timedelta(days=1)).isoformat(),
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

    def on_delete() -> None:
        store.deleted_in_phase = store.phase
        store.phase = "closed"

    cluster.on_delete = on_delete
    original = cluster.kubectl

    def guarded(plane: str, arguments: tuple[str, ...], options: dict[str, Any]) -> str:
        if arguments[:2] == ("get", "pod"):
            # The job is every managed-job variant's first mutation
            # (test_general_live_managed_job_start); the watcher comes after.
            assert FakeFailJob.instances, "the watcher was looked up before the job"
        return original(plane, arguments, options)

    cluster.kubectl = guarded  # type: ignore[method-assign]
    ctx = _context(tmp_path, Regional(cluster, cpu_python=store.respond))
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


def test_worker_restart_passes_when_the_claim_survives_the_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    spec = ctx.spec
    assert (
        spec.approvals == ("CONTROL_PLANE_RESTART",)
        and spec.flags == "X"
        and spec.risk == "H"
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert store.deleted_in_phase == "open", "the Pod went while the claim was open"
    assert len(cluster.deletes) == 1
    assert result["claim_state_at_delete"] == wr.OPEN
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
    job = FakeFailJob.instances[-1]
    assert job.mode == "exit:1" and job.restart_budget == 1
    assert ctx.cleanup.pending() == [f"job-{job.name}"]
    assert (ctx.evidence.case_dir / "open-claim-records.json").is_file(), (
        "the records at the open claim are evidence"
    )
    assert result["claim_window"] == {
        "reserved_at": CLAIMED_AT.isoformat(),
        "dispatched_at": (CLAIMED_AT + timedelta(milliseconds=400)).isoformat(),
        "closed_at": CLOSED_AT.isoformat(),
        "open_seconds": 7200.0,
        "ledger_updated_at": None,
    }
    window = json.loads((ctx.evidence.case_dir / "open-claim-window.json").read_text())
    assert window["timing"] == [] and set(window["stages"]) == set(result["stages"])
    assert window["deleted_at"] < window["deletion_returned_at"], (
        "the delete bracket is on the window evidence"
    )


def test_worker_restart_is_inconclusive_when_the_restart_ended_before_the_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    store.phase = "closed"
    store.claimed_at = T0 + timedelta(seconds=1)
    store.closed_at = T0 + timedelta(seconds=3, milliseconds=200)
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(
        VariantInconclusive, match="closed before the watcher"
    ) as raised:
        ctx.spec.execute(ctx)
    assert "2.2 s" in str(raised.value), str(raised.value)
    assert cluster.deletes == [], "no watcher Pod is deleted outside the window"


def test_worker_restart_is_inconclusive_when_the_delete_missed_the_dated_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first read was OPEN, but the settled records date the claim closed
    before the delete request went out: the condition was not established,
    whatever the oracle would have said, and the stages stay as evidence."""

    store, cluster = Store(), Cluster()
    store.claimed_at = T0 + timedelta(seconds=1)
    store.closed_at = T0 + timedelta(seconds=3)
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(VariantInconclusive) as raised:
        ctx.spec.execute(ctx)
    detail = str(raised.value)
    assert detail.startswith("the watcher was not replaced inside the open claim: "), (
        detail
    )
    assert "after the claim closed" in detail and "2.0 s" in detail, detail
    assert store.deleted_in_phase == "open", "the delete did go out on an OPEN read"
    assert len(cluster.deletes) == 1
    window = json.loads((ctx.evidence.case_dir / "open-claim-window.json").read_text())
    assert window["open_seconds"] == 2.0
    assert window["timing"] == [detail.split(": ", 1)[1]]
    assert set(window["stages"]) == {
        "watcher_replaced",
        "single_restart",
        "attempt_once",
        "notifications",
        "watcher_restored",
    }
    assert window["stages"]["single_restart"] == [], (
        "the oracle itself was clean; only the precondition failed"
    )


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
# the product facts the window reasoning rests on
# --------------------------------------------------------------------------- #
def test_open_claim_is_one_control_loop_round_trip_in_the_product() -> None:
    """The harness dates the window instead of trying to widen it because of
    these product facts; a change to any of them is the cue to revisit.

    The data plane's RESTART_WORKLOAD returns SUCCEEDED in the call that
    creates the restart copy, without reading a Pod of it (pinned by the
    product's own tests/execution/test_restart_safety.py
    ``test_terminal_pytorch_restart_creates_retry_object``: ``core_api`` is an
    unused double). The dispatcher is woken by the command's terminal status,
    so the step is recorded SUCCEEDED within the round trip rather than at the
    next poll. A product-side hold after the submission would run under the
    step's waiting cap; a start barrier the harness then puts on the restart
    copy must stay below it with margin (``N + margin < cap``).
    """

    assert WAKING_REMOTE_COMMAND_STATUSES == frozenset({"SUCCEEDED", "FAILED"})
    assert WorkflowDispatcherConfig(enabled=False).poll_interval_seconds == 5.0
    cap = ProductionExecutorConfig(
        enabled=False, executor_id="pin", allowed_operations=frozenset()
    ).step_waiting_limit(WorkflowOperation.RESTART_WORKLOAD)
    assert cap == 600, (
        "the RESTART_WORKLOAD waiting cap any post-submission hold runs under"
    )
    assert wr.CLAIM_POLL_SECONDS == 5
    assert wr.TERMINAL_STEPS == frozenset({"SUCCEEDED", "FAILED"})
