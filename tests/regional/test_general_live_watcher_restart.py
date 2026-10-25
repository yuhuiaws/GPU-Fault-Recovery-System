"""GF-GENERAL-LIVE-029 ``worker_restart``: the watcher replaced mid-claim.

The claim-state reader, the open-claim wait (open, closed-first, never), the
UID-fenced delete's exact request, the replacement wait, and the graders over
scripted completion records; then the variant end to end over a fake job, a
fake watcher inventory that flips on the delete, and a phase-driven store.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

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


def _workflow(
    request_id: str, op: str, statuses: list[str], status: str = "SUCCEEDED"
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "official_steps": [{"operation": op, "parameters": {}}],
        "step_executions": [
            {"step_index": 0, "operation": op, "status": item, "details": {}}
            for item in statuses
        ],
    }


def _entry(
    *,
    restart: list[str] | None,
    recovery_status: str = "SUCCEEDED",
    decision: str = "PLAN_CREATED",
    stops: int = 1,
    recoveries: int = 1,
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
                        f"wf-r{index}", "RESTART_WORKLOAD", restart, recovery_status
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
    assert (ctx.evidence.case_dir / "open-claim-timeline.json").is_file(), (
        "the wait writes its timeline"
    )

    closed = _records(_entry(restart=["SUCCEEDED"]))
    with pytest.raises(VariantInconclusive, match="closed before the watcher"):
        wr.wait_open_claim(ctx, lambda: closed, ATTEMPT, timeout_seconds=600)

    never = _records(_entry(restart=None), count=0)
    with pytest.raises(VariantInconclusive, match="no open restart claim appeared"):
        wr.wait_open_claim(ctx, lambda: never, ATTEMPT, timeout_seconds=30)


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
            entry = _entry(restart=self.restart_statuses)
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
    ctx = _context(tmp_path, Regional(cluster, cpu_python=store.respond))
    return ctx


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


def test_worker_restart_is_inconclusive_when_the_restart_ended_before_the_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, cluster = Store(), Cluster()
    store.phase = "closed"
    ctx = _variant(tmp_path, monkeypatch, store, cluster)
    with pytest.raises(VariantInconclusive, match="closed before the watcher"):
        ctx.spec.execute(ctx)
    assert cluster.deletes == [], "no watcher Pod is deleted outside the window"


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
