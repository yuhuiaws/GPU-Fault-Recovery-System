"""The writer restore HA-003 and HA-010 run after their own failover.

Live 2026-10-13 HA-010's failover left the writer in another zone than every
control-plane Pod until an operator failed back by hand. Both cases now fail
the writer back to the pre-case instance in cleanup, under the injection's
quiet rule, record ``writer-restore.json``, and fail when placement is not
restored. Everything here runs on a fake RDS, store and clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import aurora_writer_restore as restore
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional.regional_commands import RegionalFixtureError

QUIET = {"queue": {"depth": 0}, "remote_commands": {"open_by_cluster": {}}}
BUSY = {"queue": {"depth": 0}, "remote_commands": {"open_by_cluster": {"gpu-a": 1}}}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Rds:
    """Two members; a named failover settles on its target after one poll."""

    def __init__(self, writer: str = "db-b", *, unsettled: int = 0) -> None:
        self.writer = writer
        self.unsettled = unsettled
        self.requests: list[str] = []
        self.ignore_requests = False
        self.land_on: str | None = None

    def snapshot(self) -> dict[str, Any]:
        if self.unsettled:
            self.unsettled -= 1
            return {"status": "failing-over", "writer": None}
        return {"status": "available", "writer": self.writer}

    def fail_over_to(self, target: str) -> dict[str, Any]:
        self.requests.append(target)
        if not self.ignore_requests:
            self.writer = self.land_on or target
        return {"DBCluster": {"DBClusterIdentifier": "unit"}}

    def wait_writer_left(
        self, previous: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.writer == previous:
            raise RegionalFixtureError(f"Aurora failover did not converge: {previous}")
        return self.snapshot(), [{"writer": self.writer}]

    def bound(self) -> restore.WriterRestoreRds:
        return restore.WriterRestoreRds(
            snapshot=self.snapshot,
            fail_over_to=self.fail_over_to,
            wait_writer_left=self.wait_writer_left,
        )


def _restore(
    rds: Rds, tmp_path: Path, states: list[dict[str, Any]] | None = None
) -> tuple[Clock, list[dict[str, Any]], Any]:
    clock = Clock()
    reads: list[dict[str, Any]] = []
    queue = list(states or [QUIET])

    def read_state() -> dict[str, Any]:
        state = queue.pop(0) if len(queue) > 1 else queue[0]
        reads.append(state)
        return state

    try:
        result: Any = restore.restore_writer(
            rds.bound(),
            original_writer="db-a",
            case_dir=tmp_path,
            read_state=read_state,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
    except RegionalFixtureError as exc:
        result = exc
    return clock, reads, result


def _evidence(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / restore.EVIDENCE_FILE).read_text())


def test_a_moved_writer_is_failed_back_after_the_failover_settles(
    tmp_path: Path,
) -> None:
    rds = Rds(unsettled=2)

    _, reads, record = _restore(rds, tmp_path, [BUSY, QUIET])

    assert rds.requests == ["db-a"] and rds.writer == "db-a"
    assert isinstance(record, dict), record
    assert record["action"] == "failover" and record["errors"] == []
    assert record["writer_before_restore"] == "db-b"
    assert record["rds_after"] == {"status": "available", "writer": "db-a"}
    assert len(reads) == 2, "a busy store is waited out, not refused at once"
    assert _evidence(tmp_path) == record
    assert json.loads((tmp_path / restore.QUIET_EVIDENCE_FILE).read_text()) == QUIET


def test_a_writer_that_never_left_needs_no_failover(tmp_path: Path) -> None:
    rds = Rds(writer="db-a")

    _, reads, record = _restore(rds, tmp_path)

    assert rds.requests == [] and reads == [], "nothing to restore, nothing read"
    assert record["action"] == "none" and record["errors"] == []


def test_work_in_flight_refuses_the_restore_after_a_bounded_wait(
    tmp_path: Path,
) -> None:
    rds = Rds()

    clock, reads, error = _restore(rds, tmp_path, [BUSY])

    assert isinstance(error, RegionalFixtureError), error
    assert "writer restore refused: remote commands are open" in str(error)
    assert "gpu-fault-admin aurora-failback" in str(error)
    assert rds.requests == [], "no failover lands on a node action in flight"
    assert clock.now >= restore.RESTORE_QUIET_WAIT_SECONDS
    assert len(reads) > 1, "the quiet rule is polled, not read once"
    assert "writer restore refused" in _evidence(tmp_path)["errors"][0]


@pytest.mark.parametrize("defect", ["unsettled", "ignored", "elsewhere"])
def test_a_restore_that_cannot_restore_placement_raises(
    tmp_path: Path, defect: str
) -> None:
    rds = Rds()
    if defect == "unsettled":
        rds.unsettled = 10_000
    elif defect == "ignored":
        rds.ignore_requests = True
    else:
        rds.land_on = "db-c"

    _, _, error = _restore(rds, tmp_path)

    assert isinstance(error, RegionalFixtureError), error
    expected = {
        "unsettled": "Aurora did not settle before the writer restore",
        "ignored": "Aurora failover did not converge",
        "elsewhere": "the writer is 'db-c' after the restore",
    }[defect]
    assert expected in str(error), error
    assert _evidence(tmp_path)["errors"], "a failed restore is evidence too"


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "target": "db-1",
        "action": "failover",
        "rds_after": {"status": "available", "writer": "db-1"},
    }
    record.update(overrides)
    return record


def test_a_writer_back_on_the_pre_case_instance_restores_placement() -> None:
    assert restore.placement_restore_errors(_record(), original_writer="db-1") == []
    untouched = _record(action="none")
    assert restore.placement_restore_errors(untouched, original_writer="db-1") == [], (
        "a writer that never left needs no failover"
    )


@pytest.mark.parametrize(
    ("overrides", "original", "fragment"),
    [
        ({"rds_after": {"status": "available", "writer": "db-2"}}, "db-1", "db-2"),
        (
            {"rds_after": {"status": "failing-over", "writer": "db-1"}},
            "db-1",
            "not available",
        ),
        ({"rds_after": None}, "db-1", "after the restore"),
        ({"target": "db-2"}, "db-1", "targeted 'db-2'"),
        ({}, "", "pre-case writer is unknown"),
    ],
)
def test_a_writer_left_elsewhere_is_named_with_the_failback(
    overrides: dict[str, Any], original: str, fragment: str
) -> None:
    errors = restore.placement_restore_errors(
        _record(**overrides), original_writer=original
    )
    assert any(fragment in item for item in errors), errors
    if (overrides.get("rds_after") or {}).get("writer") == "db-2":
        assert "gpu-fault-admin aurora-failback" in " ".join(errors), errors


def test_the_quiet_rule_is_the_injection_rule() -> None:
    assert restore.store_quiet_errors(QUIET) == []
    assert restore.store_quiet_errors(BUSY) == ["remote commands are open"]
    backlog = {**QUIET, "queue": {"fault_backlog_depth": 2, "depth": 9}}
    assert restore.store_quiet_errors(backlog) == [
        "the processor fault queue holds work"
    ]
    telemetry_only = {**QUIET, "queue": {"fault_backlog_depth": 0, "depth": 9}}
    assert restore.store_quiet_errors(telemetry_only) == [], (
        "routine telemetry is not work in flight"
    )


def test_ha003_binds_its_rds_calls_with_a_named_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    settings = SimpleNamespace(
        regional=SimpleNamespace(region="us-west-2"), rds_cluster_id="unit-rds"
    )
    monkeypatch.setattr(
        ha003, "aws_rds", lambda _settings, *args: calls.append(args) or {}
    )
    monkeypatch.setattr(
        ha003, "rds_snapshot", lambda _settings: {"status": "available"}
    )

    def waited(_settings: Any, **kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
        calls.append(("wait", kwargs["previous_writer"], kwargs["timeout_seconds"]))
        return {"writer": "db-a"}, []

    monkeypatch.setattr(ha003, "wait_rds_failover", waited)
    bound = ha003.writer_restore_rds(settings)  # type: ignore[arg-type]

    assert bound.snapshot() == {"status": "available"}
    bound.fail_over_to("db-a")
    bound.wait_writer_left("db-b")

    assert calls == [
        (
            "failover-db-cluster",
            "--db-cluster-identifier",
            "unit-rds",
            "--target-db-instance-identifier",
            "db-a",
        ),
        ("wait", "db-b", restore.RESTORE_TIMEOUT_SECONDS),
    ]
