"""A run's data settles before its purge; what lands after it is purged too.

Live 2026-10-14: action-raw-32x4 purged its data (total 0) and deregistered;
57 s later the correlated suite's registration found 117 records and 39 links
on the same synthetic clusters and refused. They were 39 drill notifications
of the action run (each with its delivery and result, plus the deduplication
link) that the collector-silence scan wrote after the purge committed: the scan
had listed the run's Agents before. The shared purge now waits for the run's
data to settle, probes again after the purge and purges what landed late,
reporting it by kind; a registration adopts an earlier run's late residue only
when it is provably that run's drill residue.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_data as data
from scripts.perf import regional_capacity_settle as settle
from tests.regional._perf_caller_support import CapacityWire, action_args

QUIET = settle.QUIET_SECONDS
CLUSTERS = ["perf-cap-000", "perf-cap-001"]


class RunData:
    """The synthetic clusters' rows as the probe and the purge see them."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()
        self.sources: Counter[str] = Counter()
        self.links = 0
        self.tables: dict[str, int] = {}
        self.nonterminal = 0
        self.open_queue_rows = 0
        self.owner: str | None = None
        self.proven = False
        # Writers that land once a purge has committed, one per purge.
        self.after_purge: list[Callable[[RunData], None]] = []
        # Writers that land before the next probe, one per probe.
        self.before_probe: list[Callable[[RunData], None]] = []
        self.calls: list[tuple[str, str, bool]] = []

    def notifications(self, count: int, source: str = "collector-silent") -> None:
        """What one ``save_notification_if_absent`` plus a delivered send leave."""

        for kind in ("notification", "notification_delivery", "notification_result"):
            self.kinds[kind] += count
        self.sources[source] += count
        self.links += count

    def probe(self) -> dict[str, Any]:
        records = sum(self.kinds.values())
        return {
            "records": records,
            "links": self.links,
            **self.tables,
            "total": records + self.links + sum(self.tables.values()),
            "by_kind": dict(self.kinds),
            "notification_sources": dict(self.sources),
            "nonterminal": self.nonterminal,
            "open_queue_rows": self.open_queue_rows,
            "residue_owner": self.owner,
            "residue_owner_proven": self.proven,
        }

    def invoke(
        self,
        _control: Any,
        *,
        run_id: str,
        cluster_ids: list[str],
        cleanup: bool,
        force_nonterminal: bool = False,
    ) -> dict[str, Any]:
        assert cluster_ids == CLUSTERS, "the run's exact clusters"
        self.calls.append(("purge" if cleanup else "probe", run_id, force_nonterminal))
        if not cleanup:
            if self.before_probe:
                self.before_probe.pop(0)(self)
            return self.probe()
        removed = {
            "run_id": run_id,
            "cluster_ids": cluster_ids,
            "total": 0,
            "deleted_records": sum(self.kinds.values()),
            "deleted_links": self.links,
            "forced_nonterminal": self.nonterminal if force_nonterminal else 0,
            "dropped_partitions": [],
            "deleted_by_kind": dict(self.kinds),
            "deleted_table_rows": {k: v for k, v in self.tables.items() if v},
            "notification_sources": dict(self.sources),
        }
        self.kinds.clear()
        self.sources.clear()
        self.links = 0
        self.tables = {}
        self.nonterminal = self.open_queue_rows = 0
        if self.after_purge:
            self.after_purge.pop(0)(self)
        return removed


def finished_run() -> RunData:
    """An action run at teardown: everything terminal, its advisories delivered."""

    rows = RunData()
    rows.kinds.update({"workflow": 4, "remote_command": 40, "incident": 4, "agent": 16})
    rows.notifications(48)
    rows.tables = {"gpu_fault_processor_queue": 0, "gpu_fault_gpu_metric_latest": 0}
    return rows


def run_artifacts(tmp_path: Path, run_id: str = "act-a") -> Path:
    (tmp_path / "registry-registration-intent.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cluster_ids": CLUSTERS,
                "data_empty_before_registration": True,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def purge(
    tmp_path: Path, rows: RunData, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[float], list[str]]:
    monkeypatch.setattr(data, "invoke", rows.invoke)
    sleeps: list[float] = []
    logs: list[str] = []
    document = settle.purge_run_data(
        lambda *_args, **_kwargs: "",
        run_id="act-a",
        artifacts=run_artifacts(tmp_path),
        log=logs.append,
        sleep=sleeps.append,
    )
    return document, sleeps, logs


def test_a_late_write_after_the_purge_is_caught_and_purged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = finished_run()
    # The collector-silence scan listed 13 of the run's Agents before the purge
    # committed and writes their advisories after it.
    rows.after_purge = [lambda late: late.notifications(13)]

    document, sleeps, logs = purge(tmp_path, rows, monkeypatch)

    assert [call[0] for call in rows.calls] == [
        "probe",  # settle: the run's data as the run left it
        "probe",  # ... the same a quiet window later
        "purge",
        "probe",  # a quiet window after the purge: the late advisories
        "probe",  # settle again: nothing else is landing
        "purge",
        "probe",  # nothing more
    ], "settle, purge, catch the late write, purge again, prove it gone"
    assert sleeps == [QUIET] * 4, "every wait is one quiet window"
    assert all(call[2] for call in rows.calls if call[0] == "purge"), (
        "both purges may remove open synthetic work (the executors are gone)"
    )
    assert logs == [
        "run act-a data appeared after its purge; purge 2 removed "
        "notification=13 [collector-silent 13], notification_delivery=13, "
        "notification_result=13; links 13"
    ], "the late purge names what it removed by kind, source and links"
    recorded = json.loads((tmp_path / "capacity-data-cleanup.json").read_text())
    assert recorded["settle"]["outcome"] == "settled"
    assert recorded["deleted_by_kind"]["notification"] == 48, "the first purge"
    assert [item["summary"] for item in recorded["late_purges"]] == [
        logs[0].split("removed ", 1)[1]
    ], "capacity-data-cleanup.json keeps every later purge"


def test_late_writes_that_never_stop_fail_the_teardown_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = finished_run()
    rows.after_purge = [lambda late: late.notifications(2)] * 10

    with pytest.raises(RuntimeError, match="kept appearing after 4 purges"):
        purge(tmp_path, rows, monkeypatch)

    assert [call[0] for call in rows.calls].count("purge") == settle.LATE_PURGES + 1, (
        "the late purges are bounded"
    )
    assert not (tmp_path / "capacity-data-cleanup.json").exists(), (
        "a cleanup still racing a writer is no cleanup record"
    )


def test_the_purge_waits_while_the_runs_data_is_still_moving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = finished_run()
    rows.nonterminal = 3
    rows.open_queue_rows = 2

    def finish(late: RunData) -> None:
        late.nonterminal = late.open_queue_rows = 0
        late.kinds["workflow_event"] += 3

    # The run's last workflows end and their results land over two windows.
    rows.before_probe = [lambda _rows: None, finish, lambda late: late.notifications(1)]

    document, sleeps, _logs = purge(tmp_path, rows, monkeypatch)

    assert document["settle"] == {
        "outcome": "settled",
        "probes": 4,
        "waited_seconds": 3 * QUIET,
        "last": document["settle"]["last"],
        "open_work": False,
    }, "open work, then changing counts, then two equal probes: three windows"
    assert sleeps == [QUIET] * 4, "the three settle windows and the one after the purge"


def test_open_work_that_nothing_moves_stops_the_wait_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = finished_run()
    rows.nonterminal = 5  # LEASED commands of executors that were deleted

    document, sleeps, logs = purge(tmp_path, rows, monkeypatch)

    assert document["settle"]["outcome"] == "stuck"
    assert document["settle"]["waited_seconds"] == settle.STUCK_WINDOWS * QUIET, (
        "two unchanged windows prove nobody will finish the open work"
    )
    assert document["forced_nonterminal"] == 5, "the purge removes it as before"
    assert logs[0].startswith("run act-a data did not settle (stuck after 60 s"), logs
    assert sleeps == [QUIET] * (settle.STUCK_WINDOWS + 1), (
        "the stuck wait, then the quiet window after the purge"
    )


def test_an_empty_run_purges_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = RunData()

    document, sleeps, logs = purge(tmp_path, rows, monkeypatch)

    assert document["settle"]["outcome"] == "empty"
    assert (sleeps, logs) == ([], []), "nothing there, nothing in flight"
    assert [call[0] for call in rows.calls] == ["probe", "purge"]


def test_the_settle_wait_is_bounded_by_counted_windows() -> None:
    rows = finished_run()
    rows.before_probe = [lambda late: late.notifications(1)] * 50
    sleeps: list[float] = []

    outcome = settle.wait_until_settled(
        lambda: rows.invoke(None, run_id="act-a", cluster_ids=CLUSTERS, cleanup=False),
        sleep=sleeps.append,
    )

    assert outcome["outcome"] == "bound"
    assert sum(sleeps) == settle.SETTLE_SECONDS, "at most the settle bound"


def residue() -> RunData:
    """The 2026-10-14 residue: 39 drill advisories of the deregistered action run."""

    rows = RunData()
    rows.notifications(39)
    rows.owner = "act20261014T054453"
    rows.proven = True
    return rows


def test_a_registration_adopts_an_earlier_runs_proven_drill_residue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = residue()
    monkeypatch.setattr(data, "invoke", rows.invoke)
    logs: list[str] = []
    sleeps: list[float] = []
    found = rows.invoke(None, run_id="corract-b", cluster_ids=CLUSTERS, cleanup=False)

    adopted = settle.adopt_earlier_run_residue(
        lambda *_args, **_kwargs: "",
        run_id="corract-b",
        cluster_ids=CLUSTERS,
        found=found,
        log=logs.append,
        sleep=sleeps.append,
    )

    assert rows.calls == [
        ("probe", "corract-b", False),  # the registration's data preflight
        ("probe", "act20261014T054453", False),  # the same a quiet window later
        ("purge", "act20261014T054453", False),
        ("probe", "corract-b", False),
    ], "settle, then that run's own purge, then the registration's proof"
    assert adopted["total"] == 0, "the registration starts on empty clusters"
    assert adopted["adopted_residue"]["owner_run_id"] == "act20261014T054453"
    assert adopted["adopted_residue"]["purged"]["deleted_by_kind"] == {
        "notification": 39,
        "notification_delivery": 39,
        "notification_result": 39,
    }
    assert "drill residue of earlier run act20261014T054453" in logs[0], logs
    assert logs[-1].startswith("purged earlier run act20261014T054453's residue: "), (
        logs
    )
    assert sleeps == [QUIET], "it settles before the purge"


@pytest.mark.parametrize(
    ("owner", "proven", "why"),
    [
        (None, False, "no earlier run in the registry history owns it"),
        ("act-a", False, "not provably run act-a's drill residue"),
        ("corract-b", True, "not provably run corract-b's drill residue"),
    ],
    ids=["no-owner", "not-drill-residue", "own-run"],
)
def test_residue_that_is_not_provably_an_earlier_runs_still_refuses(
    monkeypatch: pytest.MonkeyPatch, owner: str | None, proven: bool, why: str
) -> None:
    rows = residue()
    rows.owner, rows.proven = owner, proven
    rows.tables = {"gpu_fault_gpu_metric_latest": 3}
    monkeypatch.setattr(data, "invoke", rows.invoke)

    with pytest.raises(RuntimeError) as caught:
        settle.adopt_earlier_run_residue(
            lambda *_args, **_kwargs: "",
            run_id="corract-b",
            cluster_ids=CLUSTERS,
            found=rows.probe(),
            log=lambda _line: None,
            sleep=lambda _seconds: None,
        )

    message = str(caught.value)
    assert message.startswith(settle.REFUSAL), message
    assert why in message and "gpu_fault_gpu_metric_latest=3" in message, message
    assert rows.calls == [], "nothing is purged"


def test_the_action_suite_registers_after_adopting_an_earlier_runs_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    rows = residue()
    rows.owner = "act-earlier"

    def invoke(control: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs["run_id"] == "act-earlier" or (
            not kwargs["cleanup"] and rows.probe()["total"]
        ):
            kwargs["cluster_ids"] = CLUSTERS
            return rows.invoke(control, **kwargs)
        return wire.data(control, **kwargs)

    monkeypatch.setattr(data, "invoke", invoke)

    assert action.main(action_args(tmp_path)) == 0, "the run proceeds on clean clusters"

    preflight = json.loads((tmp_path / "registry-data-preflight.json").read_text())
    assert preflight["adopted_residue"]["owner_run_id"] == "act-earlier"
    intent = json.loads((tmp_path / "registry-registration-intent.json").read_text())
    assert intent["data_empty_before_registration"] is True
    assert ("purge", "act-earlier", False) in rows.calls, "the earlier run's purge"
