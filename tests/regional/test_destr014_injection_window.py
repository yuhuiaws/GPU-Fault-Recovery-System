"""DESTR-014 writes its two XIDs inside one aggregation window, node-c first.

An eks-managed run wrote them 9 s apart against a 5 s window: two host-probe
calls in a row, each bracketing its exec with three ownership reads. node-b's
XID 46 then reached a job our own STOP_WORKLOADS had already stopped (the
watcher records that stop at once), opened a workflow of its own, and the case
judged 28 consequences of the split. The runner overlaps the two calls now and
judges the pair on the stored events' own clocks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr014_branch_exhaustion as case
from tests.regional.test_destr014_branch_exhaustion import FAULT, SIBLING

NOW = datetime.now(timezone.utc)


class Probe:
    """One host probe: records each call's start and end in a shared log."""

    def __init__(
        self,
        log: list[tuple[str, ...]],
        name: str,
        *,
        on_start: Callable[[], None] | None = None,
        gate: threading.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self.log = log
        self.name = name
        self.on_start = on_start
        self.gate = gate
        self.error = error

    def execute(self, command: str, *arguments: str, **_: Any) -> dict[str, Any]:
        self.log.append(("start", self.name, command, *arguments))
        if self.on_start is not None:
            self.on_start()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never released this probe call"
        if self.error is not None:
            raise self.error
        self.log.append(("end", self.name))
        return {"ok": True}


def live_run(sibling: Probe, fault: Probe) -> Any:
    return SimpleNamespace(
        settings=SimpleNamespace(
            sibling_node=SIBLING,
            fault_node=FAULT,
            sibling_pci_bdf="0000:02:00",
            fault_pci_bdf="0000:01:00",
        ),
        marker="destr014-marker-a1",
        run_id="destr014-run",
        inject_sibling=sibling,
        inject_fault=fault,
    )


def starts(log: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [item[1:] for item in log if item[0] == "start"]


def test_node_b_is_written_while_node_c_is_still_inside_its_probe_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Live, one probe call is about 8 s of kubectl round trips; node-b's write
    # must not wait for it. node-b's start releases node-c's call, so the run
    # only finishes if the two calls overlapped.
    monkeypatch.setattr(case, "INJECTION_STAGGER_SECONDS", 0.05)
    log: list[tuple[str, ...]] = []
    gate = threading.Event()
    returned = case.inject_in_window(
        live_run(
            Probe(log, "node-c", gate=gate), Probe(log, "node-b", on_start=gate.set)
        )
    )
    assert starts(log) == [
        (
            "node-c",
            "write-xid79",
            "--marker",
            "destr014-marker-a1",
            "--drill-id",
            "destr014-run-c",
            "--pci-bdf",
            "0000:02:00",
        ),
        (
            "node-b",
            "write-xid46",
            "--marker",
            "destr014-marker-a1",
            "--drill-id",
            "destr014-run-b",
            "--pci-bdf",
            "0000:01:00",
        ),
    ]
    node_b_start = next(
        i for i, item in enumerate(log) if item[:2] == ("start", "node-b")
    )
    assert node_b_start < log.index(("end", "node-c")), log
    assert sorted(returned) == sorted([SIBLING, FAULT])


def test_node_b_starts_at_once_when_node_c_returned_before_the_stagger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(case, "INJECTION_STAGGER_SECONDS", 30.0)
    log: list[tuple[str, ...]] = []
    started = time.monotonic()
    case.inject_in_window(live_run(Probe(log, "node-c"), Probe(log, "node-b")))
    assert time.monotonic() - started < 5, "node-b waited out the stagger"
    assert [item[:2] for item in starts(log)] == [
        ("node-c", "write-xid79"),
        ("node-b", "write-xid46"),
    ]


def test_a_failed_node_c_write_never_writes_node_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(case, "INJECTION_STAGGER_SECONDS", 30.0)
    log: list[tuple[str, ...]] = []
    failing = Probe(log, "node-c", error=RuntimeError("kmsg write refused"))
    with pytest.raises(RuntimeError, match="kmsg write refused"):
        case.inject_in_window(live_run(failing, Probe(log, "node-b")))
    assert [item[0] for item in starts(log)] == ["node-c"], log


def stored(node: str, at: datetime | None) -> dict[str, Any]:
    return {"event": {"node_id": node, "observed_at": at.isoformat()} if at else {}}


def window_errors(sibling_at: datetime | None, fault_at: datetime | None) -> list[str]:
    return case.injection_window_errors(
        stored(SIBLING, sibling_at),
        stored(FAULT, fault_at),
        sibling_node=SIBLING,
        fault_node=FAULT,
        aggregation_window_seconds=5,
    )


def test_a_staggered_pair_inside_the_window_is_the_case_set_up() -> None:
    assert window_errors(NOW, NOW + timedelta(seconds=1.5)) == []


def test_the_sequential_pair_is_named_as_outside_the_window() -> None:
    assert window_errors(NOW, NOW + timedelta(seconds=9)) == [
        "the two injections are 9.0s apart, outside the 5s aggregation window"
    ]


def test_node_b_first_breaks_the_injection_order() -> None:
    assert window_errors(NOW + timedelta(seconds=1), NOW) == [
        "node-b's XID 46 was observed before node-c's XID 79; the injection "
        "order was not kept"
    ]


def test_an_event_without_its_own_clock_is_incomplete() -> None:
    assert window_errors(NOW, None) == ["injection timestamps are incomplete"]
