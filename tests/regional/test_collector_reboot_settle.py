"""The post-reboot settle wait of COLLECT-004 (live 2026-09-28).

kubelet reports Ready right after the boot and HyperPod restarts it ~30 s
later; a probe Pod re-created in between loses its exec channel and the case
fails "inventory reboot failed: HostProbeError: command failed (1)". The wait
holds until the node has been Ready long enough on the new boot and the node
installer has reconciled that boot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.collector_inventory_reboot import (
    wait_node_settled_after_reboot,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _snapshot(
    *, ready: str = "True", ready_age: float = 120.0, installer: str | None = "boot-b"
) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(seconds=ready_age)
    return {
        "name": "node-a",
        "uid": "uid-a",
        "boot_id": "boot-b",
        "ready": ready,
        "ready_since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "installer_boot_id": installer,
    }


def _regional(snapshots: list[dict[str, Any]]) -> Any:
    rows = iter(snapshots)
    return SimpleNamespace(node_snapshot=lambda node: next(rows))


def test_a_node_ready_long_enough_and_reconciled_is_settled_at_once() -> None:
    clock = _Clock()
    result = wait_node_settled_after_reboot(
        _regional([_snapshot()]), node="node-a", boot_id="boot-b", clock=clock
    )
    assert result["polls"] == 1 and result["installer_reconciled"] is True, result
    assert clock.sleeps == [], "a settled node was waited on"


def test_the_wait_holds_across_the_second_kubelet_start() -> None:
    clock = _Clock()
    result = wait_node_settled_after_reboot(
        _regional(
            [
                _snapshot(ready_age=20.0, installer="boot-a"),  # kubelet's first start
                _snapshot(ready="False", ready_age=1.0, installer="boot-a"),
                _snapshot(
                    ready_age=10.0, installer="boot-b"
                ),  # second start, too fresh
                _snapshot(ready_age=50.0, installer="boot-b"),
            ]
        ),
        node="node-a",
        boot_id="boot-b",
        clock=clock,
    )
    assert result["polls"] == 4, result
    assert clock.sleeps == [5.0, 5.0, 5.0], clock.sleeps


def test_the_installer_must_have_reconciled_the_new_boot() -> None:
    clock = _Clock()
    with pytest.raises(RegionalFixtureError, match="did not settle"):
        wait_node_settled_after_reboot(
            _regional([_snapshot(installer="boot-a")] * 4),
            node="node-a",
            boot_id="boot-b",
            timeout_seconds=12.0,
            clock=clock,
        )
    assert clock.sleeps == [5.0, 5.0, 2.0], clock.sleeps


def test_a_further_boot_change_is_a_failure_not_a_wait() -> None:
    snapshot = _snapshot()
    snapshot["boot_id"] = "boot-c"
    with pytest.raises(RegionalFixtureError, match="boot changed again"):
        wait_node_settled_after_reboot(
            _regional([snapshot]), node="node-a", boot_id="boot-b", clock=_Clock()
        )


def test_older_projections_without_the_fields_are_accepted() -> None:
    result = wait_node_settled_after_reboot(
        _regional([{"boot_id": "boot-b", "ready": "True"}]),
        node="node-a",
        boot_id="boot-b",
        clock=_Clock(),
    )
    assert (
        result["installer_reconciled"] is False and result["ready_age_seconds"] is None
    )
