"""DESTR-010 Fabric Manager restart: preflight refusals, ledger and the finally steps."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from tests.regional.test_destr010_fabric_reset_shape import (
    AFTER,
    BASELINE,
    RESTORE_COMMAND_ID,
)

NODE = {
    "name": "node-a",
    "uid": "uid-node",
    "ready": "True",
    "unschedulable": False,
    "taints": [],
}


def preflight_frame() -> dict[str, Any]:
    return {
        "node": dict(NODE),
        "workloads": [],
        "state": {
            "agent": {
                "lifecycle_state": "ACTIVE",
                "allowed_operations": ["RESTART_FABRIC_MANAGER"],
            },
            "profile": {
                "warnings": [],
                "capabilities": [
                    {
                        "capability": "fabricManagerRestart",
                        "mode": "OWN",
                        "owner": "gpu-fault-node-agent",
                        "adapter": "node-action",
                    }
                ],
            },
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        },
        "fabric": {"active_workflow_incidents": [], "recent_xid_events": []},
    }


def test_normalize_notifications_skips_entries_that_are_not_records() -> None:
    assert destr010.normalize_notifications(
        [
            None,
            "notification-a",
            {
                "notification": {
                    "notification_id": "n-1",
                    "category": "INCIDENT",
                    "subject": "XID 45 on node-a",
                },
                "result": {"status": "SENT", "provider_message_id": "m-1"},
            },
            {"notification_id": "n-2", "status": "PENDING"},
        ]
    ) == [
        {
            "notification_id": "n-1",
            "category": "INCIDENT",
            "subject": "XID 45 on node-a",
            "drill_id": None,
            "status": "SENT",
            "reason": None,
            "provider_message_id": "m-1",
            "provider_message_id_present": True,
        },
        {"notification_id": "n-2", "status": "PENDING"},
    ]


def test_preflight_refuses_xid_events_inside_the_companion_window() -> None:
    frame = preflight_frame()
    assert destr010.validate_preflight(**frame) == [], frame
    frame["fabric"]["recent_xid_events"] = [{"xid": 45, "observed_at": "t"}]

    assert destr010.validate_preflight(**frame) == [
        "target node has XID events inside the companion window"
    ]


def test_read_only_preflight_records_failed_focused_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = preflight_frame()

    class Regional:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "rel-1", "cluster_id": "cluster-a"}

        def store_snapshot(self, *, node: str) -> dict[str, Any]:
            return {**copy.deepcopy(frame["state"]), "release_id": "rel-1"}

        def node_snapshot(self, node: str) -> dict[str, Any]:
            return dict(frame["node"])

        def business_workloads(self, node: str) -> list[dict[str, str]]:
            return []

    reused: list[tuple[Path, bool]] = []
    monkeypatch.setattr(destr010, "RegionalLiveFixture", Regional)
    monkeypatch.setattr(
        destr010, "fabric_probe", lambda regional, node: dict(frame["fabric"])
    )
    monkeypatch.setattr(destr010.xid45_opt_in, "switch_survey", lambda regional: None)
    monkeypatch.setattr(
        destr010,
        "focused_tests",
        lambda case_dir, *, reuse: reused.append((case_dir, reuse))
        or {"passed": False, "returncode": 1},
    )
    settings = SimpleNamespace(regional=object(), node="node-a")

    result = destr010.read_only_preflight(settings, tmp_path, reuse_focused_tests=True)

    assert result["errors"] == ["focused regression tests failed"]
    assert result["release_id"] == "rel-1"
    assert reused == [(tmp_path, True)]
    written = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert written["errors"] == ["focused regression tests failed"]


def test_a_chain_ledger_without_the_client_verification_is_named() -> None:
    after = json.loads(json.dumps(AFTER))
    after["ledger"] = [
        row for row in after["ledger"] if row["operation"] != "VERIFY_NO_GPU_CLIENTS"
    ]

    errors = destr010.host_errors(
        BASELINE, after, after, RESTORE_COMMAND_ID, shape=shapes.FABRIC_RESET
    )

    assert "Node Agent ledger shows no client verification" in errors, errors


class FinallyFixture:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, action: str, *, timeout: int) -> dict[str, Any]:
        self.calls.append(action)
        return {"ActiveState": "active"}

    def cleanup(self) -> dict[str, bool]:
        self.calls.append("cleanup")
        return {"pod": False}


def test_finish_case_fails_a_node_left_cordoned_whatever_shape_errors_has() -> None:
    fixture = FinallyFixture()
    cordoned = {**NODE, "unschedulable": True}
    regional = SimpleNamespace(node_snapshot=lambda node: dict(cordoned))
    result: dict[str, Any] = {"verdict": "PASS", "errors": "kept by the phase"}

    destr010.finish_case(fixture, regional, "node-a", dict(NODE), result)

    assert result["verdict"] == "FAIL"
    assert result["errors"] == "kept by the phase"
    assert result["final_node"] == cordoned
    assert fixture.calls == ["ensure-fabric-manager-active", "cleanup"]


@pytest.fixture
def opt_in(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``xid45_opt_in``'s window close and comparison, scripted per test."""

    script: dict[str, Any] = {"closed": {"state": "CLOSED"}, "errors": []}

    def close_window(regional: Any, case_dir: Path) -> Any:
        if isinstance(script["closed"], Exception):
            raise script["closed"]
        return script["closed"]

    def window_errors(opened: Any, closed: Any) -> list[str]:
        script["compared"] = (opened, closed)
        return list(script["errors"])

    monkeypatch.setattr(destr010.xid45_opt_in, "close_window", close_window)
    monkeypatch.setattr(destr010.xid45_opt_in, "window_errors", window_errors)
    return script


def test_close_opt_in_fails_the_case_when_the_window_cannot_be_closed(
    opt_in: dict[str, Any], tmp_path: Path
) -> None:
    opt_in["closed"] = RuntimeError("ConfigMap patch conflict")
    result: dict[str, Any] = {"verdict": "PASS"}

    destr010.close_opt_in(object(), tmp_path, result)

    assert result == {
        "verdict": "FAIL",
        "opt_in_close_error": "RuntimeError: ConfigMap patch conflict",
    }


def test_close_opt_in_records_a_window_that_did_not_restore(
    opt_in: dict[str, Any], tmp_path: Path
) -> None:
    opt_in["errors"] = ["the switch kept the opt-in value"]
    result: dict[str, Any] = {
        "verdict": "PASS",
        "errors": [],
        "observe_only": {"opt_in_window": {"state": "OPEN"}},
    }

    destr010.close_opt_in(object(), tmp_path, result)

    assert result["verdict"] == "FAIL"
    assert result["errors"] == ["the switch kept the opt-in value"]
    assert result["opt_in_window_closed"] == "CLOSED"
    assert opt_in["compared"] == ({"state": "OPEN"}, {"state": "CLOSED"})


def test_close_opt_in_keeps_a_non_list_error_field_and_still_fails(
    opt_in: dict[str, Any], tmp_path: Path
) -> None:
    opt_in["errors"] = ["the switch kept the opt-in value"]
    result: dict[str, Any] = {"verdict": "PASS", "errors": None}

    destr010.close_opt_in(object(), tmp_path, result)

    assert (result["verdict"], result["errors"]) == ("FAIL", None)
    assert opt_in["compared"] == (None, {"state": "CLOSED"})
