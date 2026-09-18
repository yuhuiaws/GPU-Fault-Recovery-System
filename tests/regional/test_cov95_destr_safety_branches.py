from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from scripts.e2e.regional import run_destr012_managed_recovery_guard as managed
from scripts.e2e.regional import run_destr016_preempting_reboot as preemption
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import SPARE_LABEL
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_branches import BranchHarness
from tests.regional._cov95_destr_edge_data import change
from tests.regional._cov95_destr_exhaustion import (
    RECOVERY_IDENTITY_REFUSAL,
    ExhaustionHarness,
)
from tests.regional._cov95_destr_lifetime import LifetimeHarness
from tests.regional._cov95_destr_managed import ManagedHarness
from tests.regional.test_destr015_parallel_branch_join import NODES


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("first", "fabric_manager", "ActiveState"), "inactive", "first action"),
        (("first", "fabric_manager", "MainPID"), "100", "MainPID did not change"),
        (
            ("first", "fabric_manager", "InvocationID"),
            "old",
            "InvocationID did not change",
        ),
        (("first", "journal", "started_count"), 2, "exactly one Fabric Manager start"),
        (
            ("replay", "fabric_manager", "MainPID"),
            "300",
            "MainPID changed during replay",
        ),
        (("replay", "fabric_manager", "ActiveState"), "failed", "active after replay"),
        (("replay", "gpu_fault_timers"), ["foreign.timer"], "timer inventory changed"),
    ],
)
def test_fabric_host_contract_rejects_restart_or_replay_drift(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = ActionHarness(fabric, tmp_path, monkeypatch)
    frame = {
        "before": h.host.execute("snapshot"),
        "first": h.host.execute("snapshot", "--since", "unit"),
        "replay": h.host.execute("snapshot", "--since", "unit"),
    }
    command = "key-fabric/node-a/agent-3"
    assert (
        fabric.host_errors(frame["before"], frame["first"], frame["replay"], command)
        == []
    ), frame
    change(frame, path, value)
    errors = fabric.host_errors(
        frame["before"], frame["first"], frame["replay"], command
    )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("kmsg_writable",), False, "not writable"),
        (("compute_clients",), [{"pid": "foreign"}], "active NVIDIA compute clients"),
        (("fabric_manager", "ActiveState"), "inactive", "not active at baseline"),
    ],
)
def test_fabric_execute_refuses_invalid_host_before_injection(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = ActionHarness(fabric, tmp_path, monkeypatch)
    h.plan(tmp_path)

    def host_result(result: dict[str, Any], args: tuple[Any, ...], _kwargs: Any) -> Any:
        if args == ("snapshot",):
            change(result, path, value)
        return result

    h.transforms["host.execute"] = host_result
    code, report = h.execute(tmp_path)
    assert code == 1 and expected in report["error"], report
    assert not any(
        name == "host.execute" and args[0] == "write-xid45" for name, args, _ in h.calls
    ), h.calls
    assert any(name == "host.cleanup" for name, _, _ in h.calls), h.calls


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("step_executions",), [], "STOP_WORKLOADS did not fail"),
        (
            ("step_executions", 0, "details", "managed_job_recovery_workloads"),
            ["wl-d", "foreign"],
            "only the violating workload",
        ),
        (
            ("step_executions", 0, "details", "required_annotation"),
            "foreign",
            "wrong required annotation",
        ),
        (
            ("step_executions", 0, "details", "required_annotation_value"),
            "true",
            "wrong required annotation value",
        ),
        (
            ("step_executions", 0, "details", "remediation_commands"),
            ["kubectl unrelated"],
            "remediation command is missing",
        ),
    ],
)
def test_managed_recovery_failure_must_identify_exact_remediation(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    assert (
        managed.group_d_failure_errors(h.blocked, expected_workload_id="wl-d") == []
    ), h.blocked
    change(h.blocked["workflow"], path, value)
    errors = managed.group_d_failure_errors(h.blocked, expected_workload_id="wl-d")
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("node", "ready"), "False", "target node is not Ready"),
        (("store", "agent", "lifecycle_state"), "DRAINING", "not ACTIVE"),
        (("store", "profile", "capabilities"), [], "no gpuReset"),
        (("store", "profile", "warnings"), ["drift"], "profile has warnings"),
        (("store", "queue"), {"depth": 1}, "processor queue is not empty"),
        (("survey", "replicas"), [], "step waiting ceiling is unknown"),
    ],
)
def test_lifetime_preflight_cannot_compress_an_unproven_environment(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = LifetimeHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], h.calls
    frame = {"node": h.node, "store": h.store, "survey": h.survey_data}
    change(frame, path, value)
    preflight = h.plan(tmp_path)
    assert any(expected in error for error in preflight["errors"]), preflight
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not any(name == "window.open" for name, _ in h.calls), h.calls


@pytest.mark.parametrize(
    "defect", ["gpu-count", "critical-pod", "reset-capability", "warning"]
)
def test_parallel_preflight_refuses_incomplete_target_contract(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], h.calls
    if defect == "gpu-count":
        h.nodes[NODES[0]]["gpu_allocatable"] = 7
        expected = "does not allocate 8 GPUs"
    elif defect == "critical-pod":
        monkeypatch.setattr(
            h.regional,
            "business_workloads",
            lambda node: [{"node": node, "name": "critical"}],
        )
        expected = "critical system workload"
    elif defect == "reset-capability":
        h.profile["capabilities"] = [
            row for row in h.profile["capabilities"] if row["capability"] != "gpuReset"
        ]
        expected = "no gpuReset capability"
    else:
        h.profile["warnings"] = ["drift"]
        expected = "profile has warnings"
    preflight = h.plan(tmp_path)
    assert any(expected in error for error in preflight["errors"]), preflight
    assert not h.injected, h.calls


@pytest.mark.parametrize("defect", ["predecessor", "unready", "spare", "executor"])
def test_exhaustion_preflight_preserves_safeguard_refusal_and_reports_other_faults(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ExhaustionHarness(tmp_path, monkeypatch)
    assert h.plan(tmp_path)["errors"] == [], (
        "valid recovery identity must pass real preflight"
    )
    h.agents[h.settings.sibling_node].pop("installer_bundle_sha256")
    assert h.plan(tmp_path)["errors"] == [RECOVERY_IDENTITY_REFUSAL], h.calls
    if defect == "predecessor":
        h.predecessor_valid = False
        expected = "predecessor evidence is not PASS"
    elif defect == "unready":
        h.nodes[h.settings.fault_node]["ready"] = "False"
        expected = "not Ready and schedulable"
    elif defect == "spare":
        h.nodes[h.settings.fault_node]["labels"][SPARE_LABEL] = "true"
        expected = "labeled as a warm spare"
    else:
        monkeypatch.setattr(h.regional, "ready_pods", lambda *_a: [])
        expected = "no ready executor replica"
    preflight = h.plan(tmp_path)
    assert RECOVERY_IDENTITY_REFUSAL in preflight["errors"], preflight
    assert any(expected in error for error in preflight["errors"]), preflight
    assert not any(name.endswith(".open") for name, _ in h.calls), h.calls


@pytest.mark.parametrize(
    ("explicit", "inventory", "pci_bdf", "expected"),
    [
        ("/dev/nvidia3", [{"index": 3, "pci_bdf": "0000:03:00.0"}], "", "/dev/nvidia3"),
        ("", [{"index": 3, "pci_bdf": "0000:AB:00.0"}], "0000:ab:00.0", "/dev/nvidia3"),
    ],
)
def test_holder_device_selection_is_bound_to_observed_inventory(
    explicit: str, inventory: list[dict[str, Any]], pci_bdf: str, expected: str
) -> None:
    original = deepcopy(inventory)
    assert preemption.holder_device(explicit, inventory, pci_bdf=pci_bdf) == expected, (
        inventory
    )
    assert inventory == original, "device selection must not rewrite host inventory"


@pytest.mark.parametrize(
    ("explicit", "inventory", "pci_bdf", "expected"),
    [
        (
            "",
            [{"index": 0, "pci_bdf": "0000:01:00.0"}],
            "0000:02:00.0",
            "BDF is not unique",
        ),
        ("/dev/nvidia3", [{"index": 0}], "", "device is not in GPU inventory"),
        ("", [{"pci_bdf": "0000:01:00.0"}], "0000:01:00.0", "no device index"),
        (
            "/dev/nvidia3",
            [{"index": 0, "pci_bdf": "0000:01:00.0"}],
            "0000:01:00.0",
            "does not match",
        ),
    ],
)
def test_holder_device_selection_rejects_missing_or_conflicting_binding(
    explicit: str, inventory: list[dict[str, Any]], pci_bdf: str, expected: str
) -> None:
    with pytest.raises(RegionalFixtureError, match=expected):
        preemption.holder_device(explicit, inventory, pci_bdf=pci_bdf)
