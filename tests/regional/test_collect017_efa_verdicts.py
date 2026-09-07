"""COLLECT-017 A verdict: the control plane rebound the EFA function, not the
host fail-safe, and every documented reading matches.

Before 2026-09-07 the runner only asked ``workflow.status == SUCCEEDED``; a
300-second host timer could rebind the function first, the Node Agent would
report ``already_bound`` and the case would pass without the remediation path
ever having run on a real node.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.e2e.regional import run_collect017_efa_plugin as collect017

BDF = "0000:6e:00.0"
OTHER = "0000:6f:00.0"


def _inventory(*bdfs: str, active: int | None = None) -> dict[str, Any]:
    devices = [
        {"name": f"rdmap{i}", "pci_bdf": bdf, "driver": "efa", "active": True}
        for i, bdf in enumerate(bdfs)
    ]
    return {
        "discovered_count": len(devices),
        "active_count": len(devices) if active is None else active,
        "devices": devices,
    }


def _bundle() -> dict[str, Any]:
    steps = [
        {"operation": op, "step_index": i}
        for i, op in enumerate(collect017.EFA_REMEDIATION_STEPS)
    ]
    executions = [
        {
            "step_index": i,
            "operation": op,
            "status": "SUCCEEDED",
            "adapter_operation_id": (
                f"remote/remote-{i:06x}"
                if op in {"REMEDIATE_EFA_DRIVER", "RESTART_EFA_DEVICE_PLUGIN"}
                else f"workflow-{i:06x}"
            ),
            "details": (
                {
                    "expected_count": 2,
                    "pci_discovered_count": 2,
                    "driver_bound_count": 2,
                    "rebound_pci_bdfs": [BDF],
                    "already_bound": False,
                }
                if op == "REMEDIATE_EFA_DRIVER"
                else {}
            ),
        }
        for i, op in enumerate(collect017.EFA_REMEDIATION_STEPS)
    ]
    return {
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": steps,
            "step_executions": executions,
        },
        "incident": {
            "state": "RECOVERED",
            "official_action": "REMEDIATE_EFA_DRIVER",
            "reasons": ["EFA PCI device is present but the efa driver is not bound"],
        },
    }


def _verdict(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "bdf": BDF,
        "baseline": _inventory(BDF, OTHER),
        "unbound": _inventory(OTHER),
        "recovered": _inventory(BDF, OTHER),
        "restore": {
            "pci_bdf": BDF,
            "bound": True,
            "already_bound": True,
            "timer_was_active": True,
            "timer_fired": False,
        },
    }
    bundle = overrides.pop("bundle", None) or _bundle()
    arguments.update(overrides)
    return collect017.efa_unbind_errors(bundle, **arguments)


def test_documented_success_path_has_no_errors() -> None:
    assert _verdict() == []


def test_fail_safe_timer_rebinding_is_not_a_pass() -> None:
    # The timer fired and bound the function; the Node Agent then found it
    # already bound. Both readings must fail the case on their own.
    errors = _verdict(
        restore={
            "pci_bdf": BDF,
            "bound": True,
            "already_bound": True,
            "timer_was_active": False,
            "timer_fired": True,
        }
    )
    assert any("fail-safe timer" in item for item in errors), errors

    bundle = _bundle()
    step = bundle["workflow"]["step_executions"][2]
    step["details"] = {
        "expected_count": 2,
        "pci_discovered_count": 2,
        "driver_bound_count": 2,
        "already_bound": True,
    }
    errors = _verdict(bundle=bundle)
    assert any("already bound" in item for item in errors), errors
    assert any("rebound_pci_bdfs" in item for item in errors), errors


def test_restore_having_to_bind_is_not_a_pass() -> None:
    errors = _verdict(
        recovered=_inventory(OTHER),
        restore={
            "pci_bdf": BDF,
            "bound": True,
            "already_bound": False,
            "timer_was_active": True,
            "timer_fired": False,
        },
    )
    assert any("restore-efa had to bind" in item for item in errors), errors
    assert any("not bound to efa after the workflow" in item for item in errors), errors
    assert any("!= baseline 2" in item for item in errors), errors


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda b: b["workflow"].__setitem__("status", "FAILED"), "workflow status"),
        (lambda b: b["workflow"]["official_steps"].pop(2), "official steps"),
        (lambda b: b["workflow"].__setitem__("step_executions", []), "never executed"),
        (
            lambda b: b["workflow"]["step_executions"][2].__setitem__(
                "adapter_operation_id", "workflow-local"
            ),
            "remote node action",
        ),
        (
            lambda b: b["workflow"]["step_executions"][2]["details"].__setitem__(
                "rebound_pci_bdfs", [OTHER]
            ),
            "rebound_pci_bdfs",
        ),
        (
            lambda b: b["workflow"]["step_executions"][2]["details"].__setitem__(
                "driver_bound_count", 1
            ),
            "driver_bound_count",
        ),
        (lambda b: b["incident"].__setitem__("state", "QUARANTINED"), "incident state"),
        (
            lambda b: b["incident"].__setitem__("official_action", "RUN_DIAGNOSTICS"),
            "official_action",
        ),
        (
            lambda b: b["incident"].__setitem__("reasons", ["link down"]),
            "DRIVER_UNBOUND",
        ),
    ],
)
def test_each_documented_reading_is_checked(mutate, fragment: str) -> None:
    bundle = copy.deepcopy(_bundle())
    mutate(bundle)
    errors = _verdict(bundle=bundle)
    assert any(fragment in item for item in errors), (fragment, errors)


def test_collector_must_see_the_function_leave_and_return() -> None:
    errors = _verdict(unbound=_inventory(BDF, OTHER))
    assert any("did not observe the unbound function" in item for item in errors), (
        errors
    )
    assert any("still bound to efa after the unbind" in item for item in errors), errors

    errors = _verdict(recovered=_inventory(BDF, OTHER, active=1))
    assert errors == ["ACTIVE 1 != baseline 2"], errors


def test_fail_safe_outlives_the_workflow_and_incident_waits() -> None:
    assert collect017.EFA_RESTORE_SECONDS > (
        collect017.EFA_WORKFLOW_TIMEOUT_SECONDS
        + collect017.EFA_INCIDENT_TIMEOUT_SECONDS
    ), "a shorter fail-safe rebinds the function before the control plane can"
