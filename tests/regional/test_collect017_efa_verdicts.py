"""COLLECT-017 A verdict: the control plane rebound the EFA function, not the
host fail-safe, and every documented reading matches.

Before 2026-09-07 the runner only asked ``workflow.status == SUCCEEDED``; a
300-second host timer could rebind the function first, the Node Agent would
report ``already_bound`` and the case would pass without the remediation path
ever having run on a real node.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_collect017_efa_plugin as collect017

BDF = "0000:6e:00.0"
OTHER = "0000:6f:00.0"
NODE = "hyperpod-node-a"


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
            # Remote steps keep empty details (observed live); the Node
            # Agent's result rides on the remote command instead.
            "details": {},
        }
        for i, op in enumerate(collect017.EFA_REMEDIATION_STEPS)
    ]
    remote_commands = [
        {
            "command_id": "remote-72ab5309d3c46bcab846b290",
            "step_index": 2,
            "operation": "REMEDIATE_EFA_DRIVER",
            "status": "SUCCEEDED",
            "error": None,
            "result_details": {
                "node_results": {
                    NODE: {
                        "expected_count": 2,
                        "pci_discovered_count": 2,
                        "driver_bound_count": 2,
                        "rebound_pci_bdfs": [BDF],
                        "already_bound": False,
                    }
                }
            },
        },
        {
            "command_id": "remote-ec6a25ad721fb94c7ba97b66",
            "step_index": 3,
            "operation": "RESTART_EFA_DEVICE_PLUGIN",
            "status": "SUCCEEDED",
            "error": None,
            "result_details": {"node_results": {NODE: {"already_healthy": True}}},
        },
    ]
    return {
        "workflow": {
            "request_id": "workflow-d6cb833b779638eaafc386e4",
            "status": "SUCCEEDED",
            "official_action": "REMEDIATE_EFA_DRIVER",
            "official_steps": steps,
            "step_executions": executions,
        },
        "remote_commands": remote_commands,
        "incident": {
            "state": "RECOVERED",
            # Observed live: official_action stays None, effective_action decides.
            "official_action": None,
            "effective_action": "REMEDIATE_EFA_DRIVER",
            "reasons": ["EFA PCI device is present but the efa driver is not bound"],
        },
    }


def _remediation(bundle: dict[str, Any]) -> dict[str, Any]:
    """The Node Agent's REMEDIATE_EFA_DRIVER result for NODE inside ``bundle``."""

    result = collect017.remote_node_result(
        bundle, operation="REMEDIATE_EFA_DRIVER", node=NODE
    )
    assert result is not None, "fixture bundle lost its remediation result"
    return result


def _move_result_to_other_node(bundle: dict[str, Any]) -> None:
    """The agent answered for a different node than the one the case chose."""

    node_results = bundle["remote_commands"][0]["result_details"]["node_results"]
    node_results["other-node"] = node_results.pop(NODE)


def _verdict(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "node": NODE,
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
    _remediation(bundle).update({"already_bound": True, "rebound_pci_bdfs": []})
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
            lambda b: _remediation(b).__setitem__("rebound_pci_bdfs", [OTHER]),
            "rebound_pci_bdfs",
        ),
        (
            lambda b: _remediation(b).__setitem__("driver_bound_count", 1),
            "driver_bound_count",
        ),
        (lambda b: b.__setitem__("remote_commands", []), "no Node Agent result"),
        (lambda b: _move_result_to_other_node(b), "no Node Agent result"),
        (lambda b: b["incident"].__setitem__("state", "QUARANTINED"), "incident state"),
        (
            lambda b: b["incident"].__setitem__("effective_action", "RUN_DIAGNOSTICS"),
            "incident action",
        ),
        (
            lambda b: b["workflow"].__setitem__("official_action", "RUN_DIAGNOSTICS"),
            "workflow official_action",
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


def _plugin_bundle(steps: tuple[str, ...]) -> dict[str, Any]:
    return {
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [{"operation": op} for op in steps],
            "step_executions": [
                {"operation": op, "status": "SUCCEEDED"} for op in steps
            ],
        },
        "incident": {"state": "RECOVERED"},
    }


@pytest.mark.parametrize(
    "steps", [collect017.GPU_PLUGIN_STEPS, collect017.EFA_PLUGIN_STEPS]
)
def test_plugin_workflow_verdict_follows_the_documented_steps(
    steps: tuple[str, ...],
) -> None:
    good = _plugin_bundle(steps)
    assert collect017.plugin_workflow_errors(good, steps=steps, label="x") == []

    extra = copy.deepcopy(good)
    extra["workflow"]["official_steps"].insert(1, {"operation": "STOP_WORKLOADS"})
    errors = collect017.plugin_workflow_errors(extra, steps=steps, label="x")
    assert any("official steps" in item for item in errors), errors

    failed_step = copy.deepcopy(good)
    failed_step["workflow"]["step_executions"][1]["status"] = "FAILED"
    failed_step["workflow"]["status"] = "FAILED"
    errors = collect017.plugin_workflow_errors(failed_step, steps=steps, label="x")
    assert any(steps[1] in item and "FAILED" in item for item in errors), errors
    assert any("workflow status" in item for item in errors), errors

    open_incident = copy.deepcopy(good)
    open_incident["incident"]["state"] = "QUARANTINED"
    errors = collect017.plugin_workflow_errors(open_incident, steps=steps, label="x")
    assert errors == ["x incident state 'QUARANTINED' != RECOVERED"], errors


def test_plugin_workflow_wait_outlives_two_collector_samples() -> None:
    # 15s interval x 2 consecutive samples, plus ingestion and planning.
    assert collect017.PLUGIN_WORKFLOW_PLAN_TIMEOUT_SECONDS >= 120, (
        "the DaemonSet must stay excluded long enough for the collector to report"
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def time(self) -> float:
        return 1_700_000_000 + self.now


def _settings() -> collect017.Settings:
    return collect017.Settings(
        regional=object(),  # type: ignore[arg-type]
        node=NODE,
        site_file=Path("/tmp/site.yaml"),
        host_probe_image="img",
        predecessor_path=Path("/tmp/p.json"),
    )


def test_efa_inventory_wait_short_circuits_instead_of_falling_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unbind that never shows up used to flow into a 600s workflow wait."""

    clock = _Clock()
    monkeypatch.setattr(collect017, "time", clock)
    reads = {"count": 0}

    class Collector:
        def efa_inventory(self) -> dict[str, Any]:
            reads["count"] += 1
            return _inventory(BDF, OTHER)

        def snapshot(self) -> dict[str, Any]:
            raise AssertionError("the wait must use the light efa-inventory read")

    with pytest.raises(collect017.RegionalFixtureError, match="did not reach 1"):
        collect017.wait_efa_inventory(
            Collector(),  # type: ignore[arg-type]
            discovered_count=1,
            timeout_seconds=30,
        )
    assert reads["count"] > 1 and set(clock.sleeps) == {3}

    last = collect017.wait_efa_inventory(
        Collector(),  # type: ignore[arg-type]
        discovered_count=1,
        timeout_seconds=6,
        required=False,
    )
    assert last["discovered_count"] == 2, "the recovery read reports what it saw"


def test_restore_efa_failure_never_masks_the_case_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collect017, "time", _Clock())
    calls: list[str] = []

    class Collector:
        def snapshot(self) -> dict[str, Any]:
            return {"efa_inventory": _inventory(BDF, OTHER)}

        def efa_inventory(self) -> dict[str, Any]:
            return _inventory(BDF, OTHER)  # the unbind never takes

        def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
            calls.append(arguments[0])
            if arguments[0] == "restore-efa":
                raise RuntimeError("kubectl exec lost the pod")
            return {}

    with pytest.raises(collect017.RegionalFixtureError, match="did not reach"):
        collect017.run_efa_unbind(
            _settings(),
            object(),  # type: ignore[arg-type]
            Collector(),  # type: ignore[arg-type]
            1,
        )
    assert calls == ["unbind-efa", "restore-efa"], "the fail-safe is still disarmed"


def test_gpu_plugin_daemonset_is_restored_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restores = {"count": 0}

    class Plugin:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def discover(self) -> dict[str, Any]:
            return {"name": "nvidia-device-plugin"}

        def exclude_node(self) -> None:
            pass

        def wait_allocatable(self, expected: int) -> dict[str, Any]:
            return {"allocatable": expected}

        def restore(self) -> None:
            restores["count"] += 1

    class Regional:
        def node_snapshot(self, _node: str) -> dict[str, Any]:
            return {"gpu_allocatable": 8}

    good = _plugin_bundle(collect017.GPU_PLUGIN_STEPS)
    monkeypatch.setattr(collect017.base, "DevicePluginFixture", Plugin)
    monkeypatch.setattr(
        collect017, "_wait_planned_workflow", lambda *_, **__: {"request_id": "wf"}
    )
    monkeypatch.setattr(collect017.base, "latest_node_workflow", lambda *_, **__: good)

    result = collect017.run_gpu_plugin(_settings(), Regional())  # type: ignore[arg-type]
    assert result["errors"] == []
    assert restores["count"] == 1, "no second rollout wait on the success path"

    def never_planned(*_: Any, **__: Any) -> dict[str, Any]:
        raise collect017.RegionalFixtureError("no workflow planned")

    monkeypatch.setattr(collect017, "_wait_planned_workflow", never_planned)
    with pytest.raises(collect017.RegionalFixtureError):
        collect017.run_gpu_plugin(_settings(), Regional())  # type: ignore[arg-type]
    assert restores["count"] == 2, "the finally restores when the success path did not"
