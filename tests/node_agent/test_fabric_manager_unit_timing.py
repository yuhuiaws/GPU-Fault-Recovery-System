"""Fabric Manager's start and the collector's start, timed to NVIDIA's unit (batch 131).

NVIDIA's R580 ``nvidia-fabricmanager.service`` allows ``TimeoutStartSec=720``:
on NVL5 (HGX B200/B300) its start script polls up to 60 s for the InfiniBand
device, starts the NVLink Subnet Manager and then Fabric Manager, in the one
unit. The Agent waited 120 s for every ``systemctl start``/``restart`` it ran,
so a Fabric Manager the unit was still legitimately starting failed the restore
or the restart. It now waits 780 s (the unit's bound plus a minute) for that
unit only; every other unit keeps 120 s.

The fabric_manager_status window (``gpu_fabric_manager``) counts from the
metrics collector's first observation and assumed "the collectors start after
Fabric Manager", which the unit did not say: it is now ordered
``After=nvidia-fabricmanager.service``, ordering only.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from gpu_fault.cluster_scope import (
    FABRIC_MANAGER_START_TIMEOUT_SECONDS,
    FABRIC_MANAGER_UNIT,
)
from gpu_fault.node_agent.common import (
    UNIT_START_TIMEOUT_SECONDS,
    unit_start_timeout_seconds,
)
from gpu_fault.node_agent.fabric_manager_restart import (
    START_TIMEOUT_SECONDS,
    start_fabric_manager_after_reset,
)

from ._deployment_support import ROOT
from ._support import (
    CompletedProcess,
    GpuServiceQuiesceManager,
    NodeActionStatus,
    ServiceRunner,
    WorkflowOperation,
    command,
    envelope,
    node_action_executor,
)

FM = FABRIC_MANAGER_UNIT
SERVICES = ("nvidia-fabricmanager", "nvidia-dcgm", "gpu-fault-metrics-collector")


class TimedRunner(ServiceRunner):
    """``ServiceRunner`` that records the timeout of every call it answers."""

    def __init__(self, *, active: set[str]) -> None:
        super().__init__(active=active)
        self.timeouts: list[tuple[list[str], object]] = []

    def __call__(self, value, *, check=False, **kwargs):
        self.timeouts.append((list(value), kwargs.get("timeout")))
        if value[:2] in (["systemctl", "is-active"], ["systemctl", "is-enabled"]) and (
            value[2] != "--quiet"
        ):
            self.commands.append(value)
            answer = "active" if value[2] in self.active else "inactive"
            if value[1] == "is-enabled":
                answer = "enabled"
            return CompletedProcess(value, 0, stdout=answer + "\n", stderr="")
        return super().__call__(value, check=check, **kwargs)

    def timeout_of(self, *argv: str) -> object:
        (timeout,) = [item for command, item in self.timeouts if command == list(argv)]
        return timeout


def test_fabric_manager_alone_gets_the_unit_s_own_start_bound() -> None:
    assert FABRIC_MANAGER_START_TIMEOUT_SECONDS == 780, "R580 720 s plus a minute"
    assert UNIT_START_TIMEOUT_SECONDS == 120, "every other unit is unchanged"
    assert unit_start_timeout_seconds(FM) == 780
    assert unit_start_timeout_seconds(f"{FM}.service") == 780
    for unit in ("nvidia-dcgm", "nvidia-persistenced", "kubelet", "nvlsm"):
        assert unit_start_timeout_seconds(unit) == 120, unit
    assert START_TIMEOUT_SECONDS == 780, "the start after a full reset follows it"


def test_the_restore_starts_fabric_manager_with_780s_and_the_rest_with_120s(
    tmp_path: Path,
) -> None:
    runner = TimedRunner(active=set(SERVICES))
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=SERVICES,
        processes=(),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        sleeper=lambda _seconds: None,
    )
    manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")

    restored = manager.restore(incident_id="incident-a")

    assert restored["restored_services"] == list(SERVICES), restored
    assert runner.timeout_of("systemctl", "start", FM) == 780
    assert runner.timeout_of("systemctl", "start", "nvidia-dcgm") == 120
    assert runner.timeout_of("systemctl", "start", "gpu-fault-metrics-collector") == 120


def test_the_fail_safe_and_boot_restore_use_the_same_bounds(tmp_path: Path) -> None:
    """The fail-safe timer's restore and the boot reconcile run the same loop."""

    runner = TimedRunner(active=set(SERVICES))
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=SERVICES,
        processes=(),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
        sleeper=lambda _seconds: None,
    )
    result = manager.quiesce(incident_id="incident-a", workflow_request_id="workflow-a")

    GpuServiceQuiesceManager.restore_state_file(
        Path(result["state_path"]), runner=runner
    )

    assert runner.timeout_of("systemctl", "start", FM) == 780
    assert runner.timeout_of("systemctl", "start", "nvidia-dcgm") == 120


def test_the_fabric_manager_restart_waits_780s(tmp_path: Path) -> None:
    runner = TimedRunner(active={FM})
    agent = node_action_executor(
        tmp_path,
        "fm-restart.db",
        allowed_operations={WorkflowOperation.RESTART_FABRIC_MANAGER},
        fabric_manager_restart_enabled=True,
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RESTART_FABRIC_MANAGER)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert runner.timeout_of("systemctl", "restart", FM) == 780, (
        "the unit's start bound plus its stop, not the generic 120 s"
    )


def test_the_start_after_a_full_reset_waits_780s() -> None:
    runner = TimedRunner(active=set())

    record = start_fabric_manager_after_reset(runner, [])

    assert record["fabric_manager_start"]["attempted"] is True, record
    assert runner.timeout_of("systemctl", "start", FM) == 780


def _unit_section(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser["Unit"]


@pytest.mark.parametrize(
    "dependency", ["Requires", "Wants", "BindsTo", "Requisite", "PartOf", "Upholds"]
)
def test_the_metrics_collector_starts_after_fabric_manager_without_needing_it(
    dependency: str,
) -> None:
    """Ordering only: at boot the collector starts once Fabric Manager's start
    job (on NVL5: NVLSM, then Fabric Manager) has finished, and a node with no
    such unit, or a failed one, still runs its collector."""

    unit = _unit_section(ROOT / "deploy/systemd/gpu-fault-metrics-collector.service")

    assert f"{FM}.service" in unit.get("After", "").split(), unit.get("After")
    assert "nvidia-persistenced.service" in unit.get("After", "").split()
    assert f"{FM}.service" not in unit.get(dependency, "").split(), (
        f"{dependency}= would make Fabric Manager a dependency of the collector"
    )


def test_the_installer_ships_that_unit_verbatim() -> None:
    """The ordering reaches nodes through the node bundle and the installer."""

    installer = (ROOT / "deploy/node/install-gpu-fault-collector.sh").read_text()
    bundle = (ROOT / "deploy/node/build-node-installer-bundle.sh").read_text()

    assert (
        'install -m 0644 "${REPO_DIR}/deploy/systemd/gpu-fault-metrics-collector.'
        'service" \\\n    /etc/systemd/system/gpu-fault-metrics-collector.service'
    ) in installer, "installed as shipped"
    assert '"${REPO_DIR}/deploy/systemd/"*.service' in bundle, "bundled"
    assert "systemctl daemon-reload" in installer, "and reloaded"
