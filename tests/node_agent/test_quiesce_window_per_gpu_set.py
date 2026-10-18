"""One quiesce window per GPU set: what the Node Agent does when a second GPU of
the node turns up after the window was opened (batch 136).

The control plane used to widen a window whose QUIESCE had already run: VERIFY,
RESET and RESTORE took the second GPU, the QUIESCE kept the first. On the Agent
that window had released persistence and swept clients for the quiesced GPU
only. Pinned here, on the real executor and quiesce manager over one fake node
(systemd, the persistence daemon, the driver's report), in every persistence
regime a node boots into:

- the reproduction: inside the first GPU's window the widened VERIFY meets the
  ``nvidia-persistenced`` the window kept, holding the second GPU's device node,
  and fails closed (the ladder then reboots the node); with no daemon it passes
  on a GPU whose persistence the window never released; and the Agent refuses a
  second QUIESCE with another target set outright;
- the fixed shape: the second GPU's own window, after the first one's RESTORE,
  under the same incident and workflow: its QUIESCE releases that GPU, its VERIFY
  passes, its RESET claims that window's single reset, Fabric Manager is kept in
  both windows;
- re-send stability: the first window's commands, re-sent with their bodies, are
  answered from the ledger; the same id with a widened body is refused as a
  reused command id (HTTP 409 ``COMMAND_ID_REUSED``).
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.node_agent.app import _node_action_rejection
from gpu_fault.node_agent.quiesce import FABRIC_MANAGER_KEPT_REASON

from ._support import (
    GpuServiceQuiesceManager,
    NodeActionExecutor,
    NodeActionResult,
    NodeActionStatus,
    WorkflowOperation,
    command,
    envelope,
    node_action_executor,
)
from .test_quiesce_persistence_release import (
    DAEMON_PID,
    OTHER_BUS_ID,
    TARGET_BUS_ID,
    PersistenceNode,
)

FM = "nvidia-fabricmanager"
DAEMON_UNIT = "nvidia-persistenced"
RESTORE_UNIT = "gpu-fault-gpu-persistence"
SERVICES = (FM, "nvidia-dcgm", DAEMON_UNIT, RESTORE_UNIT, "kubelet")
KEEP = {"fabric_manager": "keep"}
#: GPU-a is the first fault (bus 53, /dev/nvidia1), GPU-b the second (bus 64,
#: /dev/nvidia0): probe order is not PCI order on purpose.
BUS = {"GPU-a": TARGET_BUS_ID, "GPU-b": OTHER_BUS_ID}
DEVICE = {"GPU-a": "/dev/nvidia1", "GPU-b": "/dev/nvidia0"}
MINOR = {"GPU-a": "1", "GPU-b": "0"}
HYPERPOD = {FM, "nvidia-dcgm", RESTORE_UNIT, "kubelet"}
EKS = HYPERPOD | {DAEMON_UNIT}
#: The regimes a node boots into: (daemon, legacy flag, active units, release
#: error). HyperPod starts ``nvidia-persistenced`` by hand and races
#: ``gpu-fault-gpu-persistence``: daemon-only when the daemon wins, daemon plus
#: the legacy flag when the unit wins. EKS runs the daemon as a systemd unit the
#: window keeps; when the release fails the window stops it as before.
REGIMES: dict[str, tuple[str | None, bool, set[str], str | None]] = {
    "eks-daemon-kept": ("systemd", False, EKS, None),
    "eks-daemon-stopped": ("systemd", False, EKS, "Failed to set persistence mode"),
    "hyperpod-daemon": ("hand", False, HYPERPOD, None),
    "hyperpod-daemon-and-legacy": ("hand", True, HYPERPOD, None),
    "legacy-only": (None, True, HYPERPOD, None),
}


class AgentNode(PersistenceNode):
    """``PersistenceNode`` plus what the executor itself asks the driver: the
    ``nvidia-smi -q -x`` report its device map and bus ids come from, the
    compute-app query, the GPU inventory and the reset."""

    def __init__(self, tmp_path: Path, regime: str) -> None:
        daemon, legacy, active, release_error = REGIMES[regime]
        super().__init__(
            tmp_path,
            daemon=daemon,
            legacy=legacy,
            active=set(active),
            release_error=release_error,
        )
        self.resets: list[str] = []

    def report(self) -> str:
        gpus = "".join(
            f'<gpu id="{BUS[uuid]}"><uuid>{uuid}</uuid>'
            f"<minor_number>{MINOR[uuid]}</minor_number>"
            f"<persistence_mode>{self.mode(BUS[uuid])}</persistence_mode>"
            f"<pci><pci_bus_id>{BUS[uuid]}</pci_bus_id></pci></gpu>"
            for uuid in sorted(BUS)
        )
        return f'<?xml version="1.0" ?><nvidia_smi_log>{gpus}</nvidia_smi_log>'

    def _nvidia_smi(self, argv: list[str], timeout: Any) -> CompletedProcess:
        if argv[1:3] == ["-q", "-x"]:
            return CompletedProcess(argv, 0, stdout=self.report(), stderr="")
        if argv[1].startswith("--query-compute-apps="):
            return CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[1] == "--query-gpu=uuid":
            return CompletedProcess(argv, 0, stdout="GPU-a\nGPU-b\n", stderr="")
        if argv[1:2] == ["--gpu-reset"]:
            self.resets.append(argv[-1])
            return CompletedProcess(argv, 0, stdout="", stderr="")
        return super()._nvidia_smi(argv, timeout)

    def fm_calls(self) -> list[list[str]]:
        return [
            item
            for item in self.commands
            if item[:1] == ["systemctl"]
            and item[-1] == FM
            and item[1] in {"stop", "start", "restart"}
        ]


def agent_on(tmp_path: Path, regime: str) -> tuple[NodeActionExecutor, AgentNode]:
    node = AgentNode(tmp_path, regime)
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=SERVICES,
        processes=(),
        failsafe_seconds=420,
        retry_seconds=60,
        settle_seconds=0,
        restore_settle_seconds=0,
        device_sweep_timeout_seconds=20,
        proc_root=str(node.proc),
        restore_command="/opt/gpu-fault/restore",
        runner=node,
        sleeper=node.tick,
    )
    agent = node_action_executor(
        tmp_path,
        "per-gpu-window.db",
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        },
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=manager,
        runner=node,
        proc_root=str(node.proc),
        device_client_samples=2,
        sleep=lambda _seconds: None,
    )
    return agent, node


def run(
    agent: NodeActionExecutor,
    operation: WorkflowOperation,
    step: int,
    gpus: list[str],
    parameters: dict[str, Any] | None = None,
) -> NodeActionResult:
    """One command the way the adapter names it: ``<workflow>/<step>/<op>/<node>``."""

    command_id = f"workflow-a/{step}/{operation.value}/node-a"
    if operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS:
        command_id += "/attempt-1"
    return agent.execute(
        envelope(
            command(
                operation, command_id=command_id, gpu_uuids=gpus, parameters=parameters
            )
        )
    )


def window(
    agent: NodeActionExecutor, first_step: int, gpus: list[str]
) -> list[NodeActionResult]:
    """QUIESCE, VERIFY, RESET and RESTORE of one per-GPU reset window."""

    return [
        run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, first_step, gpus, KEEP),
        run(agent, WorkflowOperation.VERIFY_NO_GPU_CLIENTS, first_step + 1, gpus),
        run(agent, WorkflowOperation.RESET_GPU, first_step + 2, gpus),
        run(agent, WorkflowOperation.RESTORE_GPU_SERVICES, first_step + 3, gpus),
    ]


def released(result: NodeActionResult) -> list[str]:
    return [
        item["command"] for item in result.details["persistence_release"]["commands"]
    ]


def state_file(tmp_path: Path) -> dict[str, Any]:
    (path,) = sorted((tmp_path / "quiesce").glob("quiesce-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "regime", ["eks-daemon-kept", "hyperpod-daemon", "hyperpod-daemon-and-legacy"]
)
def test_a_widened_verify_meets_the_kept_daemon_on_the_second_gpu(
    tmp_path: Path, regime: str
) -> None:
    agent, node = agent_on(tmp_path, regime)

    quiesced = run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, 3, ["GPU-a"], KEEP)
    widened = run(agent, WorkflowOperation.VERIFY_NO_GPU_CLIENTS, 4, ["GPU-a", "GPU-b"])

    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    assert released(quiesced) == [f"nvidia-smi -i {BUS['GPU-a']} -pm 0"]
    assert node.held == {BUS["GPU-b"]}, "the kept daemon still holds the second GPU"
    assert widened.status is NodeActionStatus.FAILED
    assert widened.error is not None
    assert f"GPU-b:{DAEMON_PID}:nvidia-persiste" in widened.error, widened.error
    (holder,) = widened.details["gpu_client_holders"]
    assert holder["device"] == DEVICE["GPU-b"]
    assert holder["process_name"] == "nvidia-persiste"
    assert node.resets == [], "nothing was reset: the step escalates instead"


def test_with_no_daemon_the_widened_verify_passes_on_an_unreleased_gpu(
    tmp_path: Path,
) -> None:
    agent, node = agent_on(tmp_path, "legacy-only")

    quiesced = run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, 3, ["GPU-a"], KEEP)
    widened = run(agent, WorkflowOperation.VERIFY_NO_GPU_CLIENTS, 4, ["GPU-a", "GPU-b"])

    assert widened.status is NodeActionStatus.SUCCEEDED, widened.error
    assert quiesced.details["persistence_release"]["pci_bus_ids"] == [BUS["GPU-a"]]
    after = {
        item["pci_bus_id"]: item["persistence_mode"]
        for item in quiesced.details["persistence_mode"]["after_release"]
    }
    assert after == {BUS["GPU-a"]: "Disabled", BUS["GPU-b"]: "Enabled"}, (
        "the window never released the second GPU it would now reset"
    )
    assert state_file(tmp_path)["target_device_paths"] == [DEVICE["GPU-a"]]


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_the_agent_refuses_to_widen_an_open_window(tmp_path: Path, regime: str) -> None:
    agent, node = agent_on(tmp_path, regime)
    run(agent, WorkflowOperation.QUIESCE_GPU_SERVICES, 3, ["GPU-a"], KEEP)
    calls = len(node.commands)

    widened = run(
        agent, WorkflowOperation.QUIESCE_GPU_SERVICES, 10, ["GPU-a", "GPU-b"], KEEP
    )

    assert widened.status is NodeActionStatus.FAILED
    assert "quiesce target GPU scope changed after quiesce" in str(widened.error)
    assert not any(
        item[:1] in (["kill"], ["systemctl"]) or "-pm" in item
        for item in node.commands[calls:]
    ), "nothing was stopped, released or signalled for the refused widening"


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_each_gpu_gets_its_own_window_in_the_same_incident(
    tmp_path: Path, regime: str
) -> None:
    agent, node = agent_on(tmp_path, regime)

    first = window(agent, 3, ["GPU-a"])
    second = window(agent, 12, ["GPU-b"])

    for result in (*first, *second):
        assert result.status is NodeActionStatus.SUCCEEDED, (
            result.operation.value,
            result.error,
        )
    assert node.resets == ["GPU-a", "GPU-b"], "each GPU reset once, in its own window"
    for quiesce, gpu in ((first[0], "GPU-a"), (second[0], "GPU-b")):
        assert released(quiesce) == [f"nvidia-smi -i {BUS[gpu]} -pm 0"], regime
        assert quiesce.details["persistence_release"]["pci_bus_ids"] == [BUS[gpu]]
        kept = [item["service"] for item in quiesce.details["kept_services"]]
        assert kept[0] == FM, "a per-GPU reset window keeps Fabric Manager"
        assert quiesce.details["kept_services"][0]["reason"] == (
            FABRIC_MANAGER_KEPT_REASON
        )
        if regime == "eks-daemon-kept":
            assert kept == [FM, DAEMON_UNIT], "the release went through the daemon"
        else:
            assert DAEMON_UNIT not in kept
    assert second[1].details["gpu_uuids"] == ["GPU-b"]
    assert node.fm_calls() == [], "Fabric Manager is never stopped nor started"
    assert FM in node.active
    restored = {
        item["pci_bus_id"]: item["persistence_mode"]
        for item in second[3].details["persistence_mode"]["after_restore"]
    }
    assert restored == {BUS["GPU-a"]: "Enabled", BUS["GPU-b"]: "Enabled"}
    assert not list((tmp_path / "quiesce").glob("quiesce-*.json")), (
        "both windows closed"
    )


def test_a_stopped_daemon_unit_is_stopped_and_started_once_per_window(
    tmp_path: Path,
) -> None:
    agent, node = agent_on(tmp_path, "eks-daemon-stopped")

    window(agent, 3, ["GPU-a"])
    window(agent, 12, ["GPU-b"])

    stops = [
        item for item in node.commands if item == ["systemctl", "stop", DAEMON_UNIT]
    ]
    starts = [
        item for item in node.commands if item == ["systemctl", "start", DAEMON_UNIT]
    ]
    assert len(stops) == len(starts) == 2, node.commands
    assert node.running(), "the restore started the unit the window stopped"


def test_the_first_window_re_sent_after_the_second_is_answered_from_the_ledger(
    tmp_path: Path,
) -> None:
    agent, node = agent_on(tmp_path, "eks-daemon-kept")
    first = window(agent, 3, ["GPU-a"])
    window(agent, 12, ["GPU-b"])
    calls = len(node.commands)

    replayed = window(agent, 3, ["GPU-a"])

    assert [item.model_dump() for item in replayed] == [
        item.model_dump() for item in first
    ], "the same command ids and bodies answer the results they had"
    assert node.commands[calls:] == [], "no re-send reached the node"
    with pytest.raises(ValueError, match="command_id reused") as refused:
        run(agent, WorkflowOperation.RESET_GPU, 5, ["GPU-a", "GPU-b"])
    status, body = _node_action_rejection(refused.value)
    assert (status, body["code"], body["retryable"]) == (
        409,
        "COMMAND_ID_REUSED",
        False,
    )
    assert node.resets == ["GPU-a", "GPU-b"]
