"""Re-evaluate the driver's recovery action after P2P users are drained."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

from gpu_fault.node_agent.operations.reset import ResetProgressError
from gpu_fault.node_agent.protocol import NodeActionCommand

if TYPE_CHECKING:
    from gpu_fault.node_agent.executor import NodeActionExecutor

RECOVERY_ACTIONS = frozenset(
    {"None", "Reset", "Reboot", "Drain P2P", "Drain and Reset"}
)


def parse_recovery_state(text: str, targets: list[str]) -> dict[str, str]:
    """Require one current, identifiable report per target, never guess N/A."""

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeError("GPU recovery query returned invalid XML") from exc
    states: dict[str, str] = {}
    wanted = set(targets)
    for gpu in root.findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        if uuid not in wanted:
            continue
        if uuid in states:
            raise RuntimeError(f"duplicate GPU recovery report for {uuid}")
        if (gpu.findtext("persistence_mode") or "").strip() != "Disabled":
            raise RuntimeError(f"persistence release is not confirmed for {uuid}")
        action = (gpu.findtext("gpu_recovery_action") or "").strip()
        if action not in RECOVERY_ACTIONS:
            raise RuntimeError(f"unknown GPU recovery action for {uuid}: {action!r}")
        states[uuid] = action
    if set(states) != wanted:
        raise RuntimeError("GPU recovery query omitted one or more target GPUs")
    return states


def recover_drain_p2p(
    executor: NodeActionExecutor, command: NodeActionCommand
) -> dict[str, Any]:
    if not executor.reset_enabled or not executor.service_quiesce_enabled:
        raise RuntimeError("Drain P2P recovery requires enabled reset and quiesce")
    if executor.quiesce_manager is None:
        raise RuntimeError("Drain P2P recovery requires a quiesce manager")
    targets = command.gpu_uuids
    if not targets or len(set(targets)) != len(targets):
        raise RuntimeError("Drain P2P recovery requires unique target GPU UUIDs")
    executor.quiesce_manager.assert_quiesced(
        incident_id=command.incident_id, for_reset=False
    )
    # Quiesce releases persistence through the daemon RPC (or legacy mode).
    # Verify that release and all device clients before consulting the driver.
    executor._verify_no_clients(targets, include_agent_activities=True)
    completed = executor._run_checked(
        ["nvidia-smi", "-q", "-x", "-i", ",".join(targets)],
        timeout=executor._nvidia_smi_query_timeout(),
    )
    states = parse_recovery_state(completed.stdout, targets)
    details: dict[str, Any] = {
        "gpu_recovery_actions": states,
        "persistence_disabled": True,
        "verified_no_gpu_clients": True,
        "reset_completed": [],
    }
    # Decide over the whole target set before resetting any GPU.
    if "Reboot" in states.values():
        raise ResetProgressError(
            "the driver requires node reboot after P2P drain",
            action_details={
                **details,
                "gpu_recovery_reboot_required": True,
                "reset_not_attempted": targets,
            },
        )
    reset_targets = [uuid for uuid in targets if states[uuid] != "None"]
    if not reset_targets:
        return {**details, "reset_skipped": "driver recovery action is None"}
    try:
        result = executor._execute_reset_gpu(
            command.model_copy(update={"gpu_uuids": reset_targets})
        )
    except ResetProgressError as exc:
        exc.action_details["gpu_recovery_actions"] = states
        raise
    return {**details, **result, "reset_completed": reset_targets}
