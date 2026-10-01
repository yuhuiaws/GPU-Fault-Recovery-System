"""The final device-client recheck names the holders it refuses on.

Live 2026-10-01 (regional acceptance, segment D of the active-workload reset
case): a ``RESET_GPU`` on a freshly provisioned node failed with
``ResetProgressError: OWNERSHIP_FINAL_CLIENTS_CHANGED (GPU GPU-..., 1 of 1)``
and nothing -- not the journal, not the result details, not the workflow step
-- said which process held the device. The sampled preflight already names its
holders (``<gpu>:<pid>:<comm>``); the final refusal was the one opaque place,
so an operator could not tell a lingering workload from a platform daemon
(DCGM host engine, exporter, health monitor) without a node login. The refusal
now carries a bounded, sanitized attribution that travels with the result.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.late_ownership import (
    FINAL_CLIENT_ATTRIBUTION_LIMIT,
    FINAL_CLIENT_FIELD_LIMIT,
    OWNERSHIP_PROTOCOL,
    OwnershipPermit,
    OwnershipRefused,
    execute_with_final_ownership,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import SECRET, FakeRunner, node_action_executor
from tests.node_agent.test_late_ownership_checkpoint_history import granted
from tests.regional.test_late_ownership_agent_gate import NOW
from tests.regional.test_late_ownership_agent_gate import command as command_fixture
from tests.regional.test_late_ownership_agent_gate_edges import FakeNodeAgent

LOGGER_NAME = "gpu_fault.node_agent.late_ownership"
REASON = "OWNERSHIP_FINAL_CLIENTS_CHANGED"
HOLDER = {
    "gpu_uuid": "GPU-a",
    "pid": "4242",
    "device": "/dev/nvidia0",
    "process_name": "dcgm-exporter",
}
# A value that must never travel: a holder's command line or environment can
# carry exactly this kind of string, and the journal is world-readable to ops.
TOKEN_LIKE = "AKIA-NOT-A-REAL-TOKEN-0123456789"
IDENTITY_FIELDS = {"gpu_uuid", "pid", "device", "process_name"}


def _refused(
    holders: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> OwnershipRefused:
    agent = FakeNodeAgent()
    agent.device_client_finder = lambda targets: list(holders)  # type: ignore[method-assign]
    agent.sleep = lambda seconds: None  # type: ignore[attr-defined]
    signed = SignedNodeAction(command=command_fixture(), signature="unused")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        with pytest.raises(OwnershipRefused) as refused:
            execute_with_final_ownership(agent, signed, agent.effect)
    assert "effect" not in agent.calls, "a refused recheck must not reach the handler"
    return refused.value


def test_the_refusal_names_each_persistent_holder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    refused = _refused([dict(HOLDER)], caplog)

    details = refused.action_details
    assert details["reason"] == REASON
    # The code stays the message prefix for every existing matcher; the holder
    # follows in the preflight's own ``<gpu>:<pid>:<comm>`` shape.
    assert str(refused) == f"{REASON}: GPU-a:4242:dcgm-exporter"
    assert details["persistent_device_clients"] == [HOLDER]
    assert details["persistent_device_client_count"] == 1
    assert details["node_action_not_started"] is True
    assert "refusal_cause" not in details, "a named holder is not a foreign failure"
    [record] = [item for item in caplog.records if item.name == LOGGER_NAME]
    assert record.levelno == logging.WARNING
    assert "final ownership recheck refused" in record.getMessage()
    assert "command_id=owned-command" in record.getMessage()
    assert "GPU-a:4242:/dev/nvidia0:dcgm-exporter" in record.getMessage()


def test_the_attribution_is_bounded_in_count_and_field_length(
    caplog: pytest.LogCaptureFixture,
) -> None:
    long_name = "x" * (FINAL_CLIENT_FIELD_LIMIT * 3) + "\n\x1b[0m"
    holders = [
        {**HOLDER, "pid": str(1000 + index), "process_name": long_name}
        for index in range(FINAL_CLIENT_ATTRIBUTION_LIMIT + 4)
    ]

    refused = _refused(holders, caplog)

    attributed = refused.action_details["persistent_device_clients"]
    assert len(attributed) == FINAL_CLIENT_ATTRIBUTION_LIMIT
    assert refused.action_details["persistent_device_client_count"] == len(holders)
    assert str(refused).endswith("(+4 more)"), str(refused)
    for entry in attributed:
        for value in entry.values():
            assert len(value) <= FINAL_CLIENT_FIELD_LIMIT, entry
            assert value.isprintable(), "control characters must not reach the journal"
    [record] = [item for item in caplog.records if item.name == LOGGER_NAME]
    assert "\n" not in record.getMessage(), "a holder name must not split the line"
    assert "\x1b" not in record.getMessage(), "an escape must not reach the journal"


def test_only_the_four_identity_fields_are_copied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    holder = {
        **HOLDER,
        "cmdline": f"python train.py --token {TOKEN_LIKE}",
        "environ": f"AWS_SECRET_ACCESS_KEY={TOKEN_LIKE}",
        "cgroup": "/kubepods/burstable/pod-1234",
    }

    refused = _refused([holder], caplog)

    [entry] = refused.action_details["persistent_device_clients"]
    assert set(entry) == IDENTITY_FIELDS
    assert TOKEN_LIKE not in str(refused)
    assert TOKEN_LIKE not in repr(refused.action_details)
    assert TOKEN_LIKE not in caplog.text
    assert "kubepods" not in caplog.text, (
        "the cgroup path is not part of the attribution"
    )


@pytest.mark.parametrize("device_clients", [None, []])
def test_a_refusal_without_holders_carries_no_attribution(
    device_clients: list[dict[str, str]] | None,
) -> None:
    refused = OwnershipRefused(
        "OWNERSHIP_PERMIT_EXPIRED", device_clients=device_clients
    )

    assert str(refused) == "OWNERSHIP_PERMIT_EXPIRED"
    assert "persistent_device_clients" not in refused.action_details
    assert "persistent_device_client_count" not in refused.action_details


def refused_reset_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[NodeActionExecutor, NodeActionResult, FakeRunner]:
    """Run one ``RESET_GPU`` through the real executor into a holder refusal.

    The sampled preflight sees a clean node; the holder appears only once the
    final permit has been granted, exactly the window the final recheck
    exists for. Shared with the adapter fold test so the result the workflow
    step receives is the result the executor really writes.
    """

    hardware = FakeRunner()
    granted_permits: list[str] = []

    def finder(targets: set[str]) -> list[dict[str, str]]:
        return [dict(HOLDER)] if granted_permits else []

    agent = node_action_executor(
        tmp_path,
        "holder-attribution.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=lambda: NOW,
        runner=hardware,
        require_final_ownership=True,
        agent_generation=1,
        device_client_finder=finder,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=1,
        sleep=lambda seconds: None,
    )

    def require(current: NodeActionCommand) -> OwnershipPermit:
        granted_permits.append(current.command_id)
        return granted(current, len(granted_permits), SECRET)

    monkeypatch.setattr(agent.ownership_gate, "require", require)
    value = NodeActionCommand(
        command_id="workflow/step/node-a",
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        fencing_token=3,
        operation=WorkflowOperation.RESET_GPU,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=["GPU-a"],
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
        ownership_guard=OWNERSHIP_PROTOCOL,
    )
    signed = SignedNodeAction(command=value, signature=sign_node_action(value, SECRET))
    return agent, agent.execute(signed), hardware


def test_the_node_action_result_carries_the_holders_through_the_reset_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, result, hardware = refused_reset_result(tmp_path, monkeypatch)

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert (result.error or "").startswith(
        f"ResetProgressError: {REASON}: GPU-a:4242:dcgm-exporter (GPU GPU-a, 1 of 1)"
    ), result.error
    assert result.details["reason"] == REASON
    assert result.details["persistent_device_clients"] == [HOLDER]
    assert result.details["persistent_device_client_count"] == 1
    assert result.details["node_action_not_started"] is True
    assert result.details["reset_failed"] == ["GPU-a"]
    assert [item for item in hardware.commands if "--gpu-reset" in item] == []
    # The ledger answer a later poll replays is the same attributed result.
    assert agent.ledger.get(result.command_id) == result


def test_the_default_device_client_finder_reports_the_holders_comm(
    tmp_path: Path,
) -> None:
    """``device_client_finder`` already carries ``process_name`` from ``comm``.

    Pinned because the attribution reads that field: a finder that dropped it
    would turn every refusal back into ``<gpu>:<pid>:`` with no name.
    """

    proc = tmp_path / "proc"
    (proc / "100" / "fd").mkdir(parents=True)
    (proc / "100" / "fd" / "7").symlink_to("/unit/nvidia0")
    (proc / "100" / "comm").write_text("dcgm-exporter\n", encoding="ascii")
    (proc / "100" / "cmdline").write_text(f"exporter\0--token={TOKEN_LIKE}\0")
    agent = node_action_executor(
        tmp_path,
        "finder.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        proc_root=str(proc),
        gpu_device_path_finder=lambda: {"GPU-a": "/unit/nvidia0"},
    )

    clients = agent.device_client_finder({"GPU-a"})

    assert clients == [
        {
            "gpu_uuid": "GPU-a",
            "pid": "100",
            "process_name": "dcgm-exporter",
            "device": "/unit/nvidia0",
        }
    ]
    assert TOKEN_LIKE not in repr(clients), "the command line is never scanned"
