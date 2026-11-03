"""A node-wide SXID reset names the node's whole GPU inventory (batch 244).

Live (round 15, COLLECT-014 on HyperPod, R16): the per-GPU latest row
``pcie_link_downtrain`` is published only while the collector holds a verdict
for the GPU, and a verdict needs the GPU under load; after a collector restart
on an idle GPU the row froze at its last delivery while every other row kept
refreshing. Ingest's latest-row fallback took any row inside its window as
inventory evidence. For an SXID older than the newest deliveries that window
keeps exactly the rows that stopped refreshing (the newer inventory snapshot is
"from the future" for the event), and the full-reset gates asked only for a
non-empty GPU set: a back-dated fatal SXID planned an executable
RESET_ALL_GPUS_NVSWITCHES naming GPU 0 alone on an 8-GPU node. Only the Node
Agent's exact-inventory check stopped the reset, and the ladder read that
refusal as a failed reset and opened a reboot.

Driven through the real collector routes, normalizer, policy and planner; the
Agent side through the real node-action adapter, Agent and executor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.channel_registry import FABRIC_MANAGER_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricSource,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.node_agent import NodeActionExecutor, NodeActionLedger, NodeActionResult
from gpu_fault.nvidia_logs import FabricManagerLogEvent, NvidiaKernelLogEvent
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.telemetry import TelemetryMetricLatest
from tests._builders import build_context
from tests.app_services._cov95_runtime_ingest import post
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.node_agent._support import SECRET, FakeRunner, no_device_clients

OP = WorkflowOperation
NOW = datetime.now(timezone.utc).replace(microsecond=0)
GPUS = [f"GPU-{index}" for index in range(8)]
BDFS = {gpu: f"0000:{0x18 + 0x10 * index:02x}:00.0" for index, gpu in enumerate(GPUS)}
#: GPU 0's verdict row, frozen at its last delivery before the collector restart.
FROZEN_AT = NOW - timedelta(minutes=10)
#: The fatal SXID's own time: inside the frozen row's window, and older than
#: every delivery since (the snapshot and the other rows refreshed at NOW).
BACK_DATED = FROZEN_AT + timedelta(minutes=1)
SAFETY = [OP.FREEZE_EVIDENCE, OP.MARK_UNSCHEDULABLE, OP.QUARANTINE]
FULL_RESET = "RESET_ALL_GPUS_AND_NVSWITCHES"


def _snapshot(
    context: Any,
    *,
    devices: list[str] = GPUS,
    expected: int | None = 8,
    observed_at: datetime = NOW,
) -> None:
    context.store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed_at,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="boot-a",
            expected_gpu_count=expected,
            devices=[
                GpuInventoryDevice(
                    gpu_index=index,
                    gpu_uuid=gpu,
                    pci_bdf=BDFS.get(gpu, f"0000:f{index}:00.0"),
                )
                for index, gpu in enumerate(devices)
            ],
        )
    )


def _row(context: Any, gpu: str, metric: str, observed_at: datetime) -> None:
    context.store.observe_gpu_metric(
        ("cluster-a", "node-a", gpu, metric),
        GpuMetricLatest(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed_at,
            source=GpuMetricSource.DCGM_EXPORTER,
            sample=GpuMetricSample(
                metric_name=metric,
                canonical_name=metric,
                value=0,
                gpu_uuid=gpu,
                pci_bdf=BDFS.get(gpu),
            ),
        ),
    )


def _live_node(context: Any) -> None:
    """The live node: the inventory and every per-GPU row refreshed now, GPU 0's
    verdict row frozen ten minutes ago."""

    _snapshot(context)
    for gpu in GPUS:
        _row(context, gpu, "gpu_temperature_c", NOW)
    _row(context, "GPU-0", "pcie_link_downtrain", FROZEN_AT)


def _topology(context: Any, scope: str, observed_at: datetime) -> None:
    """Trusted port topology for the SXID's switch 0 port 3 (the optional
    host-collector query): an ACCESS port names its GPU and partition."""

    labels = {
        "trusted": "true",
        "switch_id": "nvidia-nvswitch0",
        "port": "3",
        "link_scope": scope,
        "fabric_partition": "fabric-a",
    }
    if scope == "ACCESS":
        labels["gpu_uuid"] = "GPU-0"
    context.store.observe_telemetry_metric(
        TelemetryMetricLatest(
            cluster_id="cluster-a",
            node_id="node-a",
            observed_at=observed_at,
            name="nvswitch_port_topology",
            value=1,
            device="nvidia-nvswitch0/3",
            labels=labels,
        )
    )


def _sxid(
    context: Any,
    *,
    path: str = NVIDIA_KERNEL_PATH,
    sxid: int = 10003,
    classification: str = "Fatal",
    observed_at: datetime = BACK_DATED,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    values: dict[str, Any] = {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": f"b244-{sxid}-{int(observed_at.timestamp())}",
        "observed_at": observed_at,
        "message": (
            f"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): {sxid}, {classification}, "
            "Link 3"
        ),
        "product": "H100",
        "workload_state": WorkloadState.IDLE,
        "runtime_profile_version": "simulated-v1",
    }
    event = (
        NvidiaKernelLogEvent(**values, source_boot_id="boot-a")
        if path == NVIDIA_KERNEL_PATH
        # Delivered now: inside the Fabric Manager path's stale-line bound.
        else FabricManagerLogEvent(**values, source="journal", collected_at=NOW)
    )
    response = post(context, path, event)
    assert response.status_code == 200, response.text
    body = response.json()
    (normalized,) = body["normalized"]["sxid_events"]
    (decision,) = body["decisions"]
    return (
        normalized,
        decision,
        context.store.get_workflow(decision["workflow_request_id"]),
    )


def _assert_no_full_reset(normalized: dict, decision: dict, workflow: Any) -> None:
    assert normalized["participating_gpu_uuids"] == [], (
        "a partial GPU list stood in for the node's inventory: "
        f"{normalized['participating_gpu_uuids']}"
    )
    assert decision["official_action"] == FULL_RESET
    assert decision["disposition"] == "BLOCKED_MISSING_EVIDENCE", decision
    assert any(
        "complete node GPU inventory" in reason for reason in decision["reasons"]
    ), decision["reasons"]
    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.safety_only is True
    assert [step.operation for step in workflow.safety_steps] == SAFETY, (
        "the existing fail-closed path: evidence, cordon, quarantine, an operator"
    )


def _assert_full_reset(normalized: dict, decision: dict, workflow: Any) -> None:
    assert normalized["participating_gpu_uuids"] == GPUS
    assert normalized["fabric_partition"] == "cluster-a/node-a/local-nvswitch"
    assert decision["official_action"] == FULL_RESET
    assert decision["disposition"] == "EXECUTABLE", decision
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    scoped = {
        step.operation: step.gpu_uuids
        for step in workflow.official_steps
        if step.operation
        in {OP.VERIFY_NO_GPU_CLIENTS, OP.RESET_ALL_GPUS_NVSWITCHES, OP.VALIDATE_GPU}
    }
    assert scoped == {
        OP.VERIFY_NO_GPU_CLIENTS: GPUS,
        OP.RESET_ALL_GPUS_NVSWITCHES: GPUS,
        OP.VALIDATE_GPU: GPUS,
    }, scoped


@pytest.mark.parametrize("path", [NVIDIA_KERNEL_PATH, FABRIC_MANAGER_PATH])
@pytest.mark.parametrize(
    ("sxid", "classification", "topology"),
    [
        (10003, "Fatal", None),
        (19084, "Non-fatal", None),
        # A SCOPE_DEPENDENT_RESET code on a trunk link (trusted topology).
        (12001, "Fatal", "TRUNK"),
    ],
    ids=["10003", "19084", "fatal-trunk"],
)
def test_a_frozen_row_is_no_inventory_for_a_back_dated_sxid(
    path: str, sxid: int, classification: str, topology: str | None
) -> None:
    context = build_context()
    _live_node(context)
    if topology:
        _topology(context, topology, BACK_DATED - timedelta(seconds=1))

    normalized, decision, workflow = _sxid(
        context, path=path, sxid=sxid, classification=classification
    )

    _assert_no_full_reset(normalized, decision, workflow)
    assert normalized["fabric_partition"] in {None, "fabric-a"}, (
        "the node's own fabric was named on the strength of one GPU's row"
    )


@pytest.mark.parametrize(
    "evidence",
    ["fresh-snapshot", "fresh-snapshot-without-count", "rows-covering-the-node"],
)
def test_the_node_s_complete_inventory_still_plans_the_full_reset(
    evidence: str,
) -> None:
    context = build_context()
    observed_at = NOW
    if evidence == "rows-covering-the-node":
        # Every row of the node stopped at the same last delivery (the whole
        # pipeline down); the snapshot that declares the count is newer than
        # the event, so the rows answer for it.
        _snapshot(context)
        for gpu in GPUS:
            _row(context, gpu, "gpu_temperature_c", FROZEN_AT)
        observed_at = BACK_DATED
    else:
        _snapshot(context, expected=None if evidence.endswith("without-count") else 8)

    normalized, decision, workflow = _sxid(context, observed_at=observed_at)

    _assert_full_reset(normalized, decision, workflow)


@pytest.mark.parametrize(
    "evidence",
    [
        "snapshot-short-of-its-count",
        "rows-without-a-declared-count",
        "rows-naming-another-gpu-set",
        "trusted-port-topology-only",
    ],
)
def test_an_inventory_short_of_the_node_is_no_basis(evidence: str) -> None:
    context = build_context()
    observed_at = NOW
    if evidence == "snapshot-short-of-its-count":
        _snapshot(context, devices=GPUS[1:])
    elif evidence == "rows-without-a-declared-count":
        for gpu in GPUS:
            _row(context, gpu, "gpu_temperature_c", NOW)
    elif evidence == "rows-naming-another-gpu-set":
        _snapshot(context)
        for gpu in [*GPUS, "GPU-replaced"]:
            _row(context, gpu, "gpu_temperature_c", FROZEN_AT)
        observed_at = BACK_DATED
    else:
        # The SXID's port names GPU 0 and a partition; nothing else answers
        # for the node around the back-dated event.
        _live_node(context)
        _row(context, "GPU-0", "pcie_link_downtrain", NOW)
        _topology(context, "ACCESS", BACK_DATED - timedelta(seconds=1))
        observed_at = BACK_DATED

    normalized, decision, workflow = _sxid(context, observed_at=observed_at)

    _assert_no_full_reset(normalized, decision, workflow)


class _InventoryRunner(FakeRunner):
    """nvidia-smi on an 8-GPU node."""

    def __call__(self, command: list[str], **kwargs: Any) -> CompletedProcess[str]:
        if "--query-gpu=uuid" in command:
            self.commands.append(command)
            return CompletedProcess(command, 0, stdout="\n".join(GPUS), stderr="")
        return super().__call__(command, **kwargs)


class _Window:
    def assert_quiesced(self, **_: Any) -> None:
        return None


def _agent_flow(
    tmp_path: Path, requested: list[str]
) -> tuple[FlowHarness, _InventoryRunner, list[NodeActionResult]]:
    h = FlowHarness(
        [OP.QUIESCE_GPU_SERVICES, OP.RESET_ALL_GPUS_NVSWITCHES, OP.RESTORE_GPU_SERVICES]
    )
    runner = _InventoryRunner()
    agent = NodeActionExecutor(
        secret=SECRET,
        node_ids={"node-a"},
        allowed_operations={OP.RESET_ALL_GPUS_NVSWITCHES},
        reset_enabled=True,
        fabric_reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=_Window(),
        ledger=NodeActionLedger(str(tmp_path / "actions.db")),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {
            gpu: f"/dev/nvidia{index}" for index, gpu in enumerate(GPUS)
        },
        sleep=lambda _: None,
    )
    results: list[NodeActionResult] = []

    def send(_endpoint: str, envelope: Any) -> NodeActionResult:
        result = agent.execute(envelope)
        results.append(result)
        return NodeActionResult.model_validate_json(result.model_dump_json())

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, SECRET, sender=send
    )
    h.executor.adapters.insert(0, adapter)
    h.amend(
        official_steps=[
            step.model_copy(
                update={
                    "execution_owner": (
                        adapter.owner
                        if step.operation is OP.RESET_ALL_GPUS_NVSWITCHES
                        else step.execution_owner
                    ),
                    "gpu_uuids": requested,
                }
            )
            for step in h.workflow.official_steps
        ]
    )
    return h, runner, results


def _gpu_resets(runner: _InventoryRunner) -> list[list[str]]:
    return [command for command in runner.commands if "--gpu-reset" in command]


def test_the_agent_still_refuses_a_list_that_is_not_its_inventory(
    tmp_path: Path,
) -> None:
    """The backstop the defect leaned on, kept: the Agent compares the plan's
    list with its own nvidia-smi inventory before anything is claimed or
    spawned. Its refusal carries no attestation, so the ladder reads it as a
    failed reset and climbs to the reboot rung -- the reason the control plane
    must never plan a list that is not the node's whole inventory."""

    h, runner, results = _agent_flow(tmp_path, ["GPU-0"])

    assert h.execute().status is WorkflowStatus.FAILED
    (result,) = results
    assert result.retryable is False
    assert "does not match local inventory" in (result.error or ""), result.error
    assert _gpu_resets(runner) == [], "the refused reset was spawned"
    saved = h.store.get_workflow(h.workflow.request_id)
    (failure,) = [
        item
        for item in saved.step_executions
        if item.operation is OP.RESET_ALL_GPUS_NVSWITCHES
    ]
    assert failure.status is WorkflowStepStatus.FAILED
    assert [context.step.operation for context in h.adapter.calls] == [
        OP.QUIESCE_GPU_SERVICES,
        OP.RESTORE_GPU_SERVICES,
    ], "the compensation restore brings the quiesced services back"
    classified = HardwareEscalationService.classify(saved)
    assert classified is not None
    assert classified[:3] == ("reset", RecoveryAction.REBOOT_NODE, OP.RESTART_NODE), (
        classified[:3]
    )
    incident = h.store.get_incident(h.incident.incident_id)
    assert incident.state is IncidentState.ESCALATED
    assert incident.node_ids == ["node-a"]


def test_the_agent_runs_the_full_reset_on_its_whole_inventory(tmp_path: Path) -> None:
    h, runner, results = _agent_flow(tmp_path, list(GPUS))

    assert h.execute().status is WorkflowStatus.SUCCEEDED
    (result,) = results
    assert result.details["reset_gpu_uuids"] == GPUS
    assert result.details["inventory_verified_before"] is True
    assert result.details["inventory_verified_after"] is True
    assert _gpu_resets(runner) == [["nvidia-smi", "--gpu-reset"]]
