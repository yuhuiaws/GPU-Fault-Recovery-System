"""Software E2E boundary: real control/agent logic, no cluster or device I/O."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from xml.etree.ElementTree import Element, SubElement, tostring

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.execution import ProductionExecutorConfig, WorkflowStepOutcome
from gpu_fault.models import CapabilityName
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import (
    GpuServiceQuiesceManager,
    NodeActionExecutor,
    NodeActionLedger,
)
from gpu_fault.node_agent.common import PERSISTENCE_RESTORE_UNIT
from gpu_fault.store import SqliteStore
from tests._builders import asgi_client, execute_workflow
from tests.node_agent._support import ServiceRunner

SECRET = "local-xid-e2e-" + "x" * 32
NODE_OPERATIONS = {
    Op.QUIESCE_GPU_SERVICES,
    Op.VERIFY_NO_GPU_CLIENTS,
    Op.RECOVER_DRAIN_P2P,
    Op.RESET_GPU,
    Op.RESTORE_GPU_SERVICES,
}
PCI_A = "0000:81:00.0"
PCI_B = "0000:82:00.0"


class LocalHardware(ServiceRunner):
    """Only a fake command runner can reach this fixture's hardware state."""

    def __init__(self):
        super().__init__(active={PERSISTENCE_RESTORE_UNIT, FABRIC_MANAGER_UNIT})
        self.persistence = {"GPU-a": "Enabled", "GPU-b": "Enabled"}
        self.recovery_action = "None"
        self.failure = None
        self.trace = []
        self.training_running = True
        self.cordoned = False
        self.stop_ready = True
        self.validation_ready = True

    def report(self, *, recovery=False):
        root = Element("nvidia_smi_log")
        for index, (uuid, bus) in enumerate((("GPU-a", PCI_A), ("GPU-b", PCI_B))):
            if recovery and self.failure == "missing-gpu" and uuid == "GPU-a":
                continue
            gpu = SubElement(root, "gpu", id=bus)
            SubElement(gpu, "uuid").text = uuid
            SubElement(gpu, "minor_number").text = str(index)
            SubElement(gpu, "persistence_mode").text = self.persistence[uuid]
            SubElement(gpu, "gpu_recovery_action").text = (
                self.recovery_action if uuid == "GPU-a" else "None"
            )
        return tostring(root, encoding="unicode")

    def __call__(self, argv, **kwargs):
        if argv[0] == "nvidia-smi":
            self.commands.append(list(argv))
            if "-pm" in argv:
                assert argv == ["nvidia-smi", "-i", PCI_A, "-pm", "0"], argv
                assert self.cordoned and not self.training_running, self.trace
                self.trace.append("persistence-off")
                if self.failure != "persistence-stuck":
                    self.persistence["GPU-a"] = "Disabled"
                return CompletedProcess(argv, 0, "", "")
            if argv[:3] == ["nvidia-smi", "-q", "-x"]:
                recovery = "-i" in argv
                if recovery:
                    assert argv[-1] == "GPU-a", argv
                    self.trace.append("recovery-query")
                    if self.failure == "query-timeout":
                        raise TimeoutExpired(argv, 15)
                return CompletedProcess(argv, 0, self.report(recovery=recovery), "")
            if "--gpu-reset" in argv:
                assert argv == ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"], argv
                assert not self.training_running and self.cordoned, self.trace
                assert self.persistence["GPU-a"] == "Disabled", self.persistence
                self.trace.append("gpu-reset")
                if self.failure == "reset-timeout":
                    raise TimeoutExpired(argv, 120)
                return CompletedProcess(argv, 0, "", "")
            query = next((arg for arg in argv if arg.startswith("--query-")), "")
            if query == "--query-gpu=pci.bus_id,persistence_mode":
                text = (
                    f"{PCI_A}, {self.persistence['GPU-a']}\n"
                    f"{PCI_B}, {self.persistence['GPU-b']}\n"
                )
            elif query == "--query-compute-apps=gpu_uuid,pid,process_name":
                text = "GPU-a, 123, python\n" if self.training_running else ""
            elif query == "--query-gpu=uuid":
                text = "GPU-a\nGPU-b\n"
            else:
                raise AssertionError(f"unexpected NVIDIA command: {argv}")
            return CompletedProcess(argv, 0, text, "")
        assert argv[0] in {"systemctl", "systemd-run"}, argv
        if argv[:2] == ["systemctl", "start"] and argv[2] == PERSISTENCE_RESTORE_UNIT:
            self.persistence["GPU-a"] = "Enabled"
            self.trace.append("persistence-restored")
        return super().__call__(argv, **kwargs)


class LocalPlatform:
    """Scripted scheduler/provider/validation boundary, not real Kubernetes."""

    def __init__(self, hardware):
        self.hardware = hardware

    def supports(self, step):
        return step.operation not in NODE_OPERATIONS

    def execute(self, context):
        op = context.step.operation
        hw = self.hardware
        hw.trace.append(op.value)
        if op is Op.MARK_UNSCHEDULABLE or op is Op.QUARANTINE:
            hw.cordoned = True
        elif op is Op.STOP_WORKLOADS:
            if not hw.stop_ready:
                return WorkflowStepOutcome.waiting(operation_id="local-stop")
            hw.training_running = False
        elif op in {Op.VALIDATE_GPU, Op.VALIDATE_FABRIC, Op.VALIDATE_HOST}:
            if not hw.validation_ready:
                return WorkflowStepOutcome.waiting(operation_id="local-validation")
        elif op is Op.RESTART_NODE:
            assert hw.cordoned and not hw.training_running, hw.trace
        elif op is Op.RESTORE_SCHEDULING:
            assert hw.validation_ready, hw.trace
            hw.cordoned = False
        elif op is Op.RESTART_WORKLOAD:
            assert not hw.cordoned and hw.validation_ready, hw.trace
            hw.training_running = True
        elif op not in {Op.FREEZE_EVIDENCE, Op.ESCALATE_SUPPORT}:
            raise AssertionError(f"unexpected platform operation: {op}")
        return WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key)


class RecoveryRig:
    def __init__(self, directory: Path):
        self.hardware = LocalHardware()
        proc = directory / "proc"
        proc.mkdir()
        self.store = SqliteStore(str(directory / "control.db"))
        self.ledger = NodeActionLedger(str(directory / "agent.db"))
        manager = GpuServiceQuiesceManager(
            state_dir=str(directory / "quiesce"),
            services=(PERSISTENCE_RESTORE_UNIT, FABRIC_MANAGER_UNIT),
            processes=(),
            containers=(),
            device_sweep_processes=(),
            proc_root=str(proc),
            settle_seconds=0,
            restore_settle_seconds=0,
            runner=self.hardware,
            sleeper=lambda _: None,
            boot_id_reader=lambda: "local-test-boot",
        )
        self.agent = NodeActionExecutor(
            secret=SECRET,
            node_ids={"node-a"},
            allowed_operations=set(NODE_OPERATIONS),
            reset_enabled=True,
            service_quiesce_enabled=True,
            quiesce_manager=manager,
            ledger=self.ledger,
            runner=self.hardware,
            proc_root=str(proc),
            diagnostic_output_dir=str(directory / "diagnostics"),
            health_snapshot_request_dir=str(directory / "snapshots"),
            device_client_samples=1,
            device_client_sample_interval_seconds=0,
            sleep=lambda _: None,
        )
        self.envelopes = []
        self.agent_results = []
        self.node_adapter = NodeActionWorkflowAdapter(
            {"node-a": "http://local-test.invalid"}, SECRET, sender=self.send
        )
        self.context = ApplicationContext(
            store=self.store,
            production_executor_config=ProductionExecutorConfig(
                enabled=True,
                executor_id="local-e2e-executor",
                allowed_operations=frozenset(Op),
            ),
            production_adapters=[self.node_adapter, LocalPlatform(self.hardware)],
            execution_token="local-execution-" + "x" * 32,
        )
        base = default_simulated_profile()
        self.store.save_profile(
            base.model_copy(
                update={
                    "capabilities": [
                        capability.model_copy(update={"owner": self.node_adapter.owner})
                        if capability.capability is CapabilityName.GPU_RESET
                        else capability
                        for capability in base.capabilities
                    ]
                }
            )
        )
        self.executor = self.context.workflow_executor

    def send(self, endpoint, envelope):
        assert endpoint == "http://local-test.invalid", endpoint
        self.envelopes.append(envelope)
        self.hardware.trace.append(envelope.command.operation.value)
        result = self.agent.execute(envelope)
        self.agent_results.append(result)
        return result

    def payload(self, *, xid=154, active=True, **overrides):
        return {
            "event_id": "local-xid-event",
            "cluster_id": "development",
            "node_id": "node-a",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "xid": xid,
            "gpu_uuid": "GPU-a",
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": "simulated-v1",
            "workload_state": "ACTIVE" if active else "IDLE",
            "affected_workload_ids": ["training/job/train"] if active else [],
            "job_id": "train" if active else None,
            "attempt_id": "train-a001" if active else None,
            **(
                {
                    "raw_message": (
                        "NVRM: Xid (PCI:0000:81:00): 154, GPU recovery action "
                        "changed from 0x0 (None) to 0x3 (Drain P2P)"
                    )
                }
                if xid == 154
                else {}
            ),
            **overrides,
        }

    def ingest(self, payload):
        async def post():
            async with asgi_client(self.context) as client:
                response = await client.post("/v1/gpu-events/xid", json=payload)
                assert response.status_code == 200, response.text
                decision = response.json()
                if decision["disposition"] == "PENDING_CORRELATION":
                    assert decision["workflow_request_id"] is None, decision
                    assert self.envelopes == [], (
                        "correlation must not dispatch an action"
                    )
                    record = self.store.get_xid_correlation(payload["event_id"])
                    self.context.xid_correlation.now = (
                        lambda: record.deadline + timedelta(milliseconds=1)
                    )
                    assert self.context.xid_correlation.run_once() == 1
                    response = await client.post("/v1/gpu-events/xid", json=payload)
                    assert response.status_code == 200, response.text
                    decision = response.json()
                assert decision["disposition"] == "EXECUTABLE", decision
                if payload["xid"] == 154:
                    assert decision["official_action"] == "DRAIN_P2P", decision
                return decision

        return asyncio.run(post())

    def execute(self, decision):
        workflow = self.store.get_workflow(decision["workflow_request_id"])
        return execute_workflow(
            self.executor,
            workflow.request_id,
            expected_fencing_token=workflow.fencing_token,
        )

    def close(self):
        self.ledger.close()
        self.store.close()
