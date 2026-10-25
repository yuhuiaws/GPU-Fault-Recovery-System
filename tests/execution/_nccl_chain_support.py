"""Continuous local chain: raw RDMA -> API -> real node handlers -> real fabric validation."""

import hashlib
import json
import tarfile
import time
from datetime import datetime, timedelta
from threading import Lock

import pytest

from gpu_fault.adapters import ControlPlaneEvidenceAdapter, gpu_validation
from gpu_fault.adapters.node_action import hung_survey as fanout
from gpu_fault.adapters.node_action import transport
from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.host.collector import HostTelemetryCollector
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.gpu_metrics import GpuMetricsService
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowStatus
from gpu_fault.node_agent import NodeActionLedger
from gpu_fault.node_agent.operations import flight_recorder, hung_summary, hung_triage
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from gpu_fault.store import SqliteStore
from tests._builders import (
    active_workflow_executor,
    attempt_observation,
    container_observation,
)
from tests.execution._health_policy_e2e_support import JOB
from tests.execution._nccl_chain_node import OPERATIONS, ChainNode
from tests.execution._raw_health_e2e_support import RawHealthRig
from tests.node_agent._support import SECRET
from tests.node_agent.test_process_diagnostics_e2e import proc_stat


class RdmaCollector(HostTelemetryCollector):
    CONTRIBUTORS = ("_rdma",)


class NodeClock:
    def __init__(self):
        self.value = 0.0
        self.tick = lambda value: None
        self.settle = lambda: None

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        self.tick(self.value)
        # An agent sleep is where PyTorch's monitor thread gets to run: every
        # FIFO request the agent already wrote is answered before it returns.
        self.settle()
        time.sleep(0.003)


@pytest.fixture(name="chain")
def chain_fixture(tmp_path, monkeypatch):
    rigs = []

    def create(**kwargs):
        root = tmp_path / str(len(rigs))
        root.mkdir()
        rig = NcclChain(root, monkeypatch, **kwargs)
        rigs.append(rig)
        return rig

    yield create
    for rig in rigs:
        rig.close()


def runtime_container_id(node):
    return hashlib.sha256(f"trainer-container/{node}".encode()).hexdigest()


class NcclChain(RawHealthRig):
    def __init__(self, path, monkeypatch, *, nodes=3, mode="normal"):
        self.nodes = [f"node-{chr(97 + index)}" for index in range(nodes)]
        self.gpus = [f"GPU-{chr(97 + index)}" for index in range(nodes)]
        self.attempt = "attempt-a"
        self.scope_known = True
        # "annotation": the Pod names its cgroup (fixtures only); "runtime-id":
        # the watcher knows only the kubelet's container id, as on a live site.
        self.container_scope = "annotation"
        super().__init__(path, monkeypatch)
        self.node_clock = NodeClock()
        self.actors = [ChainNode(self, index, mode) for index in range(nodes)]
        self.node_clock.settle = self.settle_fifos
        self.commands, self.outcomes, self.extra_stores = [], [], []
        self.pending_node = None
        self.summary_unreachable = None
        self.triage_unreachable = None
        self.bundle_unreachable = None
        self.corrupt_node = None
        self.old_epoch_node = None
        self.pid_reuse = mode == "pid-reuse"
        self.lose_response = None
        self.lost_response = False
        self.command_io_counts = []
        self.mode = mode
        self.survey_passes = set()
        self.fabric_scope_known = True
        # A thread id makes every node answer ``rdma -j resource show qp`` with
        # an EFA QP owned by that thread of its rank process.
        self.qp_owner_thread = None
        self.clock_bound_known = True
        self.lock = Lock()
        rig = self

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return rig.clock if tz else rig.clock.replace(tzinfo=None)

        monkeypatch.setattr(transport, "datetime", Clock)
        monkeypatch.setattr(fanout, "datetime", Clock)
        for module in (hung_summary, hung_triage, flight_recorder):
            monkeypatch.setattr(module, "time", self.node_clock)
        self.node_adapter = NodeActionWorkflowAdapter(
            {node: "https://test.invalid" for node in self.nodes},
            SECRET,
            owner="simulated-runtime",
            sender=self.send,
            max_parallel_node_actions=1,
        )
        original = self.boundary.execute

        def execute(ctx):
            if ctx.step.operation in OPERATIONS:
                result = self.node_adapter.execute(ctx)
                if self.corrupt_node and ctx.step.operation is Op.COLLECT_HUNG_TRIAGE:
                    value = (result.details.get("node_results") or {}).get(
                        self.corrupt_node
                    )
                    if value:
                        value["sha256"] = "0" * 64
                self.outcomes.append((ctx.step.operation, result))
                return result
            assert ctx.step.operation is Op.ESCALATE_SUPPORT, (
                "no workload or hardware mutation is authorized by hung suspicion"
            )
            return original(ctx)

        monkeypatch.setattr(self.boundary, "execute", execute)
        monkeypatch.setattr(
            self.store,
            "list_training_progress_states",
            lambda *args, **kwargs: pytest.fail(
                "NCCL chain must not query train steps"
            ),
        )
        self.executor = self.context.workflow_executor
        counters = path / "node-a/infiniband/efa0/ports/1/hw_counters"
        counters.mkdir(parents=True, exist_ok=True)
        device = counters.parents[2] / "device"
        device.mkdir(exist_ok=True)
        (device / "uevent").write_text("DRIVER=efa\n")
        self.counters = counters
        self.rx = self.tx = 0
        self.write_counters()
        self.collector = RdmaCollector(
            self.sink,
            self.raw_context,
            node_id="node-a",
            infiniband_root=str(path / "node-a/infiniband"),
            proc_root=str(path / "node-a/proc"),
            net_class_root=str(path / "net"),
            edge_filter_enabled=False,
            expected_efa_device_count=1,
            now=lambda: self.clock,
        )
        self.collectors = [self.collector] + [
            RdmaCollector(
                self.sink,
                self.raw_context,
                node_id=node,
                infiniband_root=str(path / node / "infiniband"),
                edge_filter_enabled=False,
                expected_efa_device_count=1,
                now=lambda: self.clock,
            )
            for node in self.nodes[1:]
        ]

    def observe(self):
        self.store.save_attempt_observation(
            attempt_observation(
                "train",
                self.attempt,
                self.clock,
                expected_critical_ranks=len(self.nodes),
                workload_ids=[JOB],
                started_at=self.clock - timedelta(minutes=20),
                containers=[
                    container_observation(
                        f"pod-{chr(97 + index)}",
                        "trainer",
                        index,
                        node,
                        cgroup_path="/training/job"
                        if self.scope_known and self.container_scope == "annotation"
                        else None,
                        container_id=f"containerd://{runtime_container_id(node)}"
                        if self.container_scope == "runtime-id"
                        else None,
                        gpu_uuids=[self.gpus[index]],
                    )
                    for index, node in enumerate(self.nodes)
                ],
            )
        )

    def use_runtime_container_ids(self):
        """A live managed Pod: no cgroup annotation, only the runtime's id.

        Each node's rank then runs in the cgroup containerd's systemd driver
        names after that id (cgroup v2), as on the HyperPod/EKS nodes.
        """

        self.container_scope = "runtime-id"
        for actor in self.actors:
            cgroup = (
                "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod"
                f"{actor.index:08d}_0000_0000_0000_000000000000.slice/"
                f"cri-containerd-{runtime_container_id(actor.name)}.scope"
            )
            (actor.proc / "cgroup").write_text(f"0::{cgroup}\n")

    def node_now(self):
        return self.clock + timedelta(seconds=self.node_clock.value)

    def settle_fifos(self):
        for actor in self.actors:
            actor.settle()

    def write_counters(self):
        for name, value in (("rx_bytes", self.rx), ("tx_bytes", self.tx)):
            (self.counters / name).write_text(str(value))

    def traffic(self, rate=0, seconds=60, *, invalid=False):
        self.advance(seconds)
        self.rx += int(rate * seconds / 2)
        self.tx += int(rate * seconds / 2)
        self.write_counters()
        if invalid:
            (self.counters / "tx_bytes").unlink()
        batch = self.collector.collect_once()
        return batch

    def trigger(self):
        self.traffic(seconds=0)
        self.traffic(2_000_000, 15)
        self.traffic(seconds=130)
        for _ in range(10):
            self.traffic()
        return self.hung_workflows()[0]

    def traffic_all(self, rate=0, seconds=60):
        self.advance(seconds)
        self.rx += int(rate * seconds / 2)
        self.tx += int(rate * seconds / 2)
        for actor, collector in zip(self.actors, self.collectors):
            counters = actor.root / "infiniband/efa0/ports/1/hw_counters"
            for name, value in (("rx_bytes", self.rx), ("tx_bytes", self.tx)):
                (counters / name).write_text(str(value))
            collector.collect_once()

    def hung_workflows(self):
        return [
            workflow
            for workflow in self.store.list_workflows()
            if any(
                step.operation is Op.COLLECT_HUNG_SUMMARY
                for step in workflow.official_steps
            )
        ]

    def healthy(self, *, omit_gpu=None):
        reverse = {value[0]: key for key, value in DCGM_METRICS.items()}
        metrics = {
            "gpu_temperature_c": 40,
            "ecc_dbe_volatile_total": 0,
            "row_remap_failure": 0,
            "row_remap_pending": 0,
            "pcie_replay_total": 0,
            "clock_throttle_reasons": 0,
            "power_limit_w": 500,
            "nvlink_crc_flit_error_total": 0,
            "nvlink_recovery_error_total": 0,
            "nvlink_replay_error_total": 0,
        }
        for index, node in enumerate(self.nodes):
            self.post(
                "/v1/collector-events/host-telemetry",
                {
                    "cluster_id": "cluster-a",
                    "node_id": node,
                    "observed_at": self.clock.isoformat(),
                    "runtime_profile_version": "simulated-v1",
                    "samples": [
                        {
                            "name": name,
                            "value": value,
                            "device": "eth0"
                            if name.startswith("network_")
                            else "efa0/1",
                        }
                        for name, value in (
                            ("network_link_up", 1),
                            ("network_errors_delta", 0),
                            ("network_drops_delta", 0),
                            ("rdma_link_down", 0),
                            ("rdma_errors_delta", 0),
                        )
                    ],
                },
            )
            if node == omit_gpu:
                continue
            collector = DcgmMetricsCollector(
                self.sink, self.raw_context, node_id=node, edge_filter_enabled=False
            )
            text = "".join(
                f'{reverse[name]}{{UUID="{self.gpus[index]}",pci_bus_id="0000:81:00.0",gpu="0"}} {value}\n'
                for name, value in metrics.items()
            )
            collector.collect_text(text, observed_at=self.clock)

    def send(self, endpoint, envelope):
        command = envelope.command
        self.commands.append(command)
        if command.node_id == self.pending_node:
            return WorkflowStepOutcome.waiting(details={"node_action_state": "PENDING"})
        failed = {
            Op.COLLECT_HUNG_SUMMARY: self.summary_unreachable,
            Op.COLLECT_HUNG_TRIAGE: self.triage_unreachable,
            Op.COLLECT_DIAGNOSTIC_BUNDLE: self.bundle_unreachable,
        }.get(command.operation)
        if command.node_id == failed:
            return NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                error="owned unavailable-node fixture",
            )
        with self.lock:
            survey_pass = command.parameters.get("survey_pass")
            if (
                command.operation is Op.COLLECT_HUNG_SUMMARY
                and survey_pass not in self.survey_passes
            ):
                self.survey_passes.add(survey_pass)
                self.clock += timedelta(seconds=5)
            self.node_clock.value = 0
            actor = self.actors[self.nodes.index(command.node_id)]
            actor.base_cpu = actor.cpu_ticks
            self.node_clock.tick = actor.tick
            actor.context_file()
            if (
                self.pid_reuse
                and command.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
                and command.node_id == self.nodes[0]
            ):
                actor.start_ticks = 9999
                (actor.proc / "stat").write_text(proc_stat(100, 0, start=9999))
            before = len(actor.calls)
            result = actor.agent.execute(envelope)
            if (
                command.operation is Op.COLLECT_HUNG_SUMMARY
                and command.node_id == self.old_epoch_node
            ):
                result = result.model_copy(
                    update={
                        "details": {
                            **result.details,
                            "snapshot_epoch": "previous-attempt-epoch",
                        }
                    }
                )
            self.command_io_counts.append(
                (
                    command.command_id,
                    command.node_id,
                    command.operation,
                    len(actor.calls) - before,
                )
            )
            if (
                self.lose_response == (command.operation, command.node_id)
                and not self.lost_response
            ):
                self.lost_response = True
                actor.ledger.close()
                actor.ledger = NodeActionLedger(str(actor.root / "ledger.db"))
                actor.agent = actor.new_agent()
                return WorkflowStepOutcome.waiting(
                    details={"node_action_state": "PENDING"}
                )
            return result

    def run(self, workflow, attempts=35):
        from gpu_fault.models import WorkflowExecutionRequest

        current = self.store.get_workflow(workflow.request_id)
        # The controlled clock and task-owned fixture NICs are explicit input
        # evidence. Negative cases remove them, never assume them in production.
        steps = [
            step.model_copy(
                update={
                    "parameters": {
                        **step.parameters,
                        **(
                            {"clock_uncertainty_seconds": 0}
                            if self.clock_bound_known
                            else {}
                        ),
                        **(
                            {
                                "training_fabric_by_node": {
                                    node: {
                                        "basis": "TASK_DEVICE_BINDING",
                                        "network_interfaces": ["eth0"],
                                        "rdma_ports": ["efa0/1"],
                                    }
                                    for node in self.nodes
                                }
                            }
                            if self.fabric_scope_known
                            else {}
                        ),
                    }
                }
            )
            for step in current.official_steps
        ]
        self.store.save_workflow(current.model_copy(update={"official_steps": steps}))
        for _ in range(attempts):
            result = self.executor.execute(
                workflow.request_id,
                WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
            )
            if result.status is not WorkflowStatus.RUNNING:
                break
        return result

    def restart_executor(self):
        store = SqliteStore(str(self.path / "health.db"))
        self.extra_stores.append(store)
        self.executor = active_workflow_executor(
            store,
            [
                ControlPlaneEvidenceAdapter(owner="simulated-runtime"),
                self.boundary,
                gpu_validation.GpuValidationAdapter(
                    GpuMetricsService(store=store),
                    store=store,
                    owner="simulated-runtime",
                ),
            ],
            set(Op),
            executor_id="health-policy-test",
        )

    def archives(self):
        result = {}
        for actor in self.actors:
            for path in (actor.root / "diagnostics").glob("*.tar.gz"):
                with tarfile.open(path) as archive:
                    stream = archive.extractfile("diagnostics/manifest.json")
                    assert stream is not None
                    result[actor.name] = (
                        json.load(stream),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
        return result

    def close(self):
        # Every actor is closed even when one fails: an actor left open keeps
        # its non-daemon FIFO thread running and the worker cannot exit.
        failures = []
        for actor in self.actors:
            try:
                actor.close()
            except AssertionError as exc:
                failures.append(f"{actor.name}: {exc}")
        for store in self.extra_stores:
            store.close()
        self.store.close()
        assert not failures, failures
