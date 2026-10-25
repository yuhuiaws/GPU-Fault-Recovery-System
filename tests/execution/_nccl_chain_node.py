"""Owned proc/sysfs/FIFO fixtures; no real GPU, workload or external service."""

import fcntl
import json
import os
import pickle
import select
import struct
import termios
from subprocess import CompletedProcess
from threading import Condition, Event, Thread

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionExecutor, NodeActionLedger
from tests.execution._flight_series_support import entry
from tests.execution._process_evidence_support import stack_json
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent._support import SECRET
from tests.node_agent.test_process_diagnostics_e2e import proc_stat

OPERATIONS = {
    Op.COLLECT_HUNG_SUMMARY,
    Op.COLLECT_HUNG_TRIAGE,
    Op.COLLECT_DIAGNOSTIC_BUNDLE,
}


def pending_bytes(descriptor):
    """Bytes written to the FIFO that its reader has not consumed yet."""
    return struct.unpack("i", fcntl.ioctl(descriptor, termios.FIONREAD, bytes(4)))[0]


class ChainNode:
    def __init__(self, rig, index, mode):
        self.rig, self.index, self.mode = rig, index, mode
        self.name, self.gpu = rig.nodes[index], rig.gpus[index]
        self.cpu_ticks = self.base_cpu = 0
        self.start_ticks = 1000
        self.root = rig.path / self.name
        self.proc = process(self.root / "proc", 100)
        task = self.proc / "task/100"
        task.mkdir(parents=True)
        state = "D" if mode == "unranked-io" and index == 0 else "R"
        for path in (self.proc, task):
            (path / "stat").write_text(proc_stat(100, 0, state))
            (path / "wchan").write_text("io_schedule" if state == "D" else "0")
        (task / "status").write_text(
            "NSpid:\t100\t1\n" if mode == "tid-namespace" else "NSpid:\t100\n"
        )
        (self.proc / "io").write_text("read_bytes: 0\nwrite_bytes: 0\nwchar: 0\n")
        self.namespace = self.proc / "root/tmp"
        self.namespace.mkdir(parents=True)
        port = self.root / "infiniband/efa0/ports/1/hw_counters"
        port.mkdir(parents=True)
        device = port.parents[2] / "device"
        device.mkdir()
        (device / "uevent").write_text("DRIVER=efa\n")
        for name in ("rx_bytes", "tx_bytes"):
            (port / name).write_text("0")
        environment = (
            []
            if mode == "unranked-io"
            else [f"RANK={index}", f"WORLD_SIZE={len(rig.nodes)}"]
        )
        self.requests = 0
        self.worker = None
        self.stop = Event()
        self.answered = Condition()
        self.reader = self.holder = None
        self.errors = []
        if mode.startswith("fr-") or mode == "pid-reuse":
            environment += [
                "TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/trace",
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/dump",
            ]
            pipe = self.namespace / f"trace{index}.pipe"
            os.mkfifo(pipe)
            self.reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
            # Our own write end keeps the FIFO out of its end-of-file state.
            # Without it, once a request's writer closed, select() reported the
            # empty FIFO readable (POLLHUP); when the agent's next open() landed
            # before the following read(), that read raised EAGAIN, the thread
            # died and every later request on this node went unanswered.
            self.holder = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
            self.worker = Thread(target=self.respond)
            self.worker.start()
        if mode in {"same-role", "different-role", "cooperative", "tid-namespace"}:
            environment.append("GPU_FAULT_RANK_PROGRESS_FILE=/tmp/context.json")
        (self.proc / "environ").write_bytes(("\0".join(environment) + "\0").encode())
        if mode == "foreign-process":
            other = process(self.root / "proc", 900, cgroup="/other-job")
            (other / "stat").write_text(proc_stat(900, 0))
        self.ledger = NodeActionLedger(str(self.root / "ledger.db"))
        self.calls = []
        self.agent = self.new_agent()

    def new_agent(self):
        return NodeActionExecutor(
            secret=SECRET,
            node_ids={self.name},
            allowed_operations=OPERATIONS,
            reset_enabled=False,
            ledger=self.ledger,
            runner=self.runner,
            proc_root=str(self.root / "proc"),
            infiniband_root=str(self.root / "infiniband"),
            diagnostic_output_dir=str(self.root / "diagnostics"),
            python_stack_tool="/unit/py-spy",
            now=self.rig.node_now,
            sleep=self.rig.node_clock.sleep,
        )

    def context_file(self):
        if self.mode not in {
            "same-role",
            "different-role",
            "cooperative",
            "tid-namespace",
        }:
            return
        (self.namespace / "context.json").write_text(
            json.dumps(
                {
                    "job_id": "train",
                    "attempt_id": self.rig.attempt,
                    "rank": self.index,
                    "world_size": len(self.rig.nodes),
                    "process_start_ticks": 1000,
                    "process_pid": 100,
                    "observed_at": self.rig.clock.isoformat(),
                    "diagnostic_context": {
                        "group_name": "tp",
                        "members": list(range(len(self.rig.nodes))),
                        "role": "trainer",
                        "pipeline_stage": "1"
                        if self.mode == "different-role" and self.index == 0
                        else "0",
                        "expert_role": "none",
                        "hardware_class": "test-h100",
                        "code_version": "fixture",
                        "thread_roles": {
                            "1" if self.mode == "tid-namespace" else "100": "TRAINING"
                        },
                    },
                    "process_groups": [
                        {
                            "name": "dp",
                            "members": list(range(len(self.rig.nodes))),
                            "enqueued": 99
                            if self.index == len(self.rig.nodes) - 1
                            else 100,
                            "started": 99
                            if self.index == len(self.rig.nodes) - 1
                            else 100,
                            "completed": 98
                            if self.index == len(self.rig.nodes) - 1
                            else 99,
                        }
                    ]
                    if self.mode == "cooperative"
                    else [],
                }
            )
        )

    def respond(self):
        try:
            while not self.stop.is_set():
                # Readable only ever means request bytes (see ``holder``).
                if not select.select([self.reader], [], [], 0.05)[0]:
                    continue
                with self.answered:
                    self.answer(os.read(self.reader, 128))
                    self.answered.notify_all()
        except Exception as exc:
            self.errors.append(exc)
        finally:
            with self.answered:
                self.answered.notify_all()

    def answer(self, payload):
        self.requests += payload.count(b"1\n")
        if self.mode == "fr-second-missing" and self.requests >= 2:
            return
        done = self.index != 0 or self.mode == "fr-advancing" and self.requests >= 2
        if self.mode == "fr-whole-group":
            done = False
        document = {
            "pg_config": {"0": {"ranks": list(range(len(self.rig.nodes)))}},
            "entries": [
                {**item, "process_group": ["0", "dp"]}
                for item in [
                    entry(99, "completed"),
                    entry(100, "completed" if done else "started"),
                ]
            ],
        }
        if self.mode == "fr-pg-status":
            document["version"] = "2.10"
            document["pg_status"] = {
                "42": {
                    "last_enqueued_collective": "100",
                    "last_started_collective": "100",
                    "last_completed_collective": "100" if done else "99",
                }
            }
            document["entries"] = [
                {
                    **entry(sequence, "completed"),
                    "pg_id": 42,
                    "process_group": ["0", "dp"],
                }
                for sequence in range(1, 81)
            ] + [{**document["entries"][-1], "pg_id": 42}]
        temporary = self.namespace / "pending"
        if self.mode == "fr-pickle":
            temporary.write_bytes(pickle.dumps(document, protocol=2))
        else:
            temporary.write_text(json.dumps(document))
        temporary.replace(self.namespace / f"dump{self.index}")

    def settle(self, timeout=30.0):
        """Return once every request already in the FIFO has been answered.

        The agent waits for a dump by sleeping on the rig's node clock, which
        moves 0.25 s per 3 ms of real time: on a loaded host its whole budget
        could pass before this thread was scheduled at all.
        """
        if self.worker is None:
            return
        with self.answered:
            answered = self.answered.wait_for(
                lambda: (
                    self.errors
                    or not self.worker.is_alive()
                    or not pending_bytes(self.reader)
                ),
                timeout,
            )
        if not answered:
            self.errors.append(
                AssertionError(f"{self.name}: FIFO request unanswered for {timeout}s")
            )

    def runner(self, argv, **kwargs):
        self.calls.append(argv)
        if (
            argv[0] in {"dcgmi", "strace", "timeout", "/usr/bin/nvidia-bug-report.sh"}
            or "--gpu-reset" in argv
        ):
            raise AssertionError(f"unexpected intrusive command: {argv}")
        query = next((value for value in argv if value.startswith("--query-")), "")
        if query.startswith("--query-compute-apps"):
            suffix = ",python" if "process_name" in query else ""
            text = f"100,{self.gpu}{suffix}\n"
            if self.mode == "foreign-process":
                text += f"900,{self.gpu}{suffix}\n"
            return CompletedProcess(argv, 0, text, "")
        if query.startswith("--query-gpu"):
            if self.mode == "normalized-cpu" and self.index == 1:
                self.rig.node_clock.sleep(0.5)
            extra = ",0x0,H100" if "clocks_throttle" in query else ""
            return CompletedProcess(argv, 0, f"{self.gpu},60,20{extra}\n", "")
        thread = self.rig.qp_owner_thread
        if argv[:2] == ["rdma", "-j"] and thread is not None:
            # The rank's NCCL proxy thread created the EFA QP; the kernel
            # reports that thread's id (see network_bindings).
            task = self.proc / "task" / str(thread)
            task.mkdir(exist_ok=True)
            (task / "stat").write_text(proc_stat(thread, 0))
            row = {"ifname": "efa0", "port": 1, "state": "RTS", "pid": thread}
            return CompletedProcess(argv, 0, json.dumps([row]), "")
        if argv[0] == "/unit/py-spy":
            assert argv[-1] == "100", "must not attach to another workload"
            function = (
                "other"
                if self.mode in {"same-role", "different-role"} and self.index == 0
                else "compute"
            )
            text = stack_json(
                100, function, tid=1 if self.mode == "tid-namespace" else 100
            )
            return CompletedProcess(
                argv, 1 if self.mode == "pyspy-error" else 0, text, ""
            )
        return CompletedProcess(argv, 0, "owned diagnostic fixture\n", "")

    def tick(self, value):
        state = "D" if self.mode == "unranked-io" and self.index == 0 else "R"
        self.cpu_ticks = self.base_cpu + (
            int(value * os.sysconf("SC_CLK_TCK")) if state == "R" else 0
        )
        (self.proc / "stat").write_text(
            proc_stat(100, self.cpu_ticks, state, self.start_ticks)
        )

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
            assert not self.worker.is_alive(), "owned FIFO writer must exit"
            os.close(self.reader)
            os.close(self.holder)
        self.ledger.close()
        assert not self.errors, self.errors
