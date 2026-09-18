from __future__ import annotations

import copy
import json
from types import SimpleNamespace


class CommandModel:
    def __init__(self, module, monkeypatch) -> None:
        self.module = module
        self.v = module.verdicts
        self.events = []
        self.failure = ""
        self.cleanup_state = {}
        self.ready = {
            "owner": self.v.OWNER,
            "adapter_has_barrier_coordinator": False,
            "executor_id": "fixture-executor",
        }
        self.seed = {
            "command_id": "command-a",
            "workflow_id": "workflow-fixture",
            "incident_id": "incident-fixture",
            "event_id": "event-fixture",
            "operation": self.v.OPERATION,
            "node_ids": list(self.v.NODE_IDS),
            "registered_agents": [],
        }
        self.state = {
            "barrier_unavailable_holds_total": 3,
            "claimed_total": 3,
            "unexpected_failures": 0,
            "reported_failures": 0,
            "adapter_executed": False,
            "ledger": {"physical_count": 1, "keys": ["workflow-fixture/0/step"]},
        }
        seeded = module.seeded
        for name in (
            "require_environment",
            "preflight_residuals",
            "register_synthetic_cluster",
            "create_probe_pod",
            "seed_identity",
            "seed_command",
            "wait_executor_state",
            "wait_command",
            "read_state",
            "file_present",
            "pod_logs",
            "pod_phase",
            "delete_owned_resource",
            "command_snapshot",
            "cleanup",
            "cpu_python",
            "probe_resource_uid",
            "control",
        ):
            monkeypatch.setattr(seeded, name, getattr(self, name))
        if hasattr(module, "time"):
            monkeypatch.setattr(
                module, "time", SimpleNamespace(sleep=lambda _seconds: None)
            )

    def require_environment(self):
        self.events.append("environment")

    def preflight_residuals(self, *_args):
        self.events.append("preflight")
        if self.failure == "preflight":
            raise RuntimeError("fixture preflight refused")

    def register_synthetic_cluster(self, *_args):
        self.events.append("register")
        if self.failure == "register":
            raise RuntimeError("registry acknowledgement lost")

    def create_probe_pod(self, *_args, **_kwargs):
        self.events.append("create")
        return (
            {**self.ready, "owner": "wrong"} if self.failure == "ready" else self.ready
        )

    def seed_identity(self, _run_id):
        return copy.deepcopy(self.seed)

    def seed_command(self, *_args, **_kwargs):
        self.events.append("seed")
        if self.failure == "seed-ack":
            raise RuntimeError("seed acknowledgement lost")
        return (
            {**self.seed, "registered_agents": ["unsafe-agent"]}
            if self.failure == "seed"
            else self.seed
        )

    def wait_executor_state(self, _probe, predicate, _timeout):
        assert not predicate({}), (
            "an empty counter read must not reach the hold threshold"
        )
        assert predicate(self.state), "three recorded holds must satisfy the threshold"
        return copy.deepcopy(self.state)

    def wait_command(self, command_id, predicate, _timeout):
        self.events.append("wait-command")
        if self.module.CASE_ID.endswith("017"):
            value = {
                "status": "WAITING",
                "status_source": self.v.STATUS_SOURCE,
                "result_details": {
                    "multi_node_barrier_unavailable": True,
                    "operation": self.v.OPERATION,
                    "node_ids": list(self.v.NODE_IDS),
                    "executor_id": "fixture-executor",
                    "reason": "a multi-node barrier coordinator is unavailable",
                },
            }
        else:
            value = {"command_id": command_id, "status": "SUCCEEDED"}
        assert predicate(value), "the runner must wait for its documented command state"
        return value

    def read_state(self, _probe, path):
        if path.endswith("claim-state.json"):
            return {"counters": copy.deepcopy(self.state)}
        if path.endswith("adapter-executed.json"):
            return {"operation": self.v.OPERATION}
        return copy.deepcopy(self.state)

    def file_present(self, *_args):
        return self.failure == "marker"

    def pod_logs(self, _probe):
        return f"held: {self.v.HOLD_LOG}"

    def pod_phase(self, _probe):
        return "Failed" if self.failure == "phase" else "Running"

    def delete_owned_resource(self, *_args, **kwargs):
        self.events.append("stop-claimant")
        if kwargs:
            assert kwargs == {"require_uid": True, "expected_uid": "fixture-pod-uid"}, (
                "the claimant must stop under the recorded Pod UID"
            )

    def command_snapshot(self, _command_id):
        self.events.append("final-command")
        return {"status": "WAITING"}

    def probe_resource_uid(self, *_args):
        return "fixture-pod-uid"

    def dispatch(self):
        return {
            **self.seed,
            "first": {
                "status": "WAITING",
                "remote_command_id": "command-a",
                "remote_status": "PENDING",
            },
            "held": {
                "status": "WAITING",
                "details": {
                    "reason": "wrong"
                    if self.failure == "dispatch"
                    else self.v.HOLD_REASON,
                    "remote_command_id": "command-a",
                    "held_command_id": "command-b",
                    "remote_cluster_id": "perf-cap-000",
                    "remote_status": "PENDING",
                    "mutation_submitted_by_control_plane": False,
                },
            },
            "open_sibling_holds_total": 1,
            "open_command_ids": ["command-a"],
            "all_command_ids": ["command-a"],
        }

    def cpu_python(self, _source, *args):
        if len(args) == 6:
            self.events.append("dispatch")
            return self.dispatch()
        if len(args) == 1:
            return {"status": "RUNNING"}
        if len(args) == 3 and args[1] == self.v.OWNER:
            self.events.append("release")
            return {
                "outcome": {
                    "status": "WAITING",
                    "details": {"remote_command_id": "command-b"},
                },
                "cancelled": self.failure != "release",
                "cancelled_command": {"command_id": "command-b", "status": "FAILED"},
                "all_command_ids": ["command-a", "command-b"],
            }
        self.events.append("purge")
        if self.failure == "purge":
            raise RuntimeError("fixture purge read failed")
        return {"remaining_commands": 1 if self.failure == "purge-residual" else 0}

    def control(self, *args, **_kwargs):
        if self.failure == "metrics-all":
            raise RuntimeError("metrics unavailable")
        if args[0] == "get":
            if self.failure == "metrics-one" and "app=gpu-fault-control-worker" in args:
                raise RuntimeError("worker metrics unavailable")
            return "fixture-pod"
        return json.dumps({"metrics": f"{self.v.HOLDS_METRIC} 1"})

    def cleanup(self, _probe, _case_dir, _run_id, result, _seed, *, state, purge=None):
        self.events.append("cleanup")
        self.cleanup_state = copy.deepcopy(state)
        if purge is not None:
            try:
                purge(state["seed"])
            except Exception:
                result["verdict"] = "FAIL"
        if self.failure == "cleanup":
            result.update(verdict="FAIL", cleanup_error="fixture cleanup failed")
