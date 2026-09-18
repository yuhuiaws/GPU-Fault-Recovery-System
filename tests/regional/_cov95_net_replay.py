"""Fake Kubernetes/host protocol for the complete NET-001 runner."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net001_collector_replay as module
from tests.regional._cov95_collect_net import Clock


def replay_snapshot(host: Any, runner: Any) -> dict[str, Any]:
    host.snapshot_reads += 1
    matching = [
        {"test_ids": [marker]} for marker in host.buffered if marker in runner.test_ids
    ]
    boxes = {
        name: {
            "exists": True,
            "parent_exists": True,
            "parent_writable": True,
            "line_count": len(matching) if name == "kernel" else 0,
            "malformed_count": 0,
            "replayable_count": len(matching) if name == "kernel" else 0,
            "matching": matching if name == "kernel" else [],
        }
        for name in ("kernel", "dcgm", "host", "fabric-manager")
    }
    boxes["kernel"].update(host.kernel_overrides)
    if host.problem == "outage-ids" and host.blocked:
        boxes["kernel"]["matching"] = []
    rules = ["owned-rule"] if host.blocked else []
    if host.problem == "rule-drift" and host.blocked:
        rules = []
    result = {
        "services": {
            name: {"ActiveState": "active", "NRestarts": "0", **host.service_overrides}
            for name in module.SERVICES
        },
        "outboxes": boxes,
        "rules": rules,
        "kmsg_exists": True,
        "kmsg_writable": True,
        "existing_tagged_rules": [],
        "connectivity": {"192.0.2.1": True},
        "endpoint_ipv4": ["192.0.2.1"],
        "gpu_pci_bdf": "0000:af:00",
    }
    result.update(host.baseline_overrides)
    if host.snapshot_override is not None:
        host.snapshot_override(result)
    return result


@pytest.fixture
def replay_runner(tmp_path: Path, monkeypatch: Any) -> Any:
    cpu, gpu = tmp_path / "cpu-kubeconfig", tmp_path / "gpu-kubeconfig"
    cpu.touch()
    gpu.touch()
    settings = module.Settings(
        cpu_kubeconfig=cpu,
        cpu_context="cpu-a",
        gpu_kubeconfig=gpu,
        gpu_context="gpu-a",
        namespace="fixture",
        cluster_id="cluster-a",
        region="us-test-1",
        target_node="node-a",
        endpoint_host="control.invalid",
        host_probe_image="example@sha256:" + "a" * 64,
        predecessor={"valid": True},
    )
    runner = module.Runner(
        tmp_path, settings, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    host = SimpleNamespace(
        runner=runner,
        clock=Clock(),
        calls=[],
        manifests=[],
        resources={},
        armed=set(),
        blocked=set(),
        buffered=[],
        delivered=[],
        problem="",
        fail_at="",
        cleanup_started=False,
        cleanup_read_error=False,
        baseline_overrides={},
        kernel_overrides={},
        service_overrides={},
        store_override=None,
        snapshot_override=None,
        worker_names=["worker-b", "worker-a"],
        release_id="release-a",
        host_output=None,
        writes=0,
        snapshot_reads=0,
    )
    monkeypatch.setattr(module, "time", host.clock)

    snapshot = partial(replay_snapshot, host, runner)

    def store() -> dict[str, Any]:
        if host.store_override is not None:
            return deepcopy(host.store_override)
        markers = [value for value in host.delivered if value in runner.test_ids]
        if host.problem == "leaked-event" and host.blocked:
            markers = runner.test_ids[:1]
        if host.problem == "replay-timeout":
            markers = []
        return {
            "evidence": [{"record_id": "record-" + marker} for marker in markers],
            "incidents": [
                {
                    "workflow_request_id": "unexpected"
                    if host.problem == "workflow"
                    else None,
                    "state": "RECOVERED",
                    "decision_disposition": "MONITOR_ONLY",
                    "decision_action": "NO_ACTION",
                    "official_action": "IGNORE",
                }
                for marker in markers
            ],
            "notifications": [
                {"result_status": "SKIPPED", "provider_message_id_present": False}
                for marker in markers
            ],
        }

    def host_command(args: list[str]) -> dict[str, Any]:
        verb = args[0]
        host.calls.append(("host", args))
        if verb == "arm":
            tag = args[args.index("--tag") + 1]
            host.armed.add(tag)
            if host.fail_at == "arm-ack":
                raise module.RegionalFixtureError("rollback arm ACK lost")
            return {"timer": {"ActiveState": "active"}}
        if host.fail_at == verb:
            raise module.RegionalFixtureError(verb + " fixture failure")
        if verb in {"preflight", "snapshot"}:
            return snapshot()
        if verb == "write-kmsg":
            marker = args[args.index("--test-id") + 1]
            host.writes += 1
            if host.blocked:
                host.buffered.append(marker)
            else:
                host.delivered.extend(host.buffered)
                host.buffered.clear()
            return {"test_id": marker, "bytes_written": 1}
        tag = args[args.index("--tag") + 1]
        if verb == "block":
            host.blocked.add(tag)
            return {
                "rules": ["owned-rule"],
                "connectivity": {"192.0.2.1": host.problem == "block-reachable"},
            }
        if verb == "cleanup":
            host.blocked.discard(tag)
            host.armed.discard(tag)
            return {
                "rules": ["retained"] if host.problem == "cleanup-rules" else [],
                "connectivity": {"192.0.2.1": host.problem != "cleanup-unreachable"},
                "timer": {"ActiveState": "inactive"},
            }
        raise AssertionError(verb)

    def command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        host.calls.append(("command", argv, kwargs))
        assert argv[0] == "kubectl", (
            "all runner transport must use explicit Kubernetes scope"
        )
        is_gpu = argv[argv.index("--kubeconfig") + 1] == str(gpu)
        if is_gpu:
            assert argv[argv.index("--context") + 1] == "gpu-a"
        if "net001-node-probe" in argv:
            result = host_command(argv[argv.index("net001-node-probe") + 1 :])
            return subprocess.CompletedProcess(
                argv,
                0,
                host.host_output
                if host.host_output is not None
                else "fixture notice\n" + json.dumps(result),
                "",
            )
        if "apply" in argv:
            value = json.loads(kwargs["input_text"])
            host.manifests.append(value)
            host.resources[(value["kind"].lower(), value["metadata"]["name"])] = value
            return subprocess.CompletedProcess(argv, 0, "created", "")
        if "delete" in argv:
            host.cleanup_started = True
            index = argv.index("delete")
            kind, name = argv[index + 1 : index + 3]
            if host.problem != "resource-residual":
                host.resources.pop((kind, name), None)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "get" in argv:
            index = argv.index("get")
            kind = argv[index + 1]
            if kind == "configmap" and not is_gpu:
                result = {
                    "data": {"state.json": json.dumps({"release_id": host.release_id})}
                }
            elif "-A" in argv:
                business = host.problem == "workload" or (
                    host.problem == "late-workload" and host.writes >= 3
                )
                result = {
                    "items": [
                        {
                            "metadata": {
                                "name": "pod-a",
                                "namespace": "business" if business else "kube-system",
                            }
                        }
                    ]
                }
            elif "-l" in argv:
                result = {
                    "items": [
                        {"metadata": {"name": name}} for name in host.worker_names
                    ]
                }
            else:
                name = argv[index + 2]
                if host.cleanup_read_error:
                    if kwargs.get("check", True):
                        raise module.RegionalFixtureError("resource GET unavailable")
                    return subprocess.CompletedProcess(argv, 1, "", "GET unavailable")
                return subprocess.CompletedProcess(
                    argv, 0, name if (kind, name) in host.resources else "", ""
                )
            return subprocess.CompletedProcess(argv, 0, json.dumps(result), "")
        if "exec" in argv and not is_gpu:
            return subprocess.CompletedProcess(argv, 0, json.dumps(store()), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module, "run_fixture_command", command)
    host.snapshot = snapshot
    host.store = store
    return host
