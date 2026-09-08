# ruff: noqa: F401
from __future__ import annotations

import gzip
import hashlib
import inspect
import io
import json
import logging
import os
import pickle
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, TimeoutExpired
from threading import Event
from typing import Any, Callable
from unittest import mock
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import (
    AgentHeartbeatRejected,
    AgentHeartbeatReporter,
    GpuServiceQuiesceManager,
    NodeActionCommand,
    NodeActionExecutionState,
    NodeActionExecutor,
    NodeActionLedger,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    agent_config_digest,
    agent_config_payload,
    create_node_agent_app,
    executor_from_environment,
    heartbeat_reporter_from_environment,
    print_config_digest,
    run,
    sign_node_action,
    sign_result_query,
    validate_host_proc_root,
)
from gpu_fault.node_agent.operations.registry import OPERATION_HANDLERS

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
SECRET = "node-action-secret-" + "x" * 32


class FakeRunner:
    def __init__(
        self,
        clients: str = "",
        reset_error: str | None = None,
        reset_timeout_seconds: int | None = None,
        reset_timeout_gpu_uuid: str | None = None,
    ) -> None:
        self.clients = clients
        self.reset_error = reset_error
        # A reset that never returns: subprocess.run SIGKILLs nvidia-smi while
        # the in-kernel reset keeps going, so the outcome is unknown. Pinning a
        # GPU UUID stops a multi-GPU loop part way through.
        self.reset_timeout_seconds = reset_timeout_seconds
        self.reset_timeout_gpu_uuid = reset_timeout_gpu_uuid
        self.commands: list[list[str]] = []

    def __call__(self, command, **_):
        self.commands.append(command)
        if (
            "--gpu-reset" in command
            and self.reset_timeout_seconds is not None
            and (
                self.reset_timeout_gpu_uuid is None
                or self.reset_timeout_gpu_uuid in command
            )
        ):
            raise TimeoutExpired(command, self.reset_timeout_seconds)
        if "--gpu-reset" in command and self.reset_error:
            raise CalledProcessError(255, command, stderr=self.reset_error)
        stdout = (
            self.clients
            if "--query-compute-apps=gpu_uuid,pid,process_name" in command
            else "GPU-a\n"
            if "--query-gpu=uuid" in command
            else "GPU-a, 0\n"
            if "--query-gpu=uuid,index" in command
            else ""
        )
        return CompletedProcess(command, 0, stdout=stdout, stderr="")


def no_device_clients(_: set[str]) -> list[dict[str, str]]:
    return []


def command(
    operation: WorkflowOperation,
    *,
    command_id: str = "workflow/step/node-a",
    fencing_token: int = 1,
    parameters: dict | None = None,
    gpu_uuids: list[str] | None = None,
) -> NodeActionCommand:
    return NodeActionCommand(
        command_id=command_id,
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        fencing_token=fencing_token,
        operation=operation,
        node_id="node-a",
        gpu_uuids=gpu_uuids if gpu_uuids is not None else ["GPU-a"],
        parameters=parameters or {},
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=2),
    )


def executor(tmp_path, runner: FakeRunner) -> NodeActionExecutor:
    return NodeActionExecutor(
        secret=SECRET,
        node_ids={"node-a"},
        allowed_operations={
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        },
        reset_enabled=True,
        ledger=NodeActionLedger(str(tmp_path / "actions.db")),
        runner=runner,
        device_client_finder=no_device_clients,
        # The three GPUs the reset tests target. A target the device map cannot
        # resolve now fails the client verification closed instead of being
        # dropped from the scan, so the fixture has to declare the whole node.
        gpu_device_path_finder=lambda: {
            "GPU-a": "/dev/nvidia0",
            "GPU-b": "/dev/nvidia1",
            "GPU-c": "/dev/nvidia2",
        },
        now=lambda: NOW,
        sleep=lambda _: None,
    )


def node_action_executor(
    tmp_path,
    ledger_name: str,
    *,
    allowed_operations: set[WorkflowOperation],
    reset_enabled: bool = False,
    now: Callable[[], datetime] | None = lambda: NOW,
    **overrides,
) -> NodeActionExecutor:
    parameters = {
        "secret": SECRET,
        "node_ids": {"node-a"},
        "allowed_operations": allowed_operations,
        "reset_enabled": reset_enabled,
        "ledger": NodeActionLedger(str(tmp_path / ledger_name)),
        **overrides,
    }
    if now is not None:
        parameters["now"] = now
    return NodeActionExecutor(**parameters)


def envelope(value: NodeActionCommand) -> SignedNodeAction:
    return SignedNodeAction(command=value, signature=sign_node_action(value, SECRET))


def result_params(command_id: str, *, issued_at: str | None = None) -> dict[str, str]:
    """Query parameters for an authenticated result poll.

    The agent under test is pinned to a fixed ``now`` for command
    validation, but the result query's replay window is checked against
    the real clock, so the timestamp here is a real one.
    """
    stamp = issued_at or datetime.now(timezone.utc).isoformat()
    return {
        "command_id": command_id,
        "issued_at": stamp,
        "signature": sign_result_query(command_id, stamp, SECRET),
    }


def submit_action(client: TestClient, signed: SignedNodeAction) -> Any:
    """POST one signed envelope to the asynchronous submit route."""

    return client.post("/v1/node-actions/submit", json=signed.model_dump(mode="json"))


def wait_for_result(client: TestClient, command_id: str, *, tries: int = 200) -> Any:
    """Poll ``/result`` until the pool stops answering PENDING.

    There is no synchronous route, so every HTTP test that needs a finished
    action polls exactly like the control plane's transport does. A 404 or a
    non-PENDING state is an answer and is returned as-is.
    """

    for _ in range(tries):
        polled = client.get("/v1/node-actions/result", params=result_params(command_id))
        if (
            polled.status_code != 200
            or polled.json()["state"] != NodeActionExecutionState.PENDING.value
        ):
            return polled
        time.sleep(0.02)
    raise AssertionError(f"node action {command_id} never left PENDING")


class Quiesced:
    """A quiesce manager stub mirroring the real fencing signature."""

    def assert_quiesced(
        self,
        *,
        incident_id: str,
        command_id: str | None = None,
        for_reset: bool = True,
        attempt: int | None = None,
    ) -> None:
        assert incident_id == "incident-a", incident_id


class ServiceRunner:
    def __init__(
        self,
        *,
        active: set[str],
        fail_stop: str | None = None,
        fail_start: str | None = None,
    ) -> None:
        self.active = set(active)
        self.fail_stop = fail_stop
        self.fail_start = fail_start
        self.commands: list[list[str]] = []

    def __call__(self, value, *, check=False, **_):
        self.commands.append(value)
        returncode = 0
        stderr = ""
        if value[:3] == ["systemctl", "is-active", "--quiet"]:
            returncode = 0 if value[3] in self.active else 3
        elif value[:2] == ["systemctl", "stop"]:
            service = value[2]
            if service == self.fail_stop:
                returncode = 1
                stderr = "stop failed"
            else:
                self.active.discard(service)
        elif value[:2] == ["systemctl", "start"]:
            service = value[2]
            if service == self.fail_start:
                returncode = 1
                stderr = "start failed"
            else:
                self.active.add(service)
        elif value[:2] == ["systemctl", "restart"]:
            service = value[2]
            if service == self.fail_start:
                returncode = 1
                stderr = "restart failed"
            else:
                self.active.add(service)
        stdout = "101\n" if value[:2] == ["systemctl", "show"] else ""
        completed = CompletedProcess(value, returncode, stdout=stdout, stderr=stderr)
        if check and returncode:
            raise CalledProcessError(returncode, value, stderr=stderr)
        return completed


def quiesce_manager(tmp_path, runner) -> GpuServiceQuiesceManager:
    return GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("nvidia-fabricmanager", "nvidia-dcgm", "kubelet"),
        processes=("nv-hostengine",),
        failsafe_seconds=30,
        retry_seconds=10,
        settle_seconds=0,
        restore_settle_seconds=0,
        restore_command="/opt/gpu-fault/restore",
        runner=runner,
    )


def quiesce_executor(tmp_path, runner) -> NodeActionExecutor:
    manager = quiesce_manager(tmp_path, runner)
    return NodeActionExecutor(
        secret=SECRET,
        node_ids={"node-a"},
        allowed_operations={
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        },
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=manager,
        ledger=NodeActionLedger(str(tmp_path / "quiesce-actions.db")),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        now=lambda: NOW,
        # The device-client sampling interval is real seconds and no quiesce
        # test asserts on it; a reset samples twice (preflight, then the
        # reset's own re-check), which is 8 s of waiting per reset.
        sleep=lambda _: None,
    )


def _triage_only_executor(tmp_path, sleep) -> NodeActionExecutor:
    return NodeActionExecutor(
        secret=SECRET,
        node_ids={"node-a"},
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        reset_enabled=False,
        ledger=NodeActionLedger(str(tmp_path / "await.db")),
        runner=lambda argv, **_: CompletedProcess(argv, 0, "", ""),
        sleep=sleep,
        now=lambda: NOW,
    )


def _flight_dump_payload() -> str:
    return json.dumps(
        {
            "entries": [
                {
                    "pg_name_": "default",
                    "collective_seq_id_": 12,
                    "profiling_name_": "nccl:all_reduce",
                    "time_discovered_started_": None,
                    "time_discovered_completed_": None,
                    "time_created_": "2026-08-16T01:00:00Z",
                }
            ]
        }
    )


def _torch_flight_dump(seq: int, state: str) -> dict[str, object]:
    """Shape torch 2.10 actually writes (verified on ml.p5en.48xlarge)."""

    return {
        "version": "2.10",
        "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}},
        "pg_status": {
            "0": {
                "last_enqueued_collective": seq,
                "last_started_collective": -1,
                "last_completed_collective": seq - 1,
            }
        },
        "entries": [
            {
                "record_id": seq - 1,
                "pg_id": 0,
                "process_group": ("0", "default_pg"),
                "collective_seq_id": seq,
                "p2p_seq_id": 0,
                "op_id": seq,
                "profiling_name": "nccl:all_reduce",
                "state": state,
                "retired": state == "completed",
                "is_p2p": False,
                "time_created_ns": 1786844121207040384,
                "time_discovered_started_ns": None,
                "time_discovered_completed_ns": (
                    1786844121735009183 if state == "completed" else None
                ),
                "input_sizes": [[16777216]],
                "output_sizes": [[16777216]],
                "timeout_ms": 600000,
            }
        ],
    }


__all__ = [name for name in globals() if not name.startswith("__")]
