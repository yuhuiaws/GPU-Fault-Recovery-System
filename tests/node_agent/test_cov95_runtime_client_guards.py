from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import FakeRunner, command, envelope
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("failure", ["missing", "permission"])
def test_unreadable_process_inventory_cannot_prove_no_gpu_clients(
    monkeypatch: pytest.MonkeyPatch,
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    failure: str,
) -> None:
    proc = tmp_path / "proc-view"
    runner = FakeRunner()
    agent = node_factory(
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        proc_root=str(proc),
        device_client_finder=None,
        runner=runner,
    )
    signed = envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS))
    original_iterdir = Path.iterdir

    def unavailable(path: Path) -> Any:
        if path == proc:
            raise PermissionError("synthetic proc visibility denied")
        return original_iterdir(path)

    with monkeypatch.context() as patches:
        if failure == "permission":
            proc.mkdir()
            patches.setattr(Path, "iterdir", unavailable)
        refused = agent.execute(signed)
    assert refused.status is NodeActionStatus.FAILED
    assert refused.retryable is True
    assert "verified_no_gpu_clients" not in refused.details
    proc.mkdir(exist_ok=True)
    recovered = agent.execute(signed)
    assert recovered.status is NodeActionStatus.SUCCEEDED
    assert recovered.attempt == 2
    assert recovered.details["device_clients_checked"] is True
    assert all("--gpu-reset" not in argv for argv in runner.commands), (
        "a visibility failure must never dispatch a reset"
    )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({}, set()),
        ({"workload_cgroup_paths": None}, set()),
        ({"workload_cgroup_paths": ["/job///", "", "/"]}, {"/job"}),
        (
            {"workload_cgroup_paths_by_node": {"node-a": ["/a/"], "node-b": ["/b/"]}},
            {"/a"},
        ),
    ],
)
@pytest.mark.parametrize("whole_node", [False, True])
def test_quiesce_scope_uses_only_the_target_node_and_normalized_cgroups(
    node_factory: Callable[..., NodeActionExecutor],
    parameters: dict[str, Any],
    expected: set[str],
    whole_node: bool,
) -> None:
    calls = []

    class Manager:
        def quiesce(self, **kwargs: Any) -> dict[str, bool]:
            calls.append(kwargs)
            return {"quiesced": True}

    agent = node_factory(
        allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Manager(),
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                gpu_uuids=[] if whole_node else ["GPU-a"],
                parameters=parameters,
            )
        )
    )
    assert result.status is NodeActionStatus.SUCCEEDED
    assert calls == [
        {
            "incident_id": "incident-a",
            "workflow_request_id": "workflow-a",
            "target_device_paths": {"/unit/nvidia0", "/unit/nvidia1"}
            if whole_node
            else {"/unit/nvidia0"},
            "workload_cgroup_paths": expected,
        }
    ]


@pytest.mark.parametrize("bad_paths", ["not-a-list", [1], {"path": "/job"}])
def test_malformed_cgroup_scope_is_refused_before_the_quiesce_manager(
    node_factory: Callable[..., NodeActionExecutor], bad_paths: Any
) -> None:
    calls = []

    class Manager:
        def quiesce(self, **kwargs: Any) -> dict:
            calls.append(kwargs)
            return {}

    agent = node_factory(
        allowed_operations={WorkflowOperation.QUIESCE_GPU_SERVICES},
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=Manager(),
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                parameters={"workload_cgroup_paths": bad_paths},
            )
        )
    )
    assert result.status is NodeActionStatus.FAILED
    assert "list of strings" in result.error
    assert calls == []


@pytest.mark.parametrize("name", ["missing", "invalid-utf8"])
def test_unreadable_process_name_does_not_hide_an_owned_gpu_descriptor(
    node_factory: Callable[..., NodeActionExecutor], tmp_path: Path, name: str
) -> None:
    proc = tmp_path / "proc"
    holder = proc / "100"
    (holder / "fd").mkdir(parents=True)
    (holder / "fd" / "7").symlink_to("/unit/nvidia0")
    (holder / "fd" / "8").write_text("closed descriptor", encoding="ascii")
    (proc / "101" / "fd").mkdir(parents=True)
    (proc / "101" / "fd" / "7").symlink_to("/unit/unrelated")
    (proc / "102").mkdir()
    (proc / "not-a-pid").mkdir()
    if name == "invalid-utf8":
        (holder / "comm").write_bytes(b"\xff")
    agent = node_factory(
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        device_client_finder=None,
    )
    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))
    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a:100:unknown" in result.error
    assert result.retryable is False


@pytest.mark.parametrize("inventory_available", [False, True])
def test_mixed_unresolvable_gpu_causes_preserve_each_targets_evidence(
    node_factory: Callable[..., NodeActionExecutor], inventory_available: bool
) -> None:
    queries = []

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        queries.append(argv)
        text = ""
        if argv == ["nvidia-smi", "-q", "-x"]:
            text = (
                "<nvidia_smi_log><gpu><minor_number>5</minor_number></gpu>"
                "<gpu><uuid>GPU-a</uuid><minor_number>N/A</minor_number></gpu>"
                "<gpu><uuid>GPU-b</uuid><minor_number>1</minor_number></gpu>"
                "</nvidia_smi_log>"
            )
        elif "--query-gpu=uuid" in argv:
            if not inventory_available:
                raise OSError("synthetic inventory unavailable")
            text = "GPU-a\nGPU-b\n"
        return CompletedProcess(argv, 0, stdout=text, stderr="")

    agent = node_factory(
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        gpu_device_path_finder=None,
        runner=runner,
    )
    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                gpu_uuids=["GPU-a", "GPU-unknown"],
            )
        )
    )
    assert result.status is NodeActionStatus.FAILED
    assert "GPU-a (the driver reports no minor number" in result.error
    assert "GPU-unknown" in result.error
    assert (
        "no GPU on this node reports this UUID"
        if inventory_available
        else "the driver reported no device node for it"
    ) in result.error
    assert sum("--query-gpu=uuid" in argv for argv in queries) == 1


def test_unsupported_minor_number_probe_falls_back_to_a_complete_index_map(
    node_factory: Callable[..., NodeActionExecutor],
) -> None:
    queries = []

    def runner(argv: list[str], **kwargs: Any) -> CompletedProcess:
        queries.append(argv)
        if argv == ["nvidia-smi", "-q", "-x"]:
            raise OSError("synthetic driver does not support XML query")
        text = "invalid\nGPU-a, 0\n" if "--query-gpu=uuid,index" in argv else ""
        return CompletedProcess(argv, 0, stdout=text, stderr="")

    agent = node_factory(
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        gpu_device_path_finder=None,
        runner=runner,
    )
    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))
    assert result.status is NodeActionStatus.SUCCEEDED
    assert sum("--query-gpu=uuid,index" in argv for argv in queries) == 1
