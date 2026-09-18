from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutor
from tests.node_agent._cov95_runtime_hung import (
    COMPUTE_QUERY,
    BundleRunner,
    capture_bundle,
    process,
)
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("explicit", [True, False])
def test_bundle_preserves_empty_or_failed_process_discovery_without_sampling_unowned_pids(
    node_factory: Callable[..., NodeActionExecutor], explicit: bool
) -> None:
    runner = BundleRunner("", returncode=3, stderr="synthetic inventory refusal")
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE}, runner=runner
    )
    manifest, contents = capture_bundle(
        agent,
        target_pids_by_node={"node-a": ["invalid"]} if explicit else [],
        target_gpu_uuids_by_pid_by_node=[],
    )
    assert manifest["gpu_processes"] == []
    assert manifest["strace_sampling"]["process_count"] == 0
    assert manifest["python_stack_sampling"]["process_count"] == 0
    assert not any(
        argv[0] in {"timeout", agent.python_stack_tool} for argv in runner.calls
    ), "an empty discovery must not expand to host-wide diagnostic attachment"
    capture = next(
        item
        for item in manifest["captures"]
        if item["file"] == "nvidia-compute-processes.csv"
    )
    assert capture["returncode"] == (0 if explicit else 3)
    assert contents["nvidia-compute-processes.csv"] == (
        b"" if explicit else b"\n[stderr]\nsynthetic inventory refusal"
    )


def test_explicit_target_with_disappeared_comm_remains_the_only_sampled_pid(
    node_factory: Callable[..., NodeActionExecutor],
) -> None:
    runner = BundleRunner("999, GPU-a, python\n")
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE}, runner=runner
    )
    manifest, _contents = capture_bundle(
        agent,
        target_pids_by_node={"node-a": ["bad", 100]},
        target_gpu_uuids_by_pid_by_node={"node-a": {"100": "GPU-a"}},
    )
    assert manifest["gpu_processes"] == [
        {
            "pid": 100,
            "gpu_uuid": "GPU-a",
            "process_name": "unknown",
            "association": "hung_triage_target",
        }
    ]
    assert not any(COMPUTE_QUERY in argv for argv in runner.calls), (
        "explicit triage targets must not be replaced by a fresh wider GPU inventory"
    )
    samples = [argv for argv in runner.calls if argv[0] == "timeout"]
    assert len(samples) == 1
    assert samples[0][samples[0].index("-p") + 1] == "100"


@pytest.mark.parametrize("inventory_unreadable", [False, True])
def test_cgroup_expansion_only_adds_readable_python_members_of_the_gpu_process_cgroup(
    node_factory: Callable[..., NodeActionExecutor],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory_unreadable: bool,
) -> None:
    root = tmp_path / "proc"
    process(root, 100)
    process(root, 101)
    process(root, 102, cgroup="/different/workload")
    process(root, 103, name="native-worker")
    (process(root, 104) / "cmdline").unlink()
    (root / "not-a-pid").mkdir()
    real_iterdir = Path.iterdir

    def iterdir(path: Path) -> Any:
        if inventory_unreadable and path == root:
            raise PermissionError("synthetic process inventory refusal")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    runner = BundleRunner("invalid\n100, GPU-a, python\n999, GPU-b, python\n")
    agent = node_factory(
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        runner=runner,
        python_stack_tool="/unit/py-spy",
    )
    manifest, _contents = capture_bundle(agent)
    expected = [100] if inventory_unreadable else [100, 101]
    assert [row["pid"] for row in manifest["gpu_processes"]] == expected
    if not inventory_unreadable:
        assert (
            manifest["gpu_processes"][1]["association"] == "training_container_cgroup"
        )
        assert manifest["gpu_processes"][1]["source_gpu_pids"] == [100]
    assert manifest["python_stack_sampling"]["process_count"] == len(expected)
    assert manifest["strace_sampling"]["process_count"] == len(expected)
    stacks = [argv for argv in runner.calls if argv[0] == "/unit/py-spy"]
    assert len(stacks) == 3 * len(expected)
    assert {int(argv[-1]) for argv in stacks} == set(expected)
    traces = [argv for argv in runner.calls if argv[0] == "timeout"]
    assert {int(argv[argv.index("-p") + 1]) for argv in traces} == set(expected)
