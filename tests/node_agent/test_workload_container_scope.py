"""A managed job's workload scope comes from its runtime container ids.

Live, HyperPod (GF-GENERAL-LIVE-011, every variant of live3 and live4,
2026-10-24/25): the watcher only knows a container's cgroup from a
``gpu-fault.io/cgroup-path`` Pod annotation that nothing sets -- the path names
the runtime's container id, which exists only once the container runs. Every
hung DAG therefore carried ``workload_cgroup_paths_by_node`` = ``{node: []}``,
both nodes' COLLECT_HUNG_SUMMARY answered ``SCOPE_UNKNOWN`` and
COLLECT_HUNG_TRIAGE was skipped (``NO_VALID_SURVEY``): ``fr_enabled`` read
"COLLECT_HUNG_TRIAGE sampled no rank process" with the flight recorder on.

What the watcher does know of every running container is the id the kubelet
reports (``containerStatuses[].containerID``). The control plane now sends it
per node and the Agent finds the container's cgroup in its own
``/proc/<pid>/cgroup``, where the runtime names the cgroup after that id.
"""

import hashlib

import pytest

from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus
from tests.node_agent._cov95_runtime_hung import COMPUTE_QUERY, process
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.node_agent._support import command, envelope
from tests.node_agent.test_hung_collection_hardening import stat_text

TRAINER = hashlib.sha256(b"trainer-container").hexdigest()
PEER = hashlib.sha256(b"second-trainer-container").hexdigest()
OTHER = hashlib.sha256(b"another-job-container").hexdigest()
POD = "kubepods-burstable-pod835a5f72_cf56_4102_912f_3dbb2ddfa995.slice"
POD_DIR = f"/kubepods.slice/kubepods-burstable.slice/{POD}"


def scope(identifier: str, runtime: str = "cri-containerd") -> str:
    """The cgroup containerd's systemd driver gives a container (cgroup v2)."""

    return f"{POD_DIR}/{runtime}-{identifier}.scope"


def rank(root, pid, cgroup, environment=b""):
    directory = process(root, pid, cgroup=cgroup)
    (directory / "stat").write_text(stat_text(pid, 1000))
    (directory / "environ").write_bytes(environment)
    return directory


def parameters(container_ids, cgroups=None):
    return {
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "snapshot_epoch": "epoch",
        # What every live hung DAG carried (hung-records.json, 2026-10-25).
        "workload_cgroup_paths_by_node": {"node-a": cgroups or []},
        **(
            {"workload_container_ids_by_node": {"node-a": container_ids}}
            if container_ids is not None
            else {}
        ),
    }


def summary(node_factory, container_ids, cgroups=None):
    agent = node_factory(allowed_operations={Op.COLLECT_HUNG_SUMMARY})
    result = agent.execute(
        envelope(
            command(
                Op.COLLECT_HUNG_SUMMARY, parameters=parameters(container_ids, cgroups)
            )
        )
    )
    assert result.status is NodeActionStatus.SUCCEEDED, result
    return result.details


def test_summary_scans_every_process_of_the_container_its_runtime_id_names(
    node_factory, tmp_path
):
    proc = tmp_path / "proc"
    # torchrun (no RANK) and its 8 local workers share the training container.
    rank(proc, 99, scope(TRAINER))
    for local in range(8):
        rank(
            proc, 100 + local, scope(TRAINER), f"RANK={local}\0WORLD_SIZE=16\0".encode()
        )
    # Another job's rank, and the runtime's monitor named after our id: neither
    # is a process of the training container.
    rank(proc, 900, scope(OTHER), b"RANK=0\0WORLD_SIZE=8\0")
    rank(proc, 950, f"{POD_DIR}/crio-conmon-{TRAINER}.scope")

    details = summary(node_factory, [f"containerd://{TRAINER}"])

    assert details["status"] == "COLLECTED", details
    assert sorted(row["pid"] for row in details["ranks"]) == [99, *range(100, 108)]
    assert all(row["process_identity"]["verified"] for row in details["ranks"]), details
    assert details["scan_complete"] is True, details
    assert details["workload_scope"] == {
        "basis": "RUNTIME_CONTAINER_ID",
        "container_count": 1,
        "resolved_container_count": 1,
        "unresolved_container_count": 0,
        "unresolved_container_ids": [],
        "cgroup_path_count": 1,
        "complete": True,
    }, details


def test_the_recorded_live_summary_without_any_scope_stays_scope_unknown(
    node_factory, tmp_path
):
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=16\0")

    details = summary(node_factory, None)

    assert details["status"] == "SCOPE_UNKNOWN", details
    assert details["ranks"] == [] and details["scan_complete"] is False, details
    assert details["workload_scope"] == {
        "basis": "UNAVAILABLE",
        "reason": "NO_CGROUP_PATH_OR_CONTAINER_ID",
    }, details


def test_a_container_without_a_process_here_is_unknown_never_an_empty_scan(
    node_factory, tmp_path
):
    rank(tmp_path / "proc", 900, scope(OTHER), b"RANK=0\0WORLD_SIZE=8\0")

    details = summary(node_factory, [TRAINER])

    assert details["status"] == "SCOPE_UNKNOWN", details
    assert details["ranks"] == [], details
    assert details["workload_scope"]["reason"] == "CONTAINER_CGROUP_NOT_FOUND"
    assert details["workload_scope"]["unresolved_container_ids"] == [TRAINER]


def test_one_resolved_container_of_two_is_an_incomplete_scan(node_factory, tmp_path):
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=16\0")

    details = summary(node_factory, [TRAINER, PEER])

    assert details["status"] == "COLLECTED", details
    assert [row["pid"] for row in details["ranks"]] == [100]
    assert details["scan_complete"] is False, "an unresolved container hides ranks"
    assert details["workload_scope"]["unresolved_container_ids"] == [PEER]
    assert details["workload_scope"]["complete"] is False


@pytest.mark.parametrize(
    "cgroup",
    [
        # containerd, systemd driver, cgroup v2 (EKS, HyperPod)
        f"0::{scope(TRAINER)}",
        # containerd or Docker, cgroupfs driver, cgroup v1: one line per
        # hierarchy, none of them the unified one.
        "\n".join(
            f"{index}:{name}:/kubepods/burstable/"
            f"pod835a5f72-cf56-4102-912f-3dbb2ddfa995/{TRAINER}"
            for index, name in enumerate(("cpu,cpuacct", "memory", "pids"), 3)
        ),
        f"0::{scope(TRAINER, 'crio')}",
        f"0::{scope(TRAINER, 'docker')}",
        # A node that is itself a container (kind): the outer id is not ours.
        f"0::/docker/{OTHER}/kubepods/burstable/pod835a5f72/{TRAINER}",
        # A cgroup the training container created below its own.
        f"0::{scope(TRAINER)}/init.scope",
    ],
)
def test_every_runtime_layout_resolves_to_the_container_cgroup(
    node_factory, tmp_path, cgroup
):
    directory = rank(tmp_path / "proc", 100, "/unused", b"RANK=0\0WORLD_SIZE=16\0")
    (directory / "cgroup").write_text(cgroup + "\n")

    details = summary(node_factory, [f"containerd://{TRAINER}"])

    assert [row["pid"] for row in details["ranks"]] == [100], details
    assert details["workload_scope"]["resolved_container_count"] == 1, details


@pytest.mark.parametrize(
    "value",
    [
        "containerd://abc123",
        f"containerd://{TRAINER.upper()}",
        f"containerd://{TRAINER}0",
        None,
        123,
    ],
)
def test_a_malformed_container_id_proves_nothing(node_factory, tmp_path, value):
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=16\0")

    details = summary(node_factory, [value])

    assert details["status"] == "SCOPE_UNKNOWN", details
    assert details["workload_scope"]["reason"] == "NO_CGROUP_PATH_OR_CONTAINER_ID"


def test_an_explicit_cgroup_path_and_a_runtime_id_are_both_scope(
    node_factory, tmp_path
):
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0WORLD_SIZE=2\0")
    rank(tmp_path / "proc", 101, "/training/job", b"RANK=1\0WORLD_SIZE=2\0")

    details = summary(node_factory, [TRAINER], cgroups=["/training/job"])

    assert [row["pid"] for row in details["ranks"]] == [100, 101], details
    assert details["workload_scope"]["explicit_cgroup_path_count"] == 1, details


def test_triage_attaches_only_to_the_container_its_runtime_id_names(
    node_factory, tmp_path
):
    rank(tmp_path / "proc", 100, scope(TRAINER), b"RANK=0\0")
    # Holds the same GPU, from another job's container: never attached.
    rank(tmp_path / "proc", 900, scope(OTHER), b"RANK=0\0")
    calls = []

    def runner(argv, **kwargs):
        from subprocess import CompletedProcess

        calls.append(argv)
        text = "100, GPU-a, python\n900, GPU-a, python\n"
        return CompletedProcess(argv, 0, text if COMPUTE_QUERY in argv else "", "")

    agent = node_factory(
        allowed_operations={Op.COLLECT_HUNG_TRIAGE},
        runner=runner,
        python_stack_tool="/unit/py-spy",
    )
    result = agent.execute(
        envelope(
            command(
                Op.COLLECT_HUNG_TRIAGE,
                parameters={
                    **parameters([f"containerd://{TRAINER}"]),
                    "require_workload_identity": True,
                },
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED, result
    [row] = result.details["ranks"]
    assert row["pid"] == 100, result.details
    assert row["process_identity"]["verified"] is True, row
    assert result.details["workload_scope"]["resolved_container_count"] == 1
    assert all(argv[-1] != "900" for argv in calls if argv[0] == "/unit/py-spy"), (
        "another job's process is never attached"
    )
