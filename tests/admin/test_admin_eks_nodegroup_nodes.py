"""The read-only node inventory of ``prepare-eks-nodegroup``: facts, checks, drift.

Per node: python3.12 venv and strace always; in host mode the DCGM packages
accepted by the pin, the CUDA build of the driver's CUDA major, nvidia-dcgm
active and ``dcgmi discovery -l`` answering; in embedded mode an active host
engine is a finding. Across nodes: DCGM build drift and driver CUDA majors.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import eks_nodegroup_nodes as nodes
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.eks_nodegroup_user_data import dcgm_pin
from tests.admin._eks_nodegroup_world import (
    GPU_GROUP,
    GPU_NAME,
    REGION,
    FakeAws,
    FakeClock,
    World,
    old_node,
)

EXACT = dcgm_pin("4.7.0-2")
VERSION = dcgm_pin("4.7.0")


def _facts(**fields: object) -> nodes.NodeFacts:
    values: dict[str, object] = {
        "instance_id": "i-1",
        "ssm_status": "Success",
        "python_venv": "ok",
        "strace": "ok",
        "driver_cuda_major": "13",
        "packages": {
            "datacenter-gpu-manager-4-core": "4.7.0-2",
            "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
        },
        "service": "active",
        "hostengine": "running",
        "dcgmi": "ok",
    }
    values.update(fields)
    return nodes.NodeFacts(**values)  # type: ignore[arg-type]


def _failed(result: dict) -> dict[str, tuple[str, str]]:
    return {
        name: (check["observed"], check["expected"])
        for name, check in result["checks"].items()
        if not check["ok"]
    }


def test_the_inventory_script_reports_every_fact_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """Run the script under ``sh`` with stub tools on PATH."""

    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in {
        "nvidia-smi": "echo '    CUDA Version                          : 13.0'",
        "rpm": (
            "printf 'gpu-fault-node package datacenter-gpu-manager-4-core 4.7.0-2\\n"
            "gpu-fault-node package datacenter-gpu-manager-4-cuda13 4.7.0-1\\n'"
        ),
        "systemctl": "echo inactive; exit 3",
        "pgrep": "exit 1",
        "dcgmi": "exit 226",
        "strace": "exit 0",
        "cloud-init": "exit 0",
        "timeout": 'shift; "$@"',
    }.items():
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (stubs / name).chmod(0o755)
    script = "\n".join(nodes.inventory_commands(wait_for_cloud_init=True)) + "\n"
    completed = subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    facts = nodes.parse_inventory("i-1", "Success", completed.stdout)
    assert facts.driver_cuda_major == "13"
    assert facts.packages == {
        "datacenter-gpu-manager-4-core": "4.7.0-2",
        "datacenter-gpu-manager-4-cuda13": "4.7.0-1",
    }
    assert (facts.service, facts.hostengine, facts.dcgmi) == (
        "inactive",
        "absent",
        "failed",
    )
    assert facts.strace == "ok"
    assert not any(
        word in script for word in ("dnf ", "install", "enable", "start", "rm ")
    ), "the inventory never installs, starts or removes anything"


def test_host_mode_passes_with_the_pinned_build_of_the_driver_cuda_major() -> None:
    result = nodes.judge(_facts(), mode="host", pin=EXACT)

    assert result["passed"] is True, result
    assert set(result["checks"]) == {
        "python3.12-venv",
        "strace",
        "datacenter-gpu-manager-4-core",
        "dcgm-cuda-flavor",
        "dcgm-cuda-version",
        "nvidia-dcgm",
        "dcgmi-discovery",
    }
    assert result["findings"] == []


@pytest.mark.parametrize(
    ("fields", "failing"),
    [
        ({"python_venv": "missing"}, {"python3.12-venv": ("missing", "ok")}),
        ({"strace": "missing"}, {"strace": ("missing", "ok")}),
        (
            {"packages": {"datacenter-gpu-manager-4-cuda13": "4.7.0-2"}},
            {"datacenter-gpu-manager-4-core": ("unknown", "4.7.0-2")},
        ),
        (
            {
                "packages": {
                    "datacenter-gpu-manager-4-core": "4.7.0-2",
                    "datacenter-gpu-manager-4-cuda13": "4.7.0-1",
                }
            },
            {"dcgm-cuda-version": ("4.7.0-1", "4.7.0-2")},
        ),
        (
            {
                "packages": {
                    "datacenter-gpu-manager-4-core": "4.7.0-2",
                    "datacenter-gpu-manager-4-cuda12": "4.7.0-2",
                }
            },
            {
                "dcgm-cuda-flavor": (
                    "datacenter-gpu-manager-4-cuda12",
                    "datacenter-gpu-manager-4-cuda13",
                ),
                "dcgm-cuda-version": ("unknown", "4.7.0-2"),
            },
        ),
        ({"service": "inactive"}, {"nvidia-dcgm": ("inactive", "active")}),
        ({"dcgmi": "failed"}, {"dcgmi-discovery": ("failed", "ok")}),
    ],
)
def test_host_mode_names_each_failing_check(
    fields: dict[str, object], failing: dict[str, tuple[str, str]]
) -> None:
    result = nodes.judge(_facts(**fields), mode="host", pin=EXACT)

    assert result["passed"] is False
    assert _failed(result) == failing


def test_a_version_pin_accepts_either_build_of_the_version() -> None:
    drifted = _facts(
        packages={
            "datacenter-gpu-manager-4-core": "4.7.0-1",
            "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
        }
    )

    assert nodes.judge(drifted, mode="host", pin=VERSION)["passed"] is True
    assert _failed(nodes.judge(drifted, mode="host", pin=EXACT)) == {
        "datacenter-gpu-manager-4-core": ("4.7.0-1", "4.7.0-2")
    }


def test_a_second_cuda_build_beside_the_driver_one_is_a_finding() -> None:
    result = nodes.judge(
        _facts(
            packages={
                "datacenter-gpu-manager-4-core": "4.7.0-2",
                "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
                "datacenter-gpu-manager-4-cuda12": "4.7.0-2",
            }
        ),
        mode="host",
        pin=EXACT,
    )

    assert result["passed"] is True
    assert result["findings"] == [
        "i-1: datacenter-gpu-manager-4-cuda12 4.7.0-2 is installed beside the "
        "driver's CUDA 13"
    ]


def test_embedded_mode_reports_an_active_host_engine_as_a_finding() -> None:
    quiet = nodes.judge(
        _facts(service="inactive", hostengine="absent", dcgmi="failed", packages={}),
        mode="embedded",
        pin=None,
    )
    active = nodes.judge(_facts(), mode="embedded", pin=None)

    assert quiet["passed"] is True and quiet["findings"] == []
    assert set(quiet["checks"]) == {"python3.12-venv", "strace"}, (
        "embedded mode checks no DCGM host package"
    )
    assert active["passed"] is True, "the conflict is reported, not a failed check"
    assert active["findings"] == [
        "i-1: a host DCGM engine is active (nvidia-dcgm active, nv-hostengine "
        "running); it conflicts with the DCGM exporter's embedded engine"
    ]


def test_a_failed_ssm_invocation_fails_the_node() -> None:
    result = nodes.judge(_facts(ssm_status="Failed"), mode="host", pin=EXACT)

    assert result["passed"] is False and _failed(result) == {}


def test_fleet_findings_name_build_drift_and_driver_differences() -> None:
    one = _facts(instance_id="i-a")
    two = _facts(
        instance_id="i-b",
        driver_cuda_major="12",
        packages={
            "datacenter-gpu-manager-4-core": "4.7.0-1",
            "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
        },
    )

    assert nodes.fleet_findings([one, two]) == [
        "datacenter-gpu-manager-4-core differs across the nodes: i-a 4.7.0-2, "
        "i-b 4.7.0-1",
        "the driver CUDA major differs across the nodes: i-a 13, i-b 12",
    ]
    assert nodes.fleet_findings([one, _facts(instance_id="i-c")]) == []


def _read(world: World, **options: object) -> dict:
    clock = FakeClock()
    return nodes.read_nodes(
        FakeAws(world),
        region=REGION,
        cluster=GPU_NAME,
        nodegroup=GPU_GROUP,
        mode=str(options.get("mode", "embedded")),
        pin=options.get("pin"),  # type: ignore[arg-type]
        after_roll=bool(options.get("after_roll", False)),
        timeout_seconds=300.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )


def test_read_nodes_inventories_every_running_instance_over_ssm() -> None:
    world = World()
    world.node_facts = {
        "i-old1": old_node(
            packages={
                "datacenter-gpu-manager-4-core": "4.7.0-1",
                "datacenter-gpu-manager-4-cuda13": "4.7.0-1",
            }
        ),
        "i-old2": old_node(
            packages={
                "datacenter-gpu-manager-4-core": "4.7.0-2",
                "datacenter-gpu-manager-4-cuda13": "4.7.0-2",
            }
        ),
    }

    result = _read(world)

    assert result["outcome"] == "passed"
    assert [item["instance_id"] for item in result["nodes"]] == ["i-old1", "i-old2"]
    assert world.sent["instances"] == ["i-old1", "i-old2"]
    assert world.sent["commands"] == nodes.inventory_commands(wait_for_cloud_init=False)
    assert result["findings"] == [
        "datacenter-gpu-manager-4-core differs across the nodes: i-old1 4.7.0-1, "
        "i-old2 4.7.0-2",
        "datacenter-gpu-manager-4-cuda13 differs across the nodes: i-old1 4.7.0-1, "
        "i-old2 4.7.0-2",
    ], "the 4.7.0-1 / 4.7.0-2 drift of the current nodes is shown"
    assert result["driver_cuda_majors"] == ["13"]
    assert all(flag is False for argv, flag in world.calls), (
        "the inventory is a read: no call is marked mutating"
    )


def test_a_long_node_group_name_keeps_the_ssm_comment_within_its_limit() -> None:
    world = World()
    long_name = "g" * 63
    world.instances = {long_name: ["i-1"]}
    clock = FakeClock()

    result = nodes.read_nodes(
        FakeAws(world),
        region=REGION,
        cluster=GPU_NAME,
        nodegroup=long_name,
        mode="embedded",
        pin=None,
        after_roll=False,
        timeout_seconds=60.0,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )

    assert result["outcome"] == "passed", result


def test_read_nodes_retries_an_unregistered_instance_then_reads_it() -> None:
    world = World()
    world.ssm_send_errors = ["InvalidInstanceId: Instances not in a valid state"]

    result = _read(world, after_roll=True)

    assert result["outcome"] == "passed"
    assert world.sent["commands"][0].startswith(
        "timeout 900 cloud-init status --wait"
    ), "after a roll the inventory first waits for cloud-init"


def test_read_nodes_is_unavailable_without_ssm_and_skipped_without_nodes() -> None:
    denied = World()
    denied.ssm_send_errors = ["AccessDeniedException: not authorized"]
    empty = World()
    empty.instances = {}

    unavailable = _read(denied)
    skipped = _read(empty)

    assert unavailable["outcome"] == "unavailable"
    assert "AccessDeniedException" in unavailable["reason"]
    assert "AmazonSSMManagedInstanceCore" in unavailable["remedy"]
    assert "kubectl debug node/" in unavailable["remedy"]
    assert skipped == {
        "outcome": "skipped",
        "reason": "no running instance",
        "nodes": [],
    }


def test_an_invocation_that_never_finishes_is_bounded() -> None:
    class Slow(World):
        def ssm_get_command_invocation(self, options):  # type: ignore[no-untyped-def]
            return {"Status": "InProgress"}

    result = _read(Slow())

    assert result["outcome"] == "failed"
    assert {item["ssm_status"] for item in result["nodes"]} == {"TimedOutWaiting"}


def test_an_unexpected_invocation_error_is_unavailable() -> None:
    class Broken(World):
        def ssm_get_command_invocation(self, options):  # type: ignore[no-untyped-def]
            raise BootstrapError("ThrottlingException")

    assert _read(Broken())["outcome"] == "unavailable"
