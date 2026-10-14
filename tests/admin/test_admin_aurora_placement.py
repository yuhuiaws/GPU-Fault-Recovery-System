"""``status``'s Aurora writer placement finding.

Live 2026-10-13: HA-010's injected failover left the writer in us-west-2b under
a control plane entirely in us-west-2a (+~1.2 ms per database round trip,
single-cluster burst admission p99 x3) and nothing said so. ``status`` now
reads where the writer runs against the zones of the nodes running
gpu-fault-api-ha and gpu-fault-control-worker and prints a finding -- never a
failing check -- naming the supported remedy. The reads are bounded, read-only
and fail soft: an unreadable placement is a finding line too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import aurora_placement as placement
from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import status_report
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._aurora_placement_support import CLUSTER_ID, World, site
from tests.admin.test_admin_site import site_file


def _findings(world: World, tmp_path: Path) -> list[str]:
    return placement.placement_findings(
        site(tmp_path),  # type: ignore[arg-type]
        state_dir=tmp_path / "state",
        runner=world,  # type: ignore[arg-type]
    )


def test_a_writer_in_the_control_plane_zone_is_no_finding(tmp_path: Path) -> None:
    world = World(writer="aurora-a-writer")

    assert _findings(world, tmp_path) == []
    assert all(
        timeout == placement.READ_TIMEOUT_SECONDS for timeout in world.timeouts
    ), f"every read is bounded: {world.timeouts}"
    assert [call[:3] for call in world.calls] == [
        ["kubectl", "--kubeconfig", str(tmp_path / "cpu.kubeconfig")],
        ["kubectl", "--kubeconfig", str(tmp_path / "cpu.kubeconfig")],
        ["aws", "rds", "describe-db-clusters"],
        ["aws", "rds", "describe-db-instances"],
    ]
    pod_list = world.calls[0]
    assert pod_list[pod_list.index("-l") + 1] == (
        "app in (gpu-fault-api-ha,gpu-fault-control-worker)"
    )
    assert "--field-selector=status.phase=Running" in pod_list
    assert world.calls[1][3:7] == ["get", "node", "cpu-1", "cpu-2"], (
        "only the nodes the Pods run on"
    )


def test_a_writer_in_another_zone_is_a_finding_that_names_the_failback(
    tmp_path: Path,
) -> None:
    (line,) = _findings(World(), tmp_path)

    assert line.startswith(
        "Aurora writer aurora-a-reader runs in us-west-2b but the CPU control "
        "plane (gpu-fault-api-ha, gpu-fault-control-worker) runs in us-west-2a: "
        "every database round trip crosses zones; fail it back to "
        "aurora-a-writer (us-west-2a, promotion tier 0) with gpu-fault-admin "
        f"aurora-failback --state-dir {tmp_path / 'state'} --reference <change>"
    ), line
    assert line.endswith("then the same with --confirm AURORA_FAILBACK"), line


def test_the_preferred_member_is_available_in_zone_and_lowest_tier_first(
    tmp_path: Path,
) -> None:
    world = World(
        writer="db-c",
        zones={"db-a": "us-west-2a", "db-b": "us-west-2a", "db-c": "us-west-2c"},
        tiers={"db-a": 2, "db-b": 0, "db-c": 0},
        statuses={"db-a": "available", "db-b": "rebooting", "db-c": "available"},
    )
    found = placement.read_writer_placement(
        site(tmp_path),  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
    )

    assert found.placed is False
    preferred = found.preferred()
    assert preferred is not None and preferred.instance_id == "db-a", (
        "a rebooting tier-0 member is no failback target"
    )
    world.statuses["db-b"] = "available"
    world.tiers["db-a"] = None
    again = placement.read_writer_placement(
        site(tmp_path),  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
    )
    best = again.preferred()
    assert best is not None and best.instance_id == "db-b", "tier 0 beats no tier"
    assert again.evidence()["members"][0] == {
        "instance_id": "db-a",
        "availability_zone": "us-west-2a",
        "status": "available",
        "promotion_tier": None,
        "writer": False,
    }


def test_a_control_plane_spanning_the_writer_zone_is_placed(tmp_path: Path) -> None:
    world = World(nodes={"cpu-1": "us-west-2a", "cpu-2": "us-west-2b"})

    assert _findings(world, tmp_path) == []


def test_no_available_member_in_the_control_plane_zone_says_so(tmp_path: Path) -> None:
    world = World(
        statuses={"aurora-a-writer": "rebooting", "aurora-a-reader": "available"}
    )

    (line,) = _findings(world, tmp_path)

    assert "no available Aurora member is in us-west-2a to fail back to" in line, line
    assert "once one is, run gpu-fault-admin aurora-failback" in line, line


@pytest.mark.parametrize(
    "defect",
    ["kubectl", "describe-db-clusters", "no-pods", "no-zone", "no-writer", "two"],
)
def test_an_unreadable_placement_is_a_finding_line_not_a_crash(
    tmp_path: Path, defect: str
) -> None:
    world = World()
    if defect in {"kubectl", "describe-db-clusters"}:
        world.failure = defect
    elif defect == "no-pods":
        world.pods = []
    elif defect == "no-zone":
        world.nodes["cpu-2"] = ""
    elif defect == "no-writer":
        world.writer = "nobody"
    else:
        world.zones["aurora-a-third"] = "us-west-2c"
        world.tiers["aurora-a-third"] = 2
        world.statuses["aurora-a-third"] = "available"
        original = world.run

        def two_writers(arguments, **kwargs):
            raw = original(arguments, **kwargs)
            if "describe-db-clusters" not in arguments:
                return raw
            document = json.loads(raw)
            for member in document["DBClusters"][0]["DBClusterMembers"]:
                member["IsClusterWriter"] = member["DBInstanceIdentifier"] != (
                    "aurora-a-writer"
                )
            return json.dumps(document)

        world.run = two_writers  # type: ignore[method-assign]

    (line,) = _findings(world, tmp_path)

    assert line.startswith("Aurora writer placement could not be read: "), line
    expected = {
        "kubectl": "unit outage",
        "describe-db-clusters": "unit outage",
        "no-pods": "no Running gpu-fault-api-ha or gpu-fault-control-worker Pod",
        "no-zone": "control-plane node(s) cpu-2 carry no zone label",
        "no-writer": f"Aurora cluster {CLUSTER_ID} reports 0 writers",
        "two": f"Aurora cluster {CLUSTER_ID} reports 2 writers",
    }[defect]
    assert expected in line, line


def test_the_status_reader_runs_behind_a_read_only_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World()

    def mutating_reader(_site, runner):
        runner.run(["aws", "rds", "failover-db-cluster"], mutate=True)
        raise AssertionError("the read-only runner must refuse first")

    monkeypatch.setattr(placement, "read_writer_placement", mutating_reader)

    (line,) = _findings(world, tmp_path)

    assert line.startswith("Aurora writer placement could not be read: "), line
    assert world.calls == [], "a mutation never reaches a command"


def test_the_cluster_lookup_must_answer_for_the_named_cluster() -> None:
    world = World()
    world.answered_cluster = "another-cluster"

    with pytest.raises(
        BootstrapError, match=f"cannot describe Aurora cluster {CLUSTER_ID}"
    ):
        placement.aurora_members(
            world,  # type: ignore[arg-type]
            aws_region="us-west-2",
            cluster_id=CLUSTER_ID,
        )


def _status_report() -> str:
    document = {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [{"name": "cpu_workloads", "status": "PASS"}],
        },
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def test_status_prints_the_finding_and_keeps_the_engine_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    world = World()
    monkeypatch.setattr(placement, "CommandRunner", lambda: world)
    monkeypatch.setattr(status_report, "status_findings", lambda _site: [])
    monkeypatch.setattr(
        admin_cli,
        "run_driver",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, stdout=_status_report()
        ),
    )
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )

    assert admin_cli.run(arguments) == 0, "a finding is not a failing check"

    captured = capsys.readouterr()
    assert json.loads(captured.out)["healthy"] is True
    findings = [
        line
        for line in captured.err.splitlines()
        if line.startswith("gpu-fault-admin status: finding: ")
    ]
    assert any(
        "Aurora writer aurora-a-reader runs in us-west-2b" in line
        and f"aurora-failback --state-dir {tmp_path} " in line
        for line in findings
    ), captured.err
    assert not any("failover-db-cluster" in " ".join(call) for call in world.calls), (
        "status never fails anything over"
    )
