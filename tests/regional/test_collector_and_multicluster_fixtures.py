from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from scripts.e2e.regional import run_collect016_training_recovery as collect016
from scripts.e2e.regional import run_collect017_efa_plugin as collect017
from scripts.e2e.regional import run_collector_acceptance as collect
from scripts.e2e.regional import run_collector_destructive as collect_destructive
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e002
from scripts.e2e.regional import run_iso006_cluster_offline as iso006
from scripts.e2e.regional.multi_cluster_fixture import (
    ClusterTarget,
    MultiClusterSettings,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.probes import cluster_network_probe, collector_node_probe

ROOT = Path(__file__).resolve().parents[2]


def _kubeconfig(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("apiVersion: v1\n", encoding="utf-8")
    return path


def test_collector_case_registries_cover_the_requested_order() -> None:
    expected_low_risk = {
        "GF-REGIONAL-COLLECT-001",
        "GF-REGIONAL-COLLECT-002",
        "GF-REGIONAL-COLLECT-003",
        "GF-REGIONAL-COLLECT-005",
        "GF-REGIONAL-COLLECT-009",
        "GF-REGIONAL-COLLECT-010",
        "GF-REGIONAL-COLLECT-011",
        "GF-REGIONAL-COLLECT-012",
    }
    expected_destructive = {
        "GF-REGIONAL-COLLECT-004",
        "GF-REGIONAL-COLLECT-008",
        "GF-REGIONAL-COLLECT-013",
        "GF-REGIONAL-COLLECT-014",
        "GF-REGIONAL-COLLECT-015",
    }

    assert set(collect.CASE_IDS) == expected_low_risk, collect.CASE_IDS
    assert set(collect_destructive.CASE_IDS) == expected_destructive, (
        collect_destructive.CASE_IDS
    )
    assert collect016.CASE_ID == "GF-REGIONAL-COLLECT-016", collect016.CASE_ID
    assert collect017.CASE_ID == "GF-REGIONAL-COLLECT-017", collect017.CASE_ID


@pytest.mark.parametrize("case_id", collect.CASE_IDS)  # type: ignore[untyped-decorator]
def test_collector_runner_is_plan_only_with_case_specific_confirmation(
    tmp_path: Path, case_id: str
) -> None:
    nodes = ["node-a", "node-b"] if case_id.endswith("-011") else ["node-a"]
    arguments = collect.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--case",
            case_id,
            *[value for node in nodes for value in ("--node", node)],
        ]
    )

    assert arguments.execute is False, case_id
    assert collect.CONFIRMATIONS[case_id].endswith("_EXECUTE"), case_id


def test_collector_host_probe_has_narrow_action_allowlists() -> None:
    assert collector_node_probe.ALLOWED_XIDS == {13, 31, 46, 48, 54, 62, 63, 78, 109}, (
        collector_node_probe.ALLOWED_XIDS
    )
    assert 99999 in collector_node_probe.ALLOWED_SXIDS, (
        collector_node_probe.ALLOWED_SXIDS
    )
    assert collector_node_probe.ALLOWED_SERVICES == {
        "gpu-fault-dcgm-collector.service",
        "gpu-fault-fabric-manager-collector.service",
        "gpu-fault-host-collector.service",
        "gpu-fault-kernel-collector.service",
    }, collector_node_probe.ALLOWED_SERVICES


def test_cluster_network_probe_chain_and_restore_unit_are_deterministic() -> None:
    first = cluster_network_probe.chain_name("iso006-run-a")
    second = cluster_network_probe.chain_name("iso006-run-a")

    assert first == second, (first, second)
    assert len(first) <= 28, first
    assert cluster_network_probe.restore_unit("iso006-run-a").startswith(
        "gpu-fault-network-restore-"
    ), cluster_network_probe.restore_unit("iso006-run-a")


def test_cluster_network_probe_arms_restore_before_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_network_probe, "run", fake_run)

    cluster_network_probe.block(
        argparse.Namespace(
            run_id="iso006-run-a",
            control_plane_cidr=["10.0.0.0/24"],
            restore_seconds=180,
        )
    )

    timer_index = next(
        index for index, command in enumerate(calls) if command[0] == "systemd-run"
    )
    block_index = next(
        index
        for index, command in enumerate(calls)
        if command[:4] == ["iptables", "-I", "OUTPUT", "1"]
    )
    assert timer_index < block_index


def test_cluster_network_probe_cleans_partial_block_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        calls.append(command)
        if command[:2] == ["iptables", "-A"]:
            raise cluster_network_probe.ProbeError("synthetic rule failure")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_network_probe, "run", fake_run)

    with pytest.raises(cluster_network_probe.ProbeError, match="synthetic"):
        cluster_network_probe.block(
            argparse.Namespace(
                run_id="iso006-run-a",
                control_plane_cidr=["10.0.0.0/24"],
                restore_seconds=180,
            )
        )

    assert ["iptables", "-F", cluster_network_probe.chain_name("iso006-run-a")] in calls
    assert [
        "systemctl",
        "stop",
        cluster_network_probe.restore_unit("iso006-run-a") + ".timer",
    ] in calls


def test_multi_cluster_fixture_rejects_same_physical_target(tmp_path: Path) -> None:
    cpu = _kubeconfig(tmp_path, "cpu.kubeconfig")
    gpu = _kubeconfig(tmp_path, "gpu.kubeconfig")
    target = ClusterTarget(
        cluster_id="cluster-a", gpu_kubeconfig=gpu, gpu_context="context-a"
    )

    with pytest.raises(ValueError, match="distinct cluster IDs"):
        MultiClusterSettings(
            cpu_kubeconfig=cpu,
            namespace="gpu-fault-system",
            region="us-west-2",
            cluster_a=target,
            cluster_b=target,
        )


def test_multi_cluster_registry_requires_distinct_eks_and_hyperpod_identities() -> None:
    registrations = [
        {
            "cluster_id": "cluster-a",
            "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
            "hyperpod_cluster_name": "hyperpod-a",
        },
        {
            "cluster_id": "cluster-b",
            "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-b",
            "hyperpod_cluster_name": "hyperpod-b",
        },
    ]

    assert registrations_are_distinct_physical_clusters(registrations), (
        "distinct EKS and HyperPod identities were rejected"
    )
    assert not registrations_are_distinct_physical_clusters(
        [
            registrations[0],
            {
                **registrations[1],
                "eks_cluster_arn": registrations[0]["eks_cluster_arn"],
            },
        ]
    ), "duplicate EKS identity was accepted as a second physical cluster"
    assert not registrations_are_distinct_physical_clusters(
        [
            registrations[0],
            {
                **registrations[1],
                "hyperpod_cluster_name": registrations[0]["hyperpod_cluster_name"],
            },
        ]
    ), "duplicate HyperPod identity was accepted as a second physical cluster"


def test_multi_cluster_runners_are_plan_only(tmp_path: Path) -> None:
    common = [
        "--run-dir",
        str(tmp_path),
        "--cluster-a",
        "cluster-a",
        "--gpu-a-kubeconfig",
        "/tmp/a.kubeconfig",
        "--gpu-a-context",
        "context-a",
        "--cluster-b",
        "cluster-b",
        "--gpu-b-kubeconfig",
        "/tmp/b.kubeconfig",
        "--gpu-b-context",
        "context-b",
    ]
    iso = iso006.parser().parse_args([*common, "--control-plane-cidr", "10.0.0.0/24"])
    e2e = e2e002.parser().parse_args(common)

    assert iso.execute is False, iso
    assert e2e.execute is False, e2e
    assert iso006.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-013", (
        iso006.PREDECESSOR_CASE_ID
    )
    assert e2e002.PREDECESSOR_CASE_ID == "GF-REGIONAL-ISO-006", (
        e2e002.PREDECESSOR_CASE_ID
    )


def test_collector_promoted_scripts_contain_no_site_specific_topology() -> None:
    paths = (
        ROOT / "scripts/e2e/regional/run_collector_acceptance.py",
        ROOT / "scripts/e2e/regional/run_collector_destructive.py",
        ROOT / "scripts/e2e/regional/run_collect016_training_recovery.py",
        ROOT / "scripts/e2e/regional/run_collect017_efa_plugin.py",
        ROOT / "scripts/e2e/regional/run_iso006_cluster_offline.py",
        ROOT / "scripts/e2e/regional/run_e2e002_multicluster_fault.py",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "/secure/gpu-fault-bootstrap" not in source, path
        assert "514385905925" not in source, path
        assert "gpu-fault-gpu-1-" not in source, path
