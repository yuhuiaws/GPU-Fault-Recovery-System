"""WORKLOAD/ISO/E2E command-line admission edges and executor Pod identity.

Each CLI refusal happens before authorization: ISO-001 without a distinct
secondary cluster, the admin-status case without a state directory, E2E-001
with a mutable host probe image, and a deployment whose release identity is
blank. The executor inventory reader refuses Pods outside the namespace and
Pods without an ``executor`` container.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_workload_acceptance as runner
from scripts.e2e.regional.regional_commands import RegionalFixtureError


def install_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    *,
    release_id: str = "unit-release",
) -> list[str]:
    events: list[str] = []
    targets = {
        name: SimpleNamespace(cluster_id=name, context="context-" + name)
        for name in ("a", "b")
    }
    for name in ("site", "cpu.kubeconfig", "gpu.kubeconfig"):
        (tmp_path / name).write_text("unit fixture\n")
    site = SimpleNamespace(
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        target=lambda name: targets[name],
        regional=lambda target: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": release_id,
                "cluster_id": target.cluster_id,
            }
        ),
    )
    monkeypatch.setattr(runner, "WorkloadSite", lambda path: site)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner.os, "umask", lambda mode: None)
    monkeypatch.setattr(
        runner,
        "predecessor_path",
        lambda *a: events.append("predecessor") or (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--run-dir",
            str(tmp_path),
            "--site",
            str(tmp_path / "site"),
            "--plan",
            *arguments,
        ],
    )
    return events


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        (
            ["--case", "GF-REGIONAL-ISO-001", "--cluster-id", "a"],
            "secondary-cluster-id",
        ),
        (
            [
                "--case",
                "GF-REGIONAL-ISO-001",
                "--cluster-id",
                "a",
                "--secondary-cluster-id",
                "a",
            ],
            "must differ",
        ),
        (["--case", "GF-REGIONAL-WORKLOAD-002", "--cluster-id", "a"], "--state-dir"),
        (
            [
                "--case",
                "GF-REGIONAL-E2E-001",
                "--cluster-id",
                "a",
                "--host-probe-image",
                "registry.invalid/probe:latest",
            ],
            "immutable --host-probe-image",
        ),
    ],
)
def test_cli_refuses_incomplete_case_arguments_before_reading_the_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], fragment: str
) -> None:
    events = install_cli(monkeypatch, tmp_path, arguments)
    with pytest.raises(runner.WorkloadAcceptanceError, match=fragment):
        runner.main()
    assert events == [], "argument refusals precede any predecessor lookup"


def test_cli_refuses_a_deployment_without_a_release_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = install_cli(
        monkeypatch,
        tmp_path,
        ["--case", "GF-REGIONAL-WORKLOAD-001", "--cluster-id", "a"],
        release_id="   ",
    )
    with pytest.raises(runner.WorkloadAcceptanceError, match="release identity"):
        runner.main()
    assert events == [], "a blank release stops before the predecessor is resolved"


def executor_pod(
    *, namespace: str = "gpu-fault-system", container: str = "executor"
) -> dict[str, Any]:
    return {
        "metadata": {"name": "executor-0", "uid": "uid-0", "namespace": namespace},
        "spec": {"nodeName": "node-a", "containers": [{"name": container}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": container,
                    "ready": True,
                    "containerID": "containerd://" + "c" * 64,
                    "imageID": "registry.invalid/executor@sha256:" + "d" * 64,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": "2026-01-01T00:00:00Z"}},
                }
            ],
        },
    }


def regional_with(pods: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(
        settings=SimpleNamespace(namespace="gpu-fault-system"),
        kubectl=lambda *a, **k: json.dumps({"items": pods}),
    )


def test_executor_pods_must_live_in_the_release_namespace() -> None:
    regional = regional_with([executor_pod(namespace="default")])
    with pytest.raises(RegionalFixtureError, match="namespace or node is missing"):
        runner.e2e_executor_pods(regional)


def test_executor_pods_must_carry_an_executor_container() -> None:
    regional = regional_with([executor_pod(container="sidecar")])
    with pytest.raises(RegionalFixtureError, match="executor container is missing"):
        runner.e2e_executor_pods(regional)


def test_executor_pods_report_sorted_container_identities() -> None:
    pods = runner.e2e_executor_pods(regional_with([executor_pod()]))
    assert [pod["name"] for pod in pods] == ["executor-0"]
    assert pods[0]["containers"][0]["name"] == "executor"
    assert pods[0]["containers"][0]["restart_count"] == 0
