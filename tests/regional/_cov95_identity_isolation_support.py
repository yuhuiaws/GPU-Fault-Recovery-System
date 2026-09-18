from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_iso006_cluster_offline as offline
from tests.regional.test_multi_cluster_fixture_review import pod_document


def arguments(module: Any, tmp_path: Path) -> Any:
    for name in ("cpu", "a", "b", "site"):
        (tmp_path / name).write_text("synthetic fixture", encoding="ascii")
    values = [
        "--run-dir",
        str(tmp_path),
        "--site-file",
        str(tmp_path / "site"),
        "--cpu-kubeconfig",
        str(tmp_path / "cpu"),
        "--namespace",
        "gpu-system",
        "--region",
        "us-west-2",
        "--cluster-a",
        "a",
        "--gpu-a-kubeconfig",
        str(tmp_path / "a"),
        "--gpu-a-context",
        "context-a",
        "--cluster-b",
        "b",
        "--gpu-b-kubeconfig",
        str(tmp_path / "b"),
        "--gpu-b-context",
        "context-b",
    ]
    if module is offline:
        values.extend(
            [
                "--host-probe-image",
                "unit@sha256:" + "a" * 64,
                "--control-plane-cidr",
                "10.0.0.0/24",
            ]
        )
    return module.parser().parse_args(values)


def preflight_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any
) -> tuple[Any, list[Any], list[dict[str, Any]], list[tuple[str, ...]]]:
    settings = module.configure(arguments(module, tmp_path))
    reads = []

    class Regional:
        def __init__(self, target: Any) -> None:
            self.target = target
            self.nodes = [
                {
                    "name": f"node-{target.cluster_id}-{index}",
                    "uid": f"uid-{target.cluster_id}-{index}",
                    "ready": "True",
                    "unschedulable": False,
                    "labels": {},
                    "taints": [],
                }
                for index in range(3)
            ]
            self.workloads: list[Any] = []

        def evidence_identity(self) -> dict[str, str]:
            return {"release_id": "unit-release", "cluster_id": self.target.cluster_id}

        def gpu_nodes(self) -> list[dict[str, Any]]:
            return self.nodes

        def gpu_workloads(self) -> list[Any]:
            return self.workloads

        def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
            return [{"name": app, "uid": "uid-" + app}]

        def kubectl(self, *args: str) -> str:
            reads.append(args)
            assert args[1:3] == ("get", "pod")
            document = pod_document()
            pod = document["items"][0]
            pod["metadata"]["name"] = (
                args[4] if args[0] == "cpu" else "executor-" + self.target.cluster_id
            )
            pod["spec"]["containers"][0]["env"] = [
                {
                    "name": "GPU_FAULT_EXECUTOR_ID",
                    "value": self.target.cluster_id + "/executor",
                }
            ]
            return json.dumps(document)

    regions = [Regional(settings.multi.cluster_a), Regional(settings.multi.cluster_b)]
    monkeypatch.setattr(
        module.MultiClusterSettings,
        "regional",
        lambda self, target: regions[0] if target.cluster_id == "a" else regions[1],
    )
    registrations = [
        {
            "cluster_id": name,
            "enabled": True,
            "synthetic": False,
            "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/" + name,
            "hyperpod_cluster_name": "hp-" + name,
        }
        for name in ("a", "b")
    ]
    monkeypatch.setattr(module, "registration_snapshot", lambda *args: registrations)
    monkeypatch.setattr(
        module, "predecessor_evidence", lambda *args, **kwargs: {"valid": True}
    )
    monkeypatch.setattr(
        module, "focused_tests", lambda *args, **kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        module,
        "record_focused_tests",
        lambda details, tests: details.update({"focused_tests": tests}),
    )
    if module is offline:
        monkeypatch.setattr(
            module,
            "claim_sample",
            lambda region: {"status": 200, "latency_seconds": 0.1},
        )
    return settings, regions, registrations, reads
