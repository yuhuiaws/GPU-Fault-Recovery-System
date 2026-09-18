from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gpu_fault.training_submit_cli import RenderedTrainingWorkload, render_workload
from tests.hyperpod.test_training_submit_cli import container_template


def workload(kind: str = "Job", count: Any = 1) -> dict[str, Any]:
    if kind == "PyTorchJob":
        spec = {
            "pytorchReplicaSpecs": {
                "Master": {"replicas": count, "template": container_template()}
            }
        }
    elif kind == "JobSet":
        spec = {
            "replicatedJobs": [
                {
                    "name": "workers",
                    "replicas": count,
                    "template": {"spec": {"template": container_template()}},
                }
            ]
        }
    else:
        spec = {
            "completions": count,
            "completionMode": "Indexed",
            "template": container_template(),
        }
    return {
        "apiVersion": {
            "Job": "batch/v1",
            "PyTorchJob": "kubeflow.org/v1",
            "JobSet": "jobset.x-k8s.io/v1alpha2",
        }[kind],
        "kind": kind,
        "metadata": {"name": "training"},
        "spec": spec,
    }


def manifest_file(tmp_path: Path, document: Any = None) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(workload() if document is None else document), encoding="utf-8"
    )
    return path


def render(path: Path, **overrides: Any) -> RenderedTrainingWorkload:
    arguments: dict[str, Any] = {
        "job_id": "training",
        "attempt_id": "training-a001",
        "attempt_number": 1,
        "runtime_profile_version": "profile-v1",
        "expected_critical_ranks": None,
        "training_container": None,
        "restart_budget": None,
    }
    arguments.update(overrides)
    return render_workload(path, **arguments)
