from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault import training_submit_cli
from gpu_fault.training_submit_cli import TrainingSubmitError, inject_metadata, run


def container_template() -> dict:
    return {
        "spec": {
            "restartPolicy": "Never",
            "containers": [{"name": "trainer", "image": "train:latest"}],
        }
    }


def test_injects_pytorchjob_templates_with_global_rank_offsets() -> None:
    document = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": "train"},
        "spec": {
            "pytorchReplicaSpecs": {
                "Worker": {"replicas": 3, "template": container_template()},
                "Master": {"replicas": 1, "template": container_template()},
            }
        },
    }

    result = inject_metadata(
        document,
        job_id="logical-job",
        attempt_id="logical-job-a001",
        runtime_profile="hyperpod-v1",
        restart_budget=2,
    )

    replicas = result["spec"]["pytorchReplicaSpecs"]
    master = replicas["Master"]["template"]["metadata"]
    worker = replicas["Worker"]["template"]["metadata"]
    assert master["annotations"]["gpu-fault.io/rank-offset"] == "0"
    assert worker["annotations"]["gpu-fault.io/rank-offset"] == "1"
    assert worker["annotations"]["gpu-fault.io/expected-critical-ranks"] == "4"
    assert worker["labels"]["gpu-fault.io/job-id"] == "logical-job"
    assert worker["annotations"]["gpu-fault.io/restart-budget"] == "2"


def test_injects_jobset_offsets_and_strides() -> None:
    document = {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {"name": "train"},
        "spec": {
            "replicatedJobs": [
                {
                    "name": "launcher",
                    "replicas": 1,
                    "template": {"spec": {"template": container_template()}},
                },
                {
                    "name": "workers",
                    "replicas": 2,
                    "template": {
                        "spec": {
                            "completionMode": "Indexed",
                            "completions": 4,
                            "template": container_template(),
                        }
                    },
                },
            ]
        },
    }

    result = inject_metadata(
        document,
        job_id="logical-job",
        attempt_id="logical-job-a001",
        runtime_profile="hyperpod-v1",
    )
    replicated = result["spec"]["replicatedJobs"]
    launcher = replicated[0]["template"]["spec"]["template"]["metadata"]
    workers = replicated[1]["template"]["spec"]["template"]["metadata"]
    assert launcher["annotations"]["gpu-fault.io/rank-offset"] == "0"
    assert workers["annotations"]["gpu-fault.io/rank-offset"] == "1"
    assert workers["annotations"]["gpu-fault.io/rank-job-stride"] == "4"
    assert workers["annotations"]["gpu-fault.io/expected-critical-ranks"] == "9"


def test_rejects_non_indexed_multi_rank_job() -> None:
    document = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "train"},
        "spec": {"completions": 2, "template": container_template()},
    }
    with pytest.raises(TrainingSubmitError, match="Indexed"):
        inject_metadata(
            document,
            job_id="logical-job",
            attempt_id="logical-job-a001",
            runtime_profile="hyperpod-v1",
        )


def test_run_renders_and_submits_once(tmp_path, capsys) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "train"},
                "spec": {"template": container_template()},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    args = argparse.Namespace(
        manifest=manifest,
        job_id="logical-job",
        attempt_id=None,
        attempt_number=2,
        runtime_profile_version="hyperpod-v1",
        expected_critical_ranks=None,
        training_container=None,
        restart_budget=3,
        namespace="training",
        context="hp",
        kubeconfig=None,
        dry_run=False,
    )
    assert run(args, runner=runner) == 0
    assert calls[0][0] == [
        "kubectl",
        "--context",
        "hp",
        "--namespace",
        "training",
        "apply",
        "-f",
        "-",
    ]
    submitted = yaml.safe_load(calls[0][1]["input"])
    assert submitted["metadata"]["namespace"] == "training"
    labels = submitted["spec"]["template"]["metadata"]["labels"]
    assert labels["gpu-fault.io/attempt-id"] == "logical-job-a002"
    assert calls[0][1]["check"] is False
    assert "job-id: logical-job" in capsys.readouterr().err


def test_dry_run_overrides_manifest_namespace(tmp_path, capsys) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "train", "namespace": "source"},
                "spec": {"template": container_template()},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest,
        job_id="logical-job",
        attempt_id=None,
        attempt_number=1,
        runtime_profile_version="hyperpod-v1",
        expected_critical_ranks=None,
        training_container=None,
        restart_budget=1,
        namespace="target",
        context=None,
        kubeconfig=None,
        dry_run=True,
    )

    assert run(args) == 0
    assert yaml.safe_load(capsys.readouterr().out)["metadata"]["namespace"] == "target"


def test_site_supplies_runtime_profile_and_rejects_mismatch(
    tmp_path, monkeypatch
) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("site", encoding="utf-8")
    monkeypatch.setattr(
        training_submit_cli,
        "load_site",
        lambda _path: SimpleNamespace(
            release_config={"runtime_profile": {"version": "profile-v2"}}
        ),
    )
    args = argparse.Namespace(site=site, runtime_profile_version=None)

    assert training_submit_cli.resolve_runtime_profile_version(args) == "profile-v2"
    args.runtime_profile_version = "profile-v1"
    with pytest.raises(TrainingSubmitError, match="does not match"):
        training_submit_cli.resolve_runtime_profile_version(args)
