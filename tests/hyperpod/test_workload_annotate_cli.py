from __future__ import annotations

import argparse

import yaml

from gpu_fault.workload_annotate_cli import run


def _manifest() -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "train"},
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{"name": "trainer", "image": "train:latest"}],
                }
            }
        },
    }


def _args(manifest, output=None) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        output=output,
        job_id="customer-training",
        attempt_id=None,
        attempt_number=2,
        runtime_profile_version="hyperpod-v1",
        expected_critical_ranks=None,
        training_container=None,
        restart_budget=1,
        namespace=None,
    )


def test_writes_managed_manifest_without_submitting(tmp_path, capsys) -> None:
    source = tmp_path / "job.yaml"
    source.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
    output = tmp_path / "job.managed.yaml"

    assert run(_args(source, output)) == 0

    document = yaml.safe_load(output.read_text("utf-8"))
    assert document["metadata"]["labels"] == {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": "customer-training",
        "gpu-fault.io/attempt-id": ("customer-training-a002"),
    }
    template = document["spec"]["template"]["metadata"]
    assert template["labels"]["gpu-fault.io/critical"] == "true"
    assert template["labels"]["gpu-fault.io/role"] == "worker"
    assert template["annotations"] == {
        "gpu-fault.io/expected-critical-ranks": "1",
        "gpu-fault.io/training-container": "trainer",
        "gpu-fault.io/runtime-profile-version": "hyperpod-v1",
        "gpu-fault.io/rank-offset": "0",
        "gpu-fault.io/restart-budget": "1",
    }
    stderr = capsys.readouterr().err
    assert "job-id: customer-training" in stderr
    assert "attempt-id: customer-training-a002" in stderr
    assert f"output: {output}" in stderr


def test_prints_managed_manifest_to_stdout(tmp_path, capsys) -> None:
    source = tmp_path / "job.yaml"
    source.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")

    assert run(_args(source)) == 0

    captured = capsys.readouterr()
    document = yaml.safe_load(captured.out)
    assert (
        document["spec"]["template"]["metadata"]["labels"]["gpu-fault.io/attempt-id"]
        == "customer-training-a002"
    )
    assert "output:" not in captured.err


def test_can_override_namespace(tmp_path, capsys) -> None:
    source = tmp_path / "job.yaml"
    document = _manifest()
    document["metadata"]["namespace"] = "source"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    args = _args(source)
    args.namespace = "target"

    assert run(args) == 0

    rendered = yaml.safe_load(capsys.readouterr().out)
    assert rendered["metadata"]["namespace"] == "target"
