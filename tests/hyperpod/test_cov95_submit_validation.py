from __future__ import annotations

import copy
from typing import Any
from uuid import UUID

import pytest
import yaml

from gpu_fault import training_submit_cli as submit
from tests.hyperpod._cov95_submit_support import manifest_file, render, workload
from tests.hyperpod.test_training_submit_cli import container_template


@pytest.mark.parametrize("kind", ["Job", "PyTorchJob", "JobSet"])
@pytest.mark.parametrize("count", [True, False, 1.25, float("inf"), float("nan")])
def test_rank_count_rejects_lossy_numeric_coercion(tmp_path, kind, count) -> None:
    path = manifest_file(tmp_path, workload(kind, count))
    with pytest.raises(submit.TrainingSubmitError, match="must be an integer"):
        render(path)


@pytest.mark.parametrize("kind", ["Job", "PyTorchJob", "JobSet"])
@pytest.mark.parametrize(
    ("count", "diagnostic"),
    [
        (None, "must be an integer"),
        ("many", "must be an integer"),
        (0, "must be greater than zero"),
        (-1, "must be greater than zero"),
    ],
)
def test_rank_count_rejects_missing_invalid_or_nonpositive_values(
    tmp_path, kind, count, diagnostic
) -> None:
    path = manifest_file(tmp_path, workload(kind, count))
    with pytest.raises(submit.TrainingSubmitError, match=diagnostic):
        render(path)


@pytest.mark.parametrize(
    ("document", "diagnostic"),
    [
        ({"kind": "Deployment"}, "supported kinds"),
        ({"kind": "Job"}, "Job requires spec.template"),
        ({"kind": "PyTorchJob"}, "requires spec.pytorchReplicaSpecs"),
        (
            {"kind": "PyTorchJob", "spec": {"pytorchReplicaSpecs": {"Master": None}}},
            "invalid PyTorch replica spec",
        ),
        (
            {"kind": "PyTorchJob", "spec": {"pytorchReplicaSpecs": {"Master": {}}}},
            "requires template",
        ),
        ({"kind": "JobSet"}, "requires spec.replicatedJobs"),
        (
            {"kind": "JobSet", "spec": {"replicatedJobs": [None]}},
            "invalid replicatedJob",
        ),
        (
            {"kind": "JobSet", "spec": {"replicatedJobs": [{}]}},
            "requires template.spec.template",
        ),
        ({"kind": "Job", "spec": {"template": {}}}, "Pod template requires spec"),
        (
            {"kind": "Job", "spec": {"template": {"spec": {"containers": []}}}},
            "require --training-container",
        ),
    ],
)
def test_invalid_workload_shape_fails_before_submission(
    tmp_path, document, diagnostic
) -> None:
    with pytest.raises(submit.TrainingSubmitError, match=diagnostic):
        render(manifest_file(tmp_path, document))


def test_jobset_requires_indexed_completion_and_retains_global_rank_stride(
    tmp_path,
) -> None:
    document = workload("JobSet", 2)
    job = document["spec"]["replicatedJobs"][0]
    job.pop("name")
    job_spec = job["template"]["spec"]
    job_spec["completions"] = 3
    with pytest.raises(submit.TrainingSubmitError, match="requires completionMode"):
        render(manifest_file(tmp_path, document))
    job_spec["completionMode"] = "Indexed"
    rendered = render(
        manifest_file(tmp_path, document),
        expected_critical_ranks=6,
        training_container="trainer",
    )
    actual = yaml.safe_load(rendered.manifest)["spec"]["replicatedJobs"][0]
    metadata = actual["template"]["spec"]["template"]["metadata"]
    assert metadata["labels"][submit.ROLE_LABEL] == "worker"
    assert metadata["annotations"][submit.RANK_JOB_STRIDE_ANNOTATION] == "3"
    assert metadata["annotations"][submit.EXPECTED_RANKS_ANNOTATION] == "6"
    assert submit.RESTART_BUDGET_ANNOTATION not in metadata["annotations"]


@pytest.mark.parametrize(
    ("options", "diagnostic"),
    [
        ({"attempt_number": 0}, "attempt-number must be positive"),
        ({"restart_budget": -1}, "restart-budget cannot be negative"),
        ({"expected_critical_ranks": 0}, "expected-critical-ranks must be positive"),
        ({"expected_critical_ranks": 2}, "must equal the rank count"),
        ({"job_id": "bad/job"}, "job ID must be a valid Kubernetes label"),
        ({"job_id": "x" * 64}, "job ID must be a valid Kubernetes label"),
        ({"attempt_id": "_invalid"}, "attempt ID must be a valid Kubernetes label"),
        ({"namespace": "   "}, "namespace cannot be blank"),
        ({"training_container": "absent"}, "not present in every Pod template"),
    ],
)
def test_render_rejects_inconsistent_identity_and_options(
    tmp_path, options: dict[str, Any], diagnostic: str
) -> None:
    with pytest.raises(submit.TrainingSubmitError, match=diagnostic):
        render(manifest_file(tmp_path), **options)


def test_container_choice_is_explicit_for_every_replica(tmp_path) -> None:
    document = workload("PyTorchJob")
    master = document["spec"]["pytorchReplicaSpecs"]["Master"]["template"]
    master["spec"]["containers"].extend(
        [None, {"image": "no-name"}, {"name": "sidecar", "image": "local"}]
    )
    with pytest.raises(submit.TrainingSubmitError, match="multiple containers"):
        render(manifest_file(tmp_path, document))
    document["spec"]["pytorchReplicaSpecs"]["Worker"] = {
        "replicas": "2",
        "template": container_template(),
    }
    rendered = render(
        manifest_file(tmp_path, document),
        training_container="trainer",
        restart_budget=0,
    )
    actual = yaml.safe_load(rendered.manifest)["spec"]["pytorchReplicaSpecs"]
    for role, offset in (("Master", "0"), ("Worker", "1")):
        annotations = actual[role]["template"]["metadata"]["annotations"]
        assert annotations[submit.TRAINING_CONTAINER_ANNOTATION] == "trainer"
        assert annotations[submit.RANK_OFFSET_ANNOTATION] == offset
        assert annotations[submit.EXPECTED_RANKS_ANNOTATION] == "3"
        assert annotations[submit.RESTART_BUDGET_ANNOTATION] == "0"


@pytest.mark.parametrize("text", ["", "---\n", "[]", "- one", "{}\n---\n{}"])
def test_manifest_must_contain_exactly_one_mapping(tmp_path, text) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(submit.TrainingSubmitError, match="exactly one"):
        render(path)


@pytest.mark.parametrize("text", ["{broken:", "!!python/object:untrusted {}"])
def test_manifest_uses_safe_yaml_and_reports_parse_errors(tmp_path, text) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(submit.TrainingSubmitError, match="cannot read manifest"):
        render(path)


def test_missing_manifest_reports_a_submission_error(tmp_path) -> None:
    with pytest.raises(submit.TrainingSubmitError, match="cannot read manifest"):
        render(tmp_path / "absent.yaml")


def test_generated_ids_and_metadata_preserve_workload_semantics(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(submit.uuid, "uuid4", lambda: UUID(int=1))
    document = workload()
    document["spec"]["backoffLimit"] = 0
    document["metadata"]["labels"] = {"customer": "retained"}
    original = copy.deepcopy(document)
    path = manifest_file(tmp_path, document)
    rendered = render(path, job_id=None, attempt_id=None, attempt_number=3)
    assert rendered.job_id == "train-00000000000000000000000000000001"
    assert rendered.attempt_id == f"{rendered.job_id}-a003"
    actual = yaml.safe_load(rendered.manifest)
    assert actual["metadata"]["labels"]["customer"] == "retained"
    assert actual["spec"]["backoffLimit"] == 0
    assert actual["spec"]["template"]["spec"] == original["spec"]["template"]["spec"]
    assert yaml.safe_load(path.read_text("utf-8")) == original
