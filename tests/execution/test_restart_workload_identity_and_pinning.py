"""Restart guard cases decided by what the workload objects themselves declare.

A workload labelled for another job, two workloads of different attempts, a
JobSet whose Pod templates pin a node the plan avoids (including malformed
replicated entries the guard skips), a ``nodeSelector`` hostname rebound to a
replacement node, a PyTorchJob role that is not an object, and attempt
observations without GPU UUIDs that cannot stand in for the source count.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from gpu_fault.adapters.common import (
    ANNOTATION_TARGET_GPU_COUNT,
    LABEL_ATTEMPT_ID,
    LABEL_JOB_ID,
)
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowStepStatus
from tests._builders import attempt_observation, container_observation
from tests.execution._cov95_runtime_restart import (
    RestartHarness,
    changed_parameters,
    source_workload,
)
from tests.execution.test_restart_safety import NOW

HOSTNAME = "kubernetes.io/hostname"


def _with_workloads(context: WorkflowStepContext, *ids: str) -> WorkflowStepContext:
    step = context.step.model_copy(update={"workload_ids": list(ids)})
    return replace(
        context,
        step=step,
        workflow=context.workflow.model_copy(update={"official_steps": [step]}),
    )


def _templates(workload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "job":
        return [workload["spec"]["template"]]
    if kind == "pytorchjob":
        return [
            role["template"]
            for role in workload["spec"]["pytorchReplicaSpecs"].values()
            if isinstance(role, dict)
        ]
    return [
        row["template"]["spec"]["template"]
        for row in workload["spec"]["replicatedJobs"]
    ]


def _pin_hostname(workload: dict[str, Any], kind: str, hostname: str) -> None:
    for template in _templates(workload, kind):
        template["spec"]["nodeSelector"] = {HOSTNAME: hostname}


def test_a_workload_labelled_for_another_job_is_refused() -> None:
    harness = RestartHarness("job")
    harness.api.source["metadata"]["labels"][LABEL_JOB_ID] = "train-2"

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error == (
        "restart workload job identity does not match train-1: train-2"
    )
    assert harness.api.created == {}, "a mislabelled workload was still restarted"


def test_workloads_of_two_different_attempts_are_not_restarted_together() -> None:
    harness = RestartHarness("job")
    sibling = deepcopy(harness.api.source)
    sibling["metadata"]["name"] = "training-job-two"
    sibling["metadata"]["labels"][LABEL_ATTEMPT_ID] = "attempt-b"
    harness.api.created["training-job-two"] = sibling
    harness.context = _with_workloads(
        harness.context, "training/job/training-job", "training/job/training-job-two"
    )

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error == (
        "restart workloads have inconsistent attempt IDs: attempt-a, attempt-b"
    )
    assert set(harness.api.created) == {"training-job-two"}, (
        "a restart was submitted for workloads of two attempts"
    )


def test_a_jobset_pinned_by_hostname_to_an_avoided_node_is_refused() -> None:
    """The guard reads every replicated job's Pod template: entries that are not
    objects or carry no template are skipped, the real one is pinned by its
    ``nodeSelector`` hostname to the node the plan avoids."""

    harness = RestartHarness("jobset")
    _pin_hostname(harness.api.source, "jobset", "node-bad")
    harness.api.source["spec"]["replicatedJobs"] = [
        "not-an-object",
        {"name": "no-template"},
        {"name": "no-pod-template", "template": {"spec": {}}},
        {
            "name": "zone-only",
            "template": {
                "spec": {
                    "template": {
                        "spec": {"nodeSelector": {"topology.kubernetes.io/zone": "a"}}
                    }
                }
            },
        },
        *harness.api.source["spec"]["replicatedJobs"],
    ]
    harness.context = changed_parameters(harness.context, avoid_node_ids=["node-bad"])

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details.get("reason") == "RESTART_TARGET_AVOIDED", outcome.details
    assert outcome.details.get("avoided_node_ids") == ["node-bad"], outcome.details
    assert harness.api.created == {}


def test_a_hostname_selector_follows_the_replacement_node_in_every_shape() -> None:
    for kind in ("job", "pytorchjob", "jobset"):
        harness = RestartHarness(kind)
        _pin_hostname(harness.api.source, kind, "node-old")
        harness.bind_replacement({"node-old": "node-new"})

        outcome = harness.execute()

        assert outcome.status is WorkflowStepStatus.SUCCEEDED, (kind, outcome)
        (created,) = harness.api.created.values()
        selectors = [
            template["spec"]["nodeSelector"] for template in _templates(created, kind)
        ]
        assert selectors and all(
            selector[HOSTNAME] == "node-new" for selector in selectors
        ), (kind, selectors)


def test_a_selector_without_a_hostname_is_carried_over_untouched() -> None:
    zone = {"topology.kubernetes.io/zone": "us-west-2a"}
    for kind in ("job", "pytorchjob", "jobset"):
        harness = RestartHarness(kind)
        _pin_hostname(harness.api.source, kind, "node-old")
        for template in _templates(harness.api.source, kind):
            template["spec"]["nodeSelector"] = dict(zone)
        harness.bind_replacement({"node-old": "node-new"})

        outcome = harness.execute()

        assert outcome.status is WorkflowStepStatus.SUCCEEDED, (kind, outcome)
        (created,) = harness.api.created.values()
        selectors = [
            template["spec"]["nodeSelector"] for template in _templates(created, kind)
        ]
        assert selectors and all(selector == zone for selector in selectors), (
            kind,
            selectors,
        )


def test_a_pytorchjob_role_that_is_not_an_object_is_left_as_it_was() -> None:
    harness = RestartHarness("pytorchjob")
    source = harness.api.source
    source["metadata"].setdefault("annotations", {})[ANNOTATION_TARGET_GPU_COUNT] = "6"
    source["spec"]["pytorchReplicaSpecs"]["Broken"] = "not-an-object"

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    (created,) = harness.api.created.values()
    replicas = created["spec"]["pytorchReplicaSpecs"]
    assert replicas["Broken"] == "not-an-object"
    for role in ("Master", "Worker"):
        labels = replicas[role]["template"]["metadata"]["labels"]
        assert labels[LABEL_ATTEMPT_ID] != "attempt-a", (role, labels)


def test_observations_without_gpu_uuids_do_not_stand_in_for_the_source_count() -> None:
    """The plan did not know the count (0); the only observations of the job
    name no GPU, so the count stays unknown and the restart is not approved."""

    harness = RestartHarness("job", source_gpu_count=0)
    harness.store.save_attempt_observation(
        attempt_observation(
            "train-1",
            "attempt-a",
            NOW,
            containers=[
                container_observation("pod-a", "pod-a", 0, "node-a", gpu_uuids=[])
            ],
        )
    )

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details.get("reason") == "GPU_COUNT_CHANGED", outcome.details
    assert outcome.details.get("source_gpu_count") == 0, outcome.details
    assert outcome.details.get("target_gpu_count") == 4, outcome.details
    assert outcome.details.get("required_approval_annotation") is None, (
        "an unknown source count cannot name the annotation that would approve it"
    )
    assert harness.api.created == {}


def test_the_source_shapes_used_above_restart_cleanly_without_a_selector() -> None:
    """Pins the fixture: the three shapes restart as-is, so the refusals above
    are caused by the labels, pins and roles each test adds."""

    for kind in ("job", "pytorchjob", "jobset"):
        harness = RestartHarness(kind)
        assert harness.api.source == source_workload(kind, terminal=True)
        assert harness.execute().status is WorkflowStepStatus.SUCCEEDED, kind
