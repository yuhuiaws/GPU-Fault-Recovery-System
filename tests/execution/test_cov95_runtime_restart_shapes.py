from __future__ import annotations

import json
from copy import deepcopy

import pytest
from kubernetes import client

from gpu_fault.adapters.common import (
    ANNOTATION_RESTART_BUDGET,
    ANNOTATION_RESTART_COUNT,
    ANNOTATION_TARGET_GPU_COUNT,
    LABEL_ATTEMPT_ID,
)
from gpu_fault.models import WorkflowStepStatus
from tests.execution._cov95_runtime_restart import (
    COUNTS,
    RestartHarness,
    changed_parameters,
    pod_templates,
)


@pytest.mark.parametrize("kind", list(COUNTS))
@pytest.mark.parametrize("terminal", [False, True])
def test_restart_preserves_workload_shape_and_scopes_new_attempt_metadata(
    kind: str, terminal: bool
) -> None:
    h = RestartHarness(kind, terminal=terminal)
    source = deepcopy(h.api.source)
    outcome = h.execute()
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    attempt = outcome.details["restart_attempt_id"]
    assert attempt != "attempt-a" and attempt.startswith("attempt-a-r-"), outcome
    if kind == "job" or terminal:
        assert len(h.api.created) == 1 and h.api.patches == [], (
            h.api.created,
            h.api.patches,
        )
        body = next(iter(h.api.created.values()))
        assert "status" not in body, body
        assert (
            not {
                "uid",
                "resourceVersion",
                "creationTimestamp",
                "namespace",
                "generation",
                "managedFields",
            }
            & body["metadata"].keys()
        ), body
        retry_id = f"training/{kind}/{body['metadata']['name']}"
        assert outcome.details["restarted_workload_ids"] == [retry_id], outcome
        for template in pod_templates(body, kind):
            assert json.loads(
                template["metadata"]["annotations"]["gpu-fault.io/workload-ids"]
            ) == [retry_id], template
        duplicate = h.execute()
        assert duplicate.status is WorkflowStepStatus.SUCCEEDED, duplicate
        assert len(h.api.created) == 1, h.api.created
    else:
        assert h.api.created == {} and len(h.api.patches) == 1, (
            h.api.created,
            h.api.patches,
        )
        body = h.api.patches[0]
        assert body["metadata"]["resourceVersion"] == "5", body
    assert body["metadata"]["labels"][LABEL_ATTEMPT_ID] == attempt, body
    for template in pod_templates(body, kind):
        metadata = template["metadata"]
        assert metadata["labels"][LABEL_ATTEMPT_ID] == attempt, metadata
        assert metadata["annotations"][ANNOTATION_RESTART_COUNT] == "1", metadata
        assert metadata["annotations"][ANNOTATION_RESTART_BUDGET] == "1", metadata
    assert h.api.source == source, "building a retry must not mutate the read snapshot"
    assert (
        outcome.details["notification_context"]["target_gpu_count"] == COUNTS[kind]
    ), outcome


@pytest.mark.parametrize("kind", list(COUNTS))
@pytest.mark.parametrize("pin", ["nodeName", "nodeSelector"])
def test_restart_rebinds_old_node_and_applies_avoidance_to_every_affinity_term(
    kind: str, pin: str
) -> None:
    h = RestartHarness(kind)
    expression = {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": ["node-old"],
    }
    for template in pod_templates(h.api.source, kind):
        spec = template["spec"]
        spec[pin] = (
            "node-old" if pin == "nodeName" else {"kubernetes.io/hostname": "node-old"}
        )
        spec["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [deepcopy(expression)]},
                        {
                            "matchExpressions": [
                                {"key": "disk", "operator": "In", "values": ["ssd"]}
                            ]
                        },
                    ]
                }
            }
        }
    h.context = changed_parameters(h.context, avoid_node_ids=["node-old"])
    h.bind_replacement({"node-old": "node-spare", "": "ignored", "ignored": ""})
    outcome = h.execute()
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    body = next(iter(h.api.created.values()))
    for template in pod_templates(body, kind):
        spec = template["spec"]
        assert spec[pin] == (
            "node-spare"
            if pin == "nodeName"
            else {"kubernetes.io/hostname": "node-spare"}
        ), spec
        terms = spec["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"]
        assert all(term["matchExpressions"].count(expression) == 1 for term in terms), (
            terms
        )
        assert len(terms) == 2, terms


@pytest.mark.parametrize("kind", list(COUNTS))
@pytest.mark.parametrize("binding", ["absent", "failed", "malformed"])
def test_pinned_avoided_node_needs_a_successful_observed_replacement_binding(
    kind: str, binding: str
) -> None:
    h = RestartHarness(kind)
    for template in pod_templates(h.api.source, kind):
        template["spec"]["nodeSelector"] = {"kubernetes.io/hostname": "node-old"}
    h.context = changed_parameters(h.context, avoid_node_ids=["node-old"])
    if binding != "absent":
        h.bind_replacement(
            ["not-a-mapping"] if binding == "malformed" else {"node-old": "node-spare"},
            succeeded=binding != "failed",
        )
    outcome = h.execute()
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["reason"] == "RESTART_TARGET_AVOIDED", outcome
    assert outcome.details["restart_submitted"] is False, outcome
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)


@pytest.mark.parametrize(
    ("kind", "defect"),
    [
        ("job", "override-text"),
        ("job", "override-negative"),
        ("job", "missing-template"),
        ("job", "missing-containers"),
        ("job", "container-scalar"),
        ("job", "resource-count"),
        ("pytorchjob", "replica-container"),
        ("pytorchjob", "replica-scalar"),
        ("pytorchjob", "missing-template"),
        ("jobset", "replica-container"),
        ("jobset", "replica-scalar"),
        ("jobset", "missing-template"),
    ],
)
def test_unknown_declared_gpu_count_holds_before_any_workload_mutation(
    kind: str, defect: str
) -> None:
    h = RestartHarness(kind)
    spec = h.api.source["spec"]
    if defect.startswith("override"):
        h.api.source["metadata"]["annotations"][ANNOTATION_TARGET_GPU_COUNT] = (
            "unknown" if defect == "override-text" else "-1"
        )
    elif defect == "replica-container":
        spec["pytorchReplicaSpecs" if kind == "pytorchjob" else "replicatedJobs"] = (
            "not-a-container"
        )
    elif defect == "replica-scalar":
        spec["pytorchReplicaSpecs" if kind == "pytorchjob" else "replicatedJobs"] = (
            {"Worker": "invalid"} if kind == "pytorchjob" else ["invalid"]
        )
    else:
        template = pod_templates(h.api.source, kind)[0]
        if defect == "missing-template":
            if kind == "job":
                spec["template"] = None
            elif kind == "pytorchjob":
                spec["pytorchReplicaSpecs"]["Master"]["template"] = None
            else:
                spec["replicatedJobs"][0]["template"]["spec"]["template"] = None
        elif defect == "missing-containers":
            template["spec"]["containers"] = []
        elif defect == "container-scalar":
            template["spec"]["containers"] = ["invalid"]
        else:
            template["spec"]["containers"][0]["resources"]["limits"][
                "nvidia.com/gpu"
            ] = "unknown"
    outcome = h.execute()
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "GPU_COUNT_CHANGED", outcome
    assert outcome.details["target_gpu_count"] is None, outcome
    assert outcome.details["restart_submitted"] is False, outcome
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)


def test_typed_kubernetes_job_is_serialized_into_an_independent_retry() -> None:
    h = RestartHarness("job", source_gpu_count=2)
    source = client.V1Job(
        metadata=client.V1ObjectMeta(
            name="training-job", uid="source-uid", resource_version="5"
        ),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={LABEL_ATTEMPT_ID: "attempt-a"}),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="trainer",
                            resources=client.V1ResourceRequirements(
                                limits={"nvidia.com/gpu": "2"}
                            ),
                        )
                    ],
                ),
            )
        ),
        status=client.V1JobStatus(active=0, failed=1),
    )
    h.api.source = source
    outcome = h.execute()
    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    body = next(iter(h.api.created.values()))
    assert body["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"] == {
        "nvidia.com/gpu": "2"
    }, body
    assert "uid" not in body["metadata"] and "status" not in body, body
    assert source.metadata.uid == "source-uid", source.metadata
