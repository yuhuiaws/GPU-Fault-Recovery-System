from __future__ import annotations

import copy

import pytest

from gpu_fault.admin.node_key_custody_models import CustodyError
from gpu_fault.admin.node_key_custody_pods import bound_consumer_pods
from tests.regional._security_consumer_pods import (
    converged_deployment,
    owned_pod_documents,
)


@pytest.fixture
def documents():
    deployment = converged_deployment(
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "consumer",
                "namespace": "unit-system",
                "uid": "deployment-uid",
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "metadata": {},
                    "spec": {
                        "containers": [
                            {"name": "api", "image": "image@sha256:" + "a" * 64}
                        ]
                    },
                },
            },
        }
    )
    rs, pods = owned_pod_documents(deployment)
    return deployment, rs, pods


def test_bound_consumer_census_uses_actual_controllers(documents):
    deployment, rs, pods = documents
    assert (
        bound_consumer_pods(deployment, rs, pods, namespace="unit-system")
        == pods["items"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "foreign-pod",
        "foreign-rs",
        "namespace",
        "image",
        "missing",
        "duplicate",
        "not-ready",
        "ambiguous-ready",
        "not-running",
        "changed-generation",
        "extra-container",
        "empty-selector",
        "paused",
        "pagination",
        "missing-rs",
        "malformed-rs",
        "malformed-document",
    ],
)
def test_foreign_or_incomplete_consumer_evidence_is_never_used_for_key_activation(
    documents, defect
):
    deployment, rs, pods = copy.deepcopy(documents)
    pod = pods["items"][0]
    if defect == "foreign-pod":
        pod["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif defect == "foreign-rs":
        rs["items"][0]["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif defect == "namespace":
        pod["metadata"]["namespace"] = "foreign"
    elif defect == "image":
        pod["spec"]["containers"][0]["image"] = "foreign"
    elif defect == "missing":
        pods["items"] = []
    elif defect == "duplicate":
        deployment["spec"]["replicas"] = 2
        deployment = converged_deployment(deployment)
        pods["items"].append(copy.deepcopy(pod))
    elif defect == "not-ready":
        pod["status"]["conditions"][0]["status"] = "False"
    elif defect == "ambiguous-ready":
        pod["status"]["conditions"].append({"type": "Ready", "status": "False"})
    elif defect == "not-running":
        pod["status"]["containerStatuses"][0]["state"] = {}
    elif defect == "changed-generation":
        deployment["metadata"]["generation"] += 1
    elif defect == "extra-container":
        pod["spec"]["containers"].append({"name": "foreign", "image": "foreign"})
    elif defect == "empty-selector":
        deployment["spec"]["selector"] = {}
    elif defect == "paused":
        deployment["spec"]["paused"] = True
    elif defect == "pagination":
        pods["metadata"] = {"continue": "more"}
    elif defect == "missing-rs":
        rs["items"] = []
    elif defect == "malformed-rs":
        rs["items"].append({})
    else:
        pods["items"] = [None]
    with pytest.raises(CustodyError):
        bound_consumer_pods(deployment, rs, pods, namespace="unit-system")
