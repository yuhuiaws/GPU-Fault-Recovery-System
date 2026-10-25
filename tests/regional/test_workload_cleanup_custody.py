from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import fixture_ownership as custody_module
from scripts.e2e.regional import managed_workload_fixture as managed
from tests.regional.test_fixture_ownership import (
    document,
    managed_harness,
    owner_reference,
)


def orphaned_fixture(tmp_path, monkeypatch):
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    source = api.objects[(fixture.resource, fixture.name)]
    pod = document("training", kind="Pod", uid="training-uid")
    pod["metadata"]["labels"] = source["spec"]["template"]["metadata"]["labels"]
    pod["metadata"]["ownerReferences"] = owner_reference(source)
    pod = api.add(pod)
    fixture.adopt(pod)
    assert fixture.ownership is not None
    fixture.ownership.retain_for_cleanup(pod)
    del api.objects[(fixture.resource, fixture.name)]
    pod["metadata"].pop("ownerReferences")
    pod["metadata"]["resourceVersion"] = "2"
    return fixture, api, path, pod


def test_orphan_cleanup_custody_survives_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, path, _pod = orphaned_fixture(tmp_path, monkeypatch)
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    resumed.delete()
    assert api.objects == {}
    assert api.deletes[0][2]["preconditions"] == {
        "uid": "training-uid",
        "resourceVersion": "2",
    }
    assert resumed.ownership is not None
    assert resumed.ownership.record.completed, (
        "test_orphan_cleanup_custody_survives_process_restart: expected resumed.ownership.record.completed"
    )


@pytest.mark.parametrize("field", ["uid", "owner-reference", "owner-label", "spec"])
def test_orphan_cleanup_custody_does_not_authorize_changed_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    fixture, api, path, pod = orphaned_fixture(tmp_path, monkeypatch)
    if field == "uid":
        pod["metadata"]["uid"] = "foreign-uid"
    elif field == "owner-reference":
        pod["metadata"]["ownerReferences"] = [
            {"kind": "Job", "name": "foreign", "uid": "foreign-uid", "controller": True}
        ]
    elif field == "owner-label":
        pod["metadata"]["labels"][managed.OWNER_LABEL] = "foreign"
    else:
        pod["spec"]["serviceAccountName"] = "foreign"
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    with pytest.raises(managed.RegionalFixtureError, match="ownership"):
        resumed.delete()
    assert ("pod", "training") in api.objects
    assert api.deletes == []


def test_cleanup_custody_cannot_be_created_from_an_unobserved_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, _path = managed_harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    assert fixture.ownership is not None
    with pytest.raises(managed.RegionalFixtureError, match="observed ownership"):
        fixture.ownership.retain_for_cleanup(document("unknown", kind="Pod"))
    assert api.deletes == []


def test_existing_cleanup_custody_cannot_be_rebound_to_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _api, _path, pod = orphaned_fixture(tmp_path, monkeypatch)
    pod["spec"]["serviceAccountName"] = "foreign"
    assert fixture.ownership is not None
    with pytest.raises(managed.RegionalFixtureError, match="ownership"):
        fixture.ownership.retain_for_cleanup(pod)


def _managed_pytorchjob() -> dict[str, Any]:
    """A product restart copy as the live snapshot showed it, before its own
    attempt failed: the restart's annotations, no suspend yet."""

    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "name": "gpu-fault-single-node-warm-spare-c098a41205-r-893dc810",
            "namespace": "gpu-fault-system",
            "uid": "61eef5d9-bb45-415f-815f-bf16704c04be",
            "resourceVersion": "300592001",
            "labels": {
                managed.OWNER_LABEL: "de47367c6f6844eda94fdf84d4244b32",
                "gpu-fault.io/job-id": "gl28nz-nz-a2-181de9b8",
                "gpu-fault.io/attempt-id": "gl28nz-nz-a2-181de9b8-a001-r-893dc810",
                "gpu-fault.io/managed": "true",
            },
            "annotations": {
                "gpu-fault.io/operation-id": "workflow-restart/1/RESTART_WORKLOAD",
                "gpu-fault.io/restart-count": "1",
            },
        },
        "spec": {
            "runPolicy": {"cleanPodPolicy": "None"},
            "pytorchReplicaSpecs": {"Master": {"replicas": 1}},
        },
        "status": {"conditions": [{"type": "Running", "status": "True"}]},
    }


def _contained_by_the_product(document: dict[str, Any]) -> None:
    """``STOP_WORKLOADS`` of the copy's own failed attempt, as the live
    object read at 11:36:03Z: the step's annotations and the suspend, one patch
    (``_apply_workload_mutation``)."""

    document["metadata"]["annotations"].update(
        {
            "gpu-fault.io/execution-epoch": "1",
            "gpu-fault.io/fencing-token": "1",
            "gpu-fault.io/incident-id": "inc-attempt-failure-33afa6ea13d46244dcd4",
            "gpu-fault.io/operation-id": (
                "workflow-attempt-stop-33afa6ea13d46244dcd4/1/STOP_WORKLOADS"
            ),
            "gpu-fault.io/termination-initiator-incident-id": (
                "inc-attempt-failure-33afa6ea13d46244dcd4"
            ),
            "gpu-fault.io/workflow-id": "workflow-attempt-stop-33afa6ea13d46244dcd4",
            "gpu-fault.io/workflow-step-index": "1",
        }
    )
    document["spec"]["runPolicy"]["suspend"] = True
    document["metadata"]["resourceVersion"] = "300592380"
    document["metadata"]["generation"] = 2
    document["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}


@pytest.mark.parametrize(
    "bookkeeping",
    ["containment", "relabelled-attempt", "pod-termination-annotation", "suspend-only"],
)
def test_the_products_own_bookkeeping_on_a_retained_object_keeps_its_custody(
    bookkeeping: str,
) -> None:
    """Live 2026-10-25 (eksb, 028.nonzero_exit): the cleanup retained the product's
    restart copy at 11:35:5xZ, the copy's own attempt failed and the product's
    STOP_WORKLOADS suspended and annotated it at 11:36:03Z, and the delete that
    followed was refused ("fixture cleanup ownership changed"). What the product
    writes on a managed workload it contains or restarts is not a change of
    ownership: the identity (UID, namespace, kind, name), the owner nonce, the
    job id and every other field still have to match."""

    document = _managed_pytorchjob()
    custody = custody_module.capture_cleanup_custody(document)
    if bookkeeping == "containment":
        _contained_by_the_product(document)
    elif bookkeeping == "relabelled-attempt":
        document["metadata"]["labels"]["gpu-fault.io/attempt-id"] = (
            "gl28nz-nz-a2-181de9b8-a001-r-0badcafe"
        )
    elif bookkeeping == "pod-termination-annotation":
        document["metadata"]["annotations"] = {
            "gpu-fault.io/termination-initiator-incident-id": "inc-attempt-failure-1"
        }
    else:
        document["spec"]["runPolicy"]["suspend"] = True
    custody_module.verify_cleanup_custody(
        custody,
        document,
        observed_uid=document["metadata"]["uid"],
        namespace="gpu-fault-system",
    )


@pytest.mark.parametrize(
    "change", ["owner-label", "job-id", "foreign-annotation", "spec", "namespace"]
)
def test_everything_but_the_products_bookkeeping_still_binds_the_custody(
    change: str,
) -> None:
    document = _managed_pytorchjob()
    custody = custody_module.capture_cleanup_custody(document)
    if change == "owner-label":
        document["metadata"]["labels"][managed.OWNER_LABEL] = "someone-else"
    elif change == "job-id":
        document["metadata"]["labels"]["gpu-fault.io/job-id"] = "another-job"
    elif change == "foreign-annotation":
        document["metadata"]["annotations"]["kueue.x-k8s.io/queue-name"] = "q"
    elif change == "spec":
        document["spec"]["pytorchReplicaSpecs"]["Master"]["replicas"] = 2
    else:
        document["metadata"]["namespace"] = "default"
    with pytest.raises(managed.RegionalFixtureError, match="ownership"):
        custody_module.verify_cleanup_custody(
            custody,
            document,
            observed_uid=document["metadata"]["uid"],
            namespace="gpu-fault-system",
        )
