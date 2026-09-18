from __future__ import annotations

import copy
import json
import subprocess

import pytest

from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._prerequisite_repair_support import repair_release
from tests.regional.test_installed_resource_registry import COLLECT_MODULE
from tests.regional.test_release_prerequisite_repair import REPAIR, prepare

NAMESPACE = "gpu-fault-system"
PARENT = "gpu-fault-aurora-credential-refresh"


def test_proof_jobs_are_children_of_the_registered_refresher(monkeypatch) -> None:
    instance = repair_release(monkeypatch, bootstrap=True)
    prepare(instance, bootstrap=True)
    REPAIR.bootstrap_workflow_proof(instance)
    assert len(instance.runner.created_jobs) == 2
    for job in instance.runner.created_jobs:
        assert job["metadata"]["ownerReferences"] == [
            {
                "apiVersion": "batch/v1",
                "kind": "CronJob",
                "name": PARENT,
                "uid": "previous-uid",
                "controller": False,
                "blockOwnerDeletion": False,
            }
        ]
    records = instance.state[REPAIR.REPAIR_KEY]["jobs"].values()
    assert all(record["owner_uid"] == "previous-uid" for record in records), (
        "the proof journal omitted its registered garbage-collection owner"
    )


def test_a_reparented_proof_job_is_neither_trusted_nor_deleted(monkeypatch) -> None:
    instance = repair_release(monkeypatch)
    run = instance.runner.run

    def reparent(args, **kwargs):
        result = run(args, **kwargs)
        if "get" in args and args[args.index("get") + 1] == "job" and result:
            value = json.loads(result)
            value["metadata"]["ownerReferences"][0]["uid"] = "foreign-owner"
            return json.dumps(value)
        return result

    monkeypatch.setattr(instance.runner, "run", reparent)
    with pytest.raises(ReleaseError, match="ownership or UID differs"):
        prepare(instance)
    assert instance.runner.jobs, "cleanup deleted a Job after ownership changed"
    assert "job-delete" not in instance.runner.events, (
        "a changed ownership reference reached deletion"
    )


class JobDiscovery:
    def __init__(self) -> None:
        self.parent = {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {"name": PARENT, "namespace": NAMESPACE, "uid": "owner"},
        }
        self.job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "gpu-fault-store-proof-" + "a" * 16,
                "namespace": NAMESPACE,
                "ownerReferences": [
                    {
                        "apiVersion": "batch/v1",
                        "kind": "CronJob",
                        "name": PARENT,
                        "uid": "owner",
                        "controller": False,
                    }
                ],
            },
        }

    def run(self, args, **_kwargs):
        if "-A" in args:
            assert "job" in args[1].split(","), (
                "installed-resource discovery omitted transient proof Jobs"
            )
            items = [self.parent, self.job]
        else:
            items = []
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": items}), "")


def discovered(kubectl: JobDiscovery, *, registered: bool = True) -> list[dict]:
    return COLLECT_MODULE.discover_unregistered(
        kubectl,
        plane="cpu",
        context="cpu",
        namespace=NAMESPACE,
        registered={("cronjob", PARENT)} if registered else set(),
    )


def test_registered_parent_uid_covers_its_transient_job() -> None:
    assert discovered(JobDiscovery()) == []


@pytest.mark.parametrize(
    "change",
    ["orphan", "new-parent-uid", "other-namespace", "multiple-owners", "unregistered"],
)
def test_orphaned_or_ambiguous_jobs_remain_visible_to_uninstall(change: str) -> None:
    kubectl = JobDiscovery()
    if change == "orphan":
        kubectl.job["metadata"].pop("ownerReferences")
    elif change == "new-parent-uid":
        kubectl.parent["metadata"]["uid"] = "replacement"
    elif change == "other-namespace":
        kubectl.job["metadata"]["namespace"] = "gf-regional-other"
    elif change == "multiple-owners":
        other = copy.deepcopy(kubectl.job["metadata"]["ownerReferences"][0])
        other["uid"] = "other-owner"
        kubectl.job["metadata"]["ownerReferences"].append(other)
    resources = discovered(kubectl, registered=change != "unregistered")
    assert ("job", kubectl.job["metadata"]["name"]) in {
        (item["kind"], item["name"]) for item in resources
    }, "uninstall lost a proof Job that has no unique registered owner"


def test_discovery_does_not_adopt_customer_namespace_jobs() -> None:
    kubectl = JobDiscovery()
    kubectl.job["metadata"]["namespace"] = "training"
    assert discovered(kubectl) == []
