"""Cleanup edges of the managed workload fixture the happy paths never reach.

Each test scripts one API irregularity -- a Pod gone between create and
read-back, a controller the label query no longer lists, a UID or version
that moves between two reads, Services whose inventory is malformed, a delete
whose receipt was lost -- and checks that the fixture either finishes the
cleanup it can prove or refuses with the reason, never deleting an incarnation
it did not own.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import managed_workload_fixture as managed
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from tests.regional.test_managed_workload_fixture import harness

PYTORCHJOB = "manifests/training/xid11-three-node-pytorchjob.yaml"


def _args(command: list[str]) -> list[str]:
    return command[command.index("-n") + 2 :]


def test_delete_resource_refuses_an_unknown_propagation_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)

    with pytest.raises(managed.RegionalFixtureError, match="propagation policy"):
        managed.delete_resource(
            fixture.regional, {"kind": "Pod"}, propagation_policy="Background"
        )
    assert api.deletes == []


def test_node_pinned_manifest_requires_pod_templates(tmp_path: Path) -> None:
    source = tmp_path / "job.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "kind": "PyTorchJob",
                "metadata": {"name": "job"},
                "spec": {"pytorchReplicaSpecs": ["not", "a", "mapping"]},
            }
        )
    )

    with pytest.raises(ValueError, match="has no Pod templates"):
        managed.render_node_pinned_manifest(source, tmp_path / "out.yaml", node="n")


def test_prewarm_create_refuses_a_pod_that_vanished_after_the_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case-test", run_id="run-test"
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = api.run(command, **kwargs)
        if _args(command)[0] == "create":
            api.objects.clear()
        return result

    monkeypatch.setattr(fixture.regional, "run", run)

    with pytest.raises(managed.RegionalFixtureError, match="prewarm Pod is missing"):
        prewarm.create(["node-a"])


def test_prewarm_cleanup_reports_a_pod_that_already_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _rendered = harness(tmp_path, monkeypatch)
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case-test", run_id="run-test"
    )
    prewarm.create(["node-a"])
    api.objects.clear()

    assert prewarm.cleanup() == {name: False for name in prewarm.created}
    assert api.deletes == []


def test_restart_authorization_requires_a_submitted_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _api, _rendered = harness(tmp_path, monkeypatch)

    with pytest.raises(managed.WorkloadOwnershipError, match="requires a submitted"):
        fixture.authorize_restart({})


def test_delete_includes_a_source_the_label_query_no_longer_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = _args(command)
        if args[0] == "get" and "-l" in args:
            # The label index lags: the controller is readable by name only.
            result = api.run(command, **kwargs)
            items = [
                item
                for item in json.loads(result.stdout)["items"]
                if item["metadata"]["name"] != fixture.name
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"items": items}), ""
            )
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    fixture.delete()

    assert key not in api.objects
    assert [kind for kind, _name, _options in api.deletes] == [fixture.resource]


def test_delete_refuses_a_controller_whose_uid_moved_between_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    reads = 0

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal reads
        args = _args(command)
        if args[0] == "get" and "-l" not in args and args[2] == fixture.name:
            reads += 1
            if reads == 2:
                api.objects[key]["metadata"]["uid"] = "replacement"
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    with pytest.raises(managed.RegionalFixtureError, match="UID changed"):
        fixture.delete()
    assert api.deletes == []


def test_delete_uses_the_freshest_version_of_an_owned_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    reads = 0

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal reads
        args = _args(command)
        if args[0] == "get" and "-l" not in args and args[2] == fixture.name:
            reads += 1
            if reads == 2:
                # A controller status update between the two reads.
                api.objects[key]["metadata"]["resourceVersion"] = "2"
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    fixture.delete()

    assert api.objects == {}
    assert [
        options["preconditions"]["resourceVersion"] for *_k, options in api.deletes
    ] == ["2"]


def test_delete_refuses_to_finish_while_owned_resources_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    stale = copy.deepcopy(api.objects[key])

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = _args(command)
        if args[0] == "get" and "-l" in args and api.deletes:
            # The label index still lists the controller the API already deleted.
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"items": [stale]}), ""
            )
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    with pytest.raises(managed.RegionalFixtureError, match="remain after cleanup"):
        fixture.delete()
    assert api.objects == {}
    assert len(api.deletes) == 1


def _pytorch_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, dict[str, Any]]:
    fixture, api, rendered = harness(tmp_path, monkeypatch, PYTORCHJOB)
    fixture.submit_rendered(rendered)
    job = next(obj for (kind, _n), obj in api.objects.items() if kind == "pytorchjob")
    return fixture, api, job


def _service(
    job: dict[str, Any], name: str, *, owner_uid: str | None
) -> dict[str, Any]:
    owners = (
        []
        if owner_uid is None
        else [
            {
                "apiVersion": "kubeflow.org/v1",
                "kind": "PyTorchJob",
                "name": job["metadata"]["name"],
                "uid": owner_uid,
            }
        ]
    )
    return {
        "kind": "Service",
        "metadata": {
            "name": name,
            "labels": {"training.kubeflow.org/job-name": job["metadata"]["name"]},
            "ownerReferences": owners,
        },
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [({"items": None}, "incomplete"), ({"items": ["not-an-object"]}, "invalid")],
)
def test_malformed_service_inventory_refuses_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], match: str
) -> None:
    fixture, api, _job = _pytorch_harness(tmp_path, monkeypatch)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = _args(command)
        if args[0] == "get" and args[1] == "service":
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    with pytest.raises(managed.RegionalFixtureError, match=match):
        fixture.delete()
    assert api.deletes == []


def test_an_orphaned_service_of_our_controller_is_still_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, job = _pytorch_harness(tmp_path, monkeypatch)
    api.add(_service(job, "job-master-0", owner_uid=None))

    fixture.delete()

    assert api.objects == {}
    assert [kind for kind, _name, _options in api.deletes][-1] == "service"


def test_a_service_that_left_or_was_replaced_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, job = _pytorch_harness(tmp_path, monkeypatch)
    gone = api.add(_service(job, "job-master-0", owner_uid=job["metadata"]["uid"]))
    replaced = api.add(_service(job, "job-worker-0", owner_uid=job["metadata"]["uid"]))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = _args(command)
        if args[0] == "get" and args[1] == "service" and "-l" not in args:
            if args[2] == gone["metadata"]["name"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            replaced["metadata"]["uid"] = "replacement"
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    with pytest.raises(managed.RegionalFixtureError, match="Service UID changed"):
        fixture.delete()

    assert ("service", "job-master-0") in api.objects
    assert ("service", "job-worker-0") in api.objects
    assert "service" not in [kind for kind, _name, _options in api.deletes]


def test_a_delete_whose_target_left_during_the_failure_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    attempts = 0

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        if _args(command)[0] == "delete":
            attempts += 1
            # The object disappeared under the request; the API reports failure.
            del api.objects[key]
            raise RegionalCommandFailed(1, "the object is gone")
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    fixture.delete()

    assert attempts == 1
    assert api.objects == {}


def test_a_failed_delete_re_adopts_the_changed_version_before_retrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    attempts: list[str] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if _args(command)[0] == "delete":
            options = json.loads(kwargs["input_text"])
            attempts.append(options["preconditions"]["resourceVersion"])
            if len(attempts) == 1:
                api.objects[key]["metadata"]["resourceVersion"] = "2"
                raise RegionalCommandFailed(1, "conflict")
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    fixture.delete()

    assert attempts == ["1", "2"]
    assert api.objects == {}


def _child(fixture: Any, name: str) -> dict[str, Any]:
    return {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "labels": {
                managed.OWNER_LABEL: fixture.owner,
                "gpu-fault.io/job-id": fixture.settings.job_id,
                "gpu-fault.io/attempt-id": fixture.settings.attempt_id,
            },
        },
    }


@pytest.mark.parametrize("outcome", ["vanishes", "changes-version"])
def test_a_late_child_whose_delete_fails_is_retried_or_proven_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    child_key = ("pod", "late-child")
    failed = 0
    reads_after_failure = 0

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal failed, reads_after_failure
        args = _args(command)
        if args[0] == "delete" and args[2].endswith("/jobs/" + fixture.name):
            result = api.run(command, **kwargs)
            # The controller's final dispatched request lands after its delete.
            api.add(_child(fixture, "late-child"))
            return result
        if args[0] == "delete" and args[2].endswith("/pods/late-child"):
            if failed == 0:
                failed += 1
                if outcome == "changes-version":
                    api.objects[child_key]["metadata"]["resourceVersion"] = "2"
                raise RegionalCommandFailed(1, "conflict")
        if args[0] == "get" and "-l" not in args and args[2] == "late-child":
            if failed and outcome == "vanishes":
                reads_after_failure += 1
                if reads_after_failure == 2:
                    api.objects.pop(child_key, None)
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)

    fixture.delete()

    assert api.objects == {}
    assert failed == 1
    pod_deletes = [options for kind, _n, options in api.deletes if kind == "pod"]
    if outcome == "changes-version":
        assert [item["preconditions"]["resourceVersion"] for item in pod_deletes] == [
            "2"
        ]
    else:
        assert pod_deletes == []
