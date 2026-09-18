from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import managed_workload_fixture as managed
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from tests.regional.test_managed_workload_fixture import harness


def child(fixture, *, name="training") -> dict[str, Any]:
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


def test_controllers_are_orphaned_before_pods_and_gc_versions_are_refreshed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    api.add(child(fixture))
    recreated = []

    def run(command, **kwargs):
        args = command[command.index("-n") + 2 :]
        if args[0] == "delete":
            plural = args[2].split("/")[-2]
            if plural == "pods" and (fixture.resource, fixture.name) in api.objects:
                recreated.append("controller could replace the Pod")
            if plural == "jobs":
                assert json.loads(kwargs["input_text"])["propagationPolicy"] == "Orphan"
                result = api.run(command, **kwargs)
                api.objects[("pod", "training")]["metadata"]["resourceVersion"] = "2"
                return result
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)
    fixture.delete()
    assert recreated == []
    assert api.objects == {}
    assert [kind for kind, _name, _options in api.deletes] == ["job", "pod"]
    assert api.deletes[-1][2]["gracePeriodSeconds"] == 0
    assert api.deletes[-1][2]["preconditions"]["resourceVersion"] == "2"


def test_final_inflight_controller_child_is_included_in_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)

    def run(command, **kwargs):
        args = command[command.index("-n") + 2 :]
        result = api.run(command, **kwargs)
        if args[0] == "delete" and "/jobs/" in args[2]:
            api.add(child(fixture, name="late-pod"))
        return result

    monkeypatch.setattr(fixture.regional, "run", run)
    fixture.delete()
    assert api.objects == {}
    assert api.deletes[-1][:2] == ("pod", "late-pod")


@pytest.mark.parametrize(
    "change", ["version", "uid", "owner", "unchanged", "always-version"]
)
def test_delete_retries_only_changed_versions_of_the_same_owned_incarnation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    key = (fixture.resource, fixture.name)
    attempts = []
    uid = api.objects[key]["metadata"]["uid"]

    def run(command, **kwargs):
        args = command[command.index("-n") + 2 :]
        if args[0] == "delete":
            options = json.loads(kwargs["input_text"])
            attempts.append(options)
            if len(attempts) == 1 or change == "always-version":
                metadata = api.objects[key]["metadata"]
                if change != "unchanged":
                    metadata["resourceVersion"] = str(len(attempts) + 1)
                if change == "uid":
                    metadata["uid"] = "foreign-incarnation"
                if change == "owner":
                    metadata["labels"][managed.OWNER_LABEL] = "foreign-owner"
                raise RegionalCommandFailed(1, "API rejected the delete")
        return api.run(command, **kwargs)

    monkeypatch.setattr(fixture.regional, "run", run)
    if change == "version":
        fixture.delete()
        assert api.objects == {}
        assert [item["preconditions"]["resourceVersion"] for item in attempts] == [
            "1",
            "2",
        ]
    else:
        with pytest.raises(managed.RegionalFixtureError):
            fixture.delete()
        assert key in api.objects
        assert len(attempts) == (3 if change == "always-version" else 1)
    assert all(item["preconditions"]["uid"] == uid for item in attempts), (
        'test_delete_retries_only_changed_versions_of_the_same_owned_incarnation: expected all(item["preconditions"]["uid"] == uid for item in att...'
    )


def test_ownership_change_after_controller_removal_preserves_foreign_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered = harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    api.add(child(fixture))

    def run(command, **kwargs):
        args = command[command.index("-n") + 2 :]
        result = api.run(command, **kwargs)
        if args[0] == "delete" and "/jobs/" in args[2]:
            metadata = api.objects[("pod", "training")]["metadata"]
            metadata["labels"][managed.OWNER_LABEL] = "foreign-owner"
            metadata["resourceVersion"] = "2"
        return result

    monkeypatch.setattr(fixture.regional, "run", run)
    with pytest.raises(managed.RegionalFixtureError, match="ownership"):
        fixture.delete()
    assert ("pod", "training") in api.objects
    assert [kind for kind, _name, _options in api.deletes] == ["job"]
