from __future__ import annotations

from pathlib import Path

import pytest

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
