from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional.test_fixture_ownership import (
    document,
    managed_harness,
    owner_reference,
)


def test_empty_annotations_are_omitted_before_creation_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, _path = managed_harness(tmp_path, monkeypatch)
    source = yaml.safe_load(rendered)
    source["metadata"]["annotations"] = {}
    fixture.submit_rendered(yaml.safe_dump(source))
    assert "annotations" not in api.created[0]["metadata"]
    assert fixture.ownership is not None
    creation = fixture.ownership.record.creations[fixture.resource + "/" + fixture.name]
    assert creation.approved is True
    assert "annotations" not in creation.expected["metadata"]
    fixture.delete()
    assert api.objects == {}


def test_nonempty_annotations_remain_required_and_rejected_controller_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, _path = managed_harness(tmp_path, monkeypatch)
    source = yaml.safe_load(rendered)
    source["metadata"]["annotations"] = {"unit.example/required": "preserve"}

    def run(command, **kwargs):
        args = command[command.index("-n") + 2 :]
        result = api.run(command, **kwargs)
        if args[0] == "create":
            controller = api.objects[(fixture.resource, fixture.name)]
            controller["metadata"].pop("annotations")
            child = document("unapproved-child", kind="Pod", uid="unapproved-child-uid")
            child["metadata"]["labels"] = controller["spec"]["template"]["metadata"][
                "labels"
            ]
            child["metadata"]["ownerReferences"] = owner_reference(controller)
            api.add(child)
            result.stdout = json.dumps(controller)
        elif args[0] == "delete":
            assert json.loads(kwargs["input_text"])["propagationPolicy"] == "Foreground"
            api.objects.pop(("pod", "unapproved-child"))
        return result

    monkeypatch.setattr(fixture.regional, "run", run)
    with pytest.raises(RegionalFixtureError, match="declared intent"):
        fixture.submit_rendered(yaml.safe_dump(source))
    assert fixture.ownership is not None
    creation = fixture.ownership.record.creations[fixture.resource + "/" + fixture.name]
    assert creation.approved is False
    fixture.delete()
    assert api.objects == {}
    assert [kind for kind, _name, _options in api.deletes] == [fixture.resource]
    assert fixture.ownership.record.completed is True
    assert (
        fixture.ownership.record.creations[
            fixture.resource + "/" + fixture.name
        ].approved
        is False
    )
