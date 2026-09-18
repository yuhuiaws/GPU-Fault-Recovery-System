from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import fixture_ownership as custody
from scripts.e2e.regional import managed_workload_fixture as managed
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional.test_managed_workload_fixture import Kubernetes, harness


def document(
    name: str = "owned", *, kind: str = "Job", uid: str = "uid-owned"
) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1" if kind == "Job" else "v1",
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": "gpu-fault-system",
            "uid": uid,
            "resourceVersion": "1",
        },
        "spec": {},
    }


def owner_reference(parent: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "apiVersion": parent["apiVersion"],
            "kind": parent["kind"],
            "name": parent["metadata"]["name"],
            "uid": parent["metadata"]["uid"],
            "controller": True,
        }
    ]


def ownership(tmp_path: Path) -> custody.FixtureOwnership:
    return custody.FixtureOwnership(
        tmp_path / "ownership.json", {"namespace": "gpu-fault-system", "run_id": "run"}
    )


def prepare(state: custody.FixtureOwnership) -> dict[str, Any]:
    value = document()
    state.begin()
    state.intend(value)
    state.acknowledge(value)
    return value


class AcknowledgingKubernetes(Kubernetes):
    def run(self, command: list[str], **kwargs: Any) -> Any:
        result = super().run(command, **kwargs)
        args = command[command.index("-n") + 2 :]
        if args[0] == "create" and "-o" in args:
            value = json.loads(kwargs["input_text"])
            result.stdout = json.dumps(
                self.objects[(value["kind"].lower(), value["metadata"]["name"])]
            )
        return result


def managed_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[managed.ManagedWorkloadFixture, AcknowledgingKubernetes, str, Path]:
    old, _, rendered = harness(tmp_path, monkeypatch)
    api = AcknowledgingKubernetes()
    monkeypatch.setattr(old.regional, "run", api.run)
    path = tmp_path / "workload-state.json"
    fixture = managed.ManagedWorkloadFixture(
        old.regional, old.settings, state_path=path
    )
    return fixture, api, rendered, path


def test_creation_custody_survives_controller_loss_without_reauthorizing_submit(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    value = prepare(state)
    resumed = custody.FixtureOwnership(state.path, state.binding)
    assert resumed.resuming and resumed.record.owner == state.record.owner
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        resumed.begin()
    with pytest.raises(RegionalFixtureError, match="repeated"):
        resumed.intend(document("other"))
    resumed.observe(value, lambda *_: None)
    resumed.complete()
    assert resumed.record.completed, "confirmed cleanup must remain durable"
    assert state.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("stage", ["fresh", "started", "complete"])
def test_repeated_and_unstarted_creations_are_rejected(
    stage: str, tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    if stage != "fresh":
        prepare(state)
    if stage == "complete":
        state.complete()
    with pytest.raises(RegionalFixtureError):
        state.intend(document())
    if stage != "fresh":
        with pytest.raises(RegionalFixtureError, match="cleanup-only"):
            state.begin()


@pytest.mark.parametrize(
    "defect",
    ["wrong-name", "wrong-namespace", "missing-rv", "empty-rv", "missing-uid", "kind"],
)
def test_ack_must_be_a_direct_matching_creation(defect: str, tmp_path: Path) -> None:
    state = ownership(tmp_path)
    state.begin()
    state.intend(document())
    value = document()
    if defect == "kind":
        value["kind"] = "Pod"
    elif defect == "missing-uid":
        value["metadata"].pop("uid")
    else:
        key, replacement = {
            "wrong-name": ("name", "different"),
            "wrong-namespace": ("namespace", "other"),
            "missing-rv": ("resourceVersion", None),
            "empty-rv": ("resourceVersion", ""),
        }[defect]
        value["metadata"][key] = replacement
    with pytest.raises(RegionalFixtureError, match="identity|acknowledgement"):
        state.acknowledge(value)
    with pytest.raises(RegionalFixtureError, match="unknown"):
        state.require_acknowledged()
    assert state.record.creations["job/owned"].uid is None


def test_ack_cannot_be_replaced_and_unknown_ack_does_not_authorize_adoption(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    state.begin()
    state.intend(document())
    with pytest.raises(RegionalFixtureError, match="unknown"):
        state.observe(document(), lambda *_: None)
    state.acknowledge(document())
    with pytest.raises(RegionalFixtureError, match="acknowledgement"):
        state.acknowledge(document())


def test_child_must_lead_to_the_original_acknowledged_controller(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    parent = prepare(state)
    child = document("child", kind="Pod", uid="uid-child")
    child["metadata"]["ownerReferences"] = owner_reference(parent)
    state.observe(child, lambda *_: parent)
    assert state.record.observed["pod/child"] == "uid-child"
    child["metadata"]["uid"] = "replacement"
    with pytest.raises(RegionalFixtureError, match="observed UID"):
        state.observe(child, lambda *_: parent)


@pytest.mark.parametrize(
    "defect",
    ["no-owner", "two-owners", "non-controller", "kind", "name", "uid", "namespace"],
)
def test_foreign_or_ambiguous_controller_is_never_adopted(
    defect: str, tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    parent = prepare(state)
    child = document("child", kind="Pod")
    owners = owner_reference(parent)
    child["metadata"]["ownerReferences"] = owners
    if defect == "no-owner":
        child["metadata"].pop("ownerReferences")
    elif defect == "two-owners":
        owners.append(copy.deepcopy(owners[0]))
    elif defect == "non-controller":
        owners[0]["controller"] = False
    elif defect == "namespace":
        child["metadata"]["namespace"] = "foreign"
    else:
        owners[0][defect] = {"kind": "Secret", "name": "", "uid": ""}[defect]
    with pytest.raises(RegionalFixtureError, match="owner"):
        state.observe(child, lambda *_: parent)
    assert not state.record.observed, (
        "unproven children cannot acquire deletion custody"
    )


@pytest.mark.parametrize("replacement", [None, "new-uid"])
def test_missing_or_recreated_parent_refuses_child(
    replacement: str | None, tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    parent = prepare(state)
    child = document("child", kind="Pod")
    child["metadata"]["ownerReferences"] = owner_reference(parent)
    current = None if replacement is None else document(uid=replacement)
    with pytest.raises(RegionalFixtureError, match="owner changed"):
        state.observe(child, lambda *_: current)


def test_root_recreation_is_not_accepted_even_with_identical_labels(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    prepare(state)
    with pytest.raises(RegionalFixtureError, match="creation UID"):
        state.observe(document(uid="replacement"), lambda *_: None)


@pytest.mark.parametrize("cycle", [True, False])
def test_owner_traversal_is_bounded_and_rejects_cycles(
    cycle: bool, tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    prepare(state)
    objects = [document(f"child-{index}", uid=f"child-{index}") for index in range(9)]
    for index, item in enumerate(objects):
        item["metadata"]["ownerReferences"] = owner_reference(
            objects[0 if cycle else min(index + 1, 8)]
        )
    lookup = {item["metadata"]["name"]: item for item in objects}
    with pytest.raises(RegionalFixtureError, match="chain"):
        state.observe(objects[0], lambda _kind, name: lookup[name])


@pytest.mark.parametrize("defect", ["binding", "shape", "invalid-json", "link", "mode"])
def test_corrupt_or_untrusted_local_journal_is_rejected(
    defect: str, tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    if defect == "binding":
        binding = {**state.binding, "run_id": "different"}
    else:
        binding = state.binding
        if defect == "shape":
            data = json.loads(state.path.read_text())
            data["creations"]["wrong/key"] = {"expected": document(), "uid": None}
            state.path.write_text(json.dumps(data))
        elif defect == "invalid-json":
            state.path.write_text("{")
        elif defect == "mode":
            state.path.chmod(0o644)
        else:
            target = tmp_path / "target"
            state.path.rename(target)
            state.path.symlink_to(target)
    with pytest.raises(RegionalFixtureError):
        custody.FixtureOwnership(state.path, binding)


def test_binding_is_rechecked_before_every_mutation(tmp_path: Path) -> None:
    binding: dict[str, Any] = {"namespace": "gpu-fault-system"}
    state = custody.FixtureOwnership(
        tmp_path / "record.json", binding.copy(), current_binding=lambda: binding
    )
    binding["namespace"] = "other"
    with pytest.raises(RegionalFixtureError, match="connection or source"):
        state.begin()


def test_journal_size_is_bounded(tmp_path: Path) -> None:
    state = ownership(tmp_path)
    state.begin()
    value = document()
    value["spec"]["large"] = "x" * 200000
    with pytest.raises(RegionalFixtureError, match="too large"):
        state.intend(value)


@pytest.mark.parametrize("raw", ["[]", "{", '{"a":1,"a":2}', " " * 1048577])
def test_create_receipt_is_not_an_unbounded_or_ambiguous_document(raw: str) -> None:
    with pytest.raises(RegionalFixtureError, match="bounded object"):
        custody.creation_document(raw)


@pytest.mark.parametrize("value", [{}, {"metadata": []}, {"metadata": {"uid": ""}}])
def test_creation_identity_requires_a_complete_resource(
    value: dict[str, Any], tmp_path: Path
) -> None:
    state = ownership(tmp_path)
    prepare(state)
    with pytest.raises(RegionalFixtureError, match="identity"):
        state.acknowledge(value)


def test_real_managed_fixture_resumes_only_cleanup_and_deletes_owned_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)
    assert fixture.submit_rendered(rendered)["stdout"] == "created"
    source = api.objects[(fixture.resource, fixture.name)]
    child = document("training", kind="Pod", uid="training-uid")
    child["metadata"]["labels"] = source["spec"]["template"]["metadata"]["labels"]
    child["metadata"]["ownerReferences"] = owner_reference(source)
    api.add(child)
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        resumed.submit_rendered(rendered)
    resumed.delete()
    assert not api.objects, "source and proven descendants must all be removed"
    assert api.deletes[-1][0:2] == ("pod", "training")
    assert resumed.ownership is not None and resumed.ownership.record.completed
    managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    ).delete()
    assert len(api.deletes) == 2


def test_missing_managed_create_ack_remains_unresolved_across_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)
    api.fail_create_ack = True
    with pytest.raises(RegionalFixtureError, match="receipt lost"):
        fixture.submit_rendered(rendered)
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    with pytest.raises(RegionalFixtureError, match="unknown"):
        resumed.delete()
    assert api.objects and not api.deletes


@pytest.mark.parametrize("lost_ack", [True, False])
def test_prewarm_uses_original_create_uid_after_restart(
    lost_ack: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _, _ = managed_harness(tmp_path, monkeypatch)
    path = tmp_path / "prewarm-state.json"
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )
    api.fail_create_ack = lost_ack
    if lost_ack:
        with pytest.raises(RegionalFixtureError, match="receipt lost"):
            prewarm.create(["node-a"])
    else:
        assert prewarm.create(["node-a"])["created"] == ["node-a"]
    resumed = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        resumed.create(["node-a"])
    if lost_ack:
        with pytest.raises(RegionalFixtureError, match="unknown"):
            resumed.cleanup()
        assert not api.deletes, "a missing creation ACK cannot authorize deletion"
    else:
        assert not any(resumed.cleanup().values()), (
            "prewarm cleanup must prove no residue"
        )
        assert not api.objects, "prewarm cleanup must remove the original Pod"
