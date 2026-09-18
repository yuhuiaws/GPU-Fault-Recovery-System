"""Peer-review regressions and hermetic edge coverage for fixture custody."""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.e2e.regional import destr008_cleanup_identity as cleanup
from scripts.e2e.regional import fixture_ownership as custody
from scripts.e2e.regional import managed_workload_fixture as managed
from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture
from tests.regional.test_fixture_ownership import (
    AcknowledgingKubernetes,
    document,
    managed_harness,
    owner_reference,
    ownership,
    prepare,
)
from tests.regional.test_managed_workload_fixture import harness


@pytest.fixture(autouse=True)
def fixture_ownership_hermetic_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Also usable as a plugin when running the existing fixture tests together."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fixture ownership tests require fake external I/O")

    monkeypatch.setattr(custody, "host_identity", lambda: "a" * 64)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def intended(state: custody.FixtureOwnership) -> dict[str, Any]:
    value = document()
    value["metadata"].pop("uid")
    value["metadata"].pop("resourceVersion")
    value["metadata"]["labels"] = {managed.OWNER_LABEL: state.record.owner}
    value["spec"] = {
        "parallelism": 1,
        "template": {
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "training",
                        "image": "example.invalid/training@sha256:" + "b" * 64,
                        "command": ["owned-training"],
                        "resources": {"limits": {"nvidia.com/gpu": 1}},
                    }
                ],
            }
        },
    }
    return value


def acknowledgement(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["metadata"].update(uid="uid-owned", resourceVersion="1")
    return result


def delete_only_record(state: custody.FixtureOwnership, key: str, uid: str) -> str:
    value = state.record.creations[key].model_dump(mode="json")
    assert value["uid"] == uid, "a valid CREATE ACK must retain creation custody"
    assert value["approved"] is False, "rejected creation must not authorize execution"
    digest = value["ack_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    return digest


def rejected_creation(state: custody.FixtureOwnership) -> dict[str, Any]:
    value = intended(state)
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    ack["spec"]["parallelism"] = 2
    with pytest.raises(RegionalFixtureError):
        state.acknowledge(ack)
    delete_only_record(state, "job/owned", ack["metadata"]["uid"])
    return ack


def mutate_create_reply(
    monkeypatch: pytest.MonkeyPatch,
    regional: RegionalLiveFixture,
    api: AcknowledgingKubernetes,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    original = api.run

    def changed(command: list[str], **kwargs: Any) -> Any:
        result = original(command, **kwargs)
        if "create" in command:
            value = json.loads(kwargs["input_text"])
            current = api.objects[(value["kind"].lower(), value["metadata"]["name"])]
            mutate(current)
            result.stdout = json.dumps(current)
        return result

    monkeypatch.setattr(regional, "run", changed)


@pytest.mark.parametrize("missing", ["cpu", "gpu"])
def test_binding_refuses_an_unreadable_connection_before_creating_a_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    fixture, api, _ = harness(tmp_path, monkeypatch)
    getattr(fixture.regional.settings, missing + "_kubeconfig").unlink()
    with pytest.raises(RegionalFixtureError, match="connection identity"):
        managed.ManagedWorkloadFixture(
            fixture.regional, fixture.settings, state_path=tmp_path / "journal.json"
        )
    assert not (tmp_path / "journal.json").exists(), (
        "unreadable connection identity must not create a custody journal"
    )
    assert not api.created and not api.deletes


def test_binding_records_actual_owned_file_contents_and_observed_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, _, _ = harness(tmp_path, monkeypatch)
    bound = custody.fixture_binding(
        fixture.regional, purpose="unit-test", inputs={"operation": "owned"}
    )
    assert bound["host_sha256"] == "a" * 64
    assert bound["inputs"] == {"operation": "owned"}
    for name in ("cpu_kubeconfig", "gpu_kubeconfig"):
        path = getattr(fixture.regional.settings, name)
        assert bound["connections"]["arguments:" + name] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_server_assigned_metadata_and_defaulted_fields_do_not_change_the_intent(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    ack["metadata"]["creationTimestamp"] = "2026-09-13T00:00:00Z"
    ack["spec"]["backoffLimit"] = 6
    assert state.acknowledge(ack) == "uid-owned"
    state.observe(ack, lambda *_: pytest.fail("root needs no owner traversal"))
    repeated = copy.deepcopy(ack)
    repeated["metadata"]["resourceVersion"] = "2"
    repeated["status"] = {"active": 1}
    state.observe(repeated, lambda *_: pytest.fail("root needs no owner traversal"))
    assert state.record.creations["job/owned"].expected == value
    assert state.record.observed == {"job/owned": "uid-owned"}
    assert state.record.creations["job/owned"].model_dump()["approved"] is True
    assert state.rejected_resources(lambda _kind, _name: repeated) == []


@pytest.mark.parametrize(
    ("resource", "requested", "admitted"),
    [
        ("cpu", "100m", "0.1"),
        ("cpu", 0.1, "100m"),
        ("cpu", 1, "1000m"),
        ("memory", "1Gi", "1024Mi"),
        ("nvidia.com/gpu", 1, "1"),
    ],
)
def test_quantity_normalization_preserves_the_approved_resource_contract(
    tmp_path: Path, resource: str, requested: Any, admitted: Any
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    resources = value["spec"]["template"]["spec"]["containers"][0]["resources"]
    resources["requests"] = {resource: requested}
    resources["limits"] = {resource: requested}
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    resources = ack["spec"]["template"]["spec"]["containers"][0]["resources"]
    resources["requests"][resource] = admitted
    resources["limits"][resource] = admitted
    assert state.acknowledge(ack) == "uid-owned"
    assert state.record.creations["job/owned"].model_dump()["approved"] is True
    state.observe(ack, lambda *_: None)
    assert state.record.observed == {"job/owned": "uid-owned"}


@pytest.mark.parametrize(
    ("requested", "admitted"),
    [
        ("1", "not-a-quantity"),
        ("1", True),
        (True, "1"),
        ("NaN", "NaN"),
        ("Infinity", "Infinity"),
        ("1", "NaN"),
        ("1", {"unexpected": "mapping"}),
    ],
)
def test_invalid_quantities_retain_delete_only_custody_without_approval(
    tmp_path: Path, requested: Any, admitted: Any
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    value["spec"]["template"]["spec"]["containers"][0]["resources"] = {
        "limits": {"cpu": requested}
    }
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    ack["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"] = (
        admitted
    )
    with pytest.raises(RegionalFixtureError):
        state.acknowledge(ack)
    delete_only_record(state, "job/owned", "uid-owned")
    assert state.rejected_resources(lambda *_: ack) == [ack]
    with pytest.raises(RegionalFixtureError):
        state.observe(ack, lambda *_: None)


@pytest.mark.parametrize(
    "authority",
    [
        {"uid": "known", "ack_sha256": None, "approved": False},
        {"uid": None, "ack_sha256": "a" * 64, "approved": False},
        {"uid": None, "ack_sha256": None, "approved": True},
    ],
)
def test_creation_authority_requires_a_paired_uid_and_fingerprint(
    authority: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="authority is incomplete"):
        custody.Creation(expected=document(), **authority)


@pytest.mark.parametrize("defect", ["extra-container", "mapping", "scalar-type"])
def test_declared_structural_changes_are_never_execution_approval(
    tmp_path: Path, defect: str
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    template = ack["spec"]["template"]["spec"]
    if defect == "extra-container":
        template["containers"].append(
            {"name": "undeclared", "image": "example.invalid/undeclared"}
        )
    elif defect == "mapping":
        template["containers"] = {"unexpected": "mapping"}
    else:
        ack["spec"]["parallelism"] = "1"
    with pytest.raises(RegionalFixtureError):
        state.acknowledge(ack)
    delete_only_record(state, "job/owned", "uid-owned")
    assert state.rejected_resources(lambda *_: ack) == [ack]


@pytest.mark.parametrize(
    "defect",
    ["api-version", "owner-label", "command", "image", "resources", "parallelism"],
)
def test_creation_ack_must_preserve_every_declared_intent_field(
    tmp_path: Path, defect: str
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    container = ack["spec"]["template"]["spec"]["containers"][0]
    if defect == "api-version":
        ack["apiVersion"] = "unrelated.example/v1"
    elif defect == "owner-label":
        ack["metadata"]["labels"][managed.OWNER_LABEL] = "another-run"
    elif defect == "parallelism":
        ack["spec"]["parallelism"] = 2
    else:
        container[defect] = {
            "command": ["unrelated-workload"],
            "image": "example.invalid/other@sha256:" + "c" * 64,
            "resources": {"limits": {"nvidia.com/gpu": 8}},
        }[defect]
    with pytest.raises(RegionalFixtureError):
        state.acknowledge(ack)
    if defect == "api-version":
        record = state.record.creations["job/owned"].model_dump()
        assert record["uid"] is None and record["ack_sha256"] is None
        assert record["approved"] is False
        with pytest.raises(RegionalFixtureError):
            state.require_acknowledged()
    else:
        digest = delete_only_record(state, "job/owned", "uid-owned")
        with pytest.raises(RegionalFixtureError):
            state.observe(ack, lambda *_: None)
        assert not state.record.observed, (
            "unapproved creation must not enter observed execution custody"
        )
        resumed = custody.FixtureOwnership(state.path, state.binding)
        assert delete_only_record(resumed, "job/owned", "uid-owned") == digest


def test_restart_cannot_backfill_a_lost_create_ack_from_a_current_snapshot(
    tmp_path: Path,
) -> None:
    original = ownership(tmp_path)
    value = intended(original)
    original.begin()
    original.intend(value)
    resumed = custody.FixtureOwnership(original.path, original.binding)
    with pytest.raises(RegionalFixtureError):
        resumed.acknowledge(acknowledgement(value))
    with pytest.raises(RegionalFixtureError, match="unknown"):
        resumed.require_acknowledged()


@pytest.mark.parametrize(
    "defect", ["api-version", "kind", "name", "namespace", "uid", "resource-version"]
)
def test_invalid_ack_identity_never_gains_delete_only_custody(
    tmp_path: Path, defect: str
) -> None:
    state = ownership(tmp_path)
    value = intended(state)
    state.begin()
    state.intend(value)
    ack = acknowledgement(value)
    if defect == "api-version":
        ack["apiVersion"] = "unrelated.example/v1"
    elif defect == "kind":
        ack["kind"] = "Pod"
    else:
        key, replacement = {
            "name": ("name", "another-object"),
            "namespace": ("namespace", "another-namespace"),
            "uid": ("uid", ""),
            "resource-version": ("resourceVersion", None),
        }[defect]
        ack["metadata"][key] = replacement
    with pytest.raises(RegionalFixtureError):
        state.acknowledge(ack)
    creation = state.record.creations["job/owned"].model_dump()
    assert creation["uid"] is None and creation["ack_sha256"] is None
    assert creation["approved"] is False
    with pytest.raises(RegionalFixtureError):
        state.require_acknowledged()
    with pytest.raises(RegionalFixtureError):
        state.observe(acknowledgement(value), lambda *_: None)
    assert not state.record.observed, (
        "invalid creation identity must not be adopted from a later read"
    )


def test_delete_only_fingerprint_survives_restart_and_mutable_api_metadata(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    ack = rejected_creation(state)
    digest = delete_only_record(state, "job/owned", "uid-owned")
    current = copy.deepcopy(ack)
    current["metadata"]["resourceVersion"] = "2"
    current["metadata"]["managedFields"] = [{"manager": "owned-fake-controller"}]
    current["status"] = {"active": 1}
    resumed = custody.FixtureOwnership(state.path, state.binding)
    reads: list[tuple[str, str]] = []

    def read(kind: str, name: str) -> dict[str, Any]:
        reads.append((kind, name))
        return copy.deepcopy(current)

    assert resumed.rejected_resources(read) == [current]
    assert reads == [("job", "owned")]
    assert delete_only_record(resumed, "job/owned", "uid-owned") == digest
    with pytest.raises(RegionalFixtureError):
        resumed.observe(current, read)
    assert not resumed.record.observed, (
        "resumed delete-only custody must not authorize observation"
    )
    assert resumed.rejected_resources(lambda *_: None) == []
    resumed.complete()
    repeated = custody.FixtureOwnership(state.path, state.binding)
    assert repeated.rejected_resources(lambda *_: None) == []
    assert delete_only_record(repeated, "job/owned", "uid-owned") == digest
    assert repeated.record.completed, (
        "delete-only cleanup completion must survive journal reload"
    )


@pytest.mark.parametrize(
    "defect",
    ["api-version", "kind", "name", "namespace", "uid", "spec", "labels", "annotation"],
)
def test_delete_only_cleanup_requires_the_captured_raw_ack_fingerprint(
    tmp_path: Path, defect: str
) -> None:
    state = ownership(tmp_path)
    current = rejected_creation(state)
    digest = delete_only_record(state, "job/owned", "uid-owned")
    if defect == "api-version":
        current["apiVersion"] = "unrelated.example/v1"
    elif defect == "kind":
        current["kind"] = "Pod"
    elif defect == "spec":
        current["spec"]["parallelism"] = 3
    elif defect == "labels":
        current["metadata"]["labels"][managed.OWNER_LABEL] = "a-new-owner"
    elif defect == "annotation":
        current["metadata"]["annotations"] = {"another-controller": "changed"}
    else:
        current["metadata"][defect] = "foreign-" + defect
    current["metadata"]["resourceVersion"] = "2"
    resumed = custody.FixtureOwnership(state.path, state.binding)
    with pytest.raises(RegionalFixtureError):
        resumed.rejected_resources(lambda *_: current)
    if defect in {"kind", "name"}:
        assert resumed.deletion_only(current) is False
    else:
        with pytest.raises(RegionalFixtureError):
            resumed.deletion_only(current)
    assert delete_only_record(resumed, "job/owned", "uid-owned") == digest
    assert not resumed.record.completed, (
        "raw ACK fingerprint drift must not mark cleanup complete"
    )


@pytest.mark.parametrize("api_version", [None, "", "unrelated.example/v1"])
def test_controller_reference_api_group_must_match_the_acknowledged_parent(
    tmp_path: Path, api_version: str | None
) -> None:
    state = ownership(tmp_path)
    parent = prepare(state)
    child = document("child", kind="Pod", uid="uid-child")
    child["metadata"]["ownerReferences"] = owner_reference(parent)
    reference = child["metadata"]["ownerReferences"][0]
    if api_version is None:
        reference.pop("apiVersion")
    else:
        reference["apiVersion"] = api_version
    with pytest.raises(RegionalFixtureError):
        state.observe(child, lambda _kind, _name: parent)
    assert not state.record.observed, (
        "foreign owner API group must not establish child custody"
    )


def test_parent_read_must_match_the_requested_kind_and_name_not_only_its_uid(
    tmp_path: Path,
) -> None:
    state = ownership(tmp_path)
    parent = prepare(state)
    child = document("child", kind="Pod", uid="uid-child")
    child["metadata"]["ownerReferences"] = owner_reference(parent)
    child["metadata"]["ownerReferences"][0]["name"] = "another-parent"
    with pytest.raises(RegionalFixtureError):
        state.observe(child, lambda _kind, _name: parent)
    assert not state.record.observed, (
        "mismatched parent identity must not establish child custody"
    )


def test_controller_chain_may_include_an_owned_jobset_and_job(tmp_path: Path) -> None:
    state = ownership(tmp_path)
    root = document("root", kind="JobSet", uid="uid-root")
    root["apiVersion"] = "jobset.x-k8s.io/v1alpha2"
    state.begin()
    state.intend(root)
    state.acknowledge(root)
    job = document("worker", uid="uid-worker")
    job["metadata"]["ownerReferences"] = owner_reference(root)
    pod = document("pod", kind="Pod", uid="uid-pod")
    pod["metadata"]["ownerReferences"] = owner_reference(job)
    parents = {("job", "worker"): job, ("jobset", "root"): root}
    state.observe(pod, lambda kind, name: parents[(kind, name)])
    resumed = custody.FixtureOwnership(state.path, state.binding)
    resumed.observe(pod, lambda kind, name: parents[(kind, name)])
    resumed.complete()
    resumed.complete()
    assert resumed.record.completed, (
        "repeated completion must preserve the owned controller-chain state"
    )
    assert resumed.record.observed == {"pod/pod": "uid-pod"}


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_creation_receipts_must_be_standard_json(constant: str) -> None:
    with pytest.raises(RegionalFixtureError):
        custody.creation_document('{"spec":{"value":' + constant + "}}")


def test_resumed_managed_cleanup_refuses_changed_declared_workload_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    current = api.objects[(fixture.resource, fixture.name)]
    assert current["spec"]["backoffLimit"] == 0
    current["spec"]["backoffLimit"] = 99
    current["metadata"]["resourceVersion"] = "2"
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    with pytest.raises(RegionalFixtureError):
        resumed.delete()
    assert not api.deletes, "declared workload drift must prevent all cleanup deletions"
    assert (fixture.resource, fixture.name) in api.objects


def test_managed_submission_cannot_replace_a_mismatched_ack_with_a_current_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, _ = managed_harness(tmp_path, monkeypatch)
    original = api.run

    def wrong_ack(command: list[str], **kwargs: Any) -> Any:
        result = original(command, **kwargs)
        if "create" in command:
            value = json.loads(result.stdout)
            assert value["spec"]["backoffLimit"] == 0
            value["spec"]["backoffLimit"] = 99
            result.stdout = json.dumps(value)
        return result

    monkeypatch.setattr(fixture.regional, "run", wrong_ack)
    with pytest.raises(RegionalFixtureError):
        fixture.submit_rendered(rendered)
    current = api.objects[(fixture.resource, fixture.name)]
    assert current["spec"]["backoffLimit"] == 0
    assert fixture.ownership is not None
    delete_only_record(
        fixture.ownership,
        fixture.resource + "/" + fixture.name,
        current["metadata"]["uid"],
    )
    with pytest.raises(RegionalFixtureError):
        fixture.delete()
    assert not api.deletes, (
        "current GET must not authorize deletion against a mismatched CREATE ACK"
    )


@pytest.mark.parametrize("resume", [False, True])
def test_unapproved_managed_creation_is_delete_only_and_cleanup_is_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)

    def change(current: dict[str, Any]) -> None:
        current["spec"]["backoffLimit"] = 99

    mutate_create_reply(monkeypatch, fixture.regional, api, change)
    with pytest.raises(RegionalFixtureError):
        fixture.submit_rendered(rendered)
    current = api.objects[(fixture.resource, fixture.name)]
    assert fixture.ownership is not None
    digest = delete_only_record(
        fixture.ownership,
        fixture.resource + "/" + fixture.name,
        current["metadata"]["uid"],
    )
    target = (
        managed.ManagedWorkloadFixture(
            fixture.regional, fixture.settings, state_path=path
        )
        if resume
        else fixture
    )
    with pytest.raises(RegionalFixtureError):
        target.workload()
    current["metadata"]["resourceVersion"] = "2"
    current["status"] = {"failed": 1}
    target.delete()
    assert not api.objects and len(api.deletes) == 1
    assert api.deletes[0][2]["preconditions"] == {
        "uid": current["metadata"]["uid"],
        "resourceVersion": "2",
    }
    repeated = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    repeated.delete()
    assert len(api.deletes) == 1
    assert repeated.ownership is not None
    assert (
        delete_only_record(
            repeated.ownership,
            fixture.resource + "/" + fixture.name,
            current["metadata"]["uid"],
        )
        == digest
    )
    assert repeated.ownership.record.completed, (
        "delete-only workload cleanup must remain completed on repeat"
    )


@pytest.mark.parametrize("defect", ["uid", "spec", "labels"])
def test_unapproved_managed_cleanup_refuses_drift_without_any_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    fixture, api, rendered, path = managed_harness(tmp_path, monkeypatch)

    def change(current: dict[str, Any]) -> None:
        current["spec"]["backoffLimit"] = 99

    mutate_create_reply(monkeypatch, fixture.regional, api, change)
    with pytest.raises(RegionalFixtureError):
        fixture.submit_rendered(rendered)
    current = api.objects[(fixture.resource, fixture.name)]
    if defect == "uid":
        current["metadata"]["uid"] = "uid-foreign"
    elif defect == "spec":
        current["spec"]["backoffLimit"] = 0
    else:
        current["metadata"]["labels"][managed.OWNER_LABEL] = "foreign"
    current["metadata"]["resourceVersion"] = "2"
    resumed = managed.ManagedWorkloadFixture(
        fixture.regional, fixture.settings, state_path=path
    )
    with pytest.raises(RegionalFixtureError):
        resumed.delete()
    assert api.objects and not api.deletes


@pytest.mark.parametrize("drift", ["none", "approved", "rejected"])
def test_prewarm_validates_all_approved_and_delete_only_targets_before_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    fixture, api, _, _ = managed_harness(tmp_path, monkeypatch)
    path = tmp_path / "prewarm.json"
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )

    def change_second(current: dict[str, Any]) -> None:
        if current["metadata"]["name"].endswith("-1"):
            current["spec"]["activeDeadlineSeconds"] = 1

    mutate_create_reply(monkeypatch, fixture.regional, api, change_second)
    with pytest.raises(RegionalFixtureError):
        prewarm.create(["node-a", "node-b"])
    assert len(api.created) == 2
    first = api.objects[("pod", prewarm.prefix + "-0")]
    second = api.objects[("pod", prewarm.prefix + "-1")]
    assert prewarm.ownership is not None
    assert (
        prewarm.ownership.record.creations[
            "pod/" + first["metadata"]["name"]
        ].model_dump()["approved"]
        is True
    )
    digest = delete_only_record(
        prewarm.ownership,
        "pod/" + second["metadata"]["name"],
        second["metadata"]["uid"],
    )
    if drift != "none":
        changed = first if drift == "approved" else second
        changed["spec"]["activeDeadlineSeconds"] = 99
        changed["metadata"]["resourceVersion"] = "2"
    else:
        second["metadata"]["resourceVersion"] = "2"
        second["status"] = {"phase": "Failed"}
    resumed = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )
    if drift != "none":
        with pytest.raises(RegionalFixtureError):
            resumed.cleanup()
        assert not api.deletes, "one valid resource must not hide another's drift"
        assert len(api.objects) == 2
    else:
        assert not any(resumed.cleanup().values()), (
            "validated prewarm cleanup must report no remaining resources"
        )
        assert not api.objects and len(api.deletes) == 2
        repeated = managed.ImagePrewarmFixture(
            fixture.regional, case_id="case", run_id="run", state_path=path
        )
        assert not any(repeated.cleanup().values()), (
            "repeated prewarm cleanup must report no remaining resources"
        )
        assert len(api.deletes) == 2
        assert repeated.ownership is not None
        assert (
            delete_only_record(
                repeated.ownership,
                "pod/" + second["metadata"]["name"],
                second["metadata"]["uid"],
            )
            == digest
        )
        assert repeated.ownership.record.completed, (
            "prewarm cleanup completion must persist across repeat runs"
        )


def test_managed_cleanup_never_adopts_a_different_api_group_owner_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, rendered, _ = managed_harness(tmp_path, monkeypatch)
    fixture.submit_rendered(rendered)
    source = api.objects[(fixture.resource, fixture.name)]
    child = document("foreign-pod", kind="Pod", uid="uid-foreign-pod")
    child["metadata"]["labels"] = copy.deepcopy(
        source["spec"]["template"]["metadata"]["labels"]
    )
    child["metadata"]["ownerReferences"] = owner_reference(source)
    child["metadata"]["ownerReferences"][0]["apiVersion"] = "unrelated.example/v1"
    api.add(child)
    with pytest.raises(RegionalFixtureError):
        fixture.delete()
    assert not api.deletes, (
        "foreign owner API group must prevent all managed cleanup deletions"
    )
    assert ("pod", "foreign-pod") in api.objects


def test_resumed_prewarm_cleanup_refuses_changed_declared_execution_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, api, _, _ = managed_harness(tmp_path, monkeypatch)
    path = tmp_path / "prewarm.json"
    prewarm = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )
    prewarm.create(["node-a"])
    current = next(iter(api.objects.values()))
    current["spec"]["activeDeadlineSeconds"] = 1
    current["metadata"]["resourceVersion"] = "2"
    resumed = managed.ImagePrewarmFixture(
        fixture.regional, case_id="case", run_id="run", state_path=path
    )
    with pytest.raises(RegionalFixtureError):
        resumed.cleanup()
    assert not api.deletes, "prewarm deadline drift must prevent cleanup deletions"
    assert api.objects, "resources must remain present after prewarm cleanup refusal"


@pytest.mark.parametrize("field", ["uid", "owner"])
def test_cleanup_rechecks_node_identity_before_reactivating_an_agent(
    field: str,
) -> None:
    root = {
        "incident_id": "incident-owned",
        "cluster_id": "cluster-a",
        "node_ids": ["node-a"],
        "job_id": "job-owned",
        "attempt_id": "attempt-owned",
    }
    original: dict[str, Any] = {
        "uid": "uid-owned",
        "annotations": {"gpu-fault.io/incident-id": "incident-owned"},
    }
    replaced = copy.deepcopy(original)
    if field == "uid":
        replaced["uid"] = "uid-foreign"
    else:
        replaced["annotations"]["gpu-fault.io/incident-id"] = "incident-foreign"
    calls: list[str] = []
    snapshots = 0

    def snapshot(node: str) -> dict[str, Any]:
        nonlocal snapshots
        assert node == "node-a"
        snapshots += 1
        return copy.deepcopy(original if snapshots == 1 else replaced)

    warm = SimpleNamespace(
        node_snapshot=snapshot,
        incident_by_id=lambda name: copy.deepcopy(root),
        release_spares=lambda *args: calls.append("spares.release"),
        reactivate_agent=lambda *args: calls.append("agent.reactivate"),
        wait_agent_active=lambda *args: calls.append("agent.active"),
        create_restore_workflow=lambda **kwargs: pytest.fail(
            "foreign identity reached restore creation"
        ),
    )
    result = case.restore_fault_node(
        cast(WarmSpareLiveFixture, warm),
        settings=cast(
            case.Settings,
            SimpleNamespace(
                fault_node="node-a",
                spare_node="node-b",
                regional=SimpleNamespace(cluster_id="cluster-a"),
            ),
        ),
        incident_id="incident-owned",
        profile_version="profile-v1",
    )
    assert result["errors"], "changed Node/owner must stop cleanup"
    assert "agent.reactivate" not in calls, (
        "foreign identity was checked only after changing agent lifecycle"
    )


@pytest.mark.parametrize("owner", ["", "incident-owned"])
def test_cleanup_node_accepts_only_the_unchanged_recorded_identity(owner: str) -> None:
    snapshot = {
        "uid": "uid-owned",
        "annotations": {"gpu-fault.io/incident-id": owner} if owner else {},
    }
    reads: list[str] = []

    def read(node: str) -> dict[str, Any]:
        reads.append(node)
        return copy.deepcopy(snapshot)

    cleanup.require_cleanup_node(
        cast(WarmSpareLiveFixture, SimpleNamespace(node_snapshot=read)),
        node="node-a",
        uid="uid-owned",
        owner=owner,
    )
    assert reads == ["node-a"]


def test_cleanup_family_does_not_follow_unreadable_workflow_ancestry() -> None:
    root = {
        "incident_id": "incident-owned",
        "cluster_id": "cluster-a",
        "node_ids": ["node-a"],
        "job_id": "job-owned",
        "attempt_id": "attempt-owned",
    }
    owner = "inc-support-after-workflow-owned"
    records = {
        "incident-owned": root,
        owner: {**root, "incident_id": owner, "event_id": owner.removeprefix("inc-")},
    }
    reads = []

    def unavailable(name: str, *, timeout_seconds: int) -> dict[str, Any]:
        reads.append((name, timeout_seconds))
        raise RegionalFixtureError("owned fake workflow read failed")

    warm = SimpleNamespace(
        incident_by_id=lambda name: copy.deepcopy(records[name]),
        wait_workflow_id=unavailable,
    )
    with pytest.raises(RegionalFixtureError, match="workflow read failed"):
        cleanup.require_cleanup_family(
            cast(WarmSpareLiveFixture, warm),
            incident_id="incident-owned",
            owner=owner,
            cluster_id="cluster-a",
            node="node-a",
            profile_version="profile-v1",
        )
    assert reads == [("workflow-owned", 1)]
