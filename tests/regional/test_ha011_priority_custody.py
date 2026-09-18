from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import ha011_manifests as manifests
from scripts.e2e.regional import ha011_resources as resources
from scripts.e2e.regional import run_ha011_busy_cpu_takeover as runner
from scripts.e2e.regional.regional_commands import RegionalCommandTimeout
from tests.regional._cov95_ha011_support import IMAGE_ID, Clock
from tests.regional.test_cov95_ha011_resources import prepare


def test_priority_class_is_owned_zero_never_and_deleted_after_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    priority = fake.object("PriorityClass", settings.priority_class_name)
    namespace = fake.object("Namespace", settings.isolated_namespace)
    assert priority["metadata"]["ownerReferences"] == [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "name": settings.isolated_namespace,
            "uid": namespace["metadata"]["uid"],
            "controller": False,
            "blockOwnerDeletion": False,
        }
    ]
    assert priority["value"] == 0
    assert resources.canonical_resource(priority)["globalDefault"] is False
    assert priority["preemptionPolicy"] == "Never"
    pod = fake.object("Pod", contracts.POD_NAME)
    assert pod["spec"]["priorityClassName"] == settings.priority_class_name
    assert pod["spec"]["preemptionPolicy"] == "Never"
    lifetime.arm(IMAGE_ID)
    result = lifetime.cleanup()
    assert result["namespace_absent"] is result["priority_class_absent"] is True
    assert ("PriorityClass", None, settings.priority_class_name) not in fake.objects
    deletes = [args[2] for _namespace, args in fake.calls if args[0] == "delete"]
    assert deletes == [
        "/api/v1/namespaces/" + settings.isolated_namespace,
        "/apis/scheduling.k8s.io/v1/priorityclasses/" + settings.priority_class_name,
    ]


@pytest.mark.parametrize("when", ["preflight", "start"])
def test_existing_priority_class_is_never_adopted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, when: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    existing = manifests.priority_class_manifest(settings, "foreign-namespace-uid")
    existing["metadata"]["uid"] = "foreign-priority"
    fake.put(existing)
    if when == "preflight":
        result = runner.read_only_preflight(settings, tmp_path)
        assert any("PriorityClass" in error for error in result["errors"]), (
            'test_existing_priority_class_is_never_adopted: expected any("PriorityClass" in error for error in result["errors"])'
        )
    else:
        with pytest.raises(contracts.ProofError, match="PriorityClass"):
            lifetime.start(objects)
    assert fake.created == []
    assert not fake.deleted, (
        "test_existing_priority_class_is_never_adopted: expected no fake.deleted"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "uid",
        "namespace-owner",
        "owner-label",
        "priority",
        "global",
        "policy",
        "finalizer",
    ],
)
def test_priority_drift_never_reaches_arm_or_authorizes_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    priority = fake.object("PriorityClass", settings.priority_class_name)
    if defect == "uid":
        priority["metadata"]["uid"] = "replacement"
    elif defect == "namespace-owner":
        priority["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif defect == "owner-label":
        priority["metadata"]["labels"][contracts.LABEL] = "foreign"
    elif defect == "priority":
        priority["value"] = 100
    elif defect == "global":
        priority["globalDefault"] = True
    elif defect == "policy":
        priority["preemptionPolicy"] = "PreemptLowerPriority"
    else:
        priority["metadata"]["finalizers"] = ["unapproved/hold"]
    with pytest.raises(contracts.ProofError):
        lifetime.arm(IMAGE_ID)
    assert not fake.armed, (
        "test_priority_drift_never_reaches_arm_or_authorizes_cleanup: expected no fake.armed"
    )
    with pytest.raises(contracts.ProofError):
        lifetime.cleanup()
    assert not fake.deleted, (
        "test_priority_drift_never_reaches_arm_or_authorizes_cleanup: expected no fake.deleted"
    )


def test_acknowledged_priority_uid_survives_failed_post_create_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    original = fake.dispatch
    failed = False

    def dispatch(args, namespace, text):
        nonlocal failed
        if (
            not failed
            and args[:2] == ["get", "PriorityClass"]
            and ("PriorityClass", None, settings.priority_class_name) in fake.objects
        ):
            failed = True
            raise RegionalCommandTimeout(args, 1)
        return original(args, namespace, text)

    fake.dispatch = dispatch
    with pytest.raises(RegionalCommandTimeout):
        lifetime.start(objects)
    assert lifetime.receipts["PriorityClass/" + settings.priority_class_name] == (
        "owned-priorityclass"
    )
    assert lifetime.cleanup()["priority_class_absent"] is True


def test_namespace_absence_does_not_skip_owned_priority_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    fake.objects.pop(("Namespace", None, settings.isolated_namespace))
    assert lifetime.cleanup()["priority_class_absent"] is True
    assert ("PriorityClass", None, settings.priority_class_name) not in fake.objects


@pytest.mark.parametrize("phase", ["gone", "deleting"])
def test_namespace_gc_of_the_owned_class_is_observed_without_name_deletion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    original = fake.dispatch
    key = ("PriorityClass", None, settings.priority_class_name)
    deleting_reads = 0

    def dispatch(args, namespace, text):
        nonlocal deleting_reads
        if args[:2] == ["get", "PriorityClass"] and key in fake.objects:
            if fake.objects[key]["metadata"].get("deletionTimestamp"):
                deleting_reads += 1
                if deleting_reads >= 2:
                    fake.objects.pop(key)
        answer = original(args, namespace, text)
        if args[0] == "delete" and args[2].startswith("/api/v1/namespaces/"):
            if phase == "gone":
                fake.objects.pop(key)
            else:
                fake.objects[key]["metadata"]["deletionTimestamp"] = "deleting"
        return answer

    fake.dispatch = dispatch
    assert lifetime.cleanup()["priority_class_absent"] is True
    assert sum(args[0] == "delete" for _namespace, args in fake.calls) == 1


def test_priority_creation_without_ack_is_not_adopted_during_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    original = fake.dispatch

    def dispatch(args, namespace, text):
        answer = original(args, namespace, text)
        if args[0] == "create" and answer == "owned-priorityclass":
            raise RegionalCommandTimeout(args, 1)
        return answer

    fake.dispatch = dispatch
    with pytest.raises(RegionalCommandTimeout):
        lifetime.start(objects)
    with pytest.raises(contracts.ProofError, match="custody"):
        lifetime.cleanup()
    assert not fake.deleted, (
        "test_priority_creation_without_ack_is_not_adopted_during_cleanup: expected no fake.deleted"
    )


@pytest.mark.parametrize("change", ["replacement", "timeout"])
def test_priority_deletion_wait_rejects_replacement_or_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    settings, fake, _api, lifetime, objects = prepare(monkeypatch, tmp_path)
    lifetime.start(objects)
    original = fake.dispatch
    retained = copy.deepcopy(fake.object("PriorityClass", settings.priority_class_name))

    def dispatch(args, namespace, text):
        answer = original(args, namespace, text)
        if args[0] == "delete" and "/priorityclasses/" in args[2]:
            if change == "replacement":
                retained["metadata"]["uid"] = "replacement"
            fake.put(retained)
        return answer

    fake.dispatch = dispatch
    monkeypatch.setattr(resources, "time", Clock(step=2))
    with pytest.raises(contracts.ProofError, match="PriorityClass"):
        lifetime.cleanup()
