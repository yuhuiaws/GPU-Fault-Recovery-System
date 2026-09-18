"""Remaining admission, crash, and deletion-only controller boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.host_health import NodeHealthIngestionResult
from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api


@pytest.fixture
def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    return build_api(tmp_path, monkeypatch)


@pytest.mark.parametrize("field", ["release_requested", "stopped", "cleanup_seconds"])
def test_inconsistent_saved_job_transitions_are_refused(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], field: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    value = api.journal()
    value["jobs"][0][field] = {
        "release_requested": False,
        "stopped": True,
        "cleanup_seconds": 10,
    }[field]
    write_json_atomic(watchdog.path, value)
    with pytest.raises(RegionalFixtureError, match="journal"):
        lifecycle.has_saved_plan(api.directory, plan.run_id)


def test_missing_configuration_file_is_not_a_bound_connection(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.regional.settings.cpu_kubeconfig.unlink()
    with pytest.raises(RegionalFixtureError, match="incomplete"):
        api.watchdog(plan, runtime)
    assert not api.calls, "incomplete connection identity must precede all API work"


def test_missing_ownership_proof_blocks_even_the_initial_journal_write(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    monkeypatch.setattr(locking, "ownership_held", lambda _path: False)
    with pytest.raises(RegionalFixtureError, match="ownership"):
        api.watchdog(plan, runtime)
    assert not api.calls, (
        "local ownership is a prerequisite, not an optional annotation"
    )


def test_oversized_captured_source_does_not_create_an_unreadable_journal(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    (api.source / wire.SOURCE_FILES[0]).write_text(
        "#" + "x" * lifecycle.MAX_DOCUMENT_BYTES
    )
    plan = plan.model_copy(update={"probe_sha256": wire.source_sha256(api.source)})
    with pytest.raises(RegionalFixtureError, match="bounded size"):
        api.watchdog(plan, runtime)
    assert not api.calls, (
        "oversized local state must fail before creating remote resources"
    )


def test_ready_but_changed_config_identity_blocks_execution_and_cleanup(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    api.objects["configmap", "worker-postgres"]["metadata"]["resourceVersion"] = "11"
    with pytest.raises(RegionalFixtureError, match="CPU runtime changed"):
        watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="configuration changed"):
        watchdog.cleanup()


@pytest.mark.parametrize("response", ["[]", "x" * 262145])
def test_full_control_response_must_be_a_bounded_object(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    original = api.kube

    def changed(plane: str, *args: str, **kwargs: Any) -> str:
        if args[:3] == ("get", "configmap", watchdog.name):
            return response
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(api.regional, "kubectl", changed)
    with pytest.raises(RegionalFixtureError, match="response"):
        watchdog.control_client()


@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_all_run_names_are_preflighted_before_creating_any_reference(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    api.objects[kind, watchdog.name] = {
        "metadata": {
            "name": watchdog.name,
            "uid": "preexisting",
            "resourceVersion": "1",
        }
    }
    with pytest.raises(RegionalFixtureError, match="already occupied"):
        watchdog.arm()
    assert not [
        call for call in api.calls if call[0] in {"create", "patch", "delete"}
    ], "an unowned Job must not be given freshly created ConfigMaps or RBAC"


def test_a_name_race_between_preflight_and_create_is_also_refused(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    original = api.kube
    reads = 0

    def raced(plane: str, *args: str, **kwargs: Any) -> str:
        nonlocal reads
        if args[:3] == ("get", "configmap", watchdog.name):
            reads += 1
            if reads == 2:
                return '{"metadata":{"uid":"foreign"}}'
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(api.regional, "kubectl", raced)
    with pytest.raises(RegionalFixtureError, match="already occupied"):
        watchdog.arm()
    assert not [call for call in api.calls if call[0] == "create"], (
        "late name collisions must not be adopted"
    )


def test_deletion_only_support_cannot_be_reused_by_arm_or_cleanup_observer(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.create_changes["serviceaccount"] = {"automountServiceAccountToken": True}
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="deletion-only"):
        watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="approved for execution"):
        watchdog.resume_cleanup(seconds=10)
    assert not api.journal()["jobs"], "unapproved support cannot start any observer"
    watchdog.cleanup()


@pytest.mark.parametrize("kind", ["serviceaccount", "role", "rolebinding"])
def test_disappeared_acknowledged_support_is_not_recreated(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    del api.objects[kind, watchdog.name]
    creates = len([call for call in api.calls if call[0] == "create"])
    with pytest.raises(RegionalFixtureError, match="disappeared"):
        watchdog.arm()
    assert len([call for call in api.calls if call[0] == "create"]) == creates


@pytest.mark.parametrize("missing", ["control", "support-record"])
def test_incomplete_resource_journal_cannot_validate_a_running_observer(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], missing: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    saved = api.journal()
    key = ("configmap/" if missing == "control" else "role/") + watchdog.name
    del saved["support"][key]
    write_json_atomic(watchdog.path, saved)
    expected_error = (
        "private journal is invalid"
        if missing == "control"
        else "supporting resource is missing"
    )
    with pytest.raises(RegionalFixtureError, match=expected_error):
        watchdog.validate_running()


def test_missing_control_object_cannot_be_recovered_from_saved_labels(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    del api.objects["configmap", watchdog.name]
    with pytest.raises(RegionalFixtureError, match="disappeared"):
        watchdog.control_client()


@pytest.mark.parametrize("change", ["missing", "deleting", "unapproved-plan"])
def test_current_job_must_remain_present_and_exact(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    if change == "missing":
        del api.objects["job", watchdog.name]
    elif change == "deleting":
        api.objects["job", watchdog.name]["metadata"]["deletionTimestamp"] = (
            api.clock.timestamp()
        )
    else:
        saved = api.journal()
        saved["jobs"][0]["expected"]["spec"]["backoffLimit"] = 2
        write_json_atomic(watchdog.path, saved)
    with pytest.raises(RegionalFixtureError, match="disappeared|deleting|approved"):
        watchdog.validate_running()


def test_stopped_observer_cannot_be_awaited_without_explicit_cleanup_resumption(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        watchdog.cleanup()
    with pytest.raises(RegionalFixtureError, match="unique current"):
        watchdog.wait_quiescence()


class ParentGone(BaseException):
    pass


def test_restart_after_gate_cas_uses_saved_pod_and_does_not_release_twice(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    original = api.kube

    def interrupted(plane: str, *args: str, **kwargs: Any) -> str:
        value = original(plane, *args, **kwargs)
        if args[:2] == ("patch", "pod"):
            raise ParentGone()
        return value

    monkeypatch.setattr(api.regional, "kubectl", interrupted)
    with pytest.raises(ParentGone):
        watchdog.arm()
    saved = api.journal()["jobs"][0]
    assert saved["release_requested"] and not saved["release_confirmed"], (
        "the gate intent and exact Pod UID must survive a disappearing parent"
    )
    monkeypatch.setattr(api.regional, "kubectl", original)
    api.watchdog(plan, runtime).arm()
    assert (
        len([call for call in api.calls if call[0] == "patch" and call[1][1] == "pod"])
        == 1
    ), "resumption must confirm the prior CAS rather than authorize a new Pod"


def test_unconfirmed_gate_record_cannot_claim_running_authority(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    saved = api.journal()
    saved["jobs"][0]["release_confirmed"] = False
    write_json_atomic(watchdog.path, saved)
    with pytest.raises(RegionalFixtureError, match="verified released"):
        watchdog.validate_running()


@pytest.mark.parametrize("mode", ["pending", "ended"])
def test_startup_waits_for_running_and_refuses_a_pod_that_ended(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    api, plan, runtime = setup
    original = api.kube
    sleep = api.clock.sleep

    def changed(plane: str, *args: str, **kwargs: Any) -> str:
        result = original(plane, *args, **kwargs)
        if args[:2] == ("patch", "pod"):
            pod = api.objects["pod", args[2]]
            pod["status"]["phase"] = "Pending" if mode == "pending" else "Failed"
            if mode == "pending":
                api.objects["job", pod["metadata"]["ownerReferences"][0]["name"]][
                    "status"
                ]["ready"] = 0
        return result

    def ready(seconds: float) -> None:
        sleep(seconds)
        for (kind, _), pod in api.objects.items():
            if kind == "pod" and pod["metadata"]["ownerReferences"][0]["kind"] == "Job":
                pod["status"]["phase"] = "Running"
                api.objects["job", pod["metadata"]["ownerReferences"][0]["name"]][
                    "status"
                ]["ready"] = 1

    monkeypatch.setattr(api.regional, "kubectl", changed)
    monkeypatch.setattr(api.clock, "sleep", ready)
    watchdog = api.watchdog(plan, runtime)
    if mode == "ended":
        with pytest.raises(RegionalFixtureError, match="ended before arming"):
            watchdog.arm()
    else:
        watchdog.arm()
        assert api.clock.elapsed >= 1, (
            "readiness must be observed after the pending state"
        )


@pytest.mark.parametrize(
    "change",
    [
        {
            "apiVersion": "v1",
            "kind": "PodList",
            "metadata": {"resourceVersion": "1", "continue": "next"},
            "items": [],
        },
        {
            "apiVersion": "v1",
            "kind": "PodList",
            "metadata": {"resourceVersion": "1"},
            "items": [
                {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "metadata": {"namespace": "foreign", "ownerReferences": []},
                }
            ],
        },
    ],
)
def test_observer_discovery_remains_complete_after_worker_capability_passed(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: dict[str, Any]
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    api.list_override = change
    with pytest.raises(RegionalFixtureError, match="discovery"):
        watchdog.wait_quiescence()


def test_cleanup_wait_does_not_accept_an_earlier_observer_receipt(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    original = api.kube
    sleep = api.clock.sleep

    def delayed(plane: str, *args: str, **kwargs: Any) -> str:
        result = original(plane, *args, **kwargs)
        if args[:2] == ("patch", "pod"):
            api.frozen_status = True
        return result

    def unfreeze(seconds: float) -> None:
        sleep(seconds)
        api.frozen_status = False

    monkeypatch.setattr(api.regional, "kubectl", delayed)
    watchdog.sleep = unfreeze
    receipt = watchdog.resume_cleanup(seconds=10)
    assert receipt.cleanup is not None and receipt.sequence > 1, (
        "only a receipt from the new cleanup observer may authorize closure"
    )


@pytest.mark.parametrize("change", ["attempt", "window"])
def test_cleanup_receipt_must_match_the_current_job_attempt(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    original = api.kube

    def wrong(plane: str, *args: str, **kwargs: Any) -> str:
        result = original(plane, *args, **kwargs)
        if args[:3] == ("get", "configmap", watchdog.name):
            value = json.loads(result)
            if value["data"]["status.json"] != "null":
                receipt = wire.decode(wire.Receipt, value["data"]["status.json"])
                if receipt.cleanup is not None:
                    updates: dict[str, Any] = (
                        {"attempt_id": "foreign-attempt"}
                        if change == "attempt"
                        else {"deadline_at": receipt.cleanup.started_at + 15}
                    )
                    value["data"]["status.json"] = wire.encode(
                        receipt.model_copy(
                            update={
                                "cleanup": receipt.cleanup.model_copy(update=updates)
                            }
                        )
                    )
                    return json.dumps(value)
        return result

    monkeypatch.setattr(api.regional, "kubectl", wrong)
    with pytest.raises(RegionalFixtureError, match="earlier observer"):
        watchdog.resume_cleanup(seconds=10)
    assert watchdog.record.quiescence is None, (
        "an unrelated cleanup receipt must not become retirement proof"
    )


@pytest.mark.parametrize("state", ["failed-receipt", "failed-pod", "still-monitoring"])
def test_quiescence_requires_successful_observer_and_stopped_monitoring(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], state: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    api.auto_quiet = state == "still-monitoring"
    control.request_close()
    api.progress()
    api.frozen_status = True
    snapshot = control.read()
    assert snapshot.receipt is not None, "the fixture needs a parsed observer receipt"
    if state == "failed-receipt":
        receipt = wire.receipt(
            plan,
            snapshot.control,
            snapshot.receipt,
            uid=control.uid,
            now=int(api.clock.now()),
            state="FAILED",
            error_code="FIXTURE_FAILURE",
            monitoring=False,
        )
        api.objects["configmap", watchdog.name]["data"]["status.json"] = wire.encode(
            receipt
        )
    elif state == "failed-pod":
        pod_name = api.journal()["jobs"][0]["pod"]["name"]
        api.objects["pod", pod_name]["status"]["phase"] = "Failed"
    else:
        api.objects["configmap", watchdog.name]["data"]["status.json"] = wire.encode(
            snapshot.receipt.model_copy(update={"monitoring": True})
        )
    expected_error = (
        "control schema validation failed"
        if state == "still-monitoring"
        else "quiescence|terminate"
    )
    with pytest.raises(RegionalFixtureError, match=expected_error):
        watchdog.wait_quiescence()
    assert watchdog.record.quiescence is None, (
        "failure must not publish terminal quiescence"
    )


@pytest.mark.parametrize("invalid", ["[]", "not-json", "new-claim"])
def test_cleanup_cpu_boundary_refuses_new_producer_authority(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], invalid: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.cleanup()
    payload = invalid
    if invalid == "new-claim":
        before = wire.initial_control(plan)
        changed = wire.claim_submission(
            plan, before, claim_id="forbidden", now=int(api.clock.now())
        )
        payload = wire.encode(
            [
                {
                    "op": "test",
                    "path": "/data/control.json",
                    "value": wire.encode(before),
                },
                {
                    "op": "replace",
                    "path": "/data/control.json",
                    "value": wire.encode(changed),
                },
            ]
        )
    mutations = [call for call in api.calls if call[0] != "get"]
    with pytest.raises(RegionalFixtureError, match="producer authority"):
        watchdog.cpu(
            "patch",
            "configmap",
            watchdog.name,
            "--type=json",
            stdin=payload.encode(),
            timeout=30,
        )
    assert [call for call in api.calls if call[0] != "get"] == mutations, (
        "cleanup's local tombstone must stop producer claims before the API"
    )


def test_late_ack_is_allowed_during_cleanup_but_never_a_second_claim(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    claim = control.claim()
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        watchdog.cleanup()
    response = {
        "status": 200,
        "body": NodeHealthIngestionResult(
            batch_id=plan.event_id,
            duplicate=False,
            incident_ids=["incident-a"],
            workflow_request_ids=["workflow-a"],
        ).model_dump(mode="json"),
    }
    control.acknowledge(claim, response)
    with pytest.raises(RegionalFixtureError):
        control.claim()
    proof = watchdog.resume_cleanup(seconds=10)
    assert proof.root is not None and proof.root.workflow_request_id == "workflow-a", (
        "a matching late ACK may resolve the old source but not create a new one"
    )
    assert proof.case_failed and proof.producer_revoked, (
        "cleanup remains irrevocably failed/closed"
    )


@pytest.mark.parametrize("mode", ["lost-ack", "refused", "delayed", "stuck"])
def test_unapproved_ack_deletion_has_its_own_bounded_confirmation(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    api, plan, runtime = setup
    api.create_changes["configmap"] = {"metadata/labels": {}}
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    if mode == "lost-ack":
        api.lost_delete.add("configmap")
    elif mode == "refused":
        api.failed_delete.add("configmap")
    else:
        original = api.kube
        requested = False
        reads = 0

        def pending(plane: str, *args: str, **kwargs: Any) -> str:
            nonlocal requested, reads
            if args[0] == "delete":
                options = json.loads(kwargs["input_text"])
                item = api.objects["configmap", watchdog.name]
                assert options["preconditions"] == {
                    "uid": item["metadata"]["uid"],
                    "resourceVersion": item["metadata"]["resourceVersion"],
                }, "deletion-only authority must still use UID/RV preconditions"
                item["metadata"]["deletionTimestamp"] = api.clock.timestamp()
                requested = True
                return "{}"
            if requested and args[:3] == ("get", "configmap", watchdog.name):
                reads += 1
                if mode == "delayed" and reads == 3:
                    del api.objects["configmap", watchdog.name]
            return original(plane, *args, **kwargs)

        monkeypatch.setattr(api.regional, "kubectl", pending)
    if mode in {"refused", "stuck"}:
        with pytest.raises(RegionalFixtureError, match="acknowledged|disappear"):
            watchdog.cleanup()
        assert not watchdog.record.closed, (
            "unconfirmed deletion must keep the journal open"
        )
    else:
        watchdog.cleanup()
        assert watchdog.record.closed, "only actual disappearance confirms deletion"


def test_cleanup_budget_exhaustion_never_creates_another_job(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    monkeypatch.setattr(lifecycle, "MAX_JOBS", 1)
    creates = len([call for call in api.calls if call[0] == "create"])
    with pytest.raises(RegionalFixtureError, match="budget"):
        watchdog.resume_cleanup(seconds=10)
    assert len([call for call in api.calls if call[0] == "create"]) == creates, (
        "the bounded journal cannot overflow by silently adding an observer"
    )


def test_cleanup_observer_name_collision_is_not_adopted(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    name = watchdog.name + "-c1"
    api.objects["job", name] = {
        "metadata": {"name": name, "uid": "foreign", "resourceVersion": "1"}
    }
    with pytest.raises(RegionalFixtureError, match="already occupied"):
        watchdog.resume_cleanup(seconds=10)
    assert len(watchdog.record.jobs) == 1, (
        "an unacknowledged foreign Job must not enter the journal"
    )


def test_post_delete_full_read_must_also_confirm_absence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    old = copy.deepcopy(api.objects["job", watchdog.name])
    original = api.kube
    deleted = False

    def inconsistent(plane: str, *args: str, **kwargs: Any) -> str:
        nonlocal deleted
        if deleted and args[:3] == ("get", "job", watchdog.name) and args[-1] == "json":
            return json.dumps(old)
        result = original(plane, *args, **kwargs)
        if args[0] == "delete" and "/jobs/" in args[2]:
            deleted = True
        return result

    monkeypatch.setattr(api.regional, "kubectl", inconsistent)
    with pytest.raises(RegionalFixtureError, match="deletion was not confirmed"):
        watchdog.cleanup()
    assert not watchdog.record.closed, (
        "a metadata-only absence must not replace the full confirmation"
    )


def test_pod_cessation_confirmation_must_fit_its_own_deadline(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    api.keep_pods = True
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    original = api.kube

    def slow(plane: str, *args: str, **kwargs: Any) -> str:
        result = original(plane, *args, **kwargs)
        if args[0] == "delete" and "/pods/" in args[2]:
            api.clock.sleep(lifecycle.STOP_SECONDS + 1)
        return result

    monkeypatch.setattr(api.regional, "kubectl", slow)
    with pytest.raises(RegionalFixtureError, match="cessation was not confirmed"):
        watchdog.cleanup()
    assert ("serviceaccount", watchdog.name) in api.objects, (
        "the API references must survive an unconfirmed cessation deadline"
    )


def test_never_armed_but_submitting_source_cannot_be_retired(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.auto_arm = False
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="deadline"):
        watchdog.arm()
    value = api.objects["configmap", watchdog.name]
    control = wire.decode(wire.Control, value["data"]["control.json"])
    control = wire.claim_submission(
        plan, control, claim_id="unresolved", now=int(api.clock.now())
    )
    value["data"]["control.json"] = wire.encode(control)
    with pytest.raises(RegionalFixtureError, match="unresolved producer"):
        watchdog.cleanup()
    assert ("configmap", watchdog.name) in api.objects, (
        "an unresolved source must retain its control record"
    )


@pytest.mark.parametrize("kind", ["role", "configmap"])
def test_already_absent_acknowledged_support_is_confirmed_without_recreation(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    name = watchdog.name + "-code" if kind == "configmap" else watchdog.name
    del api.objects[kind, name]
    creates = len([call for call in api.calls if call[0] == "create"])
    watchdog.cleanup()
    assert watchdog.record.closed, (
        "confirmed absence can retire an already acknowledged resource"
    )
    assert len([call for call in api.calls if call[0] == "create"]) == creates, (
        "cleanup must not reconstruct a vanished supporting resource"
    )


def test_running_pod_full_spec_drift_is_rejected_by_the_admission_fingerprint(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    api.frozen_status = True
    name = api.journal()["jobs"][0]["pod"]["name"]
    api.objects["pod", name]["spec"]["containers"][0]["env"].append(
        {"name": "UNAPPROVED_ENV", "value": "fixture-only"}
    )
    with pytest.raises(
        RegionalFixtureError, match="Pod changed after its recorded admission"
    ):
        watchdog.validate_running()
    with pytest.raises(
        RegionalFixtureError, match="Pod changed after its recorded admission"
    ):
        watchdog.cleanup()
    assert ("job", watchdog.name) in api.objects, (
        "labels and UID cannot substitute for the originally verified complete spec"
    )
