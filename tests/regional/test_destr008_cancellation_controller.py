"""Private controller lifecycle against a fake CPU API, not a live runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api


@pytest.fixture
def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    return build_api(tmp_path, monkeypatch)


def test_arm_close_quiescence_and_control_last_cleanup(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    assert not lifecycle.has_saved_plan(api.directory, plan.run_id), (
        "an absent journal cannot be treated as an authorized previous run"
    )
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    assert control.assert_armed().state == "ARMED", (
        "the real control client must see a fresh receipt"
    )
    watchdog.validate_running()
    assert lifecycle.has_saved_plan(api.directory, plan.run_id), (
        "arming must have durable identity"
    )
    assert lifecycle.load_saved_plan(api.directory, plan.run_id) == (plan, runtime), (
        "resumption must recover the original plan and runtime"
    )
    assert watchdog.control_client().identity() == control.identity(), (
        "public access must preserve the original control UID"
    )
    control.request_close()
    receipt = watchdog.wait_quiescence()
    assert receipt.state == "QUIESCENT" and not receipt.case_failed, (
        "ordinary close must require actual successful observer completion"
    )
    watchdog.cleanup()
    watchdog.cleanup()
    assert api.journal()["closed"], "cleanup must be durably complete and idempotent"
    deleted = [args[2] for verb, args, _ in api.calls if verb == "delete"]
    assert "/jobs/" in deleted[0], "observer Jobs must stop before supporting resources"
    assert deleted[-1].endswith("/configmaps/" + watchdog.name), (
        "the mutable control map must be the final deleted resource"
    )
    assert not any(
        kind in {"job", "role", "rolebinding", "serviceaccount"}
        or (kind == "pod" and name.startswith(watchdog.name))
        for kind, name in api.objects
    ), "no owned observer or supporting capability may survive cleanup"
    assert {
        value["metadata"]["uid"]
        for (kind, _), value in api.objects.items()
        if kind == "pod"
    } == {"cpu-worker-0-uid", "cpu-worker-1-uid"}, (
        "cleanup must leave the real source worker population untouched"
    )


def test_claim_converges_while_the_daemon_heartbeat_churns_status(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.heartbeat_churn = True
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    assert control.assert_armed().state == "ARMED", "arming must persist first"
    claim_id = control.claim()
    assert claim_id.startswith("claim-"), "the claim must acquire submission authority"
    submitting = control.read()
    assert submitting.control.producer.state == "SUBMITTING", (
        "the parent CAS must land control.json despite concurrent status heartbeats"
    )
    assert submitting.control.producer.claim_id == claim_id, (
        "the persisted claim must be the one the parent minted"
    )


def test_completed_arm_is_idempotent_without_resetting_control(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    first = watchdog.arm()
    creates = len([call for call in api.calls if call[0] == "create"])
    assert watchdog.arm().identity() == first.identity(), (
        "retries must not replace the control"
    )
    assert len([call for call in api.calls if call[0] == "create"]) == creates, (
        "an acknowledged resource must not be recreated"
    )


def test_lost_gate_patch_ack_is_reconciled_only_for_the_previously_verified_pod(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.gate_lost_ack = True
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    record = api.journal()
    assert record["jobs"][0]["release_confirmed"], (
        "known-UID gate CAS may be verified by readback"
    )
    assert record["jobs"][0]["pod"]["shape_sha256"], (
        "gated admission proof must predate CAS"
    )


@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_lost_creation_ack_is_never_adopted_from_metadata(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    api.lost_create.add(kind)
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    api.lost_create.clear()
    fresh = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        fresh.arm()
    with pytest.raises(RegionalFixtureError):
        fresh.cleanup()
    assert not [call for call in api.calls if call[0] == "delete"], (
        "a lost creation ACK must not be converted into deletion authority"
    )


@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_unapproved_ack_retains_deletion_only_identity(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    changes: dict[str, dict[str, Any]] = {
        "configmap": {"immutable": True},
        "serviceaccount": {"automountServiceAccountToken": True},
        "role": {"rules/0/verbs": ["get", "patch", "create"]},
        "rolebinding": {"roleRef/name": "foreign"},
        "job": {"spec/template/spec/containers/0/image": "unapproved:latest"},
    }
    api.create_changes[kind] = changes[kind]
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    record = api.journal()
    item = (
        record["jobs"][-1]["resource"]
        if kind == "job"
        else record["support"][kind + "/" + watchdog.name]
    )
    assert item["uid"] and item["ack_sha256"] and not item["approved"], (
        "direct ACK custody must survive admission refusal"
    )
    assert not [
        call for call in api.calls if call[0] == "patch" and call[1][1] == "pod"
    ], "an unapproved ACK must never release a Pod"
    watchdog.cleanup()
    assert api.journal()["closed"], (
        "known unchanged unapproved resources remain removable"
    )


def test_failed_job_can_be_stopped_then_replaced_by_cleanup_only_observer(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    job = api.objects["job", watchdog.name]
    job["status"] = {"failed": 1, "conditions": [{"type": "Failed", "status": "True"}]}
    with pytest.raises(RegionalFixtureError):
        watchdog.validate_running()
    receipt = watchdog.resume_cleanup(seconds=30)
    assert receipt.state == "QUIESCENT" and receipt.case_failed, (
        "fresh cleanup proves cessation without clearing the original failed case"
    )
    assert (
        receipt.cleanup is not None
        and receipt.cleanup.deadline_at - receipt.cleanup.started_at == 30
    ), "the cleanup observer must use its own exact bounded attempt"
    record = api.journal()
    assert record["jobs"][0]["stopped"], (
        "the old observer must stop before its successor is created"
    )
    assert record["jobs"][1]["cleanup_id"] == receipt.cleanup.attempt_id, (
        "receipt must bind the fresh observer"
    )
    control = watchdog.control_client()
    assert control.quiescence() == receipt, (
        "the parent receives the original bound control for fence closure"
    )
    with pytest.raises(RegionalFixtureError):
        control.claim()
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    watchdog.cleanup()


def test_completion_wait_allows_unaccounted_success_but_requires_final_counts(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    api.completion_lag = 2
    control.request_close()
    receipt = watchdog.wait_quiescence()
    assert receipt.state == "QUIESCENT" and api.completion_lag == 0, (
        "pending Job accounting may delay proof but cannot replace final completion"
    )
    watchdog.cleanup()


@pytest.mark.parametrize("failure", ["failed", "malformed"])
@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_other_unknown_creation_outcomes_keep_intent_and_never_adopt(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], failure: str, kind: str
) -> None:
    api, plan, runtime = setup
    getattr(api, failure + "_create").add(kind)
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    item = (
        api.journal()["jobs"][-1]["resource"]
        if kind == "job"
        else api.journal()["support"][kind + "/" + watchdog.name]
    )
    assert item["uid"] is None, "an incomplete ACK must not create a guessed identity"
    with pytest.raises(RegionalFixtureError, match="unknown"):
        watchdog.cleanup()
    assert not [call for call in api.calls if call[0] == "delete"], (
        "uncertainty must preserve earlier references"
    )


@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_unapproved_ack_with_missing_run_label_still_has_uid_custody(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    api.create_changes[kind] = {"metadata/labels": {}}
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    watchdog.cleanup()
    assert api.journal()["closed"], (
        "direct ACK custody, not a label, authorizes deletion-only cleanup"
    )


@pytest.mark.parametrize("change", ["uid", "spec"])
def test_unapproved_ack_later_drift_is_not_deleted(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: str
) -> None:
    api, plan, runtime = setup
    api.create_changes["role"] = {"rules/0/verbs": ["get", "patch", "create"]}
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    role = api.objects["role", watchdog.name]
    if change == "uid":
        role["metadata"]["uid"] = "replacement"
    else:
        role["rules"][0]["verbs"].append("delete")
    with pytest.raises(RegionalFixtureError, match="differs|changed"):
        watchdog.cleanup()
    assert ("role", watchdog.name) in api.objects, (
        "deletion-only ACK custody cannot follow later drift"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"spec/hostPID": True},
        {"spec/containers/0/envFrom": [{"secretRef": {"name": "unapproved"}}]},
        {"spec/containers/0/command": ["/bin/sh"]},
        {"spec/containers/0/image": "cpu:latest"},
        {"spec/schedulingGates": []},
        {"spec/nodeName": "prebound"},
        {"spec/initContainers": [{"name": "extra", "image": "cpu:latest"}]},
    ],
)
def test_real_pod_admission_drift_is_never_released(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: dict[str, Any]
) -> None:
    api, plan, runtime = setup
    api.pod_changes = change
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    assert not [
        call for call in api.calls if call[0] == "patch" and call[1][1] == "pod"
    ], "actual Pod drift must fail before any gate patch"
    watchdog.cleanup()
    assert api.journal()["closed"], (
        "an unapproved child may only be stopped under its exact acknowledged Job"
    )


@pytest.mark.parametrize("feature", ["no-pod", "no-armed", "gate-not-applied"])
def test_startup_is_bounded_and_partial_arm_can_be_cleaned_without_quiescence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], feature: str
) -> None:
    api, plan, runtime = setup
    if feature == "no-pod":
        api.omit_pods = True
    elif feature == "no-armed":
        api.auto_arm = False
    else:
        api.gate_no_apply = True
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="deadline|unconfirmed"):
        watchdog.arm()
    assert not api.journal()["ever_armed"], (
        "a startup failure cannot claim successful arming"
    )
    assert api.clock.elapsed <= lifecycle.ARM_SECONDS, "startup waits must be bounded"
    watchdog.cleanup()
    assert api.journal()["closed"] and api.journal()["quiescence"] is None, (
        "resource-only cleanup must not manufacture a Store quiescence receipt"
    )


def test_pod_discovery_reads_only_this_jobs_pods_from_the_typed_list(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    # A live namespace held 17 Pods (462 KiB as kubectl JSON), past the bounded
    # document, and kubectl v1.35 prints a v1/List without a resourceVersion;
    # discovery must ask the server for this Job's own Pods (attempt 11).
    api, plan, runtime = setup
    for index in range(40):
        api.objects["pod", f"foreign-{index}"] = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"foreign-{index}",
                "namespace": runtime.namespace,
                "uid": f"foreign-{index}-uid",
                "resourceVersion": "9",
                "labels": {"app": "unrelated"},
                "ownerReferences": [
                    {
                        "apiVersion": "apps/v1",
                        "kind": "ReplicaSet",
                        "name": "unrelated",
                        "uid": "unrelated-rs-uid",
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
                "managedFields": [{"fieldsV1": {"f:" + "x" * 8000: {}}}],
            },
            "spec": {"containers": [{"name": "app", "image": "unrelated:1"}]},
            "status": {"phase": "Running"},
        }
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    job_uid = api.objects["job", watchdog.name]["metadata"]["uid"]
    discovery = [
        args
        for verb, args, _ in api.calls
        if verb == "get" and args[1] == "--raw" and "/pods?" in args[2]
    ]
    assert discovery, "Pod discovery must read the server's typed Pod list"
    assert all(
        args[2].endswith(
            "/pods?labelSelector=batch.kubernetes.io%2Fcontroller-uid%3D" + job_uid
        )
        for args in discovery
    ), "discovery must be narrowed to the Pods labelled with this Job's uid"
    assert not [
        args
        for verb, args, _ in api.calls
        if verb == "get" and args[1:4] == ("pod", "-o", "json")
    ], "a namespace-wide kubectl Pod list is neither bounded nor a typed list"


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"apiVersion": "v1", "kind": "PodList", "metadata": {}, "items": []},
        {
            "apiVersion": "v1",
            "kind": "PodList",
            "metadata": {"resourceVersion": "1", "continue": "more"},
            "items": [],
        },
        {
            "apiVersion": "v1",
            "kind": "PodList",
            "metadata": {"resourceVersion": "1"},
            "items": [None],
        },
    ],
)
def test_incomplete_pod_listing_cannot_authorize_a_gate(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], response: Any
) -> None:
    api, plan, runtime = setup
    api.list_override = response
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    assert not [call for call in api.calls if call[0] == "patch"], (
        "partial or malformed discovery cannot authorize a Pod"
    )


@pytest.mark.parametrize("field", ["uid", "owner", "node", "spec", "gone", "duplicate"])
def test_running_identity_or_spec_drift_blocks_execution_and_cleanup(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], field: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    pod_name = api.journal()["jobs"][0]["pod"]["name"]
    pod = api.objects["pod", pod_name]
    if field == "uid":
        pod["metadata"]["uid"] = "replacement"
    elif field == "owner":
        pod["metadata"]["ownerReferences"][0]["uid"] = "other-job"
    elif field == "node":
        pod["spec"]["nodeName"] = "other-node"
    elif field == "spec":
        pod["spec"]["containers"][0]["command"] = ["/bin/sh"]
    elif field == "gone":
        del api.objects["pod", pod_name]
    else:
        import copy

        extra = copy.deepcopy(pod)
        extra["metadata"].update(name=pod_name + "-other", uid="other-pod")
        api.objects["pod", pod_name + "-other"] = extra
    with pytest.raises(RegionalFixtureError):
        watchdog.validate_running()
    if field != "gone":
        with pytest.raises(RegionalFixtureError):
            watchdog.cleanup()
        assert ("job", watchdog.name) in api.objects, (
            "changed Pod ownership must not be silently adopted"
        )


@pytest.mark.parametrize(
    "status",
    [
        {"active": True},
        {"active": 2},
        {"failed": 1},
        {"terminating": 1},
        {"active": 0, "ready": 1},
        {"active": 1, "succeeded": 1},
        {"conditions": {}},
        {"conditions": [{"type": "Foreign", "status": "True"}]},
        {"conditions": [{"type": "Failed", "status": "Unknown"}]},
        {"conditions": [{"type": "Failed", "status": "True"}]},
        {"conditions": [{"type": "Complete", "status": "True"}]},
        {
            "conditions": [
                {"type": "Complete", "status": "False"},
                {"type": "Complete", "status": "False"},
            ]
        },
        {"uncountedTerminatedPods": {"failed": ["pod-a"]}},
        {"uncountedTerminatedPods": {"succeeded": ["foreign"]}},
        {"uncountedTerminatedPods": {"succeeded": None}},
        {"uncountedTerminatedPods": {"extra": []}},
    ],
)
def test_job_false_counts_and_unknown_progress_are_rejected(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], status: dict[str, Any]
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    api.objects["job", watchdog.name]["status"] = status
    with pytest.raises(RegionalFixtureError, match="progress|counts"):
        watchdog.validate_running()


def test_orphan_pod_must_be_removed_before_supporting_refs(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.keep_pods = True
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    watchdog.cleanup()
    deleted = [args[2] for verb, args, _ in api.calls if verb == "delete"]
    assert "/jobs/" in deleted[0] and "/pods/" in deleted[1], (
        "a Job deletion ACK is not proof that its Pod stopped"
    )
    assert all("/roles/" not in path for path in deleted[:2]), (
        "the observer's references must remain until actual Pod cessation"
    )


@pytest.mark.parametrize(
    "kind", ["job", "rolebinding", "role", "serviceaccount", "configmap"]
)
def test_delete_ack_loss_is_reconciled_only_by_actual_absence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    api.lost_delete.add(kind)
    watchdog.cleanup()
    assert api.journal()["closed"], (
        "known-UID DELETE can be confirmed by absence after ACK loss"
    )


@pytest.mark.parametrize(
    "kind", ["job", "rolebinding", "role", "serviceaccount", "configmap"]
)
def test_failed_deletion_preserves_records_for_retry(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], kind: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    api.failed_delete.add(kind)
    with pytest.raises(RegionalFixtureError):
        watchdog.cleanup()
    assert not api.journal()["closed"], "a failed deletion cannot be reported as clean"
    api.failed_delete.clear()
    api.watchdog(plan, runtime).cleanup()
    assert api.journal()["closed"], "cleanup must resume from acknowledged identities"
