"""The real fence controller keeps protection until its own actions are quiescent."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional import destr008_admission as admission
from scripts.e2e.regional.destr008_admission import (
    API_VERSION,
    POOL_STATE,
    RESERVATION,
    RESOURCES,
    FenceBinding,
    require_admission_api,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_admission import build_harness
from tests.regional._destr008_control import build_control


def test_activation_fence_has_independent_durable_ownership_and_ordered_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    identity = fence.arm()
    assert set(harness.objects) == set(RESOURCES), (
        "both server-side fence resources must exist"
    )
    assert identity["node_uid"] == harness.binding.node_uid, (
        "the fence must bind the actual target UID"
    )
    assert fence.path.stat().st_mode & 0o777 == 0o600, (
        "ownership journals must be private"
    )
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    watchdog.submit(command_status=RemoteCommandStatus.PENDING)
    saved = json.loads(fence.path.read_text())
    assert saved["action_started"] is True, "action intent must survive parent loss"
    restored = harness.fence()
    receipt = watchdog.finish()
    restored.close(watchdog.control)
    assert harness.objects == {}, (
        "all and only the owned admission objects must be removed"
    )
    deletions = [
        args[2]
        for operation, args in harness.calls
        if operation == "kube" and args[0] == "delete"
    ]
    assert "validatingadmissionpolicybindings" in deletions[0], (
        "detach the binding before its policy"
    )
    assert "validatingadmissionpolicies" in deletions[1], (
        "the policy is the final cleanup step"
    )
    assert json.loads(fence.path.read_text())["phase"] == "CLOSED", (
        "completion must be durable"
    )
    assert json.loads(fence.path.read_text())[
        "retirement_receipt"
    ] == receipt.model_dump(mode="json")
    restored.close()
    assert (
        len(
            [row for row in harness.calls if row[0] == "kube" and row[1][0] == "delete"]
        )
        == 2
    ), "completed cleanup must not delete resources again"
    with pytest.raises(RegionalFixtureError, match="closing"):
        harness.fence().arm()


@pytest.mark.parametrize(
    "change",
    [
        {"state": "ARMED"},
        {"run_id": "foreign-run"},
        {"fence": {}},
        {"commands_active": 1},
        {"commands_active": False},
        {"workflows_active": 1},
        {"workflows_active": False},
        {"producer_revoked": False},
        {"producer_revoked": 1},
    ],
)
def test_unknown_or_inflight_work_never_releases_the_activation_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    watchdog.submit()
    proof = {**watchdog.finish().model_dump(mode="json"), **change}
    watchdog.rewrite("status.json", proof)
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        harness.fence().close(watchdog.control)
    assert set(harness.objects) == set(RESOURCES), (
        "uncertain activity must leave the server fence intact"
    )
    assert not any(
        row[0] == "kube" and row[1][0] == "delete" for row in harness.calls
    ), "no cleanup mutation may precede proven quiescence"


def test_parent_loss_without_any_quiescence_receipt_preserves_protection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    with pytest.raises(RegionalFixtureError, match="watchdog control"):
        harness.fence().close()
    watchdog.rewrite("status.json", None)
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        harness.fence().close(watchdog.control)
    assert len(harness.objects) == 2, (
        "elapsed time and parent loss cannot remove the guard"
    )


@pytest.mark.parametrize("kind", list(RESOURCES))
def test_lost_create_ack_cannot_be_reconstructed_from_public_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.lost_create.add(kind)
    with pytest.raises(TimeoutError):
        harness.fence().arm()
    before = copy.deepcopy(harness.objects)
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().arm()
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().close()
    assert harness.objects == before, (
        "unconfirmed creation must preserve resources for reconciliation"
    )


def test_successful_create_ack_supports_resume_without_another_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    identity = harness.fence().arm()
    assert identity["policy_uid"] and identity["binding_uid"], (
        "the actual create acknowledgements must establish both UID proofs"
    )
    resumed = harness.fence()
    resumed.arm()
    assert (
        len(
            [row for row in harness.calls if row[0] == "kube" and row[1][0] == "create"]
        )
        == 2
    ), "a fresh controller cannot duplicate an acknowledged creation"
    resumed.close()
    assert harness.objects == {}, "pre-action rollback may remove its proven resources"


def test_unobserved_create_outcome_is_not_retried_or_reported_as_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.failed_create.add("validatingadmissionpolicy")
    with pytest.raises(TimeoutError):
        harness.fence().arm()
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().arm()
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().close()
    creates = [
        row for row in harness.calls if row[0] == "kube" and row[1][0] == "create"
    ]
    assert len(creates) == 1, "an uncertain first submission cannot authorize another"


@pytest.mark.parametrize(
    "field", ["uid", "labels", "annotations", "spec", "deletion", "version"]
)
def test_resource_drift_is_refused_before_actions_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    policy = harness.objects["validatingadmissionpolicy"]
    if field == "spec":
        policy["spec"]["failurePolicy"] = "Ignore"
    elif field == "deletion":
        policy["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    elif field == "version":
        policy["metadata"]["resourceVersion"] = 7
    elif field in {"labels", "annotations"}:
        policy["metadata"][field] = {}
    else:
        policy["metadata"]["uid"] = "recreated"
    with pytest.raises(RegionalFixtureError, match="ownership or spec"):
        fence.protect_action(watchdog.control)
    watchdog.finish()
    with pytest.raises(RegionalFixtureError, match="ownership or spec"):
        fence.close(watchdog.control)
    assert set(harness.objects) == set(RESOURCES), (
        "known ownership drift must stop cleanup before either deletion"
    )


@pytest.mark.parametrize("drift", ["uid", "name", "cordon", "reservation", "pool"])
def test_target_drift_keeps_an_active_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    watchdog.finish()
    if drift in {"uid", "name"}:
        harness.node["metadata"][drift] = "recreated"
    elif drift == "cordon":
        harness.node["spec"]["unschedulable"] = False
    elif drift == "reservation":
        harness.node["metadata"]["annotations"][RESERVATION] = "foreign"
    else:
        harness.node["metadata"]["annotations"][POOL_STATE] = "ALLOCATED"
    with pytest.raises(RegionalFixtureError, match="identity or allocation"):
        fence.close(watchdog.control)
    assert len(harness.objects) == 2, (
        "a replaced or allocated node requires reconciliation"
    )


@pytest.mark.parametrize(
    "change", [{"state": "UNKNOWN"}, {"safe_dry_run_acknowledged": 1}, {"extra": True}]
)
def test_incomplete_or_untyped_probe_receipt_does_not_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.probe_changes = change
    fence = harness.fence()
    with pytest.raises(RegionalFixtureError, match="probe failed"):
        fence.arm()
    watchdog = harness.watchdog(fence, bind=False)
    with pytest.raises(RegionalFixtureError, match="not been armed"):
        fence.protect_action(watchdog.control)
    assert json.loads(fence.path.read_text())["action_started"] is False, (
        "unproven fences cannot start a fixture"
    )
    fence.close()


def test_foreign_preexisting_policy_is_not_adopted_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    existing = harness.install(fence.objects[0])
    with pytest.raises(RegionalFixtureError, match="already in use"):
        fence.arm()
    fence.close()
    assert harness.objects["validatingadmissionpolicy"] == existing, (
        "no creation intent means no cleanup authority"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"kind": "APIResourceList", "groupVersion": API_VERSION, "resources": []},
        {"kind": "APIResourceList", "groupVersion": "other/v1", "resources": []},
        {
            "kind": "APIResourceList",
            "groupVersion": API_VERSION,
            "resources": [{"name": "validatingadmissionpolicies", "namespaced": True}],
        },
    ],
)
def test_missing_or_incomplete_admission_support_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.discovery = value if value is not None else []
    with pytest.raises(RegionalFixtureError, match="API|fences"):
        require_admission_api(harness.regional)
    assert harness.objects == {}, "unsupported APIs cannot create a partial guard"


def test_partial_cleanup_resumes_without_recreating_a_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    harness.failed_delete.add("validatingadmissionpolicy")
    with pytest.raises(TimeoutError):
        fence.close()
    assert set(harness.objects) == {"validatingadmissionpolicy"}, (
        "a failed deletion must retain its exact remaining identity"
    )
    harness.failed_delete.clear()
    harness.fence().close()
    assert harness.objects == {}, "only remaining cleanup is retried"


def test_context_drift_prevents_any_further_api_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    before = len(harness.calls)
    harness.scope["cluster_id"] = "foreign-cluster"
    with pytest.raises(RegionalFixtureError, match="cluster identity"):
        fence.close()
    assert len(harness.calls) == before, (
        "wrong-cluster cleanup must stop before the API"
    )


def test_invalid_journal_and_recreated_closed_name_are_not_new_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    original = json.loads(fence.path.read_text())
    broken = {**original, "schema_version": True}
    fence.path.write_text(json.dumps(broken))
    with pytest.raises(RegionalFixtureError, match="journal"):
        harness.fence()
    fence.path.write_text(json.dumps(original))
    fence.close()
    recreated = harness.install(copy.deepcopy(fence.objects[0]))
    recreated["metadata"]["uid"] = "foreign"
    with pytest.raises(RegionalFixtureError, match="recreated"):
        harness.fence().close()
    assert len(harness.objects) == 1, (
        "old terminal evidence never deletes a recreated object"
    )


def test_run_label_length_is_checked_before_resource_generation() -> None:
    with pytest.raises(RegionalFixtureError, match="too long"):
        FenceBinding("a" * 64, "cluster", "node", "uid", "release")


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_changed_kubeconfig_stops_before_even_a_release_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plane: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    path = getattr(harness.regional.settings, f"{plane}_kubeconfig")
    path.write_text("changed fixture identity\n")
    before = len(harness.calls)
    monkeypatch.setattr(
        harness.regional,
        "evidence_identity",
        lambda: pytest.fail("changed connections cannot issue a release read"),
    )
    with pytest.raises(RegionalFixtureError, match="connection identity"):
        fence.protect_action(watchdog.control)
    with pytest.raises(RegionalFixtureError, match="journal identity"):
        harness.fence()
    assert len(harness.calls) == before, "connection drift must precede every API call"


@pytest.mark.parametrize("field", ["gpu_context", "namespace", "region"])
def test_changed_endpoint_settings_cannot_start_a_second_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    previous = harness.regional.settings
    harness.regional.settings = replace(
        previous,
        gpu_context="other-target" if field == "gpu_context" else previous.gpu_context,
        namespace="other-target" if field == "namespace" else previous.namespace,
        region="other-target" if field == "region" else previous.region,
    )
    before = len(harness.calls)
    with pytest.raises(RegionalFixtureError, match="connection identity"):
        fence.protect_action(watchdog.control)
    with pytest.raises(RegionalFixtureError, match="journal identity"):
        harness.fence()
    assert len(harness.calls) == before, "same run ID cannot authorize another target"


def test_missing_kubeconfig_is_not_a_valid_new_connection_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.regional.settings.gpu_kubeconfig.unlink()
    monkeypatch.setattr(
        harness.regional,
        "evidence_identity",
        lambda: pytest.fail("unreadable configuration cannot initiate any API call"),
    )
    with pytest.raises(RegionalFixtureError, match="kubeconfig identity"):
        harness.fence()


def test_changed_target_binding_cannot_hide_the_existing_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    harness.binding = replace(harness.binding, node_uid="replacement-node-uid")
    before = len(harness.calls)
    with pytest.raises(RegionalFixtureError, match="journal identity"):
        harness.fence()
    assert len(harness.calls) == before, (
        "an old run remains bound to its original target"
    )


def test_foreign_cluster_binding_is_rejected_without_api_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.binding = replace(harness.binding, cluster_id="foreign-cluster")
    monkeypatch.setattr(
        harness.regional,
        "evidence_identity",
        lambda: pytest.fail("inconsistent target cannot initiate a release read"),
    )
    with pytest.raises(RegionalFixtureError, match="binding is inconsistent"):
        harness.fence()
    assert harness.calls == [], "the configured cluster and bound cluster must agree"


@pytest.mark.parametrize("field", ["cluster_id", "release_id"])
def test_live_identity_must_match_the_requested_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    harness.scope[field] = "other-identity"
    with pytest.raises(RegionalFixtureError, match="live release or cluster"):
        harness.fence()
    assert harness.calls == [], "a live identity mismatch cannot create resources"


def test_journal_symlink_is_not_a_resume_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    original = fence.path.with_suffix(".original")
    fence.path.rename(original)
    fence.path.symlink_to(original)
    with pytest.raises(RegionalFixtureError, match="cannot be a link"):
        harness.fence()
    assert len(harness.objects) == 2, "journal substitution must not release the guard"


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "UNKNOWN"},
        {"action_started": 1},
        {"attempted": {}},
        {"resources": []},
        {"attempted": ["unknown-resource"]},
        {"attempted": ["validatingadmissionpolicy", "validatingadmissionpolicy"]},
        {
            "attempted": [
                "validatingadmissionpolicybinding",
                "validatingadmissionpolicy",
            ]
        },
        {"resources": {}},
        {"resources": {"validatingadmissionpolicy": ""}},
        {"resources": {"validatingadmissionpolicy": 42}},
        {"action_started": True, "phase": "PREPARING"},
    ],
)
def test_malformed_or_inconsistent_journal_cannot_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    document = json.loads(fence.path.read_text())
    fence.path.write_text(json.dumps({**document, **change}))
    before = len(harness.calls)
    with pytest.raises(RegionalFixtureError, match="journal identity"):
        harness.fence()
    assert len(harness.calls) == before, (
        "invalid journal state cannot authorize an API mutation"
    )


def test_non_object_resource_readback_cannot_be_treated_as_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    original = harness.regional.kubectl

    def changed(plane: str, *args: str, **kwargs: Any) -> str:
        if args[:2] == ("get", "validatingadmissionpolicy"):
            return "[]"
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(harness.regional, "kubectl", changed)
    with pytest.raises(RegionalFixtureError, match="not an object"):
        fence.protect_action(watchdog.control)
    assert len(harness.objects) == 2, (
        "unreadable policy state must preserve both resources"
    )


def test_unarmed_controller_cannot_publish_complete_fence_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    with pytest.raises(RegionalFixtureError, match="incomplete resource"):
        harness.fence().identity()


def test_stale_controller_reloads_started_action_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    first = harness.fence()
    first.arm()
    stale = harness.fence()
    watchdog = harness.watchdog(first)
    first.protect_action(watchdog.control)
    with pytest.raises(RegionalFixtureError, match="watchdog control"):
        stale.close()
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        stale.close(watchdog.control)
    assert json.loads(first.path.read_text())["action_started"] is True, (
        "stale cached false cannot overwrite durable action intent"
    )
    assert len(harness.objects) == 2, (
        "stale controller must not remove the active fence"
    )


def test_create_ack_uid_cannot_be_replaced_by_a_later_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    original = harness.regional.kubectl

    def replace_after_ack(plane: str, *args: str, **kwargs: Any) -> str:
        response = original(plane, *args, **kwargs)
        if args[0] == "create":
            harness.objects["validatingadmissionpolicy"]["metadata"]["uid"] = (
                "recreated"
            )
        return response

    monkeypatch.setattr(harness.regional, "kubectl", replace_after_ack)
    fence = harness.fence()
    with pytest.raises(RegionalFixtureError, match="ownership or spec"):
        fence.arm()
    recorded = json.loads(fence.path.read_text())["resources"]
    assert recorded["validatingadmissionpolicy"] == "uid-validatingadmissionpolicy", (
        "the create ACK's original UID, not a replacement, is the only authority"
    )
    assert (
        harness.objects["validatingadmissionpolicy"]["metadata"]["uid"] == "recreated"
    ), "replacement objects must remain untouched"


def test_delete_uses_the_revision_from_the_full_spec_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    original = harness.regional.kubectl

    def change_during_cleanup(plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "get" and args[-1] == "jsonpath={.metadata}":
            resource = harness.objects[args[1]]
            resource["metadata"]["resourceVersion"] = "2"
            resource["spec"]["validationActions"] = ["Warn"]
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(harness.regional, "kubectl", change_during_cleanup)
    with pytest.raises(RuntimeError, match="changed after its full cleanup check"):
        fence.close()
    assert len(harness.objects) == 2, "a changed revision must stop before DELETE"


@pytest.mark.parametrize(
    "field", ["run_id", "cluster_id", "release_id", "fence", "namespace"]
)
def test_binding_requires_the_full_matching_watchdog_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    values = {
        "run_id": harness.binding.run_id,
        "cluster_id": harness.binding.cluster_id,
        "release_id": harness.binding.release_id,
        "spare_node": harness.binding.node,
        "fence": fence.identity(),
        "namespace": harness.regional.settings.namespace,
    }
    values[field] = (
        {**values["fence"], "policy_uid": "foreign"} if field == "fence" else "foreign"
    )
    watchdog = build_control(**values)
    assert watchdog.tick().state == "ARMED"
    with pytest.raises(RegionalFixtureError, match="does not bind"):
        fence.bind_watchdog(watchdog.control)
    assert "watchdog" not in json.loads(fence.path.read_text())
    assert len(harness.objects) == 2


def test_different_controller_and_six_field_dictionary_are_not_retirement_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    other = build_control(
        namespace=harness.regional.settings.namespace,
        uid="different-controller",
        run_id=harness.binding.run_id,
        cluster_id=harness.binding.cluster_id,
        release_id=harness.binding.release_id,
        spare_node=harness.binding.node,
        fence=fence.identity(),
    )
    other.tick()
    with pytest.raises(RegionalFixtureError, match="already bound"):
        fence.bind_watchdog(other.control)
    fence.protect_action(watchdog.control)
    with pytest.raises(RegionalFixtureError, match="cannot change"):
        fence.bind_watchdog(other.control)
    watchdog.finish()
    with pytest.raises(RegionalFixtureError, match="watchdog control"):
        fence.close(other.control)
    with pytest.raises(RegionalFixtureError, match="watchdog control"):
        fence.close(
            {
                "state": "QUIESCENT",
                "run_id": harness.binding.run_id,
                "fence": fence.identity(),
                "commands_active": 0,
                "workflows_active": 0,
                "producer_revoked": True,
            }
        )
    assert len(harness.objects) == 2, (
        "unbound or abbreviated proof cannot retire protection"
    )
    fence.close(watchdog.control)
    assert not harness.objects, (
        "the bound watchdog proof must retire both fence resources"
    )


def test_stale_full_receipt_needs_new_real_cleanup_observation_before_fence_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    watchdog.submit()
    old = watchdog.finish()
    watchdog.clock.advance(31)
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        harness.fence().close(watchdog.control)
    assert len(harness.objects) == 2
    watchdog.cleanup(attempt_id="fresh-close-proof")
    first = watchdog.tick()
    assert first.state == "REVOKED" and first.quiet_since == int(watchdog.clock())
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        fence.close(watchdog.control)
    watchdog.clock.advance(5)
    fresh = watchdog.tick()
    assert fresh.state == "QUIESCENT" and fresh.sequence > old.sequence
    fence.close(watchdog.control)
    saved = json.loads(fence.path.read_text())["retirement_receipt"]
    assert saved == fresh.model_dump(mode="json") and saved["case_failed"] is True
    assert not harness.objects, "fresh quiescence must retire both fence resources"


def test_actual_leased_command_prevents_fence_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    watchdog.submit(command_status=RemoteCommandStatus.LEASED)
    watchdog.control.request_close()
    observed = watchdog.tick()
    assert observed.commands_active == 1
    with pytest.raises(RegionalFixtureError, match="quiescence"):
        fence.close(watchdog.control)
    assert (
        watchdog.store.get_remote_command("command-a").status
        is RemoteCommandStatus.LEASED
    )
    assert len(harness.objects) == 2


@pytest.mark.parametrize("invalid", ["not-object", "uid", "version", "spec"])
def test_invalid_direct_create_ack_never_authorizes_metadata_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    original = harness.regional.kubectl

    def invalid_ack(plane: str, *args: str, **kwargs: Any) -> str:
        response = original(plane, *args, **kwargs)
        if args[0] != "create":
            return response
        value = json.loads(response)
        if invalid == "not-object":
            return "[]"
        if invalid == "uid":
            value["metadata"]["uid"] = None
        elif invalid == "version":
            value["metadata"]["resourceVersion"] = 1
        else:
            value["spec"] = {}
        return json.dumps(value)

    monkeypatch.setattr(harness.regional, "kubectl", invalid_ack)
    with pytest.raises(RegionalFixtureError):
        harness.fence().arm()
    actual = copy.deepcopy(harness.objects)
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().arm()
    with pytest.raises(RegionalFixtureError, match="unknown"):
        harness.fence().close()
    assert harness.objects == actual and len(actual) == 1


def test_fence_cannot_publish_a_journal_without_controller_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    with pytest.raises(RegionalFixtureError, match="ownership is required"):
        fence.save()
    assert not fence.path.exists() and not harness.objects


def test_oversized_independent_probe_is_refused_before_resource_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    oversized = tmp_path / "oversized-probe"
    oversized.write_bytes(b"x" * 65537)
    monkeypatch.setattr(admission, "PROBE", oversized)
    with pytest.raises(RegionalFixtureError, match="size limit"):
        harness.fence()
    assert not harness.calls and not harness.objects


def test_missing_acknowledged_policy_is_not_an_armed_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    del harness.objects["validatingadmissionpolicy"]
    with pytest.raises(RegionalFixtureError, match="resource is missing"):
        fence.protect_action(watchdog.control)
    assert json.loads(fence.path.read_text())["action_started"] is False
    assert "validatingadmissionpolicybinding" in harness.objects


def test_foreign_full_receipt_cannot_be_authorized_by_corrupting_the_local_watchdog_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = build_harness(tmp_path, monkeypatch)
    fence = harness.fence()
    fence.arm()
    watchdog = harness.watchdog(fence)
    fence.protect_action(watchdog.control)
    other = build_control(namespace=harness.regional.settings.namespace)
    other.tick()
    other.finish()
    saved = json.loads(fence.path.read_text())
    saved["watchdog"] = other.control.identity()
    fence.path.write_text(json.dumps(saved))
    with pytest.raises(RegionalFixtureError, match="belongs to another fence"):
        harness.fence().close(other.control)
    assert len(harness.objects) == 2
