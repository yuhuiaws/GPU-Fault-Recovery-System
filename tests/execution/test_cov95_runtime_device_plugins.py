from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from kubernetes import client

from gpu_fault.models import WorkflowStepStatus
from tests.execution._cov95_runtime_plugins import PluginHarness
from tests.execution._cov95_runtime_restart import ApiError


@pytest.mark.parametrize("family", ["gpu", "efa"])
@pytest.mark.parametrize("expected", [None, 0, -1, "unknown"])
def test_plugin_restart_refuses_unknown_expected_inventory_before_reading_or_deleting(
    family, expected
) -> None:
    h = PluginHarness(family, expected_count=expected)
    result = h.execute()
    assert (
        result.status is WorkflowStepStatus.FAILED
        and result.details["safety_rejection"]
    ), result
    assert result.details["resource_name"] == h.resource, result
    assert h.core.queries == [] and h.core.deleted == [] and h.core.patches == [], (
        "unproven inventory cannot authorize a device-plugin mutation",
        h.core.deleted,
    )


@pytest.mark.parametrize("family", ["gpu", "efa"])
def test_multiple_plugin_pods_are_refused_without_choosing_one_to_delete(
    family: str,
) -> None:
    h = PluginHarness(family)
    h.core.pods.append({"metadata": {"name": "plugin-second", "uid": "pod-second"}})
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    assert "found 2" in (result.error or ""), result
    assert h.core.deleted == [] and h.annotations == {}, (h.core.deleted, h.annotations)


@pytest.mark.parametrize("started", [None, "invalid", "future"])
def test_foreign_restart_with_unproven_age_stays_refused(started: str | None) -> None:
    h = PluginHarness()
    h.annotations[h.key("operation")] = "foreign-operation"
    h.annotations[h.key("incident")] = "foreign-incident"
    if started is not None:
        h.annotations[h.key("started-at")] = (
            (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            if started == "future"
            else started
        )
    before = deepcopy(h.annotations)
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    assert "another device plugin restart" in (result.error or ""), result
    assert h.annotations == before and h.core.deleted == [], (
        h.annotations,
        h.core.deleted,
    )


@pytest.mark.parametrize("scene", ["healthy", "no-pod", "replace-pod"])
def test_stale_restart_takeover_preserves_owner_attribution_and_observed_state(
    scene: str,
) -> None:
    h = PluginHarness()
    h.annotations.update(
        {
            h.key("operation"): "stale-operation",
            h.key("started-at"): (
                datetime.now(timezone.utc) - timedelta(minutes=2)
            ).isoformat(),
        }
    )
    if scene == "healthy":
        h.node["status"]["allocatable"][h.resource] = "8"
    if scene == "no-pod":
        h.core.pods = []
    result = h.execute()
    assert result.status is (
        WorkflowStepStatus.SUCCEEDED
        if scene == "healthy"
        else WorkflowStepStatus.WAITING
    ), result
    observed = result.details["node_results"]["node-a"]
    assert observed["took_over_incident"] == "stale-operation", observed
    assert h.core.deleted == (
        [("kube-system", "plugin-old")] if scene == "replace-pod" else []
    ), h.core.deleted
    if scene == "healthy":
        assert observed["already_healthy"] is True and h.annotations == {}, observed
    else:
        assert h.annotations[h.key("operation")] == h.context.idempotency_key, (
            h.annotations
        )


@pytest.mark.parametrize("started", [None, "invalid"])
def test_owned_restart_with_unknown_start_time_does_not_delete_the_pod_again(
    started: str | None,
) -> None:
    h = PluginHarness()
    h.annotations[h.key("operation")] = h.context.idempotency_key
    if started is not None:
        h.annotations[h.key("started-at")] = started
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert (
        result.details["node_results"]["node-a"]["waiting_for_replacement_pod"] is True
    ), result
    assert h.core.deleted == [], "a waiting restart must not reissue its Pod deletion"


@pytest.mark.parametrize("status", [404, 500])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_delete_failure_preserves_the_original_error_and_does_not_claim_recovery(
    status: int, cleanup_fails: bool
) -> None:
    h = PluginHarness()
    h.core.delete_error = ApiError(status)
    h.core.clear_error = cleanup_fails
    if status == 404:
        result = h.execute()
        assert result.status is WorkflowStepStatus.WAITING, result
        assert h.annotations[h.key("operation")] == h.context.idempotency_key, (
            h.annotations
        )
    else:
        with pytest.raises(ApiError, match="fake Kubernetes status 500"):
            h.execute()
        assert bool(h.annotations) is cleanup_fails, h.annotations
    assert h.core.deleted == [("kube-system", "plugin-old")], h.core.deleted


def test_exhausted_node_patch_race_never_deletes_a_plugin_without_ownership() -> None:
    h = PluginHarness()
    h.core.conflicts = 100
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details["node_results"] == {
        "node-a": {"patch_conflict_retry": True}
    }, result
    assert h.core.deleted == [] and h.annotations == {}, (h.core.deleted, h.annotations)
    assert h.core.patch_attempts < 100, (
        "patch conflicts must have a finite retry budget"
    )


def test_typed_inventory_and_pod_readiness_confirm_replacement_and_clear_ownership() -> (
    None
):
    h = PluginHarness(
        "efa",
        typed=True,
        plugin_namespace="plugins",
        plugin_label_selector="plugin=unit-efa",
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    h.node["status"]["allocatable"][h.resource] = "8"
    h.core.pods = [
        client.V1Pod(
            metadata=client.V1ObjectMeta(name="plugin-new", uid="pod-new"),
            status=client.V1PodStatus(
                conditions=[client.V1PodCondition(type="Ready", status="True")]
            ),
        )
    ]
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    assert second.details["node_results"]["node-a"]["replacement_pod_uids"] == [
        "pod-new"
    ], second
    assert h.annotations == {}, h.annotations
    assert (
        h.core.queries == [("plugins", "plugin=unit-efa", "spec.nodeName=node-a")] * 2
    ), h.core.queries
    assert h.core.deleted == [("plugins", "plugin-old")], h.core.deleted
