from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from gpu_fault.models import WorkflowStepStatus
from tests.execution._cov95_runtime_restart import ApiError, RestartHarness


@pytest.mark.parametrize("kind", ["job", "pytorchjob", "jobset"])
@pytest.mark.parametrize("change", ["missing", "gone", "deleting", "uid"])
def test_restart_refuses_source_changes_during_the_last_pre_mutation_read(
    kind: str, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RestartHarness(kind)
    read = h.api.read
    reads = 0

    def raced_read(name, namespace):
        nonlocal reads
        reads += 1
        if reads == 2:
            if change in {"missing", "gone"}:
                raise ApiError(404 if change == "missing" else 410)
            if change == "deleting":
                h.api.source["metadata"]["deletionTimestamp"] = "2026-09-12T12:00:00Z"
            else:
                h.api.source["metadata"]["uid"] = "new-source-uid"
        return read(name, namespace)

    monkeypatch.setattr(h.api, "read", raced_read)
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    expected = {
        "missing": "RESTART_SOURCE_WORKLOAD_NOT_FOUND",
        "gone": "RESTART_SOURCE_WORKLOAD_NOT_FOUND",
        "deleting": "RESTART_SOURCE_WORKLOAD_DELETING",
        "uid": "RESTART_SOURCE_WORKLOAD_IDENTITY_DRIFT",
    }[change]
    assert result.details["reason"] == expected, result
    assert result.details["restart_submitted"] is False, result
    assert result.details["source_workload_ids"] == h.context.step.workload_ids, result
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)


@pytest.mark.parametrize("kind", ["job", "pytorchjob", "jobset"])
def test_restart_refresh_transport_failure_never_becomes_false_absence(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RestartHarness(kind)
    read = h.api.read
    reads = 0

    def raced_read(name, namespace):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise ApiError(500)
        return read(name, namespace)

    monkeypatch.setattr(h.api, "read", raced_read)
    with pytest.raises(ApiError, match="fake Kubernetes status 500"):
        h.execute()
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)


@pytest.mark.parametrize("kind", ["pytorchjob", "jobset"])
@pytest.mark.parametrize("resource_version", [None, "6"])
def test_patch_conflict_refreshes_only_resource_version_while_preserving_uid(
    kind: str, resource_version: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RestartHarness(kind, terminal=False)
    original = h.api.patch_namespaced_custom_object
    attempts = []

    def patch(group, version, namespace, plural, name, body):
        attempts.append(deepcopy(body))
        if len(attempts) == 1:
            h.api.source["metadata"]["resourceVersion"] = resource_version
            raise ApiError(409)
        return original(group, version, namespace, plural, name, body)

    monkeypatch.setattr(h.api, "patch_namespaced_custom_object", patch)
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert len(attempts) == 2, attempts
    assert attempts[0]["metadata"]["resourceVersion"] == "5", attempts
    if resource_version is None:
        assert "resourceVersion" not in attempts[1]["metadata"], attempts
    else:
        assert attempts[1]["metadata"]["resourceVersion"] == "6", attempts
    assert all(body["metadata"].get("uid") == "source-uid" for body in attempts), (
        "resource-version retries must retain the original object's immutable UID",
        attempts,
    )
    assert len(h.api.patches) == 1 and h.api.created == {}, (
        h.api.patches,
        h.api.created,
    )


@pytest.mark.parametrize("kind", ["pytorchjob", "jobset"])
def test_patch_conflict_cannot_retarget_a_recreated_workload_with_the_same_name(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = RestartHarness(kind, terminal=False)
    original = h.api.patch_namespaced_custom_object
    attempts = []

    def patch(group, version, namespace, plural, name, body):
        attempts.append(deepcopy(body))
        if len(attempts) == 1:
            h.api.source["metadata"].update(uid="recreated-uid", resourceVersion="6")
            raise ApiError(409)
        return original(group, version, namespace, plural, name, body)

    monkeypatch.setattr(h.api, "patch_namespaced_custom_object", patch)
    with pytest.raises(ValueError, match="identity.*changed"):
        h.execute()
    assert len(attempts) == 1, (
        "a UID change must be refused before another patch is sent"
    )
    assert h.api.patches == [] and h.api.created == {}, (h.api.patches, h.api.created)
    assert h.api.source["metadata"]["uid"] == "recreated-uid", h.api.source


@pytest.mark.parametrize("kind", ["pytorchjob", "jobset"])
def test_patch_without_resource_version_still_carries_the_source_uid(kind: str) -> None:
    h = RestartHarness(kind, terminal=False)
    h.api.source["metadata"].pop("resourceVersion")
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    [body] = h.api.patches
    assert "resourceVersion" not in body["metadata"], body
    assert body["metadata"]["uid"] == "source-uid", (
        "the immutable UID must keep a versionless patch bound to the observed object",
        body,
    )


@pytest.mark.parametrize(
    "workload_id",
    [
        "",
        "training/job/name/extra",
        "training/deployment/name",
        "/job/name",
        "training/job/",
    ],
)
def test_invalid_workload_identifiers_are_refused_before_api_reads(
    workload_id: str,
) -> None:
    h = RestartHarness("job")
    step = h.context.step.model_copy(update={"workload_ids": [workload_id]})
    h.context = replace(
        h.context,
        step=step,
        workflow=h.context.workflow.model_copy(update={"official_steps": [step]}),
    )
    with pytest.raises(ValueError, match="workload"):
        h.execute()
    assert h.api.reads == [], "an invalid namespace or kind must not reach Kubernetes"


@pytest.mark.parametrize(
    "workload_id", ["training/training-job", "training/JoB/training-job"]
)
def test_legacy_and_case_normalized_job_identifiers_stay_usable(
    workload_id: str,
) -> None:
    h = RestartHarness("job")
    step = h.context.step.model_copy(update={"workload_ids": [workload_id]})
    h.context = replace(
        h.context,
        step=step,
        workflow=h.context.workflow.model_copy(update={"official_steps": [step]}),
    )
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert len(h.api.created) == 1 and h.api.patches == [], (
        h.api.created,
        h.api.patches,
    )
