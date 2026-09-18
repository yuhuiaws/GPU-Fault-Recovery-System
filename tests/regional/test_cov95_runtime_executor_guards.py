from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from tests.execution.test_cluster_executor_batching import (
    FakeClient,
    FakeNodeAdapter,
    all_succeed,
    compound_command,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def make_executor(
    tmp_path: Path, client: FakeClient, adapters: list[Any], **changes: Any
) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client,
        adapters,
        **{
            "executor_id": "unit-executor",
            "allowed_namespaces": {"training"},
            "claim_state_path": str(tmp_path / "claim.json"),
            "liveness_state_path": str(tmp_path / "alive"),
            **changes,
        },
    )


@pytest.mark.parametrize(
    ("setting", "value", "reason"),
    [
        ("poll_seconds", 0, "poll seconds"),
        ("claim_wait_seconds", -1, "claim wait"),
        ("claim_wait_seconds", 31, "claim wait"),
        ("lease_seconds", 9, "lease seconds"),
        ("lease_seconds", 7201, "lease seconds"),
        ("batch_size", 0, "batch size"),
        ("batch_size", 26, "batch size"),
        ("max_concurrent_commands", 0, "concurrency"),
        ("max_concurrent_commands", 26, "concurrency"),
        ("claim_backoff_max_seconds", 1, "claim backoff max"),
        ("lease_renewal_failure_limit", 0, "renewal failure limit"),
        ("lease_renewal_failure_limit", 101, "renewal failure limit"),
        ("transport_degraded_backoff_after", 0, "transport degraded backoff"),
        ("transport_degraded_backoff_after", 101, "transport degraded backoff"),
        ("max_execution_seconds", 0, "max execution seconds"),
        ("max_execution_seconds", 86401, "max execution seconds"),
        ("liveness_interval_seconds", 0, "liveness interval"),
    ],
)
def test_invalid_executor_limits_refuse_before_claiming_or_writing_breadcrumbs(
    tmp_path: Path, setting: str, value: int, reason: str
) -> None:
    command = compound_command()
    client = FakeClient(command)
    adapter = FakeNodeAdapter(all_succeed())
    with pytest.raises(ClusterExecutorError, match=reason):
        make_executor(tmp_path, client, [adapter], **{setting: value})
    assert client.pending == [command]
    assert client.completed == []
    assert client.renewals == 0
    assert adapter.contexts == []
    assert not (tmp_path / "claim.json").exists(), (
        "invalid config must not mark a claim"
    )
    assert not (tmp_path / "alive").exists(), "invalid config must not mark liveness"


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("workflow-fence", "fencing token does not match"),
        ("incident-fence", "fencing token does not match"),
        ("no-namespaces", "no allowed workload namespaces"),
        ("foreign-namespace", "workload namespace is not allowed"),
    ],
)
def test_claimed_command_scope_failure_is_reported_without_entering_an_adapter(
    tmp_path: Path, defect: str, reason: str
) -> None:
    payload = compound_command().model_dump(mode="json")
    payload["batched_steps"] = []
    allowed = {"training"}
    if defect == "workflow-fence":
        payload["workflow"]["fencing_token"] += 1
    elif defect == "incident-fence":
        payload["incident"]["fencing_token"] += 1
    else:
        payload["step"]["workload_ids"] = ["other/job/unowned"]
        if defect == "no-namespaces":
            allowed = set()
    claimed = RemoteActionCommand.model_validate(payload)
    client = FakeClient(claimed)
    adapter = FakeNodeAdapter(all_succeed())
    executor = make_executor(tmp_path, client, [adapter], allowed_namespaces=allowed)
    assert executor.run_once() == 1
    result = client.reported()
    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-rejected"
    assert reason in result.error
    assert adapter.contexts == []
    assert client.progress_posts == []
    assert executor.unexpected_failures == 0


def test_cross_cluster_claim_is_neither_renewed_executed_nor_reported(
    tmp_path: Path,
) -> None:
    claimed = compound_command().model_copy(
        update={"cluster_id": "cluster-b", "batched_steps": []}
    )
    client = FakeClient(claimed)
    adapter = FakeNodeAdapter(all_succeed())
    executor = make_executor(tmp_path, client, [adapter])

    assert executor.run_once() == 1, "run_once counts received claims, not executions"
    assert client.renewals == 0, "foreign authority cannot authorize a renewal"
    assert adapter.contexts == [], "a foreign claim cannot enter an adapter"
    assert client.progress_posts == [], "a foreign claim cannot post progress"
    assert client.completed == [], "a result cannot cross the cluster binding"
    counters = executor.metrics_snapshot()
    assert counters["claimed_total"] == 1, "the refused claim must still be counted"
    assert counters["lease_lost_total"] == 1, "foreign lease authority must be lost"
    assert counters["results_withheld_total"] == 1, "the refused result is withheld"
    assert counters["unexpected_failures"] == 0, "admission refusal is not a defect"
    assert counters["reported_failures"] == 0, "there was no result POST to fail"
    assert executor.last_cycle_advanced is False, "refusal must not claim progress"


def test_allowed_workload_namespace_reaches_exactly_the_claimed_adapter(
    tmp_path: Path,
) -> None:
    payload = compound_command().model_dump(mode="json")
    payload["batched_steps"] = []
    payload["step"]["workload_ids"] = ["training/job/owned"]
    claimed = RemoteActionCommand.model_validate(payload)
    client = FakeClient(claimed)
    adapter = FakeNodeAdapter(all_succeed())
    executor = make_executor(tmp_path, client, [adapter])
    assert executor.run_once() == 1
    assert client.reported().status is RemoteCommandStatus.SUCCEEDED
    assert [context.step.workload_ids for context in adapter.contexts] == [
        ["training/job/owned"]
    ]
    assert client.progress_posts == []
