from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import attempt_observation, container_observation
from tests.execution.test_restart_safety import (
    BatchApi,
    PyTorchCustomApi,
    UnusedApi,
    restart_context,
)

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("source", ["same-job", "foreign-job", "latest-own-attempt"])
def test_unknown_source_gpu_count_is_inferred_only_from_the_authorized_job(
    source: str,
) -> None:
    store = InMemoryStore()
    store.save_attempt_observation(
        attempt_observation(
            "train-1" if source == "same-job" else "unrelated-job",
            "shared-attempt",
            NOW,
            containers=[
                container_observation(
                    "source-pod", "trainer", 0, "node-a", gpu_uuids=["GPU-observed"]
                )
            ],
        )
    )
    target_count = 2 if source == "latest-own-attempt" else 1
    if source == "latest-own-attempt":
        store.save_attempt_observation(
            attempt_observation(
                "train-1",
                "another-own-attempt",
                NOW,
                containers=[
                    container_observation(
                        "own-pod",
                        "trainer",
                        0,
                        "node-a",
                        gpu_uuids=["GPU-own-a", "GPU-own-b"],
                    )
                ],
            )
        )
    batch = BatchApi(target_count)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    context = restart_context(
        "source-binding",
        source_gpu_count=0,
        restart_budget=1,
        source_attempt_id="shared-attempt",
    )
    outcome = adapter.execute(context)
    if source == "foreign-job":
        assert outcome.status is WorkflowStepStatus.WAITING, outcome
        assert outcome.details["reason"] == "GPU_COUNT_CHANGED", outcome
        assert outcome.details["source_gpu_count"] == 0, outcome
        assert outcome.details["restart_submitted"] is False, outcome
        assert batch.created == {}, batch.created
    else:
        assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
        assert context.step.parameters["source_gpu_count"] == target_count, context
        assert len(batch.created) == 1, batch.created
    assert context.request.restart_authorization is not None, context.request
    assert context.request.restart_authorization.source_gpu_count == 0, (
        "observation refinement must not rewrite the signed authorization"
    )


@pytest.mark.parametrize("kind", ["job", "pytorchjob"])
def test_retry_lookup_transport_failure_cannot_authorize_a_new_workload(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = BatchApi(1)
    custom = PyTorchCustomApi()
    custom.workload["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    queried: list[str] = []
    if kind == "job":
        read_job = batch.read_namespaced_job

        def read(name: str, namespace: str) -> Any:
            queried.append(name)
            if name != "training-job":
                raise TimeoutError("fake retry lookup lost its response")
            return read_job(name, namespace)

        monkeypatch.setattr(batch, "read_namespaced_job", read)
    else:
        read_custom = custom.get_namespaced_custom_object

        def get(
            group: str, version: str, namespace: str, plural: str, name: str
        ) -> Any:
            queried.append(name)
            if name != "training-job":
                raise TimeoutError("fake retry lookup lost its response")
            return read_custom(group, version, namespace, plural, name)

        monkeypatch.setattr(custom, "get_namespaced_custom_object", get)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=custom, store=InMemoryStore()
    )
    context = restart_context(
        "retry-lookup",
        source_gpu_count=1 if kind == "job" else 24,
        restart_budget=1,
        workload_id=f"training/{kind}/training-job",
        source_attempt_id="attempt-a",
    )
    with pytest.raises(TimeoutError, match="retry lookup lost its response"):
        adapter.execute(context)
    assert len(queried) == 3 and queried[-1].startswith("training-job-r-"), queried
    assert batch.created == custom.created == {}, (batch.created, custom.created)
    assert custom.patches == [], custom.patches
