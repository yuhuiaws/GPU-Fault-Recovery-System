from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.regional import RegionalClusterLifecycle
from tests.app.test_route_tenancy_security import EXECUTION_TOKEN, regional_context
from tests.app_services._cov95_runtime_api import full_app
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime
from tests.regional._regional_support import TOKEN_A, registration

HEADERS = {"X-GPU-Fault-Execution-Token": EXECUTION_TOKEN}
NEW_TOKEN = "synthetic-new-cluster-token-" + "n" * 32


@pytest.mark.parametrize(
    "defect",
    ["not-join-state", "new-active", "active-to-pending", "draining-to-active", "path"],
)
def test_join_transition_refuses_unapproved_lifecycle_or_path_changes(
    defect: str,
) -> None:
    context = regional_context()
    current = registration("cluster-a", TOKEN_A)
    desired = RegionalClusterLifecycle.PENDING
    if defect == "not-join-state":
        desired = RegionalClusterLifecycle.DRAINING
    elif defect == "new-active":
        current = registration("cluster-new", NEW_TOKEN)
        desired = RegionalClusterLifecycle.ACTIVE
    elif defect == "draining-to-active":
        current = current.model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.DRAINING}
        )
        context.store.save_regional_cluster(current)
        desired = RegionalClusterLifecycle.ACTIVE
    with full_app(context) as (_app, client):
        before = context.store.get_regional_registry_head()
        path_cluster = "foreign" if defect == "path" else current.cluster_id
        response = client.post(
            f"/v1/regional/registry/clusters/{path_cluster}/transition",
            headers=HEADERS,
            json={
                "registration": current.model_dump(mode="json"),
                "lifecycle_state": desired.value,
                "reason": "unit rejected transition",
            },
        )
        assert response.status_code == (400 if defect == "path" else 409)
        assert context.store.get_regional_registry_head() == before


@pytest.mark.parametrize("exhausted", [False, True])
def test_join_conflicts_have_a_bounded_retry_and_do_not_publish_an_optimistic_head(
    monkeypatch: pytest.MonkeyPatch, exhausted: bool
) -> None:
    context = regional_context()
    new = registration("cluster-new", NEW_TOKEN)
    with full_app(context) as (_app, client):
        before = context.store.get_regional_registry_head()
        publish = context.store.publish_regional_registry_revision
        attempts = []

        def conflicted(revision: Any, **kwargs: Any) -> Any:
            attempts.append(revision)
            if exhausted or len(attempts) == 1:
                raise ValueError("synthetic competing registry writer")
            return publish(revision, **kwargs)

        monkeypatch.setattr(
            context.store, "publish_regional_registry_revision", conflicted
        )
        response = client.post(
            "/v1/regional/registry/clusters/cluster-new/transition",
            headers=HEADERS,
            json={
                "registration": new.model_dump(mode="json"),
                "lifecycle_state": "PENDING",
                "reason": "unit pending membership",
            },
        )
        if exhausted:
            assert response.status_code == 409
            assert "changed repeatedly" in response.json()["detail"]
            assert len(attempts) == 8
            assert context.store.get_regional_registry_head() == before
        else:
            assert response.status_code == 200
            assert response.json()["cluster_states"]["cluster-new"] == "PENDING"
            assert len(attempts) == 2
            assert (
                context.store.get_regional_registry_head().generation
                == before.generation + 1
            )
