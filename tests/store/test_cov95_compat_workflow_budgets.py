from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.store import RemediationBudgetError, WorkflowLeaseError
from gpu_fault.store.shared.errors import StaleFencingTokenError
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)
from tests.store._cov95_compat_workflows import save_pair


def test_workflow_claim_keeps_one_live_owner_and_rotates_epoch_at_expiry(compat_store):
    _, workflow = save_pair(compat_store, "lease")
    first = compat_store.claim_workflow(
        workflow.request_id,
        "owner",
        workflow.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
    )
    with pytest.raises(StaleFencingTokenError, match="fencing"):
        compat_store.claim_workflow(
            workflow.request_id, "owner", workflow.fencing_token + 1, now=NOW
        )
    with pytest.raises(WorkflowLeaseError, match="another executor"):
        compat_store.claim_workflow(
            workflow.request_id, "other-owner", workflow.fencing_token, now=NOW
        )
    assert compat_store.get_workflow(workflow.request_id) == first, (
        "refused owner or generation changes must preserve the current lease"
    )
    renewed = compat_store.claim_workflow(
        workflow.request_id,
        "owner",
        workflow.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
    )
    assert renewed.execution_epoch == first.execution_epoch, (
        "the same live owner must not consume another execution epoch"
    )
    replacement = compat_store.claim_workflow(
        workflow.request_id,
        "other-owner",
        workflow.fencing_token,
        now=NOW + timedelta(seconds=30),
        lease_duration=timedelta(seconds=30),
    )
    assert replacement.execution_owner_id == "other-owner", (
        "the exact expiry boundary must allow a new executor to take responsibility"
    )
    assert replacement.execution_epoch == first.execution_epoch + 1, (
        "taking over an expired lease must fence the previous executor"
    )


@pytest.mark.parametrize("invalid", ["unclaimed", "owner", "expired"])
def test_budget_extension_requires_a_current_execution_lease(compat_store, invalid):
    _, workflow = save_pair(compat_store, "invalid-extension")
    if invalid != "unclaimed":
        compat_store.claim_workflow(
            workflow.request_id,
            "owner",
            workflow.fencing_token,
            now=NOW,
            lease_duration=timedelta(seconds=30),
        )
    before = compat_store.get_workflow(workflow.request_id)

    with pytest.raises(WorkflowLeaseError, match="lease"):
        compat_store.extend_remediation_budget(
            workflow.request_id,
            "other-owner" if invalid == "owner" else "owner",
            {"rack": 1},
            now=NOW + timedelta(seconds=30) if invalid == "expired" else NOW,
        )
    assert compat_store.get_workflow(workflow.request_id) == before, (
        f"the {invalid} executor must not change budget or execution state"
    )


def test_budget_extension_refusal_preserves_held_scopes_until_capacity_is_available(
    compat_store,
):
    _, target = save_pair(compat_store, "target")
    _, peer = save_pair(compat_store, "peer", node_ids=["other-node"])
    claimed = compat_store.claim_workflow(
        target.request_id,
        "target-owner",
        target.fencing_token,
        now=NOW,
        lease_duration=timedelta(minutes=5),
        remediation_budget_claims={"region": 2, "node:local": 1},
    )
    compat_store.claim_workflow(
        peer.request_id,
        "peer-owner",
        peer.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims={"region": 2, "rack": 1},
    )

    with pytest.raises(RemediationBudgetError) as raised:
        compat_store.extend_remediation_budget(
            target.request_id, "target-owner", {"rack": 1}, now=NOW
        )
    assert raised.value.scope == "rack", "the refusal must name the saturated new scope"
    assert compat_store.get_workflow(target.request_id) == claimed, (
        "a failed extension must not release existing scopes or advance the lease epoch"
    )

    extended = compat_store.extend_remediation_budget(
        target.request_id, "target-owner", {"rack": 1}, now=NOW + timedelta(seconds=30)
    )
    assert extended.remediation_budget_claims == ["node:local", "rack", "region"], (
        "an extension must add the new scope without discarding existing claims"
    )
    assert extended.remediation_budget_limits == {
        "region": 2,
        "node:local": 1,
        "rack": 1,
    }, "the stored budget must retain the limits of both old and added scopes"
    assert extended.execution_epoch == claimed.execution_epoch, (
        "adding capacity must not alter execution identity"
    )
    assert (
        compat_store.extend_remediation_budget(
            target.request_id,
            "target-owner",
            {"rack": 1},
            now=NOW + timedelta(seconds=30),
        )
        == extended
    ), "repeated extension must not count the workflow against itself"
    assert compat_store.get_workflow(target.request_id) == extended, (
        "the returned extension must be the persisted workflow state"
    )
