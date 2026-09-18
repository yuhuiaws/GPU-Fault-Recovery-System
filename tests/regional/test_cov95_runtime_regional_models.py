from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.regional import (
    RegionalClusterRegistration,
    RegionalExecutorReadinessRequest,
    RegionalRegistryRevision,
    RemoteCommandClaimRequest,
    RemoteCommandProgress,
)
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A, T0
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"agent_endpoint_allowed_cidrs": ["invalid"]}, "valid networks"),
        ({"agent_endpoint_allowed_cidrs": []}, "requires agent endpoint CIDRs"),
        ({"token_sha256": "z" * 64}, "must be hexadecimal"),
        ({"synthetic": True}, "requires run ID and expiration"),
        (
            {
                "synthetic": True,
                "synthetic_run_id": "unit",
                "synthetic_expires_at": T0.replace(tzinfo=None),
            },
            "include timezone",
        ),
        ({"synthetic_run_id": "unit"}, "non-synthetic"),
        ({"synthetic_expires_at": T0}, "non-synthetic"),
        (
            {
                "retiring_token_sha256": "b" * 64,
                "token_rotation_expires_at": T0 + timedelta(days=1),
                "updated_at": T0.replace(tzinfo=None),
            },
            "timezone-aware updated_at",
        ),
    ],
)
def test_cluster_registration_rejects_unproved_network_token_and_synthetic_scope(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RegionalClusterRegistration.model_validate(
            {**CLUSTER_A.model_dump(mode="python"), **changes}
        )


@pytest.mark.parametrize(
    "method", ["is_active", "accepts_token", "accepted_token_slots"]
)
def test_registration_time_decisions_refuse_ambiguous_naive_clocks(method: str) -> None:
    with pytest.raises(ValueError, match="requires timezone"):
        getattr(CLUSTER_A, method)(T0.replace(tzinfo=None))


def test_synthetic_registration_expiry_removes_both_activity_and_authentication() -> (
    None
):
    registration = RegionalClusterRegistration.model_validate(
        {
            **CLUSTER_A.model_dump(mode="python"),
            "synthetic": True,
            "synthetic_run_id": "unit-owned",
            "synthetic_expires_at": T0 + timedelta(seconds=1),
        }
    )
    assert registration.is_active(T0) is True
    assert registration.accepts_token(T0) is True
    assert registration.is_active(T0 + timedelta(seconds=1)) is False
    assert registration.accepts_token(T0 + timedelta(seconds=1)) is False
    assert registration.token_matches("unrelated", T0 + timedelta(seconds=1)) is False


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("cluster-duplicate", "cluster IDs must be unique"),
        ("member-duplicate", "member IDs must be unique"),
        ("generation", "generation must increase"),
        ("digest", "content digest mismatch"),
    ],
)
def test_revision_model_rejects_ambiguous_or_changed_commit_identity(
    defect: str, message: str
) -> None:
    revision = RegionalRegistryRevision.build(
        generation=2,
        previous_generation=1,
        registrations=[CLUSTER_A],
        required_member_ids=["unit"],
        reason="unit",
        created_at=T0,
    )
    payload = revision.model_dump(mode="python")
    if defect == "cluster-duplicate":
        payload["registrations"] *= 2
    elif defect == "member-duplicate":
        payload["required_member_ids"] *= 2
    elif defect == "generation":
        payload["previous_generation"] = 2
    else:
        payload["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        RegionalRegistryRevision.model_validate(payload)


@pytest.mark.parametrize(
    "model", [RemoteCommandClaimRequest, RegionalExecutorReadinessRequest]
)
@pytest.mark.parametrize("owners", [[""], [" owner"], ["owner "], ["owner", "owner"]])
def test_executor_owner_contract_rejects_empty_untrimmed_or_duplicate_owners(
    model: Any, owners: list[str]
) -> None:
    with pytest.raises(ValueError, match="trimmed|unique"):
        model(executor_id="unit", execution_owners=owners)


@pytest.mark.parametrize("index", ["-1", "wrong", "1.5"])
def test_compound_progress_requires_unambiguous_step_indexes(index: str) -> None:
    with pytest.raises(ValueError, match="step indexes"):
        RemoteCommandProgress(
            executor_id="unit",
            lease_token="unit",
            batched_results={index: {"status": "SUCCEEDED", "details": {}}},
        )
