"""Cluster token rotation through an overlapping accept-set.

Rotating a single-digest credential forces the control plane and the data plane
to switch at the same instant, and every executor, collector and node agent that
has not switched yet gets a 401 in between. These tests pin the overlap that
removes that window: the new token is current immediately, the old digest keeps
working until a bounded deadline, and letting the deadline lapse withdraws the
old credential instead of cutting off whoever already moved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import (
    MAX_TOKEN_ROTATION_WINDOW,
    TOKEN_SLOT_CURRENT,
    TOKEN_SLOT_RETIRING,
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    cluster_token_sha256,
    regional_registry_content_sha256,
)
from gpu_fault.regional_registry import sync_regional_cluster_registry
from tests._builders import asgi_client, build_context, build_store

# The app-level cases go through the real authentication path, which reads the
# wall clock, so the reference instant has to be the current one rather than a
# fixed date that would put every window in the past.
NOW = datetime.now(timezone.utc)
OLD_TOKEN = "o" * 40
NEW_TOKEN = "n" * 40
OTHER_TOKEN = "x" * 40


def rotating(
    *,
    current: str = NEW_TOKEN,
    retiring: str | None = OLD_TOKEN,
    updated_at: datetime = NOW,
    expires_at: datetime | None = None,
    **overrides,
) -> RegionalClusterRegistration:
    deadline = expires_at
    if retiring is not None and deadline is None:
        deadline = updated_at + timedelta(minutes=30)
    return RegionalClusterRegistration(
        cluster_id="cluster-a",
        region="us-west-2",
        hyperpod_cluster_name="hp-cluster-a",
        eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/cluster-a",
        token_sha256=cluster_token_sha256(current),
        retiring_token_sha256=(
            None if retiring is None else cluster_token_sha256(retiring)
        ),
        token_rotation_expires_at=deadline,
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        created_at=updated_at,
        updated_at=updated_at,
        **overrides,
    )


def context_for(registration: RegionalClusterRegistration):
    context = build_context(execution_token="e" * 32)
    context.regional_mode = True
    context.store.save_regional_cluster(registration)
    return context


def headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-GPU-Fault-Cluster-ID": "cluster-a"}


def authenticate(registration: RegionalClusterRegistration, token: str) -> int:
    """Return the status of a read-only cluster-token route for ``token``.

    The submission lookup is used rather than ``claim`` because it answers 200
    for every state in which the caller is authenticated, so the status carries
    only the authentication outcome.
    """

    async def scenario() -> int:
        async with asgi_client(context_for(registration)) as client:
            response = await client.get(
                "/v1/regional/executors/hyperpod-submissions",
                headers=headers(token),
                params={"cluster_name": "hp-cluster-a", "idempotency_key": "missing"},
            )
        return response.status_code

    return asyncio.run(scenario())


def test_overlap_window_accepts_both_tokens() -> None:
    registration = rotating()

    assert authenticate(registration, NEW_TOKEN) == 200
    assert authenticate(registration, OLD_TOKEN) == 200
    assert authenticate(registration, OTHER_TOKEN) == 403


def test_expired_rotation_window_withdraws_only_the_retiring_token() -> None:
    # The window closed an hour ago. Nothing else changed, so the executors that
    # already moved keep working and only the withdrawn credential stops.
    registration = rotating(
        updated_at=NOW - timedelta(hours=2), expires_at=NOW - timedelta(hours=1)
    )

    assert authenticate(registration, NEW_TOKEN) == 200
    assert authenticate(registration, OLD_TOKEN) == 403


def test_registration_without_rotation_accepts_only_the_current_token() -> None:
    registration = rotating(retiring=None)

    assert authenticate(registration, NEW_TOKEN) == 200
    assert authenticate(registration, OLD_TOKEN) == 403


@pytest.mark.parametrize(
    "state", [RegionalClusterLifecycle.REVOKED, RegionalClusterLifecycle.ROLLED_BACK]
)
def test_revoked_cluster_rejects_every_slot(state: RegionalClusterLifecycle) -> None:
    registration = rotating(lifecycle_state=state)

    assert authenticate(registration, NEW_TOKEN) == 403
    assert authenticate(registration, OLD_TOKEN) == 403


def test_retiring_slot_use_is_reported_so_rotation_can_be_finished(caplog) -> None:
    """The only runtime signal that a caller still holds the old token.

    Dropping the retiring digest while any executor, collector or node agent is
    still presenting it locks that cluster out, and nothing else in the control
    plane says whether that is the case.
    """

    registration = rotating()

    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.factory"):
        assert authenticate(registration, OLD_TOKEN) == 200
    retiring_warnings = [
        record.getMessage()
        for record in caplog.records
        if "retiring token" in record.getMessage()
    ]

    assert len(retiring_warnings) == 1, caplog.text
    assert "cluster-a" in retiring_warnings[0]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.factory"):
        assert authenticate(registration, NEW_TOKEN) == 200

    assert [
        record.getMessage()
        for record in caplog.records
        if "retiring token" in record.getMessage()
    ] == []


def test_clusters_view_redacts_both_digests_but_publishes_the_window() -> None:
    registration = rotating()

    async def scenario() -> dict:
        async with asgi_client(context_for(registration)) as client:
            response = await client.get(
                "/v1/regional/clusters",
                headers={"X-GPU-Fault-Execution-Token": "e" * 32},
            )
        assert response.status_code == 200
        return response.json()[0]

    item = asyncio.run(scenario())

    assert "token_sha256" not in item, item
    assert "retiring_token_sha256" not in item, item
    assert item["retiring_token_sha256_present"] is True
    assert item["retiring_token_sha256_length"] == 64
    assert (
        datetime.fromisoformat(item["token_rotation_expires_at"])
        == registration.token_rotation_expires_at
    )


def test_matched_slot_names_the_credential_that_opened_the_door() -> None:
    registration = rotating()

    assert registration.matched_token_slot(NEW_TOKEN, NOW) == TOKEN_SLOT_CURRENT
    assert registration.matched_token_slot(OLD_TOKEN, NOW) == TOKEN_SLOT_RETIRING
    assert registration.matched_token_slot(OTHER_TOKEN, NOW) is None
    assert registration.accepted_token_slots(NOW) == (
        TOKEN_SLOT_CURRENT,
        TOKEN_SLOT_RETIRING,
    )
    assert registration.accepted_token_slots(NOW + timedelta(hours=1)) == (
        TOKEN_SLOT_CURRENT,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"retiring_token_sha256": cluster_token_sha256(OLD_TOKEN)},
            "requires both retiring_token_sha256",
        ),
        (
            {"token_rotation_expires_at": NOW + timedelta(minutes=5)},
            "requires both retiring_token_sha256",
        ),
        (
            {
                "retiring_token_sha256": cluster_token_sha256(NEW_TOKEN),
                "token_rotation_expires_at": NOW + timedelta(minutes=5),
            },
            "must differ from token_sha256",
        ),
        (
            {
                "retiring_token_sha256": "z" * 64,
                "token_rotation_expires_at": NOW + timedelta(minutes=5),
            },
            "must be hexadecimal",
        ),
        (
            {
                "retiring_token_sha256": cluster_token_sha256(OLD_TOKEN),
                "token_rotation_expires_at": datetime(2026, 9, 3, 13),
            },
            "must include timezone",
        ),
        (
            {
                "retiring_token_sha256": cluster_token_sha256(OLD_TOKEN),
                "token_rotation_expires_at": NOW - timedelta(minutes=5),
            },
            "must be after updated_at",
        ),
        (
            {
                "retiring_token_sha256": cluster_token_sha256(OLD_TOKEN),
                "token_rotation_expires_at": (
                    NOW + MAX_TOKEN_ROTATION_WINDOW + timedelta(minutes=1)
                ),
            },
            "must not exceed 7 days",
        ),
    ],
)
def test_rotation_metadata_is_validated_at_registration(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RegionalClusterRegistration(
            cluster_id="cluster-a",
            region="us-west-2",
            hyperpod_cluster_name="hp-cluster-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:123456789012:cluster/cluster-a",
            token_sha256=cluster_token_sha256(NEW_TOKEN),
            agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
            created_at=NOW,
            updated_at=NOW,
            **overrides,
        )


def test_rotation_window_is_bounded_against_the_stored_update_time() -> None:
    """A stored revision must stay loadable however old it is.

    Registry revisions are immutable and their content digest is re-verified on
    every load, so a window bounded against the wall clock would turn an old
    revision illegal days after the fact and take the control plane down with it.
    """

    old = rotating(
        updated_at=NOW - timedelta(days=400),
        expires_at=NOW - timedelta(days=400) + timedelta(minutes=30),
    )

    assert old.accepted_token_slots(NOW) == (TOKEN_SLOT_CURRENT,)
    assert (
        RegionalClusterRegistration.model_validate(old.model_dump(mode="json")) == old
    )


def test_registry_digest_ignores_unset_rotation_fields() -> None:
    """Revisions published before rotation existed keep their digest.

    The content digest covers the registration dump, so a new optional field
    would otherwise invalidate every stored revision and fail every load with a
    digest mismatch.
    """

    plain = rotating(retiring=None)
    payload = plain.model_dump(mode="json")
    del payload["retiring_token_sha256"]
    del payload["token_rotation_expires_at"]
    legacy = hashlib.sha256(
        json.dumps([payload], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert regional_registry_content_sha256([plain]) == legacy
    assert regional_registry_content_sha256([rotating()]) != legacy


def test_registry_sync_hashes_a_plaintext_retiring_token() -> None:
    store = build_store()
    # `updated_at` defaults to construction time, and the window is bounded
    # against it, so the deadline has to be relative to now rather than fixed.
    observed = datetime.now(timezone.utc)
    deadline = observed + timedelta(minutes=30)

    configured = sync_regional_cluster_registry(
        store,
        [
            {
                "cluster_id": "cluster-a",
                "region": "us-west-2",
                "hyperpod_cluster_name": "hp-cluster-a",
                "eks_cluster_arn": (
                    "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a"
                ),
                "token": NEW_TOKEN,
                "retiring_token": OLD_TOKEN,
                "token_rotation_expires_at": deadline.isoformat(),
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
        now=observed,
    )

    assert len(configured) == 1, configured
    registration = configured[0]

    assert registration.token_sha256 == cluster_token_sha256(NEW_TOKEN)
    assert registration.retiring_token_sha256 == cluster_token_sha256(OLD_TOKEN)
    assert registration.matched_token_slot(OLD_TOKEN, observed) == TOKEN_SLOT_RETIRING
