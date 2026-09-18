from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import RegionalRegistryMember, RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import (
    active_registry_member_ids,
    registry_revision_converged,
)

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
STALE_SECONDS = 90


def revision(required: list[str]) -> RegionalRegistryRevision:
    return RegionalRegistryRevision.build(
        generation=5,
        registrations=[],
        previous_generation=4,
        required_member_ids=required,
        reason="synthetic registry convergence",
        created_at=NOW - timedelta(seconds=120),
    )


def member(
    member_id: str,
    target: RegionalRegistryRevision,
    *,
    age: float = 0,
    ready: bool = True,
    generation: int = 5,
    digest_matches: bool = True,
) -> RegionalRegistryMember:
    return RegionalRegistryMember(
        member_id=member_id,
        service_role="control-worker",
        release_id="synthetic-registry-release",
        generation=generation,
        content_sha256=target.content_sha256 if digest_matches else "b" * 64,
        ready=ready,
        started_at=NOW - timedelta(minutes=5),
        last_seen_at=NOW - timedelta(seconds=age),
    )


@pytest.mark.parametrize(
    "age,converged",
    [
        pytest.param(89, False, id="fresh"),
        pytest.param(90, False, id="at-stale-boundary"),
        pytest.param(90.000001, True, id="past-stale-boundary"),
        pytest.param(120, True, id="departed"),
    ],
)
def test_required_laggard_stops_blocking_only_after_its_heartbeat_expires(
    age: float, converged: bool
) -> None:
    target = revision(["survivor", "departed"])
    members = [
        member("survivor", target),
        member("departed", target, age=age, generation=4),
    ]

    assert (
        registry_revision_converged(
            target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
        )
        is converged
    ), "only a provably stale member may leave a live fleet's barrier"
    assert active_registry_member_ids(
        members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ) == (["survivor"] if converged else ["departed", "survivor"]), (
        "convergence and fleet liveness must use the same inclusive boundary"
    )


@pytest.mark.parametrize("required", [[], ["survivor"]], ids=["empty", "populated"])
@pytest.mark.parametrize(
    "ready,generation,digest_matches,age",
    [
        pytest.param(False, 5, True, 0, id="not-ready"),
        pytest.param(True, 4, True, 0, id="same-digest-old-generation"),
        pytest.param(True, 6, True, 0, id="newer-generation"),
        pytest.param(True, 5, False, 0, id="wrong-digest"),
        pytest.param(True, 5, True, -1, id="future-heartbeat"),
    ],
)
def test_new_active_members_cannot_bypass_the_publish_time_barrier(
    required: list[str], ready: bool, generation: int, digest_matches: bool, age: float
) -> None:
    target = revision(required)
    members = [
        member("survivor", target),
        member(
            "newcomer",
            target,
            ready=ready,
            generation=generation,
            digest_matches=digest_matches,
            age=age,
        ),
    ]

    assert not registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ), "every active process must provide a fresh ready ACK for the exact revision"


@pytest.mark.parametrize("required", [[], ["survivor"]], ids=["empty", "populated"])
def test_new_active_members_that_ack_the_revision_do_not_block(
    required: list[str],
) -> None:
    target = revision(required)
    members = [member("survivor", target), member("newcomer", target)]

    assert registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ), "post-publish arrivals may join the ACK set without changing the revision"
    assert target.required_member_ids == required, (
        "the publish-time membership snapshot must remain immutable"
    )


def test_missing_required_row_is_not_departure_evidence() -> None:
    target = revision(["survivor", "missing"])
    members = [member("survivor", target)]

    assert not registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ), "absence must not invent a last heartbeat or a safe departure"


@pytest.mark.parametrize("required", [[], ["departed"]], ids=["empty", "populated"])
@pytest.mark.parametrize("rows_present", [False, True], ids=["no-rows", "stale-rows"])
def test_only_an_empty_required_set_and_no_known_members_converges_without_an_ack(
    required: list[str], rows_present: bool
) -> None:
    target = revision(required)
    members = [member("departed", target, age=120)] if rows_present else []

    assert registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ) is (not required and not members), "a known fleet needs a live revision witness"


def test_a_replacement_can_ack_after_every_publish_time_member_departed() -> None:
    target = revision(["departed"])
    members = [
        member("departed", target, age=120, generation=4),
        member("replacement", target),
    ]

    assert registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ), "a live replacement proves the revision after the old process expires"


@pytest.mark.parametrize(
    "ready,generation,digest_matches",
    [
        pytest.param(False, 5, True, id="not-ready"),
        pytest.param(True, 4, True, id="same-digest-old-generation"),
        pytest.param(True, 6, True, id="newer-generation"),
        pytest.param(True, 5, False, id="wrong-digest"),
    ],
)
def test_live_required_members_still_need_an_exact_ready_ack(
    ready: bool, generation: int, digest_matches: bool
) -> None:
    target = revision(["required"])
    members = [
        member(
            "required",
            target,
            ready=ready,
            generation=generation,
            digest_matches=digest_matches,
        )
    ]

    assert not registry_revision_converged(
        target, members, observed_at=NOW, stale_seconds=STALE_SECONDS
    ), "departed-member handling must not weaken ready/generation/digest checks"
