"""``register_profile`` writes a Runtime Profile version once and never again.

``POST /v1/runtime-profiles`` used to store the compiled profile with
``save_profile``, an upsert keyed by ``profile_version``: a second registration
of a live version with other content replaced the policy every Agent, collector
and event naming that version reads. A version is immutable once written. The
same policy again (the registration anchor ``cluster_id`` aside, as the release
engine compares) answers the stored profile; other content is refused and
nothing is written. The three backends answer alike; on PostgreSQL the compare
and the insert run under one advisory lock, so two replicas registering the
same version at once cannot both write.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    EffectiveRuntimeProfile,
    Environment,
)
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import NotFoundError, RuntimeProfileConflict
from tests._builders import build_store, copy_model
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

VERSION = "unit-profile-v1"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "profile-guard.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _profile(
    *,
    cluster_id: str = "cluster-a",
    mode: CapabilityMode = CapabilityMode.OWN,
    version: str = VERSION,
) -> EffectiveRuntimeProfile:
    return EffectiveRuntimeProfile(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        profile_version=version,
        capabilities=[
            EffectiveCapability(
                capability=CapabilityName.GPU_RESET,
                mode=mode,
                owner="gpu-fault-node-agent",
                adapter="node-action",
            ),
            EffectiveCapability(
                capability=CapabilityName.SUPPORT_ESCALATION,
                mode=CapabilityMode.OWN,
                owner="gpu-fault-support-escalation",
                adapter="fixed-template-email",
            ),
        ],
    )


def test_a_new_version_is_stored_and_answered(store) -> None:
    profile = _profile()

    assert store.register_profile(profile) == profile
    assert store.get_profile(VERSION) == profile


def test_the_same_content_again_answers_the_stored_profile(store) -> None:
    store.register_profile(_profile())

    assert store.register_profile(_profile()) == _profile(), (
        "a repeated registration is idempotent"
    )
    assert store.get_profile(VERSION) == _profile()


def test_the_same_policy_under_another_anchor_keeps_the_first_anchor(store) -> None:
    store.register_profile(_profile(cluster_id="cluster-a"))

    answered = store.register_profile(_profile(cluster_id="cluster-b"))

    assert answered.cluster_id == "cluster-a", (
        "the anchor is the member a version was first registered under; the "
        "same policy registered under another member answers the stored row"
    )
    assert store.get_profile(VERSION).cluster_id == "cluster-a"


def test_other_content_under_a_stored_version_is_refused_and_writes_nothing(
    store,
) -> None:
    store.register_profile(_profile())
    observe = _profile(mode=CapabilityMode.OBSERVE)

    with pytest.raises(RuntimeProfileConflict) as refused:
        store.register_profile(observe)

    message = str(refused.value)
    assert VERSION in message and "capabilities (gpuReset)" in message, message
    assert "new profile_version" in message, message
    assert store.get_profile(VERSION) == _profile(), "the stored policy is unchanged"
    with pytest.raises(NotFoundError):
        store.get_profile("unit-profile-v2")
    assert store.register_profile(
        copy_model(observe, profile_version="unit-profile-v2")
    ) == copy_model(observe, profile_version="unit-profile-v2"), (
        "the changed policy registers under a new version"
    )


def test_concurrent_registrations_of_one_version_store_exactly_one(store) -> None:
    """Eight writers race on one version with eight different policies."""

    owners = [f"owner-{index}" for index in range(8)]
    candidates = [
        _profile().model_copy(
            update={
                "capabilities": [
                    item.model_copy(update={"owner": owner})
                    for item in _profile().capabilities
                ]
            }
        )
        for owner in owners
    ]
    start = threading.Barrier(len(candidates))

    def register(candidate: EffectiveRuntimeProfile):
        start.wait(timeout=30)
        try:
            return store.register_profile(candidate)
        except RuntimeProfileConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        outcomes = list(pool.map(register, candidates))

    stored = [item for item in outcomes if isinstance(item, EffectiveRuntimeProfile)]
    refused = [item for item in outcomes if isinstance(item, RuntimeProfileConflict)]
    assert len(stored) == 1 and len(refused) == len(candidates) - 1, outcomes
    assert store.get_profile(VERSION) == stored[0], (
        "the one registration that won is the version every reader sees"
    )
