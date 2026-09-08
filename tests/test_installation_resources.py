from __future__ import annotations

import pytest

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)


def _resource() -> InstallationResource:
    return InstallationResource(
        site_id="test-site",
        resource_key="rds/aurora",
        resource_type="rds_cluster",
        resource_id="gpu-fault-aurora",
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
    )


def _sealed_snapshot() -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[_resource()]
    )
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def test_a_sealed_snapshot_passes_the_provenance_binding() -> None:
    """M-12: a snapshot whose seal matches its content is authoritative."""

    snapshot = _sealed_snapshot()

    assert snapshot.require_source_binding() is snapshot


def test_an_unsealed_snapshot_is_refused_as_authoritative() -> None:
    """M-12: a record read back without a seal has no verifiable origin.

    ``source_sha256`` is optional so a fresh build can be minted then sealed, but
    a snapshot trusted as installed state must carry the seal -- otherwise its
    content could have been substituted wholesale between write and read.
    """

    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[_resource()]
    )

    assert snapshot.source_sha256 is None
    with pytest.raises(ValueError, match="not sealed"):
        snapshot.require_source_binding()


def test_a_reseated_seal_that_no_longer_matches_is_refused() -> None:
    """M-12: a seal that the content does not hash to is a hard rejection.

    ``model_copy`` (the path the builders use to seal) bypasses validation, so a
    tampered snapshot can carry a stale seal in memory; the load-time check must
    catch the mismatch the constructor never saw.
    """

    snapshot = _sealed_snapshot().model_copy(update={"source_sha256": "0" * 64})

    with pytest.raises(ValueError, match="digest mismatch"):
        snapshot.require_source_binding()
