from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceStatus,
)
from gpu_fault.store import InMemoryStore, SqliteStore


def _resource(
    *,
    resource_id: str = "sg-123",
    status: InstallationResourceStatus = InstallationResourceStatus.ACTIVE,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id="site-a",
        resource_key="aws/nlb/security-group",
        resource_type="security_group",
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
        status=status,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture(params=("memory", "sqlite"))
def installation_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    store = SqliteStore(str(tmp_path / "installation-resources.db"))
    try:
        yield store
    finally:
        store.close()


def test_installation_resource_store_supports_status_updates(
    installation_store,
) -> None:
    resource = _resource()
    installation_store.save_installation_resource(resource)

    updated = resource.model_copy(
        update={
            "status": InstallationResourceStatus.DELETE_PENDING,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    installation_store.save_installation_resource(updated)

    assert (
        installation_store.get_installation_resource(
            "site-a", "aws/nlb/security-group"
        ).status
        is InstallationResourceStatus.DELETE_PENDING
    )
    assert installation_store.list_installation_resources("site-a") == [updated], (
        "installation resource list did not return the updated record"
    )


def test_installation_resource_identity_is_immutable(installation_store) -> None:
    installation_store.save_installation_resource(_resource())

    with pytest.raises(ValueError, match="identity cannot change"):
        installation_store.save_installation_resource(_resource(resource_id="sg-456"))
