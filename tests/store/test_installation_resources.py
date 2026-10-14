from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceConflict,
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


def _row(key: str, resource_id: str, **updates: object) -> InstallationResource:
    return _resource(resource_id=resource_id).model_copy(
        update={"resource_key": key, **updates}
    )


def test_a_registry_batch_is_saved_all_or_nothing(installation_store) -> None:
    """One sync, one transaction (batch 91): a row that would change a recorded
    identity refuses the whole batch, so no row ahead of it is rewritten."""

    first, second = _row("aws/a", "sg-a"), _row("aws/b", "sg-b")
    assert installation_store.save_installation_resources([first, second]) == [
        first,
        second,
    ]
    pending = first.model_copy(
        update={"status": InstallationResourceStatus.DELETE_PENDING}
    )
    with pytest.raises(InstallationResourceConflict) as refused:
        installation_store.save_installation_resources(
            [pending, _row("aws/b", "sg-replaced"), _row("aws/c", "sg-c")]
        )
    assert refused.value.conflicts == (("aws/b", ("resource_id",)),)
    assert installation_store.list_installation_resources("site-a") == [
        first,
        second,
    ], "a refused batch wrote some of its rows"


def test_an_identity_conflict_names_every_refused_key_and_field(
    installation_store,
) -> None:
    installation_store.save_installation_resources(
        [_row("aws/a", "sg-a"), _row("aws/b", "sg-b")]
    )
    with pytest.raises(ValueError, match="identity cannot change") as refused:
        installation_store.save_installation_resources(
            [
                _row(
                    "aws/a",
                    "sg-a",
                    dependencies=["aws/b"],
                    ownership=InstallationResourceOwnership.EXTERNAL,
                    delete_policy=InstallationResourceDeletePolicy.PRESERVE,
                ),
                _row("aws/b", "sg-b", resource_arn="arn:aws:ec2:::sg-b"),
            ]
        )
    assert str(refused.value) == (
        "installation resource identity cannot change: "
        "aws/a (ownership, delete_policy, dependencies); aws/b (resource_arn)"
    )
    assert refused.value.response_content()["conflicts"] == [
        {
            "resource_key": "aws/a",
            "fields": ["ownership", "delete_policy", "dependencies"],
        },
        {"resource_key": "aws/b", "fields": ["resource_arn"]},
    ]


def test_a_single_row_save_names_its_conflicting_fields(installation_store) -> None:
    installation_store.save_installation_resource(_resource())
    with pytest.raises(InstallationResourceConflict) as refused:
        installation_store.save_installation_resource(
            _resource(resource_id="sg-456").model_copy(update={"region": "us-west-2"})
        )
    assert refused.value.conflicts == (
        ("aws/nlb/security-group", ("resource_id", "region")),
    )


def test_a_registry_batch_refuses_a_key_named_twice(installation_store) -> None:
    with pytest.raises(ValueError, match="keys must be unique"):
        installation_store.save_installation_resources(
            [_row("aws/a", "sg-a"), _row("aws/a", "sg-a")]
        )
    assert installation_store.list_installation_resources() == []
