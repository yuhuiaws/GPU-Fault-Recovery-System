from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from gpu_fault.regional import (
    RegionalClusterRegistration,
    cluster_token_sha256,
)


LOGGER = logging.getLogger(__name__)


def sync_regional_cluster_registry(
    store: Any,
    values: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> list[RegionalClusterRegistration]:
    """Make the durable registry exactly match the validated config."""

    configured: list[RegionalClusterRegistration] = []
    configured_ids: set[str] = set()
    observed = now or datetime.now(timezone.utc)
    for value in values:
        item = dict(value)
        token = item.pop("token", None)
        if token:
            item["token_sha256"] = cluster_token_sha256(str(token))
        registration = RegionalClusterRegistration(**item)
        if registration.synthetic and not registration.is_active(observed):
            LOGGER.warning(
                "skipped expired synthetic regional cluster registration: %s",
                registration.cluster_id,
            )
            continue
        configured.append(registration)
        configured_ids.add(registration.cluster_id)

    removed: list[str] = []
    for cluster_id in store.list_regional_cluster_ids():
        if cluster_id in configured_ids:
            continue
        store.delete_regional_cluster(cluster_id)
        removed.append(cluster_id)
    if removed:
        LOGGER.warning(
            "removed regional cluster registrations absent from "
            "the declarative registry: %s",
            ", ".join(sorted(removed)),
        )
    for registration in configured:
        store.save_regional_cluster(registration)
    return sorted(
        configured,
        key=lambda item: item.cluster_id,
    )
