from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from gpu_fault.regional import (
    RegionalClusterRegistration,
    cluster_token_sha256,
)
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

# The fields an operator configures. Lifecycle state and the two timestamps
# are runtime facts the durable registry owns, so they are excluded: a Secret
# re-read at start-up carries fresh timestamps and no lifecycle, and comparing
# the full revision digest would report drift on every boot.
_CONFIG_DIGEST_FIELDS = (
    "cluster_id",
    "region",
    "hyperpod_cluster_name",
    "eks_cluster_arn",
    "token_sha256",
    "retiring_token_sha256",
    "enabled",
    "allowed_namespaces",
    "agent_endpoint_allowed_cidrs",
)


def regional_registry_config_sha256(
    registrations: Iterable[RegionalClusterRegistration],
) -> str:
    """Digest of the operator-configured view of a registry.

    Used to tell whether the ``GPU_FAULT_REGIONAL_CLUSTERS_JSON`` Secret and
    the durable Aurora head still describe the same clusters (review H3).
    """

    payload = [
        {
            field: getattr(item, field)
            for field in _CONFIG_DIGEST_FIELDS
            if getattr(item, field) is not None
        }
        for item in sorted(registrations, key=lambda value: value.cluster_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def configured_regional_registrations(
    values: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> list[RegionalClusterRegistration]:
    """Normalize the validated Secret entries into registrations."""

    configured: list[RegionalClusterRegistration] = []
    observed = now or datetime.now(timezone.utc)
    for value in values:
        item = dict(value)
        token = item.pop("token", None)
        if token:
            item["token_sha256"] = cluster_token_sha256(str(token))
        # `retiring_token` lets the config carry both credentials in plaintext
        # during a rotation, so an operator never has to compute a digest by
        # hand to keep executors that have not been rolled yet authenticating.
        retiring_token = item.pop("retiring_token", None)
        if retiring_token:
            item["retiring_token_sha256"] = cluster_token_sha256(str(retiring_token))
        registration = RegionalClusterRegistration(**item)
        if registration.synthetic and not registration.is_active(observed):
            LOGGER.warning(
                "skipped expired synthetic regional cluster registration: %s",
                registration.cluster_id,
            )
            continue
        configured.append(registration)
    return sorted(configured, key=lambda item: item.cluster_id)


def sync_regional_cluster_registry(
    store: Any,
    values: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> list[RegionalClusterRegistration]:
    """Make the durable registry exactly match the validated config.

    Once a durable head exists it is the truth and the Secret is only a
    bootstrap seed; the caller compares the two with
    :func:`regional_registry_config_sha256` and reports drift.
    """

    try:
        head = store.get_regional_registry_head()
    except (AttributeError, NotFoundError):
        pass
    else:
        revision = store.get_regional_registry_revision(head.generation)
        if revision.content_sha256 != head.content_sha256:
            raise RuntimeError("durable regional registry head digest mismatch")
        return list(revision.registrations)

    configured = configured_regional_registrations(values, now=now)
    configured_ids = {registration.cluster_id for registration in configured}
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
    return configured
