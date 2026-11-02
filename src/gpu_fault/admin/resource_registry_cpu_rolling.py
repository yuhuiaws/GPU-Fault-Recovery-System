"""The CPU rolling update in the installation ledger (batch 240).

Two kinds of record, both from the deploy's journal
(``cpu_rolling_update.load_journal``):

* every rollback alarm the deploy put, as a ``cloudwatch_alarm`` row
  (CREATED/DELETE, tagged ``gpu-fault:site-id``): uninstall deletes it and the
  residue gate checks it is gone. A renamed pair keeps its old names in the
  journal, so their rows -- and their deletion -- stay;
* the CPU instance group's pre-install ``ScheduledUpdateConfig``, recorded
  before the deploy first changed it, as ``pre_install_scheduled_update.<group>``
  attributes of the ``cpu_hyperpod`` row: uninstall restores it before the
  alarms go.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gpu_fault.admin.cpu_rolling_update import load_journal
from gpu_fault.admin.resource_records import record
from gpu_fault.admin.site import RenderedSite
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

__all__ = [
    "ALARM_TYPE",
    "PRE_INSTALL_PREFIX",
    "cpu_rolling_rows",
    "pre_install_records",
]

ALARM_TYPE = "cloudwatch_alarm"
PRE_INSTALL_PREFIX = "pre_install_scheduled_update."


def _journal(site: RenderedSite) -> Mapping[str, Any]:
    return load_journal(site.source.parent)


def cpu_rolling_rows(
    resources: Sequence[InstallationResource],
    site: RenderedSite,
    *,
    region: str,
    account_id: str,
) -> list[InstallationResource]:
    """``resources`` with the alarm rows added and the pre-install schedules
    laid onto the ``cpu_hyperpod`` row."""

    journal = _journal(site)
    rows = list(resources)
    for name, entry in sorted((journal.get("alarms") or {}).items()):
        if not isinstance(entry, Mapping) or not entry.get("arn"):
            continue
        rows.append(
            record(
                site_id=site.registry_site_id,
                resource_key=f"aws/cloudwatch/alarm/{name}",
                resource_type=ALARM_TYPE,
                resource_id=str(name),
                resource_arn=str(entry["arn"]),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                attributes={"purpose": "cpu-rolling-update-rollback-alarm"},
            )
        )
    pre_install = {
        f"{PRE_INSTALL_PREFIX}{group}": json.dumps(
            entry["pre_install"].get("scheduled_update_config"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for group, entry in sorted((journal.get("groups") or {}).items())
        if isinstance(entry, Mapping) and isinstance(entry.get("pre_install"), Mapping)
    }
    if not pre_install:
        return rows
    cluster = str(site.release_config["cpu_hyperpod_cluster_name"])
    return [
        item.model_copy(update={"attributes": {**item.attributes, **pre_install}})
        if item.resource_type == "cpu_hyperpod" and item.resource_id == cluster
        else item
        for item in rows
    ]


def pre_install_records(resources: Iterable[InstallationResource]) -> dict[str, Any]:
    """group -> recorded ``ScheduledUpdateConfig`` from a ledger's
    ``cpu_hyperpod`` row (what uninstall restores)."""

    found: dict[str, Any] = {}
    for item in resources:
        if item.resource_type != "cpu_hyperpod":
            continue
        for key, value in item.attributes.items():
            if key.startswith(PRE_INSTALL_PREFIX):
                found[key[len(PRE_INSTALL_PREFIX) :]] = json.loads(value)
    return found
