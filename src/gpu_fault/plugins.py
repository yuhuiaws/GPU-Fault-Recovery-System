from __future__ import annotations

import logging
from enum import StrEnum
from importlib import metadata
from typing import Any

LOGGER = logging.getLogger(__name__)


class PluginGroup(StrEnum):
    # Each entry point is a ``gpu_fault.collector_registry.CollectorDescriptor``
    # named after its CLI command; the collector CLI merges them at start-up.
    COLLECTORS = "gpu_fault.collectors"
    COLLECTOR_SINKS = "gpu_fault.collector_sinks"
    WORKFLOW_ADAPTERS = "gpu_fault.workflow_adapters"
    NOTIFICATION_BUILDERS = "gpu_fault.notification_builders"
    METRIC_CONTRIBUTORS = "gpu_fault.metric_contributors"


# Loading a plugin executes its module in-process, so any importable entry
# point registered under one of these groups would run arbitrary code inside
# the control plane. These utility groups accept only shipped identifiers
# (the ``[project.entry-points.*]`` tables in pyproject.toml); installed metadata
# never grants trust. Sinks, adapters and notification builders still use
# explicit application composition. Only METRIC_CONTRIBUTORS is auto-discovered
# by the runtime, and it ships no entry-point plugins today (its built-ins use
# ``register()``), so its empty allowlist rejects every discovered entry point
# (security review M-26).
#
# COLLECTORS is deliberately excluded (``None`` == unrestricted): it is not
# auto-loaded in the control plane. A collector plugin only loads when selected
# by ``gpu-fault-collector <command>`` or explicitly validated with
# ``gpu-fault-collector validate-plugins``. Its entry-point name must equal the
# command (see ``collector_registry_with_plugins``); tightening it needs a
# config-driven allowlist wired through the collector CLI (see report).
_PLUGIN_ALLOWLIST: dict[str, frozenset[str] | None] = {
    PluginGroup.COLLECTORS.value: None,
    PluginGroup.COLLECTOR_SINKS.value: frozenset({"http", "sqs"}),
    PluginGroup.WORKFLOW_ADAPTERS.value: frozenset(
        {
            "control-plane-evidence",
            "gpu-validation",
            "hyperpod",
            "kubernetes",
            "managed-recovery",
            "node-action",
            "support",
        }
    ),
    PluginGroup.NOTIFICATION_BUILDERS.value: frozenset(
        {
            "dcgm-diagnostic",
            "diagnostic-inconclusive",
            "efa-rdma",
            "hardware-escalation",
            "hardware-inventory",
            "host-resource",
            "hyperpod-advisory",
            "not-applicable",
            "nvlink74-mechanical",
            "nvlink74-support",
            "restart-guard",
            "sxid-event",
            "warm-spare",
            "xid-investigatory",
        }
    ),
    PluginGroup.METRIC_CONTRIBUTORS.value: frozenset(),
}


def _allowlist(group_name: str) -> frozenset[str] | None:
    """Allowed identifiers for a group, or ``None`` when unrestricted.

    An unknown group is treated as an empty allowlist (fail closed).
    """
    if group_name not in _PLUGIN_ALLOWLIST:
        LOGGER.warning(
            "plugin group %s has no allowlist; refusing every plugin", group_name
        )
        return frozenset()
    return _PLUGIN_ALLOWLIST[group_name]


def discover_plugins(
    group: PluginGroup | str,
) -> dict[str, metadata.EntryPoint]:
    """Read allowlisted entry-point metadata; never import plugin modules."""

    group_name = str(group)
    allowed = _allowlist(group_name)
    selected = metadata.entry_points().select(group=group_name)
    result: dict[str, metadata.EntryPoint] = {}
    for entry_point in selected:
        if allowed is not None and entry_point.name not in allowed:
            LOGGER.warning(
                "refusing plugin %s:%s that is not on the allowlist",
                group_name,
                entry_point.name,
            )
            continue
        if entry_point.name in result:
            raise RuntimeError(f"duplicate plugin {group_name}:{entry_point.name}")
        result[entry_point.name] = entry_point
    return result


def load_plugin(
    group: PluginGroup | str,
    name: str,
) -> Any:
    group_name = str(group)
    allowed = _allowlist(group_name)
    if allowed is not None and name not in allowed:
        LOGGER.warning(
            "refusing to load plugin %s:%s that is not on the allowlist",
            group_name,
            name,
        )
        raise LookupError(f"plugin {group_name}:{name} is not on the allowlist")
    plugins = discover_plugins(group)
    try:
        entry_point = plugins[name]
    except KeyError as exc:
        raise LookupError(f"plugin {group_name}:{name} is not installed") from exc
    return entry_point.load()
