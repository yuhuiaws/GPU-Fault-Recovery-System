from __future__ import annotations

from enum import StrEnum
from importlib import metadata
from typing import Any


class PluginGroup(StrEnum):
    # Each entry point is a ``gpu_fault.collector_registry.CollectorDescriptor``
    # named after its CLI command; the collector CLI merges them at start-up.
    COLLECTORS = "gpu_fault.collectors"
    COLLECTOR_SINKS = "gpu_fault.collector_sinks"
    WORKFLOW_ADAPTERS = "gpu_fault.workflow_adapters"
    NOTIFICATION_BUILDERS = "gpu_fault.notification_builders"
    METRIC_CONTRIBUTORS = "gpu_fault.metric_contributors"


def discover_plugins(
    group: PluginGroup | str,
) -> dict[str, metadata.EntryPoint]:
    group_name = str(group)
    selected = metadata.entry_points().select(group=group_name)
    result: dict[str, metadata.EntryPoint] = {}
    for entry_point in selected:
        if entry_point.name in result:
            raise RuntimeError(f"duplicate plugin {group_name}:{entry_point.name}")
        result[entry_point.name] = entry_point
    return result


def load_plugin(
    group: PluginGroup | str,
    name: str,
) -> Any:
    plugins = discover_plugins(group)
    try:
        entry_point = plugins[name]
    except KeyError as exc:
        raise LookupError(f"plugin {group}:{name} is not installed") from exc
    return entry_point.load()
