from __future__ import annotations

from importlib import metadata

import pytest

from gpu_fault.plugins import PluginGroup, discover_plugins, load_plugin


class _EntryPoints(list):
    def select(self, *, group: str):
        return [item for item in self if item.group == group]


def test_plugin_discovery_and_loading(monkeypatch) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="example",
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    assert list(discover_plugins(PluginGroup.COLLECTOR_SINKS)) == ["example"]
    assert (
        load_plugin(PluginGroup.COLLECTOR_SINKS, "example").__name__ == "HttpEventSink"
    )


def test_duplicate_plugin_names_fail_closed(monkeypatch) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="same",
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
            metadata.EntryPoint(
                name="same",
                value="gpu_fault.collectors.sinks:SqsEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    with pytest.raises(RuntimeError, match="duplicate plugin"):
        discover_plugins(PluginGroup.COLLECTOR_SINKS)


def test_missing_plugin_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "entry_points", lambda: _EntryPoints())
    with pytest.raises(LookupError, match="is not installed"):
        load_plugin(PluginGroup.WORKFLOW_ADAPTERS, "missing")
