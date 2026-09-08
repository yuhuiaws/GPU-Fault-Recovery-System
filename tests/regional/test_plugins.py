from __future__ import annotations

import logging
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
                name="http",
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    assert list(discover_plugins(PluginGroup.COLLECTOR_SINKS)) == ["http"]
    assert load_plugin(PluginGroup.COLLECTOR_SINKS, "http").__name__ == "HttpEventSink"


def test_duplicate_plugin_names_fail_closed(monkeypatch) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="http",
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
            metadata.EntryPoint(
                name="http",
                value="gpu_fault.collectors.sinks:SqsEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    with pytest.raises(RuntimeError, match="duplicate plugin"):
        discover_plugins(PluginGroup.COLLECTOR_SINKS)


def test_missing_plugin_is_explicit(monkeypatch) -> None:
    """An allowlisted-but-uninstalled plugin fails with an install hint."""
    monkeypatch.setattr(metadata, "entry_points", lambda: _EntryPoints())
    with pytest.raises(LookupError, match="is not installed"):
        load_plugin(PluginGroup.WORKFLOW_ADAPTERS, "kubernetes")


# --- M-26: only allowlisted plugin identifiers may load ---


def test_unallowlisted_plugin_is_refused_by_discovery(monkeypatch, caplog) -> None:
    """A rogue entry point in a known group must never be discovered/loaded."""
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="http",  # legitimate
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
            metadata.EntryPoint(
                name="evil",  # attacker-registered
                value="attacker_pkg.payload:pwn",
                group=PluginGroup.COLLECTOR_SINKS,
            ),
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    with caplog.at_level(logging.WARNING):
        discovered = discover_plugins(PluginGroup.COLLECTOR_SINKS)

    assert list(discovered) == ["http"]
    assert "evil" in caplog.text


def test_unallowlisted_plugin_load_is_rejected(monkeypatch) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="evil",
                value="attacker_pkg.payload:pwn",
                group=PluginGroup.COLLECTOR_SINKS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    with pytest.raises(LookupError, match="not on the allowlist"):
        load_plugin(PluginGroup.COLLECTOR_SINKS, "evil")


def test_unknown_group_refuses_every_plugin(monkeypatch, caplog) -> None:
    """A group with no declared allowlist must load nothing (fail closed)."""
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="anything",
                value="attacker_pkg.payload:pwn",
                group="gpu_fault.not_a_real_group",
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    with caplog.at_level(logging.WARNING):
        assert discover_plugins("gpu_fault.not_a_real_group") == {}


def test_allowlisted_plugin_still_loads(monkeypatch) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(
                name="node-action",
                value="gpu_fault.collectors.sinks:HttpEventSink",
                group=PluginGroup.WORKFLOW_ADAPTERS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)
    assert list(discover_plugins(PluginGroup.WORKFLOW_ADAPTERS)) == ["node-action"]
    assert (
        load_plugin(PluginGroup.WORKFLOW_ADAPTERS, "node-action").__name__
        == "HttpEventSink"
    )
