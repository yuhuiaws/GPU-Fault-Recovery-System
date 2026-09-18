from __future__ import annotations

import logging
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

from gpu_fault.plugins import PluginGroup, discover_plugins, load_plugin
from scripts.component_wheels import component_definition

ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize("source", ["project", "control_plane"])
def test_shipped_builtin_entry_points_are_discovered_without_importing(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    entries = (
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "entry-points"
        ]
        if source == "project"
        else component_definition(source).entry_points
    )
    points = metadata.EntryPoints(
        metadata.EntryPoint(name=name, value=value, group=group)
        for group, declared in entries.items()
        for name, value in declared.items()
    )

    def unexpected_load(_entry: metadata.EntryPoint) -> object:
        pytest.fail("discovery must not import any plugin")

    monkeypatch.setattr(metadata, "entry_points", lambda: points)
    monkeypatch.setattr(metadata.EntryPoint, "load", unexpected_load)

    for group, declared in entries.items():
        actual = discover_plugins(group)
        assert {name: point.value for name, point in actual.items()} == declared


def test_shipped_diagnostic_inconclusive_builder_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.notifications.diagnostic_inconclusive import (
        DiagnosticInconclusiveEmailBuilder,
    )

    points = metadata.EntryPoints(
        [
            metadata.EntryPoint(
                name="diagnostic-inconclusive",
                value=(
                    "gpu_fault.notifications.diagnostic_inconclusive:"
                    "DiagnosticInconclusiveEmailBuilder"
                ),
                group=PluginGroup.NOTIFICATION_BUILDERS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: points)

    assert (
        load_plugin(PluginGroup.NOTIFICATION_BUILDERS, "diagnostic-inconclusive")
        is DiagnosticInconclusiveEmailBuilder
    )


def test_duplicate_diagnostic_builder_is_rejected_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = metadata.EntryPoints(
        metadata.EntryPoint(
            name="diagnostic-inconclusive",
            value=f"{module}:builder",
            group=PluginGroup.NOTIFICATION_BUILDERS,
        )
        for module in ("first", "second")
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: points)

    def unexpected_load(_entry: metadata.EntryPoint) -> object:
        pytest.fail("duplicate plugins must be refused before import")

    monkeypatch.setattr(metadata.EntryPoint, "load", unexpected_load)
    with pytest.raises(RuntimeError, match="duplicate plugin"):
        discover_plugins(PluginGroup.NOTIFICATION_BUILDERS)


@pytest.mark.parametrize(
    "group",
    [
        PluginGroup.COLLECTOR_SINKS,
        PluginGroup.WORKFLOW_ADAPTERS,
        PluginGroup.NOTIFICATION_BUILDERS,
        PluginGroup.METRIC_CONTRIBUTORS,
    ],
)
def test_installed_metadata_cannot_grant_new_plugin_trust(
    monkeypatch: pytest.MonkeyPatch, group: PluginGroup
) -> None:
    points = metadata.EntryPoints(
        [
            metadata.EntryPoint(
                name="unapproved-extension",
                value="external_package:factory",
                group=group,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: points)

    assert discover_plugins(group) == {}
    with pytest.raises(LookupError, match="not on the allowlist"):
        load_plugin(group, "unapproved-extension")


def test_notification_builtin_does_not_widen_the_metrics_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = metadata.EntryPoints(
        [
            metadata.EntryPoint(
                name="diagnostic-inconclusive",
                value="external_package:factory",
                group=PluginGroup.METRIC_CONTRIBUTORS,
            )
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: points)

    assert discover_plugins(PluginGroup.METRIC_CONTRIBUTORS) == {}
    with pytest.raises(LookupError, match="not on the allowlist"):
        load_plugin(PluginGroup.METRIC_CONTRIBUTORS, "diagnostic-inconclusive")
