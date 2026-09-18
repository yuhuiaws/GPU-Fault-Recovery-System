"""Collector discovery is metadata-only; loading follows explicit selection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from gpu_fault import collector_registry, collectors_cli
from gpu_fault.collector_registry import CollectorDescriptor, discover_collector_plugins
from gpu_fault.collectors.sinks import EventSink
from gpu_fault.plugins import PluginGroup

BUILT: list[str] = []
RAN: list[str] = []
NOT_A_FACTORY = 17


class EntryPoints(list[metadata.EntryPoint]):
    def select(self, *, group: str) -> list[metadata.EntryPoint]:
        return [point for point in self if point.group == group]


class Collector:
    def __init__(self, command: str) -> None:
        self.command = command

    def run(self) -> None:
        RAN.append(self.command)


def build_collector(sink: EventSink, arguments: argparse.Namespace) -> Collector:
    BUILT.append(arguments.command)
    return Collector(arguments.command)


def descriptor_for(name: str) -> CollectorDescriptor:
    return replace(
        collector_registry.COLLECTOR_REGISTRY["training-progress"],
        cli_command=name,
        export_name=None,
        factory=f"{__name__}:build_collector",
        needs_context=False,
    )


def install_plugins(
    monkeypatch: pytest.MonkeyPatch,
    descriptors: dict[str, object],
    *,
    names: list[str] | None = None,
) -> list[str]:
    points = EntryPoints(
        metadata.EntryPoint(
            name=name,
            value=f"test_collector_plugin_{name}:descriptor",
            group=PluginGroup.COLLECTORS,
        )
        for name in (names if names is not None else descriptors)
    )
    loaded: list[str] = []

    def load(point: metadata.EntryPoint) -> object:
        loaded.append(point.name)
        value = descriptors[point.name]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(metadata, "entry_points", lambda: points)
    monkeypatch.setattr(metadata.EntryPoint, "load", load)
    return loaded


@pytest.fixture(autouse=True)
def no_collector_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    BUILT.clear()
    RAN.clear()

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("metadata-only commands must not construct a sink or context")

    monkeypatch.setattr(collectors_cli, "sink_from_environment", unexpected)
    monkeypatch.setattr(collectors_cli, "context_from_environment", unexpected)


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [
        (["--help"], 0),
        (["kernel", "--help"], 0),
        (["broken", "--help"], 0),
        (["validate-plugins", "--help"], 0),
        (["not-installed"], 2),
        (["broken", "--not-an-option"], 2),
    ],
)
def test_help_and_parse_errors_do_not_import_plugins(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], exit_code: int
) -> None:
    loaded = install_plugins(
        monkeypatch, {"broken": RuntimeError("unselected plugin import failed")}
    )
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", *arguments])
    with pytest.raises(SystemExit) as failure:
        collectors_cli.main()
    assert failure.value.code == exit_code
    assert loaded == []
    assert BUILT == RAN == []


def test_outbox_inspection_does_not_import_plugins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    loaded = install_plugins(
        monkeypatch, {"broken": RuntimeError("unselected plugin import failed")}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-collector",
            "outbox",
            "--outbox-path",
            str(tmp_path / "events.ndjson"),
            "stats",
        ],
    )
    collectors_cli.main()
    assert json.loads(capsys.readouterr().out)["depth"] == 0
    assert loaded == []
    assert BUILT == RAN == []


@pytest.mark.parametrize("command", ["kernel", "selected"])
def test_running_a_collector_loads_only_its_selected_plugin(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    loaded = install_plugins(
        monkeypatch,
        {
            "broken": RuntimeError("unselected plugin import failed"),
            "selected": descriptor_for("selected"),
        },
    )
    monkeypatch.setitem(
        collector_registry.COLLECTOR_REGISTRY,
        "kernel",
        replace(
            collector_registry.COLLECTOR_REGISTRY["kernel"],
            factory=f"{__name__}:build_collector",
            needs_context=False,
            needs_product_discovery=False,
        ),
    )
    monkeypatch.setattr(collectors_cli, "sink_from_environment", lambda: object())
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", command])
    collectors_cli.main()
    assert loaded == ([] if command == "kernel" else ["selected"])
    assert BUILT == RAN == [command]


@pytest.mark.parametrize("name", ["kernel", "outbox", "validate-plugins"])
def test_discovery_rejects_command_collisions_before_loading_any_module(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    loaded = install_plugins(
        monkeypatch,
        {"other": RuntimeError("must not import"), name: descriptor_for(name)},
    )
    with pytest.raises(RuntimeError, match="collides with a CLI command"):
        discover_collector_plugins()
    assert loaded == []


def test_duplicate_names_fail_during_metadata_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = install_plugins(
        monkeypatch, {"same": descriptor_for("same")}, names=["same", "same"]
    )
    with pytest.raises(RuntimeError, match="duplicate plugin"):
        discover_collector_plugins()
    assert loaded == []


def test_selected_registry_rejects_missing_plugins_without_importing_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = install_plugins(
        monkeypatch, {"broken": RuntimeError("unselected plugin import failed")}
    )
    with pytest.raises(LookupError, match="missing is not installed"):
        collector_registry.collector_registry_with_plugins(selected_command="missing")
    assert loaded == []


@pytest.mark.parametrize("command", ["selected", "validate-plugins"])
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("type", "must be a CollectorDescriptor"),
        ("name", "two must agree"),
        ("channel", "unknown collector channel"),
        ("factory", "not callable"),
    ],
)
def test_selected_loading_and_explicit_validation_keep_descriptor_checks(
    monkeypatch: pytest.MonkeyPatch, command: str, invalid: str, message: str
) -> None:
    descriptor: object = descriptor_for("selected")
    if invalid == "type":
        descriptor = object()
    elif invalid == "name":
        descriptor = descriptor_for("different-name")
    elif invalid == "channel":
        descriptor = replace(
            descriptor_for("selected"), channel_paths=("/v1/collector-events/unknown",)
        )
    elif invalid == "factory":
        descriptor = replace(
            descriptor_for("selected"), factory=f"{__name__}:NOT_A_FACTORY"
        )
        monkeypatch.setattr(collectors_cli, "sink_from_environment", lambda: object())
    loaded = install_plugins(monkeypatch, {"selected": descriptor})
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", command])
    with pytest.raises(RuntimeError, match=message):
        collectors_cli.main()
    assert loaded == ["selected"]
    assert BUILT == RAN == []


@pytest.mark.parametrize("via_cli", [False, True])
def test_explicit_validation_loads_all_plugins_without_running_collectors(
    monkeypatch: pytest.MonkeyPatch, via_cli: bool
) -> None:
    loaded = install_plugins(
        monkeypatch, {name: descriptor_for(name) for name in ("first", "second")}
    )
    if via_cli:
        monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", "validate-plugins"])
        collectors_cli.main()
    else:
        collector_registry.validate_collector_plugins()
    assert loaded == ["first", "second"]
    assert BUILT == RAN == []


def test_explicit_validation_refuses_an_unselected_broken_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = install_plugins(
        monkeypatch,
        {
            "selected": descriptor_for("selected"),
            "broken": RuntimeError("unselected plugin import failed"),
        },
    )
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", "validate-plugins"])
    with pytest.raises(RuntimeError, match="unselected plugin import failed"):
        collectors_cli.main()
    assert loaded == ["selected", "broken"]
    assert BUILT == RAN == []
