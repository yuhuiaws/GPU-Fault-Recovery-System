"""Fresh-process collection and guard checks with optional drivers unavailable."""

from __future__ import annotations

import argparse
import gzip
import importlib
import importlib.abc
import json
import sys
from pathlib import Path
from typing import Any

DRIVERS = frozenset(
    {"psycopg", "psycopg2", "psycopg_pool", "psycopg_binary", "psycopg_c"}
)


class MissingPostgresDrivers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.partition(".")[0] in DRIVERS:
            raise ModuleNotFoundError(
                "optional PostgreSQL driver intentionally unavailable", name=fullname
            )


def block_drivers() -> None:
    for name in tuple(sys.modules):
        if name.partition(".")[0] in DRIVERS:
            del sys.modules[name]
    sys.meta_path.insert(0, MissingPostgresDrivers())
    for name in DRIVERS:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
        else:
            raise AssertionError("optional driver import blocker was bypassed")


class Inventory:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.nodeids: list[str] = []
        self.errors: list[dict[str, str]] = []
        self.skips: dict[str, str] = {}
        self.phases: dict[str, dict[str, str]] = {}

    def pytest_collection_finish(self, session: Any) -> None:
        self.nodeids = sorted(item.nodeid for item in session.items)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.errors.append(
                {"nodeid": report.nodeid, "detail": str(report.longrepr)[-3000:]}
            )
        elif report.skipped:
            self.skips[report.nodeid] = str(report.longrepr)

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.phases.setdefault(report.nodeid, {})[report.when] = report.outcome
        if report.failed:
            self.errors.append(
                {"nodeid": report.nodeid, "detail": str(report.longrepr)[-3000:]}
            )

    def pytest_sessionfinish(self, session: Any) -> None:
        value = {
            "exitstatus": int(session.exitstatus),
            "nodeids": self.nodeids,
            "errors": self.errors,
            "skips": self.skips,
            "phases": self.phases,
        }
        with gzip.open(self.output, "wt", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
        print(
            json.dumps(
                {
                    "exitstatus": value["exitstatus"],
                    "collected": len(self.nodeids),
                    "errors": len(self.errors),
                    "skipped_collectors": sorted(self.skips),
                },
                sort_keys=True,
            )
        )


def protect_guard_probe() -> Any:
    import builtins
    import os
    import socket
    import subprocess

    import pytest

    from scripts.e2e.regional import ha011_resources

    patch = pytest.MonkeyPatch()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("optional-driver probe reached the lower safety boundary")

    for module, name in (
        (subprocess, "run"),
        (os, "system"),
        (socket.socket, "connect"),
        (ha011_resources, "run_fixture_command"),
    ):
        patch.setattr(module, name, forbidden)
    try:
        import psycopg
    except ImportError:
        pass
    else:
        patch.setattr(psycopg, "connect", forbidden)
    original_open = builtins.open

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).startswith(
            "/etc/gpu-fault/"
        ):
            forbidden()
        return original_open(path, *args, **kwargs)

    patch.setattr(builtins, "open", guarded_open)
    return patch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-drivers", action="store_true")
    parser.add_argument("--guard")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("targets", nargs="+")
    options = parser.parse_args()
    if options.block_drivers:
        block_drivers()

    import pytest

    inventory = Inventory(options.output)
    arguments = ["-o", "addopts=", "-p", "no:terminal", "-p", "no:cacheprovider"]
    patch = None
    if options.guard or options.run_tests:
        if options.guard:
            importlib.import_module(options.guard)
            arguments.extend(["-p", options.guard])
        patch = protect_guard_probe()
        arguments.append("--noconftest")
    else:
        arguments.append("--collect-only")
    try:
        return int(pytest.main([*arguments, *options.targets], plugins=[inventory]))
    finally:
        if patch is not None:
            patch.undo()


if __name__ == "__main__":
    raise SystemExit(main())
