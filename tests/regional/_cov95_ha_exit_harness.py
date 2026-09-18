from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import processor_factory
from gpu_fault.processor import coordinator
from gpu_fault.store import SqliteStore
from scripts.e2e.regional import run_ha008_processor_exit_acceptance as acceptance
from scripts.e2e.regional import run_ha008_processor_exit_probe as probe
from tests.regional._cov95_ha001_harness import Clock


class ProbeExit(BaseException):
    def __init__(self, code: int) -> None:
        self.code = code


class InlineThread:
    def __init__(self, *, target: Any, **kwargs: Any) -> None:
        self.target = target

    def start(self) -> None:
        self.target()


class TrackedEnvironment(dict[str, str]):
    def __init__(self, patch: pytest.MonkeyPatch) -> None:
        super().__init__(os.environ)
        self.patch = patch

    def update(self, values: Any = (), /, **kwargs: str) -> None:
        for key, value in dict(values, **kwargs).items():
            self.patch.setenv(key, value)
            self[key] = value


class ExitHarness:
    """Real processor/SQLite protocol; process boundaries and clocks are simulated."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch = monkeypatch
        self.clock = Clock()
        self.epoch = datetime.now(timezone.utc)
        self.stores: list[SqliteStore] = []
        self.commands: list[list[str]] = []
        self.pid = 11
        self.returncode_override: int | None = None
        self.takeover_override: dict[str, Any] = {}
        self.omit_claim = False
        self.takeover_failure = False
        monkeypatch.setattr(processor_factory, "Thread", InlineThread)
        monkeypatch.setattr(
            processor_factory, "time", SimpleNamespace(sleep=self.clock.sleep)
        )
        monkeypatch.setattr(
            processor_factory, "os", SimpleNamespace(**{**vars(os), "_exit": self.exit})
        )
        monkeypatch.setattr(
            probe,
            "os",
            SimpleNamespace(
                **{
                    **vars(os),
                    "environ": TrackedEnvironment(monkeypatch),
                    "getpid": lambda: self.pid,
                }
            ),
        )
        monkeypatch.setattr(
            probe,
            "logging",
            SimpleNamespace(INFO=logging.INFO, basicConfig=lambda **kw: None),
        )
        monkeypatch.setattr(probe, "SqliteStore", self.open_store)
        monkeypatch.setattr(probe, "datetime", SimpleNamespace(now=self.now))
        monkeypatch.setattr(probe, "time", SimpleNamespace(sleep=self.clock.sleep))
        monkeypatch.setattr(acceptance, "time", SimpleNamespace(sleep=self.clock.sleep))
        monkeypatch.setattr(acceptance, "datetime", SimpleNamespace(now=self.now))
        monkeypatch.setattr(
            acceptance,
            "subprocess",
            SimpleNamespace(**{**vars(subprocess), "run": self.run}),
        )

    def now(self, tz: Any = None) -> datetime:
        return self.epoch + timedelta(seconds=self.clock.now)

    def open_store(self, database: str) -> SqliteStore:
        store = SqliteStore(database)
        self.stores.append(store)
        return store

    def exit(self, code: int) -> None:
        raise ProbeExit(code)

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(argv)
        if self.omit_claim:
            return subprocess.CompletedProcess(argv, 70, "", "")
        takeover = "--takeover" in argv
        if takeover and self.takeover_failure:
            return subprocess.CompletedProcess(argv, 1, "", "unit takeover failed")
        self.pid = 22 if takeover else 11
        out, err = io.StringIO(), io.StringIO()
        # An inline child needs fresh logging state, just like a new interpreter.
        manager = logging.Manager(logging.RootLogger(logging.INFO))
        logger = manager.getLogger(coordinator.LOGGER.name)
        handler = logging.StreamHandler(err)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        start = len(self.stores)
        code = 0
        try:
            with self.patch.context() as local, redirect_stdout(out):
                local.setattr(sys, "argv", argv[1:])
                local.setattr(coordinator, "LOGGER", logger)
                try:
                    probe.main()
                except ProbeExit as exc:
                    code = exc.code
        finally:
            logger.removeHandler(handler)
            handler.close()
            for store in self.stores[start:]:
                store.close()
        stdout = out.getvalue()
        if takeover:
            value = json.loads(stdout.splitlines()[-1])
            value.update(self.takeover_override)
            stdout = json.dumps(value)
        elif self.returncode_override is not None:
            code = self.returncode_override
        return subprocess.CompletedProcess(argv, code, stdout, err.getvalue())


def child_arguments(root: Path, *, fail_release: bool = False) -> list[str]:
    return [
        sys.executable,
        probe.__file__,
        "--database",
        str(root / "processor.db"),
        "--claim-file",
        str(root / "claim.json"),
        *(["--fail-release"] if fail_release else []),
    ]
