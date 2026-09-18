from __future__ import annotations

import runpy
import signal
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import api_budget as budget
from gpu_fault.admin import native_http
from tests.admin import test_cov95_api_budget as existing
from tests.admin._cov95_admission_support import Clock
from tests.admin._cov95_api_support import ShimTransport
from tests.admin.test_api_budget_handoff_model import PARENT, ProcessModel, seed_lender

shim = existing.shim


@pytest.mark.parametrize("explicit", [False, True])
def test_tool_resolution_outside_a_budget_uses_only_the_requested_path(
    tmp_path, monkeypatch, explicit
):
    tool = tmp_path / "example"
    tool.touch(mode=0o700)
    monkeypatch.delenv(budget.ROOT_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert budget.resolve_tool("example", str(tmp_path) if explicit else None) == str(
        tool
    )


@pytest.mark.parametrize("module", [budget, native_http])
def test_isolated_stdlib_entry_loads_its_own_source_without_starting_work(
    module, monkeypatch
):
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = runpy.run_path(str(module.__file__), run_name="cov95_isolated_entry")
    assert callable(namespace["main"]), (
        "standalone worker did not expose its entrypoint"
    )
    assert sys.path[0] == str(Path(module.__file__).resolve().parents[2])


def test_standalone_http_entry_refuses_missing_deadline_before_any_transport(
    monkeypatch,
):
    transport = ShimTransport()
    assert native_http.current_deadline() is None
    monkeypatch.setitem(sys.modules, "subprocess", transport.subprocess)
    monkeypatch.setitem(sys.modules, "signal", transport.signal)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["http-entry", "request"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(native_http.__file__), run_name="__main__")
    assert result.value.code == 1
    assert transport.calls == []


@pytest.mark.parametrize("protocol_error", [False, True])
def test_standalone_api_entry_reports_only_bounded_errors(
    monkeypatch, capsys, protocol_error
):
    transport = ShimTransport()
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "owned API scope was not created"
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "subprocess", transport.subprocess)
            patch.setitem(sys.modules, "signal", transport.signal)
            patch.setattr(sys, "path", list(sys.path))
            patch.setattr(
                sys, "argv", ["api-entry", "aws" if protocol_error else "unknown"]
            )
            if protocol_error:
                with sqlite3.connect(root / "budget.sqlite3") as database:
                    database.execute("PRAGMA user_version=999")
            try:
                with pytest.raises(SystemExit) as result:
                    runpy.run_path(str(budget.__file__), run_name="__main__")
                assert result.value.code == 1
            finally:
                with sqlite3.connect(root / "budget.sqlite3") as database:
                    database.execute(
                        f"PRAGMA user_version={budget.BUDGET_PROTOCOL_VERSION}"
                    )
    message = capsys.readouterr().err
    assert (
        "incompatible inherited" if protocol_error else "bounded capacity/deadline"
    ) in message
    assert transport.calls == []


def test_api_scope_disappearance_before_admission_never_spawns_cli(
    shim, monkeypatch, tmp_path
):
    with budget.deployment_api_budget():
        with monkeypatch.context() as patch:

            def which(_name, **_options):
                patch.delenv(budget.ROOT_ENV)
                return str(tmp_path / "tools/aws")

            patch.setattr(budget, "shutil", SimpleNamespace(which=which))
            with pytest.raises(
                budget.ApiBudgetError, match="lost its deployment scope"
            ):
                budget.main()
        assert shim.calls == []


@pytest.mark.parametrize("when", ["during-drain", "after-eof"])
def test_nested_cli_output_deadline_keeps_spooled_output_private_and_restores_lender(
    shim, monkeypatch, when
):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "owned API scope was not created"
        seed_lender(root, 4)
        model = ProcessModel(root / "budget.sqlite3")
        shim.os.getpid = model.os.getpid
        shim.os.pidfd_open = model.os.pidfd_open
        shim.os.close = model.os.close
        shim.signal.pidfd_send_signal = model.signal.pidfd_send_signal
        clock = Clock()
        read = shim.read
        advanced = False

        def delayed(fd, size):
            nonlocal advanced
            chunk = read(fd, size)
            if not advanced and (
                when == "during-drain"
                and chunk
                or when == "after-eof"
                and not any(shim.reads.values())
            ):
                advanced = True
                clock.value += 3601
            return chunk

        shim.os.read = delayed
        monkeypatch.setattr(budget, "Path", model.path)
        monkeypatch.setattr(budget, "time", clock)
        monkeypatch.setenv(budget.PARENT_ENV, PARENT)
        with pytest.raises(budget.ApiBudgetError, match="output deadline expired"):
            budget.main()
        assert shim.output.buffer.getvalue() == b""
        assert shim.errors.buffer.getvalue() == b""
        assert model.signals == [signal.SIGSTOP, signal.SIGCONT]
