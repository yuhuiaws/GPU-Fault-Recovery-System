from __future__ import annotations

import configparser
import json
import signal
import socket
import sqlite3
import subprocess
import time
from collections import deque
from pathlib import Path

import pytest

from gpu_fault.admin import api_budget as budget
from tests.admin._cov95_api_support import ShimTransport
from tests.admin.test_api_budget_handoff_model import PARENT, ProcessModel, seed_lender


@pytest.fixture
def shim(monkeypatch, tmp_path):
    transport = ShimTransport()
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("aws", "kubectl"):
        path = tools / name
        path.touch(mode=0o700)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.delenv(budget.ROOT_ENV, raising=False)
    transport.install(monkeypatch, ["aws", "--version"])
    return transport


@pytest.mark.parametrize("arguments", [[], ["not-a-cli"], ["/tmp/unknown"]])
def test_shim_rejects_unsupported_entry_before_transport(shim, monkeypatch, arguments):
    shim.install(monkeypatch, arguments)
    with pytest.raises(budget.ApiBudgetError, match="requires an AWS or kubectl"):
        budget.main()
    assert not shim.calls, "unsupported shim input started a process"
    assert not shim.handlers, "unsupported shim input installed signal handlers"


def test_shim_requires_scope_and_executable(shim, monkeypatch):
    with pytest.raises(budget.ApiBudgetError, match="lost its deployment scope"):
        budget.main()
    with budget.deployment_api_budget():
        monkeypatch.setenv("PATH", "/nonexistent-cov95-tools")
        with pytest.raises(budget.ApiBudgetError, match="tool is missing: aws"):
            budget.main()
    assert not shim.calls, "missing scope or executable started a process"


@pytest.mark.parametrize(
    "arguments,weight,telemetry",
    [
        (["aws", "--version"], 1, 0),
        (["aws", "help"], 1, 0),
        (["aws", "sts", "get-caller-identity"], 1, 1),
        (["aws", "s3", "cp", "fake-source", "fake-target"], 4, 1),
        (["kubectl", "version"], 1, 0),
    ],
)
def test_shim_propagates_exit_status_and_charges_backend(
    shim, monkeypatch, arguments, weight, telemetry
):
    shim.install(monkeypatch, arguments)
    with budget.deployment_api_budget(), budget.api_phase("cov95/shim"):
        assert budget.main() == 7
        command, options = shim.calls[0]
        assert command[1:] == arguments[1:]
        assert Path(command[0]).name == arguments[0]
        assert options["start_new_session"] is True
        assert callable(options["preexec_fn"]), "shim lost pre-exec ownership binding"
        assert options["stdout"] is options["stderr"] is None
        stats = budget.statistics("cov95/shim")
        backend = arguments[0]
        assert stats["peak_admitted_weight"][backend] == weight
        assert stats["backends"][backend]["commands"] == 1
        assert stats["backends"][backend]["finished_commands"] == 1
        assert stats["backends"][backend]["telemetry_expected_commands"] == telemetry
        assert options["env"][budget.PARENT_ENV]
        assert not shim.signals, "completed shim command was signalled for cleanup"


@pytest.mark.parametrize("has_default", [False, True])
def test_transfer_configuration_preserves_unrelated_profiles(
    shim, monkeypatch, tmp_path, has_default
):
    content = (
        ("[default]\nregion=us-east-1\n" if has_default else "")
        + "[profile example]\nregion=us-west-2\ns3=\n max_concurrent_requests=99\n"
        " addressing_style=path\n[sso-session example]\nsso_region=us-west-2\n"
    )
    source = tmp_path / "config"
    source.write_text(content)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(source))
    shim.install(monkeypatch, ["aws", "s3", "ls"])
    with budget.deployment_api_budget():
        assert budget.main() == 7
        target = Path(shim.calls[0][1]["env"]["AWS_CONFIG_FILE"])
        parser = configparser.RawConfigParser()
        parser.read(target)
        assert parser["profile example"]["region"] == "us-west-2"
        assert parser["sso-session example"] == {"sso_region": "us-west-2"}
        for section in ("default", "profile example"):
            s3 = configparser.RawConfigParser()
            s3.read_string("[s3]\n" + parser[section]["s3"])
            assert s3["s3"]["max_concurrent_requests"] == "4"
            assert s3["s3"]["preferred_transfer_client"] == "classic"
        assert "addressing_style = path" in parser["profile example"]["s3"]
        assert target.stat().st_mode & 0o077 == 0
        assert source.read_text() == content


@pytest.mark.parametrize("content", ["no-section", "[default]\ns3=\n not-an-option"])
def test_invalid_transfer_configuration_never_starts_cli(
    shim, monkeypatch, tmp_path, content
):
    source = tmp_path / "config"
    source.write_text(content)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(source))
    shim.install(monkeypatch, ["aws", "s3", "ls"])
    with budget.deployment_api_budget():
        with pytest.raises(budget.ApiBudgetError, match="bound AWS transfer"):
            budget.main()
        assert budget.statistics()["backends"] == {}
        assert not shim.calls, "invalid transfer configuration started a CLI"
        assert source.read_text() == content


@pytest.mark.parametrize("escalate", [False, True])
def test_shim_interruption_reaps_before_releasing_capacity(shim, escalate):
    shim.wait_failures.append(budget.ApiBudgetError("modeled interruption"))
    if escalate:
        shim.wait_failures.append(subprocess.TimeoutExpired(["fake"], 5))
    with budget.deployment_api_budget():
        with pytest.raises(budget.ApiBudgetError, match="modeled interruption"):
            budget.main()
        assert shim.signals == [signal.SIGTERM] + ([signal.SIGKILL] if escalate else [])
        stats = budget.statistics()["backends"]["aws"]
        assert stats["commands"] == stats["finished_commands"] == 1


def test_shim_recovers_capacity_between_wait_polls(shim):
    shim.wait_failures.append(subprocess.TimeoutExpired(["fake"], 0.1))
    with budget.deployment_api_budget():
        assert budget.main() == 7
    assert len(shim.waits) == 2
    assert all(0 < timeout <= 0.1 for timeout in shim.waits), (
        "CLI polling exceeded its bounded interval"
    )


@pytest.mark.parametrize(
    "failure", [OSError("spawn failed"), subprocess.SubprocessError("binding failed")]
)
def test_shim_spawn_failure_does_not_leak_admission(shim, failure):
    shim.spawn_error = failure
    with budget.deployment_api_budget():
        with pytest.raises(type(failure), match=str(failure)):
            budget.main()
        with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone() == (0,)
        assert budget.statistics()["backends"]["aws"]["finished_commands"] == 1


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_shim_signal_handlers_fail_closed(shim, signum):
    with budget.deployment_api_budget():
        budget.main()
        with pytest.raises(budget.ApiBudgetError, match="interrupted"):
            shim.handlers[signum](signum, None)


@pytest.mark.parametrize(
    "scenario", ["success", "retry-read", "oversized", "missing-pipe"]
)
def test_nested_shim_returns_output_only_after_lender_resumes(
    shim, monkeypatch, scenario
):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        seed_lender(root, 4)
        model = ProcessModel(root / "budget.sqlite3")
        shim.os.getpid = model.os.getpid
        shim.os.pidfd_open = model.os.pidfd_open
        shim.os.close = model.os.close
        shim.signal.pidfd_send_signal = model.signal.pidfd_send_signal
        monkeypatch.setattr(budget, "Path", model.path)
        monkeypatch.setenv(budget.PARENT_ENV, PARENT)
        if scenario == "retry-read":
            shim.reads[901].appendleft(BlockingIOError())
        elif scenario == "oversized":
            shim.reads[901] = deque([b"x" * 65536] * 129 + [b""])
        elif scenario == "missing-pipe":
            shim.missing_pipes = True
        if scenario in {"oversized", "missing-pipe"}:
            with pytest.raises(budget.ApiBudgetError, match="output"):
                budget.main()
            assert shim.output.buffer.getvalue() == b""
            assert shim.errors.buffer.getvalue() == b""
            assert shim.signals == [signal.SIGTERM]
        else:
            assert budget.main() == 7
            assert shim.output.buffer.getvalue() == b"out\n"
            assert shim.errors.buffer.getvalue() == b"err\n"
        assert model.signals == [signal.SIGSTOP, signal.SIGCONT]
        assert model.state == "S"
        assert budget.statistics()["backends"]["aws"]["finished_commands"] == 1


@pytest.mark.parametrize(
    "backend,weight", [("unknown", 1), ("aws", 0), ("aws", 9), ("http", 5)]
)
def test_invalid_capacity_is_rejected_without_a_lease(backend, weight):
    with budget.deployment_api_budget():
        with pytest.raises(budget.ApiBudgetError, match="invalid.*capacity"):
            with budget.api_slot(backend, weight=weight):
                pytest.fail("invalid capacity was admitted")
        assert budget.statistics()["backends"] == {}


@pytest.mark.parametrize(
    "kind",
    ["relative", "absent", "symlink", "public", "database-public", "database-symlink"],
)
def test_budget_storage_identity_is_checked_at_public_boundary(
    tmp_path, monkeypatch, kind
):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        with monkeypatch.context() as context:
            if kind in {"database-public", "database-symlink"}:
                path = root / "budget.sqlite3"
                if kind == "database-public":
                    path.chmod(0o644)
                else:
                    saved = root / "saved.sqlite3"
                    path.rename(saved)
                    path.symlink_to(saved)
            else:
                path = tmp_path / "owned"
                if kind == "relative":
                    path = Path("relative")
                elif kind != "absent":
                    path.mkdir(mode=0o755 if kind == "public" else 0o700)
                    if kind == "symlink":
                        link = tmp_path / "link"
                        link.symlink_to(path, target_is_directory=True)
                        path = link
                context.setenv(budget.ROOT_ENV, str(path))
            with pytest.raises(budget.ApiBudgetError, match="private|identity"):
                budget.budget_root()
        if kind == "database-public":
            path.chmod(0o600)
        elif kind == "database-symlink":
            path.unlink()
            saved.rename(path)


@pytest.mark.parametrize("state", ["unknown", None])
def test_incompatible_lease_states_do_not_allow_new_work(state):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        with sqlite3.connect(root / "budget.sqlite3") as database:
            database.execute(
                "INSERT INTO leases(id,state) VALUES('invalid',?)", (state,)
            )
        with pytest.raises(budget.ApiBudgetProtocolError, match="incompatible"):
            with budget.api_slot("aws"):
                pytest.fail("unknown handoff state allowed work")
        with sqlite3.connect(root / "budget.sqlite3") as database:
            database.execute("DELETE FROM leases WHERE id='invalid'")


def test_environment_phase_labels_and_unbudgeted_calls(monkeypatch):
    monkeypatch.delenv(budget.ROOT_ENV, raising=False)
    assert budget.api_environment(None) is None
    assert budget.api_environment({"PATH": "example"}) == {"PATH": "example"}
    with budget.api_slot("http") as identifier:
        assert identifier is None
    with budget.deployment_api_budget():
        root = budget.budget_root()
        with budget.api_phase("invalid label containing spaces"):
            with budget.api_slot("http"):
                pass
        assert budget.statistics("unclassified")["backends"]["http"]["commands"] == 1
        values = budget.api_environment({"PATH": f"{root}/bin:/example:{root}/bin"})
        assert values["PATH"] == f"{root}/bin:/example"
        assert budget.resolve_tool("definitely-not-a-tool", path="/example") is None


def test_csm_discards_malformed_messages_and_counts_only_valid_events():
    with budget.deployment_api_budget():
        root = budget.budget_root()
        with budget.api_slot("aws") as identifier:
            common = {"Version": 1, "ClientId": identifier}
            malformed = [
                b"not-json",
                b"[]",
                json.dumps({**common, "Version": 2}).encode(),
                json.dumps({**common, "ClientId": 7}).encode(),
                json.dumps({**common, "ClientId": "not-a-lease"}).encode(),
                json.dumps({**common, "Type": "unknown"}).encode(),
                *(
                    json.dumps(
                        {**common, "Type": "ApiCall", "AttemptCount": count}
                    ).encode()
                    for count in (None, True, -1, 101, "2")
                ),
            ]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                address = ("127.0.0.1", int((root / "csm-port").read_text()))
                for payload in malformed:
                    sender.sendto(payload, address)
                sender.sendto(
                    json.dumps({**common, "Type": "ApiCallAttempt"}).encode(), address
                )
                sender.sendto(
                    json.dumps(
                        {**common, "Type": "ApiCall", "AttemptCount": 3}
                    ).encode(),
                    address,
                )
            deadline = time.monotonic() + 3
            while budget.statistics()["backends"]["aws"]["sdk_calls"] != 1:
                assert time.monotonic() < deadline, "valid telemetry was not collected"
                time.sleep(0.01)
            stats = budget.statistics()["backends"]["aws"]
            assert (
                stats["sdk_calls"],
                stats["sdk_attempts"],
                stats["sdk_attempt_records"],
                stats["sdk_retries"],
            ) == (1, 3, 1, 2)
