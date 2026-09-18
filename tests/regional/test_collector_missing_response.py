"""Missing responses cannot stand in for transport, ownership or cleanup proof."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.collector_env_restore import restore_collector_env
from scripts.e2e.regional.regional_commands import RegionalCommandTimeout
from tests.regional._host_probe_support import ProbeApi, host_probe

RUN_ID = "collect004-missing-unit-a1"
NONCE = "synthetic-private-owner-value"
PRIVATE = "synthetic-private-diagnostic"


@pytest.fixture
def owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[ProbeApi, host.HostProbeFixture]]:
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    probe = host_probe(tmp_path)
    probe.create()
    original = deepcopy(api.objects)
    api.calls.clear()
    try:
        yield api, probe
    finally:
        api.objects = original
        api.node_uid = "node-uid"
        api.after_execute = None
        api.read_error = False
        monkeypatch.setattr(host, "run_fixture_command", api.run)
        assert not any(probe.cleanup().values()), "fake probe cleanup failed"


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("transition", ["ready", "unknown", "unready", "terminated"])
def test_empty_response_is_distinct_even_when_ready_has_not_changed(
    monkeypatch: pytest.MonkeyPatch,
    owned: tuple[ProbeApi, host.HostProbeFixture],
    returncode: int,
    transition: str,
) -> None:
    api, probe = owned
    finished = False

    def after() -> None:
        nonlocal finished
        finished = True
        if transition == "terminated":
            api.objects["pod"]["status"]["phase"] = "Failed"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = api.run(command, **kwargs)
        if command[7:9] == ["get", "node"]:
            node = json.loads(result.stdout)
            ready = (
                {"unknown": "Unknown", "unready": "False"}.get(transition, "True")
                if finished
                else "True"
            )
            node["status"] = {
                "conditions": [{"type": "Ready", "status": ready}],
                "nodeInfo": {"bootID": "original-boot"},
            }
            result.stdout = json.dumps(node)
        return result

    monkeypatch.setattr(host, "run_fixture_command", run)
    api.probe_stdout, api.probe_returncode = "", returncode
    api.after_execute = after
    with pytest.raises(host.HostProbeMissingResponseError) as error:
        probe.execute("restore-collector-env", "--run-id", RUN_ID)
    assert not isinstance(error.value, host.HostProbeTransportError), (
        "test_empty_response_is_distinct_even_when_ready_has_not_changed: expected no isinstance(error.value, host.HostProbeTransportError)"
    )
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("authorization", [True, False, None])
def test_public_probe_and_wrapper_only_defer_with_fresh_authorization(
    owned: tuple[ProbeApi, host.HostProbeFixture],
    returncode: int,
    authorization: bool | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api, probe = owned
    api.probe_stdout = ""
    api.probe_stderr = f"{NONCE} {PRIVATE}"
    api.probe_returncode = returncode
    checks: list[int] = []

    def binding() -> bool:
        checks.append(sum(args[0] == "exec" for args, _ in api.calls))
        return authorization is True

    callback = binding if authorization is not None else None
    if authorization is True:
        result = restore_collector_env(
            probe, RUN_ID, owner_nonce=NONCE, reboot_transition=callback
        )
        assert result["deferred"] is True
        assert result["restored"] is False
        assert result["cleanup_verified"] is False
        assert result["error_type"] == "HostProbeMissingResponseError"
        assert PRIVATE not in json.dumps(result)
        assert NONCE not in json.dumps(result)
    else:
        with pytest.raises(host.HostProbeMissingResponseError) as error:
            restore_collector_env(
                probe, RUN_ID, owner_nonce=NONCE, reboot_transition=callback
            )
        assert PRIVATE not in str(error.value)
        assert NONCE not in str(error.value)
    assert checks == ([] if authorization is None else [1])
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1
    assert not any(args[0] in {"create", "delete", "wait"} for args, _ in api.calls), (
        'test_public_probe_and_wrapper_only_defer_with_fresh_authorization: expected no any(args[0] in {"create", "delete", "wait"} for args, _ in...'
    )
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ('{"error":"synthetic-private-diagnostic"}', 0),
        ('{"error":"synthetic-private-diagnostic"}', 1),
        ('{"restored":true,"cleanup_verified":true}', 1),
        ("{}", 1),
        ("[]", 1),
        ("not-json", 1),
        (" \n", 1),
    ],
    ids=["guard-zero", "guard-one", "failed-receipt", "empty", "array", "bad", "blank"],
)
def test_explicit_rejection_precedes_post_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
    owned: tuple[ProbeApi, host.HostProbeFixture],
    stdout: str,
    returncode: int,
) -> None:
    api, probe = owned
    finished = False
    post_reads: list[str] = []
    checks: list[bool] = []

    def after() -> None:
        nonlocal finished
        finished = True

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if finished and command[7] == "get":
            post_reads.append(command[8])
            raise RegionalCommandTimeout(command, 60)
        return api.run(command, **kwargs)

    def binding() -> bool:
        checks.append(True)
        return True

    monkeypatch.setattr(host, "run_fixture_command", run)
    api.after_execute = after
    api.probe_stdout, api.probe_returncode = stdout, returncode
    with pytest.raises(host.HostProbeError) as error:
        restore_collector_env(
            probe, RUN_ID, owner_nonce=NONCE, reboot_transition=binding
        )
    assert type(error.value) is host.HostProbeError
    assert PRIVATE not in str(error.value)
    assert checks == []
    assert post_reads == []


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize(
    "drift",
    [
        "node-uid",
        "node-missing-uid",
        "pod-uid",
        "pod-missing-uid",
        "pod-missing",
        "pod-owner",
        "configmap-uid",
        "configmap-owner",
        "configmap-script",
        "forbidden",
    ],
)
def test_empty_response_does_not_hide_post_exec_identity_failure(
    owned: tuple[ProbeApi, host.HostProbeFixture], returncode: int, drift: str
) -> None:
    api, probe = owned

    def after() -> None:
        if drift == "forbidden":
            api.read_error = True
        elif drift.startswith("node-"):
            api.node_uid = "" if drift == "node-missing-uid" else "foreign-node"
        elif drift == "pod-missing":
            api.objects.pop("pod")
        elif drift == "pod-missing-uid":
            api.objects["pod"]["metadata"].pop("uid")
        elif drift == "configmap-script":
            api.objects["configmap"]["data"] = {"probe.py": "unowned content"}
        else:
            kind, attribute = drift.split("-")
            if attribute == "uid":
                api.objects[kind]["metadata"]["uid"] = "foreign-uid"
            else:
                api.objects[kind]["metadata"]["annotations"].pop(
                    "gpu-fault.io/probe-owner"
                )

    def forbidden_callback() -> bool:
        raise AssertionError("a rejected identity must not authorize deferral")

    api.probe_stdout, api.probe_returncode = "", returncode
    api.after_execute = after
    with pytest.raises(host.HostProbeError) as error:
        restore_collector_env(
            probe, RUN_ID, owner_nonce=NONCE, reboot_transition=forbidden_callback
        )
    assert type(error.value) is host.HostProbeError
    assert not any(args[0] in {"create", "delete"} for args, _ in api.calls), (
        'test_empty_response_does_not_hide_post_exec_identity_failure: expected no any(args[0] in {"create", "delete"} for args, _ in api.calls)'
    )


@pytest.mark.parametrize(
    "stderr",
    [
        "Forbidden",
        f'Error from server (Forbidden): pods "{PRIVATE}" is forbidden',
        "Error from server (Unauthorized)",
        "error: You must be logged in to the server",
    ],
)
def test_empty_forbidden_exec_is_not_a_missing_response(
    owned: tuple[ProbeApi, host.HostProbeFixture], stderr: str
) -> None:
    api, probe = owned
    api.probe_stdout, api.probe_returncode, api.probe_stderr = "", 1, stderr

    def callback() -> bool:
        raise AssertionError("permission rejection must not authorize deferral")

    with pytest.raises(host.HostProbeError, match="request was rejected") as error:
        restore_collector_env(
            probe, RUN_ID, owner_nonce=NONCE, reboot_transition=callback
        )
    assert type(error.value) is host.HostProbeError
    assert PRIVATE not in str(error.value)


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("stdout", [" ", "\n", "\t", "null", "{}", "[]", "bad-json"])
def test_nonempty_stdout_is_never_missing_response(
    owned: tuple[ProbeApi, host.HostProbeFixture], stdout: str, returncode: int
) -> None:
    api, probe = owned
    api.probe_stdout, api.probe_returncode = stdout, returncode
    with pytest.raises(host.HostProbeError) as error:
        probe.execute("restore-collector-env", "--run-id", RUN_ID)
    assert type(error.value) is host.HostProbeError


@pytest.mark.parametrize("returncode", [-9, 2, 124, 127, 137, 255])
def test_empty_stdout_other_exit_codes_remain_hard(
    owned: tuple[ProbeApi, host.HostProbeFixture], returncode: int
) -> None:
    api, probe = owned
    api.probe_stdout, api.probe_returncode = "", returncode
    with pytest.raises(host.HostProbeError) as error:
        probe.execute("restore-collector-env", "--run-id", RUN_ID)
    assert type(error.value) is host.HostProbeError


@pytest.mark.parametrize(
    "scenario",
    [
        "symlink",
        "dangling-symlink",
        "digest-mismatch",
        "digest-read-failure",
        "copy-failure",
        "new-copy",
        "owned-copy",
    ],
)
def test_actual_shell_setup_refuses_with_fixed_json_before_host_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owned: tuple[ProbeApi, host.HostProbeFixture],
    scenario: str,
) -> None:
    api, probe = owned
    source = probe.settings.probe_script
    target = tmp_path / f"{NONCE}-host-script.py"
    original = source.read_bytes()
    if scenario in {"symlink", "dangling-symlink"}:
        target.symlink_to(
            source if scenario == "symlink" else tmp_path / "missing-target"
        )
    elif scenario == "digest-mismatch":
        target.write_text(PRIVATE)
    elif scenario == "digest-read-failure":
        target.mkdir()
    elif scenario == "copy-failure":
        source.unlink()
    elif scenario == "owned-copy":
        target.write_bytes(original)

    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "host-action-started"
    chroot = commands / "chroot"
    chroot.write_text(
        '#!/bin/sh\nprintf started > "$TEST_MARKER"\n'
        """printf '%s\\n' '{"restored":true,"cleanup_verified":true}'\n"""
    )
    chroot.chmod(0o700)
    receipts: list[subprocess.CompletedProcess[str]] = []
    post_reads: list[str] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if receipts and command[7] == "get":
            post_reads.append(command[8])
            if scenario not in {"new-copy", "owned-copy"}:
                raise RegionalCommandTimeout(command, 60)
        result = api.run(command, **kwargs)
        if command[7] == "exec" and "host-probe" in command:
            shell = command[command.index("/bin/bash") :]
            start = shell.index("host-probe")
            shell[start + 1 : start + 3] = [str(source), str(target)]
            result = subprocess.run(
                shell,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
                env={"PATH": f"{commands}:/usr/bin:/bin", "TEST_MARKER": str(marker)},
            )
            receipts.append(result)
        return result

    def callback() -> bool:
        raise AssertionError("explicit shell rejection must not authorize deferral")

    monkeypatch.setattr(host, "run_fixture_command", run)
    if scenario in {"new-copy", "owned-copy"}:
        result = restore_collector_env(
            probe, RUN_ID, owner_nonce=NONCE, reboot_transition=callback
        )
        assert result == {"restored": True, "cleanup_verified": True}
        assert marker.read_text() == "started"
        assert target.read_bytes() == original
        assert target.stat().st_mode & 0o777 == 0o700
        assert post_reads == ["node", "pod", "configmap"]
    else:
        with pytest.raises(host.HostProbeError, match="output withheld") as error:
            restore_collector_env(
                probe, RUN_ID, owner_nonce=NONCE, reboot_transition=callback
            )
        assert type(error.value) is host.HostProbeError
        assert not marker.exists(), (
            "test_actual_shell_setup_refuses_with_fixed_json_before_host_action: expected no marker.exists()"
        )
        assert post_reads == []
        assert len(receipts) == 1
        assert receipts[0].returncode == 1
        assert json.loads(receipts[0].stdout) == {"error": "host probe setup rejected"}
        assert receipts[0].stderr == ""
        assert NONCE not in receipts[0].stdout
        assert PRIVATE not in receipts[0].stdout
        if scenario in {"symlink", "dangling-symlink"}:
            assert target.is_symlink(), (
                "test_actual_shell_setup_refuses_with_fixed_json_before_host_action: expected target.is_symlink()"
            )
        elif scenario == "digest-mismatch":
            assert target.read_text() == PRIVATE
        elif scenario == "digest-read-failure":
            assert target.is_dir(), (
                "test_actual_shell_setup_refuses_with_fixed_json_before_host_action: expected target.is_dir()"
            )
        else:
            assert not target.exists(), (
                "test_actual_shell_setup_refuses_with_fixed_json_before_host_action: expected no target.exists()"
            )
