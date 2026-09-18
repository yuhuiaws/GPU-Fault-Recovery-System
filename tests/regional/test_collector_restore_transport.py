"""Restore deferral requires an unknown response and a fresh reboot binding."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.collector_env_restore import restore_collector_env
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeFixture,
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import (
    RegionalCommandFailed,
    RegionalCommandTimeout,
)
from tests.regional._host_probe_support import ProbeApi, host_probe

RUN_ID = "collect004-transport-unit-a1"
OWNER_NONCE = "synthetic-owner-nonce-not-for-output"
PRIVATE_DETAIL = "synthetic-diagnostic-not-for-output"
RESTORE_ARGUMENTS = (
    "restore-collector-env",
    "--run-id",
    RUN_ID,
    "--owner-nonce",
    OWNER_NONCE,
)


@pytest.mark.parametrize("output", ["", "not-json"])
@pytest.mark.parametrize(
    "transition", ["pod-terminated", "node-unready", "node-unknown"]
)
def test_missing_and_malformed_responses_are_not_transport_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: str, transition: str
) -> None:
    api = ProbeApi()
    finished = False

    def after() -> None:
        nonlocal finished
        finished = True
        if transition == "pod-terminated":
            api.objects["pod"]["status"]["phase"] = "Failed"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = api.run(command, **kwargs)
        if finished and command[7:9] == ["get", "node"]:
            node = json.loads(result.stdout)
            node["status"] = {
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "Unknown"
                        if transition == "node-unknown"
                        else "False",
                    }
                ]
            }
            result.stdout = json.dumps(node)
        return result

    monkeypatch.setattr(host, "run_fixture_command", run)
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_stdout, api.probe_returncode = output, 1
    api.after_execute = after
    expected = HostProbeMissingResponseError if output == "" else HostProbeError
    with pytest.raises(expected) as error:
        probe.execute("restore-collector-env", "--run-id", RUN_ID)
    assert not isinstance(error.value, HostProbeTransportError), (
        "test_missing_and_malformed_responses_are_not_transport_proof: expected no isinstance(error.value, HostProbeTransportError)"
    )
    if output:
        assert not isinstance(error.value, HostProbeMissingResponseError), (
            "test_missing_and_malformed_responses_are_not_transport_proof: expected no isinstance(error.value, HostProbeMissingResponseError)"
        )
    assert api.exec_containers == ["probe"]


def test_explicit_host_guard_rejection_stays_hard_even_if_pod_terminates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_stdout, api.probe_returncode = '{"error":"guard rejected"}', 1
    api.after_execute = lambda: api.objects["pod"]["status"].update({"phase": "Failed"})
    with pytest.raises(HostProbeError) as caught:
        probe.execute("restore-collector-env", "--run-id", RUN_ID)
    assert not isinstance(caught.value, HostProbeTransportError), (
        "test_explicit_host_guard_rejection_stays_hard_even_if_pod_terminates: expected no isinstance(caught.value, HostProbeTransportError)"
    )


@dataclass
class _Collector:
    responses: list[dict[str, object] | Exception]
    events: list[str] = field(default_factory=list)
    calls: list[tuple[tuple[str, ...], int]] = field(default_factory=list)

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, object]:
        self.events.append("execute")
        self.calls.append((arguments, timeout))
        assert self.responses, "restore exceeded its permitted execution count"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def recreate(self) -> None:
        self.events.append("recreate")
        raise AssertionError("restore wrapper must not recreate the probe")


def _binding(collector: _Collector, *responses: bool | Exception) -> Callable[[], bool]:
    pending = list(responses)

    def check() -> bool:
        collector.events.append("binding")
        assert pending, "unexpected additional reboot-binding check"
        response = pending.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return check


def _binding_failure(kind: str) -> bool | Exception:
    if kind == "false":
        return False
    exception_type = {
        "callback": RuntimeError,
        "guard": HostProbeError,
        "transport": HostProbeTransportError,
        "missing": HostProbeMissingResponseError,
    }[kind]
    return exception_type("reboot binding could not be verified")


@pytest.mark.parametrize(
    "receipt",
    [
        {"restored": True, "cleanup_verified": True},
        {"restored": False, "reason": "recovery remains unresolved"},
        {"restored": False, "deferred": True, "cleanup_verified": False},
    ],
    ids=["restored", "unresolved", "deferred"],
)
def test_restore_preserves_receipt_and_forwards_nonce(
    receipt: dict[str, object],
) -> None:
    collector = _Collector([receipt])

    result = restore_collector_env(
        collector,
        RUN_ID,
        owner_nonce=OWNER_NONCE,
        reboot_transition=_binding(collector),
    )

    assert result is receipt, "a failed restore must not become cleanup success"
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]
    assert collector.events == ["execute"]
    assert not collector.responses, (
        "test_restore_preserves_receipt_and_forwards_nonce: expected no collector.responses"
    )


@pytest.mark.parametrize(
    "message",
    [
        "collector env backup digest mismatch",
        "collector env ownership changed",
        "host probe failed (exit 1); output withheld",
        "kubectl failed: exec: pod is Failed; wait Ready timed out",
    ],
    ids=["checksum", "ownership", "command", "transport-like-text"],
)
def test_generic_probe_error_never_recreates_or_checks_reboot(message: str) -> None:
    failure = HostProbeError(message)
    collector = _Collector([failure])

    with pytest.raises(HostProbeError) as error:
        restore_collector_env(
            collector,
            RUN_ID,
            owner_nonce=OWNER_NONCE,
            reboot_transition=_binding(collector, True),
        )

    assert error.value is failure
    assert collector.events == ["execute"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]


@pytest.mark.parametrize(
    "failure_type", [HostProbeTransportError, HostProbeMissingResponseError]
)
def test_unknown_response_without_reboot_binding_is_a_hard_error(
    failure_type: type[HostProbeError],
) -> None:
    failure = failure_type("owned probe response unavailable")
    collector = _Collector([failure])

    with pytest.raises(failure_type) as error:
        restore_collector_env(collector, RUN_ID, owner_nonce=OWNER_NONCE)

    assert error.value is failure
    assert collector.events == ["execute"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]


@pytest.mark.parametrize(
    "failure_type", [HostProbeTransportError, HostProbeMissingResponseError]
)
@pytest.mark.parametrize(
    "binding_result", ["false", "callback", "guard", "transport", "missing"]
)
def test_reboot_binding_failure_prevents_deferral(
    binding_result: str, failure_type: type[HostProbeError]
) -> None:
    failure = failure_type("owned probe response unavailable")
    decision = _binding_failure(binding_result)
    collector = _Collector([failure])
    expected = decision if isinstance(decision, Exception) else failure

    with pytest.raises(type(expected)) as error:
        restore_collector_env(
            collector,
            RUN_ID,
            owner_nonce=OWNER_NONCE,
            reboot_transition=_binding(collector, decision),
        )

    assert error.value is expected
    assert collector.events == ["execute", "binding"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]


@pytest.mark.parametrize(
    "failure_type", [HostProbeTransportError, HostProbeMissingResponseError]
)
def test_unknown_response_defers_without_recreating_or_retrying(
    failure_type: type[HostProbeError], capsys: pytest.CaptureFixture[str]
) -> None:
    failure = failure_type(f"{OWNER_NONCE} {PRIVATE_DETAIL}")
    collector = _Collector([failure])

    result = restore_collector_env(
        collector,
        RUN_ID,
        owner_nonce=OWNER_NONCE,
        reboot_transition=_binding(collector, True),
    )

    assert result["run_id"] == RUN_ID
    assert result["restored"] is False
    assert result["deferred"] is True
    assert result["cleanup_verified"] is False
    assert result.get("timer_disarmed") is not True
    assert result["error_type"] == failure_type.__name__
    assert collector.events == ["execute", "binding"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert OWNER_NONCE not in json.dumps(result)
    assert PRIVATE_DETAIL not in json.dumps(result)


@pytest.mark.parametrize(
    "failure_type", [HostProbeTransportError, HostProbeMissingResponseError]
)
@pytest.mark.parametrize(
    "binding_result", ["false", "callback", "guard", "transport", "missing"]
)
def test_later_restore_does_not_reuse_a_previous_reboot_authorization(
    binding_result: str, failure_type: type[HostProbeError]
) -> None:
    first = failure_type("first missing response")
    second = failure_type("second missing response")
    decision = _binding_failure(binding_result)
    collector = _Collector([first, second])
    binding = _binding(collector, True, decision)
    result = restore_collector_env(
        collector, RUN_ID, owner_nonce=OWNER_NONCE, reboot_transition=binding
    )
    assert result["deferred"] is True
    assert result["cleanup_verified"] is False
    expected = decision if isinstance(decision, Exception) else second

    with pytest.raises(type(expected)) as error:
        restore_collector_env(
            collector, RUN_ID, owner_nonce=OWNER_NONCE, reboot_transition=binding
        )

    assert error.value is expected
    assert collector.events == ["execute", "binding"] * 2
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)] * 2


@pytest.mark.parametrize(
    "failure_type", [HostProbeError, RuntimeError], ids=["guard", "unexpected"]
)
def test_guard_or_unexpected_failure_is_never_deferred(
    failure_type: type[Exception],
) -> None:
    failure = failure_type("restore ownership or checksum rejected")
    collector = _Collector([failure])

    with pytest.raises(failure_type) as error:
        restore_collector_env(
            collector,
            RUN_ID,
            owner_nonce=OWNER_NONCE,
            reboot_transition=_binding(collector, True),
        )

    assert error.value is failure
    assert collector.events == ["execute"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)]


@pytest.mark.parametrize("decision", [1, "true", {}, None])
def test_only_boolean_true_authorizes_deferral(decision: object) -> None:
    failure = HostProbeMissingResponseError("missing response")
    collector = _Collector([failure])
    with pytest.raises(HostProbeMissingResponseError) as error:
        restore_collector_env(
            collector,
            RUN_ID,
            owner_nonce=OWNER_NONCE,
            reboot_transition=cast(Callable[[], bool], lambda: decision),
        )
    assert error.value is failure
    assert collector.events == ["execute"]


def test_post_reboot_restore_can_disable_deferral() -> None:
    receipt: dict[str, object] = {"restored": True, "cleanup_verified": True}
    collector = _Collector([HostProbeMissingResponseError("missing response"), receipt])
    first = restore_collector_env(
        collector,
        RUN_ID,
        owner_nonce=OWNER_NONCE,
        reboot_transition=_binding(collector, True),
    )
    assert first["deferred"] is True
    result = restore_collector_env(collector, RUN_ID, owner_nonce=OWNER_NONCE)
    assert result is receipt
    assert collector.events == ["execute", "binding", "execute"]
    assert collector.calls == [(RESTORE_ARGUMENTS, 300)] * 2


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ProbeApi:
    value = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", value.run)
    return value


@pytest.fixture
def owned_probe(tmp_path: Path, api: ProbeApi) -> Iterator[HostProbeFixture]:
    probe = host_probe(tmp_path)
    probe.create()
    original_objects = deepcopy(api.objects)
    original_node_uid = api.node_uid
    api.calls.clear()
    try:
        yield probe
    finally:
        # Undo only fake API drift so public cleanup can release the local lock.
        api.objects = original_objects
        api.node_uid = original_node_uid
        api.after_execute = None
        assert not any(probe.cleanup().values()), "in-memory probe cleanup failed"


@pytest.mark.parametrize("phase", ["Failed", "Succeeded"])
def test_owned_terminal_pod_raises_transport_before_exec(
    owned_probe: HostProbeFixture, api: ProbeApi, phase: str
) -> None:
    api.objects["pod"]["status"]["phase"] = phase

    with pytest.raises(HostProbeTransportError) as error:
        owned_probe.execute(*RESTORE_ARGUMENTS)

    assert isinstance(error.value, HostProbeError), (
        "test_owned_terminal_pod_raises_transport_before_exec: expected isinstance(error.value, HostProbeError)"
    )
    assert not any(args[0] in {"exec", "create", "delete"} for args, _ in api.calls), (
        'test_owned_terminal_pod_raises_transport_before_exec: expected no any(args[0] in {"exec", "create", "delete"} for args, _ in api.calls)'
    )
    assert not api.host_script_exists, (
        "test_owned_terminal_pod_raises_transport_before_exec: expected no api.host_script_exists"
    )
    assert owned_probe.residuals()["host_script"] is False


@pytest.mark.parametrize("phase", ["Running", "Failed", "Succeeded"])
@pytest.mark.parametrize(
    "drift", ["node-uid", "pod-uid", "pod-owner", "configmap-uid", "configmap-owner"]
)
def test_identity_drift_is_a_guard_even_when_pod_is_terminal(
    owned_probe: HostProbeFixture, api: ProbeApi, phase: str, drift: str
) -> None:
    api.objects["pod"]["status"]["phase"] = phase
    if drift == "node-uid":
        api.node_uid = "foreign-node-uid"
    else:
        kind, attribute = drift.split("-")
        metadata = api.objects[kind]["metadata"]
        if attribute == "uid":
            metadata["uid"] = "foreign-resource-uid"
        else:
            metadata["annotations"]["gpu-fault.io/probe-owner"] = "foreign-owner"

    with pytest.raises(HostProbeError) as error:
        owned_probe.execute(*RESTORE_ARGUMENTS)

    assert not isinstance(error.value, HostProbeTransportError), (
        "test_identity_drift_is_a_guard_even_when_pod_is_terminal: expected no isinstance(error.value, HostProbeTransportError)"
    )
    assert not any(args[0] in {"exec", "create", "delete"} for args, _ in api.calls), (
        'test_identity_drift_is_a_guard_even_when_pod_is_terminal: expected no any(args[0] in {"exec", "create", "delete"} for args, _ in api.calls)'
    )
    assert not api.host_script_exists, (
        "test_identity_drift_is_a_guard_even_when_pod_is_terminal: expected no api.host_script_exists"
    )


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        (json.dumps({"error": PRIVATE_DETAIL}), 0),
        (json.dumps({"error": PRIVATE_DETAIL}), 1),
        (json.dumps({"restored": True, "detail": PRIVATE_DETAIL}), 1),
        (PRIVATE_DETAIL, 1),
        ("{}", 0),
        ("[]", 0),
    ],
    ids=["json-error", "nonzero-json-error", "nonzero", "invalid", "empty", "array"],
)
def test_host_command_failure_is_generic_and_withholds_arguments_and_output(
    owned_probe: HostProbeFixture,
    api: ProbeApi,
    stdout: str,
    returncode: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api.probe_stdout = stdout
    api.probe_stderr = f"{PRIVATE_DETAIL} {OWNER_NONCE}"
    api.probe_returncode = returncode

    with pytest.raises(HostProbeError) as error:
        owned_probe.execute(*RESTORE_ARGUMENTS)

    assert not isinstance(error.value, HostProbeTransportError), (
        "test_host_command_failure_is_generic_and_withholds_arguments_and_output: expected no isinstance(error.value, HostProbeTransportError)"
    )
    assert PRIVATE_DETAIL not in str(error.value)
    assert OWNER_NONCE not in str(error.value)
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1
    assert owned_probe.residuals()["host_script"] is True
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("drift", ["node-uid", "pod-uid", "pod-owner"])
def test_post_exec_drift_does_not_return_a_restore_receipt(
    owned_probe: HostProbeFixture, api: ProbeApi, drift: str
) -> None:
    api.probe_stdout = '{"restored": true, "cleanup_verified": true}'

    def change_identity() -> None:
        if drift == "node-uid":
            api.node_uid = "foreign-node-uid"
        elif drift == "pod-uid":
            api.objects["pod"]["metadata"]["uid"] = "foreign-pod-uid"
        else:
            api.objects["pod"]["metadata"]["annotations"][
                "gpu-fault.io/probe-owner"
            ] = "foreign-owner"

    api.after_execute = change_identity
    with pytest.raises(HostProbeError) as error:
        owned_probe.execute(*RESTORE_ARGUMENTS)

    assert not isinstance(error.value, HostProbeTransportError), (
        "test_post_exec_drift_does_not_return_a_restore_receipt: expected no isinstance(error.value, HostProbeTransportError)"
    )
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1
    assert not any(args[0] in {"create", "delete"} for args, _ in api.calls), (
        'test_post_exec_drift_does_not_return_a_restore_receipt: expected no any(args[0] in {"create", "delete"} for args, _ in api.calls)'
    )


def test_public_execute_timeout_is_typed_and_never_replays_the_host_action(
    owned_probe: HostProbeFixture, api: ProbeApi, capsys: pytest.CaptureFixture[str]
) -> None:
    def time_out() -> None:
        raise RegionalCommandTimeout(
            ["kubectl", "exec", *RESTORE_ARGUMENTS, PRIVATE_DETAIL], 7
        )

    api.after_execute = time_out

    with pytest.raises(HostProbeTransportError, match="timed out after 7s") as error:
        owned_probe.execute(*RESTORE_ARGUMENTS, timeout=7)

    assert isinstance(error.value, HostProbeError), (
        "test_public_execute_timeout_is_typed_and_never_replays_the_host_action: expected isinstance(error.value, HostProbeError)"
    )
    assert error.value.__cause__ is None
    assert OWNER_NONCE not in str(error.value)
    assert PRIVATE_DETAIL not in str(error.value)
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1
    assert not any(args[0] in {"create", "delete"} for args, _ in api.calls), (
        'test_public_execute_timeout_is_typed_and_never_replays_the_host_action: expected no any(args[0] in {"create", "delete"} for args, _ in ap...'
    )
    assert api.host_script_exists, (
        "test_public_execute_timeout_is_typed_and_never_replays_the_host_action: expected api.host_script_exists"
    )
    assert owned_probe.residuals()["host_script"] is True
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize(
    "message", ["Forbidden", "exec: pod is Failed; Ready timed out"]
)
def test_untyped_command_failure_is_not_inferred_to_be_transport(
    owned_probe: HostProbeFixture, api: ProbeApi, message: str
) -> None:
    def fail_command() -> None:
        raise RegionalCommandFailed(1, message)

    api.after_execute = fail_command

    with pytest.raises(HostProbeError) as error:
        owned_probe.execute(*RESTORE_ARGUMENTS)

    assert not isinstance(error.value, HostProbeTransportError), (
        "test_untyped_command_failure_is_not_inferred_to_be_transport: expected no isinstance(error.value, HostProbeTransportError)"
    )
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1
    assert not any(args[0] in {"create", "delete"} for args, _ in api.calls), (
        'test_untyped_command_failure_is_not_inferred_to_be_transport: expected no any(args[0] in {"create", "delete"} for args, _ in api.calls)'
    )
