"""An install that times out is an unknown outcome, never a fresh install.

``_retryable_action_error`` classes ``subprocess.TimeoutExpired`` as retryable,
which is right for a probe and wrong for a package manager or firmware tool:
killed at 1800 s it may still be mid-flight or half-applied, and a control
plane that reads "retryable" resubmits the command and the agent runs a second
1800 s install on top. For REMEDIATE_DRIVER, UPDATE_SOFTWARE_FIRMWARE and
REMEDIATE_EFA_DRIVER every failure from the moment the install subprocess is
spawned is therefore FAILED, ``retryable=False``, with ``outcome_unknown`` and
``manual_confirmation_required`` in the details -- the flags the escalation
ladder reads as "hand this to an operator". Failures before the spawn (the
executable cannot be started) stay retryable: nothing ran.

Public surface only: ``execute`` and the ledger's ``attempt_history``.
"""

from __future__ import annotations

from typing import Callable

from ._support import (
    CompletedProcess,
    FakeRunner,
    NodeActionStatus,
    Quiesced,
    TimeoutExpired,
    WorkflowOperation,
    command,
    envelope,
    executor,
    hashlib,
    no_device_clients,
    node_action_executor,
    pytest,
)

INSTALL_TIMEOUT_SECONDS = 1800
UNKNOWN_OUTCOME_FLAGS = ("outcome_unknown", "manual_confirmation_required")
# The nvidia-smi query ``_verify_no_clients`` runs before the install spawns.
CLIENT_PROBE = "--query-compute-apps=gpu_uuid,pid,process_name"

Failure = Callable[[list[str], int | None], BaseException]


def timeout(argv: list[str], seconds: int | None) -> BaseException:
    return TimeoutExpired(argv, seconds or 0)


def cannot_spawn(argv: list[str], _seconds: int | None) -> BaseException:
    return FileNotFoundError(2, "No such file or directory", argv[0])


class InstallRunner:
    """Plays the package manager and the verification probe.

    ``install`` names the executable whose spawn is the point of no return;
    ``failure`` is raised for it (a timeout after the spawn, or an OSError in
    place of the spawn), ``verify_failure`` for any command that matches a key
    of ``answers`` -- the post-install verification probes -- and
    ``probe_failure`` for the compute-client probe that runs *before* the
    install.
    """

    def __init__(
        self,
        install: str,
        *,
        answers: dict[str, str],
        failure: Failure | None = None,
        verify_failure: Failure | None = None,
        probe_failure: Failure | None = None,
    ) -> None:
        self.install = install
        self.answers = answers
        self.failure = failure
        self.verify_failure = verify_failure
        self.probe_failure = probe_failure
        self.commands: list[list[str]] = []

    def __call__(self, argv, *, timeout=None, **_):
        argv = list(argv)
        self.commands.append(argv)
        if CLIENT_PROBE in argv and self.probe_failure is not None:
            raise self.probe_failure(argv, timeout)
        if argv[0] == self.install:
            if self.failure is not None:
                raise self.failure(argv, timeout)
            return CompletedProcess(argv, 0, stdout="", stderr="")
        for token, stdout in self.answers.items():
            if token in argv:
                if self.verify_failure is not None:
                    raise self.verify_failure(argv, timeout)
                return CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")

    def installs(self) -> int:
        return sum(1 for argv in self.commands if argv[0] == self.install)


def _pinned(tmp_path, name: str):
    executable = tmp_path / name
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    return executable, hashlib.sha256(executable.read_bytes()).hexdigest()


def driver_install(tmp_path, **runner_options):
    executable, digest = _pinned(tmp_path, "driver-remediate")
    runner = InstallRunner(
        str(executable),
        answers={"--query-gpu=driver_version": "575.86.01\n"},
        **runner_options,
    )
    agent = node_action_executor(
        tmp_path,
        "driver-timeout.db",
        allowed_operations={WorkflowOperation.REMEDIATE_DRIVER},
        reset_enabled=True,
        service_quiesce_enabled=True,
        driver_remediation_enabled=True,
        driver_remediation_command=(str(executable), "--branch", "{target}"),
        driver_remediation_sha256=digest,
        target_driver_branch=575,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        agent_generation=3,
        sleep=lambda _: None,
    )
    signed = envelope(
        command(
            WorkflowOperation.REMEDIATE_DRIVER,
            command_id="workflow/3/REMEDIATE_DRIVER/node-a",
            parameters={"target_driver_branch": 575},
        )
    )
    return agent, runner, signed


def firmware_install(tmp_path, **runner_options):
    executable, digest = _pinned(tmp_path, "firmware-update")
    verifier, verifier_digest = _pinned(tmp_path, "firmware-version")
    runner = InstallRunner(
        str(executable), answers={str(verifier): "92.10.14\n"}, **runner_options
    )
    agent = node_action_executor(
        tmp_path,
        "firmware-timeout.db",
        allowed_operations={WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE},
        reset_enabled=True,
        service_quiesce_enabled=True,
        firmware_update_enabled=True,
        firmware_update_command=(str(executable), "--version", "{target}"),
        firmware_update_sha256=digest,
        target_firmware_version="92.10.14",
        firmware_verify_command=(str(verifier),),
        firmware_verify_sha256=verifier_digest,
        quiesce_manager=Quiesced(),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        agent_generation=3,
        sleep=lambda _: None,
    )
    signed = envelope(
        command(
            WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
            command_id="workflow/4/UPDATE_SOFTWARE_FIRMWARE/node-a",
            parameters={"target_firmware_version": "92.10.14"},
        )
    )
    return agent, runner, signed


def efa_install(tmp_path, **runner_options):
    """Two EFA functions, neither bound: ``modprobe efa`` is the install."""

    pci_root = tmp_path / "pci"
    for index in range(2):
        device = pci_root / f"0000:0{index}:00.0"
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa2\n")
    runner = InstallRunner("modprobe", answers={}, **runner_options)
    agent = node_action_executor(
        tmp_path,
        "efa-timeout.db",
        allowed_operations={WorkflowOperation.REMEDIATE_EFA_DRIVER},
        efa_driver_remediation_enabled=True,
        efa_pci_devices_root=str(pci_root),
        efa_driver_bind_path=str(tmp_path / "drivers" / "efa" / "bind"),
        runner=runner,
        agent_generation=3,
    )
    signed = envelope(
        command(
            WorkflowOperation.REMEDIATE_EFA_DRIVER,
            command_id="workflow/5/REMEDIATE_EFA_DRIVER/node-a",
            parameters={"expected_count": 2},
        )
    )
    return agent, runner, signed


INSTALLS = [
    pytest.param(driver_install, id="driver"),
    pytest.param(firmware_install, id="firmware"),
    pytest.param(efa_install, id="efa"),
]


def assert_unknown_outcome(result) -> None:
    assert result.status is NodeActionStatus.FAILED, result
    assert result.retryable is False, (
        "an install whose outcome nobody can read must never be retried: "
        f"{result.error}"
    )
    assert "unknown" in (result.error or ""), (
        f"the error must say the node state is unknown: {result.error}"
    )
    for flag in UNKNOWN_OUTCOME_FLAGS:
        assert result.details.get(flag) is True, (
            f"the escalation ladder reads {flag!r} from the details: {result.details}"
        )


@pytest.mark.parametrize("install", INSTALLS)
def test_an_install_timeout_is_an_unknown_outcome_and_the_resubmit_replays_it(
    tmp_path, install
) -> None:
    """The install was spawned and killed at its deadline; nothing runs twice."""

    agent, runner, signed = install(tmp_path, failure=timeout)

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert_unknown_outcome(first)
    assert "TimeoutExpired" in (first.error or ""), (
        f"the error must name the timeout that was hit: {first.error}"
    )
    assert second == first, "the resubmit must replay the stored result"
    assert runner.installs() == 1, (
        f"exactly one install may be spawned for one command_id: {runner.commands}"
    )
    history = agent.ledger.attempt_history(signed.command.command_id)
    assert [(row["attempt"], row["state"]) for row in history] == [(1, "FAILED")], (
        f"a second attempt row means the timeout was treated as retryable: {history}"
    )


def test_the_driver_timeout_error_names_the_deadline(tmp_path) -> None:
    agent, _runner, signed = driver_install(tmp_path, failure=timeout)

    result = agent.execute(signed)

    assert str(INSTALL_TIMEOUT_SECONDS) in (result.error or ""), (
        f"the error must name the {INSTALL_TIMEOUT_SECONDS} s deadline: {result.error}"
    )
    assert result.details["install_timeout_seconds"] == INSTALL_TIMEOUT_SECONDS, (
        result.details
    )


@pytest.mark.parametrize("install", INSTALLS)
def test_an_install_that_cannot_be_spawned_stays_retryable(tmp_path, install) -> None:
    """Nothing ran: the resubmit is the retry, and it is allowed to run."""

    agent, runner, signed = install(tmp_path, failure=cannot_spawn)

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert first.status is NodeActionStatus.FAILED, first
    assert first.retryable is True, (
        f"a spawn failure ran nothing and must stay retryable: {first.error}"
    )
    assert not any(first.details.get(flag) for flag in UNKNOWN_OUTCOME_FLAGS), (
        f"a spawn failure is a known outcome: {first.details}"
    )
    assert second.attempt == 2, f"the resubmit must open attempt 2: {second}"
    assert runner.installs() == 2, (
        f"each attempt spawns the install once: {runner.commands}"
    )


@pytest.mark.parametrize(
    "install",
    [
        pytest.param(driver_install, id="driver"),
        pytest.param(firmware_install, id="firmware"),
    ],
)
def test_a_verification_probe_that_fails_to_run_after_the_install_is_not_retryable(
    tmp_path, install
) -> None:
    """The install finished; the probe that would confirm it timed out.

    A retry would run the whole install again, so anything that would have
    been retryable *before* the spawn is an unknown outcome *after* it.
    """

    agent, runner, signed = install(tmp_path, verify_failure=timeout)

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert_unknown_outcome(first)
    assert second == first, "the resubmit must replay the stored result"
    assert runner.installs() == 1, (
        f"the install must not run a second time for a failed probe: {runner.commands}"
    )


@pytest.mark.parametrize(
    "install",
    [
        pytest.param(driver_install, id="driver"),
        pytest.param(firmware_install, id="firmware"),
    ],
)
def test_a_timed_out_client_probe_before_the_install_stays_retryable(
    tmp_path, install
) -> None:
    """The retryable side of the boundary: nothing was spawned yet.

    ``_verify_no_clients`` runs ``nvidia-smi --query-compute-apps`` before the
    installer; a slow nvidia-smi there is exactly the ``TimeoutExpired`` the
    executor is right to call retryable, and the resubmit must be allowed to
    run the install.
    """

    agent, runner, signed = install(tmp_path, probe_failure=timeout)

    first = agent.execute(signed)

    assert first.status is NodeActionStatus.FAILED, first
    assert first.retryable is True, (
        f"a probe timeout before the spawn must stay retryable: {first.error}"
    )
    assert (first.error or "").startswith("TimeoutExpired:"), first.error
    assert not any(first.details.get(flag) for flag in UNKNOWN_OUTCOME_FLAGS), (
        f"nothing was installed, so the outcome is not unknown: {first.details}"
    )
    assert runner.installs() == 0, (
        f"the install must not have been spawned: {runner.commands}"
    )
    runner.probe_failure = None
    second = agent.execute(signed)
    assert second.status is NodeActionStatus.SUCCEEDED, second.error
    assert second.attempt == 2, second
    assert runner.installs() == 1, runner.commands


def test_a_gpu_reset_timeout_keeps_its_own_classification(tmp_path) -> None:
    """RESET_GPU is not an install: its timeout stays the reset's own ruling.

    ``_run_reset_with_busy_retry`` already turns the killed ``nvidia-smi`` into
    a non-retryable ``ResetOutcomeUnknown`` that the per-GPU loop reports as a
    ``ResetProgressError`` (pinned in test_remediation.py); this pins that the
    install ruling did not reach into it.
    """

    runner = FakeRunner(reset_timeout_seconds=120)
    agent = executor(tmp_path, runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    first = agent.execute(signed)
    second = agent.execute(signed)

    assert first.status is NodeActionStatus.FAILED, first
    assert first.retryable is False, first
    assert (first.error or "").startswith("ResetProgressError:"), (
        f"the reset keeps its own error class: {first.error}"
    )
    assert "gpu reset outcome unknown after 120s" in (first.error or ""), first.error
    assert first.details["reset_outcome_unknown"] == ["GPU-a"], (
        f"the reset keeps its own per-GPU progress details: {first.details}"
    )
    assert second == first, "the resubmit must replay the stored result"
    assert len([item for item in runner.commands if "--gpu-reset" in item]) == 1, (
        "exactly one nvidia-smi --gpu-reset may run for one command"
    )
