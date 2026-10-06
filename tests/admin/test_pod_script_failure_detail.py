"""A failed in-Pod administrator script says what refused it.

Live (gpu-eks-cluster): ``submit-remediation --disposition confirm-node-action``
failed with "workflow reconcile execution failed: <sensitive output redacted>"
and nothing else -- the Pod script's own refusal (it wanted a terminal Node
Agent result the reboot-settled RESET_GPU could not have) was a ``SystemExit``
on stderr, which the admin side blacks out wholesale. The script now also
prints its refusal as JSON on stdout, which the admin side shows after the
ordinary redaction; a traceback still stays blacked out, but names its
exception class.
"""

from __future__ import annotations

from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from gpu_fault.admin import workflow_reconcile as module
from gpu_fault.admin.bootstrap_common import BootstrapError

SITE = SimpleNamespace(
    release_config={"cpu_kubeconfig": "/tmp/cpu.kubeconfig", "namespace": "gpu-fault"}
)


def _runner(monkeypatch: pytest.MonkeyPatch, *, stdout: str, stderr: str):
    def run_command(arguments, **kwargs):
        if "get" in arguments:
            return CompletedProcess(arguments, 0, "cpu-ingress-0", "")
        return CompletedProcess(arguments, 1, stdout, stderr)

    monkeypatch.setattr(module, "run_command", run_command)


def test_a_script_refusal_is_shown_to_the_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner(
        monkeypatch,
        stdout=(
            '{"error": "submit-remediation: confirmation for RESET_GPU on node-a cites '
            "neither a terminal Node Agent result, nor a validating restore, nor a reboot "
            'since the action", "error_kind": "refused"}\n'
        ),
        stderr=(
            "Traceback (most recent call last):\n  File ...\n"
            "SystemExit: submit-remediation: confirmation for RESET_GPU ... token=AKIA123\n"
        ),
    )

    with pytest.raises(BootstrapError) as failed:
        module.run_control_plane_script(SITE, {"mode": "confirm-node-action"})

    message = str(failed.value)
    assert "confirmation for RESET_GPU on node-a cites neither" in message
    assert "nor a reboot since the action" in message
    assert "AKIA123" not in message, (
        "stderr stays redacted; the refusal came from stdout"
    )


def test_a_traceback_names_only_its_exception_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner(
        monkeypatch,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "<string>", line 12, in <module>\n'
            "KeyError: 'workflow-ec95fa59 postgres://user:secret@db/gpu'\n"
        ),
    )

    with pytest.raises(BootstrapError) as failed:
        module.run_control_plane_script(SITE, {"mode": "confirm-node-action"})

    message = str(failed.value)
    assert message.endswith("KeyError: <sensitive output redacted>"), message
    assert "secret" not in message and "workflow-ec95fa59" not in message


def test_an_unreadable_failure_keeps_the_old_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner(monkeypatch, stdout="not json\n", stderr="")

    with pytest.raises(BootstrapError, match="workflow reconcile execution failed$"):
        module.run_control_plane_script(SITE, {"mode": "confirm-node-action"})
