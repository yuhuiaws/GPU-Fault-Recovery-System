"""Who ran the administrator command, resolved without ever blocking it.

Operator writes used to be attributed by a free-text ``--reference`` alone. The
identity comes from ``aws sts get-caller-identity``; a CLI that is missing, slow
or unauthenticated must degrade to a named fallback, never hold the command.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.admin import operator_identity

ARN = "arn:aws:sts::123456789012:assumed-role/Admin/alice"


def _completed(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_the_sts_arn_is_returned_when_the_cli_answers() -> None:
    seen: list[dict[str, object]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen.append({"command": command, **kwargs})
        return _completed(json.dumps({"Arn": ARN, "Account": "123456789012"}))

    arn = operator_identity.caller_identity_arn(run=run, timeout=3.0)

    assert arn == ARN, "the STS ARN was not read from the CLI output"
    assert seen[0]["command"][:3] == ["aws", "sts", "get-caller-identity"], (
        f"unexpected identity command: {seen[0]['command']}"
    )
    assert seen[0]["timeout"] == 3.0, "the resolver must bound how long it waits"


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.TimeoutExpired(cmd="aws", timeout=1),
        FileNotFoundError("aws"),
        _completed("", returncode=255),
        _completed("not json"),
        _completed(json.dumps({"Account": "1"})),
    ],
    ids=["timeout", "missing-cli", "nonzero", "invalid-json", "no-arn"],
)
def test_every_failure_yields_no_identity_instead_of_an_error(outcome: object) -> None:
    def run(_command: list[str], **_kwargs: object) -> object:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    assert operator_identity.caller_identity_arn(run=run) is None, (
        "a failed identity lookup must not raise; the write it attributes "
        "still has to happen"
    )


def test_resolution_falls_back_to_the_named_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: None
    )

    assert (
        operator_identity.resolve_operator_identity()
        == operator_identity.UNKNOWN_IDENTITY
    ), "without an ARN the identity must be the named unknown, not empty"
    assert (
        operator_identity.resolve_operator_identity(fallback="alice@host")
        == "alice@host"
    ), "an explicit fallback must be used when the ARN is unavailable"


def test_resolution_prefers_the_arn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(operator_identity, "caller_identity_arn", lambda **_kwargs: ARN)

    assert operator_identity.resolve_operator_identity(fallback="alice@host") == ARN, (
        "the STS ARN must win over the local fallback"
    )


def test_the_local_identity_names_user_and_host() -> None:
    local = operator_identity.local_operator_identity()

    assert "@" in local, f"local identity is not user@host: {local!r}"
    assert not local.startswith("@") and not local.endswith("@"), (
        f"local identity has an empty user or host: {local!r}"
    )
