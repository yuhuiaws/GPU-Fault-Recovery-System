"""The AMP definition read behind release-diff retries transient AWS failures."""

from __future__ import annotations

import pytest

from gpu_fault_release import regional_observability_rollback as observability
from gpu_fault_release.regional_observability_rollback import AmpDefinition


class _Runner:
    def __init__(self, answers: list[tuple[int, str, str]]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def probe_output(self, arguments: list[str]) -> tuple[int, str, str]:
        self.calls += 1
        return self.answers.pop(0)


class _Release:
    def __init__(self, answers: list[tuple[int, str, str]]) -> None:
        self.runner = _Runner(answers)


DEFINITION = AmpDefinition(
    describe=("aws", "amp", "describe-alert-manager-definition"),
    root="alertManagerDefinition",
    label="AMP Alertmanager",
)
NO_CREDENTIALS = (
    "aws: [ERROR]: An error occurred (NoCredentials): Unable to locate credentials."
    ' You can configure credentials by running "aws login".'
)


def test_transient_credential_failure_is_retried_then_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live 2026-09-23: one describe in release-diff answered NoCredentials (the
    instance-role fetch failed once under ~370 parallel commands) and a verified
    release was rolled back. A read gets a bounded second chance."""

    slept: list[float] = []
    monkeypatch.setattr(observability, "_RETRY_SLEEP", slept.append)
    release = _Release(
        [
            (255, "", NO_CREDENTIALS),
            (0, '{"alertManagerDefinition": {"status": {"statusCode": "ACTIVE"}}}', ""),
        ]
    )

    document = observability.describe_amp_definition(release, DEFINITION)

    assert document == {"status": {"statusCode": "ACTIVE"}}, "the second read is used"
    assert release.runner.calls == 2 and slept == [
        observability.AMP_READ_RETRY_SECONDS
    ], "exactly one retry after one backoff"


def test_persistent_transient_failure_still_raises_after_the_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observability, "_RETRY_SLEEP", lambda _seconds: None)
    release = _Release([(255, "", NO_CREDENTIALS)] * observability.AMP_READ_ATTEMPTS)

    with pytest.raises(
        observability.ReleaseError, match="cannot read the AMP Alertmanager"
    ):
        observability.describe_amp_definition(release, DEFINITION)

    assert release.runner.calls == observability.AMP_READ_ATTEMPTS, (
        "the attempts are bounded"
    )


@pytest.mark.parametrize(
    ("stderr", "calls"),
    (
        ("An error occurred (ResourceNotFoundException): no such definition", 1),
        ("An error occurred (AccessDeniedException): not authorised", 1),
    ),
)
def test_absence_and_hard_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, stderr: str, calls: int
) -> None:
    monkeypatch.setattr(observability, "_RETRY_SLEEP", lambda _seconds: None)
    release = _Release([(255, "", stderr)] * 3)

    if "ResourceNotFound" in stderr:
        assert observability.describe_amp_definition(release, DEFINITION) is None, (
            "absence is the answer, not an error"
        )
    else:
        with pytest.raises(observability.ReleaseError, match="cannot read"):
            observability.describe_amp_definition(release, DEFINITION)
    assert release.runner.calls == calls, "neither absence nor a hard error is retried"
