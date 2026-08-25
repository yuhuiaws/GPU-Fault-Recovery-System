from __future__ import annotations

import pytest

from gpu_fault.aws_errors import (
    AUTHORIZATION_ERROR_CODES,
    CREDENTIAL_EXCEPTION_NAMES,
    aws_configuration_error,
    missing_aws_credentials,
)


def _botocore_exception(name: str, *, response: dict | None = None) -> Exception:
    """A stand-in for a botocore exception class.

    The classifier matches on ``__module__`` plus class name precisely so
    it never has to import botocore (an optional extra). Synthesising the
    class here keeps these tests runnable without the SDK installed;
    ``test_classified_exception_names_still_exist_in_botocore`` is what
    catches an upstream rename.
    """

    namespace: dict = {"__module__": "botocore.exceptions"}
    exception_type = type(name, (Exception,), namespace)
    exc = exception_type(name)
    if response is not None:
        exc.response = response
    return exc


def _client_error(code: str) -> Exception:
    return _botocore_exception(
        "ClientError",
        response={
            "Error": {"Code": code, "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
    )


def test_missing_credentials_are_configuration_not_defect() -> None:
    reason = aws_configuration_error(_botocore_exception("NoCredentialsError"))

    assert reason is not None
    # The reason has to name the knob an operator turns, otherwise it is
    # no more actionable than the stack trace it replaces.
    assert "eks.amazonaws.com/role-arn" in reason
    assert "NoCredentialsError" in reason


@pytest.mark.parametrize("name", sorted(CREDENTIAL_EXCEPTION_NAMES))
def test_every_credential_exception_is_classified(name: str) -> None:
    assert aws_configuration_error(_botocore_exception(name)) is not None


@pytest.mark.parametrize("code", sorted(AUTHORIZATION_ERROR_CODES))
def test_every_authorization_code_is_classified(code: str) -> None:
    reason = aws_configuration_error(_client_error(code))

    assert reason is not None
    assert code in reason


def test_wrong_role_arn_reports_the_trust_policy() -> None:
    reason = aws_configuration_error(_client_error("InvalidIdentityToken"))

    assert reason is not None
    # Credentials resolve with a wrong ARN; the failure only appears when
    # AssumeRoleWithWebIdentity refuses them, so the remediation is the
    # trust policy rather than the annotation's presence.
    assert "trust policy" in reason


def test_wrapped_credential_error_is_still_classified() -> None:
    # The HyperPod adapter wraps provider failures in its own error, so
    # the classification has to survive one level of re-raise.
    inner = _botocore_exception("NoCredentialsError")
    try:
        try:
            raise inner
        except Exception as exc:
            raise RuntimeError("describe cluster failed") from exc
    except RuntimeError as outer:
        assert aws_configuration_error(outer) is not None


def _wrap(inner: BaseException, depth: int) -> BaseException:
    exc = inner
    for _ in range(depth):
        outer = RuntimeError("wrapped")
        outer.__cause__ = exc
        exc = outer
    return exc


def test_wrapping_within_the_bound_is_classified() -> None:
    inner = _botocore_exception("NoCredentialsError")

    assert aws_configuration_error(_wrap(inner, 3)) is not None


def test_deep_wrapping_beyond_the_bound_is_not_classified() -> None:
    inner = _botocore_exception("NoCredentialsError")

    # Bounded on purpose: an unbounded walk hangs on a cyclic chain. Four
    # levels is more than any wrapping this codebase does, so falling off
    # the end means the internal-error path, which is the safe direction.
    assert aws_configuration_error(_wrap(inner, 4)) is None


def test_cyclic_cause_chain_terminates() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first

    assert aws_configuration_error(first) is None


def test_non_aws_exception_is_not_classified() -> None:
    # A genuine executor defect must keep reaching the internal-error
    # path, otherwise this classifier hides real bugs.
    assert aws_configuration_error(AttributeError("core")) is None
    assert aws_configuration_error(KeyError("step")) is None


def test_lookalike_from_another_module_is_not_classified() -> None:
    impostor = type(
        "NoCredentialsError", (Exception,), {"__module__": "mypackage.errors"}
    )("nothing to do with AWS")

    assert aws_configuration_error(impostor) is None


def test_retryable_network_error_is_not_configuration() -> None:
    # EndpointConnectionError means "try again", not "fix the manifest";
    # classifying it would turn a transient blip into a FAILED step with
    # an operator instruction that does not apply.
    assert (
        aws_configuration_error(_botocore_exception("EndpointConnectionError")) is None
    )


def test_unrelated_client_error_code_is_not_configuration() -> None:
    assert aws_configuration_error(_client_error("Throttling")) is None


def test_malformed_client_error_response_is_tolerated() -> None:
    for response in (None, "denied", {}, {"Error": "denied"}):
        exc = _botocore_exception("ClientError")
        if response is not None:
            exc.response = response  # type: ignore[attr-defined]
        assert aws_configuration_error(exc) is None


def test_classified_exception_names_still_exist_in_botocore() -> None:
    exceptions = pytest.importorskip("botocore.exceptions")

    missing = sorted(
        name for name in CREDENTIAL_EXCEPTION_NAMES if not hasattr(exceptions, name)
    )

    # Matching by name is only safe while the names exist upstream: a
    # rename would silently return every one of these to the
    # internal-error path.
    assert missing == []


def test_missing_credentials_probe_reports_the_gap(monkeypatch) -> None:
    boto3 = pytest.importorskip("boto3")

    class FakeSession:
        def get_credentials(self):
            return None

    monkeypatch.setattr(boto3, "Session", FakeSession)

    reason = missing_aws_credentials()

    assert reason is not None
    assert "eks.amazonaws.com/role-arn" in reason


def test_missing_credentials_probe_is_quiet_when_resolvable(monkeypatch) -> None:
    boto3 = pytest.importorskip("boto3")

    class FakeSession:
        def get_credentials(self):
            return object()

    monkeypatch.setattr(boto3, "Session", FakeSession)

    assert missing_aws_credentials() is None


def test_missing_credentials_probe_makes_no_api_call(monkeypatch) -> None:
    boto3 = pytest.importorskip("boto3")
    calls = []

    class FakeSession:
        def get_credentials(self):
            calls.append("resolve")
            return object()

        def client(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("the startup probe must not contact AWS")

    monkeypatch.setattr(boto3, "Session", FakeSession)

    assert missing_aws_credentials() is None
    assert calls == ["resolve"]
