"""Tell AWS misconfiguration apart from an executor defect.

The data-plane executor once reported a real deployment gap as its own
bug: the ServiceAccount had no ``eks.amazonaws.com/role-arn`` annotation,
so HyperPod ``DescribeCluster`` raised ``NoCredentialsError``, which is
not in the ``(HyperPodAdapterError, KeyError, ValueError)`` tuple the
preflight call site catches. It escaped to the executor's catch-all and
landed in the result as ``executor_internal_error: true`` with an
``exception_type`` -- indistinguishable from "the adapter is broken",
and counted into the internal-error alerting metric.

Missing credentials, an unassumable role and a denied API call are all
things an operator fixes by changing configuration, never by changing
code. They deserve their own classification: a FAILED step whose reason
names the knob, and no internal-error count.

``botocore`` is an optional dependency (``[hyperpod]``/``[collectors]``),
so this module must not import it. It matches on the exception's module
and class name, plus the ``ClientError`` error code, which is stable
public API of the SDK's error contract.
"""

from __future__ import annotations

from typing import Any

#: ``botocore.exceptions`` classes that mean "no usable credentials" or
#: "the SDK cannot work out where/who to be". Every one of these is
#: raised before a request is signed, so none of them can be a transient
#: provider failure. Network-level errors (``EndpointConnectionError``,
#: ``ConnectTimeoutError``) are deliberately absent: those are retryable
#: and are neither misconfiguration nor a defect.
CREDENTIAL_EXCEPTION_NAMES = frozenset(
    {
        "CredentialRetrievalError",
        "InvalidConfigError",
        "MetadataRetrievalError",
        "NoCredentialsError",
        "NoRegionError",
        "PartialCredentialsError",
        "ProfileNotFound",
        "SSOTokenLoadError",
        "TokenRetrievalError",
        "UnauthorizedSSOTokenError",
    }
)

#: ``ClientError`` codes that mean the caller's identity or its policy is
#: wrong. ``InvalidIdentityToken`` is what a wrong role ARN in the IRSA
#: annotation looks like: credentials resolve, then
#: ``AssumeRoleWithWebIdentity`` refuses them.
AUTHORIZATION_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "AuthFailure",
        "ExpiredToken",
        "ExpiredTokenException",
        "InvalidClientTokenId",
        "InvalidIdentityToken",
        "InvalidSignatureException",
        "MissingAuthenticationToken",
        "MissingAuthenticationTokenException",
        "SignatureDoesNotMatch",
        "UnrecognizedClientException",
        "UnauthorizedOperation",
    }
)

#: How far up the ``__cause__``/``__context__`` chain to look. An
#: adapter that wraps a ``ClientError`` in its own error would otherwise
#: hide the classification; a bound keeps a cyclic chain from hanging.
_MAX_CHAIN_DEPTH = 4


def _error_code(exc: BaseException) -> str | None:
    response: Any = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def _classify_one(exc: BaseException) -> str | None:
    module = type(exc).__module__ or ""
    name = type(exc).__name__
    if not module.startswith("botocore"):
        return None
    if name in CREDENTIAL_EXCEPTION_NAMES:
        return (
            f"AWS credentials are not configured ({name}): the pod has "
            "no usable role. On EKS this is the ServiceAccount's "
            "eks.amazonaws.com/role-arn annotation, which must exist "
            "and be listed in the role's trust policy"
        )
    code = _error_code(exc)
    if code in AUTHORIZATION_ERROR_CODES:
        return (
            f"AWS rejected the call as unauthorized ({code}): the role "
            "the pod assumed is missing the required permission, or the "
            "eks.amazonaws.com/role-arn annotation names a role whose "
            "trust policy does not list this ServiceAccount"
        )
    return None


def aws_configuration_error(exc: BaseException) -> str | None:
    """Operator-facing reason if ``exc`` is AWS misconfiguration.

    Returns ``None`` for anything else, so callers keep their existing
    behaviour (re-raise, or report an internal error) untouched.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        reason = _classify_one(current)
        if reason is not None:
            return reason
        current = current.__cause__ or current.__context__
    return None


def missing_aws_credentials() -> str | None:
    """Reason if this process has no resolvable AWS credentials.

    A local check, not an API call: ``botocore`` resolves the credential
    chain without contacting AWS, so this is safe to run at startup and
    catches the exact deployment gap that used to surface only when a
    real fault arrived hours later. A wrong role ARN still resolves here
    and is classified later by :func:`aws_configuration_error` when the
    first request is signed.
    """

    try:
        import boto3
    except ImportError:
        return None
    try:
        credentials = boto3.Session().get_credentials()
    except Exception as exc:  # pragma: no cover - defensive
        return aws_configuration_error(exc) or (
            f"AWS credential resolution failed: {type(exc).__name__}: {exc}"
        )
    if credentials is not None:
        return None
    return (
        "no AWS credentials are resolvable in this pod: the "
        "ServiceAccount is most likely missing its "
        "eks.amazonaws.com/role-arn annotation (IRSA), so every "
        "HyperPod call would fail at the moment a real fault arrives"
    )
