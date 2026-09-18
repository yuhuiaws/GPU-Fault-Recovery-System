from __future__ import annotations

import errno
from urllib.error import URLError

import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError

from gpu_fault.dataplane_metrics import (
    MetricFamily,
    MetricsServer,
    start_metrics_server,
)
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.regional_compatibility import (
    CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    RegionalExecutorCompatibilityPolicy,
)
from gpu_fault.transport_errors import (
    retryable_transport_error,
    retryable_transport_result,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("error", "prefix"),
    [
        (
            OSError(errno.ENETUNREACH, "unit network unreachable"),
            "temporary transport failure",
        ),
        (
            EndpointConnectionError(endpoint_url="https://unit.invalid"),
            "temporary AWS transport failure",
        ),
        (URLError(TimeoutError("unit timeout")), "temporary transport failure"),
    ],
)
def test_real_transport_exceptions_produce_waiting_results_with_the_original_lease(
    error: Exception, prefix: str
) -> None:
    result = retryable_transport_result(
        error, lease_token="unit-lease", executor_id="unit-executor"
    )
    assert result is not None
    assert result.status is RemoteCommandStatus.WAITING
    assert result.lease_token == "unit-lease"
    assert result.details["reason"].startswith(prefix), result.details
    assert result.details["executor_id"] == "unit-executor"
    assert result.details["exception_type"] == type(error).__name__


@pytest.mark.parametrize("depth", [0, 4, 5, 7])
def test_exception_chain_walk_has_a_fixed_depth_budget(depth: int) -> None:
    error: Exception = TimeoutError("unit timeout")
    for _ in range(depth):
        outer = RuntimeError("unit wrapper")
        outer.__cause__ = error
        error = outer
    assert (retryable_transport_error(error) is not None) is (depth < 5)


def test_cyclic_and_nontransport_errors_are_not_misclassified_as_retryable() -> None:
    cycle = RuntimeError("unit cycle")
    cycle.__cause__ = cycle
    for error in (
        cycle,
        NoCredentialsError(),
        PermissionError("unit policy refusal"),
        ValueError("unit invalid"),
    ):
        assert retryable_transport_error(error) is None
        assert (
            retryable_transport_result(error, lease_token="unit", executor_id="unit")
            is None
        )


@pytest.mark.parametrize(
    "values",
    [
        {"GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": "0"},
        {"GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": "0"},
        {"GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "invalid"},
        {"GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S": "invalid"},
        {"GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": "invalid"},
        {"GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS": "invalid"},
    ],
)
def test_executor_compatibility_loader_rejects_invalid_pins(
    values: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="positive|SHA-256"):
        RegionalExecutorCompatibilityPolicy.from_mapping(values)


@pytest.mark.parametrize("window", [False, True])
def test_executor_compatibility_keeps_artifact_and_digest_windows_independent(
    window: bool,
) -> None:
    protocol = CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
    policy = RegionalExecutorCompatibilityPolicy(
        required_version=protocol,
        compatible_versions=frozenset({2}),
        required_artifact_sha256="a" * 64,
        compatible_artifact_sha256s=frozenset({"b" * 64}) if window else frozenset(),
        required_compatibility_digest="c" * 64,
        compatible_compatibility_digests=frozenset({"d" * 64})
        if window
        else frozenset(),
    )
    assert policy.minimum_accepted_version == 2
    assert policy.rejection_reason(protocol, "A" * 64, "C" * 64) is None
    assert "protocol version mismatch" in policy.rejection_reason(
        99, "a" * 64, "c" * 64
    )
    missing = policy.rejection_reason(protocol, None, "c" * 64)
    assert "artifact mismatch" in missing and "MISSING" in missing
    wrong_digest = policy.rejection_reason(protocol, "a" * 64, "e" * 64)
    assert "compatibility digest mismatch" in wrong_digest
    assert ("one of" in wrong_digest) is window
    if window:
        assert policy.rejection_reason(2, "B" * 64, "D" * 64) is None
    else:
        assert "artifact mismatch" in policy.rejection_reason(
            protocol, "b" * 64, "c" * 64
        )


def test_empty_compatibility_window_uses_the_artifact_fallback_only_when_configured() -> (
    None
):
    protocol = CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
    unrestricted = RegionalExecutorCompatibilityPolicy.from_mapping({})
    assert unrestricted.rejection_reason(protocol) is None
    legacy = unrestricted.rejection_reason(3)
    assert legacy is not None and "protocol version mismatch" in legacy
    pinned = RegionalExecutorCompatibilityPolicy.from_mapping(
        {"GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": "a" * 64}
    )
    assert pinned.rejection_reason(protocol, "A" * 64) is None
    assert "compatibility digest mismatch" in pinned.rejection_reason(
        protocol, "a" * 64, "b" * 64
    )


def test_unstarted_metrics_server_is_stoppable_without_binding_a_socket() -> None:
    family = MetricFamily(
        "gpu_fault_unit_", counters=[("events", "unit")], timestamps=[("seen", "unit")]
    )
    family.inc("events", 2)
    family.mark("seen", 1.5)
    server = MetricsServer(family, port=12345)
    assert server.port == 12345
    assert server.is_running is False
    server.stop()
    server.stop()
    assert server.is_running is False
    assert "gpu_fault_unit_events 2\n" in family.render()
    assert "gpu_fault_unit_seen 1.5\n" in family.render()


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_disabled_or_invalid_metrics_port_never_attempts_a_listener(
    monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    calls = []
    monkeypatch.setattr(MetricsServer, "start", lambda self: calls.append(self))
    family = MetricFamily("gpu_fault_unit_", counters=[])
    assert start_metrics_server(family, port=port) is None
    assert calls == []
