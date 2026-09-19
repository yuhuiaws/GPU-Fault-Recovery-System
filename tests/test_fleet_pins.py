"""``FleetPinRuntime``: the process-local pin snapshot behind the hot reload.

Every pin used to reach a control-plane process once, as environment the
kubelet resolved from ``gpu-fault-release-metadata`` at Pod start, so a release
that changed only data-plane artifacts still rolled three Deployments. The
runtime re-reads that ConfigMap and swaps the derived policies in place. These
tests drive the public entry points only: ``apply`` for the poller's path, an
injected reader for the poll loop, ``bind_fleet_registry`` for admission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from gpu_fault.fleet import FleetCompatibilityPolicy
from gpu_fault.fleet_pins import (
    FleetPinRuntime,
    kubernetes_config_map_reader,
    pin_content_sha256,
)
from tests._builders import copy_model
from tests.fleet._support import NOW, heartbeat, registry, signed
from tests.regional._fleet_pin_support import (
    CONFIG_MAP,
    EXECUTOR_ARTIFACT,
    NAMESPACE,
    NEW_ARTIFACT,
    NEW_EXECUTOR_ARTIFACT,
    FakeConfigMapReader,
    config_map_data,
    never_read,
    pins_from,
    startup_environment,
    wait_for,
)

DIGEST_VECTORS = [
    (
        {"b": "x,y", "a": "1", "z": ""},
        "585414de19bbac53560be605120197fd16f19f6ccc14d77b275817cf39280b65",
    ),
    # Non-ASCII, a slash and quotes: where an escaping choice would diverge.
    (
        {"b": "x,y", "a": "1", "z": 'é/"q"'},
        "5b3cde13320b3b0f790e697e4546d458df306a93d351fa175a136c39bf9db646",
    ),
]
STATUS_KEYS = {
    "source",
    "config_map",
    "content_sha256",
    "resource_version",
    "observed_at",
    "generation",
    "error",
}


def _runtime(
    environment: dict[str, str] | None = None,
    *,
    reader=never_read,
    poll_seconds: float = 1.0,
) -> FleetPinRuntime:
    return FleetPinRuntime(
        environment or startup_environment(),
        config_map=CONFIG_MAP,
        namespace=NAMESPACE,
        poll_seconds=poll_seconds,
        reader=reader,
        now=lambda: NOW,
    )


# --- Task 1: heartbeat admission follows the ConfigMap -------------------------


def test_heartbeat_admission_follows_the_applied_pin_window() -> None:
    """The same registry object admits artifact B only while the window holds it."""

    fleet = registry(now=lambda: NOW)
    runtime = _runtime()
    runtime.bind_fleet_registry(fleet)
    fleet.register(
        signed(copy_model(heartbeat("node-a"), artifact_sha256=NEW_ARTIFACT))
    )

    refused = fleet.readiness("cluster-a", ["node-a"])
    assert not refused.ready, "artifact B must be refused while only A is pinned"
    assert any(
        "artifact SHA-256 mismatch" in reason for reason in refused.nodes[0].reasons
    ), refused.nodes[0].reasons

    runtime.apply(
        config_map_data(
            **{
                "compatible-agent-artifact-sha256s": NEW_ARTIFACT,
                "compatible-agent-compatibility-digests": NEW_ARTIFACT,
            }
        ),
        resource_version="101",
    )
    widened = fleet.readiness("cluster-a", ["node-a"])
    assert widened.ready, f"the widened window must admit B: {widened.nodes[0].reasons}"

    runtime.apply(
        config_map_data(
            **{
                "compatible-agent-artifact-sha256s": "",
                "compatible-agent-compatibility-digests": "",
            }
        ),
        resource_version="102",
    )
    closed = fleet.readiness("cluster-a", ["node-a"])
    assert not closed.ready, "closing the window must refuse artifact B again"
    assert runtime.snapshot().generation == 3, runtime.status()


def test_agent_policy_replaces_only_the_pinned_fields() -> None:
    """Non-pin policy (TLS, agent version, heartbeat age) is the start-up value."""

    fleet = registry(now=lambda: NOW)
    startup = fleet.policy
    runtime = _runtime()
    runtime.bind_fleet_registry(fleet)

    runtime.apply(
        config_map_data(
            **{
                "required-agent-protocol-version": "4",
                "compatible-agent-protocol-versions": "3",
                "required-node-action-key-version": "2",
            }
        )
    )

    policy = fleet.policy
    assert policy.required_agent_protocol_version == 4, policy
    assert policy.compatible_agent_protocol_versions == frozenset({3}), policy
    assert policy.required_node_action_key_version == 2, policy
    assert policy.required_agent_version == startup.required_agent_version, policy
    assert policy.required_policy_version == startup.required_policy_version, policy
    assert policy.require_tls == startup.require_tls, policy
    assert policy.max_heartbeat_age_seconds == startup.max_heartbeat_age_seconds, policy
    assert policy.required_operations == startup.required_operations, policy


# --- Task 4: digest vector ------------------------------------------------------


@pytest.mark.parametrize(("data", "expected"), DIGEST_VECTORS)
def test_pin_content_digest_is_the_recorded_vector(
    data: dict[str, str], expected: str
) -> None:
    assert pin_content_sha256(data) == expected, data


@pytest.mark.parametrize(("data", "expected"), DIGEST_VECTORS)
def test_pin_content_digest_matches_jq_compact_sorted_output(
    data: dict[str, str], expected: str
) -> None:
    """``jq -cS '.data'`` is what the release script hashes; bytes must agree."""

    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is not on PATH")
    rendered = subprocess.run(
        [jq, "-cS", ".data"],
        input=json.dumps({"data": data}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    digest = hashlib.sha256(rendered.rstrip("\n").encode("utf-8")).hexdigest()
    assert digest == expected, rendered
    assert pin_content_sha256(data) == digest, rendered


# --- Task 5: lifecycle ----------------------------------------------------------


def test_snapshot_one_is_the_start_up_environment() -> None:
    environment = startup_environment()
    runtime = _runtime(environment)

    snapshot = runtime.snapshot()

    assert snapshot.source == "environment", snapshot
    assert snapshot.generation == 1, snapshot
    assert snapshot.resource_version is None, snapshot
    assert snapshot.observed_at is None, snapshot
    assert dict(snapshot.data) == pins_from(environment), snapshot.data
    assert snapshot.content_sha256 == pin_content_sha256(pins_from(environment)), (
        snapshot
    )
    assert runtime.status() == {
        "source": "environment",
        "config_map": CONFIG_MAP,
        "content_sha256": snapshot.content_sha256,
        "resource_version": None,
        "observed_at": None,
        "generation": 1,
        "error": None,
    }
    assert runtime.executor_policy().required_artifact_sha256 == EXECUTOR_ARTIFACT
    assert runtime.environment()["GPU_FAULT_SERVICE_ROLE"] == "worker", (
        "a non-pin variable must pass through the runtime's environment"
    )


def test_apply_with_the_same_digest_confirms_without_a_new_generation() -> None:
    runtime = _runtime()
    first = runtime.snapshot()

    confirmed = runtime.apply(pins_from(startup_environment()), resource_version="3")

    assert confirmed.generation == 1, confirmed
    assert (confirmed.source, confirmed.resource_version) == ("configmap", "3")
    assert confirmed.observed_at == NOW, confirmed
    assert confirmed.content_sha256 == first.content_sha256, confirmed
    assert runtime.snapshot() is confirmed, "the confirmation must be what is served"
    assert set(runtime.status()) == STATUS_KEYS, runtime.status()
    assert runtime.status()["observed_at"] == NOW.isoformat(), runtime.status()


def test_a_key_the_config_map_dropped_does_not_survive_from_start_up() -> None:
    """Finalize empties the ``compatible-*`` keys; the Pod's old env must not win."""

    environment = startup_environment(
        GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S=NEW_EXECUTOR_ARTIFACT
    )
    runtime = _runtime(environment)
    assert (
        NEW_EXECUTOR_ARTIFACT in runtime.executor_policy().compatible_artifact_sha256s
    )

    data = pins_from(environment)
    del data["compatible-regional-executor-artifact-sha256s"]
    runtime.apply(data)

    assert runtime.executor_policy().compatible_artifact_sha256s == frozenset(), (
        runtime.executor_policy()
    )
    assert (
        "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S"
        not in runtime.environment()
    ), dict(runtime.environment())


def test_step_batching_policy_follows_the_executor_window() -> None:
    runtime = _runtime()
    assert runtime.step_batching_policy().active is True, runtime.step_batching_policy()

    runtime.apply(
        config_map_data(**{"compatible-regional-executor-protocol-versions": "2"})
    )
    assert runtime.step_batching_policy().active is False, (
        "an admitted protocol-2 executor cannot decode a compound command"
    )

    runtime.apply(config_map_data())
    assert runtime.step_batching_policy().active is True, runtime.step_batching_policy()


def test_poll_thread_applies_a_changed_config_map_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reader = FakeConfigMapReader(config_map_data(), resource_version="7")
    runtime = _runtime(reader=reader, poll_seconds=0.01)

    with caplog.at_level(logging.INFO, logger="gpu_fault.fleet_pins"):
        runtime.start()
        try:
            assert wait_for(lambda: runtime.status()["source"] == "configmap"), (
                runtime.status()
            )
            confirmed = runtime.snapshot()
            assert confirmed.generation == 1, "same digest: a confirmation, no bump"
            assert confirmed.resource_version == "7", confirmed
            reader.data["compatible-regional-executor-artifact-sha256s"] = (
                NEW_EXECUTOR_ARTIFACT
            )
            reader.resource_version = "8"
            assert wait_for(lambda: runtime.snapshot().generation == 2), (
                runtime.status()
            )
            seen = reader.calls
            assert wait_for(lambda: reader.calls >= seen + 3), (
                "the loop must keep polling after a change"
            )
        finally:
            runtime.stop()

    applied = runtime.snapshot()
    assert applied.resource_version == "8", applied
    assert (
        NEW_EXECUTOR_ARTIFACT in runtime.executor_policy().compatible_artifact_sha256s
    )
    lines = [
        record.getMessage()
        for record in caplog.records
        if "applied" in record.getMessage()
    ]
    assert len(lines) == 1, lines
    assert applied.content_sha256 in lines[0], lines
    assert "generation=2" in lines[0], lines
    assert not any(
        thread.name == "gpu-fault-fleet-pins" for thread in threading.enumerate()
    ), "stop() must end the poll thread"
    calls = reader.calls
    time.sleep(0.05)
    assert reader.calls == calls, "a stopped runtime must not poll"


def test_read_failure_keeps_the_last_snapshot_and_reports_the_error() -> None:
    reader = FakeConfigMapReader(
        config_map_data(
            **{"compatible-regional-executor-artifact-sha256s": NEW_EXECUTOR_ARTIFACT}
        ),
        resource_version="5",
    )
    runtime = _runtime(reader=reader)
    assert runtime.refresh_once() is True, runtime.status()
    served = runtime.snapshot()

    reader.error = ConnectionError("apiserver unreachable")
    assert runtime.refresh_once() is False, runtime.status()

    status = runtime.status()
    assert status["error"] == "ConnectionError: apiserver unreachable", status
    assert status["generation"] == 2, status
    assert runtime.snapshot() is served, "a failed read must keep serving the last pins"
    assert (
        NEW_EXECUTOR_ARTIFACT in runtime.executor_policy().compatible_artifact_sha256s
    )

    reader.error = None
    assert runtime.refresh_once() is True, runtime.status()
    assert runtime.status()["error"] is None, runtime.status()


def test_invalid_pins_are_refused_and_the_last_snapshot_stays() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="artifact must be SHA-256"):
        runtime.apply(
            config_map_data(
                **{"required-regional-executor-artifact-sha256": "not-a-digest"}
            )
        )
    assert runtime.snapshot().generation == 1, runtime.status()

    reader = FakeConfigMapReader(
        config_map_data(**{"compatible-agent-artifact-sha256s": "zz"})
    )
    polled = _runtime(reader=reader)
    assert polled.refresh_once() is False, polled.status()
    assert "compatible digests must be SHA-256" in str(polled.status()["error"]), (
        polled.status()
    )
    assert polled.snapshot().generation == 1, polled.status()
    assert (
        polled.agent_policy(FleetCompatibilityPolicy()).compatible_artifact_sha256s
        == frozenset()
    ), "a refused pin set must not leak into the derived policy"


def test_from_environment_reads_the_name_namespace_and_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GPU_FAULT_FLEET_PIN_CONFIGMAP",
        "GPU_FAULT_FLEET_PIN_POLL_SECONDS",
        "GPU_FAULT_NAMESPACE",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = FleetPinRuntime.from_environment(
        startup_environment(), reader=never_read
    )
    assert (defaults.config_map, defaults.namespace, defaults.poll_seconds) == (
        "gpu-fault-release-metadata",
        "gpu-fault-system",
        2.0,
    )

    custom = FleetPinRuntime.from_environment(
        startup_environment(
            GPU_FAULT_FLEET_PIN_CONFIGMAP="pins",
            GPU_FAULT_NAMESPACE="ns",
            GPU_FAULT_FLEET_PIN_POLL_SECONDS="0.5",
        ),
        reader=never_read,
    )
    assert (custom.config_map, custom.namespace, custom.poll_seconds) == (
        "pins",
        "ns",
        0.5,
    )

    with pytest.raises(ValueError, match="GPU_FAULT_FLEET_PIN_POLL_SECONDS"):
        FleetPinRuntime.from_environment(
            startup_environment(GPU_FAULT_FLEET_PIN_POLL_SECONDS="0"), reader=never_read
        )


def test_default_reader_uses_the_cluster_api_and_needs_an_explicit_kubeconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kubernetes.client
    import kubernetes.config
    from kubernetes.config.config_exception import ConfigException

    calls: list[str] = []
    documents = [
        SimpleNamespace(
            data={"required-agent-protocol-version": "3"},
            metadata=SimpleNamespace(resource_version="42"),
        ),
        SimpleNamespace(data=None, metadata=SimpleNamespace(resource_version="43")),
    ]

    class FakeCoreV1Api:
        def read_namespaced_config_map(self, name, namespace, **kwargs):
            calls.append(f"{namespace}/{name}")
            return documents.pop(0)

    def not_in_cluster() -> None:
        raise ConfigException("Service host/port is not set.")

    monkeypatch.setattr(kubernetes.config, "load_incluster_config", not_in_cluster)
    monkeypatch.setattr(
        kubernetes.config, "load_kube_config", lambda: calls.append("kubeconfig")
    )
    monkeypatch.setattr(kubernetes.client, "CoreV1Api", FakeCoreV1Api)
    monkeypatch.delenv("KUBECONFIG", raising=False)
    reader = kubernetes_config_map_reader(CONFIG_MAP, NAMESPACE)

    with pytest.raises(ConfigException):
        reader()
    assert calls == [], "without KUBECONFIG the reader must not open a kubeconfig"

    monkeypatch.setenv("KUBECONFIG", "/nonexistent/explicit-kubeconfig")
    assert reader() == ({"required-agent-protocol-version": "3"}, "42")
    assert reader() == ({}, "43"), "an empty ConfigMap reads as an empty mapping"
    assert calls == [
        "kubeconfig",
        f"{NAMESPACE}/{CONFIG_MAP}",
        f"{NAMESPACE}/{CONFIG_MAP}",
    ]
