"""Real inventory Collector logic with an isolated sink and no host mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.collectors import sinks
from gpu_fault.collectors.host.collector import HostTelemetryCollector
from scripts.e2e.regional.collector_inventory_sampling import (
    inventory_delivery_errors,
    isolated_inventory_records,
)
from scripts.e2e.regional.probes import collector_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional import test_collector_env_safety as env_safety

env_host = env_safety.env_host


@pytest.fixture
def sampling_host(
    env_host: env_safety.EnvHost, monkeypatch: pytest.MonkeyPatch
) -> env_safety.EnvHost:
    host = env_host
    # The installer's CA file must exist: the publisher refuses a missing one
    # before it constructs any HTTP client.
    ca_file = probe.COLLECTOR_ENV.parent / "control-plane-ca.crt"
    ca_file.write_text(
        "-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n"
    )
    contents = (
        probe.COLLECTOR_ENV.read_text()
        + "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT=16\n"
        + "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS=300\n"
        + "GPU_FAULT_RUNTIME_PROFILE_VERSION=profile-fixture\n"
        + "GPU_FAULT_NODE_INSTANCE_TYPE=ml.p5en.48xlarge\n"
        + "GPU_FAULT_CONTROL_PLANE_URL=https://fixture.invalid\n"
        + "GPU_FAULT_CONTROL_PLANE_TOKEN=fixture-private-token\n"
        # The installer writes the CA as SSL_CERT_FILE (install-gpu-fault-collector.sh);
        # there is no GPU_FAULT_CONTROL_PLANE_CA_FILE in a real collector.env.
        + f"SSL_CERT_FILE={ca_file}\n"
    )
    probe.COLLECTOR_ENV.write_text(contents)
    process = probe.PROC_ROOT / "1234"
    process.mkdir()
    (process / "stat").write_text("1234 (fixture host) " + " ".join(["0"] * 20))
    (process / "environ").write_bytes(b"\0".join(contents.encode().splitlines()))
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))

    def run(
        command: list[str], *, check: bool = True, timeout: float = 180
    ) -> subprocess.CompletedProcess[str]:
        # Same signature as the real ``probe.run``: the isolated Collector must
        # reach it through an adapter that speaks the Collector's runner
        # contract, not by being handed ``run`` itself (2026-09-17 live FAIL:
        # ``run() got an unexpected keyword argument 'capture_output'``).
        if command[0] == "nvidia-smi":
            assert tuple(command) == HostTelemetryCollector.GPU_QUERY_ARGV
            assert check is False and timeout > 0
            return subprocess.CompletedProcess(
                command, 0, "\n".join(f"GPU-{index}, 0" for index in range(8)), ""
            )
        return host.run(command, check=check, timeout=timeout)

    def sleep(seconds: float) -> None:
        host.now += seconds

    monkeypatch.setattr(probe, "run", run)
    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: host.now, sleep=sleep)
    )
    monkeypatch.setattr(sinks, "HttpEventSink", env_safety.forbidden)
    return host


def sampling_arguments() -> argparse.Namespace:
    return probe.parser().parse_args(
        [
            "sample-gpu-inventory",
            "--run-id",
            "collect004-isolated-fixture",
            "--cluster-id",
            "cluster-fixture",
            "--node-id",
            "node-fixture",
            "--expected-env-sha256",
            hashlib.sha256(probe.COLLECTOR_ENV.read_bytes()).hexdigest(),
            "--expected-boot-id",
            "boot-fixture-a",
            "--expected-gpu-count",
            "9",
        ]
    )


def sample(host: env_safety.EnvHost) -> dict[str, Any]:
    arguments = sampling_arguments()
    arguments.handler(arguments)
    return host.emitted[-1]


def records(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    return isolated_inventory_records(
        receipt,
        run_id="collect004-isolated-fixture",
        cluster_id="cluster-fixture",
        node_id="node-fixture",
        boot_id="boot-fixture-a",
        baseline_sha256=receipt["identity"]["configuration"]["file"]["sha256"],
        expected_gpu_count=9,
        interval_seconds=15,
        observed_after=datetime.fromisoformat(receipt["batches"][0]["observed_at"]),
        tolerance=0.5,
    )


def publication_arguments(receipt: dict[str, Any]) -> argparse.Namespace:
    return probe.parser().parse_args(
        [
            "publish-gpu-inventory",
            "--run-id",
            receipt["run_id"],
            "--expected-sha256",
            receipt["sha256"],
            "--receipt-json",
            json.dumps(receipt),
            "--confirm",
            "PUBLISH_GPU_INVENTORY",
        ]
    )


def reseal(receipt: dict[str, Any]) -> None:
    receipt["sha256"] = probe.inventory_receipt_digest(receipt)


def test_real_collector_debounce_never_posts_or_changes_production(
    sampling_host: env_safety.EnvHost,
) -> None:
    original = probe.COLLECTOR_ENV.read_bytes()
    invocation = sampling_host.invocations.copy()
    receipt = sample(sampling_host)
    captured = records(receipt)
    assert receipt["publication_performed"] is False
    assert receipt["kind"] == "ISOLATED_GPU_INVENTORY"
    assert len(captured) == 2
    assert [row["samples"][-1]["value"] for row in captured] == [0, 1]
    assert "baseline" in captured[0]["edge_filter_reasons"]
    assert (
        datetime.fromisoformat(captured[1]["observed_at"])
        - datetime.fromisoformat(captured[0]["observed_at"])
    ).total_seconds() == 15
    assert probe.COLLECTOR_ENV.read_bytes() == original
    assert sampling_host.invocations == invocation
    assert sampling_host.mutations() == []
    assert not list(probe.ACCEPTANCE_STATE.glob("*")), (
        'test_real_collector_debounce_never_posts_or_changes_production: expected no list(probe.ACCEPTANCE_STATE.glob("*"))'
    )
    assert "fixture-private-token" not in json.dumps(receipt)
    assert "PRIVATE_FIXTURE_SENTINEL" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_id", "other"),
        ("node_id", "other"),
        ("expected_boot_id", "other"),
        ("expected_env_sha256", "0" * 64),
        ("expected_gpu_count", 8),
        ("expected_gpu_count", 10),
        ("expected_gpu_count", True),
    ],
)
def test_sampling_scope_rejection_has_no_service_or_network_action(
    sampling_host: env_safety.EnvHost, field: str, value: Any
) -> None:
    arguments = sampling_arguments()
    setattr(arguments, field, value)
    with pytest.raises(probe.ProbeError):
        probe.sample_gpu_inventory(arguments)
    assert sampling_host.mutations() == []
    assert sampling_host.emitted == []


@pytest.mark.parametrize("drift", ["boot", "env", "process"])
def test_changed_identity_during_sampling_never_yields_a_publishable_receipt(
    sampling_host: env_safety.EnvHost, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    def sleep(seconds: float) -> None:
        sampling_host.now += seconds
        if drift == "boot":
            probe.BOOT_ID_FILE.write_text("another-boot")
        elif drift == "env":
            probe.COLLECTOR_ENV.write_text(
                probe.COLLECTOR_ENV.read_text() + "# drift\n"
            )
        else:
            sampling_host.invocations[probe.HOST_COLLECTOR_UNIT] = "another-invocation"

    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: sampling_host.now, sleep=sleep)
    )
    with pytest.raises(probe.ProbeError):
        sample(sampling_host)
    assert sampling_host.mutations() == []
    assert sampling_host.emitted == []


def test_isolated_samples_carry_the_node_instance_type(
    sampling_host: env_safety.EnvHost,
) -> None:
    """Every isolated inventory sample must carry the installer's instance type.

    The planner freezes ``node_instance_type`` from the finding into the
    RESTART_NODE plan and VALIDATE_GPU caps ``expected`` at that type's
    physical GPU count; without the label the live workflow validated
    ``expected=9`` on an 8-GPU node and FAILED (2026-09-17 attempt 3).
    """
    receipt = sample(sampling_host)
    assert receipt["identity"]["node_instance_type"] == "ml.p5en.48xlarge"
    for batch in receipt["batches"]:
        for item in batch["samples"]:
            assert item["labels"]["node_instance_type"] == "ml.p5en.48xlarge"
    stripped = deepcopy(receipt)
    for item in stripped["batches"][-1]["samples"]:
        item["labels"].pop("node_instance_type")
    reseal(stripped)
    with pytest.raises(RegionalFixtureError, match="scope"):
        records(stripped)


def test_publisher_reads_the_ca_under_the_installer_key() -> None:
    """The node installer publishes the control-plane CA as ``SSL_CERT_FILE``.

    The publisher must read that key: a 2026-09-17 live attempt failed with
    "requires the configured HTTPS endpoint" because the probe looked for a
    ``GPU_FAULT_CONTROL_PLANE_CA_FILE`` that no collector.env carries.
    """
    installer = Path("deploy/node/install-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )
    assert (
        f'write_env {probe.COLLECTOR_CA_ENV_KEY} "/etc/gpu-fault/control-plane-ca.crt"'
        in installer
    )
    assert probe.COLLECTOR_CA_ENV_KEY == "SSL_CERT_FILE"


def test_collector_runner_speaks_the_bounded_process_runner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Collector calls ``runner(argv, capture_output=, text=, timeout=, check=)``.

    The probe's adapter must accept exactly that shape and forward the bound
    and the check flag to ``run``; a signature drift on either side turned every
    live sample into a collection error.
    """
    import inspect

    from gpu_fault.collectors.process import BoundedProcessRunner

    product = {
        name: parameter.kind
        for name, parameter in inspect.signature(
            BoundedProcessRunner.__call__
        ).parameters.items()
        if name != "self"
    }
    adapter = {
        name: parameter.kind
        for name, parameter in inspect.signature(
            probe.collector_runner
        ).parameters.items()
    }
    assert adapter == product
    calls: list[tuple[list[str], bool, float]] = []

    def run(
        command: list[str], *, check: bool = True, timeout: float = 180
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, check, timeout))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(probe, "run", run)
    completed = probe.collector_runner(
        ["nvidia-smi"], capture_output=True, text=True, timeout=15.5, check=False
    )
    assert completed.stdout == "ok"
    assert calls == [(["nvidia-smi"], False, 15.5)]
    with pytest.raises(probe.ProbeError, match="captured text"):
        probe.collector_runner(["nvidia-smi"], timeout=15.5)


def test_query_failure_is_not_a_missing_card_sample(
    sampling_host: env_safety.EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(
        command: list[str], *, check: bool = True, timeout: float = 180
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            raise subprocess.TimeoutExpired(command, 15)
        return sampling_host.run(command, check=check, timeout=timeout)

    monkeypatch.setattr(probe, "run", run)
    with pytest.raises(probe.ProbeError, match="could not read inventory"):
        sample(sampling_host)
    assert sampling_host.emitted == []
    assert sampling_host.mutations() == []


def test_publisher_sends_only_the_two_original_batches_once(
    sampling_host: env_safety.EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = sample(sampling_host)
    captured = deepcopy(receipt["batches"])
    calls: list[dict[str, Any]] = []

    def factory(url: str, **kwargs: Any) -> Any:
        assert url == "https://fixture.invalid"
        assert kwargs == {
            "bearer_token": "fixture-private-token",
            "timeout_seconds": 15,
            "max_attempts": 1,
            "outbox_path": None,
        }

        def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
            assert path == "/v1/collector-events/host-telemetry"
            calls.append(deepcopy(payload))
            return {}

        return SimpleNamespace(post=post)

    monkeypatch.setattr(sinks, "HttpEventSink", factory)
    arguments = publication_arguments(receipt)
    arguments.handler(arguments)
    assert calls == captured
    assert sampling_host.emitted[-1] == {
        "run_id": receipt["run_id"],
        "sample_sha256": receipt["sha256"],
        "publication_performed": True,
        "batch_ids": [item["batch_id"] for item in captured],
    }
    assert sampling_host.mutations() == []
    assert "fixture-private-token" not in json.dumps(sampling_host.emitted)


@pytest.mark.parametrize(
    "drift",
    [
        "digest",
        "run",
        "boot",
        "kind",
        "published",
        "missing",
        "extra",
        "foreign",
        "early",
        "time",
        "stale",
    ],
)
def test_publisher_refuses_drift_before_constructing_any_http_client(
    sampling_host: env_safety.EnvHost, drift: str
) -> None:
    receipt = sample(sampling_host)
    if drift == "digest":
        receipt["sha256"] = "0" * 64
    elif drift == "run":
        receipt["run_id"] = "other-run"
    elif drift == "boot":
        receipt["identity"]["boot_id"] = "other-boot"
    elif drift == "kind":
        receipt["kind"] = "LIVE"
    elif drift == "published":
        receipt["publication_performed"] = True
    elif drift == "missing":
        receipt["batches"].pop()
    elif drift == "extra":
        receipt["batches"].append(deepcopy(receipt["batches"][-1]))
    elif drift == "foreign":
        receipt["batches"][-1]["node_id"] = "another-node"
    elif drift == "early":
        receipt["batches"][0]["samples"][-1]["value"] = 1
    elif drift == "time":
        receipt["sampled_at"] = "invalid-clock"
    else:
        receipt["sampled_at"] = (
            datetime.fromisoformat(receipt["sampled_at"]) - timedelta(seconds=301)
        ).isoformat()
    if drift != "digest":
        reseal(receipt)
    arguments = publication_arguments(receipt)
    arguments.run_id = "collect004-isolated-fixture"
    with pytest.raises(probe.ProbeError):
        arguments.handler(arguments)
    assert sampling_host.mutations() == []


@pytest.mark.parametrize("failed_index", [0, 1])
def test_failed_publisher_has_no_retry_or_outbox(
    sampling_host: env_safety.EnvHost,
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int,
) -> None:
    receipt = sample(sampling_host)
    calls = []

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload["batch_id"])
        if len(calls) - 1 == failed_index:
            raise sinks.CollectorError("private transport detail", status_code=503)
        return {}

    monkeypatch.setattr(
        sinks, "HttpEventSink", lambda *args, **kwargs: SimpleNamespace(post=post)
    )
    with pytest.raises(probe.ProbeError, match="acceptance is unresolved") as error:
        publication_arguments(receipt).handler(publication_arguments(receipt))
    assert len(calls) == failed_index + 1
    assert "private transport detail" not in str(error.value)
    assert sampling_host.mutations() == []


def persisted_projection(captured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What ``HOST_INVENTORY_EVIDENCE`` returns: only the mismatch sample per batch."""
    projected = deepcopy(captured)
    for record in projected:
        record["samples"] = [
            item
            for item in record["samples"]
            if item["name"] == "gpu_inventory_mismatch"
        ]
    return projected


def test_persisted_evidence_must_match_the_captured_payloads(
    sampling_host: env_safety.EnvHost,
) -> None:
    captured = records(sample(sampling_host))
    persisted = persisted_projection(captured)
    # The control plane's evidence probe projects each batch to its
    # gpu_inventory_mismatch sample (2026-09-17 live: comparing the whole
    # five-sample batch against that projection failed every attempt).
    assert inventory_delivery_errors(captured, persisted) == []
    assert inventory_delivery_errors(captured, deepcopy(captured)), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, deepcopy(captured))"
    )
    assert inventory_delivery_errors(captured, persisted[:1]), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, persisted[:1])"
    )
    changed = deepcopy(persisted)
    changed[-1]["samples"][-1]["labels"]["expected_count"] = "20"
    assert inventory_delivery_errors(captured, changed), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, changed)"
    )
    dropped = deepcopy(persisted)
    dropped[-1]["samples"] = []
    assert inventory_delivery_errors(captured, dropped), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, dropped)"
    )
    assert inventory_delivery_errors(captured, [persisted[0], persisted[0]]), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, [persisted[0], persisted[0]])"
    )
    reordered = deepcopy(persisted)
    reordered[-1]["edge_filter_reasons"] = []
    assert inventory_delivery_errors(captured, reordered), (
        "test_persisted_evidence_must_match_the_captured_payloads: expected inventory_delivery_errors(captured, reordered)"
    )


def test_isolated_receipt_cannot_be_counted_as_an_already_delivered_live_proof(
    sampling_host: env_safety.EnvHost,
) -> None:
    receipt = sample(sampling_host)
    receipt["publication_performed"] = True
    reseal(receipt)
    with pytest.raises(RegionalFixtureError, match="identity"):
        records(receipt)
