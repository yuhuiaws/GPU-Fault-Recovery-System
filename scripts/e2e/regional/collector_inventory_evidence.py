"""Scoped inventory and collector-clock evidence for COLLECT-003/004."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from scripts.e2e.regional.collector_acceptance_fixture import (
    CollectorAcceptanceFixture,
    collector_setting,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError


INVENTORY_STATE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
names = {
    f"{resource}_inventory_{suffix}"
    for resource in ("gpu", "efa")
    for suffix in ("expected_count", "active_count")
}
signals = {}
for resource in ("gpu", "efa"):
    name = f"{resource}_inventory_mismatch"
    state = store.get_health_signal_state(f"{cluster_id}/{node_id}/{name}/node")
    signals[name] = state.model_dump(mode="json") if state is not None else None
print(json.dumps({
    "cluster_id": cluster_id,
    "node_id": node_id,
    "metrics": [
        item.model_dump(mode="json")
        for item in store.list_telemetry_metrics_latest(cluster_id, node_id)
        if item.name in names
    ],
    "signals": signals,
    "statuses": [
        item.model_dump(mode="json")
        for item in store.list_collector_statuses(cluster_id, node_id)
        if item.collector.value == "HOST_TELEMETRY"
    ],
    "captured_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True))
"""


def evidence_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def run_collect003(fixture: CollectorAcceptanceFixture) -> dict[str, Any]:
    snapshot = fixture.snapshot()
    env = snapshot["collector_env"]
    gpu_actual = len(snapshot["gpu_inventory"])
    efa = snapshot["efa_inventory"]
    expected_gpu = collector_setting(env, "GPU_FAULT_EXPECTED_GPU_COUNT")
    raw_efa = env.get("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT")
    try:
        if not isinstance(raw_efa, str):
            raise ValueError
        expected_efa = int(raw_efa)
        if expected_efa < 0:
            raise ValueError
    except ValueError:
        raise RegionalFixtureError(
            "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT is not a nonnegative integer"
        ) from None
    errors = []
    if gpu_actual != expected_gpu:
        errors.append("actual GPU count differs from collector configuration")
    if int(efa["active_count"]) != expected_efa:
        errors.append("active EFA count differs from collector configuration")
    runtime = fixture.execute("inventory-config")
    if (
        not (snapshot.get("collector_env_file") or {}).get("sha256")
        or runtime.get("file") != snapshot.get("collector_env_file")
        or not runtime.get("invocation_id")
        or set(runtime.get("file_env") or {})
        != {
            "GPU_FAULT_EXPECTED_GPU_COUNT",
            "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
            "GPU_FAULT_HOST_INTERVAL_SECONDS",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
            "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
        }
        or runtime.get("running_env") != runtime.get("file_env")
        or any(env.get(key) != value for key, value in runtime["file_env"].items())
    ):
        errors.append("running host collector configuration differs from its file")
    control_plane = fixture.regional.cpu_python(
        INVENTORY_STATE_PROBE, fixture.regional.settings.cluster_id, fixture.node
    )
    errors.extend(
        inventory_state_errors(
            control_plane,
            cluster_id=fixture.regional.settings.cluster_id,
            node=fixture.node,
            expected_gpu=expected_gpu,
            expected_efa=expected_efa,
            max_age_seconds=(
                collector_setting(env, "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS")
                + 2 * collector_setting(env, "GPU_FAULT_HOST_INTERVAL_SECONDS")
            ),
        )
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gpu_actual": gpu_actual,
        "gpu_expected": expected_gpu,
        "efa_inventory": efa,
        "efa_expected": expected_efa,
        "runtime_configuration": runtime,
        "control_plane_inventory": control_plane,
    }


def inventory_state_errors(
    proof: dict[str, Any],
    *,
    cluster_id: str,
    node: str,
    expected_gpu: int,
    expected_efa: int,
    max_age_seconds: int,
) -> list[str]:
    """No missing/stale telemetry can stand in for a healthy inventory signal."""

    captured = evidence_time(proof.get("captured_at"))
    if (
        captured is None
        or proof.get("cluster_id") != cluster_id
        or proof.get("node_id") != node
        or max_age_seconds <= 0
    ):
        return ["inventory control-plane evidence has no bound identity or clock"]

    def fresh(value: Any) -> bool:
        observed = evidence_time(value)
        return observed is not None and (
            -30 <= (captured - observed).total_seconds() <= max_age_seconds
        )

    errors = []
    statuses = proof.get("statuses")
    if (
        not isinstance(statuses, list)
        or len(statuses) != 1
        or not isinstance(statuses[0], dict)
        or statuses[0].get("cluster_id") != cluster_id
        or statuses[0].get("node_id") != node
        or statuses[0].get("collector") != "HOST_TELEMETRY"
        or not str(statuses[0].get("batch_id") or "").startswith(f"host-{node}-")
        or statuses[0].get("errors") != []
        or not fresh(statuses[0].get("ingested_at"))
        or not fresh(statuses[0].get("observed_at"))
    ):
        errors.append("fresh successful node host-collector delivery is unproven")
    metrics = proof.get("metrics")
    if not isinstance(metrics, list) or any(
        not isinstance(item, dict) for item in metrics
    ):
        return [*errors, "inventory latest-metric evidence is malformed"]
    signals = proof.get("signals")
    if not isinstance(signals, dict):
        return [*errors, "inventory signal-state evidence is missing"]
    for resource, expected in (("gpu", expected_gpu), ("efa", expected_efa)):
        name = f"{resource}_inventory_mismatch"
        state = signals.get(name)
        # The Store deliberately omits never-active signals. An explicit None
        # is acceptable only alongside fresh healthy producer metrics below.
        if name not in signals or (
            state is not None
            and (
                not isinstance(state, dict)
                or state.get("signal_key") != f"{cluster_id}/{node}/{name}/node"
                or state.get("active") is not False
                or not fresh(state.get("observed_at"))
                or not fresh(state.get("clock_at"))
            )
        ):
            errors.append(f"{name}: absence of a continuing finding is unproven")
        for suffix in ("expected_count", "active_count"):
            metric_name = f"{resource}_inventory_{suffix}"
            rows = [item for item in metrics if item.get("name") == metric_name]
            if (
                len(rows) != 1
                or rows[0].get("cluster_id") != cluster_id
                or rows[0].get("node_id") != node
                or rows[0].get("device") is not None
                or type(rows[0].get("value")) not in (int, float)
                or rows[0]["value"] != expected
                or not fresh(rows[0].get("observed_at"))
                or (rows[0].get("labels") or {}).get("expected_count") != str(expected)
                or (rows[0].get("labels") or {}).get("observed_count") != str(expected)
                or any(
                    (rows[0].get("labels") or {}).get(key) != "0"
                    for key in (
                        "consecutive_mismatch_samples",
                        "missing_count",
                        "excess_count",
                    )
                )
            ):
                errors.append(f"{metric_name}: current configured count is unproven")
    return errors


def debounce_sample_errors(
    records: list[dict[str, Any]],
    *,
    interval: int,
    required_samples: int,
    tolerance: float,
    expected_count: int | None = None,
) -> list[str]:
    """Judge first-mismatch -> threshold on the producer clock, not runner time."""

    if (
        type(interval) is not int
        or interval <= 0
        or type(required_samples) is not int
        or required_samples < 2
    ):
        return ["inventory debounce is switched off or its interval is invalid"]
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        return ["inventory debounce tolerance is invalid"]
    errors = []
    samples = []
    for record in records:
        for sample in record.get("samples") or []:
            if sample.get("name") != "gpu_inventory_mismatch":
                continue
            labels = sample.get("labels") or {}
            try:
                count = int(labels["consecutive_mismatch_samples"])
                required = int(labels["required_consecutive_samples"])
                expected = int(labels["expected_count"])
                observed = int(labels["observed_count"])
                value = float(sample["value"])
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append("inventory debounce sample labels/value are malformed")
                continue
            if expected_count is not None and expected != expected_count:
                continue
            if required != required_samples:
                errors.append(
                    "collector reports a different required_consecutive_samples"
                )
            if count < 1:
                continue
            if expected < 1 or observed != expected - 1:
                errors.append("inventory debounce sample does not match the override")
            observed_at = evidence_time(record.get("observed_at"))
            if observed_at is None or value not in (0.0, 1.0):
                errors.append("inventory debounce sample clock/value is invalid")
                continue
            samples.append((observed_at, count, value, record))
    samples.sort(key=lambda item: item[0])
    if not samples:
        return [
            *errors,
            "no bound gpu_inventory_mismatch samples reached the control plane",
        ]
    first = samples[0]
    if first[1] != 1 or first[2] != 0:
        errors.append("the first mismatching sample is missing or already fired")
    if "baseline" not in (first[3].get("edge_filter_reasons") or []):
        errors.append("first mismatch is not the restarted collector's baseline")
    threshold = next((item for item in samples if item[2] == 1), None)
    if threshold is None:
        return [*errors, "no gpu_inventory_mismatch finding reached the control plane"]
    if threshold[1] != required_samples:
        errors.append("finding did not fire on the configured threshold sample")
    earlier = [item for item in samples if item[0] <= threshold[0]]
    if any(
        second[1] <= previous[1] or second[0] <= previous[0]
        for previous, second in zip(earlier, earlier[1:])
    ):
        errors.append("collector mismatch sequence restarted or regressed")
    if any(item[2] != 0 for item in earlier[:-1]):
        errors.append("finding fired before the configured threshold")
    elapsed = (threshold[0] - first[0]).total_seconds()
    expected_elapsed = interval * (required_samples - 1)
    if not expected_elapsed <= elapsed <= expected_elapsed * (1 + tolerance):
        errors.append(
            f"collector first-to-threshold latency {elapsed:.1f}s is outside "
            f"{expected_elapsed}..{expected_elapsed * (1 + tolerance):.1f}s"
        )
    return errors
