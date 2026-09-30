from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import partial
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from gpu_fault.cluster_executor import ClusterExecutorError
from scripts.e2e.regional.capacity_acceptance_base import (
    CapError,
    CapHarnessBase,
    percentile,
    utc_now,
    write_json,
)
from scripts.e2e.regional.capacity_acceptance_traffic import cap001_monitor, cap001_send
from scripts.e2e.regional.capacity_acceptance_executor import (
    run_executor_proof,
)
from scripts.e2e.regional.capacity_wire import (
    CapacityThreadsRunning as Cap004ThreadsRunning,
)
from scripts.e2e.regional.probes import cap004_commands
from scripts.e2e.regional.capacity_acceptance_retry import run_claim_retry_proof
from scripts.e2e.regional.capacity_scrape_companion import ScrapeCompanion

# CAP-001: cluster A's per-cluster queue cap and the slack B's trickle adds.
CAP001_A_QUEUE_CAP = 20
CAP001_QUEUE_SLACK = 5
CAP001_BASELINE_REQUESTS = 15
CAP001_STORM_B_REQUESTS = 60
# The storm client keeps every connection alive for the whole minute and
# retires idle ones itself, before the probe's uvicorn does (its keep-alive is
# 5 s): a request racing the server's close is reset inside the Pod, and one
# pod-side stream error makes kubectl port-forward drop the whole tunnel.
CAP001_CLIENT_LIMITS = httpx.Limits(
    max_connections=100, max_keepalive_connections=100, keepalive_expiry=2.0
)
# CAP-002: the probe runs GPU_FAULT_STORE_IO_MAX_IN_FLIGHT=4 and the alert
# fires above 0.9, so every one of the four slots must be held; three left the
# ratio at 0.75 and the alert could never fire.
CAP002_STORE_IO_SLOTS = 4
CAP002_SATURATION_RATIO = 0.9
CAP002_ALERT_HOLD_SECONDS = 600
# The alert has `for: 5m`; once the hold is released the ratio drops to zero
# on the next scrape, but the probe must stay alive for that scrape to
# happen -- a deleted probe leaves a stale saturated sample until staleness
# (five minutes) -- so the resolve budget is seven minutes.
CAP002_RESOLVE_ATTEMPTS = 28
CAP002_RESOLVE_POLL_SECONDS = 15
# CAP-003: the knee is the first cluster count whose claim p95 crosses this
# or that stops answering 200.
CAP003_KNEE_P95_MS = 1000.0
CAP003_TARGET_CLUSTERS = 20
CAP003_HEADROOM = 2.0
CAP003_WAIT_FIELDS = (
    "requested_wait_seconds",
    "request_budget_seconds",
    "claim_wait_reserve_seconds",
    "server_max_wait_seconds",
    "effective_wait_seconds",
)


def cap003_wait_configuration(
    observed: Mapping[str, Any], requested: float
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in CAP003_WAIT_FIELDS:
        value = observed.get(key)
        if (
            value is None
            or type(value) not in {int, float}
            or not math.isfinite(value)
            or value < 0
        ):
            raise CapError("CAP003 effective wait configuration is missing or invalid")
        values[key] = float(value)
    budget = values["request_budget_seconds"]
    reserve = values["claim_wait_reserve_seconds"]
    maximum = values["server_max_wait_seconds"]
    effective = values["effective_wait_seconds"]
    expected = min(
        requested,
        maximum,
        max(0.0, budget - reserve) if budget > 0 else requested,
    )
    if (
        values["requested_wait_seconds"] != requested
        or maximum <= 0
        or effective <= 0
        or not math.isclose(effective, expected, rel_tol=0, abs_tol=1e-9)
    ):
        raise CapError("CAP003 effective wait does not match the deployed budget")
    return values


def cap001_failures(result: Mapping[str, Any]) -> list[str]:
    """Which CAP-001 pass conditions the recorded result does not meet."""

    failures: list[str] = []
    a_status = result["a_status_counts"]
    maxima = result["metric_maxima"]
    baseline_p95 = result["b_baseline_latency_ms"]["p95"]
    storm_p95 = result["b_latency_ms"]["p95"]
    factor = float(result["b_latency_factor"])
    if a_status.get(429, 0) <= 0:
        failures.append("cluster A was never rejected with 429")
    if result["a_retry_after_values"] != ["2"]:
        failures.append("cluster A Retry-After is not exactly 2")
    if result["b_baseline_status_counts"] != {202: CAP001_BASELINE_REQUESTS}:
        failures.append("cluster B baseline was not accepted end to end")
    if result["b_status_counts"] != {202: CAP001_STORM_B_REQUESTS}:
        failures.append("cluster B was not accepted end to end during the storm")
    if any(type(status) is not int or status not in {202, 429} for status in a_status):
        failures.append("cluster A saw an unexpected HTTP or transport status")
    if sum(a_status.values()) != 3000:
        failures.append("cluster A request accounting is incomplete")
    if baseline_p95 is None or storm_p95 is None:
        failures.append("cluster B latency percentiles are missing")
    elif (
        not all(math.isfinite(value) for value in (baseline_p95, storm_p95, factor))
        or baseline_p95 <= 0
        or storm_p95 < 0
        or factor < 1
    ):
        failures.append("cluster B latency evidence is invalid")
    elif storm_p95 > factor * baseline_p95:
        failures.append(
            f"cluster B storm p95 {storm_p95:.1f}ms exceeds {factor:g}x the "
            f"B-only baseline p95 {baseline_p95:.1f}ms"
        )
    if any(
        type(maxima.get(key)) not in {int, float}
        or not math.isfinite(maxima[key])
        or maxima[key] < 0
        for key in ("queue_depth", "a_rejections", "b_rejections")
    ):
        failures.append("capacity metric maxima are missing or invalid")
    if maxima.get("queue_depth", 0) > result["queue_depth_bound"]:
        failures.append(
            f"queue depth {maxima.get('queue_depth', 0):g} exceeded the bound "
            f"{result['queue_depth_bound']} (A cap + slack)"
        )
    if maxima.get("a_rejections", 0) <= 0:
        failures.append("no admission rejections were counted for cluster A")
    if maxima.get("b_rejections", 0) != 0:
        failures.append("cluster B was rejected")
    if result.get("metric_errors"):
        failures.append("capacity monitoring lost required metric samples")
    if result["queue_drained"] is not True:
        failures.append("the processor queue did not drain")
    return failures


def cap002_saturation_error(ratio: float) -> str | None:
    if math.isfinite(ratio) and CAP002_SATURATION_RATIO < ratio <= 1:
        return None
    return (
        f"CAP-002 alert hold did not saturate Store I/O: in_flight/max = "
        f"{ratio:.2f} <= {CAP002_SATURATION_RATIO}; the alert needs all "
        f"{CAP002_STORE_IO_SLOTS} slots held"
    )


def cap003_knee(results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The first tested cluster count that no longer holds up, or None."""

    for row in results:
        p95 = row["latency_ms"]["p95"]
        non_200 = {
            int(status): count
            for status, count in row["status_counts"].items()
            if int(status) != 200
        }
        if non_200:
            return {
                "cluster_count": row["cluster_count"],
                "reason": "non-200 claim responses",
                "status_counts": row["status_counts"],
                "p95_ms": p95,
            }
        if (
            p95 is None
            or not math.isfinite(p95)
            or p95 < 0
            or p95 >= CAP003_KNEE_P95_MS
        ):
            return {
                "cluster_count": row["cluster_count"],
                "reason": f"claim p95 >= {CAP003_KNEE_P95_MS:g}ms",
                "status_counts": row["status_counts"],
                "p95_ms": p95,
            }
    return None


def cap004_progression_errors(
    commands: Sequence[Mapping[str, Any]],
    progressions: Sequence[Mapping[str, Any]],
) -> list[str]:
    expected = [item.get("command_id") for item in commands]
    observed = [item.get("command_id") for item in progressions]
    if (
        not expected
        or any(not isinstance(item, str) or not item for item in [*expected, *observed])
        or len(set(expected)) != len(expected)
        or len(set(observed)) != len(observed)
        or set(expected) != set(observed)
    ):
        return ["lease progressions do not cover the exact unique command set"]
    errors = []
    for item in progressions:
        try:
            initial = datetime.fromisoformat(
                str(item["initial"]).replace("Z", "+00:00")
            )
            latest = datetime.fromisoformat(str(item["latest"]).replace("Z", "+00:00"))
            renewed = (
                initial.tzinfo is not None
                and latest.tzinfo is not None
                and latest > initial
                and type(item["renewal_count"]) is int
                and item["renewal_count"] > 0
            )
        except (KeyError, ValueError, TypeError):
            renewed = False
        if not renewed:
            errors.append(f"lease expiry did not advance for {item['command_id']}")
    return errors


def cap002_target_value(
    result: Any,
    *,
    not_before: float = 0,
    expected_labels: Mapping[str, str] | None = None,
) -> float | None:
    """Validate one AMP evaluation; callers must also prove source freshness."""
    if (
        not isinstance(result, list)
        or len(result) != 1
        or not isinstance(result[0], dict)
    ):
        return None
    if expected_labels is not None:
        labels = result[0].get("metric")
        if not isinstance(labels, dict) or any(
            labels.get(key) != value for key, value in expected_labels.items()
        ):
            return None
    value = result[0].get("value")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
    ):
        return None
    try:
        timestamp, sample = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if (
        math.isfinite(timestamp)
        and timestamp >= not_before
        and 0 <= time.time() - timestamp <= 90
        and math.isfinite(sample)
    ):
        return sample
    return None


def cap002_target_saturated(result: Any) -> bool:
    ratio = cap002_target_value(result)
    return ratio is not None and cap002_saturation_error(ratio) is None


def cap003_recommendations(
    results: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    *,
    target_clusters: int = CAP003_TARGET_CLUSTERS,
    headroom: float = CAP003_HEADROOM,
) -> dict[str, Any]:
    """Derive the operating recommendations from where the knee was measured.

    Each tested cluster polls once a second, so the largest cluster count
    below the knee is also the sustained claim rate in requests per second.
    The recommended poll interval keeps ``target_clusters`` executors under
    that rate with ``headroom`` to spare.
    """

    knee = cap003_knee(results)
    tested = [int(row["cluster_count"]) for row in results]
    if knee is None:
        sustained = max(tested, default=0)
    else:
        below = [count for count in tested if count < int(knee["cluster_count"])]
        sustained = max(below, default=0)
    poll_seconds = (
        math.ceil(headroom * target_clusters / sustained) if sustained else None
    )
    return {
        "knee": knee,
        "sustained_cluster_count_at_1rps": sustained,
        "sustained_claims_per_second": sustained,
        "target_clusters": target_clusters,
        "headroom": headroom,
        "executor_poll_seconds": poll_seconds,
        "per_cluster_count": {
            str(row["cluster_count"]): {
                "p95_ms": row["latency_ms"]["p95"],
                "p99_ms": row["latency_ms"]["p99"],
                "status_counts": row["status_counts"],
                "api_average_cpu_cores": row["api_average_cpu_cores"],
                "store_io_wait_seconds_max": row["store_io_wait_seconds_max"],
            }
            for row in results
        },
        "postgres_pool_change": (
            "none"
            if budget["budget_ratio"] < 0.8
            else "reduce per-process pools or raise max_connections"
        ),
    }


def cap002_claim(
    harness: CapHarnessBase, probe: Any, index: int, phase: str
) -> dict[str, Any]:
    """One raw claim from synthetic cluster ``index``, timed, pins included."""

    begin = time.perf_counter()
    with httpx.Client(base_url=probe.url, timeout=30) as client:
        response = client.post(
            "/v1/regional/executors/claim",
            headers=harness.cluster_headers(
                f"cap-cluster-{index:03d}", harness.tokens[index]
            ),
            json={
                "executor_id": f"cap002-{phase}-{index:03d}",
                **harness.claim_identity(),
                "execution_owners": ["gpu-fault-kubernetes-adapter"],
                "max_commands": 1,
                "lease_seconds": 10,
            },
        )
    return {
        "status": response.status_code,
        "retry_after": response.headers.get("Retry-After"),
        "latency_seconds": time.perf_counter() - begin,
    }


class CapacityAcceptanceCases(CapHarnessBase):
    def case_001(self) -> dict[str, Any]:
        probe = self.deploy_probe(
            "CAP001",
            {
                "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": str(CAP001_A_QUEUE_CAP),
                "GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH": "0",
                "GPU_FAULT_PROCESSOR_WORKERS": "1",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "4",
            },
        )
        case_dir = self.run_dir / "CAP-001"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        monitor_stop = threading.Event()
        maxima: dict[str, float] = {}
        metrics_samples: list[dict[str, Any]] = []
        metric_errors: list[str] = []

        monitor_thread = threading.Thread(
            target=cap001_monitor,
            args=(self, probe, monitor_stop, maxima, metrics_samples, metric_errors),
            daemon=True,
        )
        client = httpx.Client(
            base_url=probe.url, timeout=20, limits=CAP001_CLIENT_LIMITS
        )
        send = partial(cap001_send, self, client)

        drained = False
        try:
            monitor_thread.start()
            # B alone, one request a second, before anything else loads the
            # probe: the reference the storm-phase B latency is judged against.
            baseline_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                baseline_responses = list(
                    pool.map(
                        lambda index: send(
                            "baseline", 1, index, baseline_start + index
                        ),
                        range(CAP001_BASELINE_REQUESTS),
                    )
                )
            started = time.perf_counter()
            wall_start = time.perf_counter()
            with (
                ThreadPoolExecutor(max_workers=80) as a_pool,
                ThreadPoolExecutor(max_workers=4) as b_pool,
            ):
                futures = [
                    a_pool.submit(send, "storm", 0, index, wall_start + index / 50.0)
                    for index in range(3000)
                ]
                futures.extend(
                    b_pool.submit(send, "storm", 1, index, wall_start + index / 1.0)
                    for index in range(CAP001_STORM_B_REQUESTS)
                )
                responses = [future.result() for future in as_completed(futures)]
            elapsed = time.perf_counter() - started

            drain_deadline = time.monotonic() + 120
            while time.monotonic() < drain_deadline:
                values = self.metrics(probe.url)
                if self.metric_value(values, "gpu_fault_processor_queue_depth") == 0:
                    time.sleep(1)
                    values = self.metrics(probe.url)
                    if (
                        self.metric_value(values, "gpu_fault_processor_queue_depth")
                        == 0
                    ):
                        drained = True
                        break
                time.sleep(1)
        finally:
            monitor_stop.set()
            monitor_thread.join()
            client.close()
        write_json(case_dir / "metrics-samples.json", metrics_samples)
        # The Pod's restart count, container states, events and log tail while
        # the probe still exists; cleanup below deletes the Deployment.
        probe_state = self.probe_pod_report(probe)
        write_json(case_dir / "probe-state.json", probe_state)

        by_cluster = {
            cluster: [item for item in responses if item["cluster"] == cluster]
            for cluster in (0, 1)
        }
        baseline_latencies = [item["latency_ms"] for item in baseline_responses]
        storm_b_latencies = [item["latency_ms"] for item in by_cluster[1]]
        baseline_p95 = percentile(baseline_latencies, 0.95)
        storm_p95 = percentile(storm_b_latencies, 0.95)
        result: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 3),
            "a_status_counts": dict(
                sorted(
                    Counter(item["status"] for item in by_cluster[0]).items(),
                    key=lambda item: str(item[0]),
                )
            ),
            "b_status_counts": dict(
                sorted(
                    Counter(item["status"] for item in by_cluster[1]).items(),
                    key=lambda item: str(item[0]),
                )
            ),
            "b_baseline_status_counts": dict(
                sorted(
                    Counter(item["status"] for item in baseline_responses).items(),
                    key=lambda item: str(item[0]),
                )
            ),
            "a_retry_after_values": sorted(
                {item["retry_after"] for item in by_cluster[0] if item["status"] == 429}
            ),
            "b_baseline_latency_ms": {
                "p50": percentile(baseline_latencies, 0.50),
                "p95": baseline_p95,
                "p99": percentile(baseline_latencies, 0.99),
            },
            "b_latency_ms": {
                "p50": percentile(storm_b_latencies, 0.50),
                "p95": storm_p95,
                "p99": percentile(storm_b_latencies, 0.99),
            },
            "b_latency_factor": self.b_latency_factor,
            "b_p95_degradation_ratio": (
                storm_p95 / baseline_p95
                if storm_p95 is not None and baseline_p95
                else None
            ),
            "queue_depth_bound": CAP001_A_QUEUE_CAP + CAP001_QUEUE_SLACK,
            "metric_maxima": maxima,
            "metric_errors": metric_errors,
            "queue_drained": drained,
            "transport_retries": sum(
                int(item.get("transport_retries", 0)) for item in responses
            ),
            "probe_restart_count": (probe_state.get("pod") or {}).get("restart_count"),
            "port_forward_incidents": len(self.transport_incidents),
        }
        failures = cap001_failures(result)
        result["failures"] = failures
        result["status"] = "PASS" if not failures else "FAIL"
        write_json(case_dir / "summary.json", result)
        cleanup = self.cleanup_probe(probe)
        write_json(case_dir / "cleanup.json", cleanup)
        if failures:
            raise CapError("GF-REGIONAL-CAP-001 failed: " + "; ".join(failures))
        return result

    def _cap002_behavior(self, probe: Any, case_dir: Any) -> dict[str, Any]:
        behavior_hold = self.probe_control(
            probe,
            "/__cap__/hold",
            {"tag": "behavior", "durations": [5, 5, 5, 5]},
        )
        with ThreadPoolExecutor(max_workers=20) as pool:
            initial_results = list(
                pool.map(
                    lambda index: cap002_claim(self, probe, index, "behavior"),
                    range(20),
                )
            )

        def release_behavior_hold() -> None:
            self.probe_control(probe, "/__cap__/release", {"tag": "behavior"})

        try:
            executor_retry = run_claim_retry_proof(
                probe.url,
                self.tokens[0],
                case_dir,
                release_behavior_hold,
                executor_pins=self.executor_pins,
            )
        finally:
            release_behavior_hold()
        metrics = self.metrics(probe.url)
        status_counts = dict(
            sorted(Counter(item["status"] for item in initial_results).items())
        )
        retry_status_counts = dict(
            sorted(
                Counter(
                    item["status"]
                    for item in executor_retry["attempts"]
                    if item["status"] == 200
                ).items()
            )
        )
        retry_after_values = sorted(
            {item["retry_after"] for item in initial_results if item["status"] == 503}
        )
        behavior: dict[str, Any] = {
            "hold": behavior_hold,
            "initial_status_counts": status_counts,
            "retry_after_values": retry_after_values,
            "retry_status_counts": retry_status_counts,
            "executor_retry": executor_retry,
            "store_io_rejections": self.metric_value(
                metrics, "gpu_fault_store_io_rejections_total", reason="capacity"
            ),
        }
        behavior["passed"] = all(
            (
                status_counts.get(503, 0) > 0,
                set(status_counts) <= {200, 503},
                retry_after_values == ["2"],
                retry_status_counts == {200: 1},
                executor_retry["passed"] is True,
                behavior["store_io_rejections"] > 0,
            )
        )
        write_json(case_dir / "behavior.json", behavior)
        if not behavior["passed"]:
            raise CapError("CAP-002 503 behavior phase failed")
        return behavior

    def _cap002_remaining(self) -> float:
        deadline = self.maintenance_deadline
        if not isinstance(deadline, datetime) or deadline.utcoffset() is None:
            raise CapError("CAP-002 requires an aware maintenance deadline")
        remaining = deadline.timestamp() - time.time()
        if not math.isfinite(remaining) or remaining <= 0:
            raise CapError("CAP-002 maintenance deadline has expired")
        return remaining

    def _cap002_poll_sleep(self) -> None:
        time.sleep(min(CAP002_RESOLVE_POLL_SECONDS, self._cap002_remaining()))
        self._cap002_remaining()

    def _cap002_query(
        self,
        probe: Any,
        query: str,
        *,
        not_before: float,
        query_time: float | None = None,
    ) -> float | None:
        remaining = self._cap002_remaining()
        params = {"query": query, "timeout": f"{min(30, remaining):g}s"}
        if query_time is not None:
            params["time"] = str(query_time)
        response = self.amp_request(
            "POST",
            "/api/v1/query",
            params,
        )
        self._cap002_remaining()
        if not isinstance(response, dict) or response.get("status") != "success":
            return None
        data = response.get("data")
        if not isinstance(data, dict) or data.get("resultType") != "vector":
            return None
        return cap002_target_value(
            data.get("result"),
            not_before=not_before,
            expected_labels={"pod": probe.pod, "capacity_run": self.run_id},
        )

    def _cap002_wait_scrape_ready(
        self, probe: Any, case_dir: Path, *, selector: str, not_before: float
    ) -> bool:
        polls = []
        for attempt in range(1, CAP002_RESOLVE_ATTEMPTS + 1):
            query_time = time.time()
            up = self._cap002_query(
                probe,
                f"up{{{selector}}}",
                not_before=not_before,
                query_time=query_time,
            )
            scraped_at = (
                self._cap002_query(
                    probe,
                    f"timestamp(up{{{selector}}})",
                    not_before=not_before,
                    query_time=query_time,
                )
                if up == 1
                else None
            )
            ready = (
                up == 1
                and scraped_at is not None
                and scraped_at >= not_before
                and 0 <= time.time() - scraped_at <= 90
            )
            polls.append(
                {
                    "attempt": attempt,
                    "observed_at": utc_now(),
                    "target_up": up,
                    "scraped_at": scraped_at,
                    "ready": ready,
                }
            )
            write_json(case_dir / "scrape-ready-poll.json", polls)
            if ready:
                return True
            if attempt < CAP002_RESOLVE_ATTEMPTS:
                self._cap002_poll_sleep()
        return False

    def _cap002_target_ratio(
        self, probe: Any, *, selector: str, not_before: float
    ) -> float | None:
        in_flight = f"gpu_fault_store_io_in_flight{{{selector}}}"
        maximum = f"gpu_fault_store_io_max_in_flight{{{selector}}}"
        query = f"{in_flight} / {maximum}"
        # Arithmetic samples carry evaluation time, not source scrape time.
        # Filter both inputs before accepting a fresh saturation or zero proof.
        for source in (in_flight, maximum):
            query += (
                f" and (timestamp({source}) >= {not_before})"
                f" and (time() - timestamp({source}) >= 0)"
                f" and (time() - timestamp({source}) <= 90)"
            )
        return self._cap002_query(probe, query, not_before=not_before)

    def cap002_alert(
        self,
        probe: Any,
        case_dir: Any,
        behavior: dict[str, Any],
        *,
        selector: str,
    ) -> tuple[dict[str, Any], bool]:
        if self._cap002_remaining() < CAP002_ALERT_HOLD_SECONDS:
            raise CapError(
                "CAP-002 maintenance time is insufficient for the alert hold"
            )
        hold_started = time.time()
        alert_hold = self.probe_control(
            probe,
            "/__cap__/hold",
            {
                "tag": "alert",
                "durations": [CAP002_ALERT_HOLD_SECONDS] * CAP002_STORE_IO_SLOTS,
            },
        )
        values = self.metrics(probe.url)
        in_flight = self.metric_value(values, "gpu_fault_store_io_in_flight")
        maximum = self.metric_value(values, "gpu_fault_store_io_max_in_flight")
        ratio = in_flight / maximum if maximum else 0
        sample = {
            "observed_at": utc_now(),
            "hold": alert_hold,
            "in_flight": in_flight,
            "max_in_flight": maximum,
            "ratio": ratio,
            "rejections": self.metric_value(
                values, "gpu_fault_store_io_rejections_total", reason="capacity"
            ),
        }
        write_json(case_dir / "saturation-sample.json", sample)
        saturation_error = cap002_saturation_error(ratio)
        if saturation_error is not None:
            raise CapError(saturation_error)
        alert_poll = []
        fired = False
        started = time.monotonic()
        for attempt in range(1, CAP002_RESOLVE_ATTEMPTS + 1):
            self._cap002_poll_sleep()
            target_ratio = self._cap002_target_ratio(
                probe, selector=selector, not_before=hold_started
            )
            target_saturated = (
                target_ratio is not None
                and cap002_saturation_error(target_ratio) is None
            )
            states = self.alert_states("GpuFaultStoreIoSaturated")
            self._cap002_remaining()
            alert_poll.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "target_ratio_present": target_ratio is not None,
                    "target_saturated": target_saturated,
                    "states": states,
                }
            )
            if target_saturated and "firing" in states:
                fired = True
                break
        write_json(case_dir / "alert-poll.json", alert_poll)
        release = self.probe_control(probe, "/__cap__/release", {"tag": "alert"})
        result: dict[str, Any] = {
            "initial_status_counts": behavior["initial_status_counts"],
            "initial_retry_after_values": behavior["retry_after_values"],
            "retry_status_counts": behavior["retry_status_counts"],
            "max_store_io_ratio": ratio,
            "store_io_rejections": behavior["store_io_rejections"],
            "alert_hold_release": release,
            "alert_fired": fired,
            "alert_states": self.alert_states("GpuFaultStoreIoSaturated"),
        }
        passed = bool(behavior["passed"] and ratio > CAP002_SATURATION_RATIO and fired)
        result["status"] = "PASS" if passed else "FAIL"
        write_json(case_dir / "summary.json", {**result, "status": "PENDING"})
        return result, passed

    def _cap002_wait_resolved(
        self,
        probe: Any,
        case_dir: Path,
        *,
        selector: str,
        not_before: float,
        attempts: int = CAP002_RESOLVE_ATTEMPTS,
    ) -> bool:
        """Wait for the alert to clear *while the probe is still scraped*.

        The hold has been released, so the next scrape samples a ratio of
        zero and the alert resolves. The probe must not be deleted before
        that: a vanished target leaves its last, saturated sample in place
        until staleness marks it, which takes five minutes on its own.
        """

        resolved_poll = []
        for attempt in range(1, attempts + 1):
            ratio = self._cap002_target_ratio(
                probe, selector=selector, not_before=not_before
            )
            states = self.alert_states("GpuFaultStoreIoSaturated")
            self._cap002_remaining()
            resolved_poll.append(
                {
                    "attempt": attempt,
                    "observed_at": utc_now(),
                    "target_ratio": ratio,
                    "target_zero": ratio == 0,
                    "states": states,
                }
            )
            write_json(case_dir / "alert-resolve-poll.json", resolved_poll)
            if ratio == 0 and states == []:
                return True
            if attempt < attempts:
                self._cap002_poll_sleep()
        return False

    def case_002_v2(self) -> dict[str, Any]:
        if self.cap002_scrape_stopped is not True:
            raise CapError("CAP-002 previous scrape shutdown is unverified")
        if not self.scrape_source_binding:
            raise CapError("CAP-002 requires the plan-bound scrape source")
        self._cap002_remaining()
        started_at = time.time()
        if self.alert_states("GpuFaultStoreIoSaturated"):
            raise CapError("CAP-002 target alert is already active before the probe")
        self._cap002_remaining()
        probe = self.deploy_probe(
            "CAP002",
            {
                "GPU_FAULT_SERVICE_ROLE": "ingress",
                "GPU_FAULT_STORE_IO_WORKERS": str(CAP002_STORE_IO_SLOTS),
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": str(CAP002_STORE_IO_SLOTS),
                "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS": "1",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
            },
        )
        case_dir = self.run_dir / "CAP-002"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        result: dict[str, Any] = {"status": "FAIL", "scrape_ready": False}
        passed = False
        resolved = False
        companion: ScrapeCompanion | None = None
        scrape_started = False
        error: BaseException | None = None
        cleanup_errors: list[str] = []

        def cleanup_failed(stage: str, exc: BaseException) -> None:
            nonlocal error
            cleanup_errors.append(f"{stage}: {type(exc).__name__}")
            if error is None and not isinstance(exc, Exception):
                error = exc

        try:
            write_json(case_dir / "summary.json", result)
            deadline = self.maintenance_deadline
            if deadline is None:
                raise CapError("CAP-002 requires an aware maintenance deadline")
            companion = ScrapeCompanion(
                self,
                probe,
                expected_source=dict(self.scrape_source_binding),
                deadline=deadline,
            )
            self.cap002_scrape_stopped = False
            start = companion.start()
            scrape_started = True
            if not isinstance(start, dict):
                raise CapError("CAP-002 scrape start evidence is invalid")
            write_json(case_dir / "scrape-start.json", start)
            result["scrape_ready"] = self._cap002_wait_scrape_ready(
                probe,
                case_dir,
                selector=companion.selector,
                not_before=started_at,
            )
            if result["scrape_ready"] is not True:
                raise CapError("CAP-002 fresh exact AMP scrape readiness is unproven")
            self._cap002_remaining()
            behavior = self._cap002_behavior(probe, case_dir)
            alert_result, passed = self.cap002_alert(
                probe, case_dir, behavior, selector=companion.selector
            )
            result.update(alert_result)
        except BaseException as exc:
            error = exc
            result["error_type"] = type(exc).__name__
        finally:
            if scrape_started and companion is not None:
                for tag in ("behavior", "alert"):
                    try:
                        self.probe_control(probe, "/__cap__/release", {"tag": tag})
                    except BaseException as exc:
                        cleanup_failed(f"release {tag}", exc)
                try:
                    resolved = self._cap002_wait_resolved(
                        probe,
                        case_dir,
                        selector=companion.selector,
                        not_before=time.time(),
                    )
                except BaseException as exc:
                    cleanup_failed("alert resolution", exc)
            if companion is not None:
                try:
                    stop = companion.stop()
                    if (
                        not isinstance(stop, dict)
                        or stop.get("cleanup_complete") is not True
                        or stop.get("process_termination_proven") is not True
                    ):
                        raise CapError("CAP-002 scrape stop evidence is invalid")
                    self.cap002_scrape_stopped = True
                    write_json(case_dir / "scrape-stop.json", stop)
                except BaseException as exc:
                    cleanup_failed("scrape companion stop", exc)
            cleanup: dict[str, Any] = {"passed": False}
            if self.cap002_scrape_stopped:
                try:
                    cleanup = self.cleanup_probe(probe)
                except BaseException as exc:
                    cleanup_failed("probe cleanup", exc)
            else:
                cleanup["resources_retained"] = True
            try:
                write_json(
                    case_dir / "cleanup.json", {**cleanup, "errors": cleanup_errors}
                )
            except BaseException as exc:
                cleanup_failed("cleanup evidence", exc)
        passed = passed and resolved and not cleanup_errors and error is None
        result["status"] = "PASS" if passed else "FAIL"
        result["cleanup_errors"] = cleanup_errors
        result["alert_resolved_after_release"] = resolved
        result["scrape_stopped"] = self.cap002_scrape_stopped
        try:
            write_json(case_dir / "summary.json", result)
        except BaseException as exc:
            if error is None:
                raise
            error.add_note(f"CAP-002 summary write failed: {type(exc).__name__}")
        if error is not None:
            for cleanup_error in cleanup_errors:
                error.add_note(cleanup_error)
            raise error
        if not passed:
            raise CapError("GF-REGIONAL-CAP-002 failed")
        return result

    def case_003(self) -> dict[str, Any]:
        probe = self.deploy_probe(
            "CAP003",
            {
                "GPU_FAULT_PROCESSOR_WORKERS": "1",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": "64",
            },
        )
        case_dir = self.run_dir / "CAP-003"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        try:
            with httpx.Client(base_url=probe.url, timeout=20) as client:
                results = self.cap003_samples(probe, client, case_dir)
            long_poll = self.cap003_long_poll_samples(probe, case_dir)
            ended_at = datetime.now(timezone.utc) + timedelta(minutes=1)
            budget = self.connection_budget()
            cloudwatch = self.cloudwatch_window(started_at, ended_at)
            write_json(case_dir / "connection-budget.json", budget)
            write_json(case_dir / "aurora.json", cloudwatch)
            knee = cap003_knee(results)
            passed = knee is None and all(row["passed"] for row in long_poll)
            result: dict[str, Any] = {
                "status": "PASS" if passed else "FAIL",
                "scenarios": results,
                "production_long_poll": long_poll,
                "connection_budget": budget,
                "knee": knee,
                "recommendations": cap003_recommendations(results, budget),
            }
            write_json(case_dir / "summary.json", result)
        finally:
            cleanup = self.cleanup_probe(probe)
            write_json(case_dir / "cleanup.json", cleanup)
        if not passed:
            raise CapError("GF-REGIONAL-CAP-003 failed")
        return result

    def cap003_long_poll_samples(
        self, probe: Any, case_dir: Path
    ) -> list[dict[str, Any]]:
        wait_seconds = 20
        replicas = 2
        results = []
        # A first long poll starts the production hub's lazy LISTEN connection.
        warmup = self.executor_client(probe.url, 0)
        if warmup.claim(
            "cap003-warmup",
            max_commands=1,
            lease_seconds=10,
            execution_owners=["gpu-fault-kubernetes-adapter"],
            wait_seconds=wait_seconds,
        ):
            raise CapError(
                "CAP003 warmup found commands in the isolated empty database"
            )
        for cluster_count in (1, 5, 10, 20):
            listener_before = self.probe_control(
                probe,
                "/__cap__/claim-wakeups",
                {"tag": "cap003", "wait_seconds": wait_seconds},
            )
            wait_configuration = cap003_wait_configuration(
                listener_before, wait_seconds
            )
            effective_wait = wait_configuration["effective_wait_seconds"]
            cpu_before = self.pod_cpu_usage_usec(probe.pod)
            connections_before = self.isolated_db_connections(probe.pod)
            started = time.perf_counter()

            def poll(index: int) -> list[dict[str, Any]]:
                cluster_index = index // replicas
                client = self.executor_client(probe.url, cluster_index)
                samples = []
                for _ in range(3):
                    begin = time.perf_counter()
                    status = 200
                    try:
                        commands = client.claim(
                            f"cap003-long-{cluster_count}-{index}",
                            max_commands=1,
                            lease_seconds=10,
                            wait_seconds=wait_seconds,
                            execution_owners=["gpu-fault-kubernetes-adapter"],
                        )
                        if commands:
                            raise CapError(
                                "CAP003 long poll returned unexpected commands"
                            )
                    except ClusterExecutorError as exc:
                        status = exc.status_code or 0
                    elapsed = time.perf_counter() - begin
                    samples.append(
                        {
                            "status": status,
                            "elapsed_seconds": elapsed,
                            "overhead_ms": max(0, elapsed - effective_wait) * 1000,
                        }
                    )
                return samples

            with ThreadPoolExecutor(max_workers=cluster_count * replicas) as pool:
                samples = [
                    sample
                    for rows in pool.map(poll, range(cluster_count * replicas))
                    for sample in rows
                ]
            elapsed = time.perf_counter() - started
            listener_after = self.probe_control(
                probe,
                "/__cap__/claim-wakeups",
                {"tag": "cap003", "wait_seconds": wait_seconds},
            )
            final_wait_configuration = cap003_wait_configuration(
                listener_after, wait_seconds
            )
            cpu_after = self.pod_cpu_usage_usec(probe.pod)
            connections_after = self.isolated_db_connections(probe.pod)
            overhead = {
                key: percentile([item["overhead_ms"] for item in samples], quantile)
                for key, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            }
            passed = (
                all(
                    row["status"] == 200
                    and row["elapsed_seconds"] >= max(0, effective_wait - 1)
                    for row in samples
                )
                and final_wait_configuration == wait_configuration
                and overhead["p95"] is not None
                and overhead["p95"] < CAP003_KNEE_P95_MS
                and all(
                    item.get("listener_connected") is True
                    and item.get("listener_alive") is True
                    for item in (listener_before, listener_after)
                )
                and listener_after.get("disconnects_total")
                == listener_before.get("disconnects_total")
                and elapsed > 0
                and cpu_after >= cpu_before
                and min(connections_before, connections_after) > 0
            )
            row = {
                "cluster_count": cluster_count,
                "executor_replicas_per_cluster": replicas,
                "wait_seconds": wait_seconds,
                "effective_wait_seconds": effective_wait,
                "wait_configuration": wait_configuration,
                "requests": len(samples),
                "passed": passed,
                "status_counts": dict(Counter(item["status"] for item in samples)),
                "hold_overhead_ms": overhead,
                "samples": samples,
                "listener_before": listener_before,
                "listener_after": listener_after,
                "database_connections_before": connections_before,
                "database_connections_after": connections_after,
                "api_average_cpu_cores": (cpu_after - cpu_before) / 1_000_000 / elapsed,
            }
            results.append(row)
            write_json(case_dir / f"long-poll-n-{cluster_count}.json", row)
        return results

    def cap003_samples(
        self, probe: Any, client: httpx.Client, case_dir: Path
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for cluster_count in (1, 5, 10, 20):
            before_cpu = self.pod_cpu_usage_usec(probe.pod)
            before_connections = self.isolated_db_connections(probe.pod)
            wall_started = time.perf_counter()

            def poll(cluster_index: int) -> list[dict[str, Any]]:
                rows = []
                for sequence in range(60):
                    scheduled = wall_started + sequence
                    delay = scheduled - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    begin = time.perf_counter()
                    response = client.post(
                        "/v1/regional/executors/claim",
                        headers=self.cluster_headers(
                            f"cap-cluster-{cluster_index:03d}",
                            self.tokens[cluster_index],
                        ),
                        json={
                            "executor_id": (
                                f"cap003-{cluster_count}-{cluster_index:03d}"
                            ),
                            **self.claim_identity(),
                            "execution_owners": ["gpu-fault-kubernetes-adapter"],
                            "max_commands": 1,
                            "lease_seconds": 10,
                        },
                    )
                    if (
                        response.status_code == 200
                        and response.json().get("commands") != []
                    ):
                        raise CapError(
                            "CAP-003 empty claim returned missing or nonempty commands"
                        )
                    rows.append(
                        {
                            "status": response.status_code,
                            "latency_ms": (time.perf_counter() - begin) * 1000,
                        }
                    )
                return rows

            with ThreadPoolExecutor(max_workers=cluster_count) as pool:
                documents = list(pool.map(poll, range(cluster_count)))
            wall = time.perf_counter() - wall_started
            after_cpu = self.pod_cpu_usage_usec(probe.pod)
            after_connections = self.isolated_db_connections(probe.pod)
            if (
                wall <= 0
                or after_cpu < before_cpu
                or min(before_connections, after_connections) < 1
            ):
                raise CapError("CAP-003 CPU/connection measurements are invalid")
            flat = [item for document in documents for item in document]
            metrics = self.metrics(probe.url)
            row: dict[str, Any] = {
                "cluster_count": cluster_count,
                "requests": len(flat),
                "status_counts": dict(
                    sorted(Counter(item["status"] for item in flat).items())
                ),
                "latency_ms": {
                    "p50": percentile([item["latency_ms"] for item in flat], 0.50),
                    "p95": percentile([item["latency_ms"] for item in flat], 0.95),
                    "p99": percentile([item["latency_ms"] for item in flat], 0.99),
                },
                "wall_seconds": wall,
                "api_average_cpu_cores": ((after_cpu - before_cpu) / 1_000_000 / wall),
                "database_connections_before": before_connections,
                "database_connections_after": after_connections,
                "store_io_wait_seconds_max": self.metric_value(
                    metrics,
                    "gpu_fault_store_io_admission_wait_seconds_max",
                ),
            }
            results.append(row)
            write_json(case_dir / f"n-{cluster_count}.json", row)
        return results

    def cap004_store(self, probe: Any, mode: str) -> dict[str, Any]:
        import json

        response = self.kubectl(
            "exec",
            "-i",
            probe.pod,
            "--",
            "/opt/gpu-fault/control-plane/bin/python",
            "-",
            mode,
            self.run_id,
            probe.database,
            input_text=Path(cap004_commands.__file__).read_text(encoding="utf-8"),
        )
        result = json.loads(response.stdout)
        if not isinstance(result, dict) or result.get("run_id") != self.run_id:
            raise CapError("CAP004 Store probe returned an invalid identity")
        return result

    def case_004(self) -> dict[str, Any]:
        case_dir = self.run_dir / "CAP-004"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        result: dict[str, Any] = {
            "status": "FAIL",
            "production_executor_verified": False,
        }
        write_json(case_dir / "summary.json", result)
        probe = self.deploy_probe(
            "CAP004",
            {
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": "64",
            },
        )
        error: BaseException | None = None
        cleanup_errors: list[str] = []
        self.cap004_executor_stopped = True
        try:
            seed = self.cap004_store(probe, "seed")
            write_json(case_dir / "seed.json", seed)
            self.cap004_executor_stopped = False
            try:
                result.update(
                    run_executor_proof(
                        probe.url,
                        self.tokens[0],
                        self.run_id,
                        case_dir,
                        executor_pins=self.executor_pins,
                    )
                )
            except Cap004ThreadsRunning:
                raise
            except BaseException:
                self.cap004_executor_stopped = True
                raise
            else:
                self.cap004_executor_stopped = True
            final = self.cap004_store(probe, "inspect")
            write_json(case_dir / "terminal-commands.json", final)
            progression_errors = cap004_progression_errors(
                seed["commands"], result["lease_progressions"]
            )
            terminal_errors = cap004_commands.terminal_errors(final, self.run_id)
            result["lease_progression_errors"] = progression_errors
            result["terminal_errors"] = terminal_errors
            result["production_executor_verified"] = not (
                result["problems"] or progression_errors or terminal_errors
            )
        except BaseException as exc:
            error = exc
            result["error_type"] = type(exc).__name__
        finally:
            if self.cap004_executor_stopped:
                try:
                    closed = self.cap004_store(probe, "cleanup")
                    if any(
                        row["status"] not in {"SUCCEEDED", "FAILED"}
                        or row["lease_present"] is not False
                        for row in closed["commands"]
                    ):
                        raise CapError("CAP004 cleanup left nonterminal commands")
                    write_json(case_dir / "command-cleanup.json", closed)
                except Exception as exc:
                    cleanup_errors.append(f"command cleanup: {type(exc).__name__}")
                try:
                    write_json(case_dir / "cleanup.json", self.cleanup_probe(probe))
                except Exception as exc:
                    cleanup_errors.append(f"probe cleanup: {type(exc).__name__}")
            else:
                cleanup_errors.append(
                    "Executor threads still running; cleanup withheld"
                )
            passed = (
                error is None
                and not cleanup_errors
                and result["production_executor_verified"] is True
            )
            result["status"] = "PASS" if passed else "FAIL"
            result["cleanup_errors"] = cleanup_errors
            result["limitations"] = [
                "Executor scheduling and lease handling use a nonphysical ledger; "
                "no Node Agent, Kubernetes or provider action is exercised."
            ]
            write_json(case_dir / "summary.json", result)
        if error is not None:
            raise error
        if not passed:
            problems = result.get("problems") or [
                "lease progression, terminal state or cleanup proof failed"
            ]
            raise CapError("GF-REGIONAL-CAP-004 failed: " + "; ".join(problems))
        return result
