from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import secrets
import threading
import time
from typing import Any, Mapping

import httpx

from scripts.e2e.regional.capacity_acceptance_base import (
    CapError,
    CapHarnessBase,
    percentile,
    utc_now,
    write_json,
)


class CapacityAcceptanceCases(CapHarnessBase):
    def case_001(self) -> dict[str, Any]:
        probe = self.deploy_probe(
            "CAP001",
            {
                "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH": "20",
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

        def monitor() -> None:
            while not monitor_stop.wait(1):
                try:
                    values = self.metrics(probe.url)
                except Exception:
                    continue
                sample = {
                    "observed_at": utc_now(),
                    "queue_depth": self.metric_value(
                        values, "gpu_fault_processor_queue_depth"
                    ),
                    "a_depth": self.metric_value(
                        values,
                        "gpu_fault_processor_cluster_queue_depth",
                        cluster_id="cap-cluster-000",
                    ),
                    "b_depth": self.metric_value(
                        values,
                        "gpu_fault_processor_cluster_queue_depth",
                        cluster_id="cap-cluster-001",
                    ),
                    "a_rejections": sum(
                        value
                        for name, labels, value in values
                        if name
                        == "gpu_fault_processor_admission_rejections_by_cluster_total"
                        and labels.get("cluster_id") == "cap-cluster-000"
                    ),
                    "b_rejections": sum(
                        value
                        for name, labels, value in values
                        if name
                        == "gpu_fault_processor_admission_rejections_by_cluster_total"
                        and labels.get("cluster_id") == "cap-cluster-001"
                    ),
                }
                metrics_samples.append(sample)
                for key, value in sample.items():
                    if isinstance(value, (int, float)):
                        maxima[key] = max(maxima.get(key, 0.0), float(value))

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        started = time.perf_counter()
        client = httpx.Client(base_url=probe.url, timeout=20)

        def send(cluster_index: int, sequence: int, scheduled: float) -> dict[str, Any]:
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            observed = datetime.now(timezone.utc)
            begin = time.perf_counter()
            payload = {
                "batch_id": (f"cap001-c{cluster_index:03d}-{sequence:05d}"),
                "cluster_id": f"cap-cluster-{cluster_index:03d}",
                "node_id": f"node-c{cluster_index:03d}-{sequence:05d}",
                "observed_at": observed.isoformat(),
                "samples": [
                    {
                        "name": "network_link_up",
                        "value": 1,
                        "device": "eth0",
                    }
                ],
                "runtime_profile_version": "hyperpod-v1",
                "edge_filter_reasons": ["capacity-test"],
            }
            headers = self.cluster_headers(
                f"cap-cluster-{cluster_index:03d}", self.tokens[cluster_index]
            )
            # The probe is reached through kubectl port-forward and a pooled
            # client whose idle connections the server closes after its
            # keep-alive timeout; a dropped connection is the transport's
            # doing, not the admission decision under test. Retry it on a fresh
            # connection and keep count, so the judgement still sees every
            # request's real status (live 2026-09-07: one drop aborted the case).
            transport_retries = 0
            for attempt in range(3):
                try:
                    response = client.post(
                        "/v1/collector-events/host-telemetry",
                        headers=headers,
                        json=payload,
                    )
                    break
                except httpx.TransportError as exc:
                    transport_retries += 1
                    if attempt == 2:
                        return {
                            "cluster": cluster_index,
                            "status": f"transport-error:{type(exc).__name__}",
                            "retry_after": None,
                            "latency_ms": (time.perf_counter() - begin) * 1000,
                            "transport_retries": transport_retries,
                        }
                    time.sleep(0.05)
            return {
                "cluster": cluster_index,
                "status": response.status_code,
                "retry_after": response.headers.get("Retry-After"),
                "latency_ms": (time.perf_counter() - begin) * 1000,
                "transport_retries": transport_retries,
            }

        wall_start = time.perf_counter()
        with (
            ThreadPoolExecutor(max_workers=80) as a_pool,
            ThreadPoolExecutor(max_workers=4) as b_pool,
        ):
            futures = [
                a_pool.submit(send, 0, index, wall_start + index / 50.0)
                for index in range(3000)
            ]
            futures.extend(
                b_pool.submit(send, 1, index, wall_start + index / 1.0)
                for index in range(60)
            )
            responses = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - started
        client.close()

        drain_deadline = time.monotonic() + 120
        drained = False
        while time.monotonic() < drain_deadline:
            values = self.metrics(probe.url)
            if self.metric_value(values, "gpu_fault_processor_queue_depth") == 0:
                time.sleep(1)
                values = self.metrics(probe.url)
                if self.metric_value(values, "gpu_fault_processor_queue_depth") == 0:
                    drained = True
                    break
            time.sleep(1)
        monitor_stop.set()
        monitor_thread.join(timeout=3)
        write_json(case_dir / "metrics-samples.json", metrics_samples)

        by_cluster = {
            cluster: [item for item in responses if item["cluster"] == cluster]
            for cluster in (0, 1)
        }
        result: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 3),
            "a_status_counts": dict(
                sorted(Counter(item["status"] for item in by_cluster[0]).items())
            ),
            "b_status_counts": dict(
                sorted(Counter(item["status"] for item in by_cluster[1]).items())
            ),
            "a_retry_after_values": sorted(
                {item["retry_after"] for item in by_cluster[0] if item["status"] == 429}
            ),
            "b_latency_ms": {
                "p50": percentile([item["latency_ms"] for item in by_cluster[1]], 0.50),
                "p95": percentile([item["latency_ms"] for item in by_cluster[1]], 0.95),
                "p99": percentile([item["latency_ms"] for item in by_cluster[1]], 0.99),
            },
            "metric_maxima": maxima,
            "queue_drained": drained,
            "transport_retries": sum(
                int(item.get("transport_retries", 0)) for item in responses
            ),
        }
        passed = all(
            (
                result["a_status_counts"].get(429, 0) > 0,
                result["a_retry_after_values"] == ["2"],
                result["b_status_counts"] == {202: 60},
                not any(
                    isinstance(status, int) and status >= 500
                    for status in result["a_status_counts"]
                ),
                maxima.get("queue_depth", 0) < 10000,
                maxima.get("a_rejections", 0) > 0,
                maxima.get("b_rejections", 0) == 0,
                drained,
            )
        )
        result["status"] = "PASS" if passed else "FAIL"
        write_json(case_dir / "summary.json", result)
        cleanup = self.cleanup_probe(probe)
        write_json(case_dir / "cleanup.json", cleanup)
        if not passed:
            raise CapError("GF-REGIONAL-CAP-001 failed")
        return result

    def _cap002_claim(self, probe: Any, index: int, phase: str) -> dict[str, Any]:
        begin = time.perf_counter()
        with httpx.Client(base_url=probe.url, timeout=30) as client:
            response = client.post(
                "/v1/regional/executors/claim",
                headers=self.cluster_headers(
                    f"cap-cluster-{index:03d}", self.tokens[index]
                ),
                json={
                    "executor_id": f"cap002-{phase}-{index:03d}",
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

    def _cap002_behavior(self, probe: Any, case_dir: Any) -> dict[str, Any]:
        behavior_hold = self.probe_control(
            probe,
            "/__cap__/hold",
            {"tag": "behavior", "durations": [5, 5, 5, 5]},
        )
        with ThreadPoolExecutor(max_workers=20) as pool:
            initial_results = list(
                pool.map(
                    lambda index: self._cap002_claim(probe, index, "behavior"),
                    range(20),
                )
            )
        self.probe_control(probe, "/__cap__/release", {"tag": "behavior"})
        with ThreadPoolExecutor(max_workers=20) as pool:
            retry_results = list(
                pool.map(
                    lambda index: self._cap002_claim(probe, index, "retry"),
                    range(20),
                )
            )
        metrics = self.metrics(probe.url)
        status_counts = dict(
            sorted(Counter(item["status"] for item in initial_results).items())
        )
        retry_status_counts = dict(
            sorted(Counter(item["status"] for item in retry_results).items())
        )
        retry_after_values = sorted(
            {item["retry_after"] for item in initial_results if item["status"] == 503}
        )
        behavior: dict[str, Any] = {
            "hold": behavior_hold,
            "initial_status_counts": status_counts,
            "retry_after_values": retry_after_values,
            "retry_status_counts": retry_status_counts,
            "store_io_rejections": self.metric_value(
                metrics, "gpu_fault_store_io_rejections_total"
            ),
        }
        behavior["passed"] = all(
            (
                status_counts.get(503, 0) > 0,
                not any(status == 500 for status in status_counts),
                retry_after_values == ["2"],
                retry_status_counts == {200: 20},
                behavior["store_io_rejections"] > 0,
            )
        )
        write_json(case_dir / "behavior.json", behavior)
        if not behavior["passed"]:
            raise CapError("CAP-002 503 behavior phase failed")
        return behavior

    def _cap002_alert(
        self,
        probe: Any,
        case_dir: Any,
        behavior: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        alert_hold = self.probe_control(
            probe,
            "/__cap__/hold",
            {"tag": "alert", "durations": [600, 600, 600]},
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
                values, "gpu_fault_store_io_rejections_total"
            ),
        }
        write_json(case_dir / "saturation-sample.json", sample)
        if ratio <= 0.9:
            raise CapError("CAP-002 alert hold did not render observable 4/4")
        alert_poll = []
        fired = False
        started = time.monotonic()
        for attempt in range(1, 33):
            time.sleep(15)
            query = self.amp_request(
                "POST",
                "/api/v1/query",
                {
                    "query": (
                        "max(gpu_fault_store_io_in_flight"
                        f'{{pod="{probe.pod}"}} / '
                        "gpu_fault_store_io_max_in_flight"
                        f'{{pod="{probe.pod}"}})'
                    )
                },
            )["data"]["result"]
            states = self.alert_states("GpuFaultStoreIoSaturated")
            alert_poll.append(
                {
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "target_ratio_present": bool(query),
                    "states": states,
                }
            )
            if "firing" in states:
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
        passed = bool(behavior["passed"] and ratio > 0.9 and fired)
        result["status"] = "PASS" if passed else "FAIL"
        write_json(case_dir / "summary.json", result)
        return result, passed

    def _cap002_wait_resolved(self, case_dir: Any) -> bool:
        resolved_poll = []
        for attempt in range(1, 13):
            states = self.alert_states("GpuFaultStoreIoSaturated")
            resolved_poll.append(
                {"attempt": attempt, "observed_at": utc_now(), "states": states}
            )
            if "firing" not in states:
                write_json(case_dir / "alert-resolve-poll.json", resolved_poll)
                return True
            time.sleep(15)
        write_json(case_dir / "alert-resolve-poll.json", resolved_poll)
        return False

    def case_002_v2(self) -> dict[str, Any]:
        if self.alert_states("GpuFaultStoreIoSaturated"):
            raise CapError("CAP-002 target alert is already active before the probe")
        probe = self.deploy_probe(
            "CAP002",
            {
                "GPU_FAULT_SERVICE_ROLE": "ingress",
                "GPU_FAULT_STORE_IO_WORKERS": "4",
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": "4",
                "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS": "1",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
            },
        )
        case_dir = self.run_dir / "CAP-002"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        result: dict[str, Any] = {}
        passed = False
        try:
            behavior = self._cap002_behavior(probe, case_dir)
            result, passed = self._cap002_alert(probe, case_dir, behavior)
        finally:
            for tag in ("behavior", "alert"):
                try:
                    self.probe_control(probe, "/__cap__/release", {"tag": tag})
                except Exception:
                    pass
            write_json(case_dir / "cleanup.json", self.cleanup_probe(probe))
        resolved = self._cap002_wait_resolved(case_dir)
        result["alert_resolved_after_cleanup"] = resolved
        write_json(case_dir / "summary.json", result)
        if not passed or not resolved:
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
        results: list[dict[str, Any]] = []
        client = httpx.Client(base_url=probe.url, timeout=20)
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
                            "execution_owners": ["gpu-fault-kubernetes-adapter"],
                            "max_commands": 1,
                            "lease_seconds": 10,
                        },
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
        client.close()
        ended_at = datetime.now(timezone.utc) + timedelta(minutes=1)
        budget = self.connection_budget()
        cloudwatch = self.cloudwatch_window(started_at, ended_at)
        write_json(case_dir / "connection-budget.json", budget)
        write_json(case_dir / "aurora.json", cloudwatch)
        passed = all(
            row["status_counts"] == {200: row["requests"]}
            and (row["latency_ms"]["p95"] or float("inf")) < 1000
            for row in results
        )
        result: dict[str, Any] = {
            "status": "PASS" if passed else "FAIL",
            "scenarios": results,
            "connection_budget": budget,
            "recommendations": {
                "executor_poll_seconds": 2,
                "executor_batch_size": 5,
                "postgres_pool_change": (
                    "none"
                    if budget["budget_ratio"] < 0.8
                    else "reduce per-process pools or raise max_connections"
                ),
            },
        }
        write_json(case_dir / "summary.json", result)
        cleanup = self.cleanup_probe(probe)
        write_json(case_dir / "cleanup.json", cleanup)
        if not passed:
            raise CapError("GF-REGIONAL-CAP-003 failed")
        return result

    def _cap004_execute_claimed(
        self,
        probe: Any,
        client: httpx.Client,
        headers: dict[str, str],
        commands: list[dict[str, Any]],
    ) -> dict[str, Any]:
        competitor_stop = threading.Event()
        competitor_claimed: list[str] = []
        competitor_errors = 0
        counters_lock = threading.Lock()
        active = 0
        max_active = 0
        renewals = 0
        renewal_failures = 0
        injected_failure_status: int | None = None
        injected = False
        durations: list[float] = []
        lease_progressions: list[dict[str, Any]] = []
        result_statuses: list[int] = []
        renewal_stops: dict[str, threading.Event] = {}
        renewal_threads: dict[str, threading.Thread] = {}
        renewal_expiries: dict[str, list[Any]] = {}

        def renew(item: Mapping[str, Any]) -> None:
            nonlocal renewals, renewal_failures
            nonlocal injected_failure_status, injected
            command_id = str(item["command_id"])
            lease_token = str(item["lease_token"])
            stop_renew = renewal_stops[command_id]
            with httpx.Client(base_url=probe.url, timeout=20) as renew_client:
                while not stop_renew.wait(3):
                    token = lease_token
                    use_bad = False
                    with counters_lock:
                        if not injected:
                            injected = True
                            use_bad = True
                    if use_bad:
                        token = secrets.token_urlsafe(32)
                    try:
                        response = renew_client.post(
                            f"/v1/regional/executors/{command_id}/renew",
                            headers=headers,
                            json={
                                "executor_id": "cap004-primary",
                                "lease_token": token,
                                "lease_seconds": 10,
                            },
                        )
                    except Exception:
                        with counters_lock:
                            renewal_failures += 1
                        continue
                    with counters_lock:
                        if response.status_code == 200:
                            renewals += 1
                            renewal_expiries[command_id].append(
                                response.json().get("lease_expires_at")
                            )
                        else:
                            renewal_failures += 1
                            if use_bad:
                                injected_failure_status = response.status_code

        for item in commands:
            command_id = str(item["command_id"])
            renewal_stops[command_id] = threading.Event()
            renewal_expiries[command_id] = [item.get("lease_expires_at")]
            thread = threading.Thread(
                target=renew,
                args=(item,),
                name=f"cap004-renew-{command_id}",
                daemon=True,
            )
            renewal_threads[command_id] = thread
            thread.start()

        def competitor() -> None:
            nonlocal competitor_errors
            with httpx.Client(base_url=probe.url, timeout=20) as competitor_client:
                while not competitor_stop.wait(0.25):
                    try:
                        response = competitor_client.post(
                            "/v1/regional/executors/claim",
                            headers=headers,
                            json={
                                "executor_id": "cap004-competitor",
                                "execution_owners": ["gpu-fault-node-agent"],
                                "max_commands": 25,
                                "lease_seconds": 10,
                            },
                        )
                    except Exception:
                        with counters_lock:
                            competitor_errors += 1
                        continue
                    if response.status_code == 200:
                        competitor_claimed.extend(
                            command["command_id"]
                            for command in response.json().get("commands") or []
                        )

        competitor_thread = threading.Thread(target=competitor, daemon=True)
        competitor_thread.start()

        def execute(item: Mapping[str, Any]) -> None:
            nonlocal active, max_active
            command_id = str(item["command_id"])
            initial_expiry = item.get("lease_expires_at")
            with counters_lock:
                active += 1
                max_active = max(max_active, active)
            begin = time.perf_counter()
            try:
                time.sleep(12)
                with httpx.Client(base_url=probe.url, timeout=20) as result_client:
                    response = result_client.post(
                        f"/v1/regional/executors/{command_id}/result",
                        headers=headers,
                        json={
                            "lease_token": str(item["lease_token"]),
                            "status": "SUCCEEDED",
                            "status_source": "cap004-simulator",
                            "details": {
                                "simulated": True,
                                "operation": item["step"]["operation"],
                            },
                        },
                    )
                with counters_lock:
                    result_statuses.append(response.status_code)
            finally:
                renewal_stops[command_id].set()
                renewal_threads[command_id].join(timeout=5)
                with counters_lock:
                    active -= 1
                    durations.append(time.perf_counter() - begin)
                    expiries = renewal_expiries[command_id]
                    lease_progressions.append(
                        {
                            "command_id": command_id,
                            "initial": initial_expiry,
                            "latest": expiries[-1],
                            "renewal_count": max(0, len(expiries) - 1),
                        }
                    )

        wall_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(execute, commands))
        finally:
            for stop_renew in renewal_stops.values():
                stop_renew.set()
            for thread in renewal_threads.values():
                thread.join(timeout=5)
            competitor_stop.set()
            competitor_thread.join(timeout=5)
        final_claim = client.post(
            "/v1/regional/executors/claim",
            headers=headers,
            json={
                "executor_id": "cap004-final",
                "execution_owners": ["gpu-fault-node-agent"],
                "max_commands": 25,
                "lease_seconds": 10,
            },
        )
        final_claim.raise_for_status()
        return {
            "max_active": max_active,
            "durations": durations,
            "renewals": renewals,
            "renewal_failures": renewal_failures,
            "injected_failure_status": injected_failure_status,
            "competitor_claimed": competitor_claimed,
            "competitor_errors": competitor_errors,
            "result_statuses": result_statuses,
            "lease_progressions": lease_progressions,
            "wall": time.perf_counter() - wall_started,
            "final_commands": final_claim.json().get("commands") or [],
        }

    def case_004(self) -> dict[str, Any]:
        probe = self.deploy_probe(
            "CAP004",
            {
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "8",
                "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT": "64",
            },
        )
        case_dir = self.run_dir / "CAP-004"
        case_dir.mkdir(mode=0o700, exist_ok=True)
        seed = self.kubectl(
            "exec",
            probe.pod,
            "--",
            "/opt/gpu-fault/control-plane/bin/python",
            "/opt/cap/cap_seed_commands.py",
        )
        (case_dir / "seed.json").write_text(seed.stdout, encoding="utf-8")
        (case_dir / "seed.json").chmod(0o600)
        client = httpx.Client(base_url=probe.url, timeout=30)
        headers = self.cluster_headers("cap-cluster-000", self.tokens[0])
        claim_started = time.perf_counter()
        claim_response = client.post(
            "/v1/regional/executors/claim",
            headers=headers,
            json={
                "executor_id": "cap004-primary",
                "execution_owners": ["gpu-fault-node-agent"],
                "max_commands": 25,
                "lease_seconds": 10,
            },
        )
        claim_latency = time.perf_counter() - claim_started
        claim_bytes = len(claim_response.content)
        claim_response.raise_for_status()
        commands = claim_response.json().get("commands") or []
        if len(commands) != 25:
            raise CapError(f"CAP-004 claimed {len(commands)} commands, expected 25")
        stats = self._cap004_execute_claimed(probe, client, headers, commands)
        client.close()
        result: dict[str, Any] = {
            "claim_response_bytes": claim_bytes,
            "claim_latency_ms": claim_latency * 1000,
            "commands_claimed": len(commands),
            "max_concurrent_commands": stats["max_active"],
            "lease_seconds": 10,
            "long_commands": sum(item >= 10 for item in stats["durations"]),
            "renewals": stats["renewals"],
            "renewal_failures": stats["renewal_failures"],
            "injected_renewal_failure_status": stats["injected_failure_status"],
            "competitor_duplicate_claims": len(stats["competitor_claimed"]),
            "competitor_transport_errors": stats["competitor_errors"],
            "result_status_counts": dict(
                sorted(Counter(stats["result_statuses"]).items())
            ),
            "final_open_claims": len(stats["final_commands"]),
            "command_latency_ms": {
                "p50": percentile([value * 1000 for value in stats["durations"]], 0.50),
                "p95": percentile([value * 1000 for value in stats["durations"]], 0.95),
                "p99": percentile([value * 1000 for value in stats["durations"]], 0.99),
            },
            "batch_wall_seconds": stats["wall"],
            "lease_progressions": stats["lease_progressions"],
            "recommended_batch_size": 25,
            "recommended_max_concurrent_commands": 5,
        }
        passed = all(
            (
                len(commands) == 25,
                stats["max_active"] == 5,
                result["long_commands"] >= 5,
                stats["renewals"] > 0,
                stats["renewal_failures"] >= 1,
                stats["injected_failure_status"] in {403, 409},
                not stats["competitor_claimed"],
                stats["competitor_errors"] == 0,
                result["result_status_counts"] == {200: 25},
                not stats["final_commands"],
                all(item["renewal_count"] > 0 for item in stats["lease_progressions"]),
            )
        )
        result["status"] = "PASS" if passed else "FAIL"
        write_json(case_dir / "summary.json", result)
        write_json(case_dir / "cleanup.json", self.cleanup_probe(probe))
        if not passed:
            raise CapError("GF-REGIONAL-CAP-004 failed")
        return result
