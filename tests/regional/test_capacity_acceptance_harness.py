"""Verdict and lifecycle contracts of the CAP-001..004 harness.

The pure verdict functions are judged against recorded results; the harness
lifecycle is judged against a fake subclass that replaces every kubectl and
AWS call with a script and records the order in which evidence is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import capacity_acceptance_base as base
from scripts.e2e.regional import capacity_acceptance_cases as cases


# --------------------------------------------------------------------------- #
# CAP-001
# --------------------------------------------------------------------------- #
def _cap001_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "a_status_counts": {202: 1000, 429: 2000},
        "a_retry_after_values": ["2"],
        "b_baseline_status_counts": {202: cases.CAP001_BASELINE_REQUESTS},
        "b_status_counts": {202: cases.CAP001_STORM_B_REQUESTS},
        "b_baseline_latency_ms": {"p50": 20.0, "p95": 40.0, "p99": 50.0},
        "b_latency_ms": {"p50": 25.0, "p95": 70.0, "p99": 90.0},
        "b_latency_factor": 2.0,
        "queue_depth_bound": cases.CAP001_A_QUEUE_CAP + cases.CAP001_QUEUE_SLACK,
        "metric_maxima": {
            "queue_depth": 21.0,
            "a_rejections": 2000.0,
            "b_rejections": 0.0,
        },
        "queue_drained": True,
    }
    result.update(overrides)
    return result


def test_cap001_passes_when_b_stays_within_the_baseline_factor() -> None:
    assert cases.cap001_failures(_cap001_result()) == []


def test_cap001_flags_b_latency_beyond_the_baseline_factor() -> None:
    failures = cases.cap001_failures(
        _cap001_result(b_latency_ms={"p50": 25.0, "p95": 81.0, "p99": 90.0})
    )
    assert any("exceeds 2x the B-only baseline" in item for item in failures), failures


def test_cap001_factor_is_a_parameter_not_a_constant() -> None:
    result = _cap001_result(
        b_latency_ms={"p50": 25.0, "p95": 81.0, "p99": 90.0}, b_latency_factor=3.0
    )
    assert cases.cap001_failures(result) == []


def test_cap001_missing_baseline_is_a_failure_not_a_free_pass() -> None:
    result = _cap001_result(
        b_baseline_latency_ms={"p50": None, "p95": None, "p99": None}
    )
    assert any("missing" in item for item in cases.cap001_failures(result)), (
        "a missing baseline latency is reported as missing"
    )


def test_cap001_queue_depth_bound_is_the_a_cap_plus_slack() -> None:
    bound = cases.CAP001_A_QUEUE_CAP + cases.CAP001_QUEUE_SLACK
    assert bound < 100, "the bound must be tight, not the 10000 global cap"
    result = _cap001_result(
        metric_maxima={
            "queue_depth": bound + 1,
            "a_rejections": 1.0,
            "b_rejections": 0.0,
        }
    )
    failures = cases.cap001_failures(result)
    assert any(f"bound {bound}" in item for item in failures), failures


# --------------------------------------------------------------------------- #
# CAP-002
# --------------------------------------------------------------------------- #
def test_cap002_holds_every_store_io_slot() -> None:
    three_of_four = 3 / 4
    assert cases.cap002_saturation_error(three_of_four) is not None, (
        "three held slots leave the ratio at 0.75 and the alert cannot fire"
    )
    assert cases.cap002_saturation_error(1.0) is None
    assert cases.CAP002_STORE_IO_SLOTS == 4


def test_cap002_saturation_error_names_the_ratio() -> None:
    message = cases.cap002_saturation_error(0.75)
    assert message is not None and "0.75" in message and "4 slots" in message


def test_cap002_resolve_budget_covers_the_five_minute_for_clause() -> None:
    budget = cases.CAP002_RESOLVE_ATTEMPTS * cases.CAP002_RESOLVE_POLL_SECONDS
    assert budget >= 6 * 60


def _bare_cases(tmp_path: Path) -> cases.CapacityAcceptanceCases:
    harness = cases.CapacityAcceptanceCases.__new__(cases.CapacityAcceptanceCases)
    harness.run_dir = tmp_path
    harness.tokens = ["t"] * 20
    return harness


def test_cap002_alert_phase_holds_four_slots(tmp_path: Path, monkeypatch) -> None:
    harness = _bare_cases(tmp_path)
    probe = SimpleNamespace(url="http://probe", pod="probe-pod")
    holds: list[dict[str, Any]] = []

    def probe_control(
        _probe: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if path.endswith("/hold"):
            holds.append(payload)
        return {"tag": payload["tag"]}

    monkeypatch.setattr(harness, "probe_control", probe_control)
    monkeypatch.setattr(
        harness,
        "metrics",
        lambda _url: [
            ("gpu_fault_store_io_in_flight", {}, 4.0),
            ("gpu_fault_store_io_max_in_flight", {}, 4.0),
            ("gpu_fault_store_io_rejections_total", {}, 3.0),
        ],
    )
    monkeypatch.setattr(
        harness, "amp_request", lambda *_a, **_k: {"data": {"result": [1]}}
    )
    monkeypatch.setattr(harness, "alert_states", lambda _name: ["firing"])
    monkeypatch.setattr(cases.time, "sleep", lambda _seconds: None)
    behavior = {
        "passed": True,
        "initial_status_counts": {503: 20},
        "retry_after_values": ["2"],
        "retry_status_counts": {200: 20},
        "store_io_rejections": 3.0,
    }

    result, passed = harness.cap002_alert(probe, tmp_path, behavior)

    assert passed is True
    assert holds == [
        {"tag": "alert", "durations": [cases.CAP002_ALERT_HOLD_SECONDS] * 4}
    ]
    assert result["max_store_io_ratio"] == 1.0


def test_cap002_waits_for_the_alert_to_resolve_before_deleting_the_probe(
    tmp_path: Path, monkeypatch
) -> None:
    harness = _bare_cases(tmp_path)
    probe = SimpleNamespace(url="http://probe", pod="probe-pod")
    events: list[tuple[str, ...]] = []
    monkeypatch.setattr(harness, "alert_states", lambda _name: [])
    monkeypatch.setattr(harness, "deploy_probe", lambda _case, _env: probe)
    monkeypatch.setattr(harness, "_cap002_behavior", lambda _p, _d: {"passed": True})
    monkeypatch.setattr(
        harness, "cap002_alert", lambda _p, _d, _b: ({"status": "PASS"}, True)
    )

    def probe_control(
        _probe: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        events.append(("release", payload["tag"]))
        return {}

    def wait_resolved(_case_dir: Any, *, attempts: int = 0) -> bool:
        events.append(("wait_resolved",))
        return True

    def cleanup_probe(_probe: Any) -> dict[str, Any]:
        events.append(("cleanup_probe",))
        return {"database_dropped": True, "residual_probe_pods": []}

    monkeypatch.setattr(harness, "probe_control", probe_control)
    monkeypatch.setattr(harness, "_cap002_wait_resolved", wait_resolved)
    monkeypatch.setattr(harness, "cleanup_probe", cleanup_probe)

    result = harness.case_002_v2()

    assert events.index(("wait_resolved",)) < events.index(("cleanup_probe",)), events
    assert ("release", "alert") in events[: events.index(("wait_resolved",))]
    assert result["alert_resolved_after_release"] is True


# --------------------------------------------------------------------------- #
# CAP-003
# --------------------------------------------------------------------------- #
def _row(
    cluster_count: int, p95: float | None, status_counts: dict[int, int] | None = None
) -> dict[str, Any]:
    requests = cluster_count * 60
    return {
        "cluster_count": cluster_count,
        "requests": requests,
        "status_counts": status_counts or {200: requests},
        "latency_ms": {"p50": (p95 or 0) / 2, "p95": p95, "p99": (p95 or 0) * 1.2},
        "wall_seconds": 60.0,
        "api_average_cpu_cores": 0.1 * cluster_count,
        "database_connections_before": 2,
        "database_connections_after": 2,
        "store_io_wait_seconds_max": 0.0,
    }


BUDGET = {"budget_ratio": 0.5}


def test_cap003_without_a_knee_recommends_from_the_largest_tested_count() -> None:
    results = [_row(1, 50.0), _row(5, 80.0), _row(10, 120.0), _row(20, 300.0)]
    recommendations = cases.cap003_recommendations(results, BUDGET)

    assert cases.cap003_knee(results) is None
    assert recommendations["sustained_cluster_count_at_1rps"] == 20
    assert recommendations["executor_poll_seconds"] == 2
    assert set(recommendations["per_cluster_count"]) == {"1", "5", "10", "20"}
    assert recommendations["per_cluster_count"]["20"]["p95_ms"] == 300.0


def test_cap003_knee_at_the_latency_threshold_drives_the_recommendation() -> None:
    results = [_row(1, 50.0), _row(5, 80.0), _row(10, 120.0), _row(20, 1200.0)]
    knee = cases.cap003_knee(results)
    recommendations = cases.cap003_recommendations(results, BUDGET)

    assert knee is not None and knee["cluster_count"] == 20
    assert "p95" in knee["reason"]
    assert recommendations["sustained_cluster_count_at_1rps"] == 10
    assert recommendations["executor_poll_seconds"] == 4


def test_cap003_knee_on_a_non_200_answer() -> None:
    results = [_row(1, 50.0), _row(10, 120.0, {200: 590, 503: 10}), _row(20, 200.0)]
    knee = cases.cap003_knee(results)

    assert knee is not None and knee["cluster_count"] == 10
    assert knee["reason"] == "non-200 claim responses"


# --------------------------------------------------------------------------- #
# Harness lifecycle
# --------------------------------------------------------------------------- #
class _ScriptedHarness(base.CapHarnessBase):
    """A harness whose cluster calls are scripted; nothing leaves the process."""

    def __init__(  # noqa: D107 - bypasses CapCoreHarness.__init__ on purpose
        self,
        run_dir: Path,
        *,
        baselines: list[Any],
        case_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.case_id = "GF-REGIONAL-CAP-001"
        self.predecessor = {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
        self.region = "region"
        self.cpu_kubeconfig = "kubeconfig"
        self.namespace = "namespace"
        self.runtime_image = "image@sha256:0"
        self.resource_prefix = "gpu-fault-test"
        self.case_results = []
        self.active_probe = None
        self.b_latency_factor = 2.0
        self._baselines = iter(baselines)
        self._case_error = case_error
        self._cleanup_error = cleanup_error
        self.events: list[str] = []

    def production_baseline(self) -> dict[str, Any]:
        value = next(self._baselines)
        if isinstance(value, BaseException):
            raise value
        return dict(value)

    def create_common_resources(self) -> None:
        self.events.append("create_common")

    def cleanup_common(self) -> None:
        self.events.append("cleanup_common")
        if self._cleanup_error is not None:
            raise self._cleanup_error

    def case_001(self) -> dict[str, Any]:
        self.events.append("case")
        if self._case_error is not None:
            raise self._case_error
        return {"status": "PASS"}


def _case_document(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "GF-REGIONAL-CAP-001.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _record_writes(monkeypatch, sink: list[tuple[str, Any]]) -> None:
    original = base.write_json

    def recording(path: Path, value: Any) -> None:
        sink.append(
            (path.name, value.get("verdict") if isinstance(value, dict) else None)
        )
        original(path, value)

    monkeypatch.setattr(base, "write_json", recording)


def test_harness_writes_pending_first_and_the_verdict_last(
    tmp_path: Path, monkeypatch
) -> None:
    writes: list[tuple[str, Any]] = []
    _record_writes(monkeypatch, writes)
    harness = _ScriptedHarness(tmp_path, baselines=[{"pods": 1}, {"pods": 1}])

    assert harness.run() == 0

    verdicts = [
        verdict for name, verdict in writes if name == "GF-REGIONAL-CAP-001.json"
    ]
    assert verdicts == ["PENDING", "PASS"]
    names = [name for name, _ in writes]
    assert names.index("production-after.json") < len(names) - 1
    assert names.index("production-after.json") < names.index(
        "phase-partial-summary.json"
    )
    assert names[-2:] == ["GF-REGIONAL-CAP-001.json", "phase-partial-summary.json"]
    assert _case_document(tmp_path)["production_unchanged"] is True


def test_a_changed_production_baseline_never_leaves_a_pass_on_disk(
    tmp_path: Path,
) -> None:
    harness = _ScriptedHarness(tmp_path, baselines=[{"pods": 1}, {"pods": 2}])

    with pytest.raises(base.CapError, match="baseline changed"):
        harness.run()

    document = _case_document(tmp_path)
    assert document["verdict"] == "FAIL"
    assert document["production_unchanged"] is False


def test_cleanup_failures_are_recorded_and_fail_the_case(tmp_path: Path) -> None:
    harness = _ScriptedHarness(
        tmp_path,
        baselines=[{"pods": 1}, {"pods": 1}],
        cleanup_error=RuntimeError("configmap delete refused"),
    )

    with pytest.raises(base.CapError, match="cleanup failed"):
        harness.run()

    document = _case_document(tmp_path)
    assert document["verdict"] == "FAIL"
    assert document["cleanup_errors"] == [
        "common resource cleanup: RuntimeError: configmap delete refused"
    ]


def test_a_baseline_read_failure_does_not_mask_the_case_error(tmp_path: Path) -> None:
    harness = _ScriptedHarness(
        tmp_path,
        baselines=[{"pods": 1}, OSError("kubectl get pods timed out")],
        case_error=ValueError("cluster B was rejected"),
    )

    with pytest.raises(ValueError, match="cluster B was rejected"):
        harness.run()

    document = _case_document(tmp_path)
    assert document["verdict"] == "FAIL"
    assert document["error"] == "ValueError: cluster B was rejected"
    assert document["cleanup_errors"] == [
        "production baseline after run: OSError: kubectl get pods timed out"
    ]
    assert document["production_unchanged"] is False


def test_verdict_is_pass_only_when_everything_held() -> None:
    assert (
        base.CapHarnessBase.verdict(
            result={}, error=None, cleanup_errors=[], production_unchanged=True
        )
        == "PASS"
    )
    assert (
        base.CapHarnessBase.verdict(
            result={}, error=None, cleanup_errors=["x"], production_unchanged=True
        )
        == "FAIL"
    )
    assert (
        base.CapHarnessBase.verdict(
            result=None, error=None, cleanup_errors=[], production_unchanged=True
        )
        == "FAIL"
    )


def test_b_latency_factor_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(base.CapError, match="b_latency_factor"):
        base.CapCoreHarness(
            site_path=tmp_path / "site.yaml",
            run_dir=tmp_path,
            case_id="GF-REGIONAL-CAP-001",
            predecessor={"valid": True},
            b_latency_factor=0.5,
        )


class _ManifestHarness(base.CapProbeHarness):
    """A probe harness that records the manifests it would apply."""

    def __init__(  # noqa: D107 - bypasses CapCoreHarness.__init__ on purpose
        self, live_worker: dict[str, Any]
    ) -> None:
        self.live_worker = live_worker
        self.namespace = "namespace"
        self.run_id = "cap000000"
        self.runtime_image = "image@sha256:0"
        self.configmap_name = "gpu-fault-cap000000-scripts"
        self.applied: list[dict[str, Any]] = []

    def apply(self, value: Any) -> None:
        self.applied.append(dict(value))


def _live_worker(*volume_names: str) -> dict[str, Any]:
    return {
        "spec": {
            "template": {
                "spec": {
                    "serviceAccountName": "gpu-fault-control-plane",
                    "volumes": [
                        {"name": name, "configMap": {"name": f"gpu-fault-{name}"}}
                        for name in volume_names
                    ],
                    "containers": [
                        {
                            "image": "image@sha256:0",
                            "volumeMounts": [
                                {
                                    "name": name,
                                    "mountPath": f"/etc/gpu-fault/{name}",
                                    "readOnly": True,
                                }
                                for name in volume_names
                            ],
                        }
                    ],
                }
            }
        }
    }


def _probe_pod_spec(harness: _ManifestHarness) -> dict[str, Any]:
    harness._apply_probe_resources(
        suffix="cap001",
        deployment="gpu-fault-cap000000-cap001",
        service="gpu-fault-cap000000-cap001",
        environment=[],
    )
    (deployment,) = [m for m in harness.applied if m["kind"] == "Deployment"]
    return dict(deployment["spec"]["template"]["spec"])


def test_probe_inherits_the_workers_rds_ca_bundle_verbatim() -> None:
    # The Aurora DSN pins sslmode=verify-full to the projected CA bundle; a
    # probe without the worker's mount cannot open a single connection.
    pod = _probe_pod_spec(_ManifestHarness(_live_worker("artifact", "rds-ca-bundle")))
    assert {
        "name": "rds-ca-bundle",
        "configMap": {"name": "gpu-fault-rds-ca-bundle"},
    } in pod["volumes"]
    assert {
        "name": "rds-ca-bundle",
        "mountPath": "/etc/gpu-fault/rds-ca-bundle",
        "readOnly": True,
    } in pod["containers"][0]["volumeMounts"]
    # Only the TLS material is carried; the worker's artifact stays behind.
    assert [v["name"] for v in pod["volumes"]] == ["scripts", "work", "rds-ca-bundle"]


def test_probe_carries_nothing_when_the_release_mounts_no_ca_bundle() -> None:
    pod = _probe_pod_spec(_ManifestHarness(_live_worker("artifact")))
    assert [v["name"] for v in pod["volumes"]] == ["scripts", "work"]
    assert [m["name"] for m in pod["containers"][0]["volumeMounts"]] == [
        "scripts",
        "work",
    ]


class _BaselineHarness(base.CapHarnessBase):
    """A core harness whose kubectl reads are scripted."""

    def __init__(  # noqa: D107 - bypasses CapCoreHarness.__init__ on purpose
        self, pods: list[dict[str, Any]]
    ) -> None:
        self._pods = pods

    def kubectl_json(self, *args: str) -> Any:
        return {"items": self._pods if args[1] == "pods" else []}


def _pod(name: str, labels: dict[str, str]) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "uid": f"uid-{name}", "labels": labels},
        "status": {"phase": "Running", "containerStatuses": [{"restartCount": 0}]},
    }


def test_baseline_ignores_capacity_probes_wearing_the_worker_label() -> None:
    # A probe left behind by another run wears app=gpu-fault-control-worker;
    # its later removal must not turn a passing case into "baseline changed".
    harness = _BaselineHarness(
        [
            _pod("gpu-fault-control-worker-1", {"app": "gpu-fault-control-worker"}),
            _pod(
                "gpu-fault-cap000000-cap001-1",
                {
                    "app": "gpu-fault-control-worker",
                    "gpu-fault.io/capacity-probe": "cap000000-cap001",
                },
            ),
        ]
    )
    assert [p["name"] for p in harness.production_baseline()["pods"]] == [
        "gpu-fault-control-worker-1"
    ]
