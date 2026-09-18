from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import ha010_verdicts
from scripts.e2e.regional import run_ha002_pdb_topology as ha002
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional import run_ha007_control_worker_shutdown as ha007
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010
from scripts.e2e.regional.ha_store_probe import cpu_store_probe, store_probe_script
from scripts.e2e.regional.regional_live_fixture import component_python


def test_ha_sql_probe_uses_projected_credentials_without_printing_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "dsn"
    path.write_text("postgresql://fixture.invalid/new")
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://fixture.invalid/old")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    namespace = {}
    exec(store_probe_script('observed = os.environ["GPU_FAULT_STORE_URL"]'), namespace)
    assert namespace["observed"] == "postgresql://fixture.invalid/new"
    assert capsys.readouterr().out == ""


def test_failed_pod_read_is_not_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    def cpu(*args, **kwargs):
        assert kwargs.get("check", True) is True
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(ha002.COMMON, "cpu", cpu)
    with pytest.raises(RuntimeError, match="API unavailable"):
        ha002.pod_by_name("pod-a")


def test_owner_lookup_does_not_match_a_pod_name_prefix() -> None:
    regional = SimpleNamespace(
        ready_pods=lambda *a: [{"name": "pod-a"}, {"name": "pod-ab"}]
    )
    assert ha004.owner_pod(regional, "cluster/pod-ab")["name"] == "pod-ab"
    with pytest.raises(Exception, match="cannot map"):
        ha004.owner_pod(regional, "cluster/pod-abc")


def test_failed_auth_log_read_is_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def control(*args, **kwargs):
        assert kwargs.get("check", True) is True
        raise RuntimeError("logs unavailable")

    monkeypatch.setattr(ha009.BASE, "control", control)
    with pytest.raises(RuntimeError, match="logs unavailable"):
        ha009.auth_failures_in_logs("worker", datetime.now(timezone.utc))


def test_credential_health_probe_uses_the_role_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ha009.BASE,
        "control",
        lambda *args, **kwargs: (
            calls.append(args)
            or json.dumps(
                {
                    "healthz_status": 200,
                    "metrics_status": 200,
                    "metrics_text": "gpu_fault_postgres_pool_connections_errors_total 0",
                }
            )
        ),
    )
    result = ha009.probe_pod("worker", 8081)
    assert calls[0][-1] == "8081"
    assert result["healthz_status"] == 200


@pytest.mark.parametrize("value", ["NaN", "inf", "-1"])
def test_ha010_budgets_reject_nonfinite_or_negative_values(value: str) -> None:
    with pytest.raises(ValueError):
        ha010_verdicts.positive_seconds(value, default=90, label="budget")


def test_two_endpoints_are_not_a_continuous_failover_timeline() -> None:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    sample = {
        "livez": 200,
        "healthz": 200,
        "registry_ready": True,
        "secret_drift": False,
    }
    errors = ha010_verdicts.sampler_errors(
        [{**sample, "t": start.timestamp()}, {**sample, "t": start.timestamp() + 200}],
        pod="p",
        stale_seconds=90,
        rds_available_at=start + timedelta(seconds=30),
        failover_requested_at=start + timedelta(seconds=10),
    )
    assert any("missing samples" in error for error in errors), errors


def test_readiness_outage_uses_the_previous_state_for_irregular_samples() -> None:
    assert (
        ha010_verdicts.readiness_outage_seconds(
            [
                {"t": 1, "healthz": 200},
                {"t": 11, "healthz": 503},
                {"t": 41, "healthz": 200},
            ]
        )
        == 30
    )


def test_sampler_cleanup_waits_and_still_stops_the_other_sampler(
    tmp_path: Path,
) -> None:
    stopped = []

    def failure():
        raise RuntimeError("wait timed out")

    run = SimpleNamespace(
        samplers=[
            SimpleNamespace(pod="first", stop=failure),
            SimpleNamespace(
                pod="second",
                stop=lambda: stopped.append(True),
                process=SimpleNamespace(poll=lambda: 0),
            ),
        ],
        case_dir=tmp_path,
        preflight={"runtime_identity": {}},
        regional=SimpleNamespace(
            kubectl=lambda *a, **kw: "", verify_runtime_identity=lambda *a, **kw: {}
        ),
    )
    result = ha010.cleanup_case(run)
    assert result["errors"]
    assert stopped == [True]
    assert result["samplers_stopped"] == {"first": False, "second": True}


def test_readiness_failure_reaps_the_owned_shutdown_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    process = SimpleNamespace(
        poll=lambda: -9 if "kill" in calls else None,
        kill=lambda: calls.append("kill"),
        wait=lambda **kw: calls.append("wait") or -9,
    )
    monkeypatch.setattr(ha007.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(
        ha007,
        "_wait_for_file",
        lambda *a: ((_ for _ in ()).throw(RuntimeError("not ready"))),
    )
    with pytest.raises(RuntimeError, match="not ready"):
        ha007.run_probe(
            tmp_path,
            [1],
            budgets={"lifespan_budget_seconds": 2, "kubernetes_grace_seconds": 3},
        )
    assert calls == ["kill", "wait"]


def test_over_budget_evidence_cannot_hide_sigkill() -> None:
    errors = ha007.evaluate_run(
        {
            "duration_seconds": 8,
            "elapsed_seconds": 5,
            "returncode": -9,
            "completed": 0,
            "shutdown_failures": [ha007.WORKER_THREAD_NAME],
            "shutdown_seconds": 5,
        },
        lifespan_budget_seconds=5,
        kubernetes_grace_seconds=10,
    )
    assert any("killed by a signal" in error for error in errors), errors


def test_store_probe_uses_a_ready_pod_and_the_cpu_component_interpreter() -> None:
    calls = []

    def control(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "get":
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "ingress", "uid": "unit-uid"},
                            "spec": {"containers": [{"name": "api"}]},
                            "status": {
                                "phase": "Running",
                                "conditions": [{"type": "Ready", "status": "True"}],
                                "containerStatuses": [{"name": "api", "ready": True}],
                            },
                        }
                    ]
                }
            )
        return '{"count":0}'

    assert cpu_store_probe(control, 'print("{}")') == {"count": 0}
    assert calls[1][0][:6] == (
        "exec",
        "-i",
        "ingress",
        "--",
        component_python("cpu"),
        "-",
    )
    assert isinstance(calls[1][1]["stdin"], bytes), (
        f"CPU probe stdin must be bytes, got {type(calls[1][1]['stdin']).__name__}"
    )
