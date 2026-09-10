"""Contract tests for GF-REGIONAL-HA-010.

Every verdict is judged against synthetic evidence -- in-Pod ``/livez`` and
``/healthz`` timelines, Pod records before and after the failover, the
replacement Pod's status -- once on the intended run and once per way the run
can be wrong. Nothing here touches a cluster or RDS; the runner's live phases
only call these functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha010_verdicts as verdicts
from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010
from scripts.e2e.regional.probes import ha010_probe as probe
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
REQUESTED_AT = T0 + timedelta(seconds=10)
AVAILABLE_AT = REQUESTED_AT + timedelta(seconds=45)
STALE = 90.0
DELETED = "gpu-fault-api-ha-a"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_ha001() -> None:
    metadata = RegionalCaseMetadata(
        case_id=ha010.CASE_ID,
        title="",
        category="regional-high-availability",
        level="staging",
        risk="live-service-action",
        automation="manual",
        procedure="docs/x.md#gf-regional-ha-010",
        predecessor=ha010.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "HA010_", prefix
    assert ha010.CONFIRMATION.startswith(prefix), ha010.CONFIRMATION
    assert ha010.PREDECESSOR_CASE_ID == "GF-REGIONAL-HA-001", (
        "a CPU replica deletion under load must already be proven"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-HA-010", verdicts.CASE_ID


def test_the_three_cpu_deployments_are_the_aurora_readers() -> None:
    assert verdicts.CPU_DEPLOYMENTS == (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ), verdicts.CPU_DEPLOYMENTS
    assert verdicts.ROLLED_DEPLOYMENT in verdicts.CPU_DEPLOYMENTS, (
        "the rolled replica must belong to a sampled Deployment"
    )


# --------------------------------------------------------------------------- #
# Fabricated evidence
# --------------------------------------------------------------------------- #
def _pod(name: str, *, uid: str | None = None, **overrides: Any) -> dict[str, Any]:
    record = {
        "name": name,
        "uid": uid or f"uid-{name}",
        "ready": True,
        "restarts": 0,
        "port": 8080,
        "waiting_reasons": [],
        "ready_at": T0.isoformat(),
    }
    record.update(overrides)
    return record


def _pods(**overrides: Any) -> dict[str, list[dict[str, Any]]]:
    pods = {
        "gpu-fault-api-ha": [
            _pod("gpu-fault-api-ha-a"),
            _pod("gpu-fault-api-ha-b"),
            _pod("gpu-fault-api-ha-c"),
        ],
        "gpu-fault-control-worker": [_pod("gpu-fault-control-worker-a", port=8081)],
        "gpu-fault-telemetry-spool-worker": [
            _pod("gpu-fault-telemetry-spool-worker-a", port=8082)
        ],
    }
    pods.update(overrides)
    return pods


def _healthz(**overrides: Any) -> dict[str, Any]:
    registry = {"ready": True, "secret_drift": False, "secret_config_sha256": "a" * 64}
    registry.update(overrides)
    return {
        "http_status": 200,
        "payload": {"status": "ok", "regional_registry": registry},
    }


def _healthz_by_pod(pods: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        str(record["name"]): _healthz()
        for records in pods.values()
        for record in records
    }


def _rds(**overrides: Any) -> dict[str, Any]:
    rds = {
        "status": "available",
        "writer": "db-1",
        "members": [
            {"identifier": "db-1", "writer": True},
            {"identifier": "db-2", "writer": False},
        ],
    }
    rds.update(overrides)
    return rds


def _preflight(**overrides: Any) -> list[str]:
    pods = overrides.pop("pods", _pods())
    arguments: dict[str, Any] = {
        "pods": pods,
        "rds": _rds(),
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "healthz": _healthz_by_pod(pods),
        "predecessor_valid": True,
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def _samples(
    *,
    start: datetime = T0,
    seconds: int = 240,
    outage: tuple[int, int] = (15, 80),
    mutate: Any = None,
) -> list[dict[str, Any]]:
    """A timeline: /livez always 200, /healthz 503 during ``outage`` seconds."""

    samples = []
    for offset in range(0, seconds, 2):
        moment = start + timedelta(seconds=offset)
        unhealthy = outage[0] <= offset < outage[1]
        samples.append(
            {
                "t": moment.timestamp(),
                "livez": 200,
                "healthz": 503 if unhealthy else 200,
                "registry_ready": not unhealthy,
                "secret_drift": False,
                "status": "unhealthy" if unhealthy else "ok",
                "error": None,
            }
        )
    if mutate is not None:
        mutate(samples)
    return samples


def _sampler_errors(samples: list[dict[str, Any]]) -> list[str]:
    return verdicts.sampler_errors(
        samples,
        pod="gpu-fault-api-ha-b",
        stale_seconds=STALE,
        rds_available_at=AVAILABLE_AT,
        failover_requested_at=REQUESTED_AT,
    )


def _replacement(**overrides: Any) -> dict[str, Any]:
    record = _pod(
        "gpu-fault-api-ha-d",
        ready_at=(REQUESTED_AT + timedelta(seconds=150)).isoformat(),
    )
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def test_the_preflight_passes_on_a_quiet_ready_control_plane() -> None:
    assert _preflight() == [], _text(_preflight())


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"predecessor_valid": False}, "predecessor evidence is not PASS"),
        ({"rds": _rds(status="failing-over")}, "not available"),
        ({"rds": _rds(members=[{"identifier": "db-1", "writer": True}])}, "reader"),
        ({"rds": _rds(writer=None)}, "reader"),
        ({"queue": {"depth": 3}}, "queue is not empty"),
        ({"remote_commands": {"open_by_cluster": {"c": 1}}}, "workflow is in flight"),
        (
            {
                "pods": _pods(
                    **{
                        "gpu-fault-api-ha": [
                            _pod("gpu-fault-api-ha-a"),
                            _pod("gpu-fault-api-ha-b"),
                        ]
                    }
                )
            },
            "at least 3",
        ),
        (
            {
                "pods": _pods(
                    **{
                        "gpu-fault-control-worker": [
                            _pod("gpu-fault-control-worker-a", ready=False)
                        ]
                    }
                )
            },
            "not Ready",
        ),
        (
            {
                "pods": _pods(
                    **{
                        "gpu-fault-control-worker": [
                            _pod("gpu-fault-control-worker-a", port=None)
                        ]
                    }
                )
            },
            "http containerPort",
        ),
        ({"pods": _pods(**{"gpu-fault-telemetry-spool-worker": []})}, "no Pods"),
    ],
)
def test_the_preflight_refuses_each_unsafe_precondition(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _preflight(**overrides)
    assert any(fragment in item for item in errors), _text(errors)


def test_a_tier_the_site_runs_at_zero_replicas_is_not_a_missing_tier() -> None:
    """The telemetry spool worker is scaled to zero wherever spool admission is
    off; it has no Pod to sample and must not refuse the run (first live run)."""
    pods = _pods(**{"gpu-fault-telemetry-spool-worker": []})
    refused = _preflight(pods=pods)
    assert any("no Pods" in item for item in refused), _text(refused)
    assert (
        _preflight(
            pods=pods, scaled_to_zero=frozenset({"gpu-fault-telemetry-spool-worker"})
        )
        == []
    )
    # A tier that is scaled to zero but somehow has Pods is still judged.
    assert (
        _preflight(scaled_to_zero=frozenset({"gpu-fault-telemetry-spool-worker"})) == []
    )


def test_the_preflight_refuses_a_pod_that_already_drifts_from_its_secret() -> None:
    pods = _pods()
    healthz = _healthz_by_pod(pods)
    healthz["gpu-fault-api-ha-b"] = _healthz(secret_drift=True)
    errors = _preflight(pods=pods, healthz=healthz)
    assert any("secret_drift" in item for item in errors), _text(errors)


# --------------------------------------------------------------------------- #
# Registry (ARCH-H3)
# --------------------------------------------------------------------------- #
def test_the_registry_contract_passes_when_every_pod_agrees_with_the_secret() -> None:
    healthz = _healthz_by_pod(_pods())
    assert verdicts.registry_errors(healthz, label="x") == [], "intended run"


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        (_healthz(secret_drift=True), "secret_drift"),
        (_healthz(secret_drift=None), "secret_drift"),
        (_healthz(ready=False), "not ready"),
        (_healthz(secret_config_sha256=None), "secret_config_sha256"),
        ({"http_status": 503, "payload": {}}, "returned 503"),
    ],
)
def test_the_registry_contract_rejects_drift_unreadiness_or_a_refusal(
    payload: dict[str, Any], fragment: str
) -> None:
    errors = verdicts.registry_errors({"pod-a": payload}, label="after")
    assert any(fragment in item for item in errors), _text(errors)


def test_the_registry_contract_needs_at_least_one_pod() -> None:
    errors = verdicts.registry_errors({}, label="after")
    assert errors == ["after: no /healthz payload was read from any Pod"], errors


# --------------------------------------------------------------------------- #
# Timelines (ARCH-H1)
# --------------------------------------------------------------------------- #
def test_a_readiness_only_outage_that_recovers_in_the_stale_window_passes() -> None:
    errors = _sampler_errors(_samples())
    assert errors == [], _text(errors)


def test_a_timeline_with_no_readiness_outage_at_all_also_passes() -> None:
    errors = _sampler_errors(_samples(outage=(0, 0)))
    assert errors == [], _text(errors)


def test_a_single_livez_failure_fails_the_case() -> None:
    def mutate(samples: list[dict[str, Any]]) -> None:
        samples[20]["livez"] = 503

    errors = _sampler_errors(_samples(mutate=mutate))
    assert any("/livez answered 503" in item for item in errors), _text(errors)


def test_an_unreachable_livez_fails_the_case() -> None:
    def mutate(samples: list[dict[str, Any]]) -> None:
        samples[20]["livez"] = None
        samples[20]["error"] = "URLError: refused"

    errors = _sampler_errors(_samples(mutate=mutate))
    assert any("/livez answered None" in item for item in errors), _text(errors)


def test_a_healthz_500_is_a_crash_not_a_refusal() -> None:
    def mutate(samples: list[dict[str, Any]]) -> None:
        samples[10]["healthz"] = 500

    errors = _sampler_errors(_samples(mutate=mutate))
    assert any("/healthz answered 500" in item for item in errors), _text(errors)
    assert not any("/livez" in item for item in errors), _text(errors)


def test_an_unreachable_healthz_while_livez_answers_is_a_crash() -> None:
    def mutate(samples: list[dict[str, Any]]) -> None:
        samples[10]["healthz"] = None

    errors = _sampler_errors(_samples(mutate=mutate))
    assert any("/healthz was unreachable" in item for item in errors), _text(errors)


def test_readiness_that_does_not_return_within_the_stale_window_fails() -> None:
    # Outage from +15s to +200s: RDS is available at +55s, so recovery must
    # land by +55+90+30 = +175s; it does not.
    errors = _sampler_errors(_samples(outage=(15, 200)))
    assert any("was not 200 within 90s" in item for item in errors), _text(errors)


def test_readiness_that_returns_just_inside_the_window_passes() -> None:
    errors = _sampler_errors(_samples(outage=(15, 170)))
    assert errors == [], _text(errors)


def test_a_timeline_that_starts_late_or_stops_early_is_not_evidence() -> None:
    late = _sampler_errors(_samples(start=REQUESTED_AT + timedelta(seconds=5)))
    assert any("started after the failover" in item for item in late), _text(late)
    short = _sampler_errors(_samples(seconds=120))
    assert any("stopped before the readiness recovery" in item for item in short), (
        _text(short)
    )


def test_an_empty_timeline_is_not_evidence() -> None:
    assert _sampler_errors([]) == ["gpu-fault-api-ha-b: the probe recorded no samples"]


def test_the_final_sample_must_show_a_ready_registry_without_drift() -> None:
    def drift(samples: list[dict[str, Any]]) -> None:
        samples[-1]["secret_drift"] = True

    def unready(samples: list[dict[str, Any]]) -> None:
        samples[-1]["registry_ready"] = False

    assert any(
        "reports secret_drift" in item
        for item in _sampler_errors(_samples(mutate=drift))
    ), "drift"
    assert any(
        "ready registry" in item for item in _sampler_errors(_samples(mutate=unready))
    ), "unready"


def test_readiness_outage_and_recovery_are_measured_from_the_samples() -> None:
    samples = _samples()
    assert verdicts.readiness_outage_seconds(samples) == pytest.approx(64.0), (
        "32 two-second samples answered 503 (offsets +16s .. +78s)"
    )
    recovered = verdicts.readiness_recovery_at(samples, rds_available_at=AVAILABLE_AT)
    assert recovered == (T0 + timedelta(seconds=80)).timestamp(), recovered
    assert verdicts.readiness_outage_seconds([]) == 0.0, "no samples, no outage"


# --------------------------------------------------------------------------- #
# Pods
# --------------------------------------------------------------------------- #
def _after(deleted: str = DELETED, **overrides: Any) -> dict[str, list[dict[str, Any]]]:
    after = _pods()
    after["gpu-fault-api-ha"] = [
        item for item in after["gpu-fault-api-ha"] if item["name"] != deleted
    ] + [_pod("gpu-fault-api-ha-d")]
    for deployment, records in overrides.items():
        after[deployment] = records
    return after


def test_no_restart_passes_when_only_the_deleted_pod_was_replaced() -> None:
    errors = verdicts.restart_errors(_pods(), _after(), deleted_pod=DELETED)
    assert errors == [], _text(errors)


def test_a_moved_restart_count_fails_the_case() -> None:
    after = _after(
        **{"gpu-fault-control-worker": [_pod("gpu-fault-control-worker-a", restarts=1)]}
    )
    errors = verdicts.restart_errors(_pods(), after, deleted_pod=DELETED)
    assert any("restartCount moved by 1" in item for item in errors), _text(errors)


def test_a_recreated_or_missing_pod_fails_the_case() -> None:
    recreated = _after(
        **{
            "gpu-fault-telemetry-spool-worker": [
                _pod("gpu-fault-telemetry-spool-worker-a", uid="other", port=8082)
            ]
        }
    )
    errors = verdicts.restart_errors(_pods(), recreated, deleted_pod=DELETED)
    assert any("was recreated" in item for item in errors), _text(errors)
    missing = _after(**{"gpu-fault-control-worker": []})
    errors = verdicts.restart_errors(_pods(), missing, deleted_pod=DELETED)
    assert any("disappeared" in item for item in errors), _text(errors)


def test_the_deleted_pod_must_actually_be_gone_and_the_others_ready() -> None:
    lingering = _pods()
    lingering["gpu-fault-api-ha"].append(_pod("gpu-fault-api-ha-d"))
    errors = verdicts.restart_errors(_pods(), lingering, deleted_pod=DELETED)
    assert any("still exists" in item for item in errors), _text(errors)
    unready = _after(
        **{
            "gpu-fault-control-worker": [
                _pod("gpu-fault-control-worker-a", ready=False)
            ]
        }
    )
    errors = verdicts.restart_errors(_pods(), unready, deleted_pod=DELETED)
    assert any("not Ready afterwards" in item for item in errors), _text(errors)


def _replacement_errors(replacement: dict[str, Any]) -> list[str]:
    return verdicts.replacement_errors(
        replacement, deleted_pod=DELETED, deleted_at=REQUESTED_AT, budget_seconds=300.0
    )


def test_a_replacement_that_waited_for_aurora_passes() -> None:
    errors = _replacement_errors(_replacement())
    assert errors == [], _text(errors)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"waiting_reasons": ["CrashLoopBackOff"]}, "CrashLoopBackOff"),
        ({"restarts": 2}, "restarted 2 times"),
        ({"ready": False}, "never became Ready"),
        ({"ready_at": None}, "no ready_at"),
        (
            {"ready_at": (REQUESTED_AT + timedelta(seconds=400)).isoformat()},
            "budget is 300s",
        ),
        ({"name": DELETED}, "distinct from the deleted"),
        ({"name": ""}, "distinct from the deleted"),
    ],
)
def test_a_replacement_that_crashed_restarted_or_was_late_fails(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = _replacement_errors(_replacement(**overrides))
    assert any(fragment in item for item in errors), _text(errors)


def test_the_startup_retry_line_is_recognised_but_informational() -> None:
    lines = [
        "INFO starting",
        "WARNING regional registry bootstrap attempt 2 failed (OperationalError: x); "
        "retrying in 1.0s with 118s of start-up budget left",
    ]
    assert verdicts.startup_retry_observed(lines) is True, "backoff line"
    assert verdicts.startup_retry_observed(["INFO ready"]) is False, "no line"
    relevant = verdicts.relevant_log_lines("\n".join(lines))
    assert relevant == [lines[1]], relevant


def test_live_budgets_fall_back_to_the_shipped_defaults_and_refuse_garbage() -> None:
    assert verdicts.positive_seconds(None, default=90.0, label="x") == 90.0, "unset"
    assert verdicts.positive_seconds("", default=90.0, label="x") == 90.0, "blank"
    assert verdicts.positive_seconds("120", default=90.0, label="x") == 120.0, "set"
    with pytest.raises(ValueError, match="positive"):
        verdicts.positive_seconds("0", default=90.0, label="x")
    with pytest.raises(ValueError, match="not a number"):
        verdicts.positive_seconds("soon", default=90.0, label="x")


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def test_the_probe_flattens_a_healthz_body_into_the_verdict_record() -> None:
    record = probe.summarize_sample(
        moment=1.0,
        livez=200,
        healthz=503,
        healthz_body={
            "status": "unhealthy",
            "regional_registry": {"ready": False, "secret_drift": False},
        },
        error=None,
    )
    assert record == {
        "t": 1.0,
        "livez": 200,
        "healthz": 503,
        "registry_ready": False,
        "secret_drift": False,
        "status": "unhealthy",
        "error": None,
    }, record
    unreachable = probe.summarize_sample(
        moment=2.0, livez=None, healthz=None, healthz_body=None, error="URLError"
    )
    assert unreachable["registry_ready"] is None and unreachable["error"], unreachable


def test_the_runner_reads_the_http_port_and_container_state_from_a_pod() -> None:
    pod = {
        "metadata": {"name": "p", "uid": "u", "creationTimestamp": T0.isoformat()},
        "spec": {
            "nodeName": "n",
            "containers": [
                {"ports": [{"name": "metrics", "containerPort": 9000}]},
                {"ports": [{"name": "http", "containerPort": 8081}]},
            ],
        },
        "status": {
            "phase": "Running",
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "lastTransitionTime": T0.isoformat(),
                }
            ],
            "containerStatuses": [
                {
                    "ready": True,
                    "restartCount": 1,
                    "state": {"running": {}},
                    "lastState": {"waiting": {"reason": "CrashLoopBackOff"}},
                }
            ],
        },
    }
    record = ha010.pod_record(pod)
    assert record["port"] == 8081, record
    assert record["restarts"] == 1 and record["ready"] is True, record
    assert record["waiting_reasons"] == ["CrashLoopBackOff"], record
    assert record["ready_at"] == T0.isoformat(), record


# --------------------------------------------------------------------------- #
# Runner guard
# --------------------------------------------------------------------------- #
def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / ha010.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    preflight = {
        "release_id": "rel-1",
        "rds": _rds(),
        "pods": _pods(),
        "budgets": {"stale_seconds": 90.0, "startup_retry_seconds": 120.0},
    }
    with pytest.raises(Exception, match="plan drifted"):
        ha010.verify_plan_identity(case_dir, preflight)


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = ha010.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    assert plan.observe_seconds == verdicts.OBSERVE_SECONDS_DEFAULT, (
        plan.observe_seconds
    )
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            ha010.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--rds-cluster-id",
            "aurora-a",
        ]
    )
    assert execute.execute is True and execute.confirm == ha010.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def test_configure_bounds_the_observation_window(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    base = [
        "--run-dir",
        str(tmp_path),
        "--cpu-kubeconfig",
        str(cpu),
        "--gpu-kubeconfig",
        str(gpu),
        "--gpu-context",
        "ctx",
        "--cluster-id",
        "cluster-a",
        "--region",
        "us-west-2",
        "--rds-cluster-id",
        "aurora-a",
    ]
    with pytest.raises(Exception, match="observe seconds"):
        ha010.configure(ha010.parser().parse_args([*base, "--observe-seconds", "30"]))
    settings = ha010.configure(ha010.parser().parse_args(base))
    assert settings.environment()["GPU_FAULT_AURORA_CLUSTER_ID"] == "aurora-a", (
        settings.environment()
    )
    assert (
        settings.predecessor_path
        == (
            tmp_path / "cases" / ha010.PREDECESSOR_CASE_ID / "GF-REGIONAL-HA-001.json"
        ).resolve()
    ), settings.predecessor_path


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = ha010.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_the_runner_and_probe_are_executable_with_a_shebang_and_no_topology() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_ha010_aurora_blackout_liveness.py",
        ROOT / "scripts/e2e/regional/probes/ha010_probe.py",
        ROOT / "scripts/e2e/regional/ha010_verdicts.py",
    ):
        source = path.read_text(encoding="utf-8")
        for topology in (
            "/secure/gpu-fault-bootstrap",
            "514385905925",
            "gpu-fault-gpu-1-",
        ):
            assert topology not in source, (path.name, topology)
        if path.name.endswith("_verdicts.py"):
            continue
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        first = source.splitlines()[0]
        assert first == "#!/usr/bin/env python3", first


def test_the_failover_step_unpacks_ha003s_document_and_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``wait_rds_failover`` returns ``(document, samples)``; the first live run
    merged the tuple as a mapping and died with the failover requested and the
    Pod deleted. The step now records both."""
    from types import SimpleNamespace

    from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010

    calls: list[str] = []
    monkeypatch.setattr(
        ha010, "_quiet_control_plane", lambda run: calls.append("quiet")
    )
    monkeypatch.setattr(
        ha010.ha003, "aws_rds", lambda *_a, **_k: {"DBClusterIdentifier": "c"}
    )
    monkeypatch.setattr(
        ha010.ha003,
        "wait_rds_failover",
        lambda *_a, **_k: ({"status": "available", "writer": "db-2"}, [{"t": 1}]),
    )
    monkeypatch.setattr(
        ha010, "_first_ready_pod", lambda *_a, **_k: "gpu-fault-api-ha-a"
    )
    run = SimpleNamespace(
        settings=SimpleNamespace(
            rds_cluster_id="c", regional=SimpleNamespace(region="r")
        ),
        case_dir=tmp_path,
        preflight={"rds": {"writer": "db-1"}, "pods": {}},
        regional=SimpleNamespace(kubectl=lambda *a, **_k: calls.append(a[1]) or ""),
        failover_requested_at=None,
        deleted_pod=None,
        deleted_at=None,
        rds_available_at=None,
    )

    rds_after = ha010.request_failover(run)

    assert rds_after == {"status": "available", "writer": "db-2"}
    assert calls == ["quiet", "delete"], calls
    after = json.loads((tmp_path / "rds-after.json").read_text())
    assert after["writer"] == "db-2" and "available_at" in after, after
    samples = json.loads((tmp_path / "rds-failover-samples.json").read_text())
    assert samples == {"samples": [{"t": 1}]}


def test_a_sampler_fed_on_stdin_can_still_be_collected() -> None:
    """communicate() flushes a stdin attribute that is set; a pipe closed by
    hand is a closed file and the first live collection died on it."""
    import subprocess
    from datetime import datetime, timezone

    from scripts.e2e.regional import run_ha010_aurora_blackout_liveness as ha010

    process = subprocess.Popen(
        [sys.executable, "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ha010.feed_script(process, 'import json\nprint(json.dumps({"samples": [1, 2]}))\n')
    assert process.stdin is None, "the fed pipe must be detached"

    report = ha010.Sampler(
        pod="p", process=process, started_at=datetime.now(timezone.utc)
    ).collect(30)

    assert report["returncode"] == 0, report
    assert report["samples"] == [1, 2], report
