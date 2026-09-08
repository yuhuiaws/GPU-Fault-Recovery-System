"""The Aurora credential refresh runs synchronously, fail closed, before every
release transaction and every rollback.

RDS rotates the managed master password every 7 days; the only thing that
copies AWSCURRENT into the ``gpu-fault-aurora`` Secret is the
``gpu-fault-aurora-credential-refresh`` CronJob. Between a rotation and its
next tick every new control-plane Pod dies on ``password authentication
failed`` -- observed live on 2026-09-07, when a release re-apply hit that window,
the automatic rollback restarted control-worker into the same wall, and the
transaction landed in ``rollback-failed`` for a reason indistinguishable from a
broken release until someone read Pod logs.

Two fixes, both pinned here: the CronJob fires at least hourly, and the
orchestrator runs the refresh as a preflight (``create job --from=cronjob``,
wait, delete) that raises when the Job does not complete, so neither an
upgrade nor a rollback starts on a stale password.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault_release import regional_aurora_credentials as REFRESH
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "deploy/control-plane/regional/aurora-credential-refresh.yaml"
CRONJOB = "gpu-fault-aurora-credential-refresh"
NAMESPACE = "gpu-fault-system"


class RecordingRunner:
    """Records every command; ``probe`` answers whether the CronJob exists and
    ``fail_on`` names a kubectl verb whose command raises like the real runner."""

    dry_run = False

    def __init__(self, *, cronjob_exists: bool = True, fail_on: str | None = None):
        self.cronjob_exists = cronjob_exists
        self.fail_on = fail_on
        self.commands: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.probes: list[list[str]] = []

    def probe(self, args, **_kwargs) -> bool:
        self.probes.append(list(args))
        return self.cronjob_exists

    def run(self, args, **kwargs) -> str:
        self.commands.append(list(args))
        self.kwargs.append(dict(kwargs))
        verb = args[args.index("-n") + 2]
        if verb == self.fail_on:
            raise MODULE.ReleaseError(f"command failed (1): {args[0]}")
        return ""


def _release(runner: RecordingRunner) -> SimpleNamespace:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE, cpu_kubeconfig="/secure/cpu"),
        _cpu=lambda *arguments: ["kubectl", "--kubeconfig", "/secure/cpu", *arguments],
    )


def _verbs(runner: RecordingRunner) -> list[str]:
    return [args[args.index("-n") + 2] for args in runner.commands]


# --- the hook itself ---------------------------------------------------------


def test_the_release_object_binds_the_hook_like_the_ca_bundle() -> None:
    assert (
        MODULE.RegionalRelease._refresh_aurora_credentials
        is REFRESH.refresh_aurora_credentials
    )


def test_an_absent_cronjob_is_skipped_without_creating_anything() -> None:
    """Legacy sites never installed the CronJob; a preflight that failed there
    would block every release for a refresher that does not exist."""

    runner = RecordingRunner(cronjob_exists=False)

    note = REFRESH.refresh_aurora_credentials(_release(runner))

    assert note["status"] == "skipped"
    assert runner.probes == [
        [
            "kubectl",
            "--kubeconfig",
            "/secure/cpu",
            "-n",
            NAMESPACE,
            "get",
            "cronjob",
            CRONJOB,
        ]
    ]
    assert runner.commands == [], "nothing may be created when the CronJob is absent"


def test_a_present_cronjob_is_run_once_waited_for_and_deleted(monkeypatch) -> None:
    monkeypatch.delenv(REFRESH.REFRESH_WAIT_SECONDS_ENV, raising=False)
    runner = RecordingRunner()

    note = REFRESH.refresh_aurora_credentials(_release(runner))

    assert _verbs(runner) == ["create", "wait", "delete"]
    create, wait, delete = runner.commands
    job = note["job"]
    assert note["status"] == "refreshed"
    assert job.startswith(CRONJOB + "-") and len(job) > len(CRONJOB) + 1
    assert len(job) <= 63, "a Job name must stay a valid DNS-1123 label"
    assert create[create.index("-n") + 1] == NAMESPACE
    assert create[-3:] == ["job", f"--from=cronjob/{CRONJOB}", job]
    assert wait[-3:] == [
        "--for=condition=complete",
        f"--timeout={REFRESH.DEFAULT_REFRESH_WAIT_SECONDS}s",
        f"job/{job}",
    ]
    # The runner's own timeout backstops a hung kubectl and must outlast the
    # server-side wait so the Job, not the client, decides the verdict.
    assert runner.kwargs[1]["timeout_seconds"] > REFRESH.DEFAULT_REFRESH_WAIT_SECONDS
    assert delete[-3:] == ["job", job, "--ignore-not-found"]


def test_two_runs_never_reuse_a_job_name() -> None:
    first = RecordingRunner()
    second = RecordingRunner()

    a = REFRESH.refresh_aurora_credentials(_release(first))
    b = REFRESH.refresh_aurora_credentials(_release(second))

    assert a["job"] != b["job"], (
        "a fixed name would collide with a Job the previous run left for diagnosis"
    )


def test_a_job_that_does_not_complete_fails_closed_and_is_left_for_diagnosis(
    monkeypatch,
) -> None:
    monkeypatch.delenv(REFRESH.REFRESH_WAIT_SECONDS_ENV, raising=False)
    runner = RecordingRunner(fail_on="wait")

    with pytest.raises(MODULE.ReleaseError) as excinfo:
        REFRESH.refresh_aurora_credentials(_release(runner))

    job = runner.commands[0][-1]
    message = str(excinfo.value)
    assert job in message, "the operator must be told which Job to read"
    assert f"kubectl -n {NAMESPACE} logs job/{job}" in message
    assert "password" in message.lower()
    assert _verbs(runner) == ["create", "wait"], (
        "a Job that did not complete must be left in place, never deleted"
    )


def test_the_wait_is_an_env_override_and_refuses_garbage(monkeypatch) -> None:
    monkeypatch.setenv(REFRESH.REFRESH_WAIT_SECONDS_ENV, "45")
    runner = RecordingRunner()

    REFRESH.refresh_aurora_credentials(_release(runner))

    assert "--timeout=45s" in runner.commands[1]

    for garbage in ("0", "-5", "soon"):
        monkeypatch.setenv(REFRESH.REFRESH_WAIT_SECONDS_ENV, garbage)
        with pytest.raises(MODULE.ReleaseError, match=REFRESH.REFRESH_WAIT_SECONDS_ENV):
            REFRESH.refresh_aurora_credentials(_release(RecordingRunner()))


# --- where the orchestrator calls it ------------------------------------------


class _Reached(RuntimeError):
    pass


def test_upgrade_refreshes_credentials_before_touching_anything() -> None:
    """After the CA bundle (the refresher mounts it), before the idle check and
    before the previous release is captured: a stale password must stop the
    transaction before it has anything to roll back."""

    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        state={},
        _ensure_contexts=lambda: calls.append("contexts"),
        _require_cpu_secrets=lambda: calls.append("cpu-secrets"),
        _apply_rds_ca_bundle=lambda: calls.append("rds-ca-bundle"),
        _refresh_aurora_credentials=lambda: calls.append("aurora-refresh"),
        _remote_commands_are_idle=lambda: (calls.append("idle"), True)[1],
        _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(_Reached()),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(_Reached):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert calls == [
        "contexts",
        "cpu-secrets",
        "rds-ca-bundle",
        "aurora-refresh",
        "idle",
    ]


def test_a_failed_refresh_stops_the_upgrade_before_the_idle_check() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        state={},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _apply_rds_ca_bundle=lambda: None,
        _refresh_aurora_credentials=lambda: (_ for _ in ()).throw(
            MODULE.ReleaseError("Aurora credential refresh Job x did not complete")
        ),
        _remote_commands_are_idle=lambda: pytest.fail(
            "the transaction went on after the credential refresh failed"
        ),
        _capture_previous=lambda **_kwargs: pytest.fail("previous was captured"),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    with pytest.raises(MODULE.ReleaseError, match="credential refresh"):
        ORCHESTRATION.upgrade_release(release, diff=diff)
    assert release.state == {}, "no state may be written before the refresh passes"


def _rollback_double(calls: list[str], *, refresh=None, **stubs) -> SimpleNamespace:
    """A release double for ``rollback_release``; ``stubs`` adds the helpers a
    particular rollback path calls beyond the refresh and the state save."""

    return SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        state={},
        _refresh_aurora_credentials=refresh or (lambda: calls.append("aurora-refresh")),
        _save_state=lambda phase, **_updates: calls.append(f"state:{phase}"),
        **stubs,
    )


def test_rollback_refreshes_credentials_before_planning_any_restore(
    monkeypatch,
) -> None:
    """The compensating rollback restarts control-worker; on 2026-09-07 it did so
    into the rotated password and turned a recoverable failure into
    ``rollback-failed``. The refresh has to come before the compensation plan,
    which is what every restore phase is derived from."""

    calls: list[str] = []

    def plan(*_args, **_kwargs):
        calls.append("compensation-plan")
        raise _Reached()

    monkeypatch.setattr(ORCHESTRATION, "build_rollback_compensation_plan", plan)

    with pytest.raises(_Reached):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls), state={"metadata": {}, "cpu_wheel": "w"}
        )

    assert calls == ["aurora-refresh", "compensation-plan"]


def test_a_failed_refresh_stops_the_rollback_before_any_restore(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        ORCHESTRATION,
        "build_rollback_compensation_plan",
        lambda *_a, **_k: pytest.fail("the rollback was planned on a stale password"),
    )

    def refresh():
        raise MODULE.ReleaseError("Aurora credential refresh Job x did not complete")

    with pytest.raises(MODULE.ReleaseError, match="credential refresh"):
        ORCHESTRATION.rollback_release(
            _rollback_double(calls, refresh=refresh), state={"metadata": {}}
        )
    assert calls == []


def test_a_rollback_with_nothing_to_restore_does_not_refresh() -> None:
    """``_rollback_context`` returns early for a bootstrap that never captured a
    previous release; there is no Deployment to restart, so no Job either."""

    calls: list[str] = []
    release = _rollback_double(
        calls,
        _cleanup_bootstrap=lambda: calls.append("cleanup-bootstrap"),
        _load_state=lambda: {"phase": "bootstrap-failed"},
    )

    ORCHESTRATION.rollback_release(release, state=None)

    assert calls == ["cleanup-bootstrap"]


# --- the CronJob fires at least hourly ---------------------------------------


def _cron_fire_minutes(schedule: str) -> list[int]:
    """Minutes-of-day at which a ``M H * * *`` schedule fires."""

    minute_field, hour_field, dom, month, dow = schedule.split()
    assert (dom, month, dow) == ("*", "*", "*"), schedule
    assert minute_field.isdigit(), f"one fixed minute expected, got {minute_field!r}"
    minute = int(minute_field)
    if hour_field == "*":
        hours = range(24)
    elif hour_field.startswith("*/"):
        hours = range(0, 24, int(hour_field[2:]))
    else:
        hours = [int(item) for item in hour_field.split(",")]
    return sorted(hour * 60 + minute for hour in hours)


def test_the_cronjob_fires_at_least_hourly() -> None:
    documents = [d for d in yaml.safe_load_all(MANIFEST.read_text("utf-8")) if d]
    cronjob = next(d for d in documents if d.get("kind") == "CronJob")
    assert cronjob["metadata"]["name"] == CRONJOB
    spec = cronjob["spec"]

    fires = _cron_fire_minutes(spec["schedule"])
    gaps = [b - a for a, b in zip(fires, fires[1:])]
    gaps.append(fires[0] + 24 * 60 - fires[-1])  # wrap past midnight
    assert max(gaps) <= 60, (
        f"schedule {spec['schedule']!r} leaves a {max(gaps)}-minute window in "
        "which a rotated password is not yet in the Secret; every new "
        "control-plane Pod dies in that window"
    )
    # A late start must not run into the next tick under concurrencyPolicy Forbid.
    assert spec["concurrencyPolicy"] == "Forbid"
    assert int(spec["startingDeadlineSeconds"]) < 3600
    assert int(spec["jobTemplate"]["spec"]["activeDeadlineSeconds"]) <= 3600


def test_the_manifest_explains_the_stale_password_window() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    assert "17 5,17" not in text, "the twice-daily schedule must not come back"
    assert "password authentication failed" in text
    assert "shorter than the rotation period" in text
