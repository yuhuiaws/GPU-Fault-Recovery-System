"""Sterile regressions for the fixed-main COLLECT-018 census snapshot wait."""

from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from gpu_fault.admin import deadlines
from scripts.e2e.regional import collect018_verdicts as verdicts
from scripts.e2e.regional import collector_action_guard as guard
from scripts.e2e.regional import run_collect018_rejected_event as runner
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.collector_window_fixture import (
    CollectorWindowFixture,
    WindowSettings,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_collect_net import Clock as BaseClock
from tests.regional._cov95_collect_net import (  # noqa: F401
    no_external_effects,
)
from tests.regional._cov95_collect_windows import WindowHost

CLUSTER = "cluster-a"
NODE = "node-a"
LABELS = f'cluster_id="{CLUSTER}",channel="{verdicts.KERNEL_CHANNEL}"'


class Clock(BaseClock):
    def __init__(self) -> None:
        super().__init__(now=0)
        self.started_at = datetime.now(timezone.utc)

    def utc_now(self) -> datetime:
        return self.started_at + timedelta(seconds=self.now)


def metrics(
    *,
    erroring: str = "0",
    silent: str = "0",
    top: str | None = None,
    counter: str = "0",
    unresolved: str = "0",
) -> list[str]:
    lines = [
        f"{verdicts.FAULT_REJECTIONS_METRIC} {counter}",
        f'{verdicts.COMPLETIONS_METRIC}{{path="{verdicts.KERNEL_PATH}",status_class="4xx"}} {counter}',
        f"{verdicts.ERRORING_METRIC}{{{LABELS}}} {erroring}",
        f"{verdicts.SILENT_METRIC}{{{LABELS}}} {silent}",
        f'{verdicts.UNRESOLVED_METRIC}{{kind="{verdicts.UNPARSED_KIND}"}} {unresolved}',
    ]
    if top is not None:
        lines.append(f'{verdicts.SILENT_TOP_METRIC}{{{LABELS},node_id="{NODE}"}} {top}')
    return ["\n".join(lines)]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    value = Clock()
    monkeypatch.setattr(runner, "time", value)
    monkeypatch.setattr(runner, "utc_now", value.utc_now)
    monkeypatch.setattr(deadlines, "time", value)
    monkeypatch.setattr(guard, "time", value)
    monkeypatch.setattr(
        guard, "datetime", SimpleNamespace(now=lambda _tz: value.utc_now())
    )
    for name in (
        deadlines.DEADLINE_ENV,
        deadlines.DEADLINE_LABEL_ENV,
        deadlines.HARD_DEADLINE_ENV,
        deadlines.RECOVERY_ACTIVE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return value


def wait(fixture: Any, case_dir: Path, *, before: list[str] | None = None) -> list[str]:
    return runner.wait_silence_metrics(
        cast(CollectorWindowFixture, fixture),
        metrics_before=metrics() if before is None else before,
        cluster_id=CLUSTER,
        node=NODE,
        case_dir=case_dir,
    )


def timeline(case_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((case_dir / "erroring-gauge-timeline.json").read_text()),
    )


def test_census_converges_after_more_than_one_snapshot_interval(
    clock: Clock, tmp_path: Path
) -> None:
    settled = metrics(erroring="1")
    fixture = SimpleNamespace(
        control_plane_metrics=Mock(side_effect=[metrics()] * 7 + [settled])
    )

    assert wait(fixture, tmp_path) == settled, "return the actual converged sample"
    assert clock.now == 35, "the poll must tolerate a snapshot lag beyond 30 seconds"
    assert fixture.control_plane_metrics.call_count == 8, "one read per poll"
    proof = timeline(tmp_path)
    assert proof["settled"] is True, "convergence needs an affirmative receipt"
    assert all(row["errors"] for row in proof["entries"][:-1]), "record every lag"
    assert proof["entries"][-1]["errors"] == [], "all census checks must converge"
    assert deadlines.current_deadline() is None, "do not leak the wait budget"


@pytest.mark.parametrize("defect", ["erroring", "silent", "top"])
def test_timeout_never_does_a_final_read(
    clock: Clock, tmp_path: Path, defect: str
) -> None:
    pending = metrics(
        erroring="0" if defect == "erroring" else "1",
        silent="1" if defect == "silent" else "0",
        top="1" if defect == "top" else None,
    )
    fixture = SimpleNamespace(control_plane_metrics=Mock(return_value=pending))

    with pytest.raises(RegionalFixtureError, match="total deadline"):
        wait(fixture, tmp_path)

    assert clock.now == 120, "the snapshot timeout is an absolute total budget"
    assert fixture.control_plane_metrics.call_count == 24, "no read at or after 120s"
    assert clock.sleeps == [5] * 24, "sleeps belong to the same budget as reads"
    proof = timeline(tmp_path)
    assert proof["settled"] is False, "timeout must never leave a successful receipt"
    assert proof["entries"][-1]["errors"], "retain the last observed census failure"


@pytest.mark.parametrize("limiter", ["parent", "maintenance", "hard", "recovery"])
@pytest.mark.parametrize("expired", [False, True], ids=["short", "expired"])
def test_wait_obeys_every_existing_deadline_and_restores_its_scope(
    clock: Clock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limiter: str,
    expired: bool,
) -> None:
    if limiter in {"hard", "recovery"}:
        monkeypatch.setenv(deadlines.HARD_DEADLINE_ENV, "12")
    scope: Any = nullcontext()
    if limiter == "parent":
        scope = deadlines.deadline_scope("parent", 12)
    elif limiter == "maintenance":
        scope = guard.action_window(clock.utc_now() + timedelta(seconds=12))
    elif limiter == "recovery":
        scope = deadlines.recovery_deadline("existing recovery")
    fixture = SimpleNamespace(control_plane_metrics=Mock(return_value=metrics()))

    with scope:
        parent = deadlines.current_deadline()
        if expired:
            clock.now = 12
        with pytest.raises(RegionalFixtureError, match="deadline|maintenance|finite"):
            wait(fixture, tmp_path)
        assert deadlines.current_deadline() is parent, "restore the caller's budget"
        if limiter == "recovery":
            assert deadlines.recovery_active(), "do not exit the caller's recovery"

    assert clock.now == 12, "the 120s poll must not extend a shorter deadline"
    assert fixture.control_plane_metrics.call_count == (0 if expired else 3), (
        "never begin a read after the effective deadline"
    )
    assert clock.sleeps == ([] if expired else [5, 5, 2]), "cap the last sleep"
    assert timeline(tmp_path)["settled"] is False, "expired proof stays negative"


def test_slow_read_cannot_return_a_late_success_or_consume_cleanup_budget(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verdicts, "SILENCE_SNAPSHOT_TIMEOUT_SECONDS", 20)
    budgets: list[float] = []

    def read() -> list[str]:
        budgets.append(deadlines.remaining_timeout(60))
        clock.now += 8
        return metrics(erroring="0" if len(budgets) == 1 else "1")

    with deadlines.deadline_scope("parent", 100) as parent:
        with pytest.raises(RegionalFixtureError, match="total deadline"):
            wait(SimpleNamespace(control_plane_metrics=read), tmp_path)
        assert deadlines.current_deadline() is parent, "cleanup keeps its parent scope"
        assert parent.remaining() == 79, "do not replace the parent's absolute budget"

    assert budgets == [20, 7], "each command inherits the remaining total read budget"
    assert clock.sleeps == [5], "a late successful read must not start another poll"
    assert timeline(tmp_path)["settled"] is False, "a late read is not convergence"


def test_wall_clock_expiry_during_a_read_refuses_success(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = clock.utc_now() + timedelta(seconds=100)

    def read() -> list[str]:
        monkeypatch.setattr(guard, "datetime", SimpleNamespace(now=lambda _tz: end))
        return metrics(erroring="1")

    fixture = SimpleNamespace(control_plane_metrics=Mock(side_effect=read))
    with guard.action_window(end):
        with pytest.raises(RegionalFixtureError, match="maintenance"):
            wait(fixture, tmp_path)

    assert fixture.control_plane_metrics.call_count == 1, "do not retry after expiry"
    assert clock.sleeps == [], "a wall-clock deadline cannot be stretched by monotonic"
    assert timeline(tmp_path)["settled"] is False, "late success stays unproven"


def test_receipt_write_time_cannot_turn_expired_evidence_into_success(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_receipt(path: Path, value: dict[str, Any]) -> None:
        write_json_atomic(path, value)
        if value.get("settled"):
            clock.now = 120

    monkeypatch.setattr(runner, "write_json_atomic", slow_receipt)
    fixture = SimpleNamespace(
        control_plane_metrics=Mock(return_value=metrics(erroring="1"))
    )
    with pytest.raises(RegionalFixtureError, match="total deadline"):
        wait(fixture, tmp_path)
    assert fixture.control_plane_metrics.call_count == 1, (
        "never reread after receipt expiry"
    )
    assert clock.sleeps == [], "receipt expiry must stop the poll immediately"
    assert timeline(tmp_path)["settled"] is False, "replace the premature settled flag"


@pytest.mark.parametrize("family", ["erroring", "silent", "top"])
@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf", "-1", "0.5", "malformed"])
def test_invalid_census_samples_fail_immediately(
    clock: Clock, tmp_path: Path, family: str, value: str
) -> None:
    sample = metrics(**{"erroring": "1", family: value})
    fixture = SimpleNamespace(control_plane_metrics=Mock(return_value=sample))

    with pytest.raises(RegionalFixtureError, match="metric"):
        wait(fixture, tmp_path)

    assert fixture.control_plane_metrics.call_count == 1, "invalid data is not lag"
    assert clock.sleeps == [], "do not retry malformed or non-finite evidence"
    assert timeline(tmp_path)["settled"] is False, "invalid data must not authorize B"


@pytest.mark.parametrize(
    "defect",
    ["empty", "silent-missing", "erroring-missing", "foreign", "label", "duplicate"],
)
def test_missing_or_ambiguous_census_cannot_satisfy_the_wait(
    clock: Clock, tmp_path: Path, defect: str
) -> None:
    lines = metrics(erroring="1")[0].splitlines()
    if defect == "empty":
        lines = []
    elif defect.endswith("-missing"):
        name = (
            verdicts.SILENT_METRIC
            if defect == "silent-missing"
            else verdicts.ERRORING_METRIC
        )
        lines = [line for line in lines if not line.startswith(name)]
    elif defect == "foreign":
        lines = [line.replace(CLUSTER, "other-cluster") for line in lines]
    elif defect == "label":
        lines = [line.replace(f'cluster_id="{CLUSTER}",', "") for line in lines]
    else:
        lines.append(f"{verdicts.ERRORING_METRIC}{{{LABELS}}} 1")
    fixture = SimpleNamespace(
        control_plane_metrics=Mock(return_value=["\n".join(lines)])
    )

    with pytest.raises(RegionalFixtureError, match="metric"):
        wait(fixture, tmp_path)

    assert fixture.control_plane_metrics.call_count == 1, "no guessing missing census"
    assert clock.sleeps == [], "unknown evidence fails before another poll"


def test_missing_baseline_is_not_a_zero_silence_census(
    clock: Clock, tmp_path: Path
) -> None:
    fixture = SimpleNamespace(
        control_plane_metrics=Mock(return_value=metrics(erroring="1"))
    )
    with pytest.raises(RegionalFixtureError, match="missing"):
        wait(fixture, tmp_path, before=[])
    assert clock.sleeps == [], "an unknown baseline cannot prove silence did not rise"


def test_read_failure_is_not_retried_or_hidden_by_the_wait(
    clock: Clock, tmp_path: Path
) -> None:
    fixture = SimpleNamespace(
        control_plane_metrics=Mock(side_effect=RegionalFixtureError("read failed"))
    )
    with pytest.raises(RegionalFixtureError, match="read failed"):
        wait(fixture, tmp_path)
    assert fixture.control_plane_metrics.call_count == 1, "preserve the read failure"
    assert clock.sleeps == [], "do not treat an unknown read as a stale zero"
    assert timeline(tmp_path)["settled"] is False, "failure evidence stays negative"


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "malformed", "missing"])
def test_synchronous_counters_require_valid_evidence(phase: str, value: str) -> None:
    samples = {"before": metrics(), "after": metrics(counter="1")}
    samples[phase] = [] if value == "missing" else metrics(counter=value)
    with pytest.raises(RegionalFixtureError, match="metric"):
        verdicts.rejection_metric_errors(samples["before"], samples["after"])


def test_unobserved_4xx_baseline_requires_an_exposed_counter_family() -> None:
    before = [
        f"{verdicts.FAULT_REJECTIONS_METRIC} 0\n"
        f"# TYPE {verdicts.COMPLETIONS_METRIC} counter"
    ]
    assert verdicts.rejection_metric_errors(before, metrics(counter="1")) == [], (
        "a declared but not yet observed labelled counter legitimately starts at zero"
    )
    with pytest.raises(RegionalFixtureError, match="missing"):
        verdicts.rejection_metric_errors(
            [f"{verdicts.FAULT_REJECTIONS_METRIC} 0"], metrics(counter="1")
        )
    with pytest.raises(RegionalFixtureError, match="missing"):
        verdicts.rejection_metric_errors(metrics(), before)


@pytest.mark.parametrize("immediate_proof", [True, False], ids=["current", "late-only"])
def test_execute_keeps_the_counter_proof_separate_from_the_lagging_gauge(
    clock: Clock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, immediate_proof: bool
) -> None:
    host = WindowHost(runner)
    observed = Mock(
        side_effect=[
            metrics(),
            metrics(counter="1" if immediate_proof else "0"),
            metrics(counter="2"),
            metrics(counter="3", erroring="1"),
            metrics(counter="3", erroring="1", unresolved="1"),
        ]
    )
    monkeypatch.setattr(host, "control_plane_metrics", observed)
    end = clock.utc_now() + timedelta(hours=1)
    settings = cast(WindowSettings, host.settings)
    fixture = cast(CollectorWindowFixture, host)

    if immediate_proof:
        result = runner.execute(settings, fixture, tmp_path, 1, end)
        assert result["verdict"] == "PASS", "a delayed valid census should pass"
        assert result["stages"]["rejection_metrics"] == [], "keep immediate G1 proof"
        assert host.unparsed and host.recovered, "later phases still execute normally"
        assert observed.call_count == 5, "only the census proof uses the extra reads"
    else:
        with pytest.raises(RegionalFixtureError, match="rejection phase failed"):
            runner.execute(settings, fixture, tmp_path, 1, end)
        assert not host.unparsed and not host.recovered, (
            "late G1 counters cannot pass A"
        )
        assert observed.call_count == 4, "do not proceed to B or C after G1 failure"
        assert [call[0] for call in host.calls] == ["post-rejected-event"], (
            "the failed immediate proof must retain the existing injection stop"
        )
    assert clock.sleeps == [5], "only wait while the valid census is lagging"
