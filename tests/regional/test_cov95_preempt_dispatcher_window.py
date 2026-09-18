from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_preempt037_dispatcher_liveness as runner
from tests.regional._cov95_preempt_dispatcher import DispatcherModel, V


@pytest.mark.parametrize("layout", ["absent-env", "other-variable", "explicit-true"])
def test_dispatcher_window_restores_each_original_environment_shape(
    layout, tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    if layout == "other-variable":
        model.container["env"] = [{"name": "UNRELATED", "value": "preserved"}]
    elif layout == "explicit-true":
        model.container["env"] = [{"name": V.VARIABLE, "value": "true"}]
    original = list(model.container.get("env", []))
    result = model.execute(tmp_path)
    assert result["verdict"] == "PASS", "bounded fake stall and restore must pass"
    assert model.container.get("env", []) == original, "restore the literal environment"
    assert model.events == ["arm", "disarm"], "disarm only after restoration proof"
    assert len(model.patches) == 2, "only opening and closing may patch the Deployment"
    assert result["env_window"]["watchdog_retained"] is False, (
        "completed restoration must release the watchdog"
    )
    assert (
        result["stall_timeline"][-1]["observed_epoch"]
        - next(
            sample["observed_epoch"]
            for sample in result["stall_timeline"]
            if sample["stalled"]
        )
        >= 300
    ), "the stalled expression must hold for the full rule duration"


@pytest.mark.parametrize(
    ("failure", "message", "patches", "events"),
    [
        ("busy-before", "still in flight", 0, []),
        ("busy-after", "still in flight", 0, ["arm", "disarm"]),
        ("busy-during", "still in flight", 2, ["arm", "disarm"]),
        ("arm", "admission failed", 0, ["arm", "disarm"]),
        ("effective-disabled", "dispatch enabled", 0, []),
        ("already-stalled", "already stalled", 0, []),
        ("open-ack-loss", "acknowledgement lost", 2, ["arm", "disarm"]),
    ],
)
def test_stall_lifecycle_fails_closed_and_restores_if_window_may_exist(
    failure, message, patches, events, tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    model.failure = failure
    with pytest.raises((runner.RegionalFixtureError, RuntimeError), match=message):
        model.execute(tmp_path)
    assert len(model.patches) == patches, (
        "only a possibly opened window needs restoration"
    )
    assert model.events == events, "watchdog ownership must follow the completed stages"
    assert model.value() == "true", (
        "failed orchestration must not leave dispatch disabled"
    )


@pytest.mark.parametrize(
    ("failure", "stage", "verdict"),
    [
        ("never-stall", "stall", "FAIL"),
        ("no-recovery", "recovery", "FAIL"),
        ("restore-drift", "restore", "FAIL"),
        ("slow-recovery", "recovery", "PASS"),
    ],
)
def test_failed_or_delayed_observations_control_the_case_verdict(
    failure, stage, verdict, tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    model.failure = failure
    result = model.execute(tmp_path)
    assert result["verdict"] == verdict, "published status must follow observed stages"
    assert bool(result["stages"][stage]) is (verdict == "FAIL"), (
        "the failed stage must retain its specific evidence gap"
    )
    retained = failure == "restore-drift"
    assert result["env_window"]["watchdog_retained"] is retained, (
        "unproved restoration must retain the independent watchdog"
    )
    assert model.events == (["arm"] if retained else ["arm", "disarm"]), (
        "watchdog cleanup must agree with the recorded restoration proof"
    )


@pytest.mark.parametrize("reason", ["baseline", "deadline", "rule"])
def test_unsafe_window_is_refused_before_watchdog_or_mutation(
    reason, tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    deadline = datetime.fromtimestamp(model.now + 5_000, timezone.utc)
    if reason == "baseline":
        model.container["env"] = [{"name": V.VARIABLE, "value": "false"}]
    elif reason == "deadline":
        deadline = datetime.fromtimestamp(model.now + 1, timezone.utc)
    else:
        runner.RUNBOOK.write_text("missing anchor")
    with pytest.raises(
        runner.RegionalFixtureError, match="already|cannot hold|runbook"
    ):
        runner.execute(model, tmp_path, deadline)
    assert model.events == model.patches == [], (
        "unsafe preflight must not arm or mutate"
    )


@pytest.mark.parametrize(
    "shape", ["missing-container", "duplicate", "reference", "uid"]
)
def test_dispatcher_snapshot_refuses_ambiguous_container_or_setting(
    shape, tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    if shape == "missing-container":
        model.container["name"] = "different"
    elif shape == "duplicate":
        model.container["env"] = [{"name": V.VARIABLE, "value": "true"}] * 2
    elif shape == "reference":
        model.container["env"] = [{"name": V.VARIABLE, "valueFrom": {}}]
    else:
        model.document["metadata"]["uid"] = ""
    with pytest.raises(runner.RegionalFixtureError, match="has no|reference|identity"):
        runner.deployment_variable(model)
    assert model.patches == [], "ambiguous inventory cannot authorize a patch"


def test_assignment_allowlist_and_absent_restore_do_not_write(
    tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    baseline = runner.deployment_variable(model)
    with pytest.raises(runner.RegionalFixtureError, match="unsupported"):
        runner.set_variable(model, "UNRELATED=false", baseline=baseline)
    assert runner.set_variable(model, f"{V.VARIABLE}-", baseline=baseline) == "ready", (
        "an already restored Deployment must still be checked for rollout readiness"
    )
    assert model.patches == [], "restoring an absent value must not modify the template"


@pytest.mark.parametrize("settles", [False, True])
def test_replica_wait_repolls_mismatch_with_a_finite_deadline(
    settles, monkeypatch
) -> None:
    state = SimpleNamespace(now=0.0, reads=0)

    def replicas(_regional):
        state.reads += 1
        return [
            {
                "pod": "worker",
                "values": {
                    V.VARIABLE: "true" if settles and state.reads > 1 else "false"
                },
            }
        ]

    monkeypatch.setattr(runner, "replicas", replicas)
    monkeypatch.setattr(runner, "ROLLOUT_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(
            monotonic=lambda: state.now,
            sleep=lambda seconds: setattr(state, "now", state.now + seconds),
        ),
    )
    if settles:
        result = runner.wait_replicas(None, "true")
        assert result[0]["values"][V.VARIABLE] == "true", (
            "the wait must return the converged replica observation"
        )
    else:
        with pytest.raises(runner.RegionalFixtureError, match="did not all read"):
            runner.wait_replicas(None, "true")
    assert state.reads == 2, "replica mismatch must not wait indefinitely"


def test_retained_watchdog_evidence_does_not_claim_closed(
    tmp_path, monkeypatch
) -> None:
    model = DispatcherModel(tmp_path, monkeypatch)
    model.failure = "restore-drift"
    model.execute(tmp_path)
    record = json.loads((tmp_path / "env-window-baseline.json").read_text())
    assert record["watchdog_retained"] is True, (
        "the independent restore channel must remain"
    )
    assert "closed_at" not in record, "drifted restore cannot be recorded as closed"
