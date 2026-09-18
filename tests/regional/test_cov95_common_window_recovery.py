from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_common_windows import (
    WINDOWS,
    change_variable,
    window_fixture,
)


@pytest.mark.parametrize("module", WINDOWS)
def test_precommit_open_failure_can_retry_the_bound_baseline(
    module, tmp_path, monkeypatch
) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    transport = api.kubectl

    def reject_patch(plane, *args, **kwargs):
        if args[:2] == ("patch", "deployment"):
            raise RegionalFixtureError("conditional write unavailable")
        return transport(plane, *args, **kwargs)

    monkeypatch.setattr(api, "kubectl", reject_patch)
    with pytest.raises(RegionalFixtureError, match="conditional write"):
        module.open_window(settings, api, module.survey(api), assignments)
    saved = json.loads(settings.baseline.read_text())
    assert saved["state"] == "OPENING", "retry needs the original bound snapshot"
    assert api.patches == [], "the rejected write cannot have changed the fixture"
    monkeypatch.setattr(api, "kubectl", transport)
    opened = module.open_window(settings, api, module.survey(api), assignments)
    assert opened["state"] == "OPEN", "an uncommitted open must remain retryable"
    assert opened["baseline"] == saved["baseline"], "retry must keep the old env"
    assert len(api.patches) == 1, "only the successful attempt may patch"


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize("stage", ["before", "after"])
def test_convergence_rejects_uid_replacement_at_either_read(
    module, stage, tmp_path, monkeypatch
) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    snapshot = module.deployment_env(api)
    replaced = {**snapshot, "uid": "replacement"}
    responses = iter([replaced] if stage == "before" else [snapshot, replaced])
    monkeypatch.setattr(module, "deployment_env", lambda _api: next(responses))
    with pytest.raises(RegionalFixtureError, match="UID changed"):
        module.converge(
            settings,
            api,
            assignments,
            expected_uid=snapshot["uid"],
            sleep=lambda _: None,
        )
    assert api.patches == [], "convergence is observational, even during drift"


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize("settles", [False, True])
def test_convergence_repolls_incomplete_population_and_obeys_deadline(
    module, settles, tmp_path, monkeypatch
) -> None:
    api, settings, _assignments = window_fixture(module, tmp_path)
    api.visible_replicas = 1
    clock = SimpleNamespace(value=0.0)
    sleeps = []
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.value))

    def sleep(seconds):
        sleeps.append(seconds)
        clock.value += seconds
        if settles:
            api.visible_replicas = 2

    if settles:
        replicas = module.converge(settings, api, {}, sleep=sleep)
        assert len(replicas) == 2, "every intended replica must become observable"
    else:
        with pytest.raises(RegionalFixtureError, match="did not all report"):
            module.converge(settings, api, {}, sleep=sleep)
    assert sleeps == [5], "failed polls must be bounded rather than spin"


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize("drift", ["uid", "value"])
def test_open_cannot_claim_success_after_post_convergence_drift(
    module, drift, tmp_path, monkeypatch
) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    converge = module.converge

    def converge_then_drift(*args, **kwargs):
        result = converge(*args, **kwargs)
        if drift == "uid":
            api.deployment["metadata"]["uid"] = "replacement"
        else:
            change_variable(api, next(iter(assignments)), "foreign-value")
        return result

    monkeypatch.setattr(module, "converge", converge_then_drift)
    with pytest.raises(RegionalFixtureError, match="drifted after convergence"):
        module.open_window(settings, api, module.survey(api), assignments)
    saved = json.loads(settings.baseline.read_text())
    assert saved["state"] == "OPENING", "a drifted window cannot be marked OPEN"
    assert len(api.patches) == 1, "a changed owner must not be repaired implicitly"


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize("drift", ["uid", "value"])
def test_restore_drift_keeps_closing_record_without_completion(
    module, drift, tmp_path, monkeypatch
) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    module.open_window(settings, api, module.survey(api), assignments)
    transport = api.kubectl

    def rollout_then_drift(plane, *args, **kwargs):
        result = transport(plane, *args, **kwargs)
        if args[:2] == ("rollout", "status"):
            if drift == "uid":
                api.deployment["metadata"]["uid"] = "replacement"
            else:
                change_variable(api, next(iter(assignments)), "foreign-value")
        return result

    monkeypatch.setattr(api, "kubectl", rollout_then_drift)
    with pytest.raises(RegionalFixtureError, match="recorded baseline"):
        module.close_window(settings, api, module.survey(api))
    saved = json.loads(settings.baseline.read_text())
    assert saved["state"] == "CLOSING", "failed restoration must stay incomplete"
    assert "closed_at" not in saved, "drift must never be reported as restored"


@pytest.mark.parametrize("module", WINDOWS)
def test_completed_close_is_idempotent_but_reopened_values_are_drift(
    module, tmp_path
) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    opened = module.open_window(settings, api, module.survey(api), assignments)
    closed = module.close_window(settings, api, module.survey(api))
    repeated = module.close_window(settings, api, module.survey(api))
    assert repeated["closed_at"] == closed["closed_at"], "retain first completion time"
    assert len(api.patches) == 2, "an already restored window needs no extra mutation"
    for name, value in opened["assignments"].items():
        change_variable(api, name, value)
    with pytest.raises(RegionalFixtureError, match="closed .* window has drifted"):
        module.close_window(settings, api, module.survey(api))
    assert len(api.patches) == 2, "a closed record no longer owns these values"


@pytest.mark.parametrize("module", WINDOWS)
def test_changed_assignments_cannot_resume_an_open_record(module, tmp_path) -> None:
    api, settings, assignments = window_fixture(module, tmp_path)
    module.open_window(settings, api, module.survey(api), assignments)
    changed = copy.deepcopy(assignments)
    changed[next(iter(changed))] = "200"
    with pytest.raises(RegionalFixtureError, match="refusing to open"):
        module.open_window(settings, api, module.survey(api), changed)
    assert len(api.patches) == 1, "resume cannot silently choose a new timing window"


@pytest.mark.parametrize("module", WINDOWS)
def test_missing_baseline_is_not_an_empty_restore(module, tmp_path) -> None:
    api, settings, _assignments = window_fixture(module, tmp_path)
    with pytest.raises(RegionalFixtureError, match="no .* baseline"):
        module.close_window(settings, api, module.survey(api))
    assert api.patches == [], "unknown original values must never be guessed"
