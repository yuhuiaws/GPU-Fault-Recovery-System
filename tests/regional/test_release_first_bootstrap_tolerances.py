"""Engine tolerances a first bootstrap needs (live trace, 2026-09-12).

A fresh site has no control plane, no agents, an empty store and a CronJob that
has never ticked; these tests pin the places where the engine used to read that
emptiness as a failure.
"""

from __future__ import annotations

import pytest

from gpu_fault_release import regional_release_progress as PROGRESS
from tests.regional.test_regional_admin_checks import (
    TARGETS,
    _aurora_cronjob,
    _aurora_release,
    _aurora_role,
    _aurora_secret,
    _checks_module,
    _status,
)


def test_aurora_refresh_verify_accepts_the_secret_status_before_the_first_tick() -> (
    None
):
    """A fresh site's CronJob has no lastSuccessfulTime for up to an hour (the
    controller only records its own scheduled runs), but the bootstrap's Job has
    already written a fresh "ok" status into the Secret; verify must take it."""

    module = _checks_module()
    cronjob = _aurora_cronjob(TARGETS)
    cronjob.pop("status", None)
    release = _aurora_release(
        cronjob, _aurora_role(TARGETS), _aurora_secret(_status("ok", minutes_ago=3))
    )

    value = module.check_aurora_refresh(release, require_success=True)

    assert value.details["last_refresh_status"]["status"] == "ok"
    assert value.details["last_successful_time"] is None


def test_aurora_refresh_verify_still_needs_success_evidence() -> None:
    module = _checks_module()
    cronjob = _aurora_cronjob(TARGETS)
    cronjob.pop("status", None)
    release = _aurora_release(cronjob, _aurora_role(TARGETS), _aurora_secret(None))

    with pytest.raises(module.ReleaseError, match="never recorded"):
        module.check_aurora_refresh(release, require_success=True)
    # A stale status is no substitute for the missing tick either.
    release = _aurora_release(
        cronjob,
        _aurora_role(TARGETS),
        _aurora_secret(_status("ok", minutes_ago=4 * 60)),
    )
    with pytest.raises(module.ReleaseError, match="stale"):
        module.check_aurora_refresh(release, require_success=True)


def test_bootstrap_completion_state_gives_the_stability_window_its_restart_grace() -> (
    None
):
    """The stability window forgives collector alerts raised while agents were
    being installed only after the newest converged_at_epoch; a bootstrap
    installs every agent from nothing, so its complete state must carry one."""

    state = PROGRESS.bootstrap_completion_state(["gpu-b", "gpu-a"], now=1000.0)

    assert state["completed_cluster_ids"] == ["gpu-a", "gpu-b"]
    attempts = state["cluster_attempts"]
    assert list(attempts) == ["gpu-a", "gpu-b"]
    assert attempts["gpu-a"]["converged_at_epoch"] == 1000.0
    assert attempts["gpu-a"]["state"] in PROGRESS.CLUSTER_ATTEMPT_STATES
