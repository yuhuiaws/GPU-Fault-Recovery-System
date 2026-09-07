"""What the post-deploy stability window is allowed to roll a release back for.

Two of its criteria fired on a healthy fleet and cost a ten-minute rollback of a
four-node repeat deploy:

* the queue-growth rule failed on any strictly increasing depth, so a queue that
  went 0 -> 1 -> 1 as the restarted agents re-posted their first inventory was
  read as a backlog running away; and
* every firing critical alert failed immediately, including the two that a
  data-plane restart *causes* -- the collector goes silent and its metrics
  snapshot goes stale for as long as the new Pods take to report.

Both now need a magnitude, and the second needs the restart to be recent. Nothing
here relaxes what a real regression looks like: sustained relative growth with a
genuinely old head-of-queue still fails, and any other critical alert still fails
on the sample it appears in.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]


def _snapshot(
    *, depth: int = 0, oldest_age_seconds: float = 0.0, alerts: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": depth, "oldest_age_seconds": oldest_age_seconds},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {
            "count": len(alerts),
            "alerts": [{"alertname": name, "severity": "critical"} for name in alerts],
        },
    }


def _release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples: list[dict]):
    """A release whose window reads `samples` and whose clock advances on sleep."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_STABILITY_GRACE_ALERTS", raising=False)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    pending = iter(samples)
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(pending))
    clock = {"now": 0.0}
    monkeypatch.setattr(VALIDATION.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        VALIDATION.time,
        "sleep",
        lambda seconds: clock.update(now=clock["now"] + seconds),
    )
    return release


def test_stability_ignores_small_absolute_queue_wobble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0 -> 1 -> 1 is a fleet that came back, not a backlog.

    Every Agent re-posts inventory once its Pod is Ready, so a repeat deploy of a
    four-node fleet reliably puts one item through the queue during the window.
    The old rule needed only strict-then-non-strict increase, so that single item
    rolled the release back.
    """

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=0),
            _snapshot(depth=1, oldest_age_seconds=5.0),
            _snapshot(depth=1, oldest_age_seconds=140.0),
        ],
    )

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["sample_count"] == 3
    assert report["queue_growth"]["failed"] is False
    assert report["queue_growth"]["depths"] == [0, 1, 1]


def test_stability_fails_on_sustained_relative_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue that quadruples with an old head is still a rollback."""

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=10, oldest_age_seconds=5.0),
            _snapshot(depth=20, oldest_age_seconds=90.0),
            _snapshot(depth=40, oldest_age_seconds=150.0),
        ],
    )

    with pytest.raises(MODULE.ReleaseError, match="sustained queue growth"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_stability_keeps_a_grown_queue_that_is_being_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth grew a lot, but the head of the queue is younger than two samples.

    A burst that is being served does not need the release rolled back; the
    oldest item aging past two sample intervals is what says nothing is moving.
    """

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=10, oldest_age_seconds=1.0),
            _snapshot(depth=20, oldest_age_seconds=2.0),
            _snapshot(depth=40, oldest_age_seconds=3.0),
        ],
    )

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["queue_growth"]["failed"] is False


def test_stability_queue_growth_minimum_depth_is_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site that runs a near-empty queue can lower the absolute floor."""

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=0),
            _snapshot(depth=1, oldest_age_seconds=5.0),
            _snapshot(depth=1, oldest_age_seconds=140.0),
        ],
    )
    monkeypatch.setenv("GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH", "1")

    with pytest.raises(MODULE.ReleaseError, match="sustained queue growth"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_stability_grace_for_collector_alerts_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alerts a data-plane restart causes do not fail its own release.

    `GpuFaultCollectorSilent` and `GpuFaultCollectorMetricsSnapshotStale` are
    both consequences of the restart the release just performed, and both clear
    on their own once the new Pods report. They are ignored for ten minutes after
    the newest cluster reached data-converged and never longer.
    """

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=0),
            _snapshot(depth=0, alerts=("GpuFaultCollectorSilent",)),
            _snapshot(depth=0, alerts=("GpuFaultCollectorMetricsSnapshotStale",)),
        ],
    )
    release.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time() - 30},
            "gpu-b": {"state": "CONVERGED", "converged_at_epoch": time.time() - 60},
        }
    }

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["critical_alert_count"] == 1


def test_stability_no_grace_for_other_critical_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled remote command is never explained by a restart."""

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=0),
            _snapshot(depth=0, alerts=("GpuFaultRemoteCommandStalled",)),
        ],
    )
    release.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time()}
        }
    }

    with pytest.raises(MODULE.ReleaseError, match="GpuFaultRemoteCommandStalled"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_stability_grace_expires_and_needs_a_recorded_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No recorded convergence, or an old one, means no grace at all.

    The grace is bounded by evidence the release itself wrote. A state that never
    recorded `converged_at_epoch` -- a release engine that predates it, or a
    control-plane-only deploy that restarted no data plane -- gets the strict
    rule.
    """

    release = _release(
        tmp_path,
        monkeypatch,
        [_snapshot(depth=0), _snapshot(depth=0, alerts=("GpuFaultCollectorSilent",))],
    )
    release.state = {"phase": "complete"}

    with pytest.raises(MODULE.ReleaseError, match="GpuFaultCollectorSilent"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    stale = _release(
        stale_root,
        monkeypatch,
        [_snapshot(depth=0), _snapshot(depth=0, alerts=("GpuFaultCollectorSilent",))],
    )
    stale.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time() - 601}
        }
    }

    with pytest.raises(MODULE.ReleaseError, match="GpuFaultCollectorSilent"):
        stale.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_baseline_grace_lets_a_restarted_collector_open_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline is where the restart's own alerts appear first.

    The baseline sample is taken immediately after the release restarted the data
    plane, so `GpuFaultCollectorSilent` is likelier to be firing there than in any
    later sample. Failing it at the baseline made the in-window grace ineffective:
    the release never reached the window. The same two conditions apply -- the
    alert is named in the grace list and the recorded convergence is recent.
    """

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(depth=0, alerts=("GpuFaultCollectorSilent",)),
            _snapshot(depth=0),
            _snapshot(depth=0),
        ],
    )
    release.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time() - 30}
        }
    }

    report = release.validate_stability_window(window_seconds=120, sample_seconds=60)

    assert report["healthy"] is True
    assert report["sample_count"] == 3
    assert report["critical_clear"]["graced_alerts"] == ["GpuFaultCollectorSilent"]
    assert report["critical_clear"]["wait_seconds"] == 0, (
        "a graced baseline must not spend the critical-clear wait"
    )


def test_baseline_without_a_recorded_convergence_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No epoch, no grace -- at the baseline as much as inside the window."""

    release = _release(
        tmp_path, monkeypatch, [_snapshot(depth=0, alerts=("GpuFaultCollectorSilent",))]
    )
    release.state = {"phase": "complete"}

    with pytest.raises(MODULE.ReleaseError, match="non-settleable critical alerts"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


def test_baseline_grace_does_not_excuse_another_critical_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graced alert alongside a real one does not smuggle the real one through."""

    release = _release(
        tmp_path,
        monkeypatch,
        [
            _snapshot(
                depth=0,
                alerts=("GpuFaultCollectorSilent", "GpuFaultRemoteCommandStalled"),
            )
        ],
    )
    release.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time()}
        }
    }

    with pytest.raises(MODULE.ReleaseError, match="GpuFaultRemoteCommandStalled"):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)


@pytest.mark.parametrize("value", ["eight", "", "-1", "3.5"])
def test_queue_growth_floor_is_validated_before_the_window_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A typo fails the release now, not two minutes into the window.

    The floor used to be parsed where it is used, after the window had already
    been waited out, and a bad value surfaced as a bare `ValueError` the driver
    could not attribute to configuration. Passing no samples at all is how this
    case proves the check runs before the first snapshot: reading one would raise
    `StopIteration` instead.
    """

    release = _release(tmp_path, monkeypatch, [])
    monkeypatch.setenv("GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH", value)

    with pytest.raises(
        MODULE.ReleaseError, match="GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH"
    ):
        release.validate_stability_window(window_seconds=120, sample_seconds=60)
