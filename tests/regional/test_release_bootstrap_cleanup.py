from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_gpu_bootstrap as GPU_BOOTSTRAP
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]

RUNNING_JOB = {"metadata": {"name": "running-job"}, "status": {"conditions": []}}
FAILED_JOB = {
    "metadata": {"name": "failed-job"},
    "status": {"conditions": [{"type": "Failed", "status": "True"}]},
}
COMPLETE_JOB = {
    "metadata": {"name": "complete-job"},
    "status": {"conditions": [{"type": "Complete", "status": "True"}]},
}


class JobRunner:
    """A cluster whose installer Jobs are whatever the next listing says."""

    dry_run = False

    def __init__(self, *listings: list[dict[str, Any]]) -> None:
        self.listings = list(listings)
        self.reads = 0
        self.calls: list[list[str]] = []

    def run(self, arguments, **_kwargs):
        self.calls.append(list(arguments))
        if "get" in arguments and "jobs" in arguments:
            index = min(self.reads, len(self.listings) - 1)
            self.reads += 1
            return json.dumps({"items": self.listings[index]})
        return ""

    def deleted(self) -> list[str]:
        return [
            arguments[arguments.index("job") + 1]
            for arguments in self.calls
            if "delete" in arguments and "job" in arguments
        ]


def test_cancel_active_installer_jobs_rechecks_for_running_jobs(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = JobRunner([RUNNING_JOB], [])
    release = MODULE.RegionalRelease(config, runner)

    GPU_BOOTSTRAP.cancel_active_installer_jobs(release, config.clusters[0])

    assert runner.reads == 2
    assert runner.deleted() == ["running-job"], "active Installer Job was not cancelled"


def test_a_quiet_fleet_is_listed_once(tmp_path: Path) -> None:
    """The verifying re-read is proof that a deletion took effect.

    With nothing deleted there is nothing to verify, and the release used to pay
    the identical `get jobs` anyway -- three of them per wave, inside the one
    stretch of an upgrade that prints no output at all.
    """

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = JobRunner([COMPLETE_JOB])
    release = MODULE.RegionalRelease(config, runner)

    GPU_BOOTSTRAP.cancel_active_installer_jobs(release, config.clusters[0])

    assert runner.reads == 1
    assert runner.deleted() == []


def test_settling_cancels_the_running_job_and_clears_the_failed_one(
    tmp_path: Path,
) -> None:
    """One listing has to answer both questions a wave asks about the Jobs.

    An in-flight Job would install the previous wave's identity onto a node the
    new wave has not cleared, and a `Failed` one left behind would keep the
    Reconciler from creating this wave's replacement. A Job that completed is
    neither and must survive: deleting it would make the Reconciler reinstall a
    node that is already on this release.
    """

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = JobRunner([FAILED_JOB, RUNNING_JOB, COMPLETE_JOB], [COMPLETE_JOB])
    release = MODULE.RegionalRelease(config, runner)

    GPU_BOOTSTRAP.settle_installer_jobs(release, config.clusters[0])

    assert runner.deleted() == ["running-job", "failed-job"]
    assert runner.reads == 2, "the listing that decides is read once per settle"


def test_settling_a_quiet_fleet_reads_the_jobs_once(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = JobRunner([COMPLETE_JOB])
    release = MODULE.RegionalRelease(config, runner)

    GPU_BOOTSTRAP.settle_installer_jobs(release, config.clusters[0])

    assert runner.reads == 1
    assert runner.deleted() == []


def test_settling_fails_closed_when_a_cancelled_job_survives(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = JobRunner([RUNNING_JOB])
    release = MODULE.RegionalRelease(config, runner)

    try:
        GPU_BOOTSTRAP.settle_installer_jobs(release, config.clusters[0])
    except MODULE.ReleaseError as error:
        assert "running-job" in str(error)
    else:  # pragma: no cover - the assertion below reports the failure
        raise AssertionError(
            "a wave was allowed to start with an installer Job still in flight"
        )
