"""What the Aurora settle loop says while it waits, and how often it re-proves.

Live 2026-10-14 raising the floor from 8/32 to 124/128 ACU kept RDS in
``modifying`` for 801 s -- the whole ACU ramp -- and the loop's reads printed
nothing, so a ``config`` log showed a 13-minute gap and nothing else. The
post-roll floor proof then repeated the three-poll settle rule (38 s) for a
change the request half had already settled.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from gpu_fault.admin import aurora_capacity as aurora
from gpu_fault.admin.config import AuroraCapacityConfig

DESIRED = AuroraCapacityConfig(min_acu=124.0, max_acu=128.0)
WAIT_LINE = re.compile(
    r"^aurora-capacity-wait \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ cluster=aurora-a "
    r"proof=(?P<proof>window|floor) target=124/128 status=(?P<status>\S+) "
    r"window=(?P<window>\S+) instances=reader:available,writer:available"
    r"(?: acu=(?P<acu>\S+))? stable=(?P<stable>\d)/(?P<required>\d) "
    r"elapsed=(?P<elapsed>\d+\.\d)s$"
)


class Clock:
    """``time`` for the module under test: sleeping advances the clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RampingRds:
    """RDS as measured: the modify answers ``available`` on the new window, the
    cluster then reads ``modifying`` for ``modifying_polls`` observations (the
    ACU ramp), and every metric sample is ``sample_age`` old."""

    def __init__(self, *, modifying_polls: int, window=(8.0, 32.0)) -> None:
        self.window = window
        self.modifying_polls = modifying_polls
        self.modified = False
        self.cluster_reads = 0
        self.sample_ages = [timedelta(0)]

    def __call__(
        self, _region: str, *arguments: str, mutate: bool = False
    ) -> dict[str, Any]:
        operation = arguments[1]
        if operation == "modify-db-cluster":
            assert mutate, "the modify must be marked as a mutation"
            self.modified, self.window = True, (124.0, 128.0)
            return {"DBCluster": {"Status": "available"}}
        if operation == "describe-db-clusters":
            self.cluster_reads += 1
            ramping = self.modified and self.modifying_polls > 0
            if ramping:
                self.modifying_polls -= 1
            return {
                "DBClusters": [
                    {
                        "Status": "modifying" if ramping else "available",
                        "ServerlessV2ScalingConfiguration": {
                            "MinCapacity": self.window[0],
                            "MaxCapacity": self.window[1],
                        },
                        "DBClusterMembers": [
                            {"DBInstanceIdentifier": "w", "IsClusterWriter": True},
                            {"DBInstanceIdentifier": "r", "IsClusterWriter": False},
                        ],
                    }
                ]
            }
        if operation == "describe-db-instances":
            instance = {"DBInstanceStatus": "available"}
            return {"DBInstances": [{**instance, "DBInstanceClass": "db.serverless"}]}
        assert operation == "get-metric-statistics"
        age = (
            self.sample_ages.pop(0)
            if len(self.sample_ages) > 1
            else self.sample_ages[0]
        )
        sampled = datetime.now(UTC) - age
        return {"Datapoints": [{"Timestamp": sampled.isoformat(), "Maximum": 124.0}]}


def wait_lines(stderr: str) -> list[re.Match[str]]:
    lines = [line for line in stderr.splitlines() if line.startswith("aurora-")]
    matches = [WAIT_LINE.match(line) for line in lines]
    assert all(matches), f"malformed aurora-capacity-wait line in {lines}"
    return [match for match in matches if match is not None]


def test_the_ramp_is_narrated_when_rds_changes_and_on_a_30_second_heartbeat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = Clock()
    monkeypatch.setattr(aurora, "time", clock)
    rds = RampingRds(modifying_polls=8)

    requested = aurora.request_aurora_capacity(
        aws_region="us-west-2",
        cluster_id="aurora-a",
        expected=AuroraCapacityConfig(min_acu=8.0, max_acu=32.0),
        desired=DESIRED,
        poll_seconds=10,
        aws_json=rds,
    )

    assert requested["modified"] is True and requested["scale_up"] is True
    lines = wait_lines(capsys.readouterr().err)
    assert [(line["status"], line["stable"], line["elapsed"]) for line in lines] == [
        # Unchanged ``modifying`` polls every 10 s print only every 30 s ...
        ("modifying", "0", "0.0"),
        ("modifying", "0", "30.0"),
        ("modifying", "0", "60.0"),
        # ... and every settle step prints, so the last line says it settled.
        ("available", "1", "80.0"),
        ("available", "2", "90.0"),
        ("available", "3", "100.0"),
    ]
    assert {(line["proof"], line["window"], line["required"]) for line in lines} == {
        ("window", "124/128", str(aurora.CAPACITY_SETTLE_STABLE_POLLS))
    }
    assert all(line["acu"] is None for line in lines), (
        "the window proof read the ACU metric it leaves to await"
    )


@pytest.mark.parametrize(
    ("window_settled", "polls"),
    [(True, 1), (False, aurora.CAPACITY_SETTLE_STABLE_POLLS)],
)
def test_the_floor_needs_one_fresh_observation_once_the_window_settled(
    window_settled: bool, polls: int, capsys: pytest.CaptureFixture[str]
) -> None:
    rds = RampingRds(modifying_polls=0, window=(124.0, 128.0))

    settled = aurora.await_aurora_capacity(
        aws_region="us-west-2",
        cluster_id="aurora-a",
        desired=DESIRED,
        modified=True,
        window_settled=window_settled,
        poll_seconds=0,
        aws_json=rds,
    )

    assert rds.cluster_reads == polls
    assert {item["actual_acu"] for item in settled["instances"]} == {124.0}
    last = wait_lines(capsys.readouterr().err)[-1]
    assert (last["proof"], last["stable"], last["required"]) == (
        "floor",
        str(polls),
        str(polls),
    )
    # Each instance's sample with its age: older than three minutes is no proof.
    assert re.fullmatch(r"reader:124\.0@\d+s,writer:124\.0@\d+s", last["acu"] or ""), (
        f"the floor line does not show each sample with its age: {last['acu']}"
    )


def test_a_settled_window_still_refuses_a_stale_floor_sample(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One observation, not a weaker one: a sample older than three minutes
    does not count, so the loop polls again until a fresh one reads the floor."""

    monkeypatch.setattr(aurora, "time", Clock())
    rds = RampingRds(modifying_polls=0, window=(124.0, 128.0))
    rds.sample_ages = [timedelta(minutes=5), timedelta(minutes=5), timedelta(0)]

    aurora.await_aurora_capacity(
        aws_region="us-west-2",
        cluster_id="aurora-a",
        desired=DESIRED,
        modified=True,
        window_settled=True,
        poll_seconds=10,
        aws_json=rds,
    )

    assert rds.cluster_reads == 2, "a stale ACU sample was accepted as the floor"
    stale, fresh = wait_lines(capsys.readouterr().err)
    stale_ages = [int(age) for age in re.findall(r"@(\d+)s", stale["acu"] or "")]
    fresh_ages = [int(age) for age in re.findall(r"@(\d+)s", fresh["acu"] or "")]
    assert stale["stable"] == "0" and min(stale_ages) >= 300, stale["acu"]
    assert fresh["stable"] == "1" and len(fresh_ages) == 2, fresh["acu"]
    assert max(fresh_ages) < 180, fresh["acu"]
