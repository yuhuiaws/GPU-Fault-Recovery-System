"""The CPU rolling-update step's refusals and narrow shapes (cov95).

``tests/admin/test_admin_cpu_rolling_update.py`` drives the deploy's last step
through its main paths over the fake AWS of ``_cpu_rolling_world``. These cases
pin what is left: a damaged journal or last-run record, a cluster (not group)
in an update, an API NLB with two target groups, a sparse HealthyHostCount
series, an alarm that is foreign or composite, a schedule that cannot be
placed, a create whose read-back times out or that someone else answered with
another schedule, a restore of a group that is gone or had no schedule, an
unexpected error, a pair that cannot be defined and a describe AWS denies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cpu_rolling_update as step
from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._cpu_rolling_world import (
    CAPACITY,
    NOT_RESTORED,
    NOW,
    RELEASE_CONFIG,
    World,
    cpu_section,
    live_alarms,
)
from tests.admin.test_admin_hyperpod_rollback import (
    CPU_GROUP,
    LIVE_BEFORE,
    LIVE_ROLLING,
    OLD_TG,
    TG_ARN,
    _described,
)

OTHER_SCHEDULE = {"ScheduleExpression": "cron(0 4 ? * SAT *)"}


def finish(world: World, tmp_path: Path, **spec: Any) -> dict[str, Any]:
    return step.finish_deploy(
        RELEASE_CONFIG,
        cpu_section(**spec),
        tmp_path,
        runner=world,  # type: ignore[arg-type]
        clock=world.clock.clock(),
    )


def target() -> step.Target:
    return step.Target.of(RELEASE_CONFIG, cpu_section())


def updates(world: World) -> list[Any]:
    """The ScheduledUpdateConfig each update-cluster request sent, in order."""

    return [
        json.loads(call[call.index("--instance-groups") + 1])[0].get(
            "ScheduledUpdateConfig"
        )
        for call in world.writes
        if call[:2] == ("sagemaker", "update-cluster")
    ]


def puts(world: World) -> int:
    return sum(call == ("cloudwatch", "put-metric-alarm") for call in world.writes)


def group_schedule(world: World) -> Any:
    return world.hyperpod.describe()["InstanceGroups"][0].get("ScheduledUpdateConfig")


# --- the journal and the last-run record ------------------------------------------


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "cannot read "),
        ('{"schema_version": 2, "groups": {}}', "is not a CPU rolling-update journal"),
        ("[1, 2]", "is not a CPU rolling-update journal"),
    ],
    ids=["unparseable", "another-schema", "not-an-object"],
)
def test_a_damaged_journal_is_refused_never_dropped(
    tmp_path: Path, content: str, message: str
) -> None:
    path = step.journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(BootstrapError) as raised:
        step.load_journal(tmp_path)

    assert message in str(raised.value) and str(path) in str(raised.value)
    assert path.read_text(encoding="utf-8") == content, "the file is left for review"


def test_a_last_run_whose_time_cannot_be_read_is_unknown(tmp_path: Path) -> None:
    since = "2026-11-02T00:00:00Z"
    step.record_last_run(tmp_path, {"at": "yesterday", "status": "DONE"})

    assert step.load_last_run(tmp_path, since=since) == {
        "status": "UNKNOWN",
        "warnings": [],
    }

    step.record_last_run(tmp_path, {"at": "2026-11-02T00:05:00Z", "status": "DONE"})
    assert step.load_last_run(tmp_path, since=since)["status"] == "DONE"


# --- reads ------------------------------------------------------------------------


def test_an_updating_cluster_defers_the_schedule_like_an_updating_group(
    tmp_path: Path,
) -> None:
    group = {"SoftwareUpdateStatus": "Succeeded", "Status": "InService"}
    assert step.update_in_progress({"ClusterStatus": "Updating"}, group) == (
        "the cluster is Updating"
    )
    assert step.update_in_progress({"ClusterStatus": "InService"}, group) is None

    described = _described(LIVE_BEFORE)
    described["ClusterStatus"] = "SystemUpdating"
    world = World(described)

    record = finish(world, tmp_path)

    assert record["status"] == "DEFERRED", record
    assert record["schedule"] == [
        {
            "group": CPU_GROUP,
            "outcome": "deferred",
            "reason": "the cluster is SystemUpdating",
        }
    ]
    assert updates(world) == [], "a deferred step writes no schedule"


class TwoTargetGroups(World):
    def _elbv2_describe_target_groups(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        answer = super()._elbv2_describe_target_groups(arguments, mutate=mutate)
        answer["TargetGroups"].append(
            {"TargetGroupArn": TG_ARN.replace("fedcba", "abcdef"), "TargetType": "ip"}
        )
        return answer


def test_an_api_nlb_with_two_target_groups_gets_no_alarm(tmp_path: Path) -> None:
    world = TwoTargetGroups(_described(LIVE_BEFORE))

    with pytest.raises(BootstrapError, match="has 2 target groups"):
        step.read_api_nlb(world, target())  # type: ignore[arg-type]
    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    (warning,) = record["warnings"]
    assert warning.startswith("rollback alarms: the API NLB "), warning
    assert "the rollback alarms need exactly one" in warning
    assert puts(world) == 0 and updates(world) == [], "nothing is put or rolled"


class SparseStatistics(World):
    """HealthyHostCount answers a junk entry and a short minute, then N."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.statistics: list[str] = []

    def _cloudwatch_get_metric_statistics(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        start = arguments[arguments.index("--start-time") + 1]
        self.statistics.append(start)
        if len(self.statistics) == 1:
            return {"Datapoints": ["junk", {"Timestamp": start, "Maximum": 2.0}]}
        return {"Datapoints": [{"Timestamp": start, "Maximum": 3.0}]}


def test_a_minute_below_n_is_not_full_capacity_and_the_wait_goes_on() -> None:
    world = SparseStatistics(_described(LIVE_BEFORE))
    nlb = step.read_api_nlb(world, target())  # type: ignore[arg-type]

    evidence = step.wait_for_full_capacity(
        world,  # type: ignore[arg-type]
        target(),
        nlb,
        clock=world.clock.clock(),
    )

    assert evidence == {
        "healthy_since": NOW.isoformat(),
        "full_minute": NOW.isoformat(),
    }
    assert world.statistics == [NOW.isoformat()] * 2, (
        "the short minute was read, judged not full, and read again"
    )
    elapsed = (world.clock.now - NOW).total_seconds()
    assert elapsed == 75, "one whole minute, then one more interval"


# --- part A: the alarms -------------------------------------------------------------


def test_an_untagged_alarm_on_another_nlb_is_never_taken_over(tmp_path: Path) -> None:
    foreign = live_alarms(lb="net/someone-elses-nlb/0123456789abcdef", tg=OLD_TG)
    world = World(_described(LIVE_BEFORE), alarms=foreign)

    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    assert [item["action"] for item in record["alarms"]["alarms"]] == [
        "refused",
        "refused",
    ]
    errors = record["alarms"]["errors"]
    assert len(errors) == 2 and all(
        "exists without gpu-fault:site-id and does not watch this site's API NLB"
        in error
        for error in errors
    ), errors
    assert (
        "rollback alarms: an alarm of the pair is not this site's"
        in (record["warnings"])
    )
    assert puts(world) == 0, "a foreign alarm is not redefined"
    assert world.alarms == foreign
    assert step.load_journal(tmp_path)["alarms"] == {}, "nothing is recorded as ours"


class CompositeCapacity(World):
    def _cloudwatch_describe_alarms(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        answer = super()._cloudwatch_describe_alarms(arguments, mutate=mutate)
        answer["CompositeAlarms"] = [{"AlarmName": CAPACITY}]
        return answer


def test_a_composite_alarm_of_the_pairs_name_is_refused(tmp_path: Path) -> None:
    world = CompositeCapacity(_described(LIVE_BEFORE))

    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    actions = {item["name"]: item["action"] for item in record["alarms"]["alarms"]}
    assert actions == {CAPACITY: "refused", NOT_RESTORED: "created"}
    assert (
        f"{CAPACITY} is a composite alarm; refusing to redefine it"
        in record["alarms"]["errors"]
    )
    assert puts(world) == 1 and CAPACITY not in world.alarms
    assert updates(world) == [], "the schedule is not touched without the pair"


# --- part B: the schedule -------------------------------------------------------------


def test_a_schedule_that_cannot_be_placed_defers_the_step(tmp_path: Path) -> None:
    world = World(_described({"ScheduleExpression": "rate(7 days)"}))

    record = finish(world, tmp_path)

    assert record["status"] == "DEFERRED", record
    (item,) = record["schedule"]
    assert item["outcome"] == "deferred"
    assert item["reason"] == (
        "the schedule cannot be placed in time ('rate(7 days)' is not a cron(...) "
        "expression)"
    )
    assert updates(world) == []
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert entry["phase"] == "deferred" and entry["reason"] == item["reason"]


class SlowCreate(World):
    """The create is applied, but the cluster then stays ``Updating``."""

    def _sagemaker_update_cluster(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        answer = super()._sagemaker_update_cluster(arguments, mutate=mutate)
        sent = json.loads(arguments[arguments.index("--instance-groups") + 1])
        if sent[0].get("ScheduledUpdateConfig"):
            self._updating = 10_000
        return answer


def test_a_create_whose_read_back_times_out_but_landed_counts_as_applied(
    tmp_path: Path,
) -> None:
    world = SlowCreate(_described(LIVE_BEFORE))

    record = finish(world, tmp_path)

    assert record["status"] == "DONE", record
    assert [item["outcome"] for item in record["schedule"]] == ["applied"]
    assert updates(world) == [None, LIVE_ROLLING], "delete, then one create"
    assert group_schedule(world) == LIVE_ROLLING
    waited = sum(world.clock.slept)
    assert waited >= step.CONVERGE_TIMEOUT_SECONDS, "the read-back ran out first"
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert (entry["phase"], entry[step.AWAITING_CREATE]) == ("verified", False)


class AnsweredByAnotherSchedule(World):
    """The create fails while someone else puts a different schedule."""

    def _sagemaker_update_cluster(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        sent = json.loads(arguments[arguments.index("--instance-groups") + 1])
        if sent[0].get("ScheduledUpdateConfig"):
            group = self.hyperpod.described["InstanceGroups"][0]
            group["ScheduledUpdateConfig"] = dict(OTHER_SCHEDULE)
            raise BootstrapError("command failed (254): aws sagemaker update-cluster")
        return super()._sagemaker_update_cluster(arguments, mutate=mutate)


def test_a_group_left_with_another_schedule_is_not_written_again(
    tmp_path: Path,
) -> None:
    world = AnsweredByAnotherSchedule(_described(LIVE_BEFORE))

    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    (item,) = record["schedule"]
    assert item["outcome"] == "failed"
    assert item["reason"].startswith(
        f"the CPU instance group {CPU_GROUP} carries ScheduledUpdateConfig "
        + json.dumps(OTHER_SCHEDULE, sort_keys=True)
        + ", not the rolling one (command failed (254)"
    ), item["reason"]
    assert item["reason"].endswith("; nothing more was written"), item["reason"]
    assert updates(world) == [None, LIVE_ROLLING], "no retry over a foreign schedule"
    assert group_schedule(world) == OTHER_SCHEDULE
    assert any(
        warning.startswith(f"CPU rolling schedule of {CPU_GROUP}: failed: ")
        for warning in record["warnings"]
    ), record["warnings"]


def test_a_restore_reports_a_gone_group_and_removes_a_schedule_it_never_had(
    tmp_path: Path,
) -> None:
    """The ledger's copy names a group that left and one that had no schedule;
    a journal entry without a pre-install record does not override it."""

    world = World(_described(LIVE_ROLLING))
    path = step.journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    journal = {"schema_version": 1, "alarms": {}, "groups": {CPU_GROUP: {}}}
    journal["groups"][CPU_GROUP]["phase"] = "deferred"
    path.write_text(json.dumps(journal), encoding="utf-8")

    results = step.restore_schedule(
        world,  # type: ignore[arg-type]
        target(),
        tmp_path,
        recorded={"cpu-control-plane-gone": LIVE_BEFORE, CPU_GROUP: None},
        clock=world.clock.clock(),
    )

    assert results == [
        {"group": "cpu-control-plane-gone", "outcome": "group-gone"},
        {"group": CPU_GROUP, "outcome": "restored", "schedule": None},
    ]
    assert updates(world) == [None], "the delete only: nothing to create"
    assert group_schedule(world) is None
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert (entry["phase"], entry[step.AWAITING_CREATE]) == ("restored", False)


# --- the deploy's last step --------------------------------------------------------------


def test_an_unexpected_error_is_recorded_and_never_raised(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE))
    config = dict(RELEASE_CONFIG)
    del config["cpu_eks_arn"]

    record = step.finish_deploy(
        config,
        cpu_section(),
        tmp_path,
        runner=world,  # type: ignore[arg-type]
        clock=world.clock.clock(),
    )

    assert record["status"] == "FAILED"
    assert record["warnings"] == ["stopped: KeyError: 'cpu_eks_arn'"]
    assert record["schedule"] == []
    assert "stopped on an unexpected error" in record["summary"]
    assert record["at"] == NOW.isoformat().replace("+00:00", "Z")
    assert world.calls == [], "nothing was read before the target was known"


def test_a_pair_that_cannot_be_defined_fails_the_alarms_and_skips_the_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer than two API replicas define no pair: nothing is put, nothing
    waited for, and the schedule is not touched without its alarms."""

    define = rollback.alarm_definitions
    monkeypatch.setattr(
        rollback,
        "alarm_definitions",
        lambda **options: define(**{**options, "replicas": 1}),
    )
    world = World(_described(LIVE_BEFORE))

    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    assert record["capacity"] == {"skipped": "both alarms match: nothing is put"}
    assert record["alarms"]["alarms"] == []
    note, failure = record["warnings"]
    assert "at least 2 expected" in note, note
    assert failure == "rollback alarms: no rollback alarm could be defined"
    assert puts(world) == 0 and updates(world) == []
    assert world.clock.slept == [], "no capacity wait for a pair that is not put"


class DeniedDescribe(World):
    def _sagemaker_describe_cluster(
        self, arguments: list[str], *, mutate: bool
    ) -> dict[str, Any]:
        raise BootstrapError(
            "command failed (254): aws sagemaker describe-cluster: AccessDenied"
        )


def test_a_schedule_step_that_cannot_describe_the_cluster_fails_loudly(
    tmp_path: Path,
) -> None:
    world = DeniedDescribe(_described(LIVE_BEFORE), alarms=live_alarms())

    record = finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    assert record["alarm_states"] == {CAPACITY: "OK", NOT_RESTORED: "OK"}
    (item,) = record["schedule"]
    assert item["group"] == "*" and item["outcome"] == "failed"
    assert "AccessDenied" in item["reason"]
    assert record["warnings"] == [
        f"CPU rolling schedule of *: failed: {item['reason']}"
    ]
    assert record["summary"].startswith("CPU rolling update (rolling): * failed."), (
        record["summary"]
    )
