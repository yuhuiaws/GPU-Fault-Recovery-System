"""The deploy's last step owns the CPU rolling update (batch 240).

Part A, the rollback alarms: created at the end of a first deploy, after every
api-ha replica is a healthy target and HealthyHostCount Maximum was N for one
whole minute; re-pointed after a fresh install's new NLB; not put again when
identical; adopted when untagged on this site's NLB; refused when another
site's tag names them. Statistic Maximum, thresholds N-1 and N from the
replica count, the second window below the group's wait. A failure never
raises out of the step.

Part B, the schedule: delete, verify, create, verify against a fake service
that refuses exactly the three requests of 2026-11-02; NOOP when the group
already rolls this way; no write to a group without a schedule; each refusal
defers; a create that fails after the delete is retried once and then fails
loudly with the exact command; the journal records the pre-install state and
resumes a group a crash left without a schedule; ``unchanged`` restores it.
"""

from __future__ import annotations

import json
import shlex
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cpu_rolling_update as step
from gpu_fault.admin import cpu_rolling_update_command as verb
from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.config import AdminConfig, default_admin_config
from tests.admin._cpu_rolling_world import (
    CAPACITY,
    NOT_RESTORED,
    NOW,
    OTHER_SITE,
    RELEASE_CONFIG,
    SITE_TAG,
    FakeClock,
    World,
    cpu_section,
    live_alarms,
    old_nlb_alarms,
)
from tests.admin.test_admin_hyperpod_rollback import (
    CPU_GROUP,
    LB,
    LIVE_BEFORE,
    LIVE_ROLLING,
    TG,
    _described,
)


def _finish(world: World, tmp_path: Path, **spec: Any) -> dict[str, Any]:
    return step.finish_deploy(
        RELEASE_CONFIG,
        cpu_section(**spec),
        tmp_path,
        runner=world,  # type: ignore[arg-type]
        clock=world.clock.clock(),
    )


def _puts(world: World) -> list[tuple[str, ...]]:
    return [call for call in world.writes if call == ("cloudwatch", "put-metric-alarm")]


def _updates(world: World) -> list[tuple[str, ...]]:
    return [
        call for call in world.writes if call[:2] == ("sagemaker", "update-cluster")
    ]


def _sent_schedules(world: World) -> list[Any]:
    """The ScheduledUpdateConfig of each update-cluster request, in order."""

    return [
        json.loads(call[call.index("--instance-groups") + 1])[0].get(
            "ScheduledUpdateConfig"
        )
        for call in _updates(world)
    ]


# --- part A: the alarms --------------------------------------------------------------


def test_a_first_deploy_creates_the_pair_once_the_control_plane_is_full(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_ROLLING))

    record = _finish(world, tmp_path)

    assert record["status"] == "DONE", record
    reads = [call[:2] for call in world.calls]
    health = reads.index(("elbv2", "describe-target-health"))
    minute = reads.index(("cloudwatch", "get-metric-statistics"))
    first_put = world.calls.index(("cloudwatch", "put-metric-alarm"))
    assert health < minute < first_put, "healthy targets, then a whole minute, then put"
    put_at = world.alarms[CAPACITY]["AlarmConfigurationUpdatedTimestamp"]
    assert put_at >= (NOW + timedelta(minutes=1)).isoformat(), (
        "not before one whole minute at full capacity"
    )
    assert [item["action"] for item in record["alarms"]["alarms"]] == [
        "created",
        "created",
    ]
    capacity, restored = (world.alarms[CAPACITY], world.alarms[NOT_RESTORED])
    for alarm, threshold, minutes in ((capacity, 2.0, 5), (restored, 3.0, 25)):
        assert alarm["Statistic"] == "Maximum" == rollback.RECOMMENDED_STATISTIC
        assert (alarm["Threshold"], alarm["EvaluationPeriods"]) == (threshold, minutes)
        assert alarm["DatapointsToAlarm"] == minutes and alarm["Period"] == 60
        assert alarm["TreatMissingData"] == "missing"
        assert not alarm["AlarmActions"] and not alarm["OKActions"]
        assert {d["Name"]: d["Value"] for d in alarm["Dimensions"]} == {
            "TargetGroup": TG,
            "LoadBalancer": LB,
        }
        assert world.tags[alarm["AlarmArn"]] == [SITE_TAG], "tagged on create"
    assert record["alarm_states"] == {CAPACITY: "OK", NOT_RESTORED: "OK"}
    journal = step.load_journal(tmp_path)
    assert sorted(journal["alarms"]) == [CAPACITY, NOT_RESTORED]
    assert record["schedule"] == [{"group": CPU_GROUP, "outcome": "noop"}]
    assert not _updates(world), "the live group already rolls this way"


def test_the_thresholds_follow_the_replica_count_and_refuse_fewer_than_two() -> None:
    nlb = {"name": "n", "load_balancer": LB, "target_groups": [TG]}

    five = rollback.alarm_definitions(nlb=nlb, replicas=5, wait_seconds=1800)
    one = rollback.alarm_definitions(nlb=nlb, replicas=1, wait_seconds=1800)

    assert [item["Threshold"] for item in five["definitions"]] == [4.0, 5.0]
    assert one["definitions"] == [] and "at least 2 expected" in one["note"]


@pytest.mark.parametrize(
    "wait,expected",
    [(None, 25), (3600, 25), (1800, 25), (1200, 15), (900, 10), (600, None)],
    ids=["no-rolling-schedule", "1h", "approved-1800", "20min", "15min", "10min"],
)
def test_the_second_window_stays_below_the_groups_wait(
    wait: int | None, expected: int | None
) -> None:
    assert rollback.restore_minutes(wait) == expected
    nlb = {"name": "n", "load_balancer": LB, "target_groups": [TG]}
    pair = rollback.alarm_definitions(nlb=nlb, replicas=3, wait_seconds=wait)
    windows = [item["EvaluationPeriods"] for item in pair["definitions"]]
    assert windows == ([5] if expected is None else [5, expected])
    if wait is not None and expected is not None:
        assert expected * 60 < wait, "the alarm trips before the next batch"


def test_a_fresh_install_re_points_the_pair_at_the_new_nlb(tmp_path: Path) -> None:
    alarms = old_nlb_alarms()
    tags = {item["AlarmArn"]: [dict(SITE_TAG)] for item in alarms.values()}
    world = World(_described(LIVE_ROLLING), alarms=alarms, tags=tags)

    record = _finish(world, tmp_path)

    assert record["status"] == "DONE", record
    assert [item["action"] for item in record["alarms"]["alarms"]] == [
        "updated",
        "updated",
    ]
    for name in (CAPACITY, NOT_RESTORED):
        dimensions = {d["Name"]: d["Value"] for d in world.alarms[name]["Dimensions"]}
        assert dimensions == {"TargetGroup": TG, "LoadBalancer": LB}
    assert not _updates(world), "same names: the schedule needs no change"


def test_an_identical_pair_is_not_put_again(tmp_path: Path) -> None:
    world = World(_described(LIVE_ROLLING))
    _finish(world, tmp_path)
    world.writes.clear()

    record = _finish(world, tmp_path)

    assert [item["action"] for item in record["alarms"]["alarms"]] == [
        "unchanged",
        "unchanged",
    ]
    assert not _puts(world) and not world.writes, "nothing written at all"


def test_the_live_hand_made_pair_is_adopted_and_tagged(tmp_path: Path) -> None:
    world = World(_described(LIVE_ROLLING), alarms=live_alarms())

    record = _finish(world, tmp_path)

    assert record["status"] == "DONE", record
    results = record["alarms"]["alarms"]
    assert all(item.get("tagged") for item in results), results
    for alarm in world.alarms.values():
        assert world.tags[alarm["AlarmArn"]] == [SITE_TAG]


def test_another_sites_alarm_of_that_name_is_refused(tmp_path: Path) -> None:
    alarms = live_alarms()
    tags = {
        item["AlarmArn"]: [{"Key": "gpu-fault:site-id", "Value": OTHER_SITE}]
        for item in alarms.values()
    }
    world = World(_described(LIVE_BEFORE), alarms=alarms, tags=tags)

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED"
    assert {item["action"] for item in record["alarms"]["alarms"]} == {"refused"}
    assert any(
        "two sites in one account and Region need their own alarm names" in warning
        and "spec.cpuControlPlane.rollbackAlarms" in warning
        for warning in record["warnings"]
    ), record["warnings"]
    assert not _puts(world) and not _updates(world), (
        "neither the alarms nor the schedule"
    )


def test_a_site_override_names_its_own_pair(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE))

    record = _finish(
        world,
        tmp_path,
        rollbackAlarms={
            "capacity": "site-b-cpu-capacity",
            "notRestored": "site-b-cpu-back",
        },
    )

    assert record["status"] == "DONE", record
    assert sorted(world.alarms) == ["site-b-cpu-back", "site-b-cpu-capacity"]
    (schedule,) = [item for item in _sent_schedules(world) if item is not None]
    assert rollback.rollback_alarm_names(schedule) == [
        "site-b-cpu-capacity",
        "site-b-cpu-back",
    ]


def test_a_failure_is_recorded_never_raised(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE), nlb=False)

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED"
    assert any("LoadBalancerNotFound" in warning for warning in record["warnings"]), (
        record["warnings"]
    )
    assert record["schedule"] == [] and not world.writes
    assert (
        "the rollback alarms are not ready, so the schedule was not touched"
        in (record["summary"])
    )


def test_a_refused_put_names_the_aws_error_and_touches_no_schedule(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    world.refused_writes["put-metric-alarm"] = (
        "An error occurred (AccessDenied) when calling the PutMetricAlarm operation: "
        "not authorized to perform: cloudwatch:TagResource"
    )

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    assert [item["action"] for item in record["alarms"]["alarms"]] == [
        "failed",
        "failed",
    ]
    assert any("cloudwatch:TagResource" in item for item in record["warnings"]), (
        "the AWS reason reaches the warning"
    )
    assert not _updates(world), "no alarm, no schedule change"
    assert step.load_journal(tmp_path)["alarms"] == {}, "nothing was created"


def test_an_own_alarm_whose_update_fails_stays_in_the_ledger(tmp_path: Path) -> None:
    alarms = old_nlb_alarms()
    tags = {item["AlarmArn"]: [dict(SITE_TAG)] for item in alarms.values()}
    world = World(_described(LIVE_ROLLING), alarms=alarms, tags=tags)
    world.refused_writes["put-metric-alarm"] = "Throttling: Rate exceeded"

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED", record
    assert sorted(step.load_journal(tmp_path)["alarms"]) == [CAPACITY, NOT_RESTORED], (
        "the site's alarms are still recorded, so uninstall deletes them"
    )


def test_a_control_plane_that_never_fills_puts_no_alarm(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE), healthy=2)

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED"
    assert any("3 healthy API targets" in warning for warning in record["warnings"]), (
        record["warnings"]
    )
    assert not world.writes, "an installing control plane never gets an alarm"


# --- part B: the schedule ------------------------------------------------------------


def test_rolling_deletes_then_creates_then_reads_it_back(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE))

    record = _finish(world, tmp_path)

    assert record["status"] == "DONE", record
    assert [item["outcome"] for item in record["schedule"]] == ["applied"]
    assert _sent_schedules(world) == [None, LIVE_ROLLING], (
        "delete (the member left out), then create the rolling one"
    )
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_ROLLING, "the live shape of 00:34Z"
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert entry["pre_install"]["scheduled_update_config"] == LIVE_BEFORE
    assert entry["phase"] == "verified"
    assert "Change it with spec.cpuControlPlane.rollingUpdate" in record["summary"]


def test_the_group_keeps_its_own_cron(tmp_path: Path) -> None:
    described = _described({"ScheduleExpression": "cron(30 3 ? * WED *)"})
    world = World(described, clock=FakeClock())

    _finish(world, tmp_path)

    schedule = world.hyperpod.describe()["InstanceGroups"][0]["ScheduledUpdateConfig"]
    assert schedule["ScheduleExpression"] == "cron(30 3 ? * WED *)", "how, never when"


def test_a_group_without_a_schedule_gets_none(tmp_path: Path) -> None:
    world = World(_described(None))

    record = _finish(world, tmp_path)

    assert record["schedule"] == [{"group": CPU_GROUP, "outcome": "no-schedule"}]
    assert not _updates(world), "the product never adds a schedule"


def _with(**changes: Any) -> dict[str, Any]:
    described = _described(LIVE_BEFORE)
    described["InstanceGroups"][0].update(changes)
    return described


@pytest.mark.parametrize(
    "described,reason",
    [
        (
            _with(SoftwareUpdateStatus="InProgress"),
            "SoftwareUpdateStatus is InProgress",
        ),
        (_with(SoftwareUpdateStatus="Pending"), "SoftwareUpdateStatus is Pending"),
        (
            _with(SoftwareUpdateStatus="RollbackInProgress"),
            "SoftwareUpdateStatus is RollbackInProgress",
        ),
        (_with(Status="SystemUpdating"), "the instance group is SystemUpdating"),
        (_with(ActiveOperations={"Scaling": 1}), "ActiveOperations"),
    ],
    ids=["in-progress", "pending", "rollback", "system-updating", "write-refusal"],
)
def test_each_refusal_defers_the_step(
    described: dict[str, Any], reason: str, tmp_path: Path
) -> None:
    world = World(described)

    record = _finish(world, tmp_path)

    assert record["status"] == "DEFERRED", record
    (item,) = record["schedule"]
    assert item["outcome"] == "deferred" and reason in item["reason"], item
    assert not _updates(world), "a deferred step writes nothing"
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert entry["phase"] == "deferred" and reason in entry["reason"]


def test_the_next_window_too_close_defers_the_step(tmp_path: Path) -> None:
    clock = FakeClock()
    # Sunday 2026-11-08 06:00Z: two hours before cron(0 8 ? * SUN *).
    clock.now = clock.now.replace(day=8, hour=6, minute=0)
    world = World(_described(LIVE_BEFORE), clock=clock)

    record = _finish(world, tmp_path)

    assert record["status"] == "DEFERRED"
    (item,) = record["schedule"]
    assert (
        "the next scheduled update window starts 2026-11-08T08:00:00+00:00"
        in (item["reason"])
    ), item
    assert not _updates(world), "nothing is written that close to the window"


def test_a_create_that_fails_after_the_delete_is_retried_once_then_loud(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    world.fail_creates = 2

    record = _finish(world, tmp_path)

    assert record["status"] == "FAILED"
    (item,) = record["schedule"]
    assert item["outcome"] == "failed"
    assert item["reason"].startswith(
        f"the CPU instance group {CPU_GROUP} of cpu-control-plane now has NO "
        "ScheduledUpdateConfig -- no automatic patching. That is safe, but it must "
        "be fixed: the create failed twice"
    ), item["reason"]
    command = item["reason"].split("Run, in a maintenance window: ", 1)[1]
    words = shlex.split(command.split(" -- or re-run", 1)[0])
    assert words[:3] == ["aws", "sagemaker", "update-cluster"]
    sent = json.loads(words[words.index("--instance-groups") + 1])
    assert sent[0]["ScheduledUpdateConfig"] == LIVE_ROLLING, "the exact create"
    assert _sent_schedules(world) == [None, LIVE_ROLLING, LIVE_ROLLING], (
        "delete, then the create sent twice: retried once"
    )
    assert (
        world.hyperpod.describe()["InstanceGroups"][0].get("ScheduledUpdateConfig")
        is None
    )
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert entry["phase"] == "failed" and "NO ScheduledUpdateConfig" in entry["reason"]


def test_a_group_a_crash_left_without_a_schedule_is_created_again(
    tmp_path: Path,
) -> None:
    world = World(_described(None))
    journal = {
        "schema_version": 1,
        "alarms": {},
        "groups": {
            CPU_GROUP: {
                "pre_install": {
                    "recorded_at": "2026-11-02T01:00:00Z",
                    "scheduled_update_config": LIVE_BEFORE,
                },
                "phase": "deleted",
                step.AWAITING_CREATE: True,
            }
        },
    }
    path = step.journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(journal), encoding="utf-8")

    record = _finish(world, tmp_path)

    assert [item["outcome"] for item in record["schedule"]] == ["applied"], record
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_ROLLING, "created from the record"
    assert step.load_journal(tmp_path)["groups"][CPU_GROUP][step.AWAITING_CREATE] is (
        False
    )


def test_a_schedule_removed_by_someone_else_is_not_created_again(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    world.hyperpod.described["InstanceGroups"][0].pop("ScheduledUpdateConfig")
    world.writes.clear()

    record = _finish(world, tmp_path)

    assert record["schedule"] == [{"group": CPU_GROUP, "outcome": "no-schedule"}]
    assert not _updates(world), "the customer removed it after the step verified"


def test_unchanged_restores_what_the_step_changed_and_leaves_the_rest(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    assert world.hyperpod.describe()["InstanceGroups"][0]["ScheduledUpdateConfig"] == (
        LIVE_ROLLING
    )

    record = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert record["status"] == "DONE", record
    assert [item["outcome"] for item in record["schedule"]] == ["restored"]
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_BEFORE, (
        "the recorded pre-install state"
    )
    world.writes.clear()
    again = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})
    assert again["schedule"] == [] and not _updates(world), "left as found from now on"


def test_a_schedule_already_back_to_its_record_is_settled_without_a_write(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    group = world.hyperpod.described["InstanceGroups"][0]
    group["ScheduledUpdateConfig"] = dict(LIVE_BEFORE)  # put back by hand
    world.writes.clear()

    first = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})
    second = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert [item["outcome"] for item in first["schedule"]] == ["unchanged"], first
    assert second["schedule"] == [], "settled: not looked at again"
    assert not _updates(world), "nothing to write"


def test_unchanged_never_touches_a_schedule_the_step_did_not_change(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_ROLLING))

    record = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert record["status"] == "DONE" and record["schedule"] == []
    assert not _updates(world), "unchanged leaves the customer's schedule as found"
    assert "left as found" in record["summary"]


def test_a_refused_switch_to_unchanged_is_journaled_as_deferred(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    world.hyperpod.described["InstanceGroups"][0]["SoftwareUpdateStatus"] = "Pending"
    world.writes.clear()

    record = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert record["status"] == "DEFERRED", record
    (item,) = record["schedule"]
    assert (
        item["outcome"] == "deferred"
        and "SoftwareUpdateStatus is Pending" in (item["reason"])
    )
    assert not _updates(world), "a refused restore writes nothing"
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert entry["phase"] == "deferred", "the status check names it"
    assert "restore of the pre-install schedule is deferred" in record["summary"]


def test_a_restore_that_stops_after_its_delete_is_failed_and_resumed(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    world.fail_creates = 1
    world.writes.clear()

    record = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert record["status"] == "FAILED", record
    assert _sent_schedules(world) == [None, LIVE_BEFORE], "delete, then the create"
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group.get("ScheduledUpdateConfig") is None
    entry = step.load_journal(tmp_path)["groups"][CPU_GROUP]
    assert (entry["phase"], entry[step.AWAITING_CREATE]) == ("failed", True)

    again = _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    assert again["status"] == "DONE", again
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_BEFORE, "the next run creates it"


def test_rolling_resumes_a_group_an_interrupted_restore_left_bare(
    tmp_path: Path,
) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    world.fail_creates = 1
    _finish(world, tmp_path, rollingUpdate={"mode": "unchanged"})

    record = _finish(world, tmp_path)

    assert [item["outcome"] for item in record["schedule"]] == ["applied"], record
    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_ROLLING


def test_restore_is_refused_during_an_update(tmp_path: Path) -> None:
    world = World(_described(LIVE_BEFORE))
    _finish(world, tmp_path)
    group = world.hyperpod.described["InstanceGroups"][0]
    group["SoftwareUpdateStatus"] = "InProgress"
    world.writes.clear()

    with pytest.raises(step.BootstrapError, match="cannot restore the pre-install"):
        step.restore_schedule(
            world,  # type: ignore[arg-type]
            step.Target.of(RELEASE_CONFIG, cpu_section()),
            tmp_path,
            clock=world.clock.clock(),
        )
    assert not world.writes, "nothing was written"


# --- the verb and the admin config ---------------------------------------------------


def test_the_verb_runs_the_same_step_under_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    world = World(_described(LIVE_BEFORE))
    site = type(
        "Site", (), {"source": tmp_path / "site.yaml", "release_config": RELEASE_CONFIG}
    )()
    monkeypatch.setattr(verb, "reload_site_for_mutation", lambda value: value)
    monkeypatch.setattr(step, "Clock", lambda: world.clock.clock())

    code = verb.run_step(site, runner=world)  # type: ignore[arg-type]

    out = capsys.readouterr()
    assert code == 0, out.err
    assert json.loads(out.out)["status"] == "DONE"
    assert "Change it with spec.cpuControlPlane.rollingUpdate" in out.err
    last = step.load_last_run(tmp_path, since="2026-11-02T00:00:00Z")
    assert last["status"] == "DONE", "recorded like the deploy's run"


def test_the_admin_config_section_defaults_to_rolling_and_keeps_digests() -> None:
    default = default_admin_config()
    unchanged = default.patched(
        {"cpuControlPlane": {"rollingUpdate": {"mode": "unchanged"}}}
    )

    assert default.cpu_control_plane.rolling_update.mode == "rolling"
    assert default.cpu_control_plane.rolling_update.wait_interval_seconds == 1800
    assert (
        default.cpu_control_plane.rollback_alarms.capacity,
        default.cpu_control_plane.rollback_alarms.not_restored,
    ) == rollback.SUGGESTED_ALARM_NAMES
    assert "cpu_control_plane" not in default.as_dict(), (
        "at its default the section is left out: every desired.json keeps its digest"
    )
    assert unchanged.role_sha256() == default.role_sha256(), (
        "no role payload carries it"
    )
    assert AdminConfig.from_mapping(unchanged.as_dict()) == unchanged
    for spec, message in (
        ({"rollingUpdate": {"mode": "sometimes"}}, "must be rolling or unchanged"),
        ({"rollingUpdate": {"waitIntervalSeconds": 3601}}, "within 900..3600"),
        ({"rollingUpdate": {"waitIntervalSeconds": 600}}, "needs 10 minutes"),
        ({"rollbackAlarms": {"capacity": "x", "notRestored": "x"}}, "two different"),
        ({"rollbackAlarms": {"capacity": "has a space"}}, "letters, digits"),
    ):
        with pytest.raises(Exception, match=message):
            default.patched({"cpuControlPlane": spec}).validate()


def test_the_api_ha_roll_never_drops_a_healthy_target() -> None:
    """The capacity alarm counts healthy API targets: api-ha rolls surge
    first, so an upgrade deploy keeps N, and N is the alarms' replica count."""

    root = Path(__file__).resolve().parents[2]
    rendered = yaml.safe_load(
        (
            root
            / "deploy/control-plane/regional/generated/gpu-fault-api-ha-ingress.yaml"
        ).read_text(encoding="utf-8")
    )

    assert rendered["spec"]["strategy"] == {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1},
    }, "surge first: an upgrade never takes a ready replica down first"
    assert (
        rendered["spec"]["replicas"]
        == step.Target.of(RELEASE_CONFIG, cpu_section()).replicas
    ), "the alarms' N is the rendered replica count"


def test_the_vendor_evidence_is_pinned() -> None:
    assert step.PUT_METRIC_ALARM_EVIDENCE["url"] == (
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/APIReference/"
        "API_PutMetricAlarm.html"
    )
    assert step.PUT_METRIC_ALARM_EVIDENCE["quotes"][1] == (
        "When you update an existing alarm, its state is left unchanged, but the "
        "update completely overwrites the previous configuration of the alarm."
    )
    assert step.SOFTWARE_UPDATE_EVIDENCE["quotes"][0].endswith(
        "Pending -> InProgress -> RollbackInProgress -> Failed"
    ), "the rollback path a rollback alarm takes"
    assert step.UPDATING_SOFTWARE == {"Pending", "InProgress", "RollbackInProgress"}
