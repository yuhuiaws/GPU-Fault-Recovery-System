"""The ``cpu_rolling_update`` status check and the deploy's ``cpu_update_window``
preflight (batch 240).

The check passes only when the CPU group rolls, its AutoRollbackConfiguration
names the product's two alarms, both exist and are OK, and they watch the
current API NLB; every other state fails with the remedy, an alarm put within
the grace is PENDING, and in ``verify`` -- the deploy's gate, before the step
-- it is never registered. The window check refuses a deploy while the CPU
group is being re-imaged and warns when the deploy may overlap its next
scheduled window.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cpu_rolling_update as step
from gpu_fault_release import regional_admin_commands as commands
from gpu_fault_release import regional_cpu_rolling_update as check
from gpu_fault_release.regional_release_errors import CheckFailure
from tests.admin._cpu_rolling_world import CAPACITY, NOT_RESTORED, live_alarms
from tests.admin.test_admin_hyperpod_rollback import (
    CPU_GROUP,
    CPU_NAME,
    LB,
    LB_ARN,
    LIVE_BEFORE,
    LIVE_ROLLING,
    NLB_NAME,
    OLD_LB,
    OLD_TG,
    REGION,
    TG,
    TG_ARN,
    _described,
)

NOW = datetime(2026, 11, 2, 2, 0, tzinfo=UTC)
NAMES = (CAPACITY, NOT_RESTORED)
NLB = {"name": NLB_NAME, "load_balancer": LB, "target_groups": [TG]}


def _alarms(**options: Any) -> dict[str, Any]:
    return {
        "MetricAlarms": list(live_alarms(**options).values()),
        "CompositeAlarms": [],
    }


def _evaluate(
    schedule: dict[str, Any] | None = LIVE_ROLLING,
    alarms: dict[str, Any] | None = None,
    *,
    mode: str = "rolling",
    journal: dict[str, Any] | None = None,
    nlb: dict[str, Any] | None = NLB,
) -> tuple[str, str, dict[str, Any]]:
    return check.evaluate(
        _described(schedule),
        alarms if alarms is not None else _alarms(),
        nlb,
        mode=mode,
        names=NAMES,
        journal=journal or {"groups": {}},
        now=NOW,
    )


# --- every state of the check ----------------------------------------------------------


def test_ok_passes() -> None:
    verdict, summary, details = _evaluate()

    assert verdict == "PASS", summary
    assert summary == (
        "the CPU control plane rolls one batch at a time, rolled back on "
        f"{CAPACITY} and {NOT_RESTORED} (both OK, on the current API NLB)"
    )
    assert details["problems"] == [] and details["pending"] == []


def test_a_missing_alarm_fails_with_the_remedy() -> None:
    alarms = _alarms()
    alarms["MetricAlarms"] = alarms["MetricAlarms"][:1]

    verdict, summary, _details = _evaluate(alarms=alarms)

    assert verdict == "FAIL"
    assert f"rollback alarm {NOT_RESTORED} does not exist" in summary
    assert summary.endswith(
        "re-run gpu-fault-admin deploy, or gpu-fault-admin cpu-rolling-update "
        "--state-dir <state dir> (the deploy's same last step); gpu-fault-admin "
        "hyperpod-scheduled-update --rolling --rollback-alarm ... prints the same "
        "requests for a manual change (administrator-operations §5.0.5)"
    ), summary


def test_insufficient_data_fails_and_is_pending_just_after_a_put() -> None:
    stale = _alarms(state="INSUFFICIENT_DATA")

    verdict, summary, _details = _evaluate(alarms=stale)
    assert verdict == "FAIL" and "is INSUFFICIENT_DATA" in summary

    for alarm in stale["MetricAlarms"]:
        alarm["AlarmConfigurationUpdatedTimestamp"] = (
            NOW - timedelta(minutes=4)
        ).isoformat()
    verdict, summary, details = _evaluate(alarms=stale)
    assert verdict == "PENDING", summary
    assert summary.startswith(
        f"PENDING: rollback alarm {CAPACITY} is INSUFFICIENT_DATA"
    ), summary
    assert len(details["pending"]) == 2 and details["problems"] == []

    for alarm in stale["MetricAlarms"]:
        alarm["AlarmConfigurationUpdatedTimestamp"] = (
            NOW - check.GRACE - timedelta(minutes=1)
        ).isoformat()
    assert _evaluate(alarms=stale)[0] == "FAIL", "past the grace it fails"


def test_alarms_on_the_previous_nlb_fail() -> None:
    verdict, summary, _details = _evaluate(alarms=_alarms(lb=OLD_LB, tg=OLD_TG))

    assert verdict == "FAIL"
    assert f"not the current API NLB {LB} (target group {TG})" in summary


def test_a_schedule_not_naming_the_alarms_fails() -> None:
    other = json.loads(json.dumps(LIVE_ROLLING))
    other["DeploymentConfig"]["AutoRollbackConfiguration"] = [{"AlarmName": "theirs"}]

    verdict, summary, _details = _evaluate(other)

    assert verdict == "FAIL"
    assert (
        f"instance group {CPU_GROUP}'s AutoRollbackConfiguration does not name "
        f"{CAPACITY} and {NOT_RESTORED}"
    ) in summary


def test_an_all_at_once_schedule_fails() -> None:
    verdict, summary, _details = _evaluate(LIVE_BEFORE)

    assert verdict == "FAIL"
    assert f"instance group {CPU_GROUP} re-images every instance at once" in summary


def test_a_deferred_step_fails_with_its_reason() -> None:
    journal = {
        "groups": {
            CPU_GROUP: {
                "phase": "deferred",
                "reason": "SoftwareUpdateStatus is InProgress (an AMI update is in flight)",
            }
        }
    }

    verdict, summary, _details = _evaluate(LIVE_BEFORE, journal=journal)

    assert verdict == "FAIL"
    assert (
        f"the deploy's CPU rolling schedule step for {CPU_GROUP} was deferred: "
        "SoftwareUpdateStatus is InProgress"
    ) in summary


def test_a_step_stopped_between_delete_and_create_fails() -> None:
    journal = {
        "groups": {
            CPU_GROUP: {"phase": "deleted", step.AWAITING_CREATE: True, "reason": ""}
        }
    }

    verdict, summary, _details = _evaluate(None, journal=journal)

    assert verdict == "FAIL", "no schedule, but the step removed it: not a PASS"
    assert (
        f"the deploy's CPU rolling schedule step for {CPU_GROUP} stopped at deleted "
        "after removing the group's ScheduledUpdateConfig, before creating it again"
    ) in summary


def test_unchanged_passes_unless_a_schedule_still_names_the_alarms() -> None:
    verdict, summary, _details = _evaluate(LIVE_BEFORE, mode="unchanged")
    assert (verdict, summary.split(":")[0]) == ("PASS", "unchanged")

    stale = _alarms(lb=OLD_LB, tg=OLD_TG)
    assert _evaluate(LIVE_ROLLING, stale, mode="unchanged")[0] == "FAIL", (
        "alarms a schedule still names are judged in any mode"
    )


def test_no_schedule_passes() -> None:
    verdict, summary, _details = _evaluate(None)

    assert (verdict, summary) == (
        "PASS",
        "no CPU instance group has a ScheduledUpdateConfig",
    )


# --- the check and its reads ---------------------------------------------------------


class Runner:
    """The engine runner: ``aws ... --region R --output json`` answers."""

    def __init__(self, described: dict[str, Any], alarms: dict[str, Any]) -> None:
        self.described = described
        self.alarms = alarms
        self.calls: list[list[str]] = []
        self.fail: str | None = None

    def run(self, command: list[str], **_options: Any) -> str:
        self.calls.append(list(command))
        if self.fail:
            raise RuntimeError(self.fail)
        assert command[-4:] == ["--region", REGION, "--output", "json"], command
        key = tuple(command[1:3])
        if key == ("sagemaker", "describe-cluster"):
            return json.dumps(self.described)
        if key == ("cloudwatch", "describe-alarms"):
            return json.dumps(self.alarms)
        if key == ("elbv2", "describe-load-balancers"):
            return json.dumps({"LoadBalancers": [{"LoadBalancerArn": LB_ARN}]})
        if key == ("elbv2", "describe-target-groups"):
            return json.dumps({"TargetGroups": [{"TargetGroupArn": TG_ARN}]})
        raise AssertionError(command)


def _release(runner: Runner, state_dir: Path | None = None) -> Any:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(
            admin_state_dir=state_dir,
            aws_region=REGION,
            cpu_hyperpod_cluster_name=CPU_NAME,
            nlb={"name": NLB_NAME},
        ),
    )


def test_status_fails_the_check_with_its_evidence() -> None:
    release = _release(Runner(_described(LIVE_BEFORE), _alarms()))

    with pytest.raises(CheckFailure) as raised:
        check.check_cpu_rolling_update(release, mode="status", now=lambda: NOW)

    assert "re-images every instance at once" in str(raised.value)
    assert raised.value.details["mode"] == "rolling"


def test_status_passes_on_the_live_shape() -> None:
    release = _release(Runner(_described(LIVE_ROLLING), _alarms()))

    value = check.check_cpu_rolling_update(release, mode="status", now=lambda: NOW)

    assert value.status == "PASS" and value.summary.startswith("the CPU control plane")


def test_verify_never_fails_on_what_the_last_step_repairs() -> None:
    runner = Runner(_described(LIVE_BEFORE), _alarms())

    value = check.check_cpu_rolling_update(
        _release(runner), mode="verify", now=lambda: NOW
    )
    assert value.status == "WARN" and value.summary.startswith(
        "pending the deploy's last"
    )

    runner.fail = "AccessDenied: cloudwatch:DescribeAlarms"
    value = check.check_cpu_rolling_update(
        _release(runner), mode="verify", now=lambda: NOW
    )
    assert value.status == "WARN" and "AccessDenied" in value.summary
    with pytest.raises(RuntimeError, match="AccessDenied"):
        check.check_cpu_rolling_update(_release(runner), mode="status", now=lambda: NOW)


def test_the_check_is_registered_for_status_only() -> None:
    release = _release(Runner(_described(LIVE_ROLLING), _alarms()))

    assert check.health_checks(release, "verify") == []
    ((name, function),) = check.health_checks(release, "status")
    assert name == "cpu_rolling_update" == check.CHECK
    assert function().status in {"PASS", "WARN"}


def test_the_check_reads_the_configured_names_and_the_journal(tmp_path: Path) -> None:
    journal = step.journal_path(tmp_path)
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "alarms": {},
                "groups": {
                    CPU_GROUP: {"phase": "failed", "reason": "the create failed"}
                },
            }
        ),
        encoding="utf-8",
    )
    release = _release(Runner(_described(LIVE_ROLLING), _alarms()), tmp_path)

    with pytest.raises(CheckFailure, match="was failed: the create failed"):
        check.check_cpu_rolling_update(release, mode="status", now=lambda: NOW)


# --- the deploy's update window --------------------------------------------------------


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"SoftwareUpdateStatus": "InProgress"}, "SoftwareUpdateStatus is InProgress"),
        ({"SoftwareUpdateStatus": "Pending"}, "SoftwareUpdateStatus is Pending"),
        ({"Status": "Updating"}, "the instance group is Updating"),
        ({"Status": "SystemUpdating"}, "the instance group is SystemUpdating"),
    ],
)
def test_a_deploy_refuses_to_start_during_an_ami_update(
    changes: dict[str, Any], reason: str
) -> None:
    described = _described(LIVE_ROLLING)
    described["InstanceGroups"][0].update(changes)

    with pytest.raises(CheckFailure, match="the deploy refuses to start") as raised:
        check.cpu_update_window(_release(Runner(described, {})), now=lambda: NOW)

    assert reason in str(raised.value)


def test_a_deploy_overlapping_the_next_window_is_warned_with_the_times() -> None:
    # Sunday 07:00Z, an hour before cron(0 8 ? * SUN *): the deploy may run into it.
    sunday = datetime(2026, 11, 8, 7, 0, tzinfo=UTC)

    value = check.cpu_update_window(
        _release(Runner(_described(LIVE_ROLLING), {})), now=lambda: sunday
    )

    assert value.status == "WARN"
    assert (
        "2026-11-08T08:00:00+00:00 - about 2026-11-08T10:00:00+00:00" in value.summary
    )
    calm = check.cpu_update_window(
        _release(Runner(_described(LIVE_ROLLING), {})), now=lambda: NOW
    )
    assert calm.status == "PASS"


def test_an_unreadable_cluster_only_warns() -> None:
    runner = Runner(_described(LIVE_ROLLING), {})
    runner.fail = "throttled"

    value = check.cpu_update_window(_release(runner), now=lambda: NOW)

    assert value.status == "WARN" and "throttled" in value.summary


def test_every_deploy_preflight_report_gains_the_window_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _release(Runner(_described(LIVE_ROLLING), {}))
    release.config.site_name = "site"
    inner = {"checks": [{"name": "aws_identity", "status": "PASS", "summary": "ok"}]}
    monkeypatch.setattr(commands, "_deploy_preflight_report", lambda _release: inner)

    report = commands.build_deploy_preflight_report(release)

    assert [item["name"] for item in report["checks"]] == [
        "aws_identity",
        "cpu_update_window",
    ]
    assert report["healthy"] is True
    monkeypatch.setattr(
        commands, "_deploy_preflight_report", lambda _r: {"healthy": True}
    )
    assert commands.build_deploy_preflight_report(release) == {"healthy": True}


# --- the engine mode -------------------------------------------------------------------


def test_run_step_records_what_the_step_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def finish(release_config, cpu_control_plane, state_dir):
        seen.update(config=release_config, mode=cpu_control_plane.rolling_update.mode)
        return {"status": "DONE", "at": "2026-11-02T02:00:00Z", "warnings": []}

    monkeypatch.setattr(check, "finish_deploy", finish)
    release = _release(Runner({}, {}), tmp_path)
    release.config.cpu_eks_arn = "arn:aws:eks:us-west-2:123456789012:cluster/cpu-eks"
    release.config.site_name = "site"

    result = check.run_step(release)

    assert result["status"] == "DONE" and seen["mode"] == "rolling"
    assert seen["config"]["nlb"] == {"name": NLB_NAME}
    assert (
        step.load_last_run(tmp_path, since="2026-11-02T01:00:00Z")["status"] == "DONE"
    )
    assert check.run_step(_release(Runner({}, {})))["status"] == "SKIPPED"
