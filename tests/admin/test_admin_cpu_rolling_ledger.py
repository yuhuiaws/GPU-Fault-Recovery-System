"""The CPU rolling update in the installation ledger and in uninstall (batch 240).

The deploy records every rollback alarm it put as a ``cloudwatch_alarm`` row
(CREATED/DELETE) and the CPU group's pre-install ``ScheduledUpdateConfig`` as
attributes of the ``cpu_hyperpod`` row. Uninstall restores that schedule first
-- refused, before anything is deleted, while an AMI update is in flight --
and then deletes the alarms with a tag-proven ownership, until
``describe-alarms`` no longer finds them (the residue gate). A site whose
ledger records neither is left exactly as before.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin import cpu_rolling_update as step
from gpu_fault.admin.aws_cleanup import (
    DELETE_PRIORITY,
    SUPPORTED_RESOURCE_TYPES,
    ResourceCleaner,
)
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.resource_records import record
from gpu_fault.admin.resource_registry_cpu_rolling import (
    ALARM_TYPE,
    PRE_INSTALL_PREFIX,
    cpu_rolling_rows,
    pre_install_records,
)
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.admin.uninstall_cpu_rolling import STATE_KEY, restore_cpu_rolling_update
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin._aws_cleanup_support import ACCOUNT as CLEANUP_ACCOUNT
from tests.admin._aws_cleanup_support import REGION as CLEANUP_REGION
from tests.admin._aws_cleanup_support import TAGS, Aws, resource
from tests.admin._cpu_rolling_world import CAPACITY, NOT_RESTORED, RELEASE_CONFIG, World
from tests.admin.test_admin_hyperpod_rollback import (
    ACCOUNT,
    CPU_GROUP,
    CPU_NAME,
    LIVE_BEFORE,
    LIVE_ROLLING,
    REGION,
    _described,
)
from tests.admin.test_admin_site import site_file

SITE_KEY = "registry-site-1"


def _site(tmp_path: Path, cluster: str = CPU_NAME) -> Any:
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        registry_site_id=SITE_KEY,
        release_config={**RELEASE_CONFIG, "cpu_hyperpod_cluster_name": cluster},
    )


def _hyperpod_row(cluster: str = CPU_NAME) -> Any:
    return record(
        site_id=SITE_KEY,
        resource_key=f"cluster/cpu_hyperpod/{cluster}",
        resource_type="cpu_hyperpod",
        resource_id=cluster,
        region=REGION,
        account_id=ACCOUNT,
        ownership=InstallationResourceOwnership.EXTERNAL,
        delete_policy=InstallationResourceDeletePolicy.PRESERVE,
    )


def _journal(tmp_path: Path, **journal: Any) -> None:
    path = step.journal_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "groups": {}, "alarms": {}, **journal}),
        encoding="utf-8",
    )


def _arn(name: str) -> str:
    return f"arn:aws:cloudwatch:{REGION}:{ACCOUNT}:alarm:{name}"


# --- the ledger -----------------------------------------------------------------------


def test_the_ledger_records_every_alarm_put_and_the_pre_install_schedule(
    tmp_path: Path,
) -> None:
    _journal(
        tmp_path,
        alarms={
            CAPACITY: {"name": CAPACITY, "arn": _arn(CAPACITY)},
            "renamed-old": {"name": "renamed-old", "arn": _arn("renamed-old")},
        },
        groups={
            CPU_GROUP: {
                "pre_install": {"scheduled_update_config": LIVE_BEFORE},
                "phase": "verified",
            }
        },
    )

    rows = cpu_rolling_rows(
        [_hyperpod_row()], _site(tmp_path), region=REGION, account_id=ACCOUNT
    )

    alarms = [row for row in rows if row.resource_type == ALARM_TYPE]
    assert [row.resource_id for row in alarms] == [CAPACITY, "renamed-old"], (
        "a renamed pair keeps its old rows, so uninstall still deletes them"
    )
    for row in alarms:
        assert row.ownership is InstallationResourceOwnership.CREATED
        assert row.delete_policy is InstallationResourceDeletePolicy.DELETE
        assert row.resource_arn == _arn(row.resource_id)
        assert row.resource_key == f"aws/cloudwatch/alarm/{row.resource_id}"
        assert (row.region, row.account_id) == (REGION, ACCOUNT)
    (hyperpod,) = [row for row in rows if row.resource_type == "cpu_hyperpod"]
    assert json.loads(hyperpod.attributes[f"{PRE_INSTALL_PREFIX}{CPU_GROUP}"]) == (
        LIVE_BEFORE
    )
    assert pre_install_records(rows) == {CPU_GROUP: LIVE_BEFORE}


def test_a_site_with_no_journal_adds_nothing_to_the_ledger(tmp_path: Path) -> None:
    rows = [_hyperpod_row()]

    assert cpu_rolling_rows(
        rows, _site(tmp_path), region=REGION, account_id=ACCOUNT
    ) == (rows)
    assert pre_install_records(rows) == {}


# --- the cleanup ----------------------------------------------------------------------


@pytest.fixture
def site(tmp_path: Path) -> RenderedSite:
    return load_site(site_file(tmp_path))


def _alarm_row(name: str = CAPACITY) -> Any:
    return resource(
        ALARM_TYPE,
        name,
        arn=f"arn:aws:cloudwatch:{CLEANUP_REGION}:{CLEANUP_ACCOUNT}:alarm:{name}",
    )


def test_uninstall_deletes_a_tagged_alarm_until_it_is_gone(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted: list[str] = []

    def describe(arguments: list[str]) -> dict[str, Any]:
        name = arguments[arguments.index("--alarm-names") + 1]
        return {"MetricAlarms": [] if name in deleted else [{"AlarmName": name}]}

    def delete(arguments: list[str]) -> dict[str, Any]:
        deleted.append(arguments[arguments.index("--alarm-names") + 1])
        return {}

    aws = Aws(
        {
            ("cloudwatch", "describe-alarms"): describe,
            ("cloudwatch", "list-tags-for-resource"): {"Tags": TAGS},
            ("cloudwatch", "delete-alarms"): delete,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)

    ResourceCleaner(site).delete(_alarm_row())

    assert deleted == [CAPACITY]
    assert [call[2] for call in aws.calls if call[1] == "cloudwatch"][-1] == (
        "describe-alarms"
    ), "the residue check reads it back: gone"
    assert ALARM_TYPE in SUPPORTED_RESOURCE_TYPES and DELETE_PRIORITY[ALARM_TYPE] == 10


@pytest.mark.parametrize(
    "tags", [[], [{"Key": "gpu-fault:site-id", "Value": "another-site"}]]
)
def test_an_alarm_without_this_sites_tag_is_never_deleted(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch, tags: list[dict[str, str]]
) -> None:
    aws = Aws(
        {
            ("cloudwatch", "describe-alarms"): {
                "MetricAlarms": [{"AlarmName": CAPACITY}]
            },
            ("cloudwatch", "list-tags-for-resource"): {"Tags": tags},
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)

    with pytest.raises(BootstrapError, match="site"):
        ResourceCleaner(site).delete(_alarm_row())
    assert aws.mutations == []


def test_an_alarm_already_gone_is_not_a_failure(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws = Aws({("cloudwatch", "describe-alarms"): {"MetricAlarms": []}})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)

    ResourceCleaner(site).delete(_alarm_row())

    assert aws.mutations == []


def test_a_denied_alarm_read_is_not_absence(
    site: RenderedSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = subprocess.CompletedProcess(
        ["aws"], 254, stdout="", stderr="An error occurred (AccessDenied)"
    )
    aws = Aws({("cloudwatch", "describe-alarms"): denied})
    monkeypatch.setattr(aws_commands, "bounded_command", aws)

    with pytest.raises(BootstrapError, match="AccessDenied"):
        ResourceCleaner(site).delete(_alarm_row())
    assert aws.mutations == []


# --- the restore ---------------------------------------------------------------------


def _request(tmp_path: Path, disposition: str = "keep") -> Any:
    return SimpleNamespace(site=_site(tmp_path), cpu_disposition=disposition)


def _snapshot(*rows: Any) -> InstallationResourceSnapshot:
    return InstallationResourceSnapshot(site_id=SITE_KEY, resources=list(rows))


def _transitions() -> tuple[list[dict[str, Any]], Any]:
    seen: list[dict[str, Any]] = []

    def transition(_path: Path, state: dict[str, Any], phase: str, **updates: Any):
        state.update(updates)
        seen.append(dict(updates))

    return seen, transition


def test_a_site_with_nothing_recorded_is_left_exactly_as_before(tmp_path: Path) -> None:
    world = World(_described(LIVE_ROLLING))
    seen, transition = _transitions()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    restore_cpu_rolling_update(
        _request(tmp_path),
        world,
        _snapshot(_hyperpod_row()),
        tmp_path,
        state,
        transition,
    )

    assert world.calls == [] and seen == [] and STATE_KEY not in state


def test_uninstall_restores_the_recorded_schedule_before_the_alarms_go(
    tmp_path: Path, capsys
) -> None:
    world = World(_described(LIVE_ROLLING))
    hyperpod = _hyperpod_row().model_copy(
        update={
            "attributes": {f"{PRE_INSTALL_PREFIX}{CPU_GROUP}": json.dumps(LIVE_BEFORE)}
        }
    )
    seen, transition = _transitions()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    restore_cpu_rolling_update(
        _request(tmp_path),
        world,
        _snapshot(hyperpod),
        tmp_path,
        state,
        transition,
        clock=world.clock.clock(),
    )

    group = world.hyperpod.describe()["InstanceGroups"][0]
    assert group["ScheduledUpdateConfig"] == LIVE_BEFORE, (
        "the ledger's pre-install state"
    )
    assert state[STATE_KEY]["outcome"] == "done"
    assert state[STATE_KEY]["groups"] == [
        {"group": CPU_GROUP, "outcome": "restored", "schedule": LIVE_BEFORE}
    ]
    assert state[STATE_KEY]["follow_up"] == [], "nothing names the deleted alarms now"
    assert capsys.readouterr().err == ""
    world.calls.clear()
    restore_cpu_rolling_update(
        _request(tmp_path),
        world,
        _snapshot(hyperpod),
        tmp_path,
        state,
        transition,
        clock=world.clock.clock(),
    )
    assert world.calls == [], "journaled: a resume skips it"


def test_the_restore_is_refused_during_an_update_and_uninstall_stops(
    tmp_path: Path,
) -> None:
    described = _described(LIVE_ROLLING)
    described["InstanceGroups"][0]["SoftwareUpdateStatus"] = "RollbackInProgress"
    world = World(described)
    hyperpod = _hyperpod_row().model_copy(
        update={
            "attributes": {f"{PRE_INSTALL_PREFIX}{CPU_GROUP}": json.dumps(LIVE_BEFORE)}
        }
    )
    seen, transition = _transitions()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    with pytest.raises(BootstrapError, match="RollbackInProgress"):
        restore_cpu_rolling_update(
            _request(tmp_path),
            world,
            _snapshot(hyperpod),
            tmp_path,
            state,
            transition,
            clock=world.clock.clock(),
        )
    assert world.writes == [] and seen == [] and STATE_KEY not in state


def test_a_schedule_still_naming_the_alarms_gets_the_helper_command(
    tmp_path: Path, capsys
) -> None:
    world = World(_described(LIVE_ROLLING))
    alarms = [
        record(
            site_id=SITE_KEY,
            resource_key=f"aws/cloudwatch/alarm/{name}",
            resource_type=ALARM_TYPE,
            resource_id=name,
            resource_arn=_arn(name),
            region=REGION,
            account_id=ACCOUNT,
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.DELETE,
        )
        for name in (CAPACITY, NOT_RESTORED)
    ]
    seen, transition = _transitions()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    restore_cpu_rolling_update(
        _request(tmp_path),
        world,
        _snapshot(_hyperpod_row(), *alarms),
        tmp_path,
        state,
        transition,
    )

    (line,) = state[STATE_KEY]["follow_up"]
    assert line.startswith(
        f"the CPU instance group {CPU_GROUP} still names the rollback alarms "
        f"{CAPACITY}, {NOT_RESTORED} in its AutoRollbackConfiguration, and uninstall "
        "deletes them. AWS does not document what a scheduled update does when a "
        "listed alarm does not exist"
    ), line
    assert "gpu-fault-admin hyperpod-scheduled-update --state-dir" in line
    assert f"--instance-group {CPU_GROUP}" in line
    assert line in capsys.readouterr().err
    assert world.writes == [], "uninstall does not change that schedule itself"


def test_retiring_the_cpu_cluster_skips_the_restore(tmp_path: Path) -> None:
    world = World(_described(LIVE_ROLLING))
    hyperpod = _hyperpod_row().model_copy(
        update={
            "attributes": {f"{PRE_INSTALL_PREFIX}{CPU_GROUP}": json.dumps(LIVE_BEFORE)}
        }
    )
    seen, transition = _transitions()
    state: dict[str, Any] = {"phase": "REGISTRY_EXPORTED"}

    restore_cpu_rolling_update(
        _request(tmp_path, "delete"),
        world,
        _snapshot(hyperpod),
        tmp_path,
        state,
        transition,
    )

    assert state[STATE_KEY] == {
        "outcome": "skipped",
        "reason": "the CPU cluster is deleted",
    }
    assert world.calls == []
