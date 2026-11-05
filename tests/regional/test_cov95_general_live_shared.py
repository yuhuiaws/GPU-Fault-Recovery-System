"""GF-GENERAL-LIVE-005 shared-filesystem variants: refusals, controls and verdicts.

Built on the fakes of ``test_general_live_shared`` (export server, node-side
shared probe, store activity documents).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_shared as shared
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from tests.regional._general_live_fakes import store_script
from tests.regional.test_general_live_shared import (
    NODE,
    PEER,
    SIZE,
    SPECS,
    T0,
    context,
    export_source,
    failure_domain,
    hang_activity,
    labelled,
    sample,
    servers_by_suffix,
    shared_probe,
)
from tests.regional.test_general_live_shared import fake_server as fake_server

HUNG = {"statvfs": {"timed_out": True, "elapsed": 5.0}, "host_mount": {}}
ANSWERED = {"statvfs": {"used_percent": 2.0, "total_bytes": SIZE}, "host_mount": {}}


def mount_disappears_context(tmp_path: Path, **script: Any) -> tuple[Any, str]:
    probe = shared_probe(NODE)
    probe.script.update(script)
    ctx = context(
        "mount_disappears", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script(
        {
            "activity": lambda _args: hang_activity(
                mount, labels=labelled(shared.MODE_MOUNT_MISSING)
            )
        }
    )
    return ctx, mount


def test_a_umount_that_leaves_the_host_mount_is_inconclusive(tmp_path: Path) -> None:
    ctx, _ = mount_disappears_context(
        tmp_path,
        **{"nfs-umount": {"host_mount": {"fstype": "nfs4"}, "directory_present": True}},
    )

    with pytest.raises(
        VariantInconclusive, match="the test export is still in the host mount table"
    ):
        SPECS["mount_disappears"].execute(ctx)

    names = [call[0] for call in ctx.probes[f"shared@{NODE}"].calls]
    assert "nfs-remount" not in names, names


def test_a_remount_that_fails_leaves_no_recovery_to_prove(tmp_path: Path) -> None:
    ctx, mount = mount_disappears_context(
        tmp_path, **{"nfs-remount": {"host_mount": None}}
    )

    with pytest.raises(VariantInconclusive) as caught:
        SPECS["mount_disappears"].execute(ctx)

    assert str(caught.value) == (
        f"the test export could not be remounted at {mount}: no recovery to prove"
    )


def test_a_short_fill_without_byte_counts_or_error_names_only_the_percentage(
    tmp_path: Path,
) -> None:
    probe = shared_probe(NODE)
    probe.script["nfs-fill"] = {
        "fill": {"fill_bytes": None, "enospc": False},
        "statvfs": {"used_percent": 61.25},
    }
    ctx = context(
        "capacity_full", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})

    with pytest.raises(VariantInconclusive) as caught:
        SPECS["capacity_full"].execute(ctx)

    assert str(caught.value) == "the test export only reached 61.2 % used"


def test_healthy_other_mount_needs_a_second_export_that_still_answers(
    tmp_path: Path,
) -> None:
    probe = shared_probe(NODE)

    def stat(_arguments: tuple[str, ...]) -> dict[str, Any]:
        # The pause of export a hangs every NFS mount on the node.
        return dict(HUNG if servers_by_suffix()["a"].paused else ANSWERED)

    probe.script["nfs-stat"] = stat
    ctx = context(
        "healthy_other_mount", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    other = shared.mount_path(ctx, shared.OTHER)
    document = hang_activity(mount, other=other)
    document["batches"][0]["samples"] += [
        sample(shared.UNAVAILABLE, other, 0.0),
        sample(shared.USED, other, 2.0),
    ]
    ctx.regional.cpu_python_responder = store_script({"activity": document})

    with pytest.raises(VariantInconclusive, match="the second export hung too"):
        SPECS["healthy_other_mount"].execute(ctx)

    assert servers_by_suffix()["a"].calls[-2:] == ["pause", "resume"], (
        "the paused export is resumed before the refusal"
    )


def pair_context(tmp_path: Path, **probes: Any) -> Any:
    ctx = context(
        "shared_failure_domain", tmp_path, activity=None, peer=PEER, probes=probes
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)

    def per_node(arguments: tuple[str, ...]) -> dict[str, Any]:
        return hang_activity(
            mount,
            node=arguments[1],
            source=export_source(),
            domain=failure_domain(export_source()),
        )

    ctx.regional.cpu_python_responder = store_script({"activity": per_node})
    return ctx


def test_the_failure_domain_needs_the_pause_to_hang_both_clients(
    tmp_path: Path,
) -> None:
    peer_probe = shared_probe(PEER)
    peer_probe.script["nfs-stat"] = dict(ANSWERED)
    ctx = pair_context(tmp_path, **{f"shared@{PEER}": peer_probe})

    with pytest.raises(
        VariantInconclusive, match="the paused export still answered on a client"
    ):
        SPECS["shared_failure_domain"].execute(ctx)

    assert servers_by_suffix()["a"].paused is False, "resumed before the refusal"


@pytest.mark.parametrize(
    ("metadata", "snapshot", "fragment"),
    [
        ({"labels": {}}, None, f"{PEER} is a unknown node"),
        (
            None,
            {"name": PEER, "ready": "True", "unschedulable": True},
            f"{PEER} is not Ready and schedulable",
        ),
        (
            None,
            {"name": PEER, "ready": "False", "unschedulable": False},
            f"{PEER} is not Ready and schedulable",
        ),
    ],
)
def test_the_failure_domain_refuses_a_peer_that_cannot_take_the_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    fragment: str,
) -> None:
    ctx = pair_context(tmp_path)
    regional = ctx.regional
    if metadata is not None:
        monkeypatch.setattr(regional, "node_metadata", lambda node: dict(metadata))
    if snapshot is not None:
        original = regional.node_snapshot
        monkeypatch.setattr(
            regional,
            "node_snapshot",
            lambda node: dict(snapshot) if node == PEER else original(node),
        )

    with pytest.raises(VariantNotRun, match=fragment) as caught:
        SPECS["shared_failure_domain"].execute(ctx)

    assert caught.value.reason == "FIXTURE_NOT_READY"
    assert servers_by_suffix() == {}, "nothing is started for a refused peer"


@pytest.mark.parametrize(
    ("scope", "level"), [({"run_level": True}, "run-level"), ({}, "mount-scoped")]
)
def test_a_mount_cleanup_that_leaves_residue_fails_its_stack_entry(
    tmp_path: Path, scope: dict[str, Any], level: str
) -> None:
    probe = shared_probe(NODE)
    probe.script["cleanup"] = {
        "clean": False,
        "scope": scope,
        "residue": {"timers": ["gfrs-deadman.timer"]},
    }
    ctx = context(
        "stat_timeout", tmp_path, activity=None, probes={f"shared@{NODE}": probe}
    )
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})
    assert SPECS["stat_timeout"].execute(ctx)["verdict"] == "PASS"

    ctx.cleanup.run()

    failures = [str(item) for item in ctx.cleanup.failed]
    assert any(
        f"the shared mount cleanup of shared on {NODE} ({level}) left residue: "
        "{'timers': ['gfrs-deadman.timer']}" in item
        for item in failures
    ), failures
    assert servers_by_suffix()["a"].deleted is True, "the export is still deleted"


def test_export_source_errors_name_a_reading_of_another_export() -> None:
    mount = "/mnt/gfrs-live/abc/shared"
    activity = hang_activity(mount, source="10.0.9.7:/elsewhere")
    server = SimpleNamespace(ready={"ip": "10.0.9.1", "export": "/a"})

    assert shared.export_source_errors(activity, mount, server) == [
        f"{shared.UNAVAILABLE} on {mount} names export ['10.0.9.7:/elsewhere'], "
        "not 10.0.9.1:/a"
    ]


def test_failure_domain_errors_name_duplicate_records_and_ignore_other_mail() -> None:
    export = "10.0.9.1:/a"
    mount = "/mnt/gfrs-live/abc/shared"
    activities = {
        node: hang_activity(
            mount, node=node, source=export, domain=failure_domain(export)
        )
        for node in (NODE, PEER)
    }
    duplicate = dict(activities[PEER]["incidents"][-1], incident_id="inc-domain-2")
    activities[PEER]["incidents"].append(duplicate)
    activities[NODE]["notifications"].append(
        {"notification_id": "ntf-node", "category": "NODE_FAULT", "incident_id": "x"}
    )
    ids = {node: [f"inc-{node[-3:]}"] for node in (NODE, PEER)}

    errors = shared.failure_domain_errors(activities, (NODE, PEER), export, ids)

    assert errors == [
        f"2 {shared.DOMAIN_EVENT_TYPE} records for {export}: inc-domain-1, inc-domain-2"
    ]


def test_domain_state_of_a_record_the_activity_does_not_hold_is_none() -> None:
    activity = hang_activity(
        "/mnt/gfrs-live/abc/shared", source="10.0.9.1:/a", domain=failure_domain("x")
    )

    assert shared.domain_state(activity, "inc-domain-1") == "RECOVERED"
    assert shared.domain_state(activity, "inc-domain-9") is None


def test_wait_quiet_waits_one_summary_plus_four_intervals_for_the_finding(
    tmp_path: Path,
) -> None:
    ctx = context("stat_timeout", tmp_path, activity=None)
    mount = shared.mount_path(ctx, shared.PRIMARY)
    ctx.regional.cpu_python_responder = store_script({"activity": hang_activity(mount)})

    found = shared.wait_quiet(
        ctx, NODE, T0 - timedelta(minutes=5), mount, name="gap-window"
    )

    assert shared.finding_incidents(found, shared.UNAVAILABLE, device=mount), (
        "the gap window returns the read that holds the finding"
    )
