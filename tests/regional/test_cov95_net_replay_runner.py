"""NET-001 complete virtual outage, rollback failures and metadata refusal."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import run_net001_collector_replay as module
from tests.regional import _cov95_net_replay as replay_support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401

replay_runner = replay_support.replay_runner


def result(host: Any) -> dict[str, Any]:
    return json.loads((host.runner.case_dir / f"{module.CASE_ID}.json").read_text())


def test_full_outage_replays_once_then_checks_reconnect_and_resource_absence(
    replay_runner: Any,
) -> None:
    host = replay_runner
    assert host.runner.run() == 0
    report = result(host)
    assert report["verdict"] == "PASS"
    assert all(report["checks"].values()), report["checks"]
    assert len(host.runner.timeline) == 12
    assert host.clock.now == 1605
    assert host.resources == {}
    assert not host.armed and not host.blocked, "all owned rollback tags must retire"
    pod = next(value for value in host.manifests if value["kind"] == "Pod")
    assert pod["spec"]["nodeName"] == "node-a"
    assert pod["spec"]["activeDeadlineSeconds"] == module.POD_DEADLINE_SECONDS
    assert pod["spec"]["restartPolicy"] == "Never"
    assert report["release_id"] == "release-a"


def test_rollback_arm_ack_loss_still_cleans_its_tag(replay_runner: Any) -> None:
    host = replay_runner
    host.fail_at = "arm-ack"
    assert host.runner.run() == 1
    assert not host.armed, "lost arm ACK must not orphan the independent rollback timer"
    assert result(host)["verdict"] == "FAIL"


def test_cleanup_read_error_cannot_prove_resource_absence(replay_runner: Any) -> None:
    host = replay_runner
    host.cleanup_read_error = True
    assert host.runner.run() == 1
    report = result(host)
    assert report["verdict"] == "FAIL"
    assert "resource cleanup failed" in report["error"]


@pytest.mark.parametrize(
    "problem",
    [
        "workload",
        "late-workload",
        "outage-ids",
        "rule-drift",
        "leaked-event",
        "replay-timeout",
        "workflow",
        "block-reachable",
        "cleanup-rules",
        "cleanup-unreachable",
        "resource-residual",
    ],
)
def test_outage_failure_stops_and_records_cleanup_result(
    replay_runner: Any, problem: str
) -> None:
    host = replay_runner
    host.problem = problem
    assert host.runner.run() == 1
    report = result(host)
    assert report["verdict"] == "FAIL"
    assert report["error"], "failed outage or cleanup needs a recorded cause"
    assert host.resources == {} or problem == "resource-residual"


@pytest.mark.parametrize(
    "field,value",
    [
        ("kmsg_exists", False),
        ("kmsg_writable", False),
        ("existing_tagged_rules", ["stale"]),
        ("connectivity", {"192.0.2.1": False}),
    ],
)
def test_preflight_refuses_unproven_host_facts(
    replay_runner: Any, field: str, value: Any
) -> None:
    host = replay_runner
    host.baseline_overrides[field] = value
    assert host.runner.run() == 1
    assert not any(call[0] == "host" and call[1][0] == "arm" for call in host.calls), (
        "invalid baseline cannot arm the outage"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("line_count", 1),
        ("parent_exists", False),
        ("parent_writable", False),
        ("malformed_count", 1),
    ],
)
def test_preflight_refuses_nonempty_or_unwritable_outbox(
    replay_runner: Any, field: str, value: Any
) -> None:
    replay_runner.kernel_overrides[field] = value
    assert replay_runner.runner.run() == 1
    assert result(replay_runner)["error"], "invalid outbox must be rejected explicitly"


@pytest.mark.parametrize(
    "failure", ["release", "worker", "store", "inactive", "restart"]
)
def test_baseline_identity_and_service_failure_are_not_accepted(
    replay_runner: Any, failure: str
) -> None:
    host = replay_runner
    if failure == "release":
        host.release_id = ""
    elif failure == "worker":
        host.worker_names = []
    elif failure == "store":
        host.store_override = {
            "evidence": [{"record_id": "existing"}],
            "incidents": [],
            "notifications": [],
        }
    elif failure == "inactive":
        host.service_overrides["ActiveState"] = "inactive"
    else:

        def drift(snapshot: dict[str, Any]) -> None:
            if host.snapshot_reads > 1:
                snapshot["services"][module.SERVICES[0]]["NRestarts"] = "1"

        host.snapshot_override = drift
    assert host.runner.run() == 1
    assert result(host)["verdict"] == "FAIL"


@pytest.mark.parametrize("output", ["", '{"error":"probe refused"}'])
def test_host_error_protocol_is_decoded_before_proceeding(
    replay_runner: Any, output: str
) -> None:
    host = replay_runner
    host.host_output = output
    if output:
        with pytest.raises(module.CaseError, match="probe refused"):
            host.runner.host_tool("snapshot")
    else:
        assert host.runner.host_tool("snapshot") == {}


def test_duplicate_tag_cleanup_and_missing_baseline_guards(replay_runner: Any) -> None:
    host = replay_runner
    runner = host.runner
    with pytest.raises(module.CaseError, match="baseline"):
        runner.validate_services(host.snapshot())
    with pytest.raises(module.CaseError, match="baseline"):
        runner.write_event("test")
    assert runner.cleanup_tag("absent") == {}
    runner.ips = ["192.0.2.1"]
    runner.arm_and_block(runner.tag, 60)
    runner.arm_and_block(runner.tag, 60)
    assert runner.active_tags == [runner.tag]
    runner.cleanup_tag(runner.tag)
    runner.cleanup_tag(runner.tag)
    assert runner.active_tags == []


def test_final_rejects_retained_outbox_and_duplicate_evidence(
    replay_runner: Any,
) -> None:
    runner = replay_runner.runner
    runner.baseline = replay_runner.snapshot()
    for change in ("matching", "duplicate", "dcgm"):
        snapshot = replay_runner.snapshot()
        store = {
            "evidence": [{"record_id": str(index)} for index in range(3)],
            "incidents": [],
            "notifications": [],
        }
        if change == "matching":
            snapshot["outboxes"]["kernel"]["matching"] = [{"test_ids": ["test"]}]
        elif change == "duplicate":
            store["evidence"][1]["record_id"] = "0"
        else:
            snapshot["outboxes"]["dcgm"]["replayable_count"] = 1
        with pytest.raises(module.CaseError):
            runner.validate_final(snapshot, store)


@pytest.mark.parametrize(
    "field,value",
    [("cluster_id", ""), ("host_probe_image", "mutable"), ("cpu_kubeconfig", None)],
)
def test_settings_require_complete_immutable_local_scope(
    replay_runner: Any, field: str, value: Any
) -> None:
    if field == "cpu_kubeconfig":
        value = replay_runner.runner.settings.cpu_kubeconfig.parent / "missing"
    with pytest.raises(ValueError):
        replace(replay_runner.runner.settings, **{field: value})


def test_predecessor_and_deadline_fail_before_resource_creation(
    replay_runner: Any,
) -> None:
    host = replay_runner
    host.runner.settings = replace(host.runner.settings, predecessor={"valid": False})
    assert host.runner.run() == 1
    assert host.manifests == []
    host.runner.settings = replace(host.runner.settings, predecessor={"valid": True})
    host.runner.maintenance_window_end = datetime.now(timezone.utc) + timedelta(
        minutes=20
    )
    assert host.runner.run() == 1
    assert host.manifests == []
