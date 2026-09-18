from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import regional_live_fixture as live
from scripts.e2e.regional.kmsg_clock import KMSG_CLOCK_SKEW_SECONDS
from tests.regional._cov95_cap_cases import Clock
from tests.regional._cov95_common_live import LiveModel


def exec_calls(fixture: LiveModel) -> list[Any]:
    return [(args, options) for args, options in fixture.calls if "exec" in args]


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_pod_probe_retries_malformed_reads_with_a_bounded_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plane: str
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    fixture.executions = ["not-json", 'diagnostic\n{"proof":1}\n']
    assert fixture.pod_python(plane, "app", "unit probe", "argument") == {"proof": 1}
    assert len(exec_calls(fixture)) == 2
    assert clock.elapsed == 1
    assert live.component_python(plane) in exec_calls(fixture)[-1][0]
    assert exec_calls(fixture)[-1][0][-1] == "argument"


@pytest.mark.parametrize("output", ["", "[]"])
def test_failed_probe_reads_never_return_an_empty_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: str
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    fixture.executions = [output]
    with pytest.raises(live.RegionalFixtureError, match="Pod probe failed"):
        fixture.cpu_python("unit probe", attempts=2)
    assert len(exec_calls(fixture)) == 2
    assert clock.elapsed == 1


def test_probe_timeout_and_invalid_budget_do_not_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    with pytest.raises(ValueError, match="attempts"):
        fixture.cpu_python("unit", attempts=0)
    assert fixture.calls == []
    fixture.executions = [live.RegionalCommandTimeout(["unit"], 5)]
    with pytest.raises(live.RegionalFixtureError, match="not retried"):
        fixture.executor_python("unit", attempts=3)
    assert len(exec_calls(fixture)) == 1
    assert clock.elapsed == 0


def test_xid_post_is_cluster_bound_and_never_replayed_after_receipt_loss(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    with pytest.raises(live.RegionalFixtureError, match="cluster ID"):
        fixture.post_xid_event({"cluster_id": "other"})
    assert fixture.calls == []
    fixture.executions = [OSError("synthetic receipt loss")]
    payload = {"cluster_id": fixture.settings.cluster_id, "event_id": "unit-event"}
    with pytest.raises(live.RegionalFixtureError, match="Pod probe failed"):
        fixture.post_xid_event(payload)
    calls = exec_calls(fixture)
    assert len(calls) == 1
    assert json.loads(calls[0][0][-1]) == payload
    assert calls[0][1]["timeout"] == 240


def test_store_snapshot_keeps_query_scope_and_fills_only_the_release_identity(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    fixture.executions = [{"workflow": {"status": "RUNNING"}}]
    observed = datetime(2026, 9, 12, tzinfo=timezone.utc)
    snapshot = fixture.store_snapshot(
        node="node-a",
        marker="unit-marker",
        observed_after=observed,
        job_id="unit-job",
        attempt_id="unit-attempt",
        hyperpod_cluster="unit-hp",
        queue_attempts=1,
        workflow_request_ids=["workflow-a", "workflow-b"],
    )
    assert snapshot == {"workflow": {"status": "RUNNING"}, "release_id": "unit-release"}
    assert exec_calls(fixture)[0][0][-9:] == [
        fixture.settings.cluster_id,
        "node-a",
        "unit-marker",
        (observed - timedelta(seconds=KMSG_CLOCK_SKEW_SECONDS)).isoformat(),
        "unit-job",
        "unit-attempt",
        "unit-hp",
        "1",
        "workflow-a,workflow-b",
    ]


@pytest.mark.parametrize("ids", [[""], ["has,comma"]])
def test_store_scope_rejects_ambiguous_workflow_ids_before_reading(
    tmp_path: Path, ids: list[str]
) -> None:
    fixture = LiveModel(tmp_path)
    with pytest.raises(ValueError, match="workflow request IDs"):
        fixture.store_snapshot(workflow_request_ids=ids)
    assert fixture.calls == []


@pytest.mark.parametrize("terminal", [False, True])
def test_workflow_wait_retains_transient_waiting_evidence_in_its_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal: bool
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    waiting = {
        "step_index": 1,
        "operation": "COLLECT_DIAGNOSTIC_BUNDLE",
        "status": "WAITING",
        "adapter_operation_id": "unit-operation",
        "details": {"nonphysical": True},
        "error": None,
    }
    values = iter(
        [
            {"workflow": {"status": "RUNNING", "step_executions": [waiting]}},
            {"workflow": {"status": "SUCCEEDED", "step_executions": []}},
        ]
    )
    reads = []

    def read(**kwargs: Any) -> dict[str, Any]:
        reads.append(kwargs)
        return next(values)

    monkeypatch.setattr(fixture, "store_snapshot", read)
    result = fixture.wait_for_workflow(
        node="node-a",
        marker="unit",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=10,
        terminal=terminal,
        workflow_request_ids=["workflow-a"],
    )
    assert result["workflow"]["status"] == ("SUCCEEDED" if terminal else "RUNNING")
    assert result["observed_waiting_step_executions"] == [waiting]
    timeline = json.loads((tmp_path / "timeline.json").read_text())
    assert len(timeline["entries"]) == (2 if terminal else 1)
    assert timeline["observed_waiting_step_executions"] == [waiting]
    assert all(item["queue_attempts"] == 1 for item in reads), (
        "polling drained the whole queue"
    )
    assert all(item["workflow_request_ids"] == ["workflow-a"] for item in reads), (
        "a poll widened the workflow scope"
    )


def test_empty_workflow_observation_times_out_with_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    monkeypatch.setattr(fixture, "store_snapshot", lambda **_kwargs: {})
    with pytest.raises(live.RegionalFixtureError, match="requested state"):
        fixture.wait_for_workflow(
            node="node-a",
            marker="unit",
            observed_after=datetime.now(timezone.utc),
            case_dir=tmp_path,
            timeout_seconds=6,
        )
    assert len(json.loads((tmp_path / "timeline.json").read_text())["entries"]) == 2
    assert clock.elapsed == 10


def test_node_wait_requires_a_real_changed_boot_id_after_failed_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    values = iter(
        [
            live.RegionalFixtureError("synthetic read failure"),
            {"ready": "True", "boot_id": None},
            {"ready": "True", "boot_id": "old"},
            {"ready": "True", "boot_id": "new"},
        ]
    )
    reads = []

    def snapshot(name: str) -> dict[str, Any]:
        reads.append(name)
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(fixture, "node_snapshot", snapshot)
    assert fixture.wait_node_ready(
        "node-a", timeout_seconds=40, expected_boot_id="old"
    ) == {"ready": "True", "boot_id": "new"}
    assert reads == ["node-a"] * 4
    assert clock.elapsed == 30


def test_node_readiness_timeout_never_accepts_missing_boot_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    monkeypatch.setattr(
        fixture, "node_snapshot", lambda _node: {"ready": "True", "boot_id": ""}
    )
    with pytest.raises(live.RegionalFixtureError, match="did not return Ready"):
        fixture.wait_node_ready("node-a", timeout_seconds=11, expected_boot_id="old")
    assert clock.elapsed == 20


def test_provider_events_filter_mutations_and_bind_the_session_issuer(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    fixture.events = [
        {
            "Events": [
                {"EventName": "ListClusters"},
                {
                    "EventName": "BatchRebootClusterNodes",
                    "Username": "not-the-issuer",
                    "EventTime": "2026-09-12T00:00:00Z",
                    "CloudTrailEvent": json.dumps(
                        {
                            "userIdentity": {
                                "sessionContext": {
                                    "sessionIssuer": {
                                        "arn": "arn:aws:iam::000000000000:role/unit-role"
                                    }
                                }
                            }
                        }
                    ),
                },
                {
                    "EventName": "BatchDeleteClusterNodes",
                    "CloudTrailEvent": "{",
                    "Username": "legacy",
                },
            ]
        }
    ]
    now = datetime.now(timezone.utc)
    result = fixture.provider_events(now, now)
    assert [item["event_name"] for item in result] == [
        "BatchRebootClusterNodes",
        "BatchDeleteClusterNodes",
    ]
    assert result[0]["session_issuer_role_name"] == "unit-role"
    assert result[1]["session_issuer_role_name"] == ""
    assert live.provider_event_actor_matches_role(
        result[0], "arn:aws:iam::000000000000:role/unit-role"
    ), "the session issuer was not used for attribution"
    assert fixture.calls[0][1]["timeout"] == 180


def test_provider_wait_does_not_confuse_visibility_delay_with_a_proved_negative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = LiveModel(tmp_path)
    clock = Clock()
    monkeypatch.setattr(live, "time", clock)
    fixture.events = [
        {"Events": []},
        {"Events": [{"EventName": "BatchRebootClusterNodes"}]},
    ]
    now = datetime.now(timezone.utc)
    assert (
        len(
            fixture.wait_provider_events(
                now,
                event_names={"BatchRebootClusterNodes"},
                expected_count=1,
                timeout_seconds=10,
                poll_seconds=2,
            )
        )
        == 1
    )
    assert clock.elapsed == 2
    assert (
        fixture.provider_events_provisional(now, now=now + timedelta(seconds=899))
        is True
    )
    assert (
        fixture.provider_events_provisional(now, now=now + timedelta(seconds=900))
        is False
    )
    with pytest.raises(ValueError, match="expected_count"):
        fixture.wait_provider_events(now, event_names=set(), expected_count=-1)
