"""NOTIFY-005 bounds each mail kind by the node count, not all of them at once.

Live 2026-10-28 HyperPod, one-node phase: the low-utilisation mail (HEALTH_TREND,
``host-resource/...``) at 13:55:29Z, then at 13:55:42Z the inconclusive-diagnostic
advisory (ADVISORY, ``.../diagnostic-inconclusive/...``) of the same incident: the
node's diagnostic-only chain (FREEZE_EVIDENCE -> VALIDATE_GPU with the metric's
``host_diagnostic_probe``) failed on the idle GPUs, and the advisory's trigger
reasons list the seven absorbed per-GPU findings, so the query took it for a
second low-utilisation notification and ``len(records) <= len(nodes)`` failed a
correct aggregation. ``LIVE_RECORDS`` are the two records that run kept, with
the site's cluster, node and GPU names replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.notifications import DiagnosticInconclusiveEmailBuilder
from scripts.e2e.regional import run_notification_acceptance as runner
from scripts.e2e.regional.notify005_checks import (
    DIAGNOSTIC_INCONCLUSIVE,
    LOW_UTILIZATION_MAIL,
    chain_finished,
    phase_checks,
    phase_progress,
    phase_records,
    record_kind,
)

CLUSTER = "hp-cluster"
NODE = "hyperpod-i-00000000000000001"
GPUS = [f"GPU-{index:08x}-965e-fc68-40e4-225aaa3315b6" for index in range(8)]
INCIDENT = (
    f"inc-host-{NODE}-1793195727235685-host_gpu_utilization_percent-"
    f"{GPUS[0]}-low_gpu_utilization"
)
LIVE_RECORDS = [
    {
        "category": "HEALTH_TREND",
        "deduplication_key": (
            "host-resource/"
            "7be311c565f8e74d4bdbf40a473800b6255ab10fff631f04eb36cd521bb47c28"
        ),
        "gpu_devices": GPUS,
        "low_cpu_utilization": False,
        "low_gpu_utilization": True,
        "matched_nodes": [NODE],
        "names_workload": True,
        "provider_message_id_present": True,
        "status": "SENT",
    },
    {
        "category": "ADVISORY",
        "deduplication_key": (
            f"{CLUSTER}/{INCIDENT}/diagnostic-inconclusive/"
            "workflow-982358159570fbb1d82dd9ad"
        ),
        "gpu_devices": GPUS,
        "low_cpu_utilization": False,
        "low_gpu_utilization": True,
        "matched_nodes": [NODE],
        "names_workload": False,
        "provider_message_id_present": True,
        "status": "SENT",
    },
]


def live(**fields: Any) -> list[dict[str, Any]]:
    """The live records with what the query now adds (the store's values)."""

    return [
        {
            **item,
            "incident_id": INCIDENT,
            "incident_state": "RECOVERED",
            "workflow_status": "FAILED",
            **fields,
        }
        for item in LIVE_RECORDS
    ]


def mail(node: str, **fields: Any) -> dict[str, Any]:
    return {
        "category": "HEALTH_TREND",
        "deduplication_key": f"host-resource/{node}",
        "incident_id": f"inc-{node}",
        "workflow_status": "FAILED",
        "matched_nodes": [node],
        "gpu_devices": GPUS,
        "low_gpu_utilization": True,
        "status": "SENT",
        "provider_message_id_present": True,
        **fields,
    }


def advisory(node: str, incident: str | None = None, **fields: Any) -> dict[str, Any]:
    incident = incident or f"inc-{node}"
    return mail(
        node,
        category="ADVISORY",
        deduplication_key=f"{CLUSTER}/{incident}/diagnostic-inconclusive/w-{incident}",
        incident_id=incident,
        **fields,
    )


def observation(nodes: list[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "nodes": nodes,
        "pods": [{"node": node} for node in nodes],
        "metadata_errors": [],
        "notifications": phase_records(records),
        "polls": [{"low_cpu_notification_count": 0}],
    }


def test_the_live_one_node_phase_passes_per_kind() -> None:
    records = live()

    checks = phase_checks(observation([NODE], records))

    assert len(records) > 1, "two records for one node: the old bound failed here"
    assert phase_progress(records, [NODE])["kinds"] == {
        DIAGNOSTIC_INCONCLUSIVE: 1,
        LOW_UTILIZATION_MAIL: 1,
    }
    assert all(checks.values()), checks


def test_the_kinds_are_the_products_own_keys() -> None:
    notice = DiagnosticInconclusiveEmailBuilder().build(
        cluster_id=CLUSTER,
        incident_id=INCIDENT,
        workflow_id="workflow-a",
        event_id="event-a",
        node_ids=[NODE],
        operations=["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        failed_operation="VALIDATE_GPU",
        error="validation failed on nodes: n (host_diagnostic:m)",
        policy_source="SITE_HOST_RESOURCE_HEALTH",
        official_action=None,
        reasons=["absorbed LOW_GPU_UTILIZATION finding f on device GPU-x"],
    )
    record = {
        "category": notice.category,
        "deduplication_key": notice.deduplication_key,
    }

    assert record_kind(record) == DIAGNOSTIC_INCONCLUSIVE, record
    assert "LOW_GPU_UTILIZATION" in notice.body_text, "why the query selects it"
    stray = record_kind({**record, "category": "GENERAL"})
    assert stray == "UNEXPECTED:GENERAL", stray


@pytest.mark.parametrize(
    "records,failed",
    [
        pytest.param(
            [mail(NODE), *(advisory(NODE, f"inc-gpu-{index}") for index in range(8))],
            {"count_at_most_node_count", "diagnostic_notices_follow_mailed_incidents"},
            id="one-incident-per-gpu",
        ),
        pytest.param(
            [mail(NODE), mail(NODE)], {"count_at_most_node_count"}, id="two-mails"
        ),
        pytest.param(
            [mail(NODE), advisory(NODE), advisory(NODE, "inc-earlier-phase")],
            {"count_at_most_node_count", "diagnostic_notices_follow_mailed_incidents"},
            id="late-advisory-of-an-earlier-phase",
        ),
        pytest.param(
            [mail(NODE), mail(NODE, category="GENERAL")],
            {"every_notification_kind_expected"},
            id="unexpected-kind",
        ),
        pytest.param(
            [mail(NODE, workflow_status="RUNNING")],
            {"every_incident_chain_finished"},
            id="chain-running",
        ),
        pytest.param(
            [advisory(NODE)],
            {
                "every_selected_node_reported_in_its_phase",
                "notifications_aggregate_multiple_gpu_devices",
                "diagnostic_notices_follow_mailed_incidents",
            },
            id="advisory-without-mail",
        ),
        pytest.param(
            [mail(NODE, gpu_devices=GPUS[:1]), advisory(NODE)],
            {"notifications_aggregate_multiple_gpu_devices"},
            id="mail-for-one-gpu",
        ),
    ],
)
def test_a_fan_out_or_a_stray_fails_its_own_check(
    records: list[dict[str, Any]], failed: set[str]
) -> None:
    checks = phase_checks(observation([NODE], records))

    assert {name for name, passed in checks.items() if not passed} == failed, checks


@pytest.mark.parametrize(
    "status,finished",
    [
        (None, True),
        ("MISSING", True),
        ("FAILED", True),
        ("SUCCEEDED", True),
        ("BLOCKED", True),
        ("SUPERSEDED", True),
        ("PENDING", False),
        ("SAFETY_PENDING", False),
        ("RUNNING", False),
    ],
)
def test_a_chain_is_finished_once_its_workflow_no_longer_runs(
    status: str | None, finished: bool
) -> None:
    assert chain_finished({"workflow_status": status}) is finished
    assert chain_finished({}) is False, "a record read without the field"


def poll(
    monkeypatch: pytest.MonkeyPatch,
    nodes: tuple[str, ...],
    rows: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clock = {"now": 0.0}
    queries = iter(rows)
    last: dict[str, Any] = {"rows": []}

    def cpu_python(*_: Any) -> dict[str, Any]:
        last["rows"] = next(queries, last["rows"])
        return {"records": last["rows"]}

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    return runner.wait_low_utilization_notifications(
        SimpleNamespace(cpu_python=cpu_python),
        cluster_id=CLUSTER,
        needle="job",
        nodes=nodes,
        observed_after=datetime.now(timezone.utc),
        deadline_seconds=1200,
        poll_seconds=30,
        settle_seconds=60,
    )


def test_the_one_node_wait_takes_the_advisory_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = [{**item, "workflow_status": "RUNNING"} for item in live()[:1]]
    rows = [[], [], pending, live()]

    records, polls = poll(monkeypatch, (NODE,), rows)

    assert [item["kind"] for item in records] == [
        LOW_UTILIZATION_MAIL,
        DIAGNOSTIC_INCONCLUSIVE,
    ], records
    # Not complete while the chain ran, then covered for the 60 s settle.
    assert len(polls) == 6, polls
    assert polls[-1]["notification_kinds"] == {
        DIAGNOSTIC_INCONCLUSIVE: 1,
        LOW_UTILIZATION_MAIL: 1,
    }, polls
    checks = phase_checks(observation([NODE], records))
    assert all(checks.values()), checks


def test_the_three_node_wait_does_not_stop_on_the_first_nodes_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node that reports first sends its advisory before the others mail:
    four records for three nodes is not a fan-out."""

    nodes = ("node-a", "node-b", "node-c")
    rows = [
        [mail("node-a")],
        [mail("node-a"), advisory("node-a")],
        [mail("node-a"), advisory("node-a"), mail("node-b"), advisory("node-b")],
        [*(mail(node) for node in nodes), *(advisory(node) for node in nodes)],
    ]

    records, polls = poll(monkeypatch, nodes, rows)

    assert len(records) == 6, records
    assert polls[2]["reported_nodes"] == ["node-a", "node-b"], polls
    checks = phase_checks(observation(list(nodes), records))
    assert all(checks.values()), checks


def test_a_second_mail_for_a_node_ends_the_wait_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [[mail(NODE), advisory(NODE), mail(NODE)]]

    records, polls = poll(monkeypatch, (NODE,), rows)

    assert len(polls) == 1, polls
    checks = phase_checks(observation([NODE], records))
    assert checks["count_at_most_node_count"] is False, checks
