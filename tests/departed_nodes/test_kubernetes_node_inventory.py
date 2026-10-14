"""The Kubernetes Node reports an executor posts, and the absences they prove.

The control plane has no GPU kubeconfig, so "absent from the cluster's
Kubernetes Nodes" -- the first proof of a departure -- is the executor's
statement, report by report. The record must never claim a longer absence than
the reports show: a chain broken by a gap remembers nothing, a stale record
proves nothing, a name absent for less than the guard is not yet absent.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest
from pydantic import ValidationError

from gpu_fault.app import create_app
from gpu_fault.app.authorization import ExplicitAuthorizationRegistry
from gpu_fault.kubernetes_node_inventory import (
    NO_REPORT_BLOCKER,
    NODE_INVENTORY_ABSENCE_RETENTION,
    NODE_INVENTORY_MAX_NODES,
    NODE_INVENTORY_PATH,
    PRESENT_BLOCKER,
    InventoryReportRefused,
    KubernetesNodeInventoryReport,
    kubernetes_absence,
    merge_node_inventory,
    record_inventory_report,
)
from tests._builders import asgi_client, build_context, build_store
from tests.departed_nodes._support import ago, inventory_report, now
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

GUARD = timedelta(minutes=10)
A, B, C = "node-a", "node-b", "node-c"


def _chain(*reports: tuple[float, list[str]]):
    """Merge ``(minutes ago, names)`` reports in order; returns the record."""

    record = None
    for minutes, names in reports:
        at = ago(minutes=minutes)
        record = merge_node_inventory(
            record, inventory_report("cluster-a", names, observed_at=at), received_at=at
        )
    assert record is not None
    return record


def test_a_name_left_out_since_more_than_the_guard_is_proven_absent() -> None:
    record = _chain(
        *((minute, [A, B]) for minute in range(20, 12, -1)),
        *((minute, [A]) for minute in range(12, -1, -1)),
    )

    absence = kubernetes_absence(record, B, now=now(), guard=GUARD)

    assert absence.proven, absence.blocker
    assert record.absent[B].since == absence.absent_since
    assert absence.absent_since is not None
    assert now() - absence.absent_since >= GUARD
    assert record.chain_started_at < absence.absent_since, "listed, then absent"


def test_a_name_left_out_for_less_than_the_guard_is_not() -> None:
    record = _chain((15, [A, B]), (10, [A, B]), (5, [A]), (0, [A]))

    absence = kubernetes_absence(record, B, now=now(), guard=GUARD)

    assert not absence.proven, absence
    assert "of the 600s guard" in str(absence.blocker)


def test_a_listed_name_and_a_missing_record_prove_nothing() -> None:
    record = _chain((20, [A, B]), (0, [A, B]))

    assert kubernetes_absence(record, B, now=now(), guard=GUARD).blocker == (
        PRESENT_BLOCKER
    )
    assert kubernetes_absence(None, B, now=now(), guard=GUARD).blocker == (
        NO_REPORT_BLOCKER
    )


def test_a_name_the_chain_never_listed_is_absent_since_the_chain_began() -> None:
    record = _chain(*((minute, [A]) for minute in range(15, -1, -1)))

    absence = kubernetes_absence(record, C, now=now(), guard=GUARD)

    assert absence.proven, absence.blocker
    assert absence.absent_since == record.chain_started_at


def test_a_stale_record_proves_nothing_however_long_the_absence() -> None:
    record = _chain(*((minute, [A]) for minute in range(60, 9, -1)))

    absence = kubernetes_absence(record, C, now=now(), guard=GUARD)

    assert not absence.proven, absence
    assert "report is" in str(absence.blocker) and "old" in str(absence.blocker)


def test_a_gap_in_the_reports_starts_a_new_chain_that_remembers_nothing() -> None:
    record = _chain((40, [A, B]), (30, [A]), (5, [A]), (0, [A]))

    assert record.chain_started_at == record.retained_since
    assert B not in record.absent, "an absence is never carried across a gap"
    assert not kubernetes_absence(record, B, now=now(), guard=GUARD).proven, (
        "proven across a gap"
    )


def test_an_older_or_repeated_report_leaves_the_record_as_it_is() -> None:
    record = _chain((5, [A]), (0, [A]))
    older = inventory_report(
        "cluster-a", [A, B], observed_at=ago(minutes=1), executor_id="executor-b"
    )

    assert merge_node_inventory(record, older, received_at=now()) is record


def test_a_returning_name_is_no_longer_absent() -> None:
    record = _chain((20, [A, B]), (15, [A]), (10, [A]), (0, [A, B]))

    assert B not in record.absent
    assert kubernetes_absence(record, B, now=now(), guard=GUARD).blocker == (
        PRESENT_BLOCKER
    )


def test_old_absences_are_pruned_and_counted_from_the_window_start() -> None:
    start = ago(hours=26)
    record = merge_node_inventory(
        None,
        inventory_report("cluster-a", [A, B], observed_at=start),
        received_at=start,
    )
    at = start
    while at < ago(minutes=1):
        at = at + timedelta(minutes=4)
        record = merge_node_inventory(
            record, inventory_report("cluster-a", [A], observed_at=at), received_at=at
        )

    assert B not in record.absent, "an absence older than the window is pruned"
    left_at = start + timedelta(minutes=4)
    assert record.retained_since > left_at, "counted from later, never earlier"
    assert record.retained_since <= left_at + NODE_INVENTORY_ABSENCE_RETENTION
    absence = kubernetes_absence(record, B, now=at, guard=GUARD)
    assert absence.proven, absence.blocker
    assert absence.absent_since == record.retained_since


def test_a_spare_remembers_it_was_one_after_it_left() -> None:
    plain = _chain((15, [A, B]), *((minute, [A]) for minute in range(14, -1, -1)))
    spared = merge_node_inventory(
        None,
        inventory_report("cluster-a", [A, B], observed_at=ago(minutes=15), spares=[B]),
        received_at=ago(minutes=15),
    )
    for minute in range(14, -1, -1):
        spared = merge_node_inventory(
            spared,
            inventory_report("cluster-a", [A], observed_at=ago(minutes=minute)),
            received_at=ago(minutes=minute),
        )

    assert not kubernetes_absence(plain, B, now=now(), guard=GUARD).was_spare, (
        "never a spare"
    )
    assert kubernetes_absence(spared, B, now=now(), guard=GUARD).was_spare, (
        "a spare when last listed"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"node_names": [A, A]},
        {"node_names": ["Not_A_Node"]},
        {"spare_node_names": [B]},
        {
            "node_names": [
                f"node-{index}" for index in range(NODE_INVENTORY_MAX_NODES + 1)
            ]
        },
    ],
)
def test_a_malformed_report_is_refused_by_the_model(changes: dict) -> None:
    values = {
        "cluster_id": "cluster-a",
        "executor_id": "executor-a",
        "observed_at": now(),
        "node_names": [A],
        **changes,
    }
    with pytest.raises(ValidationError):
        KubernetesNodeInventoryReport.model_validate(values)


def test_a_naive_report_time_is_refused() -> None:
    with pytest.raises(ValidationError):
        KubernetesNodeInventoryReport(
            cluster_id="cluster-a",
            executor_id="executor-a",
            observed_at=now().replace(tzinfo=None),
        )


def test_a_report_whose_clock_disagrees_is_refused_and_a_good_one_recorded() -> None:
    store = build_store()
    with pytest.raises(InventoryReportRefused, match="from the control plane's clock"):
        record_inventory_report(
            store, inventory_report("cluster-a", [A], observed_at=ago(minutes=5))
        )

    receipt = record_inventory_report(
        store, inventory_report("cluster-a", [A, B], observed_at=now())
    )

    assert receipt.recorded is True and receipt.node_count == 2
    stored = store.get_kubernetes_node_inventory("cluster-a")
    assert stored is not None and stored.node_names == [A, B]
    again = record_inventory_report(
        store,
        inventory_report(
            "cluster-a", [A], observed_at=ago(seconds=30), executor_id="executor-b"
        ),
    )
    assert again.recorded is False, "the sibling's older LIST does not win"


def _context():
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


def _post(context, token: str, cluster_id: str, payload: dict) -> httpx.Response:
    async def scenario() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(
                NODE_INVENTORY_PATH,
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-GPU-Fault-Cluster-ID": cluster_id,
                },
                json=payload,
            )

    return asyncio.run(scenario())


def test_the_route_is_in_the_cluster_token_bucket() -> None:
    declared = ExplicitAuthorizationRegistry()
    declared.load(create_app(_context()).routes)

    assert declared.declared(NODE_INVENTORY_PATH, "POST") == "cluster-token"


def test_the_route_records_the_authenticated_clusters_report() -> None:
    context = _context()
    report = inventory_report("cluster-a", [A, B], observed_at=now())

    response = _post(context, TOKEN_A, "cluster-a", report.model_dump(mode="json"))

    assert response.status_code == 200, response.text
    assert response.json()["cluster_id"] == "cluster-a"
    assert response.json()["node_count"] == 2
    stored = context.store.get_kubernetes_node_inventory("cluster-a")
    assert stored is not None and stored.node_names == [A, B]


def test_the_route_refuses_another_clusters_report_and_a_wrong_clock() -> None:
    context = _context()
    foreign = inventory_report("cluster-b", [A], observed_at=now())
    skewed = inventory_report("cluster-a", [A], observed_at=ago(minutes=10))

    assert (
        _post(
            context, TOKEN_A, "cluster-a", foreign.model_dump(mode="json")
        ).status_code
        == 403
    )
    response = _post(context, TOKEN_A, "cluster-a", skewed.model_dump(mode="json"))
    assert response.status_code == 422
    assert "control plane's clock" in response.json()["detail"]
    assert _post(
        context, TOKEN_B, "cluster-a", skewed.model_dump(mode="json")
    ).status_code in (401, 403)
    assert context.store.get_kubernetes_node_inventory("cluster-a") is None
