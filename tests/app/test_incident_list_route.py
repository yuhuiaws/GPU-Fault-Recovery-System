"""``GET /v1/incidents``: the operator's list of incidents by state.

``GpuFaultIncidentsAwaitingOperator`` counts ESCALATED incidents but nothing
listed their ids: the API only served one incident by id, ``status`` does not
list incidents, and the runbook sent the operator to escalation e-mails (and,
in practice, to raw SQL inside the API Pod). The route answers that: state is
required (never an unbounded dump), the search spans every registered cluster
unless one is named, and ``truncated`` says when ``limit`` cut the list.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.app.routes.incidents import (
    INCIDENT_LIST_DEFAULT_LIMIT,
    INCIDENT_LIST_FIRST_REASON_CHARS,
    INCIDENT_LIST_MAX_LIMIT,
)
from gpu_fault.models import IncidentState
from tests._builders import asgi_client, build_store, copy_model, fault_incident
from tests.orchestration._incident_closure_support import NOW, _escalated_reset
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

TOKEN = "x" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": TOKEN}


def _context() -> ApplicationContext:
    context = ApplicationContext(store=build_store(), execution_token=TOKEN)
    # The bucket is enforced by the regional default-deny middleware, the same
    # way the sibling GET /v1/incidents/{id} is guarded.
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


def _get(context: ApplicationContext, query: str, *, headers=HEADERS):
    async def scenario():
        async with asgi_client(context) as client:
            return await client.get(f"/v1/incidents{query}", headers=headers)

    return asyncio.run(scenario())


def _ids(response) -> list[str]:
    return [item["incident_id"] for item in response.json()["incidents"]]


def _queue(context: ApplicationContext) -> None:
    """Two ESCALATED incidents on cluster-a, one on cluster-b, one QUARANTINED
    and one RECOVERED that no ESCALATED listing may show."""

    _escalated_reset(context.store, incident_id="inc-a1", node_ids=("node-a1",))
    _escalated_reset(context.store, incident_id="inc-a2", node_ids=("node-a2",))
    _escalated_reset(
        context.store,
        incident_id="inc-b1",
        node_ids=("node-b1",),
        cluster_id="cluster-b",
    )
    quarantined, _ = _escalated_reset(
        context.store, incident_id="inc-q", node_ids=("node-q",)
    )
    context.store.save_incident(
        copy_model(quarantined, state=IncidentState.QUARANTINED), expected=quarantined
    )
    recovered, _ = _escalated_reset(
        context.store, incident_id="inc-r", node_ids=("node-r",)
    )
    context.store.save_incident(
        copy_model(recovered, state=IncidentState.RECOVERED), expected=recovered
    )


def test_listing_needs_the_execution_token() -> None:
    context = _context()
    _queue(context)

    denied = _get(context, "?state=ESCALATED", headers={})

    assert denied.status_code in {401, 403}, denied.text


def test_state_is_required_so_the_route_never_dumps_every_incident() -> None:
    context = _context()
    _queue(context)

    missing = _get(context, "")
    blank = _get(context, "?state=")
    unknown = _get(context, "?state=OPEN")

    assert missing.status_code == 422, missing.text
    assert "state" in missing.json()["detail"]
    assert blank.status_code == 422, blank.text
    assert unknown.status_code == 422, unknown.text
    assert (
        "OPEN" in unknown.json()["detail"] and "ESCALATED" in unknown.json()["detail"]
    )


def test_escalated_across_every_registered_cluster_excluding_quarantined() -> None:
    context = _context()
    _queue(context)

    response = _get(context, "?state=ESCALATED")

    assert response.status_code == 200, response.text
    body = response.json()
    assert sorted(_ids(response)) == ["inc-a1", "inc-a2", "inc-b1"]
    assert body["truncated"] is False
    assert body["states"] == ["ESCALATED"]
    assert body["cluster_ids"] == ["cluster-a", "cluster-b"]
    assert body["limit"] == INCIDENT_LIST_DEFAULT_LIMIT
    assert {item["cluster_id"] for item in body["incidents"]} == {
        "cluster-a",
        "cluster-b",
    }


def test_states_are_repeatable_and_comma_separated() -> None:
    context = _context()
    _queue(context)

    repeated = _get(context, "?state=ESCALATED&state=QUARANTINED")
    comma = _get(context, "?state=QUARANTINED,RECOVERED")

    assert sorted(_ids(repeated)) == ["inc-a1", "inc-a2", "inc-b1", "inc-q"]
    assert repeated.json()["states"] == ["ESCALATED", "QUARANTINED"]
    assert sorted(_ids(comma)) == ["inc-q", "inc-r"]


def test_cluster_and_node_filters_narrow_the_list() -> None:
    context = _context()
    _queue(context)

    by_cluster = _get(context, "?state=ESCALATED&cluster_id=cluster-b")
    by_node = _get(context, "?state=ESCALATED&node_id=node-a2")
    unknown_cluster = _get(context, "?state=ESCALATED&cluster_id=cluster-z")

    assert _ids(by_cluster) == ["inc-b1"]
    assert by_cluster.json()["cluster_ids"] == ["cluster-b"]
    assert _ids(by_node) == ["inc-a2"]
    assert unknown_cluster.status_code == 200
    assert _ids(unknown_cluster) == [] and unknown_cluster.json()["truncated"] is False


def test_limit_cuts_the_merged_list_newest_first_and_flags_truncation() -> None:
    context = _context()
    _queue(context)
    newest = context.store.get_incident("inc-b1")
    context.store.save_incident(
        copy_model(newest, updated_at=NOW + timedelta(minutes=1)), expected=newest
    )

    one = _get(context, "?state=ESCALATED&limit=1")
    all_three = _get(context, "?state=ESCALATED&limit=3")
    too_many = _get(context, f"?state=ESCALATED&limit={INCIDENT_LIST_MAX_LIMIT + 1}")
    zero = _get(context, "?state=ESCALATED&limit=0")

    assert _ids(one) == ["inc-b1"], "newest updated_at first, across clusters"
    assert one.json()["truncated"] is True
    assert len(_ids(all_three)) == 3 and all_three.json()["truncated"] is False
    assert too_many.status_code == 422, too_many.text
    assert zero.status_code == 422, zero.text


def test_the_row_is_a_compact_summary_with_the_first_reason_cut() -> None:
    context = _context()
    long_reason = "r" * (INCIDENT_LIST_FIRST_REASON_CHARS + 50)
    context.store.save_incident(
        fault_incident(
            "inc-long",
            "event-long",
            node_ids=["node-l"],
            state=IncidentState.ESCALATED,
            official_action="RESET_GPU",
            workflow_request_id="wf-long",
            reasons=[long_reason, "second", "third"],
        )
    )

    response = _get(context, "?state=ESCALATED")

    [row] = response.json()["incidents"]
    assert set(row) == {
        "incident_id",
        "state",
        "cluster_id",
        "node_ids",
        "created_at",
        "updated_at",
        "event_type",
        "official_action",
        "effective_action",
        "workflow_request_id",
        "first_reason",
        "reasons_count",
    }, "a summary row, not the full FaultIncident"
    assert row["incident_id"] == "inc-long"
    assert row["state"] == "ESCALATED"
    assert row["node_ids"] == ["node-l"]
    assert row["official_action"] == "RESET_GPU"
    assert row["workflow_request_id"] == "wf-long"
    assert row["reasons_count"] == 3
    assert len(row["first_reason"]) == INCIDENT_LIST_FIRST_REASON_CHARS
    assert row["first_reason"].startswith("r" * 20) and row["first_reason"].endswith(
        "…"
    )


def test_no_registered_cluster_and_no_cluster_id_is_an_empty_list() -> None:
    context = ApplicationContext(store=build_store(), execution_token=TOKEN)
    _escalated_reset(context.store)

    response = _get(context, "?state=ESCALATED")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "incidents": [],
        "truncated": False,
        "states": ["ESCALATED"],
        "cluster_ids": [],
        "limit": INCIDENT_LIST_DEFAULT_LIMIT,
    }
