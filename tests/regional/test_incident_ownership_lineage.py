"""The ownership route vouches for a successor's whole escalation lineage.

The storeless regional executor cannot walk the chain itself; it names the
workflow asking (``successor_workflow_id``) and the control plane answers
``escalation_lineage`` -- whether that workflow descends, by ``-after-`` event
ids or ``predecessor_workflow_id`` pointers, from a workflow of the owner
incident. Live the support successor of a refused reboot successor was two
rungs from the owner and the one-hop check refused it.
"""

from __future__ import annotations

import asyncio

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from tests._builders import (
    asgi_client,
    build_context,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    ownership_fixture,
    registration,
)

PATH = "/v1/regional/executors/incident-ownership"


def _chain(store) -> None:
    """``inc-dead``/``workflow-dead`` (from the fixture) -> reboot-after ->
    support-after, all on cluster-a; plus a stranger on cluster-b."""

    for incident_id, event_id, request_id, cluster_id in (
        ("inc-reboot", "reboot-after-workflow-dead", "workflow-reboot", "cluster-a"),
        (
            "inc-support",
            "support-after-workflow-reboot",
            "workflow-support",
            "cluster-a",
        ),
        ("inc-foreign", "reboot-after-workflow-dead", "workflow-foreign", "cluster-b"),
    ):
        store.save_incident(
            fault_incident(
                incident_id,
                event_id,
                cluster_id=cluster_id,
                official_action="RESTART_BM",
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=request_id,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        store.save_workflow(
            workflow_request(
                request_id,
                incident_id,
                WorkflowStatus.RUNNING,
                1,
                runtime_profile_version="profile-v1",
                official_action="RESTART_BM",
                official_steps=[workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE)],
                created_at=NOW,
                updated_at=NOW,
            )
        )


def test_the_route_walks_the_successors_lineage_back_to_the_owner() -> None:
    context = build_context()
    context.regional_mode = True
    context.execution_token = "operator-token"
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    ownership_fixture(context.store)
    _chain(context.store)
    headers = {
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }

    async def scenario() -> None:
        async with asgi_client(context) as client:
            two_hops = await client.get(
                PATH,
                headers=headers,
                params={
                    "incident_id": "inc-dead",
                    "successor_workflow_id": "workflow-support",
                },
            )
            assert two_hops.status_code == 200
            assert two_hops.json()["escalation_lineage"] is True
            assert two_hops.json()["terminal"] is True

            one_hop = await client.get(
                PATH,
                headers=headers,
                params={
                    "incident_id": "inc-dead",
                    "successor_workflow_id": "workflow-reboot",
                },
            )
            assert one_hop.json()["escalation_lineage"] is True

            stranger = await client.get(
                PATH,
                headers=headers,
                params={
                    "incident_id": "inc-dead",
                    "successor_workflow_id": "workflow-live",
                },
            )
            assert stranger.json()["escalation_lineage"] is False

            not_asked = await client.get(
                PATH, headers=headers, params={"incident_id": "inc-dead"}
            )
            assert not_asked.json()["escalation_lineage"] is None

            unknown_successor = await client.get(
                PATH,
                headers=headers,
                params={"incident_id": "inc-dead", "successor_workflow_id": "wf-none"},
            )
            assert unknown_successor.json()["escalation_lineage"] is False

            # A workflow of another cluster may not be named as the asking
            # successor, whatever its event id claims.
            cross = await client.get(
                PATH,
                headers=headers,
                params={
                    "incident_id": "inc-dead",
                    "successor_workflow_id": "workflow-foreign",
                },
            )
            assert cross.status_code == 200
            assert cross.json()["escalation_lineage"] is False

    asyncio.run(scenario())
