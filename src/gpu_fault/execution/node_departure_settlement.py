"""Settle the node actions of a record whose nodes the platform took away.

A managed node group roll, a Spot reclaim or a HyperPod replacement removes a
GPU node -- and its instance -- while a remediation workflow waits on a node
action there: a RESTART_NODE submitted and never confirmed, a RESET the Agent
accepted and never answered. The record parks BLOCKED / NEEDS_OPERATOR on that
unresolved action (``node_action_uncertainty``), and until batch 139 nothing
could answer it: ``confirm-node-action`` needed a Ready node, the departed-node
cleanup ended only records that never changed a node, and the incident close
and the restore refused the open record -- a dead end the release and
uninstall gates kept counting.

The departure proof answers it (``departed_nodes``): the node is gone from
Kubernetes and from its provider (EC2 ``terminated`` or unknown, HyperPod no
longer lists the instance) and its Agent is not alive. Whatever the action did
went with the instance and nothing can still run on it; a name that comes back
is a new Node object (the cleanup drops its ledger line when it does).

``departure_settlement`` judges a whole record against the departures of the
nodes a caller proved -- the incident close on the operator's node evidence
(``orchestration/incident_closure.py``), and through it the departed-node
cleanup -- and ``apply_departure_settlement`` writes the answers the way
``confirm-node-action`` writes an operator confirmation (``operator_confirmed``
on the step, ``settled_by: node_departed``, one ``NODE_ACTION_CONFIRMED`` event
each), so every reader of the uncertainty resolves it as it resolves a
confirmed one. An action whose step also names a node not proven departed is
``unanswered``: that node's share of the outcome is still unknown, and the
record is not settled. Pure over the records; the callers read the sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpu_fault.departed_nodes import evidence_proves_departure
from gpu_fault.execution.node_action_confirmation import (
    NodeActionConfirmationVerdict,
    apply_node_action_confirmation,
    departure_confirmation,
    describe_node_action,
    node_action_executions,
)
from gpu_fault.execution.node_action_uncertainty import unresolved_node_action
from gpu_fault.models import WorkflowRequest, WorkflowStepExecution


@dataclass(frozen=True)
class DepartureSettlement:
    """What the departures answer on one record, and what they do not."""

    #: ``(execution, confirmation)`` per unresolved node action they answer.
    settled: tuple[tuple[WorkflowStepExecution, dict[str, Any]], ...] = ()
    #: The unresolved node actions they do not answer, described.
    unanswered: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Every unresolved node action of the record is answered."""

        return bool(self.settled) and not self.unanswered


def proven_departures(
    departures: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """The entries of ``departures`` whose evidence proves the node gone."""

    return {
        node: evidence
        for node, evidence in departures.items()
        if evidence_proves_departure(evidence, node)
    }


def departure_settlement(
    workflow: WorkflowRequest,
    departures: Mapping[str, Mapping[str, Any]],
    *,
    actor: str,
    reference: str,
    now: datetime,
) -> DepartureSettlement:
    """Judge every unresolved node action of ``workflow`` against ``departures``.

    ``departures`` maps a node to its departure evidence
    (``departed_nodes.node_evidence``); only entries that prove the departure
    count, and the caller has judged each node's Agent already. A step settles
    when every node it names departed; the confirmation names the first of them.
    """

    proven = proven_departures(departures)
    settled: list[tuple[WorkflowStepExecution, dict[str, Any]]] = []
    unanswered: list[str] = []
    for execution, step in node_action_executions(workflow):
        if not unresolved_node_action(execution):
            continue
        nodes = sorted(set(step.node_ids)) if step is not None else []
        missing = sorted(set(nodes) - set(proven))
        if not nodes or missing:
            unanswered.append(
                describe_node_action(execution, step)
                + (
                    f" ({', '.join(missing)} not proven departed)"
                    if missing
                    else " (its nodes are unknown)"
                )
            )
            continue
        settled.append(
            (
                execution,
                departure_confirmation(
                    execution,
                    step,
                    nodes[0],
                    proven,
                    actor=actor,
                    reference=reference,
                    now=now,
                ),
            )
        )
    return DepartureSettlement(settled=tuple(settled), unanswered=tuple(unanswered))


def apply_departure_settlement(
    workflow: WorkflowRequest, settlement: DepartureSettlement, *, now: datetime
) -> WorkflowRequest:
    """``workflow`` with every answer of ``settlement`` recorded and audited.

    Through ``apply_node_action_confirmation``, one verdict per confirming
    node, so the step records and the events read exactly as an operator's
    confirmation does. The status is untouched: the incident close (or the
    dispatcher's settled sweep) ends the record.
    """

    by_node: dict[str, list[tuple[WorkflowStepExecution, dict[str, Any]]]] = {}
    for execution, confirmation in settlement.settled:
        by_node.setdefault(str(confirmation["node_id"]), []).append(
            (execution, confirmation)
        )
    updated = workflow
    for node_id in sorted(by_node):
        pairs = by_node[node_id]
        updated = apply_node_action_confirmation(
            updated,
            NodeActionConfirmationVerdict(
                node_id=node_id,
                confirmations=tuple(confirmation for _, confirmation in pairs),
                executions=tuple(execution for execution, _ in pairs),
            ),
            now=now,
        )
    return updated
