from collections.abc import Sequence

from gpu_fault.notifications.common import (
    HARDWARE_ESCALATION_ALL_HELD_LINE,
    HARDWARE_ESCALATION_EMAIL_TEMPLATE,
    HARDWARE_ESCALATION_HELD_HEADLINE,
    HARDWARE_ESCALATION_HELD_NODES_LINE,
    HARDWARE_ESCALATION_NONE_HELD_LINE,
    HARDWARE_ESCALATION_NOT_HELD_HEADLINE,
    HARDWARE_ESCALATION_NOT_HELD_NODES_LINE,
    HARDWARE_ESCALATION_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class HardwareEscalationEmailBuilder:
    """Renders the fixed terminal hardware-escalation template."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        node_ids: list[str],
        workload_ids: list[str],
        reasons: list[str],
        failed_operations: list[str],
        policy_source: str,
        official_action: str | None,
        ticket_id: str,
        suspended_workloads: Sequence[str] = (),
        held_node_ids: Sequence[str] | None = None,
    ) -> AdvisoryNotification:
        """``suspended_workloads``: one line per job the remediation left
        suspended (``suspended_workloads.suspended_workload_line``).
        ``held_node_ids``: the nodes the hand-off's records leave cordoned and
        quarantined (``handoff_hold.held_nodes``); the headline, the
        node-state lines, the subject and the case draft's disposition follow
        it. ``None`` reads every named node as held."""

        held = (
            list(node_ids)
            if held_node_ids is None
            else [node for node in node_ids if node in set(held_node_ids)]
        )
        free = [node for node in node_ids if node not in held]
        isolated = bool(held) or not node_ids
        label = "hardware out of service" if isolated else "support escalation"
        body = HARDWARE_ESCALATION_EMAIL_TEMPLATE.format(
            headline=(
                HARDWARE_ESCALATION_HELD_HEADLINE
                if isolated
                else HARDWARE_ESCALATION_NOT_HELD_HEADLINE
            ),
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            node_ids=", ".join(node_ids) or "UNKNOWN",
            workloads=", ".join(workload_ids) or "NONE",
            event_type=event_type,
            event_id=event_id,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            failed_operations=(", ".join(failed_operations) or "NONE_RECORDED"),
            node_state=_node_state(held, free),
            suspended_workloads=(
                "\n".join(f"  - {item}" for item in suspended_workloads) or "  - NONE"
            ),
            ticket_id=ticket_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            template_version=HARDWARE_ESCALATION_TEMPLATE_VERSION,
        )
        case_draft = "\n".join(
            [
                f"Ticket ID: {ticket_id}",
                f"Cluster: {cluster_id}",
                f"Incident: {incident_id}",
                f"Nodes: {', '.join(node_ids) or 'UNKNOWN'}",
                f"Disposition: {_disposition(held, free)}",
                (
                    "Failed operations: "
                    + (", ".join(failed_operations) or "NONE_RECORDED")
                ),
                "Reasons:",
                *[f"- {item}" for item in reasons],
                *(
                    [
                        "Suspended workloads:",
                        *[f"- {item}" for item in suspended_workloads],
                    ]
                    if suspended_workloads
                    else []
                ),
            ]
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/hardware-escalation/"
                f"{HARDWARE_ESCALATION_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU FAULT][{label}] {cluster_id} {', '.join(node_ids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft=case_draft,
        )


def _node_state(held: Sequence[str], free: Sequence[str]) -> str:
    if not free:
        return HARDWARE_ESCALATION_ALL_HELD_LINE
    if not held:
        return HARDWARE_ESCALATION_NONE_HELD_LINE
    return "\n".join(
        [
            HARDWARE_ESCALATION_HELD_NODES_LINE.format(nodes=", ".join(held)),
            HARDWARE_ESCALATION_NOT_HELD_NODES_LINE.format(nodes=", ".join(free)),
        ]
    )


def _disposition(held: Sequence[str], free: Sequence[str]) -> str:
    if not free:
        return "HARDWARE_OFFLINE"
    if not held:
        return "NOT_ISOLATED"
    return f"HARDWARE_OFFLINE ({', '.join(held)}); NOT_ISOLATED ({', '.join(free)})"
