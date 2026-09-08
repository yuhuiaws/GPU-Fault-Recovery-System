from gpu_fault.notifications.common import (
    GPU_MECHANICAL_EMAIL_TEMPLATE,
    GPU_MECHANICAL_TEMPLATE_VERSION,
    NVLINK74_MECHANICAL_EMAIL_TEMPLATE,
    NVLINK74_MECHANICAL_TEMPLATE_VERSION,
    AdvisoryNotification,
)


class Nvlink74MechanicalEmailBuilder:
    """Renders the fixed operator task for physical inspection."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        node_ids: list[str],
        link_id: int | None,
        pci_bdf: str | None,
        occurrence_counts: dict[str, int],
        annotation: str,
        annotation_value: str,
        xid: int = 74,
    ) -> AdvisoryNotification:
        if xid != 74:
            body = GPU_MECHANICAL_EMAIL_TEMPLATE.format(
                cluster_id=cluster_id,
                incident_id=incident_id,
                workflow_id=workflow_id,
                node_ids=", ".join(node_ids) or "UNKNOWN",
                xid=xid,
                pci_bdf=pci_bdf or "UNKNOWN",
                annotation=annotation,
                annotation_value=annotation_value,
                template_version=GPU_MECHANICAL_TEMPLATE_VERSION,
            )
            return AdvisoryNotification(
                deduplication_key=(
                    f"{incident_id}/gpu-mechanical/{GPU_MECHANICAL_TEMPLATE_VERSION}"
                ),
                cluster_name=cluster_id,
                incident_id=incident_id,
                subject=(
                    f"[GPU故障][XID {xid}机械检查] {cluster_id} "
                    f"{', '.join(node_ids) or 'UNKNOWN'}"
                ),
                body_text=body,
                support_case_draft=(
                    "Operator task only; vendor support case is not "
                    "created by CHECK_MECHANICALS."
                ),
            )
        body = NVLINK74_MECHANICAL_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            node_ids=", ".join(node_ids) or "UNKNOWN",
            link_id=link_id if link_id is not None else "UNKNOWN",
            pci_bdf=pci_bdf or "UNKNOWN",
            occurrence_counts=(
                ", ".join(
                    f"{key}={value}" for key, value in sorted(occurrence_counts.items())
                )
                or "NONE"
            ),
            annotation=annotation,
            annotation_value=annotation_value,
            template_version=(NVLINK74_MECHANICAL_TEMPLATE_VERSION),
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{incident_id}/xid74-mechanical/{NVLINK74_MECHANICAL_TEMPLATE_VERSION}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU故障][XID 74机械检查] {cluster_id} "
                f"{', '.join(node_ids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft=(
                "Operator task only; vendor support case is not "
                "created by CHECK_MECHANICALS."
            ),
        )
