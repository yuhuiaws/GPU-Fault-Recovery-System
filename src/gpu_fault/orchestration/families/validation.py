from __future__ import annotations

from gpu_fault.host_health import NodeHealthFinding


class ValidationOperationService:
    @staticmethod
    def inventory_parameters(
        finding: NodeHealthFinding,
    ) -> dict:
        gpu_metrics = {
            "gpu_inventory_mismatch",
            "gpu_kubernetes_allocatable_mismatch",
        }
        if finding.metric_name not in {
            *gpu_metrics,
            "efa_inventory_mismatch",
            "efa_kubernetes_allocatable_mismatch",
        }:
            return {}
        configured = {
            "gpu_inventory_active_count": (
                finding.diagnostic_parameters.get("expected_gpu_count")
            ),
            "efa_inventory_active_count": (
                finding.diagnostic_parameters.get("expected_efa_device_count")
            ),
        }
        fallback = (
            "gpu_inventory_active_count"
            if finding.metric_name in gpu_metrics
            else "efa_inventory_active_count"
        )
        configured[fallback] = configured[fallback] or (
            finding.diagnostic_parameters.get("expected_count")
        )
        metrics = {}
        for metric_name, expected in configured.items():
            if expected is None:
                continue
            try:
                metrics[metric_name] = int(expected)
            except (TypeError, ValueError):
                continue
        if not metrics:
            return {}
        return {
            "inventory_requirements_by_node": {
                finding.node_id: {
                    "metrics": metrics,
                    "node_instance_type": (
                        finding.diagnostic_parameters.get("node_instance_type")
                    ),
                }
            }
        }
