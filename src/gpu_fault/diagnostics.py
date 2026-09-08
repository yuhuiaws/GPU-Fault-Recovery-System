from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families

from gpu_fault.models import (
    DiagnosticRequest,
    RecoveryAction,
    TriageFinding,
    TriageOutcome,
    TriageReport,
)

RESET_REQUIRED_COUNTERS = {
    "DCGM_FI_DEV_RETIRED_PENDING",
    "DCGM_FI_DEV_ROW_REMAP_PENDING",
}
DRAIN_REQUIRED_COUNTERS = {
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
    "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    "DCGM_FI_DEV_ROW_REMAP_FAILURE",
}
GPU_IDENTITY_METRICS = {
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_POWER_USAGE",
}


class KubernetesDcgmDiagnosticAdapter:
    """Runs conservative quick triage from Node state and DCGM metrics."""

    def __init__(
        self,
        store,
        core_api,
        *,
        dcgm_port: int = 9400,
        timeout_seconds: int = 10,
        fetcher: Callable[[str, int], str] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.core = core_api
        self.dcgm_port = dcgm_port
        self.timeout_seconds = timeout_seconds
        self.fetcher = fetcher or self._fetch
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._reports: dict[str, TriageReport] = {}

    def submit(self, request: DiagnosticRequest) -> str:
        self.store.save_diagnostic(request)
        findings = [self._diagnose_node(node_id) for node_id in request.node_ids]
        self._reports[request.request_id] = TriageReport(
            request_id=request.request_id,
            cluster_id=request.cluster_id,
            attempt_id=request.attempt_id,
            findings=findings,
            completed_at=self.now(),
        )
        return request.request_id

    def result(self, request_id: str) -> TriageReport | None:
        return self._reports.get(request_id)

    def _diagnose_node(self, node_id: str) -> TriageFinding:
        try:
            node = self.core.read_node(node_id)
            failed_conditions = self._failed_conditions(node)
            address = self._internal_ip(node)
            metrics_url = f"http://{address}:{self.dcgm_port}/metrics"
            metrics = self.fetcher(metrics_url, self.timeout_seconds)
            samples = [
                sample
                for family in text_string_to_metric_families(metrics)
                for sample in family.samples
            ]
            names = {sample.name for sample in samples}
            reset_required = sorted(
                {
                    sample.name
                    for sample in samples
                    if sample.name in RESET_REQUIRED_COUNTERS
                    and float(sample.value) > 0
                }
            )
            drain_required = sorted(
                {
                    sample.name
                    for sample in samples
                    if sample.name in DRAIN_REQUIRED_COUNTERS
                    and float(sample.value) > 0
                }
            )
            previously_remediated = bool(
                reset_required
            ) and self._completed_hardware_remediation(node_id)
            if reset_required or drain_required or failed_conditions:
                action = (
                    RecoveryAction.DRAIN
                    if previously_remediated or drain_required
                    else RecoveryAction.RESET_GPU
                    if reset_required
                    else RecoveryAction.QUARANTINE
                )
                return TriageFinding(
                    node_id=node_id,
                    outcome=TriageOutcome.FAIL,
                    failed_checks=[
                        *failed_conditions,
                        *reset_required,
                        *drain_required,
                    ],
                    proposed_action=action,
                    evidence_refs=[metrics_url],
                    reason=(
                        "persistent GPU memory fault requires diagnostics"
                        if previously_remediated
                        else "GPU memory fault requires drain and diagnostics"
                        if drain_required
                        else "GPU memory repair requires reset"
                        if reset_required
                        else "quick node health checks failed"
                    ),
                )
            if not names.intersection(GPU_IDENTITY_METRICS):
                return TriageFinding(
                    node_id=node_id,
                    outcome=TriageOutcome.INCONCLUSIVE,
                    failed_checks=["gpu-enumeration"],
                    evidence_refs=[metrics_url],
                    reason="DCGM scrape contained no GPU identity metrics",
                )
            return TriageFinding(
                node_id=node_id,
                outcome=TriageOutcome.PASS,
                evidence_refs=[metrics_url],
                reason="Node conditions and critical DCGM counters passed",
            )
        except Exception as exc:
            return TriageFinding(
                node_id=node_id,
                outcome=TriageOutcome.INCONCLUSIVE,
                failed_checks=["dcgm-passive-health"],
                reason=f"{type(exc).__name__}: {exc}",
            )

    def _completed_hardware_remediation(self, node_id: str) -> bool:
        from datetime import timedelta

        from gpu_fault.models import WorkflowOperation
        from gpu_fault.store import NotFoundError

        cutoff = self.now() - timedelta(days=7)
        for workflow in reversed(self.store.list_workflows(limit=1000)):
            if workflow.updated_at < cutoff:
                continue
            if not {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTART_NODE,
            }.intersection(workflow.completed_operations):
                continue
            try:
                incident = self.store.get_incident(workflow.incident_id)
            except NotFoundError:
                continue
            if node_id in incident.node_ids:
                return True
        return False

    @staticmethod
    def _metadata(node: Any) -> Any:
        return node.get("metadata", {}) if isinstance(node, dict) else node.metadata

    @staticmethod
    def _status(node: Any) -> Any:
        return node.get("status", {}) if isinstance(node, dict) else node.status

    @classmethod
    def _internal_ip(cls, node: Any) -> str:
        status = cls._status(node)
        addresses = (
            status.get("addresses", [])
            if isinstance(status, dict)
            else status.addresses
        )
        for address in addresses or []:
            kind = address.get("type") if isinstance(address, dict) else address.type
            if kind == "InternalIP":
                value = (
                    address.get("address")
                    if isinstance(address, dict)
                    else address.address
                )
                if value:
                    return value
        raise ValueError("Node has no InternalIP")

    @classmethod
    def _failed_conditions(cls, node: Any) -> list[str]:
        status = cls._status(node)
        conditions = (
            status.get("conditions", [])
            if isinstance(status, dict)
            else status.conditions
        )
        failed = []
        for condition in conditions or []:
            kind = (
                condition.get("type") if isinstance(condition, dict) else condition.type
            )
            value = (
                condition.get("status")
                if isinstance(condition, dict)
                else condition.status
            )
            unhealthy = (kind == "Ready" and str(value) != "True") or (
                kind
                in {
                    "MemoryPressure",
                    "DiskPressure",
                    "PIDPressure",
                    "NetworkUnavailable",
                }
                and str(value) == "True"
            )
            if unhealthy:
                failed.append(f"kubernetes-node:{kind}")
        return sorted(failed)

    @staticmethod
    def _fetch(url: str, timeout_seconds: int) -> str:
        with urlopen(url, timeout=timeout_seconds) as response:
            return response.read().decode()
