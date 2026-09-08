from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError

from gpu_fault.channel_registry import (
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    SPOOLABLE_CHANNEL_PATHS,
)
from gpu_fault.gpu_metrics import (
    GpuInventorySnapshot,
    GpuMetricBatch,
    GpuMetricsIngestionResult,
)
from gpu_fault.host_health import (
    HostTelemetryBatch,
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
    NodeLogBatch,
)
from gpu_fault.models import RecoveryAction, Severity
from gpu_fault.store.shared.health_signals import finding_health_signal_key
from gpu_fault.telemetry import CollectorKind, EvidenceKind

LOGGER = logging.getLogger(__name__)
TelemetryBatch = GpuMetricBatch | HostTelemetryBatch | NodeLogBatch


class TelemetryIngestionService:
    def __init__(
        self,
        context,
        telemetry_context,
        node_health,
        fault_ingestion,
    ) -> None:
        self.context = context
        self.telemetry_context = telemetry_context
        self.node_health = node_health
        self.fault_ingestion = fault_ingestion
        self.telemetry_batch_handlers = {
            GPU_METRICS_PATH: (
                GpuMetricBatch,
                self._persist_gpu_metrics,
                self._finish_gpu_metrics,
            ),
            HOST_TELEMETRY_PATH: (
                HostTelemetryBatch,
                self._persist_host_telemetry,
                self._finish_host_telemetry,
            ),
            NODE_LOG_PATH: (
                NodeLogBatch,
                self._persist_node_logs,
                self._finish_node_logs,
            ),
        }
        # Inventory is batched too, but cross-node and latest-wins, so it
        # cannot share the per-path persist/finish table above. Declare it
        # here and fail closed when a channel the collector is allowed to
        # spool has no batch path at all: without this a new spoolable
        # channel would spool on the node and then 422 on every drain.
        self.telemetry_batch_paths = frozenset(
            {
                GPU_INVENTORY_PATH,
                *self.telemetry_batch_handlers,
            }
        )
        unhandled = sorted(SPOOLABLE_CHANNEL_PATHS - self.telemetry_batch_paths)
        if unhandled:
            raise RuntimeError(
                "spoolable collector channels have no telemetry "
                f"batch handler: {unhandled}"
            )

    def _persist_gpu_metrics(
        self, batch: GpuMetricBatch, observations=None
    ) -> tuple[GpuMetricBatch, GpuMetricsIngestionResult]:
        """The half of gpu-metrics ingestion that must be one commit.

        Split out so a claim batch can persist every item in a single
        transaction; only the dispatcher wake lives after the commit, in
        ``_finish_gpu_metrics``.

        The incidents are created *here*, inside the caller's ingestion
        transaction, and not after it (F-M1 / P0-31A / P0-37A). When they
        were created after the commit, a failure or a crash between the two
        halves left a CRITICAL finding state that no incident named: the next
        batch for the node saw the same severity, ``update_gpu_findings``
        reported nothing new, and the finding stayed orphaned for as long as
        the fault persisted. On Postgres the orchestrator's own state
        transactions nest as savepoints inside ``collector_ingestion_transaction``,
        so finding, batch record, marker, incident and workflow now commit
        together or not at all; a same-batch replay after a rollback recomputes
        them once, and a same-batch replay after a commit is the duplicate
        fast path plus idempotent incident lookups.
        """

        batch = self.telemetry_context._enrich_workload_context(
            batch,
            batch.observed_at,
            observations=observations,
        )
        if batch.ingested_at is None:
            batch = batch.model_copy(update={"ingested_at": datetime.now(timezone.utc)})
        result = self.context.gpu_metrics.ingest(batch)
        self.telemetry_context._record_collector_status(
            CollectorKind.GPU_METRICS,
            batch.cluster_id,
            batch.node_id,
            batch.observed_at,
            batch.batch_id,
            result.accepted_samples,
            batch.collection_errors,
        )
        if (
            result.new_findings
            or result.xid_events
            or set(batch.edge_filter_reasons) != {"health-summary"}
        ):
            self.telemetry_context._capture_evidence(
                record_id=f"gpu-metrics/{batch.batch_id}",
                cluster_id=batch.cluster_id,
                node_id=batch.node_id,
                kind=EvidenceKind.GPU_METRICS,
                observed_at=batch.observed_at,
                payload=batch.model_dump(mode="json"),
                observations=observations,
            )
        return batch, self._ingest_gpu_findings(batch, result)

    def _finish_gpu_metrics(
        self,
        batch: GpuMetricBatch,
        result: GpuMetricsIngestionResult,
    ) -> GpuMetricsIngestionResult:
        """What runs after the ingestion transaction has committed.

        Only the dispatcher wake: waking before the commit would let the
        dispatcher poll a workflow row that is not visible yet.
        """

        del batch
        self.context.dispatcher.wake()
        return result

    def _ingest_gpu_findings(
        self,
        batch: GpuMetricBatch,
        result: GpuMetricsIngestionResult,
    ) -> GpuMetricsIngestionResult:
        """Turn one batch's new findings and XID events into incidents.

        Runs inside the ingestion transaction (see ``_persist_gpu_metrics``).
        Every path in here is idempotent by event id, so a replayed batch --
        whose result is the stored one with ``duplicate=True`` -- reaches the
        incidents it already created rather than nothing.
        """

        decisions = [
            self.fault_ingestion.ingest_xid(event) for event in result.xid_events
        ]
        node_findings = []
        for finding in result.new_findings:
            severity = (
                Severity.CRITICAL
                if finding.severity.value == "CRITICAL"
                else Severity.WARNING
            )
            action = (
                RecoveryAction(finding.automatic_action)
                if finding.automatic_action
                else self.context.orchestrator.gpu_metric_action(
                    cluster_id=finding.cluster_id,
                    node_id=finding.node_id,
                    metric_name=finding.canonical_name,
                    has_explicit_gpu=bool(finding.gpu_uuid),
                    default=(
                        RecoveryAction.QUARANTINE
                        if severity is Severity.CRITICAL
                        else RecoveryAction.RUN_DIAGNOSTICS
                    ),
                )
            )
            node_findings.append(
                NodeHealthFinding(
                    finding_id=finding.finding_id,
                    event_id=f"gpu-{finding.finding_id}",
                    cluster_id=finding.cluster_id,
                    node_id=finding.node_id,
                    observed_at=finding.observed_at,
                    category=NodeHealthCategory.GPU,
                    severity=severity,
                    reason=(
                        finding.reason
                        if finding.finding_kind == "METRIC"
                        else (
                            f"{finding.reason}; correlation_rule="
                            f"{finding.correlation_rule_id}; "
                            "component_metrics="
                            f"{','.join(finding.component_metrics)}; "
                            f"confidence={finding.confidence}"
                        )
                    ),
                    recommended_action=action,
                    metric_name=finding.canonical_name,
                    value=finding.value,
                    device=(
                        finding.gpu_uuid
                        or finding.pci_bdf
                        or (
                            ",".join(finding.affected_gpu_uuids)
                            if finding.affected_gpu_uuids
                            else None
                        )
                    ),
                    pci_bdf=finding.pci_bdf,
                    gpu_uuids=(
                        finding.affected_gpu_uuids
                        or ([finding.gpu_uuid] if finding.gpu_uuid else [])
                    ),
                    evidence_ref=finding.evidence_ref,
                    runtime_profile_version=(finding.runtime_profile_version),
                    workload_state=finding.workload_state,
                    affected_workload_ids=(finding.affected_workload_ids),
                    policy_version=finding.policy_version,
                    policy_source=finding.policy_source,
                    policy_reference=finding.policy_reference,
                    official_action=finding.official_action,
                )
            )
        if node_findings:
            self.node_health.ingest(batch.batch_id, node_findings)
        return result.model_copy(update={"decisions": decisions})

    def _persist_host_telemetry(
        self, batch: HostTelemetryBatch, observations=None
    ) -> tuple[HostTelemetryBatch, NodeHealthIngestionResult]:
        """The half of host-telemetry ingestion that must be one commit.

        See ``_persist_gpu_metrics``: the findings become incidents and
        workflows here, in the same transaction as the health-signal claim
        that produced them. The claim does not latch ``notified``; the finish
        half does, after this transaction committed, so a failed incident write
        -- on any backend, transactional or not -- leaves the signal free to
        emit again on its next sample (P0-38B).
        """

        batch = self.telemetry_context._enrich_workload_context(
            batch,
            batch.observed_at,
            observations=observations,
        )
        if batch.received_at is None:
            # Sustained-signal windows run on this clock, not the node's (F-M2).
            batch = batch.model_copy(update={"received_at": datetime.now(timezone.utc)})
        findings = self.context.node_health.evaluate_metrics(batch)
        self.telemetry_context._record_collector_status(
            CollectorKind.HOST_TELEMETRY,
            batch.cluster_id,
            batch.node_id,
            batch.observed_at,
            batch.batch_id,
            len(batch.samples),
            batch.collection_errors,
        )
        if (
            findings
            or batch.collection_errors
            or set(batch.edge_filter_reasons) != {"health-summary"}
        ):
            self.telemetry_context._capture_evidence(
                record_id=f"host-telemetry/{batch.batch_id}",
                cluster_id=batch.cluster_id,
                node_id=batch.node_id,
                kind=EvidenceKind.HOST_TELEMETRY,
                observed_at=batch.observed_at,
                payload=batch.model_dump(mode="json"),
                observations=observations,
            )
        return batch, self.node_health.ingest(batch.batch_id, findings)

    def _finish_host_telemetry(
        self, batch: HostTelemetryBatch, result: NodeHealthIngestionResult
    ) -> NodeHealthIngestionResult:
        # The commit that carried the incident and its advisory notification
        # is the delivery: latch the signals that emitted only now (P0-38B).
        # Before the wake, which is not part of delivering anything -- a
        # failed wake must not re-emit a fault that already has its incident.
        notified_at = batch.received_at or batch.observed_at
        for finding in result.findings:
            self.context.store.mark_health_signal_notified(
                finding_health_signal_key(finding), notified_at=notified_at
            )
        self.context.dispatcher.wake()
        return result

    def _persist_node_logs(
        self,
        batch: NodeLogBatch,
        observations=None,
    ) -> tuple[NodeLogBatch, NodeHealthIngestionResult]:
        batch = self.telemetry_context._enrich_workload_context(
            batch,
            batch.collected_at,
            observations=observations,
        )
        findings = (
            self.context.node_health.evaluate_logs(batch) if batch.entries else []
        )
        # See ``ingest_node_logs``: the errors decide whether this batch counts
        # as a success, and an error-only batch must not.
        self.telemetry_context._record_collector_status(
            CollectorKind.NODE_LOGS,
            batch.cluster_id,
            batch.node_id,
            batch.collected_at,
            batch.batch_id,
            len(batch.entries),
            batch.collection_errors,
        )
        # Kept for an error-only batch as well: it carries no entries and is the
        # only record that this node's log collection is failing.
        if batch.entries or batch.collection_errors:
            self.telemetry_context._capture_evidence(
                record_id=f"node-logs/{batch.batch_id}",
                cluster_id=batch.cluster_id,
                node_id=batch.node_id,
                kind=EvidenceKind.NODE_LOGS,
                observed_at=batch.collected_at,
                payload=batch.model_dump(mode="json"),
                observations=observations,
            )
        # Incidents in the same transaction as the evidence, as for the other
        # two channels (F-M1).
        return batch, self.node_health.ingest(batch.batch_id, findings)

    def _finish_node_logs(
        self,
        batch: NodeLogBatch,
        result: NodeHealthIngestionResult,
    ) -> NodeHealthIngestionResult:
        del batch
        self.context.dispatcher.wake()
        return result

    def _telemetry_body(self, value):
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else value

    def _telemetry_item_failure(
        self,
        request_id: str | None,
        path: str | None,
        exc: Exception,
    ) -> dict:
        if isinstance(exc, ValidationError):
            return {
                "request_id": request_id,
                "status": 422,
                "body": {"detail": exc.errors(include_url=False)},
            }
        if isinstance(exc, HTTPException):
            return {
                "request_id": request_id,
                "status": exc.status_code,
                "body": {"detail": exc.detail},
            }
        LOGGER.exception(
            "telemetry batch item failed request_id=%s path=%s",
            request_id,
            path,
        )
        return {
            "request_id": request_id,
            "status": 500,
            "body": {"detail": f"{type(exc).__name__}: {exc}"},
        }

    def _ingest_telemetry_item(self, request_id: str, path: str, batch) -> dict:
        """One telemetry item in its own transaction."""

        _model, persist, finish = self.telemetry_batch_handlers[path]
        try:
            with self.context.store.collector_ingestion_transaction(
                batch.cluster_id,
                batch.node_id,
                batch.batch_id,
            ):
                persisted, value = persist(batch)
            body = finish(persisted, value)
        except Exception as exc:
            return self._telemetry_item_failure(request_id, path, exc)
        return {
            "request_id": request_id,
            "status": 200,
            "body": self._telemetry_body(body),
        }

    def _ingest_telemetry_group(
        self,
        items: list[tuple[str, str, TelemetryBatch]],
    ):
        """Persist a whole claim batch of telemetry in one commit.

        What limits this deployment is commits per second, not CPU: the
        Postgres pool runs autocommit, so every statement issued outside
        a transaction is its own commit. A batch of eight used to cost
        eight ingestion transactions plus eight standalone topology
        reads; one transaction for the batch makes that one commit. The
        per-item ``collector_ingestion_transaction`` nests as a savepoint,
        so one bad payload still fails alone and its siblings still
        commit -- the same per-item statuses the caller had before.

        Items are walked in (cluster, node) order because a group holds
        several nodes' locks at once. Both telemetry paths take
        ``raw_evidence/<cluster>/<node>``, so two groups with overlapping
        node sets would deadlock if each walked its nodes in claim order.

        Incidents, workflows, markers and notifications are written inside
        each item's savepoint, so a failed incident write takes that item's
        findings back with it and its siblings keep theirs (F-M1: the
        two-phase window does not widen to the whole claim batch). Only the
        dispatcher wake runs after the commit, exactly as it does for a
        single request.
        """

        ordered = sorted(
            items,
            key=lambda item: (
                item[2].cluster_id,
                item[2].node_id,
                item[2].batch_id,
            ),
        )
        results: list[dict] = []
        persisted_items: list[tuple[str, str, object, object]] = []
        observations: dict[str, list] = {}

        def observed(cluster_id: str) -> list:
            # One topology read per cluster instead of one per item; the
            # age filter in resolve() is what bounds staleness anyway.
            if cluster_id not in observations:
                observations[cluster_id] = self.context.store.list_attempt_observations(
                    cluster_id
                )
            return observations[cluster_id]

        try:
            with self.context.store.processor_batch_transaction():
                for request_id, path, batch in ordered:
                    _model, persist, _finish = self.telemetry_batch_handlers[path]
                    try:
                        with self.context.store.collector_ingestion_transaction(
                            batch.cluster_id,
                            batch.node_id,
                            batch.batch_id,
                        ):
                            persisted, value = persist(
                                batch,
                                observed(batch.cluster_id),
                            )
                    except Exception as exc:
                        results.append(
                            self._telemetry_item_failure(request_id, path, exc)
                        )
                        continue
                    persisted_items.append((request_id, path, persisted, value))
        except Exception:
            # The batch commit failed, so nothing above is durable and no
            # follow-up has run. Retry the items one transaction at a
            # time rather than failing a whole claim batch on one
            # aborted transaction.
            LOGGER.exception(
                "telemetry batch commit failed; retrying %d items one at a time",
                len(ordered),
            )
            return [
                self._ingest_telemetry_item(request_id, path, batch)
                for request_id, path, batch in ordered
            ]
        for (
            request_id,
            path,
            batch,
            value,
        ) in persisted_items:
            _model, _persist, finish = self.telemetry_batch_handlers[path]
            try:
                body = finish(batch, value)
            except Exception as exc:
                results.append(self._telemetry_item_failure(request_id, path, exc))
                continue
            results.append(
                {
                    "request_id": request_id,
                    "status": 200,
                    "body": self._telemetry_body(body),
                }
            )
        return results

    def _ingest_processor_telemetry_batch_core(
        self,
        batch_request: dict,
    ) -> dict:
        raw_items = batch_request.get("items")
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= 64:
            raise HTTPException(
                status_code=422,
                detail="telemetry batch must contain 1..64 items",
            )
        results = []
        inventory_items = []
        group_items = []
        for raw_item in raw_items:
            request_id = raw_item.get("request_id")
            path = raw_item.get("path")
            payload = raw_item.get("payload")
            # Every payload is validated before any transaction opens, so
            # a malformed item is 422 on its own instead of aborting the
            # group commit below.
            if request_id and path in self.telemetry_batch_paths:
                model = (
                    GpuInventorySnapshot
                    if path == GPU_INVENTORY_PATH
                    else self.telemetry_batch_handlers[path][0]
                )
                try:
                    parsed = model.model_validate(payload)
                except ValidationError as exc:
                    results.append(
                        {
                            "request_id": request_id,
                            "status": 422,
                            "body": {"detail": exc.errors(include_url=False)},
                        }
                    )
                    continue
                if path == GPU_INVENTORY_PATH:
                    inventory_items.append((request_id, parsed))
                else:
                    group_items.append((request_id, path, parsed))
                continue
            results.append(
                {
                    "request_id": request_id,
                    "status": 422,
                    "body": {"detail": "unsupported telemetry batch item"},
                }
            )

        if inventory_items:
            try:
                persisted = self.telemetry_context._ingest_gpu_inventory_batch(
                    [snapshot for _, snapshot in inventory_items]
                )
                results.extend(
                    {
                        "request_id": request_id,
                        "status": 200,
                        "body": snapshot.model_dump(mode="json"),
                    }
                    for (
                        request_id,
                        _input_snapshot,
                    ), snapshot in zip(
                        inventory_items,
                        persisted,
                        strict=True,
                    )
                )
            except Exception as exc:
                LOGGER.exception("GPU inventory batch failed")
                results.extend(
                    {
                        "request_id": request_id,
                        "status": 500,
                        "body": {"detail": (f"{type(exc).__name__}: {exc}")},
                    }
                    for request_id, _ in inventory_items
                )

        if group_items:
            results.extend(self._ingest_telemetry_group(group_items))
        return {"results": results}
