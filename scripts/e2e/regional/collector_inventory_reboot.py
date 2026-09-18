"""Isolated inventory sampling followed by one scoped live recovery workflow."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeGuard, cast
from uuid import uuid4

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.collector_acceptance_fixture import (
    CollectorAcceptanceFixture,
    collector_setting,
)
from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup
from scripts.e2e.regional.collector_env_restore import restore_collector_env
from scripts.e2e.regional.collector_inventory_evidence import (
    debounce_sample_errors,
    evidence_time,
    run_collect003,
)
from scripts.e2e.regional.collector_inventory_sampling import (
    inventory_delivery_errors,
    isolated_inventory_records,
)
from scripts.e2e.regional.collector_reboot_evidence import (
    capture_reboot_scope,
    prove_reboot_scope,
    submitted_reboot_errors,
)
from scripts.e2e.regional.collector_recovery_safety import require_settled_recovery
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeMissingResponseError,
    HostProbeTransportError,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

REBOOT_EVENTS = {"BatchRebootClusterNodes", "RebootClusterNodes"}

HOST_INVENTORY_EVIDENCE = r"""
import json
import sys
from datetime import datetime
from gpu_fault.app import ApplicationContext
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, observed_after_text = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = []
raw = store.list_raw_evidence(
    cluster_id, node_id=node_id, kind=EvidenceKind.HOST_TELEMETRY, limit=500
)
if len(raw) >= 500 and raw[-1].observed_at >= observed_after:
    raise ValueError("inventory evidence scan is incomplete")
for item in raw:
    if item.observed_at < observed_after or item.kind != EvidenceKind.HOST_TELEMETRY:
        continue
    payload = item.payload
    samples = [
        sample for sample in payload.get("samples", [])
        if sample.get("name") == "gpu_inventory_mismatch"
    ]
    if not samples:
        continue
    if (
        item.cluster_id != cluster_id or item.node_id != node_id
        or payload.get("cluster_id") != cluster_id
        or payload.get("node_id") != node_id
        or payload.get("producer", "node") != "node"
        or not str(payload.get("batch_id") or "").startswith(f"host-{node_id}-")
        or item.record_id != f"host-telemetry/{payload['batch_id']}"
        or payload.get("collection_errors") != []
    ):
        raise ValueError("inventory evidence identity or collection is invalid")
    records.append({
        "record_id": item.record_id,
        "observed_at": item.observed_at.isoformat(),
        "batch_id": payload["batch_id"],
        "edge_filter_reasons": payload.get("edge_filter_reasons") or [],
        "samples": samples,
    })
records.sort(key=lambda item: item["observed_at"])
print(json.dumps({"records": records}, sort_keys=True))
"""

INVENTORY_RECOVERY_STATE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.hyperpod import hyperpod_submission_idempotency_key
from gpu_fault.store import NotFoundError

cluster_id, node_id, event_id, started_text, hyperpod_cluster = sys.argv[1:]
started = datetime.fromisoformat(started_text)
store = ApplicationContext.from_environment().store
incident = store.get_incident_by_event(event_id)
if incident is None:
    raise ValueError("inventory finding has no persisted incident")
if (
    incident.cluster_id != cluster_id or incident.node_ids != [node_id]
    or incident.event_id != event_id or incident.created_at < started
    or not incident.workflow_request_id
):
    raise ValueError("inventory incident is not owned by this injection")
rows = store.list_workflows(limit=500, newest_first=True)
if len(rows) >= 500 and rows[-1].updated_at >= started:
    raise ValueError("inventory workflow scan is incomplete")
workflows = [row for row in rows if row.incident_id == incident.incident_id]
if (
    not workflows
    or incident.workflow_request_id not in {row.request_id for row in workflows}
    or any(row.created_at < started for row in workflows)
):
    raise ValueError("inventory workflow identity or generation is invalid")
node_workflow_ids = []
for row in rows:
    if row.created_at < started:
        continue
    owner = store.get_incident(row.incident_id)
    if owner.cluster_id == cluster_id and node_id in owner.node_ids:
        node_workflow_ids.append(row.request_id)
commands = store.list_remote_commands(
    workflow_request_ids=[row.request_id for row in workflows]
)
if any(
    command.cluster_id != cluster_id or command.incident_id != incident.incident_id
    or command.workflow_request_id not in {row.request_id for row in workflows}
    for command in commands
):
    raise ValueError("inventory command identity is invalid")
submissions = []
for command in commands:
    if command.step.operation.value != "RESTART_NODE":
        continue
    key = command.result_details.get("submission_idempotency_key")
    if not key:
        key = hyperpod_submission_idempotency_key(
            command.workflow_request_id, command.step_index, command.step.operation
        )
    try:
        record = store.get_hyperpod_submission(hyperpod_cluster, key)
    except NotFoundError:
        continue
    submissions.append(record.model_dump(mode="json"))
print(json.dumps({
    "event_id": event_id,
    "incidents": [incident.model_dump(mode="json")],
    "workflows": [row.model_dump(mode="json") for row in workflows],
    "node_workflow_ids": node_workflow_ids,
    "commands": [
        command.model_dump(mode="json", exclude={"lease_token"}) for command in commands
    ],
    "submissions": submissions,
    "captured_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True, default=str))
"""


class InventoryRebootSettings(Protocol):
    @property
    def regional(self) -> RegionalLiveSettings: ...

    @property
    def node(self) -> str: ...

    @property
    def hyperpod_cluster(self) -> str: ...

    @property
    def executor_role_arn(self) -> str: ...

    @property
    def debounce_tolerance(self) -> float: ...


def mismatch_finding(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        for sample in record.get("samples") or []:
            if (
                sample.get("name") == "gpu_inventory_mismatch"
                and type(sample.get("value")) in {float, int}
                and sample["value"] == 1
            ):
                return {**record, "sample": sample}
    return None


def debounce_errors(
    records: list[dict[str, Any]],
    *,
    interval: int,
    required_samples: int,
    started_at: datetime,
    tolerance: float,
    expected_count: int | None = None,
) -> list[str]:
    del started_at
    return debounce_sample_errors(
        records,
        interval=interval,
        required_samples=required_samples,
        tolerance=tolerance,
        expected_count=expected_count,
    )


def wait_mismatch_finding(
    regional: RegionalLiveFixture,
    settings: InventoryRebootSettings,
    *,
    observed_after: datetime,
    timeout_seconds: int,
    batch_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        value = regional.cpu_python(
            HOST_INVENTORY_EVIDENCE,
            settings.regional.cluster_id,
            settings.node,
            observed_after.isoformat(),
        )
        records = [
            record
            for record in value.get("records") or []
            if batch_ids is None or record.get("batch_id") in batch_ids
        ]
        complete = batch_ids is None or batch_ids <= {
            record.get("batch_id") for record in records
        }
        if (
            complete and mismatch_finding(records) is not None
        ) or time.monotonic() >= deadline:
            return records
        time.sleep(5)


@dataclass
class InventoryWorkflow:
    """Track the product's incident and reboot, independently of test fixtures."""

    regional: RegionalLiveFixture
    settings: InventoryRebootSettings
    scope: Mapping[str, Any]
    run_id: str
    started_at: datetime
    observed_after: datetime
    expected_count: int
    baseline_sha256: str
    case_dir: Path
    attempt: int
    event_id: str | None = None
    previous_workflow_ids: set[str] = field(default_factory=set)
    last_state: dict[str, Any] = field(default_factory=dict)
    reboot_deadline_monotonic: float = field(
        default_factory=lambda: time.monotonic() + 1800
    )

    def bind_finding(self, records: list[dict[str, Any]]) -> None:
        finding = mismatch_finding(records)
        if finding is None:
            raise RegionalFixtureError("inventory injection finding is unresolved")
        sample = finding["sample"]
        batch_id = finding.get("batch_id")
        if (
            not isinstance(batch_id, str)
            or not batch_id.startswith(f"host-{self.settings.node}-")
            or finding.get("record_id") != f"host-telemetry/{batch_id}"
            or sample.get("device") is not None
            or (sample.get("labels") or {}).get("expected_count")
            != str(self.expected_count)
        ):
            raise RegionalFixtureError("inventory finding scope differs from injection")
        event_id = f"{batch_id}-gpu_inventory_mismatch-node"
        if self.event_id is not None and self.event_id != event_id:
            raise RegionalFixtureError("inventory injection finding identity changed")
        self.event_id = event_id

    def read(self) -> dict[str, Any]:
        if self.event_id is None:
            records = wait_mismatch_finding(
                self.regional,
                self.settings,
                observed_after=self.observed_after,
                timeout_seconds=0,
            )
            self.bind_finding(records)
        state = self.regional.cpu_python(
            INVENTORY_RECOVERY_STATE,
            self.settings.regional.cluster_id,
            self.settings.node,
            str(self.event_id),
            self.started_at.isoformat(),
            self.settings.hyperpod_cluster,
        )
        workflows = state.get("workflows")
        if (
            state.get("event_id") != self.event_id
            or not isinstance(workflows, list)
            or not workflows
            or any(
                not isinstance(row, dict) or not row.get("request_id")
                for row in workflows
            )
        ):
            raise RegionalFixtureError("inventory recovery snapshot is incomplete")
        identities = {row["request_id"] for row in workflows}
        node_workflow_ids = state.get("node_workflow_ids")
        if (
            not isinstance(node_workflow_ids, list)
            or any(not isinstance(item, str) or not item for item in node_workflow_ids)
            or len(set(node_workflow_ids)) != len(node_workflow_ids)
            or not identities <= set(node_workflow_ids)
        ):
            raise RegionalFixtureError(
                "inventory node workflow inventory is incomplete"
            )
        if not self.previous_workflow_ids <= identities:
            raise RegionalFixtureError("inventory recovery lost a tracked workflow")
        self.previous_workflow_ids = identities
        self.last_state = state
        write_json_atomic(
            self.case_dir / f"inventory-recovery-a{self.attempt}.json", state
        )
        return state

    def wait(self, *, terminal: bool, timeout_seconds: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = self.read()
            workflows = state["workflows"]
            if len(workflows) != 1 or len(state["node_workflow_ids"]) != 1:
                raise RegionalFixtureError(
                    "inventory injection grew "
                    f"{len(state['node_workflow_ids'])} node workflows, expected one"
                )
            workflow = cast(dict[str, Any], workflows[0])
            if not any(
                step.get("operation") == "RESTART_NODE"
                for step in workflow.get("official_steps") or []
            ):
                raise RegionalFixtureError(
                    "inventory workflow did not plan RESTART_NODE"
                )
            if not terminal:
                return workflow
            if workflow.get("status") in {
                "SUCCEEDED",
                "FAILED",
                "BLOCKED",
                "SUPERSEDED",
            }:
                return workflow
            if time.monotonic() >= deadline:
                raise RegionalFixtureError("inventory reboot workflow did not settle")
            time.sleep(5)

    def read_reboot_node(self) -> dict[str, Any]:
        current = self.regional.node_snapshot(self.settings.node)
        if current.get("uid") != self.scope["node_uid"]:
            raise RegionalFixtureError("reboot restore node UID changed")
        if (
            current.get("name") != self.settings.node
            or not isinstance(current.get("boot_id"), str)
            or not current["boot_id"]
            or current.get("ready") not in {"True", "False", "Unknown"}
        ):
            raise RegionalFixtureError("reboot node state or identity is incomplete")
        return current

    def wait_reboot(self) -> dict[str, Any]:
        while time.monotonic() < self.reboot_deadline_monotonic:
            current = self.read_reboot_node()
            if (
                current["ready"] == "True"
                and current["boot_id"] != self.scope["boot_id"]
            ):
                return current
            time.sleep(
                min(5, max(0, self.reboot_deadline_monotonic - time.monotonic()))
            )
        raise RegionalFixtureError(
            "original node did not complete reboot before deadline"
        )


class InventoryRecovery(InventoryWorkflow):
    """Compatibility for previously armed production-env override journals.

    New COLLECT-004 runs use InventoryWorkflow and never arm this protocol.
    """

    def __init__(
        self,
        regional: RegionalLiveFixture,
        settings: InventoryRebootSettings,
        scope: Mapping[str, Any],
        run_id: str,
        owner_nonce: str,
        started_at: datetime,
        observed_after: datetime,
        expected_count: int,
        baseline_sha256: str,
        case_dir: Path,
        attempt: int,
        *,
        event_id: str | None = None,
        previous_workflow_ids: set[str] | None = None,
    ) -> None:
        super().__init__(
            regional,
            settings,
            scope,
            run_id,
            started_at,
            observed_after,
            expected_count,
            baseline_sha256,
            case_dir,
            attempt,
            event_id=event_id,
            previous_workflow_ids=set(previous_workflow_ids or ()),
        )
        self.owner_nonce = owner_nonce
        self.arming: dict[str, Any] | None = None
        self.environment_restored = False
        self.mutation_not_started = False

    def read_for_cleanup(self) -> dict[str, Any]:
        if not self.environment_restored:
            raise RegionalFixtureError(
                "inventory configuration restoration is unresolved; operator hold retained"
            )
        if self.mutation_not_started:
            return {"incidents": [], "workflows": [], "commands": []}
        return self.read()

    def reboot_wait_authorized(self) -> bool:
        """A confirmed request permits bounded waiting, not a healthy-node claim."""
        if self.arming is None or time.monotonic() >= self.reboot_deadline_monotonic:
            return False
        state = self.read()
        if len(state["workflows"]) != 1 or len(state.get("submissions", [])) != 1:
            return False
        captured = evidence_time(state.get("captured_at"))
        if captured is None or not (
            -30 <= (datetime.now(timezone.utc) - captured).total_seconds() <= 30
        ):
            raise RegionalFixtureError("reboot submission read has no fresh clock")
        if submitted_reboot_errors(
            scope=self.scope,
            workflow=state["workflows"][0],
            commands=state["commands"],
            submission=state["submissions"][0],
            observed_at=captured,
        ):
            return False
        self.read_reboot_node()
        return time.monotonic() < self.reboot_deadline_monotonic

    def accept_arming(self, receipt: dict[str, Any]) -> None:
        if (
            receipt.get("run_id") != self.run_id
            or receipt.get("mutation_started") is not True
            or receipt.get("timer_armed") is not True
            or receipt.get("boot_restore_armed") is not True
            or receipt.get("cluster_id") != self.settings.regional.cluster_id
            or receipt.get("node_id") != self.settings.node
            or receipt.get("baseline_sha256") != self.baseline_sha256
            or receipt.get("boot_id") != self.scope["boot_id"]
            or not valid_sha256(receipt.get("applied_sha256"))
            or not valid_sha256(receipt.get("intent_sha256"))
        ):
            raise RegionalFixtureError("inventory override has no bound ARMED receipt")
        self.arming = receipt

    def restore(
        self, collector: CollectorAcceptanceFixture, *, allow_defer: bool = False
    ) -> dict[str, Any]:
        receipt = restore_collector_env(
            collector,
            self.run_id,
            owner_nonce=self.owner_nonce,
            reboot_transition=self.reboot_wait_authorized if allow_defer else None,
        )
        if receipt.get("deferred") is True:
            if not allow_defer or not self.reboot_wait_authorized():
                raise RegionalFixtureError("reboot deferral is not authorized")
            return receipt
        if (
            self.arming is None
            and not self.previous_workflow_ids
            and receipt.get("run_id") == self.run_id
            and receipt.get("state") in {"NOT_STARTED", "CLEANED"}
            and receipt.get("no_mutation") is True
            and receipt.get("mutation_started") is False
            and receipt.get("cleanup_verified") is True
            and receipt.get("timer_disarmed") is True
            and receipt.get("recovery_stopped") is True
        ):
            snapshot = collector.snapshot()
            if (snapshot.get("collector_env_file") or {}).get(
                "sha256"
            ) != self.baseline_sha256 or snapshot.get("boot_id") != self.scope[
                "boot_id"
            ]:
                raise RegionalFixtureError(
                    "rejected inventory setup changed its baseline"
                )
            self.environment_restored = True
            self.mutation_not_started = True
            return receipt
        if (
            receipt.get("run_id") != self.run_id
            or receipt.get("restored") is not True
            or receipt.get("cleanup_verified") is not True
            or receipt.get("timer_disarmed") is not True
            or receipt.get("state") != "CLEANED"
            or receipt.get("baseline_sha256") != self.baseline_sha256
            or not valid_sha256(receipt.get("intent_sha256"))
            or (
                self.arming is not None
                and receipt.get("intent_sha256") != self.arming["intent_sha256"]
            )
        ):
            raise RegionalFixtureError("inventory env restoration is not verified")
        self.environment_restored = True
        return receipt


def valid_sha256(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def scope_identity(
    scope: Mapping[str, Any], *, after_reboot: bool = False
) -> dict[str, Any]:
    volatile = {"observed_at"}
    if after_reboot:
        # A reboot may replace an Executor Pod; its Deployment/template/role must
        # remain identical and capture_reboot_scope revalidates its new population.
        volatile.update({"boot_id", "executor_pods"})
    identity = {key: value for key, value in scope.items() if key not in volatile}
    if not after_reboot and isinstance(identity.get("executor_pods"), list):
        # Each read may assume a fresh STS session for the already verified role.
        identity["executor_pods"] = [
            {key: value for key, value in pod.items() if key != "caller_arn"}
            for pod in identity["executor_pods"]
        ]
    return identity


def create_operation(case_dir: Path, attempt: int) -> tuple[str, str]:
    if type(attempt) is not int or attempt < 1:
        raise RegionalFixtureError("inventory attempt is invalid")
    digest = hashlib.sha256(f"{case_dir.resolve()}\0{attempt}".encode()).hexdigest()[
        :24
    ]
    run_id, nonce = f"collect004-{digest}-a{attempt}", uuid4().hex
    path = case_dir / f"inventory-operation-a{attempt}.json"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "owner_nonce": nonce}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(case_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return run_id, nonce


def require_idle_target(
    regional: RegionalLiveFixture, node: str, scope: Mapping[str, Any]
) -> None:
    current = regional.node_snapshot(node)
    if (
        current.get("uid") != scope["node_uid"]
        or current.get("boot_id") != scope["boot_id"]
        or current.get("ready") != "True"
        or current.get("unschedulable") is not False
        or current.get("taints") != []
        or current.get("ownership_annotations") != {}
        or regional.business_workloads(node)
    ):
        raise RegionalFixtureError("inventory reboot target is not idle and unchanged")


def run_collect004(
    settings: InventoryRebootSettings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    *,
    cleanup: CaseCleanup,
    expected_scope: dict[str, Any],
) -> dict[str, Any]:
    healthy = run_collect003(collector)
    if healthy["errors"]:
        raise RegionalFixtureError("live inventory baseline is not healthy")
    scope = capture_reboot_scope(
        regional,
        node=settings.node,
        hyperpod_cluster=settings.hyperpod_cluster,
        executor_role_arn=settings.executor_role_arn,
    )
    if scope_identity(scope) != scope_identity(expected_scope):
        raise RegionalFixtureError("inventory reboot scope changed after preflight")
    require_idle_target(regional, settings.node, scope)
    baseline = collector.snapshot()
    env = baseline["collector_env"]
    expected = collector_setting(env, "GPU_FAULT_EXPECTED_GPU_COUNT")
    interval = collector_setting(env, "GPU_FAULT_HOST_INTERVAL_SECONDS")
    required_samples = collector_setting(
        env, "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"
    )
    baseline_sha256 = (baseline.get("collector_env_file") or {}).get("sha256")
    observed_after = evidence_time(baseline.get("captured_at"))
    if (
        not valid_sha256(baseline_sha256)
        or observed_after is None
        or baseline.get("boot_id") != scope["boot_id"]
        or required_samples != 2
    ):
        raise RegionalFixtureError("inventory baseline identity or debounce differs")
    require_action_time(2400)
    run_id, _legacy_nonce = create_operation(case_dir, attempt)
    started_at = datetime.now(timezone.utc)
    recovery = InventoryWorkflow(
        regional,
        settings,
        scope,
        run_id,
        started_at,
        observed_after,
        expected + 1,
        baseline_sha256,
        case_dir,
        attempt,
    )
    result: dict[str, Any] = {
        "errors": [],
        "cleanup_errors": [],
        "run_id": run_id,
        "scope_before": scope,
        "inventory_before": healthy,
        "interval_seconds": interval,
        "required_consecutive_samples": required_samples,
        "sampling_mode": "isolated-collector",
        "production_configuration_modified": False,
        "production_service_restarted_by_runner": False,
    }
    publication_started = False

    def read_for_cleanup() -> dict[str, Any]:
        current = collector.snapshot()
        if (current.get("collector_env_file") or {}).get("sha256") != baseline_sha256:
            raise RegionalFixtureError(
                "production collector configuration changed; operator hold retained"
            )
        if not publication_started:
            return {"incidents": [], "workflows": [], "commands": []}
        return recovery.read()

    try:
        require_action_time(2400)
        sample = collector.execute(
            "sample-gpu-inventory",
            "--run-id",
            run_id,
            "--expected-env-sha256",
            baseline_sha256,
            "--expected-boot-id",
            scope["boot_id"],
            "--cluster-id",
            settings.regional.cluster_id,
            "--node-id",
            settings.node,
            "--expected-gpu-count",
            str(expected + 1),
            timeout=interval + 90,
        )
        captured = isolated_inventory_records(
            sample,
            run_id=run_id,
            cluster_id=settings.regional.cluster_id,
            node_id=settings.node,
            boot_id=scope["boot_id"],
            baseline_sha256=baseline_sha256,
            expected_gpu_count=expected + 1,
            interval_seconds=interval,
            observed_after=observed_after,
            tolerance=settings.debounce_tolerance,
        )
        write_json_atomic(case_dir / f"isolated-inventory-a{attempt}.json", sample)
        result["isolated_sampling"] = {
            "verified": True,
            "sample_sha256": sample["sha256"],
            "evidence": f"isolated-inventory-a{attempt}.json",
            "live_delivery_proven": False,
        }
        # The sampler has returned and cannot publish. Bind and persist the
        # exact event before the separate, finite publisher may trigger reboot.
        recovery.bind_finding(captured)
        batch_ids = frozenset(record["batch_id"] for record in captured)
        write_json_atomic(
            case_dir / f"inventory-publication-a{attempt}.json",
            {
                "run_id": run_id,
                "sample_sha256": sample["sha256"],
                "batch_ids": sorted(batch_ids),
                "event_id": recovery.event_id,
                "node_uid": scope["node_uid"],
                "boot_id": scope["boot_id"],
            },
        )
        cleanup.register_refresh(collector, read_for_cleanup)
        require_action_time(2400)
        publication_started = True
        try:
            publication = collector.execute(
                "publish-gpu-inventory",
                "--run-id",
                run_id,
                "--expected-sha256",
                sample["sha256"],
                "--receipt-json",
                json.dumps(sample, sort_keys=True, allow_nan=False),
                "--confirm",
                "PUBLISH_GPU_INVENTORY",
                timeout=90,
            )
            if (
                publication.get("run_id") != run_id
                or publication.get("sample_sha256") != sample["sha256"]
                or publication.get("publication_performed") is not True
                or publication.get("batch_ids") != [row["batch_id"] for row in captured]
            ):
                raise RegionalFixtureError("inventory publication receipt differs")
            result["publication"] = publication
        except (HostProbeMissingResponseError, HostProbeTransportError) as exc:
            # No replay: a lost response is resolved by the existing ingress
            # records and product workflow below, never by another publication.
            result["publication"] = {
                "response_unknown": True,
                "error_type": type(exc).__name__,
                "replayed": False,
            }
        records = wait_mismatch_finding(
            regional,
            settings,
            observed_after=observed_after,
            timeout_seconds=interval * (required_samples + 4) + 60,
            batch_ids=batch_ids,
        )
        result["mismatch_records"] = records
        delivery_errors = inventory_delivery_errors(captured, records)
        if delivery_errors:
            raise RegionalFixtureError("; ".join(delivery_errors))
        result["isolated_sampling"]["live_delivery_proven"] = True
        result["errors"].extend(
            debounce_errors(
                records,
                interval=interval,
                required_samples=required_samples,
                started_at=started_at,
                tolerance=settings.debounce_tolerance,
                expected_count=expected + 1,
            )
        )
        workflow = recovery.wait(terminal=False, timeout_seconds=600)
        result["restart_workflow_id"] = workflow["request_id"]
        node_after = recovery.wait_reboot()
        result["node_after"] = node_after
        if (
            node_after.get("uid") != scope["node_uid"]
            or not node_after.get("boot_id")
            or node_after["boot_id"] == scope["boot_id"]
        ):
            raise RegionalFixtureError(
                "reboot node identity or changed boot is unproven"
            )
        collector.recreate()
        workflow = recovery.wait(terminal=True, timeout_seconds=1200)
        state = recovery.read()
        result["workflow_state"] = state
        require_settled_recovery(state)
        if workflow.get("status") != "SUCCEEDED":
            result["errors"].append("inventory reboot workflow is not SUCCEEDED")
        after = collector.snapshot()
        result["collector_env_after"] = after.get("collector_env_file")
        if (after.get("collector_env_file") or {}).get("sha256") != baseline_sha256:
            result["errors"].append("collector.env differs from the baseline")
        regional.wait_provider_events(
            started_at,
            event_names=REBOOT_EVENTS,
            expected_count=1,
        )
        ended_at = datetime.now(timezone.utc)
        events = regional.provider_events(started_at, ended_at)
        result["provider_events"] = events
        submissions = state.get("submissions", [])
        if len(submissions) != 1:
            raise RegionalFixtureError("inventory reboot submission is not unique")
        proof = prove_reboot_scope(
            scope,
            events=events,
            workflow=workflow,
            commands=state["commands"],
            submission=submissions[0],
            started_at=started_at,
            ended_at=ended_at,
        )
        result["provider_proof"] = proof
        result["errors"].extend(proof["errors"])
        result["provider_events_provisional"] = regional.provider_events_provisional(
            ended_at
        )
        result["inventory_after"] = run_collect003(collector)
        result["errors"].extend(result["inventory_after"]["errors"])
        after_scope = capture_reboot_scope(
            regional,
            node=settings.node,
            hyperpod_cluster=settings.hyperpod_cluster,
            executor_role_arn=settings.executor_role_arn,
        )
        result["scope_after"] = after_scope
        if scope_identity(after_scope, after_reboot=True) != scope_identity(
            scope, after_reboot=True
        ):
            result["errors"].append(
                "provider or Kubernetes identity changed after reboot"
            )
    except Exception as exc:
        result["errors"].append(f"inventory reboot failed: {type(exc).__name__}: {exc}")
    except BaseException as exc:
        result["errors"].append(f"inventory reboot interrupted: {type(exc).__name__}")
        raise
    finally:
        # There is no production-env undo operation. Preserve any drift and
        # leave failed/unknown product recovery to the normal guarded cleanup.
        result["publication_started"] = publication_started
        result["recovery_reference"] = f"inventory-recovery-a{attempt}.json"
        result["verdict"] = (
            "PASS" if not (result["errors"] or result["cleanup_errors"]) else "FAIL"
        )
        write_json_atomic(
            case_dir / f"inventory-reboot-progress-a{attempt}.json", result
        )
    return result
