from __future__ import annotations

import json
import logging
import os
import random
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPException
from threading import Event, Lock, Thread
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

from pydantic import ValidationError

from gpu_fault.adapters import (
    HyperPodLifecycleStepAdapter,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
)
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.aws_errors import (
    aws_configuration_error,
    missing_aws_credentials,
)
from gpu_fault.env_validation import (
    validate_gpu_fault_environment,
)
from gpu_fault.env import env_bool
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.transient_errors import retryable_adapter_error
from gpu_fault.execution.fleet_preflight import (
    command_requires_fleet_preflight,
    fleet_preflight_reason,
)
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentRecord,
    AgentTransitionRequest,
    FleetReadinessReport,
    FleetReadinessRequest,
)
from gpu_fault.hyperpod import (
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
    HyperPodSubmissionRecord,
)
from gpu_fault.hyperpod_spares import (
    HyperPodSpareCoordinator,
)
from gpu_fault.logging_setup import configure_logging
from gpu_fault.models import (
    AdvisoryNotification,
    WorkflowStepExecution,
    WorkflowStepStatus,
    execution_phase,
)
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS
from gpu_fault.regional import (
    RegionalExecutorReadinessRequest,
    RemoteActionCommand,
    RemoteAdvisoryNotificationRequest,
    RemoteCommandClaim,
    RemoteCommandClaimRequest,
    RemoteCommandLeaseRenewal,
    RemoteCommandResult,
    RemoteCommandStatus,
    RemoteEvidenceCaptureRequest,
    RemoteFleetRolloutFence,
    RemoteHyperPodSubmissionRequest,
    RemoteHyperPodSubmissionReservation,
    RemoteIncidentOwnershipReport,
    RemoteSpareHealthReport,
    RemoteSpareHealthRequest,
)
from gpu_fault.regional_compatibility import (
    CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
)
from gpu_fault.telemetry import RawEvidenceRecord
from gpu_fault.transport.http_client import urlopen
from gpu_fault.transport_errors import (
    retryable_transport_result,
)

LOGGER = logging.getLogger(__name__)


class ClusterExecutorError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ClusterExecutorClaimError(ClusterExecutorError):
    """The claim round trip itself failed, so no command was executed.

    ``run()`` used to log every ``run_once`` exception as "claim failed",
    including a result that could not be posted and a claim response the
    executor could not use -- which pointed the operator at the wrong side of
    the wire. Everything else that can still leave ``run_once`` is a defect
    after the claim, and says so.
    """


# Reporting a result is not a fire-once operation: the action has already run
# on the cluster, so a dropped connection on the result post costs a whole
# lease of latency and then a second execution after the command is re-claimed.
# Three attempts is enough for the baseline "Remote end closed connection"
# (~1/min) and for a control-plane rollout's 503 window, and short enough that
# the lease (>= 10 s, 120 s in production) is still ours while retrying.
# Three attempts means two sleeps, so there are two delays and not three: the
# worst case with jitter is 0.75 + 1.5 = 2.25s of extra latency on the command's
# own worker thread.
_RESULT_REPORT_ATTEMPTS = 3
_RESULT_REPORT_BACKOFF_SECONDS = (0.5, 1.0)
_RETRYABLE_REPORT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
# Transport failures reach the caller as several unrelated families, and only
# HTTPError used to be converted. ``socket.timeout`` is an alias of
# ``TimeoutError`` on 3.12 and is named here for readers, not for coverage.
# ``HTTPException`` covers the connection that dies mid-response
# (``IncompleteRead``, ``BadStatusLine``): no verdict was received, so it has to
# be retryable rather than escaping raw.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    URLError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
    ConnectionError,
    HTTPException,
)


def _retryable_report_failure(exc: BaseException) -> bool:
    """Whether re-posting a result could plausibly land.

    Retry only failures that carry no verdict about the result: no HTTP status
    at all (the request never got an answer), or one of the statuses that means
    "not now". A 403, 409 or 422 is the control plane's decision -- re-sending
    it would just be refused again.
    """

    if isinstance(exc, ClusterExecutorError):
        return (
            exc.status_code is None or exc.status_code in _RETRYABLE_REPORT_STATUS_CODES
        )
    return isinstance(exc, _TRANSPORT_ERRORS)


def _report_backoff_seconds(attempt: int) -> float:
    """Jittered delay before report attempt ``attempt + 1``.

    Jitter matters here because both replicas can be reporting into the same
    control-plane rollout window; an un-jittered 0.5/1 retries in lockstep.
    """

    base = _RESULT_REPORT_BACKOFF_SECONDS[
        min(attempt, len(_RESULT_REPORT_BACKOFF_SECONDS)) - 1
    ]
    return base + random.random() * (base / 2)


def _validation_summary(exc: ValidationError) -> str:
    """A bounded, operator-readable summary of why a payload did not parse.

    ``str(ValidationError)`` for a rejected enum value prints every accepted
    member, which is far too long for a result the control plane stores and an
    operator reads in a step's error field.
    """

    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {str(error.get('msg', ''))[:120]}")
    detail = "; ".join(parts) or "no field-level detail"
    return ("this executor could not parse the claimed command: " + detail)[:500]


def _persistent_store_from_environment():
    store_url = os.getenv("GPU_FAULT_STORE_URL", "").strip()
    if not store_url:
        return None
    if store_url.startswith("sqlite:///"):
        from gpu_fault.store import SqliteStore

        return SqliteStore(store_url.removeprefix("sqlite:///"))
    if store_url.startswith(("postgresql://", "postgres://")):
        from gpu_fault.store import PostgresStore

        return PostgresStore(
            store_url,
            pool_min_size=int(os.getenv("GPU_FAULT_POSTGRES_POOL_MIN_SIZE", "1")),
            pool_max_size=int(os.getenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "4")),
            pool_timeout_seconds=float(
                os.getenv(
                    "GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS",
                    "2",
                )
            ),
        )
    raise ClusterExecutorError("GPU_FAULT_STORE_URL must use sqlite:/// or PostgreSQL")


class RegionalExecutorClient:
    def __init__(
        self,
        base_url: str,
        cluster_id: str,
        token: str,
        *,
        timeout_seconds: float = 15,
        ca_file: str | None = None,
        executor_artifact_sha256: str | None = None,
        executor_compatibility_digest: str | None = None,
    ) -> None:
        self.base_url = self._clean_value("base_url", base_url).rstrip("/")
        self.cluster_id = self._clean_value("cluster_id", cluster_id)
        self.token = self._clean_value("token", token)
        self.timeout_seconds = timeout_seconds
        self.executor_artifact_sha256 = (
            self._clean_value(
                "executor_artifact_sha256",
                executor_artifact_sha256,
            )
            if executor_artifact_sha256
            else None
        )
        self.executor_compatibility_digest = (
            self._clean_value(
                "executor_compatibility_digest",
                executor_compatibility_digest,
            )
            if executor_compatibility_digest
            else self.executor_artifact_sha256
        )
        self.ssl_context = (
            ssl.create_default_context(cafile=ca_file) if ca_file else None
        )

    @staticmethod
    def _clean_value(name: str, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ClusterExecutorError(f"{name} must not be empty")
        if any(ord(character) < 32 or ord(character) == 127 for character in cleaned):
            raise ClusterExecutorError(f"{name} must not contain control characters")
        return cleaned

    def _send(self, request: Request) -> bytes:
        """One control-plane round trip; every failure becomes one exception type.

        Only ``HTTPError`` used to be converted, so every caller of this client
        and of the regional proxies built on it had a second, unhandled
        exception family to know about: a ``URLError``, a read timeout or a TLS
        error propagated raw. That is how a timeout on the result post escaped
        the command worker thread, failed ``run_once``, and left the command
        LEASED until it expired and was executed a second time. A failure with
        no answer at all carries ``status_code=None``, which is what callers
        read as "no verdict from the control plane, safe to retry".
        """

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
                ssl_context=self.ssl_context,
            ) as response:
                body: bytes = response.read()
                return body
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ClusterExecutorError(
                f"regional control plane rejected request ({exc.code}): {detail}",
                status_code=exc.code,
            ) from exc
        except _TRANSPORT_ERRORS as exc:
            raise ClusterExecutorError(
                f"regional control plane request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":"), default=str).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "X-GPU-Fault-Cluster-ID": self.cluster_id,
            },
            method="POST",
        )
        return json.loads(self._send(request) or b"{}")

    def _get(self, path: str) -> Any:
        request = Request(
            self.base_url + path,
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GPU-Fault-Cluster-ID": self.cluster_id,
            },
            method="GET",
        )
        return json.loads(self._send(request) or b"null")

    def claim(
        self,
        executor_id: str,
        *,
        execution_owners: list[str] | None = None,
        max_commands: int,
        lease_seconds: int,
    ) -> list[RemoteActionCommand]:
        payload = RemoteCommandClaimRequest(
            executor_id=executor_id,
            executor_protocol_version=(CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION),
            executor_artifact_sha256=self.executor_artifact_sha256,
            executor_compatibility_digest=self.executor_compatibility_digest,
            execution_owners=execution_owners or [],
            max_commands=max_commands,
            lease_seconds=lease_seconds,
        )
        response = self._post(
            "/v1/regional/executors/claim",
            payload.model_dump(mode="json"),
        )
        return self._claimed_commands(response)

    def _claimed_commands(self, response: dict[str, Any]) -> list[RemoteActionCommand]:
        """Parse each claimed command on its own, failing only the unparseable.

        The control plane commits the leases before this executor validates
        anything, so validating the batch as one document made a single command
        this build cannot parse (a value added to an enum, a
        control-plane-only field, a hotfix build past the compatibility digest)
        drop every command leased alongside it. Those commands then expired
        together, were re-claimed together with the poison command, and failed
        again -- with no verdict ever reaching the workflow that owns it.

        A response that is not even shaped like a claim is a different failure
        and still raises: claiming nothing would be indistinguishable from an
        empty queue.
        """

        raw_commands = response.get("commands")
        if not isinstance(raw_commands, list):
            return list(RemoteCommandClaim.model_validate(response).commands)
        commands: list[RemoteActionCommand] = []
        for item in raw_commands:
            try:
                commands.append(RemoteActionCommand.model_validate(item))
            except ValidationError as exc:
                self._reject_unparseable_command(item, exc)
        return commands

    def _reject_unparseable_command(self, item: Any, exc: ValidationError) -> None:
        """Fail one command this executor cannot parse, on its own lease."""

        command_id = item.get("command_id") if isinstance(item, dict) else None
        lease_token = item.get("lease_token") if isinstance(item, dict) else None
        summary = _validation_summary(exc)
        if not isinstance(command_id, str) or not command_id:
            LOGGER.error(
                "regional claim returned a command this executor cannot even "
                "identify, so it cannot be failed either: %s",
                summary,
            )
            return
        LOGGER.error(
            "regional cluster executor cannot parse a claimed command; "
            "failing it instead of dropping its batch: command=%s: %s",
            command_id,
            summary,
        )
        if not isinstance(lease_token, str) or not lease_token:
            # Without the lease token the control plane must reject the
            # result, so the command is left to expire rather than posted.
            LOGGER.error(
                "the unparseable command carried no lease token, so it can "
                "only expire: command=%s",
                command_id,
            )
            return
        try:
            self._post(
                "/v1/regional/executors/" + quote(command_id, safe="") + "/result",
                RemoteCommandResult(
                    lease_token=lease_token,
                    status=RemoteCommandStatus.FAILED,
                    status_source="executor-rejected",
                    error=summary,
                ).model_dump(mode="json"),
            )
        except Exception:
            # The command keeps its lease and expires; the next claim tries
            # again. Never let this fail the rest of the batch -- the whole
            # point of this method is that the siblings' leases were already
            # committed by the control plane.
            LOGGER.exception(
                "could not report the unparseable command as failed: command=%s",
                command_id,
            )

    def readiness(
        self,
        executor_id: str,
        *,
        execution_owners: list[str],
        last_successful_claim_age_seconds: float | None,
    ) -> dict:
        """Ask the control plane whether this executor is useful.

        Any non-2xx becomes ClusterExecutorError, which is what the
        readiness probe needs: a wrong token (403), an unknown cluster
        (403), a broken trust chain (URLError) and a backlog this
        executor cannot claim (503) all fail the same way, instead of the
        anonymous /healthz probe that passed through all four.
        """

        return self._post(
            "/v1/regional/executors/readiness",
            RegionalExecutorReadinessRequest(
                executor_id=executor_id,
                executor_protocol_version=(CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION),
                executor_artifact_sha256=self.executor_artifact_sha256,
                executor_compatibility_digest=(self.executor_compatibility_digest),
                execution_owners=execution_owners,
                last_successful_claim_age_seconds=(last_successful_claim_age_seconds),
            ).model_dump(mode="json"),
        )

    def complete(
        self,
        command: RemoteActionCommand,
        result: RemoteCommandResult,
    ) -> RemoteActionCommand:
        response = self._post(
            f"/v1/regional/executors/{command.command_id}/result",
            result.model_dump(mode="json"),
        )
        return RemoteActionCommand.model_validate(response)

    def renew(
        self,
        command: RemoteActionCommand,
        executor_id: str,
        lease_seconds: int,
    ) -> RemoteActionCommand:
        if not command.lease_token:
            raise ClusterExecutorError("cannot renew a command without a lease token")
        response = self._post(
            f"/v1/regional/executors/{command.command_id}/renew",
            RemoteCommandLeaseRenewal(
                executor_id=executor_id,
                lease_token=command.lease_token,
                lease_seconds=lease_seconds,
            ).model_dump(mode="json"),
        )
        return RemoteActionCommand.model_validate(response)


class RegionalFleetRegistry:
    """Fleet registry proxy backed by the regional control-plane API."""

    def __init__(self, client: RegionalExecutorClient, *, now=None) -> None:
        self.client = client
        self.store = self
        # HyperPodLifecycleStepAdapter reads registry.now() for the
        # post-reboot and post-replacement stabilization windows. The
        # local FleetRegistry exposes it, so the regional proxy must
        # too, otherwise every stabilization check raises
        # AttributeError and the executor reports the step FAILED.
        self.now = now or (lambda: datetime.now(timezone.utc))

    def fleet_rollout_fence_deployments(self, cluster_id: str) -> list[str]:
        """Ask the control plane whether a rollout fences destructive work.

        ``store = self`` above lets shared code call store-shaped methods on
        this proxy, and the fleet rollout fence used to read
        ``store.list_active_fleet_deployments``, which this proxy does not
        have. Because that fence fails closed, the ``AttributeError`` held
        every destructive remote command on every regional cluster
        indefinitely -- observed on 2026-09-04, when a ``STOP_WORKLOADS``
        command was re-claimed and re-held for ten minutes until the workload
        it was meant to stop had finished on its own.

        Answering it here instead of adding a store method keeps the
        supersession rule on the control plane and keeps other clusters'
        deployment records off this cluster's wire.
        """

        if cluster_id != self.client.cluster_id:
            raise ClusterExecutorError(
                "cannot read the fleet rollout fence for another cluster"
            )
        response = self.client._get("/v1/regional/executors/fleet-rollout-fence")
        return list(
            RemoteFleetRolloutFence.model_validate(response).fencing_deployment_ids
        )

    def list_agents(self, cluster_id: str | None = None) -> list[AgentRecord]:
        requested_cluster = cluster_id or self.client.cluster_id
        if requested_cluster != self.client.cluster_id:
            raise ClusterExecutorError("cannot read agents for another cluster")
        response = self.client._get(
            "/v1/fleet/agents?" + urlencode({"cluster_id": requested_cluster})
        )
        return [AgentRecord.model_validate(item) for item in (response or [])]

    def get_agent(self, cluster_id: str, node_id: str) -> AgentRecord:
        if cluster_id != self.client.cluster_id:
            raise ClusterExecutorError("cannot read an agent from another cluster")
        try:
            response = self.client._get(
                "/v1/fleet/agents/"
                + quote(cluster_id, safe="")
                + "/"
                + quote(node_id, safe="")
            )
        except ClusterExecutorError as exc:
            # By status, not by message text: any error body that happened to
            # quote "(404)" -- an upstream's own status, a proxy's detail
            # string -- used to read as "this node has no agent record", which
            # is a permanent verdict built out of a transient failure.
            if exc.status_code == 404:
                raise KeyError((cluster_id, node_id)) from exc
            raise
        return AgentRecord.model_validate(response)

    def endpoint(self, cluster_id: str, node_id: str) -> tuple[str, int]:
        """Resolve a live agent's endpoint from fleet heartbeats.

        Mirrors FleetRegistry.endpoint: readiness already applies the
        whole agent policy, so a ready node is safe to address.
        """
        report = self.readiness(cluster_id, [node_id])
        node = report.nodes[0]
        if not node.ready or node.endpoint is None:
            raise ValueError(
                f"node {node_id} is not fleet-ready: " + ", ".join(node.reasons)
            )
        return node.endpoint, node.generation or 0

    def maintenance_endpoint(
        self,
        cluster_id: str,
        node_id: str,
        expected_generation: int,
    ) -> str:
        """Resolve a quiesced agent without requiring a fresh heartbeat.

        A quiesced agent stops heartbeating, so readiness() would reject
        it. FleetRegistry.maintenance_endpoint solves this by calling
        _policy_reasons(check_liveness=False), but that policy lives on
        the control plane and is not exposed as a route. The checks that
        do not depend on the local policy object are replicated here:
        generation fencing and ACTIVE lifecycle. Liveness is
        deliberately not checked -- that is the point of this method.

        Skipping the version-compatibility half of the policy is safe
        because this method is only reachable after a SUCCEEDED
        QUIESCE_GPU_SERVICES step in the same workflow, which went
        through readiness() and therefore the full control-plane policy.
        The caller pins the generation recorded at quiesce time, so a
        match proves this is still that same agent process rather than a
        replacement at a recycled address.
        """
        record = self.get_agent(cluster_id, node_id)
        if record.generation != expected_generation:
            raise ValueError(
                f"agent generation changed from "
                f"{expected_generation} to {record.generation}"
            )
        if record.lifecycle_state is not AgentLifecycleState.ACTIVE:
            raise ValueError(f"agent lifecycle state is {record.lifecycle_state.value}")
        if not record.endpoint:
            raise ValueError(f"agent for {node_id} has no endpoint")
        return record.endpoint

    def readiness(self, cluster_id: str, node_ids: list[str]) -> FleetReadinessReport:
        response = self.client._post(
            "/v1/fleet/readiness",
            FleetReadinessRequest(cluster_id=cluster_id, node_ids=node_ids).model_dump(
                mode="json"
            ),
        )
        return FleetReadinessReport.model_validate(response)

    def drain_agent(
        self,
        cluster_id: str,
        node_id: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        return self._transition(cluster_id, node_id, "drain", request)

    def revoke_agent(
        self,
        cluster_id: str,
        node_id: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        return self._transition(cluster_id, node_id, "revoke", request)

    def _transition(
        self,
        cluster_id: str,
        node_id: str,
        action: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        if cluster_id != self.client.cluster_id:
            raise ClusterExecutorError("cannot transition an agent in another cluster")
        response = self.client._post(
            f"/v1/regional/executors/agents/{node_id}/{action}",
            request.model_dump(mode="json"),
        )
        return AgentRecord.model_validate(response)

    def spare_health_reasons(
        self,
        *,
        cluster_id: str,
        node_aliases: list[str],
        incident_id: str | None,
        observed_after,
    ) -> list[str]:
        response = self.client._post(
            "/v1/regional/executors/spares/health",
            RemoteSpareHealthRequest(
                cluster_id=cluster_id,
                node_aliases=node_aliases,
                incident_id=incident_id,
                observed_after=observed_after,
            ).model_dump(mode="json"),
        )
        return RemoteSpareHealthReport.model_validate(response).reasons

    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification:
        """Satisfy hyperpod_spares.NotificationSink over the API.

        HyperPodSpareCoordinator probes its store for this method and
        silently downgrades to a log line when it is absent. Since the
        regional executor passes this proxy as its store, without this
        method a warm-spare capacity shortage never pages anyone: the
        fault node stays isolated and the workflow stays blocked with
        no operator-visible signal anywhere in the control plane.
        """

        response = self.client._post(
            "/v1/regional/executors/advisory-notifications",
            RemoteAdvisoryNotificationRequest(
                cluster_id=self.client.cluster_id,
                notification=notification,
            ).model_dump(mode="json"),
        )
        return AdvisoryNotification.model_validate(response)

    def capture_evidence(
        self,
        request: RemoteEvidenceCaptureRequest,
    ) -> RawEvidenceRecord:
        if request.cluster_id != self.client.cluster_id:
            raise ClusterExecutorError("cannot persist evidence for another cluster")
        response = self.client._post(
            "/v1/regional/executors/evidence",
            request.model_dump(mode="json"),
        )
        return RawEvidenceRecord.model_validate(response)


class RegionalHyperPodSubmissionStore:
    """Control-plane-backed idempotency store for provider submissions.

    HyperPodLifecycleAdapter only needs three methods to make reboot and
    replace idempotent across restarts. In the default regional
    deployment the executor has no database (REMOTE_STATE=true), and the
    Pod that submits a reboot is routinely the Pod that reboot restarts,
    so the record has to live on the CPU control plane.
    """

    def __init__(self, client: RegionalExecutorClient) -> None:
        self.client = client

    def reserve_hyperpod_submission(self, record):
        response = self.client._post(
            "/v1/regional/executors/hyperpod-submissions/reserve",
            RemoteHyperPodSubmissionRequest(
                cluster_id=self.client.cluster_id,
                record=record,
            ).model_dump(mode="json"),
        )
        reservation = RemoteHyperPodSubmissionReservation.model_validate(response)
        return reservation.record, reservation.reserved

    def save_hyperpod_submission(self, record) -> None:
        self.client._post(
            "/v1/regional/executors/hyperpod-submissions/outcome",
            RemoteHyperPodSubmissionRequest(
                cluster_id=self.client.cluster_id,
                record=record,
            ).model_dump(mode="json"),
        )

    def get_hyperpod_submission(self, cluster_name: str, idempotency_key: str):
        # A real read, not a reserve: reserving would write a record for
        # a key that has none yet, and its placeholder identity would
        # then make the caller's own submission look like a conflicting
        # reuse of the key.
        response = self.client._get(
            "/v1/regional/executors/hyperpod-submissions?"
            + urlencode(
                {
                    "cluster_name": cluster_name,
                    "idempotency_key": idempotency_key,
                }
            )
        )
        if not response:
            raise KeyError((cluster_name, idempotency_key))
        return HyperPodSubmissionRecord.model_validate(response)


class RegionalIncidentOwnershipProvider:
    """Answers node/workload takeover questions over the control-plane API.

    KubernetesWorkflowAdapter refuses to isolate a node (or mutate a
    workload) that still carries another incident's ownership
    annotations, unless that incident's workflow already reached a
    terminal status. That check used to require a local store, and the
    regional executor has none (REMOTE_STATE=true), so the takeover
    branch was dead code: a node whose workflow died before
    RESTORE_SCHEDULING kept its annotations and every later incident on
    that node failed MARK_UNSCHEDULABLE with "node is already isolated by
    another incident/token" until an operator deleted the annotations by
    hand.
    """

    def __init__(self, client: RegionalExecutorClient) -> None:
        self.client = client

    def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
        response = self.client._get(
            "/v1/regional/executors/incident-ownership?"
            + urlencode({"incident_id": incident_id})
        )
        if not response:
            return RemoteIncidentOwnershipReport(incident_id=incident_id)
        return RemoteIncidentOwnershipReport.model_validate(response)

    def incident_workflow_is_terminal(self, incident_id: str) -> bool:
        report = self.incident_ownership(incident_id)
        return report.known and report.terminal


class CommandLeaseWatch:
    """This executor's local view of one claimed command's lease.

    Updated by the renewal thread, read by the executing thread (through the
    node-action lease guard) and by ``_execute_and_report`` before it posts.
    ``hold_reason()`` is non-None once the executor can no longer vouch for
    the command: renewal failed ``failure_limit`` times in a row, the lease
    window passed without a renewal landing, or the control plane asked for
    cancellation. The first two mean another executor may already own the
    command; the third means nothing new should start.
    """

    def __init__(
        self,
        *,
        lease_seconds: int,
        failure_limit: int,
        clock: Callable[[], float],
    ) -> None:
        self.lease_seconds = lease_seconds
        self.failure_limit = failure_limit
        self.clock = clock
        self._lock = Lock()
        self.expires_at = clock() + lease_seconds
        self.consecutive_failures = 0
        self.lost_reason: str | None = None
        self.cancellation_reason: str | None = None

    def renewed(self, response: Any) -> str | None:
        """Record a successful renewal; return the cancellation reason if any."""

        with self._lock:
            self.consecutive_failures = 0
            self.expires_at = self.clock() + self.lease_seconds
            requested_at = getattr(response, "cancellation_requested_at", None)
            if requested_at is not None and self.cancellation_reason is None:
                self.cancellation_reason = (
                    getattr(response, "cancellation_reason", None)
                    or "cancellation requested by the control plane"
                )
                return self.cancellation_reason
        return None

    def renewal_failed(self, error: BaseException) -> bool:
        """Count one failed renewal; True when the lease is now treated as lost."""

        with self._lock:
            self.consecutive_failures += 1
            if (
                self.lost_reason is None
                and self.consecutive_failures >= self.failure_limit
            ):
                self.lost_reason = (
                    f"lease renewal failed {self.consecutive_failures} time(s) "
                    f"in a row: {type(error).__name__}: {error}"
                )
                return True
        return False

    def lost(self) -> bool:
        return self.lost_reason is not None or self.clock() >= self.expires_at

    def hold_reason(self) -> str | None:
        with self._lock:
            if self.lost_reason is not None:
                return self.lost_reason
            if self.clock() >= self.expires_at:
                return (
                    f"lease expired locally after {self.lease_seconds}s "
                    "without a successful renewal"
                )
            return self.cancellation_reason


# ARCH-A4b: the regional executor has no store, so a stale warm-spare
# reservation can only be judged by its timestamp. One day mirrors the
# control-plane controller's default; five minutes between sweeps is far
# below the TTL and costs one node list per sweep.
SPARE_RESERVATION_TTL_SECONDS = 86400.0
SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS = 300.0


class SpareReservationSweep:
    """Reclaim warm-spare reservations whose owner can no longer be asked.

    The control plane's ``HyperPodSpareHealthController`` reads the owning
    workflow from its store; the regional executor is storeless, so
    ``SpareReservationReclaimer`` runs here with ``store=None`` and only the
    ``reserved-at`` TTL decides. A reservation without that annotation is kept
    (no evidence of staleness), and a spare running GPU pods is never touched.
    """

    def __init__(
        self,
        coordinator: HyperPodSpareCoordinator,
        *,
        ttl_seconds: float = SPARE_RESERVATION_TTL_SECONDS,
        interval_seconds: float = SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        from gpu_fault.spare_health import SpareReservationReclaimer

        self.coordinator = coordinator
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.reclaimer = SpareReservationReclaimer(
            coordinator,
            None,
            now=now or (lambda: datetime.now(timezone.utc)),
            ttl_seconds=ttl_seconds,
        )
        self.reclaimed_total = 0
        self._next_due: float | None = None

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    def run(self) -> list[str]:
        """Sweep every spare-labelled node once; returns the nodes released."""

        self._next_due = self.clock() + self.interval_seconds
        released: list[str] = []
        for node in self.coordinator.lifecycle.list_nodes(enrich=True):
            if (
                node.kubernetes_labels.get(self.coordinator.spare_label)
                != self.coordinator.spare_label_value
            ):
                continue
            node_name = self.coordinator._kubernetes_node_name(node)
            if node_name is None:
                continue
            kubernetes_node = self.coordinator.core.read_node(node_name)
            reservation = self.coordinator._annotation(kubernetes_node)
            if not reservation:
                continue
            reason = self.reclaimer.reason(node_name, kubernetes_node, reservation)
            if reason is None:
                continue
            self.coordinator.release([node_name], reservation)
            self.reclaimed_total += 1
            released.append(node_name)
            LOGGER.warning(
                "reclaimed stale spare reservation: node=%s incident=%s reason=%s",
                node_name,
                reservation,
                reason,
            )
        return released


class ClusterActionExecutor:
    def __init__(
        self,
        client: RegionalExecutorClient,
        adapters: list,
        *,
        executor_id: str,
        allowed_namespaces: set[str],
        poll_seconds: float = 2,
        lease_seconds: int = 120,
        batch_size: int = 5,
        max_concurrent_commands: int = 5,
        confirm_cluster_name: str | None = None,
        claim_backoff_max_seconds: float = 60,
        claim_state_path: str | None = None,
        lease_renewal_failure_limit: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        spare_reservation_sweep: SpareReservationSweep | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ClusterExecutorError("cluster executor poll seconds must be positive")
        if not 10 <= lease_seconds <= 7200:
            raise ClusterExecutorError(
                "cluster executor lease seconds must be between 10 and 7200"
            )
        if not 1 <= batch_size <= 25:
            raise ClusterExecutorError(
                "cluster executor batch size must be between 1 and 25"
            )
        if not 1 <= max_concurrent_commands <= 25:
            raise ClusterExecutorError(
                "cluster executor concurrency must be between 1 and 25"
            )
        if claim_backoff_max_seconds < poll_seconds:
            raise ClusterExecutorError(
                "claim backoff max must not be less than poll seconds"
            )
        if not 1 <= lease_renewal_failure_limit <= 100:
            raise ClusterExecutorError(
                "cluster executor lease renewal failure limit must be between 1 and 100"
            )
        self.lease_renewal_failure_limit = lease_renewal_failure_limit
        self.clock = clock
        # Only the report backoff sleeps inside a command's own thread, so it
        # is injectable: a test must not really wait half a second to prove the
        # retry happened.
        self.sleep = sleep
        self.spare_reservation_sweep = spare_reservation_sweep
        self.client = client
        self.adapters = adapters
        self.executor_id = executor_id
        self.allowed_namespaces = allowed_namespaces
        # Independent of anything the command carries: read from this
        # executor's own environment so a command for another HyperPod
        # cluster fails the adapter's confirmation gate.
        self.confirm_cluster_name = confirm_cluster_name
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.batch_size = batch_size
        self.max_concurrent_commands = max_concurrent_commands
        self.claim_backoff_max_seconds = claim_backoff_max_seconds
        self.execution_owners = sorted(
            {
                owner
                for adapter in adapters
                if (owner := getattr(adapter, "owner", None))
            }
        )
        if len(self.execution_owners) != len(adapters):
            raise ClusterExecutorError(
                "every local adapter must declare a unique owner"
            )
        # Liveness and observability counters. A regional executor can be
        # Ready and claiming nothing at all, so operators need a signal
        # that is tied to actual work rather than to process startup.
        #
        # Every one of them is written from more than one thread -- the claim
        # loop, one worker per claimed command, one lease renewer per command
        # -- and ``x += 1`` is a read and a write with a bytecode boundary in
        # between, so concurrent increments were silently lost and the
        # breadcrumb under-reported exactly when the executor was busiest.
        # ``_increment`` is the only way they change.
        self._counter_lock = Lock()
        self.claimed_total = 0
        self.reported_failures = 0
        self.unexpected_failures = 0
        self.lease_renewal_failures = 0
        # Commands whose lease this executor stopped trusting (renewals
        # exhausted or the window passed), results it therefore did not
        # post, cancellations seen in a renew response, and multi-node
        # barrier steps held because no coordinator is wired.
        self.lease_lost_total = 0
        self.results_withheld_total = 0
        self.cancellations_observed_total = 0
        self.barrier_unavailable_holds_total = 0
        # Adapter exceptions that say nothing about the step (Kubernetes
        # 409/429/5xx, urllib3 timeouts) reported WAITING instead of FAILED
        # (ARCH-B1 reached from the regional topology).
        self.retryable_adapter_errors_total = 0
        self.last_successful_claim_at: datetime | None = None
        # Whether the last claim cycle moved any command off WAITING. run()
        # takes the idle path when it did not, so a held command polls at
        # poll_seconds instead of as fast as the control plane will answer.
        self.last_cycle_advanced = True
        # The readiness probe runs in a separate process (kubectl exec),
        # so the claim timestamp has to leave this one. Written to the
        # container filesystem, not the store: the default regional
        # executor has no store of its own.
        self.claim_state_path = claim_state_path or os.getenv(
            "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
            "/tmp/executor-claim-state.json",
        )
        self.fleet_registry = next(
            (
                registry
                for adapter in adapters
                if (registry := getattr(adapter, "registry", None)) is not None
            ),
            None,
        )

    def _increment(self, counter: str, amount: int = 1) -> None:
        """Add to one shared counter under the counter lock."""

        with self._counter_lock:
            setattr(self, counter, getattr(self, counter) + amount)

    def metrics_snapshot(self) -> dict[str, Any]:
        """Every executor counter, for the claim breadcrumb and operators.

        The executor Pod has no /metrics listener of its own; the readiness
        probe already reads the claim-state breadcrumb out of process, so the
        counters ride along in that file (``kubectl exec ... cat``) until a
        scrape endpoint exists. Read under the counter lock so the breadcrumb
        cannot catch a half-applied increment.
        """

        with self._counter_lock:
            return self._counters()

    def _counters(self) -> dict[str, Any]:
        return {
            "claimed_total": self.claimed_total,
            "reported_failures": self.reported_failures,
            "unexpected_failures": self.unexpected_failures,
            "lease_renewal_failures": self.lease_renewal_failures,
            "lease_lost_total": self.lease_lost_total,
            "results_withheld_total": self.results_withheld_total,
            "cancellations_observed_total": (self.cancellations_observed_total),
            "barrier_unavailable_holds_total": (self.barrier_unavailable_holds_total),
            "retryable_adapter_errors_total": self.retryable_adapter_errors_total,
            "spare_reservations_reclaimed_total": (
                self.spare_reservation_sweep.reclaimed_total
                if self.spare_reservation_sweep is not None
                else 0
            ),
            "last_successful_claim_at": (
                self.last_successful_claim_at.isoformat()
                if self.last_successful_claim_at is not None
                else None
            ),
        }

    def _record_successful_claim(self, claimed_at: datetime) -> None:
        try:
            path = self.claim_state_path
            temporary = f"{path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "executor_id": self.executor_id,
                        "execution_owners": self.execution_owners,
                        "last_successful_claim_at": (claimed_at.isoformat()),
                        "counters": self.metrics_snapshot(),
                    },
                    handle,
                )
            os.replace(temporary, path)
        except OSError:
            # Never fail a claim cycle over the readiness breadcrumb.
            # The probe treats a missing or stale file as not-ready,
            # which is the correct direction to fail in.
            LOGGER.warning(
                "could not write executor claim state to %s",
                self.claim_state_path,
                exc_info=True,
            )

    def run_once(self) -> int:
        try:
            commands = self.client.claim(
                self.executor_id,
                execution_owners=self.execution_owners,
                max_commands=self.batch_size,
                lease_seconds=self.lease_seconds,
            )
        except Exception as exc:
            # Tag the phase for run()'s log. Everything after this point either
            # handles its own failures or is an executor defect, and both used
            # to be reported to the operator as "claim failed".
            raise ClusterExecutorClaimError(
                f"{type(exc).__name__}: {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc
        # A successful claim round-trip proves the token, the TLS trust
        # chain and the control-plane route all work, even when the
        # queue is empty. That is the only useful executor liveness
        # signal; the readiness marker file only proves pip install ran.
        self.last_successful_claim_at = datetime.now(timezone.utc)
        self._increment("claimed_total", len(commands))
        self._record_successful_claim(self.last_successful_claim_at)
        self.last_cycle_advanced = True
        if commands:
            with ThreadPoolExecutor(
                max_workers=min(
                    self.max_concurrent_commands,
                    len(commands),
                ),
                thread_name_prefix="gpu-fault-command",
            ) as pool:
                futures = [
                    pool.submit(self._execute_and_report, command)
                    for command in commands
                ]
                statuses = [future.result() for future in futures]
            # A command that reports WAITING is re-claimable at once, so a
            # batch that only waited puts run() straight back into claim()
            # with nothing changed: on 2026-09-04 a single held
            # STOP_WORKLOADS drove 25 claim/execute/complete round trips a
            # second across two replicas, and 759 identical log lines a
            # minute. Waiting is not progress, so it takes the idle path.
            self.last_cycle_advanced = any(
                status is not RemoteCommandStatus.WAITING for status in statuses
            )
        return len(commands)

    def _execute_and_report(self, command: RemoteActionCommand) -> RemoteCommandStatus:
        """One command, start to reported: never raises into ``run_once``.

        Anything that leaves this method comes back out of ``future.result()``
        in ``run_once``, which loses every *sibling* command's report and sends
        ``run()`` into claim backoff while this command sits LEASED until it
        expires and is executed again. ``_execute`` answers its own failures
        with a result, so whatever arrives here -- a claimed command with no
        lease token, a thread that could not be started, a defect in the
        withhold or report path -- is an executor-side defect, counted as one
        and held WAITING for the next lease holder.
        """

        try:
            return self._execute_under_lease(command)
        except Exception:
            self._increment("unexpected_failures")
            LOGGER.exception(
                "regional cluster executor raised outside command execution: "
                "command=%s cluster=%s",
                command.command_id,
                command.cluster_id,
            )
            return RemoteCommandStatus.WAITING

    def _execute_under_lease(self, command: RemoteActionCommand) -> RemoteCommandStatus:
        stop = Event()
        watch = CommandLeaseWatch(
            lease_seconds=self.lease_seconds,
            failure_limit=self.lease_renewal_failure_limit,
            clock=self.clock,
        )
        renewer = Thread(
            target=self._renew_lease,
            args=(command, stop, watch),
            name=f"lease-{command.command_id[:24]}",
            daemon=True,
        )
        renewer.start()
        # The node-action adapter reads this before every send, so a lost
        # lease stops new node actions without widening the adapter API.
        guard_token = active_lease_guard.set(watch.hold_reason)
        try:
            result = self._execute(command)
            if watch.lost():
                # Another executor may hold this command by now. The agent
                # ledger keeps whatever ran; the next lease holder polls it
                # by command_id. Posting here would race that holder's result
                # under a lease this process no longer owns.
                if watch.lost_reason is None:
                    # Expired on the local clock without the renewer ever
                    # declaring it lost; count it here, once.
                    self._increment("lease_lost_total")
                self._increment("results_withheld_total")
                LOGGER.warning(
                    "regional cluster executor withheld a result under a lost "
                    "lease: command=%s cluster=%s operation=%s status=%s "
                    "reason=%s",
                    command.command_id,
                    command.cluster_id,
                    command.step.operation.value,
                    result.status.value,
                    watch.hold_reason(),
                )
                return RemoteCommandStatus.WAITING
            if not self._report_result(command, result, watch):
                # The action ran but its verdict never landed. WAITING keeps
                # this cycle off the fast path (the command is still open on
                # the control plane) instead of claiming that it advanced.
                return RemoteCommandStatus.WAITING
            return result.status
        finally:
            active_lease_guard.reset(guard_token)
            stop.set()
            renewer.join(timeout=2)

    def _report_result(
        self,
        command: RemoteActionCommand,
        result: RemoteCommandResult,
        watch: CommandLeaseWatch,
    ) -> bool:
        """Post one result, retrying a transport failure under a live lease.

        The action has already happened on the cluster, so this post is the
        only place its verdict exists. Dropping it on the first ``URLError``
        left the command LEASED until expiry and then re-executed by whichever
        replica re-claimed it -- a second RESET_GPU or STOP_WORKLOADS for one
        workflow step. Retrying is safe because ``complete_remote_command`` is
        idempotent: a command that is already terminal is returned unchanged.

        The retry is bounded three ways, because a report that keeps failing
        must not become a spin: at most ``_RESULT_REPORT_ATTEMPTS`` attempts,
        only for failures with no verdict in them (no HTTP status, or one of
        the transient statuses), and only while the lease is still ours -- past
        that, another executor may already own the command and this result
        would race its result, which is exactly what the withhold rule above
        exists to prevent.
        """

        for attempt in range(1, _RESULT_REPORT_ATTEMPTS + 1):
            try:
                self.client.complete(command, result)
                return True
            except Exception as exc:
                hold_reason = watch.hold_reason()
                last_attempt = attempt >= _RESULT_REPORT_ATTEMPTS
                if (
                    not _retryable_report_failure(exc)
                    or last_attempt
                    or hold_reason is not None
                ):
                    # A rejected result (stale lease, stale fencing token,
                    # already terminal) and an unreachable control plane both
                    # end here, and neither may discard the results of the
                    # remaining commands in this batch.
                    LOGGER.exception(
                        "regional cluster executor could not report result: "
                        "command=%s cluster=%s operation=%s status=%s "
                        "attempts=%d lease_hold=%s",
                        command.command_id,
                        command.cluster_id,
                        command.step.operation.value,
                        result.status.value,
                        attempt,
                        hold_reason,
                    )
                    self._increment("reported_failures")
                    return False
                delay = _report_backoff_seconds(attempt)
                LOGGER.warning(
                    "regional cluster executor could not report result; "
                    "retrying in %.2fs: command=%s cluster=%s operation=%s "
                    "status=%s attempt=%d: %s: %s",
                    delay,
                    command.command_id,
                    command.cluster_id,
                    command.step.operation.value,
                    result.status.value,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                self.sleep(delay)
                # The backoff is where a lease actually runs out: the local
                # deadline can be crossed and the renewer can hit its failure
                # limit while this thread sleeps. Re-check before re-posting,
                # or the retry sends a result under a lease another replica may
                # already own -- the very race the withhold rule prevents. This
                # is a withheld result, not a reporting failure: nothing was
                # refused, and the next lease holder will redo the step.
                hold_reason = watch.hold_reason()
                if hold_reason is not None:
                    self._increment("results_withheld_total")
                    LOGGER.warning(
                        "regional cluster executor withheld an unreported "
                        "result: the lease lapsed during the report backoff: "
                        "command=%s cluster=%s operation=%s status=%s "
                        "attempts=%d reason=%s",
                        command.command_id,
                        command.cluster_id,
                        command.step.operation.value,
                        result.status.value,
                        attempt,
                        hold_reason,
                    )
                    return False
        # Unreachable: the final attempt always takes the terminal branch
        # above, which is where ``reported_failures`` is counted. Fail loudly
        # rather than returning an uncounted False if that ever changes -- the
        # ``_execute_and_report`` boundary turns this into WAITING.
        raise AssertionError(
            "the result report loop must terminate inside its final attempt"
        )

    def _renew_lease(
        self,
        command: RemoteActionCommand,
        stop: Event,
        watch: CommandLeaseWatch | None = None,
    ) -> None:
        # ``lease_seconds`` is validated to 10..7200 at construction, so a third
        # of it is never below 3.3s; the only clamp that can bind is the 30s
        # ceiling that keeps a long lease from going unrenewed for minutes.
        interval = min(30.0, self.lease_seconds / 3)
        while not stop.wait(interval):
            try:
                renewed = self.client.renew(
                    command,
                    self.executor_id,
                    self.lease_seconds,
                )
            except Exception as exc:
                self._increment("lease_renewal_failures")
                LOGGER.exception(
                    "regional command lease renewal failed: command=%s cluster=%s",
                    command.command_id,
                    command.cluster_id,
                )
                if watch is not None and watch.renewal_failed(exc):
                    self._increment("lease_lost_total")
                    LOGGER.error(
                        "regional command lease treated as lost; no further "
                        "node actions will start and the result will not be "
                        "posted: command=%s cluster=%s reason=%s",
                        command.command_id,
                        command.cluster_id,
                        watch.lost_reason,
                    )
                    return
                continue
            if watch is None:
                continue
            cancellation = watch.renewed(renewed)
            if cancellation is not None:
                self._increment("cancellations_observed_total")
                LOGGER.warning(
                    "regional command cancellation requested during execution; "
                    "no further node actions will start: command=%s cluster=%s "
                    "reason=%s",
                    command.command_id,
                    command.cluster_id,
                    cancellation,
                )

    def sweep_spare_reservations(self) -> None:
        """Run the periodic spare sweep when due; never raises (ARCH-A4b)."""

        sweep = self.spare_reservation_sweep
        if sweep is None or not sweep.due():
            return
        try:
            sweep.run()
        except Exception:  # noqa: BLE001 - housekeeping must not stop claims
            LOGGER.warning("spare reservation sweep failed", exc_info=True)

    def run(self) -> None:
        consecutive_failures = 0
        while True:
            try:
                count = self.run_once()
                consecutive_failures = 0
                self.sweep_spare_reservations()
            except Exception as exc:
                consecutive_failures += 1
                delay = min(
                    self.claim_backoff_max_seconds,
                    self.poll_seconds * (2 ** min(consecutive_failures - 1, 8)),
                )
                # A result that could not be posted no longer reaches here at
                # all, but a defect after the claim still can, and calling it
                # "claim failed" sent the operator to the wrong side of the
                # wire (F11).
                phase = (
                    "claim failed"
                    if isinstance(exc, ClusterExecutorClaimError)
                    else "cycle failed after a successful claim"
                )
                if (
                    consecutive_failures == 1
                    or consecutive_failures & (consecutive_failures - 1) == 0
                ):
                    LOGGER.exception(
                        "regional cluster executor %s; "
                        "retrying in %.1fs (consecutive=%d)",
                        phase,
                        delay,
                        consecutive_failures,
                    )
                else:
                    LOGGER.warning(
                        "regional cluster executor %s, still failing: "
                        "%s: %s; retrying in %.1fs "
                        "(consecutive=%d)",
                        phase,
                        type(exc).__name__,
                        exc,
                        delay,
                        consecutive_failures,
                    )
                time.sleep(delay)
                continue
            if count == 0 or not self.last_cycle_advanced:
                time.sleep(self.poll_seconds)

    def _execute(self, command: RemoteActionCommand) -> RemoteCommandResult:
        lease_token = command.lease_token
        if not lease_token:
            raise ClusterExecutorError("claimed command has no lease token")
        try:
            self._validate(command)
            if self.fleet_registry is not None and command_requires_fleet_preflight(
                command.step.operation
            ):
                steps = (
                    command.workflow.safety_steps
                    if command.workflow.executes_safety_steps
                    else command.workflow.official_steps
                )
                preflight_error = fleet_preflight_reason(
                    self.fleet_registry,
                    command.workflow,
                    command.incident,
                    steps,
                )
                if preflight_error is not None:
                    LOGGER.warning(
                        "remote command held before destructive action: "
                        "command=%s workflow=%s operation=%s reason=%s",
                        command.command_id,
                        command.workflow_request_id,
                        command.step.operation.value,
                        preflight_error,
                    )
                    return RemoteCommandResult(
                        lease_token=lease_token,
                        status=RemoteCommandStatus.WAITING,
                        details={
                            "fleet_preflight_blocked": True,
                            "reason": preflight_error,
                        },
                    )
            matches = [
                adapter for adapter in self.adapters if adapter.supports(command.step)
            ]
            if len(matches) != 1:
                raise ClusterExecutorError(
                    "remote command requires exactly one local adapter; "
                    f"found {len(matches)}"
                )
            barrier_hold = self._barrier_hold(command, matches[0], lease_token)
            if barrier_hold is not None:
                return barrier_hold
            workflow = command.workflow
            if command.result_details:
                previous = WorkflowStepExecution(
                    step_index=command.step_index,
                    operation=command.step.operation,
                    status=WorkflowStepStatus.WAITING,
                    phase=execution_phase(workflow),
                    adapter_operation_id=(f"remote/{command.command_id}"),
                    details=command.result_details,
                )
                workflow = workflow.model_copy(
                    update={
                        "step_executions": [
                            *workflow.step_executions,
                            previous,
                        ]
                    }
                )
            outcome = matches[0].execute(
                WorkflowStepContext(
                    workflow=workflow,
                    incident=command.incident,
                    step=command.step,
                    step_index=command.step_index,
                    request=self._execution_request(command),
                    idempotency_key=command.idempotency_key,
                )
            )
            status = {
                WorkflowStepStatus.WAITING: (RemoteCommandStatus.WAITING),
                WorkflowStepStatus.SUCCEEDED: (RemoteCommandStatus.SUCCEEDED),
                WorkflowStepStatus.FAILED: (RemoteCommandStatus.FAILED),
            }[outcome.status]
            details = dict(outcome.details or {})
            error = outcome.error
            if status is RemoteCommandStatus.FAILED and not error:
                # A FAILED outcome without a message is still the adapter's
                # verdict, not an executor defect. Left as None it failed the
                # result model's validation inside this try block and was
                # caught below as an executor-internal-error -- a stack trace,
                # an unexpected-failure count and an alert for a refusal the
                # adapter merely forgot to describe.
                error = (
                    f"{command.step.operation.value} adapter reported FAILED "
                    "without an error message"
                )
                details["error_message_missing"] = True
            return RemoteCommandResult(
                lease_token=lease_token,
                status=status,
                details=details,
                error=error,
            )
        except ClusterExecutorError as exc:
            retryable = self._retryable_control_plane_result(exc, command, lease_token)
            if retryable is not None:
                return retryable
            # A control-plane request that never got an answer now arrives here
            # as a ClusterExecutorError with no status code instead of a raw
            # URLError, so the transport classification has to be consulted on
            # this branch too. Without it, wrapping the transport error would
            # have turned every proxy timeout inside an adapter from a WAITING
            # hold into a FAILED step -- a healthy GPU declared unrecoverable
            # because a read timed out.
            retryable = self._retryable_result(exc, command, lease_token)
            if retryable is not None:
                return retryable
            # Rejections the executor raises on purpose: cluster
            # mismatch, stale fencing token, no or ambiguous adapter.
            # These are legitimate FAILED results, not executor bugs.
            LOGGER.warning(
                "regional cluster executor rejected command: "
                "command=%s cluster=%s operation=%s owner=%s nodes=%s: %s",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                command.step.execution_owner,
                ",".join(command.step.node_ids),
                exc,
            )
            return RemoteCommandResult(
                lease_token=lease_token,
                status=RemoteCommandStatus.FAILED,
                status_source="executor-rejected",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            retryable = self._retryable_result(exc, command, lease_token)
            if retryable is not None:
                return retryable
            configuration_reason = aws_configuration_error(exc)
            if configuration_reason is not None:
                # A missing IRSA annotation, an unassumable role or a
                # denied API call is a deployment gap, not a defect: no
                # stack trace, no internal-error count (which alerting
                # watches), and a reason that names the knob. Counting
                # these as executor bugs is what once made "the
                # ServiceAccount has no role-arn" look like an adapter
                # crash for the operator reading the step details.
                LOGGER.error(
                    "regional cluster executor is misconfigured for "
                    "AWS: command=%s cluster=%s operation=%s nodes=%s: "
                    "%s (%s)",
                    command.command_id,
                    command.cluster_id,
                    command.step.operation.value,
                    ",".join(command.step.node_ids),
                    configuration_reason,
                    type(exc).__name__,
                )
                return RemoteCommandResult(
                    lease_token=lease_token,
                    status=RemoteCommandStatus.FAILED,
                    status_source="executor-configuration-error",
                    error=configuration_reason,
                    details={
                        "configuration_error": True,
                        "executor_id": self.executor_id,
                        "exception_type": type(exc).__name__,
                    },
                )
            # Anything else is an executor-side defect (a missing
            # attribute, a bad adapter wiring, an unhandled provider
            # error). Reporting it as a bare FAILED string hides the
            # difference between "the action was refused" and "the
            # executor is broken", so record a stack trace, tag the
            # result, and count it for alerting.
            self._increment("unexpected_failures")
            LOGGER.exception(
                "regional cluster executor raised while executing: "
                "command=%s cluster=%s operation=%s owner=%s nodes=%s",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                command.step.execution_owner,
                ",".join(command.step.node_ids),
            )
            return RemoteCommandResult(
                lease_token=lease_token,
                status=RemoteCommandStatus.FAILED,
                status_source="executor-internal-error",
                error=f"{type(exc).__name__}: {exc}",
                details={
                    "executor_internal_error": True,
                    "executor_id": self.executor_id,
                    "exception_type": type(exc).__name__,
                },
            )

    def _retryable_result(
        self,
        exc: BaseException,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        """WAITING for an exception that says nothing about the step, else None.

        Transport failures keep their established shape. An adapter error
        ARCH-B1 classifies as retryable (a Kubernetes 409/429/5xx, a urllib3
        timeout raised inside the adapter) gets the same treatment from the
        regional topology: the command stays WAITING and is re-claimed,
        bounded by the control plane's per-step waiting cap exactly like a
        transport retry (ARCH-E2E-1 finding 1).
        """

        retryable = retryable_transport_result(
            exc,
            lease_token=lease_token,
            executor_id=self.executor_id,
        )
        if retryable is not None:
            LOGGER.warning(
                "regional cluster executor transport failed; "
                "command remains retryable: command=%s cluster=%s "
                "operation=%s nodes=%s: %s",
                command.command_id,
                command.cluster_id,
                command.step.operation.value,
                ",".join(command.step.node_ids),
                retryable.details["reason"],
            )
            return retryable
        if not retryable_adapter_error(exc):
            return None
        self._increment("retryable_adapter_errors_total")
        LOGGER.warning(
            "regional cluster executor adapter raised a retryable "
            "error; command remains retryable: command=%s cluster=%s "
            "operation=%s nodes=%s: %s: %s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            type(exc).__name__,
            exc,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-retryable-adapter-error",
            details={
                "retryable_adapter_error": True,
                "reason": "RETRYABLE_ADAPTER_ERROR",
                "executor_id": self.executor_id,
                "exception_type": type(exc).__name__,
                "adapter_error": str(exc)[:200],
            },
        )

    def _barrier_hold(
        self,
        command: RemoteActionCommand,
        adapter: Any,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        """Hold a multi-node barrier step this topology cannot coordinate.

        ``BarrierCoordinator`` persists barrier state through the control-plane
        store, which the regional executor does not have (REMOTE_STATE=true)
        and which the control plane exposes read-only over the API. The
        node-action adapter is therefore built with ``barriers=None`` here,
        and ``_execute_multi_node_reset`` would answer with a bare FAILED that
        reads like the reset itself failed. Refuse at the claim boundary
        instead, with a reason and a counter.
        """

        if not (
            command.step.operation in MULTI_NODE_BARRIER_OPERATIONS
            and len(command.step.node_ids) > 1
        ):
            return None
        if getattr(adapter, "barriers", None) is not None:
            return None
        self._increment("barrier_unavailable_holds_total")
        reason = (
            f"{command.step.operation.value} across {len(command.step.node_ids)} "
            "nodes needs a multi-node barrier coordinator, and this regional "
            "executor has none (barrier state lives in the control-plane "
            "store); split the step per node or run it from a topology with "
            "a barrier coordinator"
        )
        LOGGER.warning(
            "remote command held: multi-node barrier unavailable: command=%s "
            "cluster=%s operation=%s nodes=%s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-barrier-unavailable",
            details={
                "multi_node_barrier_unavailable": True,
                "operation": command.step.operation.value,
                "node_ids": list(command.step.node_ids),
                "reason": reason,
                "executor_id": self.executor_id,
            },
        )

    def _retryable_control_plane_result(
        self,
        exc: ClusterExecutorError,
        command: RemoteActionCommand,
        lease_token: str,
    ) -> RemoteCommandResult | None:
        if exc.status_code not in {408, 425, 429} and (
            exc.status_code is None or exc.status_code < 500
        ):
            return None
        LOGGER.warning(
            "regional cluster executor control-plane request failed "
            "transiently; command remains retryable: command=%s "
            "cluster=%s operation=%s nodes=%s status=%s",
            command.command_id,
            command.cluster_id,
            command.step.operation.value,
            ",".join(command.step.node_ids),
            exc.status_code,
        )
        return RemoteCommandResult(
            lease_token=lease_token,
            status=RemoteCommandStatus.WAITING,
            status_source="executor-retryable-control-plane",
            details={
                "retryable_control_plane_error": True,
                "status_code": exc.status_code,
                "reason": str(exc),
                "executor_id": self.executor_id,
                "exception_type": type(exc).__name__,
            },
        )

    def _validate(self, command: RemoteActionCommand) -> None:
        if command.cluster_id != self.client.cluster_id:
            raise ClusterExecutorError(
                "command cluster does not match executor cluster"
            )
        if (
            command.fencing_token != command.workflow.fencing_token
            or command.incident.fencing_token != command.fencing_token
        ):
            raise ClusterExecutorError(
                "command fencing token does not match workflow/incident"
            )
        for workload_id in command.step.workload_ids:
            namespace = workload_id.split("/", 1)[0]
            if not self.allowed_namespaces:
                raise ClusterExecutorError(
                    "executor has no allowed workload namespaces"
                )
            if namespace not in self.allowed_namespaces:
                raise ClusterExecutorError(
                    f"workload namespace is not allowed: {namespace}"
                )

    def _execution_request(self, command: RemoteActionCommand):
        from gpu_fault.models import (
            WorkflowExecutionRequest,
        )

        # confirm_cluster_name is the HyperPod adapter's second, human-
        # intent confirmation: it must come from the executor's own
        # configuration, not from the command being executed. Copying
        # command.cluster_id into it made the command confirm itself, so
        # the check could never fail and added no safety at all.
        return WorkflowExecutionRequest(
            expected_fencing_token=command.fencing_token,
            confirm_cluster_name=self.confirm_cluster_name,
            restart_authorization=command.restart_authorization,
        )


def _regional_client_from_environment(
    *, timeout_seconds: float = 15
) -> RegionalExecutorClient:
    return RegionalExecutorClient(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
        os.environ["GPU_FAULT_CLUSTER_ID"],
        os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        timeout_seconds=timeout_seconds,
        ca_file=os.getenv("GPU_FAULT_CONTROL_PLANE_CA_FILE") or None,
        executor_artifact_sha256=(
            os.getenv("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256") or None
        ),
        executor_compatibility_digest=(
            os.getenv("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST") or None
        ),
    )


def executor_from_environment() -> ClusterActionExecutor:
    cluster_id = os.environ["GPU_FAULT_CLUSTER_ID"]
    executor_id = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_ID",
        f"{cluster_id}/{socket.gethostname()}",
    )
    use_remote_state = env_bool("GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE", True)
    store = None if use_remote_state else _persistent_store_from_environment()
    regional_client = _regional_client_from_environment()
    fleet_registry = RegionalFleetRegistry(regional_client)
    spare_coordinator: HyperPodSpareCoordinator | None = None
    kubernetes_adapter = KubernetesWorkflowAdapter(
        owner=os.getenv(
            "GPU_FAULT_KUBERNETES_OWNER",
            "gpu-fault-kubernetes-adapter",
        ),
        store=store,
        notification_sink=(fleet_registry if store is None else store),
        evidence_sink=(fleet_registry if store is None else None),
        workload_log_tail_lines=int(
            os.getenv("GPU_FAULT_WORKLOAD_LOG_TAIL_LINES", "2000")
        ),
        workload_log_max_bytes=int(
            os.getenv("GPU_FAULT_WORKLOAD_LOG_MAX_BYTES", "262144")
        ),
        workload_log_s3_uri=(os.getenv("GPU_FAULT_WORKLOAD_LOG_S3_URI") or None),
        workload_log_s3_max_bytes=int(
            os.getenv(
                "GPU_FAULT_WORKLOAD_LOG_S3_MAX_BYTES",
                "104857600",
            )
        ),
        # With REMOTE_STATE the adapter has no store, so "has the
        # incident that owns this node already finished?" has to be
        # asked over the API. Without it the takeover branch never
        # fires and a node annotated by a dead workflow is permanently
        # unusable.
        ownership_provider=(
            RegionalIncidentOwnershipProvider(regional_client)
            if store is None
            else None
        ),
    )
    adapters = [kubernetes_adapter]
    hyperpod_confirm_cluster = None
    # Node agent addressing must come from fleet heartbeats, not from a
    # hand-maintained GPU_FAULT_NODE_AGENT_ENDPOINTS map: nodes are
    # replaced over a cluster's life, and a stale map fails closed only
    # at the moment a node action is dispatched, with no prior signal.
    # The control-plane wiring already passes a registry (api.py:615).
    node_action_adapter = None
    if env_bool("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER"):
        node_action_adapter = NodeActionWorkflowAdapter.from_environment(
            registry=fleet_registry,
        )
        adapters.append(node_action_adapter)
    if env_bool("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"):
        hyperpod_confirm_cluster = os.getenv(
            "GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", ""
        ).strip()
        if not hyperpod_confirm_cluster:
            raise ClusterExecutorError(
                "GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER is required "
                "when the HyperPod adapter is enabled"
            )
        # Fail at startup, not at the first real fault. Enabling the
        # adapter without the ServiceAccount's role-arn annotation used
        # to look completely healthy -- Pod Ready, logs clean -- until a
        # GPU broke hours later and REPLACE_NODE died on
        # NoCredentialsError. This costs one local credential-chain
        # resolution and no API call.
        credential_gap = missing_aws_credentials()
        if credential_gap is not None:
            raise ClusterExecutorError(
                "the HyperPod adapter is enabled but "
                + credential_gap
                + "; annotate serviceaccount "
                "gpu-fault-cluster-executor with "
                "eks.amazonaws.com/role-arn=<role> and restart, or set "
                "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER=false"
            )
        lifecycle = HyperPodLifecycleAdapter(
            HyperPodAdapterConfig.from_environment(),
            # Reboot and replace restart this very Pod, so submission
            # idempotency cannot live in its memory. With REMOTE_STATE
            # the record is held by the control plane; with a local
            # store it is held there.
            store=(
                RegionalHyperPodSubmissionStore(regional_client)
                if store is None
                else store
            ),
        )
        # Reboot confirmation always needs the fleet registry to compare
        # the pre-submit boot/incarnation baseline with the new Agent
        # heartbeat. Spare failover controls only replacement
        # allocation; it must not disable the only automatic RESTART_NODE
        # confirmation path.
        registry = fleet_registry
        if registry is None:
            raise ClusterExecutorError(
                "HyperPod lifecycle execution requires a fleet "
                "registry for reboot confirmation"
            )
        if env_bool("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"):
            if not use_remote_state:
                raise ClusterExecutorError(
                    "regional HyperPod spare failover requires "
                    "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true"
                )
            spare_coordinator = HyperPodSpareCoordinator(
                lifecycle,
                registry.store,
                kubernetes_adapter.core,
                registry=registry,
                remote_health_provider=registry,
                spare_label=os.getenv(
                    "GPU_FAULT_HYPERPOD_SPARE_LABEL",
                    "gpu-fault.io/spare",
                ),
                spare_label_value=os.getenv(
                    "GPU_FAULT_HYPERPOD_SPARE_LABEL_VALUE",
                    "true",
                ),
            )
        adapters.append(
            HyperPodLifecycleStepAdapter(
                lifecycle,
                owner=os.getenv(
                    "GPU_FAULT_HYPERPOD_OWNER",
                    "gpu-fault-hyperpod-adapter",
                ),
                registry=registry,
                spare_coordinator=spare_coordinator,
                node_action_adapter=node_action_adapter,
                kubernetes_adapter=kubernetes_adapter,
                store=store,
                notification_sink=(fleet_registry if store is None else store),
                post_reboot_stabilization_seconds=int(
                    os.getenv(
                        "GPU_FAULT_HYPERPOD_POST_REBOOT_STABILIZATION_SECONDS",
                        "60",
                    )
                ),
            )
        )
    namespaces = {
        value.strip()
        for value in os.getenv("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", "").split(",")
        if value.strip()
    }
    sweep = SpareReservationSweep(spare_coordinator) if spare_coordinator else None
    return ClusterActionExecutor(
        regional_client,
        adapters,
        executor_id=executor_id,
        allowed_namespaces=namespaces,
        spare_reservation_sweep=sweep,
        poll_seconds=float(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS",
                "2",
            )
        ),
        lease_seconds=int(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS",
                "120",
            )
        ),
        batch_size=int(os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_BATCH_SIZE", "5")),
        max_concurrent_commands=int(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_MAX_CONCURRENT_COMMANDS",
                "5",
            )
        ),
        # Only set when this executor actually owns HyperPod mutations.
        # Left None otherwise so an unexpectedly routed RESTART_NODE or
        # REPLACE_NODE fails the adapter's confirmation gate instead of
        # confirming itself.
        confirm_cluster_name=(hyperpod_confirm_cluster),
        lease_renewal_failure_limit=int(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_LEASE_FAILURE_LIMIT", "3")
        ),
    )


def readiness_probe() -> int:
    """Authenticated readiness for the executor Pod.

    Replaces ``urlopen(CONTROL_PLANE_URL + "/healthz")``, which was
    anonymous and therefore could not fail for any executor-specific
    reason: a wrong per-cluster token, a revoked registration, a broken
    trust chain to the control plane, or a backlog whose execution owners
    this executor does not implement all left the Pod Ready and claiming
    nothing. Runs as a separate process under the probe, so the last
    successful claim is read from the breadcrumb the claim loop writes.
    """

    configure_logging()
    client = _regional_client_from_environment(
        timeout_seconds=float(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_READINESS_TIMEOUT_SECONDS",
                "8",
            )
        ),
    )
    state_path = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        "/tmp/executor-claim-state.json",
    )
    executor_id = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_ID",
        f"{os.environ['GPU_FAULT_CLUSTER_ID']}/{socket.gethostname()}",
    )
    owners: list[str] = []
    claim_age: float | None = None
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        # No claim has completed yet (or the file is unreadable). The
        # control plane still checks registration, the token and the
        # backlog; it just cannot judge claim freshness.
        state = {}
    if isinstance(state, dict):
        executor_id = state.get("executor_id") or executor_id
        raw_owners = state.get("execution_owners")
        if isinstance(raw_owners, list):
            owners = [owner for owner in raw_owners if isinstance(owner, str)]
        claimed_at = state.get("last_successful_claim_at")
        if isinstance(claimed_at, str):
            try:
                claim_age = max(
                    0.0,
                    (
                        datetime.now(timezone.utc) - datetime.fromisoformat(claimed_at)
                    ).total_seconds(),
                )
            except ValueError:
                claim_age = None
    try:
        report = client.readiness(
            executor_id,
            execution_owners=owners,
            last_successful_claim_age_seconds=claim_age,
        )
    except Exception as exc:
        LOGGER.error("executor readiness probe failed: %s", exc)
        return 1
    if not report.get("ready"):
        LOGGER.error(
            "executor is not ready: %s",
            "; ".join(report.get("reasons") or ["unspecified"]),
        )
        return 1
    return 0


def main() -> None:
    configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-cluster-executor")
    executor_from_environment().run()
