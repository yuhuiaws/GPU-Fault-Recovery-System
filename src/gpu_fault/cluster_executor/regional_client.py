"""The regional executor's wire to the control plane, and the proxies on it.

``RegionalExecutorClient`` is the only HTTP client the data-plane executor
owns: claim, renew, complete, readiness, and the fleet/spare/evidence routes
the proxies below call through ``_get``/``_post``. Every failure it raises is a
``ClusterExecutorError`` carrying the HTTP status (or ``None`` for no answer at
all), which is what every caller reads as "verdict" versus "retry".

The three ``Regional*`` proxies give the storeless executor the store-shaped
objects the shared adapters expect (fleet registry, HyperPod submission store,
incident ownership) by answering over the same client.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
from datetime import datetime, timezone
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

from pydantic import ValidationError

from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentRecord,
    AgentTransitionRequest,
    FleetReadinessReport,
    FleetReadinessRequest,
)
from gpu_fault.hyperpod import HyperPodSubmissionRecord
from gpu_fault.models import AdvisoryNotification
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

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")


class ClusterExecutorError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
