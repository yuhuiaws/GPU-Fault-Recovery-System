"""Live audit of the regional remote-command protocol (GF-REGIONAL-CMD-001..016).

Runs inside an API Pod against ``http://127.0.0.1:8080`` and the live store.
Every command it seeds carries a test-only ``execution_owner`` unless the case
asserts something about a real owner (CMD-003 default owner, CMD-004 owner
filter), and every command it leases it hands back as ``WAITING`` before the
case ends, so nothing it touches is left leased against the production queue.

The audit refuses to start unless the target cluster has zero open remote
commands and the production executor cannot claim: either the operator asserts
``--isolated-cluster`` or reports ``--executor-ready-replicas 0`` (read from the
executor Deployment outside the Pod, where kubectl exists). Each case is judged
on its own -- a failing case is recorded ``FAIL`` with its error and the next
case still runs -- and, given ``--run-dir``, each case writes
``cases/<id>/<id>.json`` so the CMD cases can carry evidence and satisfy the
predecessor chain.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from gpu_fault.app import ApplicationContext
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowStepContext,
)
from gpu_fault.execution.restart_budget_preflight import reserve_restart_budgets
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RegionalRemoteWorkflowAdapter, RemoteActionCommand
from gpu_fault.store import NotFoundError

try:
    ROOT: Path | None = Path(__file__).resolve().parents[3]
except NameError:  # piped into the Pod as ``python3 -``; no checkout there
    ROOT = None
if ROOT is not None and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDITED_CASE_IDS = tuple(
    [
        *(f"GF-REGIONAL-CMD-{number:03d}" for number in range(1, 11)),
        *(f"GF-REGIONAL-CMD-{number:03d}" for number in range(12, 17)),
    ]
)
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
NODE_AGENT_OWNER = "gpu-fault-node-agent"
HYPERPOD_OWNER = "gpu-fault-hyperpod-adapter"
REMOTE_COMMAND_ID = re.compile(r"^remote-[0-9a-f]{24}$")


class ProtocolAuditError(RuntimeError):
    """A case assertion that did not hold, or a preflight that refused to run."""


def expect(condition: object, message: str) -> None:
    """Fail the current case with ``message`` unless ``condition`` holds.

    A module-level function rather than a method so the case bodies stay
    callable against a bare namespace (the unit tests drive ``run_006`` that
    way), and an explicit exception rather than ``assert`` so ``python -O``
    cannot turn the audit into a run that passes everything.
    """

    if not condition:
        raise ProtocolAuditError(message)


class LiveProtocolAudit:
    def __init__(
        self,
        *,
        cluster_id: str,
        other_cluster_id: str,
        executor_sha256: str,
        executor_digest: str,
        run_dir: Path | None = None,
        isolated_cluster: bool = False,
        executor_ready_replicas: int | None = None,
        release_id: str = "",
    ) -> None:
        self.cluster_id = cluster_id
        self.other_cluster_id = other_cluster_id
        self.executor_sha256 = executor_sha256
        self.executor_digest = executor_digest
        self.run_dir = run_dir
        self.isolated_cluster = isolated_cluster
        self.executor_ready_replicas = executor_ready_replicas
        self.release_id = release_id
        self.run_id = f"cmd-audit-{uuid4().hex[:12]}"
        # A step owner no deployed executor advertises, so a seeded command can
        # only ever be claimed by this audit's own claim calls.
        self.test_owner = f"{self.run_id}-owner"
        self.context = ApplicationContext.from_environment()
        self.store = self.context.store
        raw_registry = json.loads(os.environ["GPU_FAULT_REGIONAL_CLUSTERS_JSON"])
        self.registry = {item["cluster_id"]: item for item in raw_registry}
        self.tokens = {
            cluster_id: str(item["token"]) for cluster_id, item in self.registry.items()
        }
        self.created_commands: set[str] = set()
        self.created_workflows: set[str] = set()
        self.created_incidents: set[str] = set()
        self.results: dict[str, Any] = {}
        self.preflight_result: dict[str, Any] = {}

    def close(self) -> None:
        for command_id in self.created_commands:
            self.store._delete("remote_command", command_id)
        for workflow_id in self.created_workflows:
            self.store._delete("workflow", workflow_id)
        for incident_id in self.created_incidents:
            self.store._delete("incident", incident_id)

    def record(self, case_id: str, **details: Any) -> None:
        self.results[case_id] = details

    # ------------------------------------------------------------------ #
    # isolation preflight
    # ------------------------------------------------------------------ #
    def open_commands_in_cluster(self) -> int:
        stats = self.store.remote_command_stats()
        open_by_cluster = stats.get("open_by_cluster") or {}
        return int(open_by_cluster.get(self.cluster_id, 0))

    def preflight(self) -> dict[str, Any]:
        """Refuse to run against a queue a production executor could still drain.

        The claim route hands out real leases: a claim the audit makes before
        seeding would take a production command for the lease window, and a
        seeded command with a real adapter owner would be taken by a production
        executor. Both need the target cluster's queue empty *and* the executor
        unable to claim -- asserted by the operator (``--isolated-cluster``) or
        observed as zero Ready replicas of the executor Deployment.
        """

        open_commands = self.open_commands_in_cluster()
        errors = []
        if open_commands:
            errors.append(
                f"target cluster has {open_commands} open remote command(s); "
                "the audit leases and seeds against this queue"
            )
        if not self.isolated_cluster and self.executor_ready_replicas != 0:
            errors.append(
                "production executor may still claim: pass --isolated-cluster or "
                "--executor-ready-replicas 0 after scaling the executor "
                "Deployment down (observed "
                f"{self.executor_ready_replicas!r} ready replicas)"
            )
        result = {
            "open_remote_commands": open_commands,
            "isolated_cluster": self.isolated_cluster,
            "executor_ready_replicas": self.executor_ready_replicas,
            "test_owner": self.test_owner,
            "errors": errors,
        }
        self.preflight_result = result
        if errors:
            raise ProtocolAuditError("preflight refused: " + "; ".join(errors))
        return result

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        *,
        cluster_id: str | None,
        token: str | None,
        payload: Any = None,
    ) -> tuple[int, Any]:
        headers: dict[str, str] = {}
        if cluster_id is not None:
            headers["X-GPU-Fault-Cluster-ID"] = cluster_id
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            "http://127.0.0.1:8080" + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return response.status, json.loads(raw or b"null")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw or b"null")

    def claim(
        self,
        *,
        cluster_id: str | None = None,
        token: str | None = None,
        executor_id: str | None = None,
        owners: list[str] | None = None,
        max_commands: Any = 1,
        lease_seconds: Any = 60,
    ) -> tuple[int, Any]:
        target = cluster_id or self.cluster_id
        return self._request(
            "POST",
            "/v1/regional/executors/claim",
            cluster_id=target,
            token=token or self.tokens[target],
            payload={
                "executor_id": executor_id or f"{self.run_id}-executor",
                "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
                "executor_artifact_sha256": self.executor_sha256,
                "executor_compatibility_digest": self.executor_digest,
                "execution_owners": ([self.test_owner] if owners is None else owners),
                "max_commands": max_commands,
                "lease_seconds": lease_seconds,
            },
        )

    def complete(
        self,
        command_id: str,
        *,
        cluster_id: str | None = None,
        token: str | None = None,
        payload: dict[str, Any],
    ) -> tuple[int, Any]:
        target = cluster_id or self.cluster_id
        return self._request(
            "POST",
            f"/v1/regional/executors/{command_id}/result",
            cluster_id=target,
            token=token or self.tokens[target],
            payload=payload,
        )

    def hand_back(self, commands: list[dict[str, Any]]) -> int:
        """Return every leased command as WAITING so no lease outlives its case."""

        returned = 0
        for command in commands:
            status, _ = self.complete(
                str(command["command_id"]),
                payload={
                    "lease_token": self._lease_token(command),
                    "status": "WAITING",
                    "details": {"audit": self.run_id, "returned": True},
                },
            )
            expect(status == 200, f"could not hand back {command['command_id']}")
            returned += 1
        return returned

    # ------------------------------------------------------------------ #
    # seeding
    # ------------------------------------------------------------------ #
    def _pair(
        self,
        suffix: str,
        *,
        owner: str,
        cluster_id: str,
        fencing_token: int,
        step: WorkflowStepSpec | None = None,
        not_before: datetime | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, WorkflowStepSpec]:
        incident_id = f"{self.run_id}-{suffix}-incident"
        workflow_id = f"{self.run_id}-{suffix}-workflow"
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=f"{self.run_id}-{suffix}-event",
            event_type="CMD_PROTOCOL_PROBE",
            cluster_id=cluster_id,
            node_ids=[f"{self.run_id}-nonexistent-node"],
            policy_version="cmd-probe",
            policy_source="cmd-probe",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=workflow_id,
            fencing_token=fencing_token,
        )
        selected = step or WorkflowStepSpec(
            operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
            execution_owner=owner,
            node_ids=[f"{self.run_id}-nonexistent-node"],
        )
        workflow = WorkflowRequest(
            request_id=workflow_id,
            incident_id=incident_id,
            runtime_profile_version="cmd-probe",
            status=WorkflowStatus.RUNNING,
            fencing_token=fencing_token,
            not_before=not_before,
            official_steps=[selected],
        )
        return incident, workflow, selected

    def persist_pair(
        self,
        suffix: str,
        *,
        owner: str,
        step: WorkflowStepSpec | None = None,
    ) -> WorkflowRequest:
        """Persist an incident/workflow pair the production dispatcher cannot take.

        ``not_before`` an hour out keeps the dispatcher's eligibility filter off
        the record for the life of the audit; only this process executes it.
        """

        incident, workflow, _ = self._pair(
            suffix,
            owner=owner,
            cluster_id=self.cluster_id,
            fencing_token=1,
            step=step,
            not_before=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        self.store.save_incident_and_workflow(incident, workflow)
        self.created_incidents.add(incident.incident_id)
        self.created_workflows.add(workflow.request_id)
        return workflow

    def seed(
        self,
        suffix: str,
        *,
        owner: str | None = None,
        cluster_id: str | None = None,
        created_at: datetime | None = None,
        fencing_token: int = 1,
        persist_workflow: bool = False,
    ) -> RemoteActionCommand:
        target = cluster_id or self.cluster_id
        incident, workflow, step = self._pair(
            suffix,
            owner=owner or self.test_owner,
            cluster_id=target,
            fencing_token=fencing_token,
            not_before=(
                datetime.now(timezone.utc) + timedelta(hours=1)
                if persist_workflow
                else None
            ),
        )
        if persist_workflow:
            self.store.save_incident_and_workflow(incident, workflow)
            self.created_incidents.add(incident.incident_id)
            self.created_workflows.add(workflow.request_id)
        command = RemoteActionCommand(
            command_id=f"{self.run_id}-{suffix}-command",
            cluster_id=target,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=fencing_token,
            idempotency_key=f"{self.run_id}/{suffix}",
            step=step,
            workflow=workflow,
            incident=incident,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=created_at or datetime.now(timezone.utc),
        )
        stored = self.store.ensure_remote_command(command)
        self.created_commands.add(stored.command_id)
        return stored

    @staticmethod
    def _lease_token(command: dict[str, Any]) -> str:
        token = command.get("lease_token")
        expect(isinstance(token, str) and token, "claimed command has no lease token")
        return str(token)

    def _node_agent_record_absent(self) -> bool:
        try:
            self.store.get_agent(self.cluster_id, f"{self.run_id}-nonexistent-node")
        except NotFoundError:
            return True
        return False

    # ------------------------------------------------------------------ #
    # cases
    # ------------------------------------------------------------------ #
    def run_001(self) -> None:
        # A small test-owner backlog gives the batch bound something to bite
        # on; every command a 200 hands out goes straight back as WAITING.
        seeded = [self.seed(f"cmd001-{index}").command_id for index in range(3)]
        observed: dict[str, int] = {}
        returned = 0
        for value in (0, 1, 25, 26, -1, "5", 5.5):
            status, body = self.claim(max_commands=value)
            observed[repr(value)] = status
            if status == 200:
                commands = list(body["commands"])
                expect(
                    all(item["command_id"] in seeded for item in commands),
                    "claim returned a command this audit did not seed",
                )
                returned += self.hand_back(commands)
            if value in {1, 25}:
                expect(
                    status == 200 and len(body["commands"]) <= int(value),
                    f"max_commands={value!r} answered {status}",
                )
            elif value == "5":
                expect(
                    status == 200 and len(body["commands"]) <= 5,
                    f"max_commands='5' answered {status}",
                )
            else:
                expect(status == 422, f"max_commands={value!r} answered {status}")
        self.record(
            "GF-REGIONAL-CMD-001",
            statuses=observed,
            seeded=seeded,
            leased_and_returned=returned,
        )

    def run_002(self) -> None:
        invalid = {}
        for value in (9, 7201, 0):
            status, _ = self.claim(lease_seconds=value)
            invalid[str(value)] = status
            expect(status == 422, f"lease_seconds={value} answered {status}")
        command = self.seed("cmd002")
        leases = {}
        for value in (10, 600, 601, 7200):
            before = datetime.now(timezone.utc)
            status, body = self.claim(
                executor_id=f"{self.run_id}-lease-{value}",
                lease_seconds=value,
            )
            after = datetime.now(timezone.utc)
            expect(
                status == 200 and len(body["commands"]) == 1,
                f"lease_seconds={value} did not lease the seeded command",
            )
            claimed = body["commands"][0]
            expires = datetime.fromisoformat(
                claimed["lease_expires_at"].replace("Z", "+00:00")
            )
            lower = (expires - after).total_seconds()
            upper = (expires - before).total_seconds()
            expect(
                value - 3 <= lower <= value + 3 and value - 3 <= upper <= value + 3,
                f"lease_seconds={value} produced a deadline {lower:.1f}s out",
            )
            leases[str(value)] = round(lower, 3)
            status, _ = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(claimed),
                    "status": "WAITING",
                    "details": {"lease_seconds": value},
                },
            )
            expect(status == 200, "WAITING hand-back was refused")
        status, body = self.claim(lease_seconds=10)
        expect(
            status == 200 and len(body["commands"]) == 1,
            "10s lease did not lease the seeded command",
        )
        leased = body["commands"][0]
        time.sleep(11)
        expired_status, _ = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        expect(expired_status == 409, f"expired lease answered {expired_status}")
        self.record(
            "GF-REGIONAL-CMD-002",
            invalid=invalid,
            observed_lease_seconds=leases,
            expired_result_status=expired_status,
        )

    def run_003(self) -> None:
        values = [
            ([""], 422),
            ([" gpu-fault-kubernetes-adapter"], 422),
            (["gpu-fault-kubernetes-adapter "], 422),
            (["a", "a"], 422),
            ([f"owner-{index}" for index in range(33)], 422),
            ([f"owner-{index}" for index in range(32)], 200),
        ]
        observed = {}
        for owners, expected in values:
            status, body = self.claim(owners=owners)
            observed[str(len(owners)) + ":" + repr(owners[:2])] = status
            if status == 200:
                self.hand_back(list(body["commands"]))
            expect(status == expected, f"owners {owners[:2]!r} answered {status}")
        protected = self.seed("cmd003", owner=NODE_AGENT_OWNER)
        status, body = self.claim(owners=[])
        expect(
            status == 200 and body["commands"] == [],
            "an empty owner list leased a node-agent command",
        )
        # The other half of the default: an empty list means the kubernetes
        # adapter, so a kubernetes-adapter command IS returned.
        default_owned = self.seed("cmd003-default", owner=KUBERNETES_OWNER)
        status, body = self.claim(owners=[], max_commands=25)
        claimed_ids = [item["command_id"] for item in body["commands"]]
        self.hand_back(list(body["commands"]))
        expect(
            status == 200 and claimed_ids == [default_owned.command_id],
            f"an empty owner list leased {claimed_ids!r}, expected the "
            "kubernetes-adapter command alone",
        )
        self.record(
            "GF-REGIONAL-CMD-003",
            statuses=observed,
            empty_owner_claimed=[],
            protected_command=protected.command_id,
            empty_owner_default_claimed=claimed_ids,
        )

    def run_004(self) -> None:
        commands = {
            owner: self.seed(f"cmd004-{index}", owner=owner)
            for index, owner in enumerate(
                (KUBERNETES_OWNER, NODE_AGENT_OWNER, HYPERPOD_OWNER)
            )
        }
        status, body = self.claim(owners=[NODE_AGENT_OWNER], max_commands=5)
        first_owners = [item["step"]["execution_owner"] for item in body["commands"]]
        self.hand_back(list(body["commands"]))
        expect(
            status == 200 and first_owners == [NODE_AGENT_OWNER],
            f"node-agent claim leased {first_owners!r}",
        )
        status, remaining = self.claim(
            owners=[KUBERNETES_OWNER, HYPERPOD_OWNER],
            max_commands=5,
        )
        owners = [item["step"]["execution_owner"] for item in remaining["commands"]]
        self.hand_back(list(remaining["commands"]))
        expect(
            status == 200 and set(owners) == {KUBERNETES_OWNER, HYPERPOD_OWNER},
            f"adapter claim leased {owners!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-004",
            command_ids={key: value.command_id for key, value in commands.items()},
            claimed_owners=[NODE_AGENT_OWNER, *owners],
        )

    def run_005(self) -> None:
        now = datetime.now(timezone.utc)
        expected = []
        for index in range(5):
            command = self.seed(
                f"cmd005-{index}",
                created_at=now - timedelta(seconds=(5 - index) * 5),
            )
            expected.append(command.command_id)
        # Two commands created the same instant: the store breaks the tie on
        # command_id, so the one seeded second must still come out first.
        tie = now - timedelta(seconds=1)
        tie_b = self.seed("cmd005-tie-b", created_at=tie)
        tie_a = self.seed("cmd005-tie-a", created_at=tie)
        expected.extend(sorted([tie_b.command_id, tie_a.command_id]))
        status, body = self.claim(max_commands=25)
        actual = [item["command_id"] for item in body["commands"]]
        self.hand_back(list(body["commands"]))
        expect(
            status == 200 and actual == expected,
            f"claim order {actual!r} is not FIFO with an id tie-break {expected!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-005",
            expected=expected,
            actual=actual,
            tie_break_sample={
                "created_at": tie.isoformat(),
                "seeded_order": [tie_b.command_id, tie_a.command_id],
                "claimed_order": actual[-2:],
            },
        )

    def run_006(self) -> None:
        command = self.seed("cmd006")
        status, first = self.claim(executor_id="cmd006-a1", lease_seconds=10)
        one = first["commands"][0]
        token_one = self._lease_token(one)
        status, _ = self.complete(
            command.command_id,
            payload={
                "lease_token": token_one,
                "status": "WAITING",
                "details": {"round": 1},
            },
        )
        expect(status == 200, "first WAITING hand-back was refused")
        status, second = self.claim(executor_id="cmd006-a2", lease_seconds=10)
        two = second["commands"][0]
        token_two = self._lease_token(two)
        stale_one, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_one, "status": "SUCCEEDED"},
        )
        expect(stale_one == 409, f"stale token answered {stale_one}")
        time.sleep(15)
        stale_two, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_two, "status": "SUCCEEDED"},
        )
        expect(stale_two == 409, f"expired token answered {stale_two}")
        status, third = self.claim(executor_id="cmd006-a3", lease_seconds=60)
        three = third["commands"][0]
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(three),
                "status": "SUCCEEDED",
            },
        )
        expect(
            final_status == 200 and final["status"] == "SUCCEEDED",
            f"final result answered {final_status}",
        )
        expect(
            final["last_lease_owner"] == "cmd006-a3",
            f"final lease owner is {final.get('last_lease_owner')!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-006",
            stale_token_one=stale_one,
            expired_token_two=stale_two,
            final_owner=final["last_lease_owner"],
        )

    def run_007(self) -> None:
        command = self.seed("cmd007")
        status, claim = self.claim(executor_id="cmd007", lease_seconds=60)
        leased = claim["commands"][0]
        token = self._lease_token(leased)
        first_status, first = self.complete(
            command.command_id,
            payload={"lease_token": token, "status": "SUCCEEDED"},
        )
        expect(first_status == 200, f"terminal result answered {first_status}")
        variants: list[dict[str, Any]] = [
            {"lease_token": token, "status": "SUCCEEDED"},
            {"lease_token": "wrong-token", "status": "SUCCEEDED"},
            {
                "lease_token": "wrong-token",
                "status": "FAILED",
                "error": "must not overwrite",
            },
        ]
        replay_statuses = []
        for payload in variants:
            status, body = self.complete(command.command_id, payload=payload)
            replay_statuses.append(status)
            expect(
                status == 200 and body == first,
                f"replay {payload!r} answered {status} with a changed record",
            )
        self.record(
            "GF-REGIONAL-CMD-007",
            first_status=first_status,
            replay_statuses=replay_statuses,
            final=first,
        )

    def run_008(self) -> None:
        command = self.seed("cmd008")
        status, claim = self.claim(executor_id="cmd008", lease_seconds=60)
        leased = claim["commands"][0]
        token = self._lease_token(leased)
        payloads: dict[str, dict[str, Any]] = {
            "PENDING": {"lease_token": token, "status": "PENDING"},
            "LEASED": {"lease_token": token, "status": "LEASED"},
            "FAILED_NO_ERROR": {"lease_token": token, "status": "FAILED"},
            "FAILED_EMPTY_ERROR": {
                "lease_token": token,
                "status": "FAILED",
                "error": "",
            },
            "MISSING_TOKEN": {"status": "WAITING"},
            "EXTRA_FIELD": {
                "lease_token": token,
                "status": "WAITING",
                "foo": 1,
            },
        }
        observed = {}
        for name, payload in payloads.items():
            status, _ = self.complete(command.command_id, payload=payload)
            observed[name] = status
            expect(status == 422, f"{name} answered {status}")
            expect(
                self.store.get_remote_command(command.command_id).status.value
                == "LEASED",
                f"{name} changed the command status",
            )
        waiting, body = self.complete(
            command.command_id,
            payload={"lease_token": token, "status": "WAITING"},
        )
        expect(
            waiting == 200 and body["status"] == "WAITING",
            f"legal WAITING answered {waiting}",
        )
        self.record(
            "GF-REGIONAL-CMD-008",
            invalid_statuses=observed,
            waiting_status=waiting,
        )

    def run_009(self) -> None:
        missing_status, missing = self.complete(
            f"{self.run_id}-does-not-exist",
            payload={"lease_token": "missing", "status": "SUCCEEDED"},
        )
        other = self.seed("cmd009-other", cluster_id=self.other_cluster_id)
        cross_status, cross = self.complete(
            other.command_id,
            payload={"lease_token": "missing", "status": "SUCCEEDED"},
        )
        expect(
            missing_status == cross_status == 404,
            f"unknown answered {missing_status}, foreign answered {cross_status}",
        )
        expect(
            missing["detail"] == cross["detail"],
            "a foreign command is distinguishable from an unknown one: "
            f"{missing.get('detail')!r} != {cross.get('detail')!r}",
        )
        expect(
            self.store.get_remote_command(other.command_id).status.value == "PENDING",
            "the foreign command changed status",
        )
        self.record(
            "GF-REGIONAL-CMD-009",
            missing={"status": missing_status, "detail": missing["detail"]},
            cross_cluster={"status": cross_status, "detail": cross["detail"]},
        )

    def run_010(self) -> None:
        node_absent_before = self._node_agent_record_absent()
        command = self.seed("cmd010", persist_workflow=True)
        status, claim = self.claim(executor_id="cmd010", lease_seconds=60)
        expect(
            status == 200 and len(claim["commands"]) == 1,
            "the seeded command was not leased",
        )
        leased = claim["commands"][0]
        workflow = self.store.get_workflow(command.workflow_request_id)
        incident = self.store.get_incident(command.incident_id)
        workflow = workflow.model_copy(
            update={
                "fencing_token": 2,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        incident = incident.model_copy(
            update={
                "fencing_token": 2,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.store.save_incident_and_workflow(incident, workflow)
        stale_status, stale = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        # The full batch: FIFO must not be allowed to hide the fenced command
        # behind other backlog, so ask for everything and look for its id.
        status, reclaimed = self.claim(executor_id="cmd010-reclaim", max_commands=25)
        reclaimed_ids = [item["command_id"] for item in reclaimed["commands"]]
        self.hand_back(list(reclaimed["commands"]))
        expect(stale_status == 409, f"stale fencing result answered {stale_status}")
        expect(
            "fencing token is stale" in str(stale.get("detail")),
            f"stale detail is {stale.get('detail')!r}",
        )
        expect(
            command.command_id not in reclaimed_ids,
            "the fenced command was leased again",
        )
        after_workflow = self.store.get_workflow(command.workflow_request_id)
        after_incident = self.store.get_incident(command.incident_id)
        after_command = self.store.get_remote_command(command.command_id)
        expect(
            after_workflow.fencing_token == 2
            and after_workflow.status is WorkflowStatus.RUNNING
            and after_incident.fencing_token == 2,
            "the fenced workflow/incident changed under the stale result",
        )
        expect(
            after_command.status.value == "LEASED"
            and after_command.lease_owner == "cmd010"
            and after_command.fencing_token == 1,
            f"the fenced command moved: {after_command.status.value} "
            f"owner={after_command.lease_owner!r}",
        )
        node_absent_after = self._node_agent_record_absent()
        expect(
            node_absent_before and node_absent_after,
            "a node agent record appeared for the probe node",
        )
        self.record(
            "GF-REGIONAL-CMD-010",
            complete_status=stale_status,
            reclaimed=False,
            reclaim_batch_ids=reclaimed_ids,
            workflow_after={
                "status": after_workflow.status.value,
                "fencing_token": after_workflow.fencing_token,
                "not_before": (
                    after_workflow.not_before.isoformat()
                    if after_workflow.not_before
                    else None
                ),
            },
            command_after={
                "status": after_command.status.value,
                "lease_owner": after_command.lease_owner,
                "fencing_token": after_command.fencing_token,
            },
            node_agent_record_absent={
                "before": node_absent_before,
                "after": node_absent_after,
            },
        )

    def _adapter_context(
        self,
        suffix: str,
        *,
        fencing_token: int = 1,
        step: WorkflowStepSpec | None = None,
    ) -> WorkflowStepContext:
        incident, workflow, selected = self._pair(
            suffix,
            owner=NODE_AGENT_OWNER,
            cluster_id=self.cluster_id,
            fencing_token=fencing_token,
            step=step,
        )
        incident = incident.model_copy(update={"workflow_request_id": None})
        return WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=selected,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=fencing_token),
            idempotency_key=f"{self.run_id}/{suffix}",
        )

    def run_012(self) -> None:
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={NODE_AGENT_OWNER},
        )
        first_context = self._adapter_context("cmd012", fencing_token=1)
        first = adapter.execute(first_context)
        first_id = str(first.details["remote_command_id"])
        self.created_commands.add(first_id)
        created_at = self.store.get_remote_command(first_id).created_at
        retry_ids = []
        created_at_after_retries = []
        for _attempt in range(2):
            retry = adapter.execute(first_context)
            retry_ids.append(str(retry.details["remote_command_id"]))
            created_at_after_retries.append(
                self.store.get_remote_command(first_id).created_at
            )
        second_context = self._adapter_context("cmd012", fencing_token=2)
        second = adapter.execute(second_context)
        second_id = str(second.details["remote_command_id"])
        self.created_commands.add(second_id)
        expect(
            all(item == first_id for item in retry_ids),
            f"retries minted new command ids: {retry_ids!r}",
        )
        expect(
            all(item == created_at for item in created_at_after_retries),
            "a retry rewrote the persisted command's created_at",
        )
        expect(second_id != first_id, "a new fencing token reused the command id")
        expect(
            REMOTE_COMMAND_ID.fullmatch(first_id) is not None
            and REMOTE_COMMAND_ID.fullmatch(second_id) is not None,
            f"command ids are not remote-<24 hex>: {first_id!r}, {second_id!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-012",
            first_id=first_id,
            duplicate_ids=retry_ids,
            created_at_unchanged=True,
            next_fencing_id=second_id,
            limitation=(
                "identity reuse is exercised as three adapter.execute calls on "
                "one step context; the dispatcher's own retry path is "
                "NOT_EXERCISED here"
            ),
        )

    def run_013(self) -> None:
        command = self.seed("cmd013")
        previous: dict[str, int] = {}
        for round_number in range(1, 21):
            status, claim = self.claim(
                executor_id=f"cmd013-{round_number}",
                lease_seconds=60,
            )
            expect(
                status == 200 and len(claim["commands"]) == 1,
                f"round {round_number} did not lease the command",
            )
            leased = claim["commands"][0]
            if round_number > 1:
                expect(
                    leased["result_details"] == previous,
                    f"round {round_number} lost the previous WAITING details",
                )
            previous = {"probe_round": round_number}
            status, _ = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(leased),
                    "status": "WAITING",
                    "details": previous,
                },
            )
            expect(status == 200, f"round {round_number} WAITING answered {status}")
        status, claim = self.claim(executor_id="cmd013-final", lease_seconds=60)
        leased = claim["commands"][0]
        expect(
            leased["result_details"] == {"probe_round": 20},
            "the final claim lost the round-20 details",
        )
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        expect(
            final_status == 200 and final["status"] == "SUCCEEDED",
            f"final result answered {final_status}",
        )
        self.record(
            "GF-REGIONAL-CMD-013",
            waiting_rounds=20,
            final_status=final["status"],
            limitation="no maximum WAITING round count",
        )

    def run_014(self) -> None:
        """The delegating step's negative claim, read back from the store.

        The adapter's in-memory outcome is what the pytest proxy checks; the
        live case has to prove the executor *persisted* it, so the workflow is
        written (dispatcher-proof via ``not_before``) and executed here, and the
        step execution is read from the store.
        """

        workflow = self.persist_pair("cmd014", owner=NODE_AGENT_OWNER)
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={NODE_AGENT_OWNER},
        )
        executor = ProductionWorkflowExecutor(
            self.store,
            [adapter],
            ProductionExecutorConfig(
                enabled=True,
                executor_id=f"{self.run_id}-cmd014-executor",
                allowed_operations=frozenset(
                    {WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT}
                ),
            ),
        )
        result = executor.execute(
            workflow.request_id,
            WorkflowExecutionRequest(expected_fencing_token=1),
        )
        persisted = self.store.get_workflow(workflow.request_id)
        executions = [
            item for item in persisted.step_executions if item.step_index == 0
        ]
        expect(len(executions) == 1, "no persisted step execution for step 0")
        execution = executions[0]
        details = dict(execution.details)
        command_id = str(details.get("remote_command_id") or "")
        if command_id:
            self.created_commands.add(command_id)
        expect(
            execution.status.value == "WAITING",
            f"persisted step status is {execution.status.value}",
        )
        expected = {
            "remote_command_id": command_id,
            "remote_cluster_id": self.cluster_id,
            "remote_status": "PENDING",
            "mutation_submitted_by_control_plane": False,
        }
        expect(
            REMOTE_COMMAND_ID.fullmatch(command_id) is not None
            and all(details.get(key) == value for key, value in expected.items()),
            f"persisted step details {details!r} do not carry {expected!r}",
        )
        self.record(
            "GF-REGIONAL-CMD-014",
            workflow_status=result.status.value,
            persisted_step_status=execution.status.value,
            details=details,
        )

    def run_015(self) -> None:
        """Restart budget is decided once, at claim time; dispatch only reads it.

        Three refusals, none of which mints a remote command or touches a
        workload: an occupied budget refuses the claim preflight
        (``reserve_restart_budgets`` -> RESTART_BUDGET_EXHAUSTED); a dispatch
        that holds no reservation fails closed at the adapter
        (``issue_restart_authorization`` -> RESTART_RESERVATION_MISSING) instead
        of reserving there; and a step without ``source_gpu_count`` is refused
        at the gate before either site reads the budget row.
        """

        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={KUBERNETES_OWNER},
        )
        job_id = f"{self.run_id}-budget"
        priming_reservation = f"{self.run_id}-existing-reservation"
        try:
            state, reserved = self.store.reserve_job_restart(
                self.cluster_id,
                job_id,
                1,
                priming_reservation,
            )
            expect(
                reserved and state.restart_count == 1,
                "the priming restart reservation was refused",
            )
            full = WorkflowStepSpec(
                operation=WorkflowOperation.RESTART_WORKLOAD,
                execution_owner=KUBERNETES_OWNER,
                workload_ids=["training/job/nonexistent-cmd015"],
                parameters={
                    "cluster_id": self.cluster_id,
                    "job_id": job_id,
                    "source_attempt_id": f"{job_id}-a1",
                    "source_gpu_count": 1,
                    "restart_budget": 1,
                },
            )
            # (a) The claim preflight against the occupied budget: the only
            # site that reserves, and so the only site that can say exhausted.
            claim = self._adapter_context("cmd015-exhausted", step=full)
            failure = reserve_restart_budgets(
                self.store,
                claim.workflow,
                claim.incident,
                claim.workflow.official_steps,
            )
            expect(
                failure is not None,
                "the claim preflight admitted an exhausted budget",
            )
            assert failure is not None
            exhausted = failure.outcome
            expect(
                failure.step_index == 0
                and exhausted.status.value == "FAILED"
                and exhausted.details.get("reason") == "RESTART_BUDGET_EXHAUSTED"
                and "restart budget exhausted" in str(exhausted.error)
                and "1/1" in str(exhausted.error),
                f"exhausted budget answered {exhausted.status.value}: "
                f"{exhausted.error!r}",
            )
            state = self.store.get_restart_budget(self.cluster_id, job_id)
            expect(
                state.restart_count == 1
                and state.reservation_ids == [priming_reservation],
                f"the refused claim changed the budget row: {state.reservation_ids!r}",
            )
            # (b) Dispatch without a reservation: the adapter reads the
            # preflight's reservation back and fails closed when there is none.
            unreserved = adapter.execute(
                self._adapter_context("cmd015-unreserved", step=full)
            )
            expect(
                unreserved.status.value == "FAILED"
                and unreserved.details.get("reason") == "RESTART_RESERVATION_MISSING"
                and "restart reservation missing" in str(unreserved.error)
                and unreserved.details.get("restart_count") == 1
                and unreserved.details.get("restart_budget") == 1,
                f"unreserved dispatch answered {unreserved.status.value}: "
                f"{unreserved.error!r} {unreserved.details!r}",
            )
            # (c) The gate refuses an incomplete safety context before any
            # budget read, in the claim preflight's words.
            missing = full.model_copy(
                update={
                    "parameters": {
                        key: value
                        for key, value in full.parameters.items()
                        if key != "source_gpu_count"
                    }
                }
            )
            incomplete = adapter.execute(
                self._adapter_context("cmd015-missing", step=missing)
            )
            expect(
                incomplete.status.value == "FAILED"
                and incomplete.details.get("reason") == "RESTART_SAFETY_CONTEXT_MISSING"
                and "restart safety context is missing: source_gpu_count"
                in str(incomplete.error),
                f"missing context answered {incomplete.status.value}: "
                f"{incomplete.error!r}",
            )
            command_incidents = {
                f"{self.run_id}-cmd015-exhausted-incident",
                f"{self.run_id}-cmd015-unreserved-incident",
                f"{self.run_id}-cmd015-missing-incident",
            }
            leaked = [
                command.command_id
                for command in self.store.list_remote_commands()
                if command.incident_id in command_incidents
            ]
            for command_id in leaked:
                self.created_commands.add(command_id)
            expect(not leaked, f"refused restarts still minted commands: {leaked!r}")
        finally:
            self.store._delete(
                "restart_budget",
                self.store._restart_budget_key(self.cluster_id, job_id),
            )
        self.record(
            "GF-REGIONAL-CMD-015",
            exhausted_error=exhausted.error,
            exhausted_details=dict(exhausted.details),
            unreserved_error=unreserved.error,
            unreserved_details=dict(unreserved.details),
            missing_context_error=incomplete.error,
            commands_created=0,
            restart_budget_deleted=True,
        )

    def run_016(self) -> None:
        run_hyperpod_submission_case(self)

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #
    def _cleanup_since(
        self,
        commands_before: set[str],
        workflows_before: set[str],
        incidents_before: set[str],
    ) -> None:
        for command_id in self.created_commands - commands_before:
            self.store._delete("remote_command", command_id)
            self.created_commands.discard(command_id)
        for workflow_id in self.created_workflows - workflows_before:
            self.store._delete("workflow", workflow_id)
            self.created_workflows.discard(workflow_id)
        for incident_id in self.created_incidents - incidents_before:
            self.store._delete("incident", incident_id)
            self.created_incidents.discard(incident_id)

    def write_case_evidence(self, case_id: str) -> Path | None:
        if self.run_dir is None:
            return None
        document = case_evidence_document(
            case_id=case_id,
            details=self.results.get(case_id) or {},
            run_id=self.run_id,
            cluster_id=self.cluster_id,
            other_cluster_id=self.other_cluster_id,
            preflight=self.preflight_result,
            release_id=self.release_id,
        )
        return write_case_evidence(self.run_dir, document)

    def run(self) -> dict[str, Any]:
        """Run every case, each judged on its own; the summary carries the verdict.

        One case's failure used to abort the remaining cases and swallow the
        report; now the failure is recorded against that case and the loop
        goes on. The per-case cleanup still runs whatever happened, so a failed
        case leaves no seeded object behind for the next one to trip on.
        """

        if not self.preflight_result:
            self.preflight()
        for case_id in AUDITED_CASE_IDS:
            method = getattr(self, f"run_{case_id.rsplit('-', 1)[1]}")
            commands_before = set(self.created_commands)
            workflows_before = set(self.created_workflows)
            incidents_before = set(self.created_incidents)
            try:
                method()
            except Exception as exc:  # noqa: BLE001 -- recorded as the case verdict
                self.results[case_id] = {
                    **dict(self.results.get(case_id) or {}),
                    "verdict": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"FAIL {case_id}: {type(exc).__name__}: {exc}", flush=True)
            else:
                self.results[case_id] = {
                    **dict(self.results.get(case_id) or {}),
                    "verdict": "PASS",
                }
                print(f"PASS {case_id}", flush=True)
            finally:
                self._cleanup_since(commands_before, workflows_before, incidents_before)
            self.write_case_evidence(case_id)
        verdicts = {
            case_id: self.results[case_id]["verdict"] for case_id in self.results
        }
        return {
            "run_id": self.run_id,
            "cluster_id": self.cluster_id,
            "other_cluster_id": self.other_cluster_id,
            "release_id": self.release_id or None,
            "preflight": self.preflight_result,
            "verdict": (
                "PASS"
                if verdicts and all(value == "PASS" for value in verdicts.values())
                else "FAIL"
            ),
            "verdicts": verdicts,
            "results": self.results,
        }


def case_evidence_document(
    *,
    case_id: str,
    details: dict[str, Any],
    run_id: str,
    cluster_id: str,
    other_cluster_id: str,
    preflight: dict[str, Any],
    release_id: str = "",
) -> dict[str, Any]:
    """One CMD case's evidence document, from the case's recorded details."""

    body = dict(details)
    document: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "fault-acceptance",
        "case_id": case_id,
        "verdict": body.pop("verdict", "FAIL"),
        "run_id": run_id,
        "cluster_id": cluster_id,
        "other_cluster_id": other_cluster_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "details": body,
    }
    if release_id:
        document["release_id"] = release_id
    if "error" in body:
        document["error"] = body["error"]
    return document


def _local_write_json(path: Path, document: dict[str, Any]) -> None:
    """The in-Pod fallback: same path layout, all-or-nothing rename, no scope."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def write_case_evidence(run_dir: Path, document: dict[str, Any]) -> Path:
    """Write ``cases/<id>/<id>.json`` under ``run_dir``.

    In a checkout the shared writer is used, so the acceptance scope guard sees
    the document like any other case's. Inside the Pod there is no checkout;
    the same layout is written directly and the operator copies it out (or
    re-derives it from the printed summary with ``--write-evidence``).
    """

    case_id = str(document["case_id"])
    try:
        from scripts.e2e.regional.acceptance_runner_common import (
            write_json_atomic,
        )
        from scripts.e2e.regional.regional_case_contract import (
            case_evidence_path,
        )
    except ImportError:
        path = run_dir / "cases" / case_id / f"{case_id}.json"
        _local_write_json(path, document)
        return path
    path = case_evidence_path(run_dir, case_id)
    write_json_atomic(path, document)
    return path


def write_evidence_from_summary(
    summary: dict[str, Any],
    run_dir: Path,
    *,
    release_id: str = "",
) -> list[Path]:
    """Operator-side: turn the printed run summary into per-case evidence files."""

    results = summary.get("results")
    if not isinstance(results, dict) or not results:
        raise ProtocolAuditError("summary carries no per-case results")
    written = []
    for case_id in AUDITED_CASE_IDS:
        details = results.get(case_id)
        if not isinstance(details, dict):
            continue
        document = case_evidence_document(
            case_id=case_id,
            details=details,
            run_id=str(summary.get("run_id") or ""),
            cluster_id=str(summary.get("cluster_id") or ""),
            other_cluster_id=str(summary.get("other_cluster_id") or ""),
            preflight=dict(summary.get("preflight") or {}),
            release_id=release_id or str(summary.get("release_id") or ""),
        )
        written.append(write_case_evidence(run_dir, document))
    return written


def run_hyperpod_submission_case(audit: LiveProtocolAudit) -> None:
    key = f"{audit.run_id}-cmd016"
    cluster_name = audit.registry[audit.cluster_id]["hyperpod_cluster_name"]
    other_name = audit.registry[audit.other_cluster_id]["hyperpod_cluster_name"]
    node = f"{audit.run_id}-node"
    record = {
        "cluster_name": cluster_name,
        "idempotency_key": key,
        "action": "REBOOT",
        "requested_node_identifiers": [node],
    }
    request = {"cluster_id": audit.cluster_id, "record": record}
    first_status, first = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload=request,
    )
    second_status, second = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload=request,
    )
    cross_status, _ = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.other_cluster_id,
        token=audit.tokens[audit.other_cluster_id],
        payload={
            "cluster_id": audit.other_cluster_id,
            "record": record,
        },
    )
    mismatch_status, _ = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/reserve",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.other_cluster_id,
            "record": {
                **record,
                "cluster_name": other_name,
            },
        },
    )
    query = urllib.parse.urlencode(
        {
            "cluster_name": cluster_name,
            "idempotency_key": key,
        }
    )
    get_status, current = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    missing_query = urllib.parse.urlencode(
        {
            "cluster_name": cluster_name,
            "idempotency_key": f"{key}-missing",
        }
    )
    missing_status, missing = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{missing_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    # A GET on a key nobody reserved must not create the record it looked for.
    missing_again_status, missing_again = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{missing_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    outcome_record = {**record, "state": "SUBMITTED"}
    outcome_status, outcome = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": outcome_record,
        },
    )
    unreserved_key = f"{key}-unreserved"
    unreserved_status, unreserved = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": {
                **outcome_record,
                "idempotency_key": unreserved_key,
            },
        },
    )
    unreserved_query = urllib.parse.urlencode(
        {"cluster_name": cluster_name, "idempotency_key": unreserved_key}
    )
    unreserved_get_status, unreserved_after = audit._request(
        "GET",
        f"/v1/regional/executors/hyperpod-submissions?{unreserved_query}",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
    )
    changed_status, changed = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": {
                **outcome_record,
                "requested_node_identifiers": [f"{node}-changed"],
            },
        },
    )
    anonymous = {}
    for method, path, payload in (
        (
            "POST",
            "/v1/regional/executors/hyperpod-submissions/reserve",
            request,
        ),
        (
            "GET",
            f"/v1/regional/executors/hyperpod-submissions?{query}",
            None,
        ),
        (
            "POST",
            "/v1/regional/executors/hyperpod-submissions/outcome",
            {
                "cluster_id": audit.cluster_id,
                "record": outcome_record,
            },
        ),
    ):
        status, _ = audit._request(
            method,
            path,
            cluster_id=None,
            token=None,
            payload=payload,
        )
        anonymous[method + " " + path.split("?")[0]] = status
        expect(status in {401, 403}, f"anonymous {method} {path} answered {status}")
    expect(
        first_status == second_status == 200,
        f"reserve answered {first_status} then {second_status}",
    )
    expect(first["reserved"] is True, "the first reserve did not reserve")
    expect(
        second["reserved"] is False and second["record"]["state"] == "INTENDED",
        "the duplicate reserve did not return the INTENDED record",
    )
    expect(
        cross_status == mismatch_status == 403,
        f"cross-cluster answered {cross_status}, header/body mismatch "
        f"{mismatch_status}",
    )
    expect(
        get_status == 200 and current["state"] == "INTENDED",
        f"GET answered {get_status}",
    )
    expect(
        missing_status == 200 and missing is None,
        f"GET of a missing key answered {missing_status}: {missing!r}",
    )
    expect(
        missing_again_status == 200 and missing_again is None,
        "a GET of a missing key created a record",
    )
    expect(
        outcome_status == 200 and outcome["record"]["state"] == "SUBMITTED",
        f"outcome answered {outcome_status}",
    )
    expect(
        unreserved_status == 409 and "unreserved" in unreserved["detail"],
        f"unreserved outcome answered {unreserved_status}",
    )
    expect(
        unreserved_get_status == 200 and unreserved_after is None,
        "a refused unreserved outcome left a record behind",
    )
    expect(
        changed_status == 409 and "does not match" in changed["detail"],
        f"changed request answered {changed_status}",
    )
    audit.record(
        "GF-REGIONAL-CMD-016",
        reserve=[first_status, second_status],
        cross_cluster=cross_status,
        header_body_mismatch=mismatch_status,
        get=get_status,
        missing_get=missing_status,
        missing_get_repeat=missing_again_status,
        missing_get_created_record=missing_again is not None,
        outcome=outcome_status,
        unreserved=unreserved_status,
        unreserved_record_after=unreserved_after,
        changed_request=changed_status,
        anonymous=anonymous,
        cloudtrail={
            "status": "NOT_EVALUATED",
            "reason": (
                "the audit runs inside the API Pod without AWS credentials; "
                "the operator confirms no HyperPod UpdateClusterSoftware / "
                "BatchDeleteClusterNodes event for the probe key in CloudTrail"
            ),
            "idempotency_key": key,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument("--cluster-id", default="")
    result.add_argument("--other-cluster-id", default="")
    result.add_argument("--executor-sha256", default="")
    result.add_argument("--executor-digest", default="")
    result.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write cases/<id>/<id>.json per CMD case under this directory",
    )
    result.add_argument(
        "--release-id",
        default="",
        help="release the evidence is bound to (from the release state ConfigMap)",
    )
    result.add_argument(
        "--write-evidence",
        type=Path,
        default=None,
        metavar="SUMMARY_JSON",
        help=(
            "operator-side: read a printed run summary and write the per-case "
            "evidence under --run-dir without touching any cluster"
        ),
    )
    isolation = result.add_mutually_exclusive_group()
    isolation.add_argument(
        "--isolated-cluster",
        action="store_true",
        help="the operator asserts no production executor serves this cluster",
    )
    isolation.add_argument(
        "--executor-ready-replicas",
        type=int,
        default=None,
        help=(
            "Ready replicas of the executor Deployment as observed outside the "
            "Pod; the audit only runs when this is 0"
        ),
    )
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.write_evidence is not None:
        if arguments.run_dir is None:
            raise SystemExit("--write-evidence needs --run-dir")
        summary = json.loads(arguments.write_evidence.read_text(encoding="utf-8"))
        written = write_evidence_from_summary(
            summary, arguments.run_dir, release_id=arguments.release_id
        )
        print(json.dumps({"written": [str(path) for path in written]}, indent=2))
        return 0 if summary.get("verdict") == "PASS" else 1
    for name in (
        "cluster_id",
        "other_cluster_id",
        "executor_sha256",
        "executor_digest",
    ):
        if not getattr(arguments, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required to run the audit")
    audit = LiveProtocolAudit(
        cluster_id=arguments.cluster_id,
        other_cluster_id=arguments.other_cluster_id,
        executor_sha256=arguments.executor_sha256,
        executor_digest=arguments.executor_digest,
        run_dir=arguments.run_dir,
        isolated_cluster=arguments.isolated_cluster,
        executor_ready_replicas=arguments.executor_ready_replicas,
        release_id=arguments.release_id,
    )
    try:
        summary = audit.run()
    finally:
        audit.close()
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
