from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    FaultIncident,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteActionCommand,
)


class LiveProtocolAudit:
    def __init__(
        self,
        *,
        cluster_id: str,
        other_cluster_id: str,
        executor_sha256: str,
        executor_digest: str,
    ) -> None:
        self.cluster_id = cluster_id
        self.other_cluster_id = other_cluster_id
        self.executor_sha256 = executor_sha256
        self.executor_digest = executor_digest
        self.run_id = f"cmd-audit-{uuid4().hex[:12]}"
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

    def close(self) -> None:
        for command_id in self.created_commands:
            self.store._delete("remote_command", command_id)
        for workflow_id in self.created_workflows:
            self.store._delete("workflow", workflow_id)
        for incident_id in self.created_incidents:
            self.store._delete("incident", incident_id)

    def record(self, case_id: str, **details: Any) -> None:
        self.results[case_id] = details

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
                "executor_protocol_version": 2,
                "executor_artifact_sha256": self.executor_sha256,
                "executor_compatibility_digest": self.executor_digest,
                "execution_owners": (
                    ["gpu-fault-kubernetes-adapter"] if owners is None else owners
                ),
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

    def seed(
        self,
        suffix: str,
        *,
        owner: str = "gpu-fault-kubernetes-adapter",
        cluster_id: str | None = None,
        created_at: datetime | None = None,
        fencing_token: int = 1,
        persist_workflow: bool = False,
    ) -> RemoteActionCommand:
        target = cluster_id or self.cluster_id
        incident_id = f"{self.run_id}-{suffix}-incident"
        workflow_id = f"{self.run_id}-{suffix}-workflow"
        command_id = f"{self.run_id}-{suffix}-command"
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=f"{self.run_id}-{suffix}-event",
            event_type="CMD_PROTOCOL_PROBE",
            cluster_id=target,
            node_ids=[f"{self.run_id}-nonexistent-node"],
            policy_version="cmd-probe",
            policy_source="cmd-probe",
            workflow_request_id=workflow_id,
            fencing_token=fencing_token,
        )
        step = WorkflowStepSpec(
            operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
            execution_owner=owner,
            node_ids=[f"{self.run_id}-nonexistent-node"],
        )
        workflow = WorkflowRequest(
            request_id=workflow_id,
            incident_id=incident_id,
            status=WorkflowStatus.RUNNING,
            fencing_token=fencing_token,
            official_steps=[step],
        )
        if persist_workflow:
            self.store.save_incident_and_workflow(incident, workflow)
            self.created_incidents.add(incident_id)
            self.created_workflows.add(workflow_id)
        command = RemoteActionCommand(
            command_id=command_id,
            cluster_id=target,
            workflow_request_id=workflow_id,
            incident_id=incident_id,
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
        assert isinstance(token, str) and token
        return token

    def run_001(self) -> None:
        observed: dict[str, int] = {}
        for value in (0, 1, 25, 26, -1, "5", 5.5):
            status, body = self.claim(max_commands=value)
            observed[repr(value)] = status
            if value in {1, 25}:
                assert status == 200 and len(body["commands"]) <= int(value)
            elif value == "5":
                assert status in {200, 422}
            else:
                assert status == 422
        self.record("GF-REGIONAL-CMD-001", statuses=observed)

    def run_002(self) -> None:
        invalid = {}
        for value in (9, 7201, 0):
            status, _ = self.claim(lease_seconds=value)
            invalid[str(value)] = status
            assert status == 422
        command = self.seed("cmd002")
        leases = {}
        for value in (10, 600, 601, 7200):
            before = datetime.now(timezone.utc)
            status, body = self.claim(
                executor_id=f"{self.run_id}-lease-{value}",
                lease_seconds=value,
            )
            after = datetime.now(timezone.utc)
            assert status == 200 and len(body["commands"]) == 1
            claimed = body["commands"][0]
            expires = datetime.fromisoformat(
                claimed["lease_expires_at"].replace("Z", "+00:00")
            )
            lower = (expires - after).total_seconds()
            upper = (expires - before).total_seconds()
            assert value - 3 <= lower <= value + 3
            assert value - 3 <= upper <= value + 3
            leases[str(value)] = round(lower, 3)
            status, _ = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(claimed),
                    "status": "WAITING",
                    "details": {"lease_seconds": value},
                },
            )
            assert status == 200
        status, body = self.claim(lease_seconds=10)
        assert status == 200 and len(body["commands"]) == 1
        leased = body["commands"][0]
        time.sleep(11)
        expired_status, _ = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        assert expired_status == 409
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
            status, _ = self.claim(owners=owners)
            observed[str(len(owners)) + ":" + repr(owners[:2])] = status
            assert status == expected
        command = self.seed("cmd003", owner="gpu-fault-node-agent")
        status, body = self.claim(owners=[])
        assert status == 200 and body["commands"] == []
        self.record(
            "GF-REGIONAL-CMD-003",
            statuses=observed,
            empty_owner_claimed=[],
            protected_command=command.command_id,
        )

    def run_004(self) -> None:
        commands = {
            owner: self.seed(f"cmd004-{index}", owner=owner)
            for index, owner in enumerate(
                (
                    "gpu-fault-kubernetes-adapter",
                    "gpu-fault-node-agent",
                    "gpu-fault-hyperpod-adapter",
                )
            )
        }
        status, body = self.claim(owners=["gpu-fault-node-agent"], max_commands=5)
        assert status == 200
        assert [item["step"]["execution_owner"] for item in body["commands"]] == [
            "gpu-fault-node-agent"
        ]
        status, remaining = self.claim(
            owners=[
                "gpu-fault-kubernetes-adapter",
                "gpu-fault-hyperpod-adapter",
            ],
            max_commands=5,
        )
        owners = [item["step"]["execution_owner"] for item in remaining["commands"]]
        assert status == 200 and set(owners) == {
            "gpu-fault-kubernetes-adapter",
            "gpu-fault-hyperpod-adapter",
        }
        self.record(
            "GF-REGIONAL-CMD-004",
            command_ids={key: value.command_id for key, value in commands.items()},
            claimed_owners=["gpu-fault-node-agent", *owners],
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
        status, body = self.claim(max_commands=5)
        actual = [item["command_id"] for item in body["commands"]]
        assert status == 200 and actual == expected
        self.record("GF-REGIONAL-CMD-005", expected=expected, actual=actual)

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
        assert status == 200
        status, second = self.claim(executor_id="cmd006-a2", lease_seconds=10)
        two = second["commands"][0]
        token_two = self._lease_token(two)
        stale_one, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_one, "status": "SUCCEEDED"},
        )
        assert stale_one == 409
        time.sleep(11)
        stale_two, _ = self.complete(
            command.command_id,
            payload={"lease_token": token_two, "status": "SUCCEEDED"},
        )
        assert stale_two == 409
        status, third = self.claim(executor_id="cmd006-a3", lease_seconds=60)
        three = third["commands"][0]
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(three),
                "status": "SUCCEEDED",
            },
        )
        assert final_status == 200 and final["status"] == "SUCCEEDED"
        assert final["last_lease_owner"] == "cmd006-a3"
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
        variants = [
            {"lease_token": token, "status": "SUCCEEDED"},
            {"lease_token": "wrong-token", "status": "SUCCEEDED"},
            {
                "lease_token": "wrong-token",
                "status": "FAILED",
                "error": "must not overwrite",
            },
        ]
        for payload in variants:
            status, body = self.complete(command.command_id, payload=payload)
            assert status == 200 and body == first
        self.record(
            "GF-REGIONAL-CMD-007",
            first_status=first_status,
            replay_statuses=[200, 200, 200],
            final=first,
        )

    def run_008(self) -> None:
        command = self.seed("cmd008")
        status, claim = self.claim(executor_id="cmd008", lease_seconds=60)
        leased = claim["commands"][0]
        token = self._lease_token(leased)
        payloads = {
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
            assert status == 422
            assert self.store.get_remote_command(command.command_id).status.value == (
                "LEASED"
            )
        waiting, body = self.complete(
            command.command_id,
            payload={"lease_token": token, "status": "WAITING"},
        )
        assert waiting == 200 and body["status"] == "WAITING"
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
        assert missing_status == cross_status == 404
        assert self.store.get_remote_command(other.command_id).status.value == "PENDING"
        self.record(
            "GF-REGIONAL-CMD-009",
            missing={"status": missing_status, "detail": missing["detail"]},
            cross_cluster={"status": cross_status, "detail": cross["detail"]},
        )

    def run_010(self) -> None:
        command = self.seed("cmd010", persist_workflow=True)
        status, claim = self.claim(executor_id="cmd010", lease_seconds=60)
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
        status, reclaimed = self.claim(executor_id="cmd010-reclaim")
        assert stale_status == 409
        assert "fencing token is stale" in stale["detail"]
        assert all(
            item["command_id"] != command.command_id for item in reclaimed["commands"]
        )
        self.record(
            "GF-REGIONAL-CMD-010",
            complete_status=stale_status,
            reclaimed=False,
        )

    def _adapter_context(
        self,
        suffix: str,
        *,
        fencing_token: int = 1,
        step: WorkflowStepSpec | None = None,
    ) -> WorkflowStepContext:
        incident_id = f"{self.run_id}-{suffix}-incident"
        workflow_id = f"{self.run_id}-{suffix}-workflow"
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=f"{self.run_id}-{suffix}-event",
            event_type="CMD_PROTOCOL_PROBE",
            cluster_id=self.cluster_id,
            node_ids=[f"{self.run_id}-nonexistent-node"],
            policy_version="cmd-probe",
            policy_source="cmd-probe",
            fencing_token=fencing_token,
        )
        selected = step or WorkflowStepSpec(
            operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
            execution_owner="gpu-fault-node-agent",
            node_ids=[f"{self.run_id}-nonexistent-node"],
        )
        workflow = WorkflowRequest(
            request_id=workflow_id,
            incident_id=incident_id,
            status=WorkflowStatus.RUNNING,
            fencing_token=fencing_token,
            official_steps=[selected],
        )
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
            owners={"gpu-fault-node-agent"},
        )
        first_context = self._adapter_context("cmd012", fencing_token=1)
        first = adapter.execute(first_context)
        duplicate = adapter.execute(first_context)
        first_id = str(first.details["remote_command_id"])
        duplicate_id = str(duplicate.details["remote_command_id"])
        self.created_commands.add(first_id)
        second_context = self._adapter_context("cmd012", fencing_token=2)
        second = adapter.execute(second_context)
        second_id = str(second.details["remote_command_id"])
        self.created_commands.add(second_id)
        pattern = re.compile(r"^remote-[0-9a-f]{24}$")
        assert first_id == duplicate_id
        assert second_id != first_id
        assert pattern.fullmatch(first_id) and pattern.fullmatch(second_id)
        self.record(
            "GF-REGIONAL-CMD-012",
            first_id=first_id,
            duplicate_id=duplicate_id,
            next_fencing_id=second_id,
        )

    def run_013(self) -> None:
        command = self.seed("cmd013")
        previous: dict[str, int] = {}
        for round_number in range(1, 21):
            status, claim = self.claim(
                executor_id=f"cmd013-{round_number}",
                lease_seconds=60,
            )
            assert status == 200 and len(claim["commands"]) == 1
            leased = claim["commands"][0]
            if round_number > 1:
                assert leased["result_details"] == previous
            previous = {"probe_round": round_number}
            status, _ = self.complete(
                command.command_id,
                payload={
                    "lease_token": self._lease_token(leased),
                    "status": "WAITING",
                    "details": previous,
                },
            )
            assert status == 200
        status, claim = self.claim(executor_id="cmd013-final", lease_seconds=60)
        leased = claim["commands"][0]
        assert leased["result_details"] == {"probe_round": 20}
        final_status, final = self.complete(
            command.command_id,
            payload={
                "lease_token": self._lease_token(leased),
                "status": "SUCCEEDED",
            },
        )
        assert final_status == 200 and final["status"] == "SUCCEEDED"
        self.record(
            "GF-REGIONAL-CMD-013",
            waiting_rounds=20,
            final_status=final["status"],
            limitation="no maximum WAITING round count",
        )

    def run_014(self) -> None:
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={"gpu-fault-node-agent"},
        )
        outcome = adapter.execute(self._adapter_context("cmd014"))
        command_id = str(outcome.details["remote_command_id"])
        self.created_commands.add(command_id)
        assert outcome.status.value == "WAITING"
        assert outcome.details == {
            "remote_command_id": command_id,
            "remote_cluster_id": self.cluster_id,
            "remote_status": "PENDING",
            "mutation_submitted_by_control_plane": False,
        }
        self.record("GF-REGIONAL-CMD-014", details=outcome.details)

    def run_015(self) -> None:
        adapter = RegionalRemoteWorkflowAdapter(
            self.store,
            owners={"gpu-fault-kubernetes-adapter"},
        )
        job_id = f"{self.run_id}-budget"
        state, reserved = self.store.reserve_job_restart(
            self.cluster_id,
            job_id,
            1,
            f"{self.run_id}-existing-reservation",
        )
        assert reserved and state.restart_count == 1
        full = WorkflowStepSpec(
            operation=WorkflowOperation.RESTART_WORKLOAD,
            execution_owner="gpu-fault-kubernetes-adapter",
            workload_ids=["training/job/nonexistent-cmd015"],
            parameters={
                "cluster_id": self.cluster_id,
                "job_id": job_id,
                "source_attempt_id": f"{job_id}-a1",
                "source_gpu_count": 1,
                "restart_budget": 1,
            },
        )
        exhausted = adapter.execute(
            self._adapter_context("cmd015-exhausted", step=full)
        )
        assert exhausted.status.value == "FAILED"
        assert exhausted.details["reason"] == "RESTART_BUDGET_EXHAUSTED"
        assert "restart budget exhausted" in str(exhausted.error)
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
        assert incomplete.status.value == "FAILED"
        assert "restart safety context is missing: source_gpu_count" in str(
            incomplete.error
        )
        command_incidents = {
            f"{self.run_id}-cmd015-exhausted-incident",
            f"{self.run_id}-cmd015-missing-incident",
        }
        assert not [
            command
            for command in self.store.list_remote_commands()
            if command.incident_id in command_incidents
        ]
        self.store._delete(
            "restart_budget",
            self.store._restart_budget_key(self.cluster_id, job_id),
        )
        self.record(
            "GF-REGIONAL-CMD-015",
            exhausted_error=exhausted.error,
            missing_context_error=incomplete.error,
            commands_created=0,
        )

    def run_016(self) -> None:
        run_hyperpod_submission_case(self)

    def run(self) -> dict[str, Any]:
        for method in (
            self.run_001,
            self.run_002,
            self.run_003,
            self.run_004,
            self.run_005,
            self.run_006,
            self.run_007,
            self.run_008,
            self.run_009,
            self.run_010,
            self.run_012,
            self.run_013,
            self.run_014,
            self.run_015,
            self.run_016,
        ):
            commands_before = set(self.created_commands)
            workflows_before = set(self.created_workflows)
            incidents_before = set(self.created_incidents)
            try:
                method()
                print(f"PASS {method.__name__}", flush=True)
            finally:
                for command_id in self.created_commands - commands_before:
                    self.store._delete("remote_command", command_id)
                    self.created_commands.discard(command_id)
                for workflow_id in self.created_workflows - workflows_before:
                    self.store._delete("workflow", workflow_id)
                    self.created_workflows.discard(workflow_id)
                for incident_id in self.created_incidents - incidents_before:
                    self.store._delete("incident", incident_id)
                    self.created_incidents.discard(incident_id)
        return {
            "run_id": self.run_id,
            "cluster_id": self.cluster_id,
            "other_cluster_id": self.other_cluster_id,
            "results": self.results,
        }


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
    unreserved_status, unreserved = audit._request(
        "POST",
        "/v1/regional/executors/hyperpod-submissions/outcome",
        cluster_id=audit.cluster_id,
        token=audit.tokens[audit.cluster_id],
        payload={
            "cluster_id": audit.cluster_id,
            "record": {
                **outcome_record,
                "idempotency_key": f"{key}-unreserved",
            },
        },
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
        assert status in {401, 403}
    assert first_status == second_status == 200
    assert first["reserved"] is True
    assert second["reserved"] is False
    assert second["record"]["state"] == "INTENDED"
    assert cross_status == mismatch_status == 403
    assert get_status == 200 and current["state"] == "INTENDED"
    assert missing_status == 200 and missing is None
    assert outcome_status == 200 and outcome["record"]["state"] == "SUBMITTED"
    assert unreserved_status == 409
    assert "unreserved" in unreserved["detail"]
    assert changed_status == 409
    assert "does not match" in changed["detail"]
    audit.record(
        "GF-REGIONAL-CMD-016",
        reserve=[first_status, second_status],
        cross_cluster=cross_status,
        header_body_mismatch=mismatch_status,
        get=get_status,
        missing_get=missing_status,
        outcome=outcome_status,
        unreserved=unreserved_status,
        changed_request=changed_status,
        anonymous=anonymous,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cluster-id", required=True)
    result.add_argument("--other-cluster-id", required=True)
    result.add_argument("--executor-sha256", required=True)
    result.add_argument("--executor-digest", required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    audit = LiveProtocolAudit(
        cluster_id=arguments.cluster_id,
        other_cluster_id=arguments.other_cluster_id,
        executor_sha256=arguments.executor_sha256,
        executor_digest=arguments.executor_digest,
    )
    try:
        print(json.dumps(audit.run(), indent=2))
    finally:
        audit.close()


if __name__ == "__main__":
    main()
