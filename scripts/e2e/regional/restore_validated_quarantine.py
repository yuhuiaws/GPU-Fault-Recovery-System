from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveRuntimeProfile,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.orchestration import OPERATION_CAPABILITY
from gpu_fault.store import NotFoundError, PostgresStore

RESTORE_OPERATIONS = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)
# The modes under which a capability names an executable owner -- the same
# pair ``WorkflowBuilder.owner`` accepts when it compiles a workflow.
EXECUTABLE_MODES = {CapabilityMode.OWN, CapabilityMode.DELEGATE}


class RestoreStore(Protocol):
    """The store surface this script uses; ``PostgresStore`` live, any store in tests."""

    def get_incident(self, incident_id: str) -> FaultIncident: ...

    def get_workflow(self, request_id: str) -> WorkflowRequest: ...

    def get_profile(self, version: str) -> EffectiveRuntimeProfile: ...

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> None: ...


def resolve_execution_owner(
    profile: EffectiveRuntimeProfile,
    operation: WorkflowOperation,
) -> str:
    """The owner the compiler would assign ``operation`` under ``profile``.

    Mirrors ``WorkflowBuilder.owner``: the operation's capability from
    ``OPERATION_CAPABILITY``, then the first capability entry in an executable
    mode. A hard-coded owner string is what this replaced; a site whose profile
    delegates validation elsewhere got a workflow no adapter would claim.
    """

    capability: CapabilityName = OPERATION_CAPABILITY[operation]
    for item in profile.capabilities:
        if item.capability is capability and item.mode in EXECUTABLE_MODES:
            return str(item.owner)
    raise ValueError(
        f"profile {profile.profile_version} has no executable owner for "
        f"{capability.value} ({operation.value})"
    )


def build_restore_workflow(
    store: RestoreStore,
    *,
    incident_id: str,
    node_id: str,
    reason: str,
    runtime_profile_version: str,
) -> tuple[FaultIncident, WorkflowRequest]:
    """The validation-first restore for one quarantined node, ready to save.

    Refuses when the incident is not quarantined/escalated, still has an
    active workflow, or when ``runtime_profile_version`` is not a stored
    profile for the incident's cluster -- the version is what every step's
    owner is resolved from, so an unknown or foreign one cannot be accepted
    on the operator's word.
    """

    incident = store.get_incident(incident_id)
    if node_id not in incident.node_ids:
        raise ValueError(f"{node_id} is outside incident {incident_id}")
    if incident.state not in {
        IncidentState.QUARANTINED,
        IncidentState.ESCALATED,
    }:
        raise ValueError(
            "incident is not quarantined/escalated after a failed "
            f"validation: {incident.state.value}"
        )
    if incident.workflow_request_id:
        current = store.get_workflow(incident.workflow_request_id)
        if current.status in {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SAFETY_PENDING,
        }:
            raise ValueError(f"incident still has active workflow {current.request_id}")
    try:
        profile = store.get_profile(runtime_profile_version)
    except (KeyError, NotFoundError) as exc:
        raise ValueError(
            f"runtime profile {runtime_profile_version} is not stored"
        ) from exc
    if profile.cluster_id != incident.cluster_id:
        raise ValueError(
            f"runtime profile {runtime_profile_version} belongs to cluster "
            f"{profile.cluster_id}, not the incident's {incident.cluster_id}"
        )
    now = datetime.now(timezone.utc)
    workflow = WorkflowRequest(
        request_id=f"workflow-validated-restore-{uuid4()}",
        incident_id=incident.incident_id,
        runtime_profile_version=profile.profile_version,
        status=WorkflowStatus.PENDING,
        official_action="RESTORE_SCHEDULING",
        fencing_token=incident.fencing_token,
        official_steps=[
            WorkflowStepSpec(
                operation=operation,
                execution_owner=resolve_execution_owner(profile, operation),
                node_ids=[node_id],
            )
            for operation in RESTORE_OPERATIONS
        ],
        created_at=now,
        updated_at=now,
    )
    updated = incident.model_copy(
        update={
            "state": IncidentState.ACTION_PENDING,
            "workflow_request_id": workflow.request_id,
            "reasons": [*incident.reasons, reason],
            "updated_at": now,
        }
    )
    return updated, workflow


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a validation-first workflow that restores scheduling "
            "for one quarantined node."
        )
    )
    parser.add_argument(
        "--runtime-profile-version",
        required=True,
        help="content-addressed Runtime Profile used by the incident site",
    )
    parser.add_argument("incident_id")
    parser.add_argument("node_id")
    parser.add_argument("reason")
    arguments = parser.parse_args()
    store = PostgresStore(
        store_dsn(),
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        incident, workflow = build_restore_workflow(
            store,
            incident_id=arguments.incident_id,
            node_id=arguments.node_id,
            reason=arguments.reason,
            runtime_profile_version=arguments.runtime_profile_version,
        )
        store.save_incident_and_workflow(incident, workflow)
        print(workflow.request_id)
    finally:
        store.close()


if __name__ == "__main__":
    main()
