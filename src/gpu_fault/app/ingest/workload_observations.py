"""Ingest one ``POST /v1/workload-observations`` body.

The observation is the data plane's statement of where an attempt runs; it is
stored first. The placement hold (rule A, case 2: the attempt landed on a node
another incident is repairing) is the control plane's reaction to it and is
subordinate to it -- a hold that cannot be opened is logged and counted, and
the observation is still accepted, so the watcher never loses a rank because
the orchestrator could not read the node's workflows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from gpu_fault.watcher import AttemptObservation

if TYPE_CHECKING:
    from gpu_fault.execution.dispatcher import WorkflowDispatcher
    from gpu_fault.orchestration import IncidentOrchestrator
    from gpu_fault.telemetry import WorkloadTopologyService

LOGGER = logging.getLogger(__name__)


class ObservationIngestContext(Protocol):
    """The slice of ``ApplicationContext`` this ingest path reads."""

    topology: WorkloadTopologyService
    orchestrator: IncidentOrchestrator
    dispatcher: WorkflowDispatcher


def ingest_workload_observation(
    context: ObservationIngestContext, observation: AttemptObservation
) -> None:
    context.topology.observe(observation)
    if hold_attempt_placement(context, observation):
        # Process-local (F-A8): shortens the poll only when the ingress and
        # the worker share a process; the scan cadence is the real bound.
        context.dispatcher.wake()


def hold_attempt_placement(
    context: ObservationIngestContext, observation: AttemptObservation
) -> bool:
    """Open a placement hold when the observed attempt runs on nodes under
    repair; ``True`` when one was opened. Never raises."""

    orchestrator = context.orchestrator
    try:
        held = orchestrator.hold_attempt_on_repairing_nodes(observation)
    except Exception:  # noqa: BLE001 - the observation must still be accepted
        orchestrator.placement_holds_failed_total += 1
        LOGGER.exception(
            "placement hold for attempt %s of job %s could not be opened; "
            "the observation was accepted",
            observation.attempt_id,
            observation.job_id,
        )
        return False
    return held is not None
