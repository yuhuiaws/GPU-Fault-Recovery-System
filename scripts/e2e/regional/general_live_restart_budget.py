"""GF-GENERAL-LIVE-029 store-level restart-budget harness.

The restart budget is decided once, at claim time, by the control-plane
executor's preflight (``execution/restart_budget_preflight.py``): every pending
RESTART_WORKLOAD step is reserved on the job's durable ledger through
``store.reserve_job_restart`` under the step's reservation id
(``<workflow>/<index>/RESTART_WORKLOAD``). A reservation that exists already is
answered idempotently (two executors claiming one restart hold one
reservation), one past the budget is refused, and a step whose safety context
lacks a required parameter is refused before any budget read
(``RESTART_SAFETY_CONTEXT_MISSING``); ``release_job_restart`` frees a
reservation exactly once and ignores one it never held. The adapter-side gate
(``issue_restart_authorization``) repeats the context check and fails closed on
a step that holds no reservation (``RESTART_RESERVATION_MISSING``).

Two 029 variants exercise exactly that layer without a managed job:
``concurrent_claim`` (several claimers reserve the same restart at once, a rival
step races for the last slot, the release is raced too) and ``missing_context``
(the preflight and the gate refuse a step without ``source_gpu_count`` and the
ledger stays untouched; the same step with its context is reserved and
authorized, as the control). The harness is ``BUDGET_PROBE``, run by the
installed control-plane interpreter in an API Pod (``RegionalLiveFixture.
cpu_python`` with ``attempts=1``: a retry after a lost receipt would mutate
twice). It calls the product's own store API and preflight functions on a
*test* job id (``gl29…``) in the live cluster's namespace and never saves an
incident, workflow, command or event -- the workflow and incident the preflight
is handed exist in memory only. This is the one runner module that writes to the
store, and the only row it writes is that test job's restart ledger.

Cleanup releases every reservation the run made through the product API and
then removes the emptied test row with the store's own ``_delete`` primitive:
the product keeps ledgers forever and offers no delete, and this is the
primitive ``command_protocol_probes.cleanup_audit_records`` uses for the
``restart_budget`` rows the command-protocol audit seeds. A row still holding a
reservation this run did not make is left in place and reported, and a job id
outside the ``gl29`` test prefix is never written or deleted.
"""

from __future__ import annotations

import hashlib
from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun

#: How many executors claim the same restart at once (the product runs two or
#: three executor replicas; the store serialises them on one advisory lock).
CLAIMERS = 4
TEST_JOB_PREFIX = "gl29"
CONTEXT_MISSING = "RESTART_SAFETY_CONTEXT_MISSING"
RESERVATION_MISSING = "RESTART_RESERVATION_MISSING"

#: Runs in an API Pod under the product's interpreter; one JSON object on the
#: last stdout line. Modes: ``concurrent``, ``missing-context``, ``cleanup``,
#: ``read``. Every writing mode refuses a job id outside the test prefix and
#: (except cleanup) a ledger row that already exists.
BUDGET_PROBE = r"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from gpu_fault.app import ApplicationContext
from gpu_fault.execution.restart_budget_preflight import (
    issue_restart_authorization,
    reservation_id,
    reserve_restart_budgets,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RestartAuthorization,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import NotFoundError

OWNER = "gpu-fault-kubernetes-adapter"
TEST_PREFIX = "gl29"


def ledger(store, cluster_id, job_id):
    try:
        return store.get_restart_budget(cluster_id, job_id).model_dump(mode="json")
    except NotFoundError:
        return None


def checked_test_job(job_id):
    if not job_id.startswith(TEST_PREFIX):
        raise ValueError(f"refusing to touch a restart ledger outside the test prefix: {job_id}")
    return job_id


def race(count, action):
    barrier = threading.Barrier(count)

    def one(index):
        barrier.wait(timeout=60)
        try:
            return {"index": index, **action(index)}
        except Exception as exc:  # noqa: BLE001 - reported per claimer
            return {"index": index, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=count) as pool:
        return sorted(pool.map(one, range(count)), key=lambda item: item["index"])


def reserve(store, cluster_id, job_id, budget, reservation):
    state, reserved = store.reserve_job_restart(cluster_id, job_id, budget, reservation)
    return {
        "reserved": reserved,
        "restart_count": state.restart_count,
        "reservation_ids": list(state.reservation_ids),
    }


def release(store, cluster_id, job_id, reservation):
    state = store.release_job_restart(cluster_id, job_id, reservation)
    return {"restart_count": state.restart_count, "reservation_ids": list(state.reservation_ids)}


def concurrent(store, cluster_id, job_id, budget, workflow_id, claimers):
    if not 2 <= claimers <= 8:
        raise ValueError("claimers must be within 2..8")
    before = ledger(store, cluster_id, job_id)
    document = {
        "mode": "concurrent",
        "cluster_id": cluster_id,
        "job_id": job_id,
        "budget": budget,
        "claimers": claimers,
        "ledger": {"before": before},
    }
    if before is not None:
        document["refused"] = "preexisting ledger: an earlier attempt left the test row"
        return document
    first = f"{workflow_id}/0/RESTART_WORKLOAD"
    rival = f"{workflow_id}/1/RESTART_WORKLOAD"
    phases = {}
    phases["claim"] = {
        "results": race(claimers, lambda i: reserve(store, cluster_id, job_id, budget, first)),
        "ledger": ledger(store, cluster_id, job_id),
    }
    phases["rival"] = {
        "results": race(claimers, lambda i: reserve(store, cluster_id, job_id, budget, rival)),
        "ledger": ledger(store, cluster_id, job_id),
    }
    phases["release"] = {
        "results": race(claimers, lambda i: release(store, cluster_id, job_id, first)),
        "ledger": ledger(store, cluster_id, job_id),
    }
    phases["false_release"] = {
        "results": [
            {"index": 0, "reservation": rival, **release(store, cluster_id, job_id, rival)},
            {"index": 1, "reservation": first, **release(store, cluster_id, job_id, first)},
        ],
        "ledger": ledger(store, cluster_id, job_id),
    }
    document["reservation_ids"] = {"first": first, "rival": rival}
    document["phases"] = phases
    return document


def restart_step(cluster_id, job_id, attempt_id, budget, omit=()):
    parameters = {
        "cluster_id": cluster_id,
        "job_id": job_id,
        "source_attempt_id": attempt_id,
        "source_gpu_count": 8,
        "restart_budget": budget,
    }
    for key in omit:
        parameters.pop(key, None)
    return WorkflowStepSpec(
        operation=WorkflowOperation.RESTART_WORKLOAD,
        execution_owner=OWNER,
        workload_ids=[f"training/job/{job_id}"],
        parameters=parameters,
    )


def pair(cluster_id, job_id, workflow_id, attempt_id, step):
    # In memory only: never saved, never dispatched. The preflight reads the
    # incident's cluster and the workflow's id, steps and step records.
    incident = FaultIncident(
        incident_id=f"{workflow_id}-incident",
        event_id=f"{workflow_id}-event",
        event_type="GENERAL_LIVE_BUDGET_PROBE",
        cluster_id=cluster_id,
        node_ids=[f"{workflow_id}-nonexistent-node"],
        job_id=job_id,
        attempt_id=attempt_id,
        policy_version="general-live-probe",
        policy_source="general-live-probe",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow_id,
    )
    workflow = WorkflowRequest(
        request_id=workflow_id,
        incident_id=incident.incident_id,
        runtime_profile_version="general-live-probe",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[step],
    )
    return incident, workflow


def outcome_view(value):
    details = dict(getattr(value, "details", None) or {})
    status = getattr(value, "status", None)
    return {
        "status": getattr(status, "value", status),
        "error": getattr(value, "error", None),
        "reason": details.get("reason"),
        "missing_parameters": details.get("missing_parameters"),
        "details": details,
    }


def gate_view(value):
    if isinstance(value, RestartAuthorization):
        return {
            "authorized": True,
            "reservation_id": value.reservation_id,
            "restart_count": value.restart_count,
            "restart_budget": value.restart_budget,
        }
    return {"authorized": False, **outcome_view(value)}


def preflight_view(failure):
    if failure is None:
        return None
    return {"step_index": failure.step_index, **outcome_view(failure.outcome)}


def missing_context(store, cluster_id, job_id, budget, workflow_id, attempt_id, missing):
    omitted = [item for item in missing.split(",") if item]
    if not omitted:
        raise ValueError("missing-context needs at least one parameter to omit")
    before = ledger(store, cluster_id, job_id)
    document = {
        "mode": "missing-context",
        "cluster_id": cluster_id,
        "job_id": job_id,
        "budget": budget,
        "omitted": omitted,
        "ledger": {"before": before},
    }
    if before is not None:
        document["refused"] = "preexisting ledger: an earlier attempt left the test row"
        return document
    incomplete = restart_step(cluster_id, job_id, attempt_id, budget, omit=omitted)
    complete = restart_step(cluster_id, job_id, attempt_id, budget)
    incident, workflow = pair(cluster_id, job_id, workflow_id, attempt_id, incomplete)
    reservation = reservation_id(workflow, 0)
    document["reservation_id"] = reservation
    # (a) The claim preflight: the only site that reserves, refusing before it
    # reads the budget row.
    document["preflight"] = preflight_view(
        reserve_restart_budgets(
            store, workflow, incident, workflow.official_steps, phase="official"
        )
    )
    document["ledger"]["after_preflight"] = ledger(store, cluster_id, job_id)
    # (b) The adapter-side gate on the same step, in the preflight's words.
    document["gate"] = gate_view(
        issue_restart_authorization(store, incident, incomplete, reservation, workflow=workflow)
    )
    document["ledger"]["after_gate"] = ledger(store, cluster_id, job_id)
    # (c) A complete step that holds no reservation fails closed at the gate.
    document["unreserved"] = gate_view(
        issue_restart_authorization(store, incident, complete, reservation, workflow=workflow)
    )
    # (d) Control: the same step with its context is reserved and authorized.
    control_incident, control_workflow = pair(cluster_id, job_id, workflow_id, attempt_id, complete)
    control = {
        "preflight": preflight_view(
            reserve_restart_budgets(
                store,
                control_workflow,
                control_incident,
                control_workflow.official_steps,
                phase="official",
            )
        )
    }
    control["gate"] = gate_view(
        issue_restart_authorization(
            store, control_incident, complete, reservation, workflow=control_workflow
        )
    )
    document["control"] = control
    document["ledger"]["after_control"] = ledger(store, cluster_id, job_id)
    return document


def cleanup(store, cluster_id, job_id, workflow_id):
    before = ledger(store, cluster_id, job_id)
    released = []
    if before is not None:
        for reservation in list(before["reservation_ids"]):
            if reservation.startswith(f"{workflow_id}/"):
                release(store, cluster_id, job_id, reservation)
                released.append(reservation)
    remaining = ledger(store, cluster_id, job_id)
    deleted = False
    if (
        remaining is not None
        and remaining["restart_count"] == 0
        and not remaining["reservation_ids"]
    ):
        key = store._restart_budget_key(cluster_id, job_id)
        store._delete("restart_budget", key)
        deleted = store._get_optional("restart_budget", key) is None
        remaining = ledger(store, cluster_id, job_id)
    return {
        "mode": "cleanup",
        "cluster_id": cluster_id,
        "job_id": job_id,
        "before": before,
        "released": released,
        "deleted": deleted,
        "remaining": remaining,
        "clean": remaining is None,
    }


def main(argv):
    mode, *rest = argv
    store = ApplicationContext.from_environment().store
    if mode == "concurrent":
        cluster_id, job_id, budget, workflow_id, claimers = rest
        return concurrent(
            store, cluster_id, checked_test_job(job_id), int(budget), workflow_id, int(claimers)
        )
    if mode == "missing-context":
        cluster_id, job_id, budget, workflow_id, attempt_id, missing = rest
        return missing_context(
            store, cluster_id, checked_test_job(job_id), int(budget), workflow_id, attempt_id, missing
        )
    if mode == "cleanup":
        cluster_id, job_id, workflow_id = rest
        return cleanup(store, cluster_id, checked_test_job(job_id), workflow_id)
    if mode == "read":
        cluster_id, job_id = rest
        return {"mode": "read", "ledger": ledger(store, cluster_id, job_id)}
    raise ValueError(f"unknown budget probe mode {mode!r}")


print(json.dumps(main(sys.argv[1:]), sort_keys=True, default=str))
"""


class SeededRestartBudget:
    """The restart ledger of one variant run's test job, through the product API.

    ``tag`` tells the variants' job ids apart (``gl29cc-…``, ``gl29mc-…``); the
    id is derived from the run id and attempt, so a rerun of the same attempt
    meets its own residue (the probe refuses a preexisting row and the cleanup
    removes it). The cleanup is registered on the context's LIFO stack before
    the first write; a variant that cleans up itself discards ``cleanup_name``.
    """

    def __init__(
        self,
        ctx: VariantContext,
        *,
        tag: str,
        budget: int = 1,
        claimers: int = CLAIMERS,
    ) -> None:
        if not tag.isalnum():
            raise ValueError("a budget harness tag is alphanumeric")
        self.ctx = ctx
        self.cluster_id = ctx.settings.regional.cluster_id
        digest = hashlib.sha256(ctx.run_id.encode()).hexdigest()[:8]
        self.job_id = f"{TEST_JOB_PREFIX}{tag}-{digest}-a{int(ctx.attempt)}"
        self.workflow_id = f"{self.job_id}-workflow"
        self.attempt_id = f"{self.job_id}-a001"
        self.budget = int(budget)
        self.claimers = int(claimers)
        self.cleanup_name = f"budget-{self.job_id}"
        self._registered = False

    def _write(self, *arguments: str, name: str) -> dict[str, Any]:
        if not self._registered:
            self.ctx.cleanup.push(self.cleanup_name, self.cleanup)
            self._registered = True
        value = self.ctx.regional.cpu_python(
            BUDGET_PROBE, *arguments, attempts=1, timeout=240
        )
        self.ctx.evidence.write(name, value)
        refused = value.get("refused")
        if refused:
            raise VariantNotRun(
                "FIXTURE_NOT_READY",
                f"{refused} ({self.cluster_id}/{self.job_id}); the cleanup removes it",
            )
        return value

    def concurrent(self) -> dict[str, Any]:
        """Race ``claimers`` claims of one restart, a rival step and the release."""

        return self._write(
            "concurrent",
            self.cluster_id,
            self.job_id,
            str(self.budget),
            self.workflow_id,
            str(self.claimers),
            name="budget-concurrent",
        )

    def missing_context(self, *missing: str) -> dict[str, Any]:
        """Preflight and gate a RESTART_WORKLOAD step without ``missing``."""

        omitted = missing or ("source_gpu_count",)
        return self._write(
            "missing-context",
            self.cluster_id,
            self.job_id,
            str(self.budget),
            self.workflow_id,
            self.attempt_id,
            ",".join(omitted),
            name="budget-missing-context",
        )

    def read(self) -> dict[str, Any]:
        return self.ctx.regional.cpu_python(
            BUDGET_PROBE, "read", self.cluster_id, self.job_id
        )

    def cleanup(self) -> dict[str, Any]:
        """Release the run's reservations and remove the emptied test row."""

        value = self.ctx.regional.cpu_python(
            BUDGET_PROBE,
            "cleanup",
            self.cluster_id,
            self.job_id,
            self.workflow_id,
            attempts=1,
            timeout=240,
        )
        self.ctx.evidence.write("budget-cleanup", value)
        return value


# --------------------------------------------------------------------------- #
# verdict helpers over the probe's documents
# --------------------------------------------------------------------------- #
def _ledger_errors(label: str, ledger: Any, count: int, ids: list[str]) -> list[str]:
    if not isinstance(ledger, dict):
        return [f"{label}: the test job has no restart ledger"]
    errors = []
    if ledger.get("restart_count") != count or ledger.get("reservation_ids") != ids:
        errors.append(
            f"{label}: ledger is {ledger.get('restart_count')} {ledger.get('reservation_ids')}, "
            f"expected {count} {ids}"
        )
    if ledger.get("restart_count") != len(ledger.get("reservation_ids") or []):
        errors.append(f"{label}: restart_count does not match the reservation ids")
    return errors


def concurrent_claim_errors(document: dict[str, Any]) -> list[str]:
    """Why the concurrent claim/release race did not hold the ledger invariants."""

    if document.get("refused"):
        return [f"the probe refused: {document['refused']}"]
    phases = document.get("phases") or {}
    ids = document.get("reservation_ids") or {}
    first = ids.get("first")
    claimers = int(document.get("claimers") or 0)
    errors: list[str] = []
    claim = phases.get("claim") or {}
    results = claim.get("results") or []
    if claimers < 2 or len(results) != claimers:
        errors.append(
            f"the claim race ran {len(results)} claimers, expected {claimers} >= 2"
        )
    for item in results:
        if item.get("error"):
            errors.append(f"claimer {item.get('index')} failed: {item['error']}")
        elif item.get("reserved") is not True:
            errors.append(f"claimer {item.get('index')} was refused the one restart")
        elif item.get("restart_count") != 1 or item.get("reservation_ids") != [first]:
            errors.append(
                f"claimer {item.get('index')} saw {item.get('restart_count')} "
                f"{item.get('reservation_ids')} (a double deduction or a stranger)"
            )
    errors += _ledger_errors(
        "after the concurrent claim", claim.get("ledger"), 1, [first]
    )
    rival = phases.get("rival") or {}
    for item in rival.get("results") or []:
        if item.get("error"):
            errors.append(f"rival claimer {item.get('index')} failed: {item['error']}")
        elif item.get("reserved") is not False:
            errors.append(
                f"rival claimer {item.get('index')} was granted a restart past the budget"
            )
    errors += _ledger_errors("after the rival claims", rival.get("ledger"), 1, [first])
    release = phases.get("release") or {}
    for item in release.get("results") or []:
        if item.get("error"):
            errors.append(f"releaser {item.get('index')} failed: {item['error']}")
        elif item.get("restart_count") != 0 or item.get("reservation_ids"):
            errors.append(
                f"releaser {item.get('index')} saw {item.get('restart_count')} "
                f"{item.get('reservation_ids')} after the release"
            )
    errors += _ledger_errors(
        "after the concurrent release", release.get("ledger"), 0, []
    )
    false = phases.get("false_release") or {}
    for item in false.get("results") or []:
        if item.get("error"):
            errors.append(f"false release {item.get('index')} failed: {item['error']}")
    errors += _ledger_errors("after releasing unheld ids", false.get("ledger"), 0, [])
    for name, phase in phases.items():
        for item in phase.get("results") or []:
            count = item.get("restart_count")
            if isinstance(count, int) and count < 0:
                errors.append(f"{name}: restart_count went negative")
        count = (phase.get("ledger") or {}).get("restart_count")
        if isinstance(count, int) and count < 0:
            errors.append(f"{name}: the ledger's restart_count went negative")
    return errors


def missing_context_errors(document: dict[str, Any]) -> list[str]:
    """Why a context-less restart step was not refused fail-closed at preflight."""

    if document.get("refused"):
        return [f"the probe refused: {document['refused']}"]
    errors: list[str] = []
    ledgers = document.get("ledger") or {}
    preflight = document.get("preflight")
    if not preflight:
        errors.append(
            "the claim preflight admitted a RESTART_WORKLOAD step without its safety context"
        )
    else:
        if preflight.get("reason") != CONTEXT_MISSING:
            errors.append(
                f"the preflight refused with {preflight.get('reason')}, expected {CONTEXT_MISSING}"
            )
        if preflight.get("status") != "FAILED":
            errors.append(
                f"the preflight refusal is {preflight.get('status')}, not FAILED"
            )
        if sorted(preflight.get("missing_parameters") or []) != sorted(
            document.get("omitted") or []
        ):
            errors.append("the refusal does not name exactly the omitted parameters")
    if ledgers.get("before") is not None:
        errors.append("the test job already had a restart ledger before the probe")
    if ledgers.get("after_preflight") is not None:
        errors.append(
            "the refused preflight wrote a restart ledger row (the budget was touched "
            "before the context check)"
        )
    gate = document.get("gate") or {}
    if gate.get("authorized") or gate.get("reason") != CONTEXT_MISSING:
        errors.append(
            f"the adapter gate answered {gate.get('reason')} for a step without context, "
            f"expected {CONTEXT_MISSING}"
        )
    if ledgers.get("after_gate") is not None:
        errors.append("the adapter gate wrote a restart ledger row")
    unreserved = document.get("unreserved") or {}
    if unreserved.get("authorized") or unreserved.get("reason") != RESERVATION_MISSING:
        errors.append(
            f"a complete step holding no reservation was answered {unreserved.get('reason')}, "
            f"expected {RESERVATION_MISSING} (fail closed)"
        )
    control = document.get("control") or {}
    if control.get("preflight") is not None:
        errors.append(f"the complete context was refused: {control.get('preflight')}")
    if not (control.get("gate") or {}).get("authorized"):
        errors.append("the complete, reserved context was not authorized by the gate")
    errors += _ledger_errors(
        "after the control reservation",
        ledgers.get("after_control"),
        1,
        [document.get("reservation_id")],
    )
    return errors


def cleanup_errors(document: dict[str, Any]) -> list[str]:
    if document.get("clean") is not True:
        return [f"the test restart ledger remains: {document.get('remaining')}"]
    return []


__all__ = [
    "BUDGET_PROBE",
    "CLAIMERS",
    "CONTEXT_MISSING",
    "RESERVATION_MISSING",
    "SeededRestartBudget",
    "TEST_JOB_PREFIX",
    "cleanup_errors",
    "concurrent_claim_errors",
    "missing_context_errors",
]
