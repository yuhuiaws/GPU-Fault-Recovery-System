"""Control-plane read scripts of GF-REGIONAL-DESTR-017.

Split out of ``run_destr017_out_of_band_reboot_fence.py`` so the runner stays
a driver. Every script here runs inside a control-plane Pod through
``RegionalLiveFixture.cpu_python`` and only reads: the escalation ladder, the
fenced workflow's settled aftermath, and the retired-generation reconcile in
plan mode.
"""

from __future__ import annotations

from pathlib import Path

ESCALATION_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

request_id = sys.argv[1]
store = ApplicationContext.from_environment().store


def pair(event_id):
    incident = store.get_incident_by_event(event_id)
    if incident is None:
        return None
    workflow = None
    if incident.workflow_request_id:
        try:
            workflow = store.get_workflow(incident.workflow_request_id)
        except NotFoundError:
            workflow = None
    return {
        "incident": incident.model_dump(mode="json"),
        "workflow": (
            workflow.model_dump(mode="json") if workflow is not None else None
        ),
    }


# The escalation ladder names its follow-on records deterministically:
# ``<rung>-after-<predecessor request id>``. Reading all four says both which
# rung was taken and that none of the hardware rungs was.
print(json.dumps({
    name: pair("%s-after-%s" % (name, request_id))
    for name in ("support", "reboot", "replace", "drain")
}, sort_keys=True, default=str))
"""

# Fetch the fenced RESET_GPU workflow, its origin incident, that incident's
# recovery successors (workflows that descend from the fenced one and are not
# the operator support escalation), and the fenced workflow's own remote
# commands -- all keyed off the request id captured at WAITING, so the verdict
# never has to follow the incident pointer to a self-heal successor.
AFTERMATH_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

fenced_id = sys.argv[1]
store = ApplicationContext.from_environment().store

fenced = None
try:
    fenced = store.get_workflow(fenced_id)
except NotFoundError:
    fenced = None

incident = None
if fenced is not None and getattr(fenced, "incident_id", None):
    try:
        incident = store.get_incident(fenced.incident_id)
    except NotFoundError:
        incident = None

successors = []
for workflow in store.list_workflows(None, limit=500, newest_first=True):
    if (
        getattr(workflow, "predecessor_workflow_id", None) == fenced_id
        and not str(workflow.request_id).startswith("workflow-support-after-")
    ):
        successors.append(workflow.model_dump(mode="json"))

commands = [
    item.model_dump(mode="json", exclude={"lease_token"})
    for item in store.list_remote_commands()
    if item.workflow_request_id == fenced_id
]

print(json.dumps({
    "fenced": fenced.model_dump(mode="json") if fenced is not None else None,
    "incident": incident.model_dump(mode="json") if incident is not None else None,
    "recovery_successors": successors,
    "commands": commands,
}, sort_keys=True, default=str))
"""


def reconcile_plan_script() -> str:
    """The shipped retired-generation planner, plan mode only.

    ``gpu_fault.admin.workflow_reconcile`` ships the module's own source plus a
    stdin driver so an operator can plan against a runtime that predates it; this
    runner reuses the same source and calls the same entry point, but only ever
    the ``plan`` half. Applying a revocation is an operator decision and this
    case records NOT_APPLIED.
    """

    from gpu_fault import retired_generation

    source = Path(retired_generation.__file__).read_text(encoding="utf-8")
    return source + (
        "\n\nimport json as _json\n"
        "from gpu_fault.app import ApplicationContext as _Context\n"
        "_store = _Context.from_environment().store\n"
        "print(_json.dumps(build_retired_generation_plan(_store), "
        "sort_keys=True, default=str))\n"
    )
