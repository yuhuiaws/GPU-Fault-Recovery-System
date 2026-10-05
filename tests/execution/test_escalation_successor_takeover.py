"""The storeless executor takes a node over for the predecessor's own
escalation successor, on the control plane's word.

The regional executor owns no store; ``RemoteIncidentOwnershipReport`` is its
only view of the previous owner. The report now says whether the hold is a
*terminal* quarantine (RMA class, a spare failover's original) -- which no
successor lifts -- or the plain QUARANTINED state of a workflow that failed
after its cordon, which the ``reboot-after`` / ``replace-after`` successor of
that very workflow may take over. A report without the field (an older
control plane) licenses no takeover of a quarantine at all.
"""

from __future__ import annotations

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.regional import RemoteIncidentOwnershipReport
from tests._builders import build_store
from tests.execution._support import UnusedApi
from tests.execution.test_isolation_precheck import (
    OWNED,
    PREDECESSOR,
    SUCCESSOR,
    SUCCESSOR_EVENT,
    UNRELATED_EVENT,
    TwoNodeCore,
    _records,
)


class ReportProvider:
    def __init__(self, report: RemoteIncidentOwnershipReport) -> None:
        self.report = report
        self.queried: list[str] = []

    def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
        self.queried.append(incident_id)
        return self.report


def _report(**overrides: object) -> RemoteIncidentOwnershipReport:
    values: dict[str, object] = {
        "incident_id": PREDECESSOR,
        "known": True,
        "workflow_request_id": "wf-reset-drill",
        "workflow_status": "FAILED",
        "terminal": True,
        "incident_state": "QUARANTINED",
        "quarantine_hold": True,
        "terminal_quarantine": False,
    }
    values.update(overrides)
    return RemoteIncidentOwnershipReport.model_validate(values)


def _storeless(
    core: TwoNodeCore, provider: ReportProvider
) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=None,
        ownership_provider=provider,
    )


def test_the_successor_takes_over_a_non_terminal_quarantine_on_the_reported_word() -> (
    None
):
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    provider = ReportProvider(_report())
    context = _records(build_store(), successor_nodes=[OWNED], event_id=SUCCESSOR_EVENT)

    outcome = _storeless(core, provider).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert core.nodes[OWNED]["metadata"]["annotations"]["gpu-fault.io/incident-id"] == (
        SUCCESSOR
    )
    assert provider.queried == [PREDECESSOR], (
        "one lookup serves the pre-check and the patch"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"terminal_quarantine": True}, "a terminal quarantine stays with its owner"),
        (
            {"terminal_quarantine": None},
            "an older control plane cannot vouch for the hold: no takeover",
        ),
        (
            {"terminal": False, "workflow_status": "RUNNING"},
            "the owner is still running",
        ),
        ({"known": False}, "an unknown owner is a live owner"),
        (
            {"workflow_request_id": "wf-some-other-workflow"},
            "the successor escalates another workflow, not this owner's",
        ),
    ],
)
def test_the_reported_hold_refuses_everything_but_the_continuing_successor(
    overrides: dict[str, object], reason: str
) -> None:
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    context = _records(build_store(), successor_nodes=[OWNED], event_id=SUCCESSOR_EVENT)

    outcome = _storeless(core, ReportProvider(_report(**overrides))).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, reason
    assert "already isolated by another incident" in (outcome.error or ""), reason
    assert core.nodes[OWNED]["metadata"]["annotations"]["gpu-fault.io/incident-id"] == (
        PREDECESSOR
    ), reason


def test_an_unrelated_incident_still_cannot_take_a_quarantined_node() -> None:
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    context = _records(build_store(), successor_nodes=[OWNED], event_id=UNRELATED_EVENT)

    outcome = _storeless(core, ReportProvider(_report())).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True


def test_a_report_without_a_quarantine_hold_transfers_as_before() -> None:
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    context = _records(build_store(), successor_nodes=[OWNED], event_id=UNRELATED_EVENT)
    report = _report(incident_state="ESCALATED", quarantine_hold=False)

    outcome = _storeless(core, ReportProvider(report)).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, (
        "a terminal owner that holds no quarantine never blocked a takeover"
    )
