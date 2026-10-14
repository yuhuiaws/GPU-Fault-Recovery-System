"""The Kubernetes Node names a GPU cluster reports, and the absences they prove.

The control plane holds no GPU kubeconfig, so whether a node is still in its
cluster's Kubernetes can only be the data plane's statement. Each cluster's
executor LISTs the cluster's Nodes -- all of them, unfiltered, the question
``kubectl get nodes`` answers for the administrator -- every
``NODE_INVENTORY_INTERVAL_SECONDS`` and posts the names under the cluster token
(``KubernetesNodeInventoryReport``, ``NODE_INVENTORY_PATH``).

The control plane keeps one record per cluster (``KubernetesNodeInventory``)
and, report by report, the absences: for every name an earlier report of the
same unbroken chain listed and the latest does not, the first report that no
longer listed it (``absent``). A chain breaks when two reports arrive further
apart than ``NODE_INVENTORY_MAX_GAP``; the next report starts a new chain and
remembers nothing, so an absence is never carried across a gap in which the
node could have come back. ``kubernetes_absence`` turns the record into the
departed-node cleanup's Kubernetes evidence: absent from the latest report,
the latest report fresh (``NODE_INVENTORY_FRESHNESS``), and absent from every
report since a moment at least the guard ago. A name the chain never listed has
been absent since the chain started (``retained_since``); one whose absence
began before the retention window was pruned counts from that window's start.
Both are the conservative reading: never longer than the truth.

Every time here is the control plane's (``received_at``); the executor's own
LIST time is kept for the evidence (``observed_at``) and checked against the
receiving clock (``NODE_INVENTORY_MAX_CLOCK_SKEW``), so a report cannot claim a
moment the control plane did not see.

The control plane and the executor use this module. The node runtime wheel
carries it too, only because the store contract names its record types
(``store.contracts``, which the Agent's fleet code imports), so a change here
is a Node Runtime change as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from gpu_fault.models import StrictModel

if TYPE_CHECKING:
    from gpu_fault.store.contracts import DepartedNodeStore

#: Where an executor posts its cluster's Node names (cluster token).
NODE_INVENTORY_PATH = "/v1/regional/executors/node-inventory"
#: How often an executor reports; one unfiltered Node LIST per report.
NODE_INVENTORY_INTERVAL_SECONDS = 60.0
#: The most names one report may carry; a larger cluster is refused, not cut.
NODE_INVENTORY_MAX_NODES = 5000
#: A cluster whose latest report is older than this proves no absence.
NODE_INVENTORY_FRESHNESS = timedelta(minutes=5)
#: Two reports further apart than this break the chain of observations.
NODE_INVENTORY_MAX_GAP = timedelta(minutes=5)
#: How long an absence is remembered by name; older ones count from the window.
NODE_INVENTORY_ABSENCE_RETENTION = timedelta(hours=24)
#: How far a report's LIST time may stand from the control plane's clock.
NODE_INVENTORY_MAX_CLOCK_SKEW = timedelta(minutes=2)
#: A Kubernetes Node name (an RFC 1123 subdomain).
NODE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")

NO_REPORT_BLOCKER = (
    "no Kubernetes node report from the cluster's executor; Kubernetes "
    "absence is unproven"
)
PRESENT_BLOCKER = "node is still in Kubernetes"


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(timezone.utc)


class KubernetesNodeInventoryReport(StrictModel):
    """One executor's statement: these are every Node the cluster has now.

    ``spare_node_names`` are the listed nodes declared as warm spares (the
    ``gpu-fault.io/spare=true`` label) or holding a spare reservation: a
    departed spare is the administrator's to release, never the cleanup's.
    """

    cluster_id: str = Field(min_length=1, max_length=128)
    executor_id: str = Field(min_length=1, max_length=253)
    observed_at: datetime
    resource_version: str | None = Field(default=None, max_length=64)
    node_names: list[str] = Field(
        default_factory=list, max_length=NODE_INVENTORY_MAX_NODES
    )
    spare_node_names: list[str] = Field(
        default_factory=list, max_length=NODE_INVENTORY_MAX_NODES
    )

    @field_validator("observed_at")
    @classmethod
    def _observed_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, "observed_at")

    @field_validator("node_names", "spare_node_names")
    @classmethod
    def _names_are_node_names(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("a node is reported at most once")
        invalid = [name for name in value if NODE_NAME.fullmatch(name) is None]
        if invalid:
            raise ValueError(f"not a Kubernetes Node name: {invalid[0]!r}")
        return sorted(value)

    @model_validator(mode="after")
    def _spares_are_listed(self) -> KubernetesNodeInventoryReport:
        if not set(self.spare_node_names) <= set(self.node_names):
            raise ValueError("a spare node must be a listed node")
        return self


class AbsentNode(StrictModel):
    """A name an earlier report of the chain listed and the latest does not."""

    #: ``received_at`` of the first report that no longer listed it.
    since: datetime
    #: The last report that listed it marked it a declared or reserved spare.
    was_spare: bool = False


class KubernetesNodeInventory(StrictModel):
    """A cluster's latest Node names, with the absences its chain remembers."""

    cluster_id: str
    reporter: str
    observed_at: datetime
    received_at: datetime
    resource_version: str | None = None
    node_names: list[str] = Field(default_factory=list)
    spare_node_names: list[str] = Field(default_factory=list)
    #: ``received_at`` of the chain's first report.
    chain_started_at: datetime
    reports_in_chain: int = Field(default=1, ge=1)
    #: Every absence that began at or after this moment is in ``absent``.
    retained_since: datetime
    absent: dict[str, AbsentNode] = Field(default_factory=dict)


class KubernetesNodeInventoryReceipt(StrictModel):
    """What the control plane holds for the cluster after a report.

    ``recorded`` is False for a report no newer than the stored one (the
    sibling executor's, or a retry): the record is unchanged.
    """

    cluster_id: str
    recorded: bool
    observed_at: datetime
    received_at: datetime
    node_count: int = Field(ge=0)
    absent_count: int = Field(ge=0)
    chain_started_at: datetime
    reports_in_chain: int = Field(ge=1)

    @classmethod
    def of(
        cls, inventory: KubernetesNodeInventory, *, recorded: bool
    ) -> KubernetesNodeInventoryReceipt:
        return cls(
            cluster_id=inventory.cluster_id,
            recorded=recorded,
            observed_at=inventory.observed_at,
            received_at=inventory.received_at,
            node_count=len(inventory.node_names),
            absent_count=len(inventory.absent),
            chain_started_at=inventory.chain_started_at,
            reports_in_chain=inventory.reports_in_chain,
        )


def report_refusal(
    report: KubernetesNodeInventoryReport, *, received_at: datetime
) -> str | None:
    """Why the control plane must not record ``report`` now, or None."""

    skew = report.observed_at - received_at
    if abs(skew) > NODE_INVENTORY_MAX_CLOCK_SKEW:
        return (
            f"report observed_at {report.observed_at.isoformat()} is "
            f"{abs(skew.total_seconds()):.0f}s from the control plane's clock "
            f"(limit {NODE_INVENTORY_MAX_CLOCK_SKEW.total_seconds():.0f}s)"
        )
    return None


class InventoryReportRefused(ValueError):
    """A report the control plane does not record; ``str()`` says why."""


def record_inventory_report(
    store: DepartedNodeStore,
    report: KubernetesNodeInventoryReport,
    *,
    received_at: datetime | None = None,
) -> KubernetesNodeInventoryReceipt:
    """Take one executor report into the cluster's record, or refuse it."""

    received = received_at or datetime.now(timezone.utc)
    refusal = report_refusal(report, received_at=received)
    if refusal is not None:
        raise InventoryReportRefused(refusal)
    inventory = store.record_kubernetes_node_inventory(report, received_at=received)
    return KubernetesNodeInventoryReceipt.of(
        inventory,
        recorded=(
            inventory.observed_at == report.observed_at
            and inventory.reporter == report.executor_id
        ),
    )


def merge_node_inventory(
    previous: KubernetesNodeInventory | None,
    report: KubernetesNodeInventoryReport,
    *,
    received_at: datetime,
) -> KubernetesNodeInventory:
    """The cluster's record once ``report`` is taken into account.

    A report no newer than the stored one (the sibling executor's, or a retry)
    leaves the record as it is. A report after a gap longer than
    ``NODE_INVENTORY_MAX_GAP`` starts a new chain that remembers nothing.
    """

    if previous is not None and report.observed_at <= previous.observed_at:
        return previous
    listed = set(report.node_names)
    base: dict[str, Any] = {
        "cluster_id": report.cluster_id,
        "reporter": report.executor_id,
        "observed_at": report.observed_at,
        "received_at": received_at,
        "resource_version": report.resource_version,
        "node_names": list(report.node_names),
        "spare_node_names": list(report.spare_node_names),
    }
    if previous is None or received_at - previous.received_at > NODE_INVENTORY_MAX_GAP:
        return KubernetesNodeInventory(
            **base, chain_started_at=received_at, retained_since=received_at
        )
    spares = set(previous.spare_node_names)
    absent = {
        name: node for name, node in previous.absent.items() if name not in listed
    }
    for name in previous.node_names:
        if name not in listed and name not in absent:
            absent[name] = AbsentNode(since=received_at, was_spare=name in spares)
    cutoff = received_at - NODE_INVENTORY_ABSENCE_RETENTION
    retained_since = previous.retained_since
    if any(node.since < cutoff for node in absent.values()):
        absent = {name: node for name, node in absent.items() if node.since >= cutoff}
        retained_since = max(retained_since, cutoff)
    return KubernetesNodeInventory(
        **base,
        chain_started_at=previous.chain_started_at,
        reports_in_chain=previous.reports_in_chain + 1,
        retained_since=retained_since,
        absent=dict(sorted(absent.items())),
    )


@dataclass(frozen=True)
class KubernetesAbsence:
    """What the record says about one node, as departed-node evidence."""

    node_id: str
    #: The moment since which every report of the chain left the node out.
    absent_since: datetime | None
    #: The latest report: its LIST time (executor) and receipt (control plane).
    observed_at: datetime | None
    received_at: datetime | None
    was_spare: bool
    #: Why the record does not (yet) prove the node absent; None when it does.
    blocker: str | None

    @property
    def proven(self) -> bool:
        return self.blocker is None


def kubernetes_absence(
    inventory: KubernetesNodeInventory | None,
    node_id: str,
    *,
    now: datetime,
    guard: timedelta,
) -> KubernetesAbsence:
    """Judge whether ``inventory`` proves ``node_id`` gone from Kubernetes.

    Required, all of it: a record, fresh on ``now``; the node not in its
    latest report; and the node left out of every report since at least
    ``guard`` before that report. Anything short names the blocker.
    """

    if inventory is None:
        return KubernetesAbsence(node_id, None, None, None, False, NO_REPORT_BLOCKER)
    age = now - inventory.received_at
    if node_id in inventory.node_names:
        return KubernetesAbsence(
            node_id,
            None,
            inventory.observed_at,
            inventory.received_at,
            node_id in inventory.spare_node_names,
            PRESENT_BLOCKER,
        )
    entry = inventory.absent.get(node_id)
    since = entry.since if entry is not None else inventory.retained_since
    was_spare = bool(entry is not None and entry.was_spare)
    blocker: str | None = None
    if age > NODE_INVENTORY_FRESHNESS:
        blocker = (
            f"the cluster's latest Kubernetes node report is "
            f"{age.total_seconds():.0f}s old (limit "
            f"{NODE_INVENTORY_FRESHNESS.total_seconds():.0f}s); Kubernetes "
            "absence is unproven"
        )
    else:
        span = inventory.received_at - since
        if span < guard:
            blocker = (
                f"node has been absent from Kubernetes reports for "
                f"{max(0.0, span.total_seconds()):.0f}s of the "
                f"{guard.total_seconds():.0f}s guard"
            )
    return KubernetesAbsence(
        node_id,
        since,
        inventory.observed_at,
        inventory.received_at,
        was_spare,
        blocker,
    )
