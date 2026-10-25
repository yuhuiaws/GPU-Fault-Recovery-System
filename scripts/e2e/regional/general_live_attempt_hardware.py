"""GF-GENERAL-LIVE-029 ``hardware_chain_continues``: a trusted hardware fault
on the node of a managed job whose restart budget is already spent.

The spec (``docs/components/general-fault-live-acceptance.md`` 029): an
exhausted budget withholds RESTART_WORKLOAD *only*; the established STOP, the
necessary hardware recovery and its validation still run with real receipts,
one notification is saved, and the final incident state is consistent with the
node's isolation. The product implements that in
``execution/restart_budget_preflight.withhold_exhausted_restart`` (逻辑 4): the
restart's FAILED record (``RESTART_BUDGET_EXHAUSTED``) is written at claim time,
its index superseded, the repair steps go ahead, and the workflow ends FAILED
with the incident ESCALATED once RESTORE_SCHEDULING has run
(``executor._fail_with_reason``).

Mechanism (every piece reused, nothing reinvented). Run order: the managed
job is created first -- the runner's managed-job contract
(``tests/regional/test_general_live_managed_job_start.py``): its ``kubectl
create`` is the variant's first cluster mutation and no store record is read
before it -- then the read-only GPU survey, then the ledger seed, then the
fault; the ledger cleanup is registered before the job so the LIFO stack still
releases it after the job is gone:

* the budget: ``ExhaustedRestartBudget`` builds on lane SMALL's
  ``SeededRestartBudget`` -- same ``gl29<tag>-<digest>-a<attempt>`` test job id,
  same proven cleanup through the product API -- and seeds that job's ledger to
  ``restart_count == budget`` with ``store.reserve_job_restart`` (the call the
  executor's claim preflight makes), then proves exhaustion with one more
  reservation the product must refuse. The ledger row *is* the managed job's:
  the job is created under that id;
* the job: ``ManagedTrainingJob``, the single-node warm-spare trainer (real
  CUDA load on the node's eight GPUs, heartbeats), registered with
  ``restart_budget`` 1 and waited RUNNING, so the kernel XID meets an ACTIVE
  sole attempt and the chain compiles STOP_WORKLOADS and RESTART_WORKLOAD
  (``app/ingest/faults._enrich_workload_context``,
  ``orchestration/workflow_builder.catalog_operations``);
* the fault: lane K's kmsg kernel-source replay (``general_live_kernel_xid.
  inject_kernel_xid`` with this module's reset-grade catalog gate,
  ``KERNEL_SOURCE_REPLAY``) delivers one reviewed ``NVRM: Xid`` line naming the
  target GPU's PCI BDF; the real kernel collector, the real NVIDIA_KERNEL
  catalog, GPU-identity and workload enrichment and the real orchestrator do
  the rest. The window closes as soon as the decision exists.

The XID: **46** ``ROBUST_CHANNEL_GPU_TIMEOUT_ERROR`` ("GPU stopped processing"),
pinned catalog ``src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml``
lines 462-473: ``products: [A100, H100, B100, GB200]``, ``immediateAction:
RESET_GPU``, ``investigatoryAction: CONTACT_SUPPORT``, no xid154Linkage. The
engine maps RESET_GPU directly (``policy/models.DIRECT_ACTION_MAP``,
``engine._direct_resolution``) with GPU containment and no pre-actions, so the
compiled plan is the plain single-GPU reset chain. It is the code the product's
live reset chain was proven with on this hardware (GF-REGIONAL-DESTR-001
``run_destr001_gpu_reset.py`` writes XID 46; ``app/ingest/faults.
_enrich_gpu_identity`` records a live eks-managed XID 46 reset). The line is the
reviewed ``xid_message`` format lane K already replays live -- never an invented
payload; the NVLink5 register formats (XID 144-150) are not involved.

Contrast with GF-REGIONAL-ISO-007 (never run live): a *software* XID 11 replay
against a job with budget 0, graded "whole workflow FAILED on
RESTART_BUDGET_EXHAUSTED, exactly one FAILED step, no remote restart". The
spec'd difference this variant asserts explicitly (``contrast_errors``): here
the fault is hardware-grade, the budget was *consumed* (1/1 on the ledger), and
the workflow still ends FAILED on the budget -- but STOP_WORKLOADS, RESET_GPU
and VALIDATE_GPU carry SUCCEEDED receipts, RESTART_WORKLOAD is the only FAILED
step, the node is restored by the product's own RESTORE_SCHEDULING, and the
incident is ESCALATED (not QUARANTINED).

Grading reads only the product's records: the ``FaultPolicyDecision``
(EXECUTABLE RESET_GPU, KERNEL_LOG, the target GPU), the workflow's step
executions, remote commands, the incident state against the node snapshot, the
BUDGET_EXHAUSTED notification, the attempt's completion records (terminal
STOPPED by the incident, NO_ACTION, no passive containment, ledger unchanged)
and fresh GPU telemetry after the reset. FAIL when the hardware chain is
skipped or RESTART_WORKLOAD runs; INCONCLUSIVE when the reviewed line never
became a KERNEL_LOG event.

Cleanup is LIFO: the kmsg window (closed early; its own ``systemd-run
--on-active`` deadman restores the collector if the runner dies), the managed
job and its prewarm Pods, then the seeded ledger (release this run's
reservations, remove the emptied row; a row that is not provably gone raises
and fails the variant). The ledger row and the job are control-plane/cluster
test objects: a run that dies leaves a ``gl29hc-…`` row that withholds restarts
of that test job only, and the next attempt refuses it (NOT_RUN) while its
cleanup removes it -- the same residue rule as lane SMALL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e2e.regional.general_live_common import (
    activity,
    restore_incidents,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    GpuSurvey,
    run_gpu_survey,
    state_now,
)
from scripts.e2e.regional.general_live_kernel_xid import (
    LINK_XIDS,
    SITE_FAMILY,
    WORKFLOW_XIDS,
    CatalogEntry,
    catalog_entry,
    inject_kernel_xid,
    kmsg_pci_bdf,
    product_family,
    wait_xid_decisions,
)
from scripts.e2e.regional.general_live_registry import LIVE_REPLAY, VariantSpec
from scripts.e2e.regional.general_live_restart_budget import (
    SeededRestartBudget,
    cleanup_errors,
)
from scripts.e2e.regional.general_live_store import attempt_records
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    executions,
    final_execution,
    node_isolated,
    operations,
    parse_time,
    subsequence,
    verdict,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-029"
VARIANT = "hardware_chain_continues"
#: The reviewed kernel XID (see the module docstring for the catalog citation).
HARDWARE_XID = 46
CATALOG_CITATION = (
    "src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml lines 462-473: "
    "xid 46 ROBUST_CHANNEL_GPU_TIMEOUT_ERROR 'GPU stopped processing', products "
    "A100/H100/B100/GB200, immediateAction RESET_GPU, investigatoryAction "
    "CONTACT_SUPPORT"
)
#: The only catalog Immediate Action the hardware chain may be driven with: the
#: engine maps it to RESET_GPU directly, GPU containment, no pre-actions.
RESET_GRADE_ACTIONS = frozenset({"RESET_GPU"})
#: XID 95 is RESET_GPU too, but the engine adds MARK_UNSCHEDULABLE/STOP_WORKLOAD
#: pre-actions with ALL_APPLICATIONS containment: not the plain reset chain.
PRE_ACTION_XIDS = frozenset({95})
#: The job's restart budget; the ledger is seeded to exactly this many.
BUDGET = 1
JOB_TAG = "hc"
GPU_COUNT = 8
GPU_DUTY_PERCENT = 30
SINGLE_NODE_MANIFEST = "single-node-warm-spare-pytorchjob.yaml"
WITHHELD_REASON = "RESTART_BUDGET_EXHAUSTED"
#: The compiled reset chain on a node with an ACTIVE managed attempt
#: (``workflow_builder.catalog_operations`` for RESET_GPU), in plan order.
HARDWARE_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
)
#: Receipts the spec names: these must have SUCCEEDED, whatever the plan.
REAL_RECEIPTS = ("STOP_WORKLOADS", "RESET_GPU", "VALIDATE_GPU")
#: Repair steps that must have SUCCEEDED when the plan carries them.
REPAIR_RECEIPTS = (
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESTORE_GPU_SERVICES",
    "RESTORE_SCHEDULING",
)
#: Operations no workflow of the node may plan for a single-GPU XID.
FORBIDDEN_OPERATIONS = frozenset(
    {
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_VM",
        "QUARANTINE",
    }
)
RUNNING_BUDGET_SECONDS = 900
#: Stop, quiesce, one reset with busy retries, restore, validate, uncordon.
CHAIN_BUDGET_SECONDS = 1800
TERMINAL_BUDGET_SECONDS = 600
#: Kept free at the end of the window for the LIFO cleanup.
CLEANUP_RESERVE_SECONDS = 420
LABEL = "029-hardware-chain"

#: Seeds one test job's restart ledger to exhausted through the product's own
#: ledger API in an API Pod (``cpu_python``, ``attempts=1``). Refuses a job id
#: outside the ``gl29`` test prefix and a ledger row that already exists; the
#: last reservation is the product's own refusal of a restart past the budget.
SEED_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

TEST_PREFIX = "gl29"


def ledger(store, cluster_id, job_id):
    try:
        return store.get_restart_budget(cluster_id, job_id).model_dump(mode="json")
    except NotFoundError:
        return None


def seed(store, cluster_id, job_id, budget, workflow_id):
    if not job_id.startswith(TEST_PREFIX):
        raise ValueError(f"refusing to seed a restart ledger outside the test prefix: {job_id}")
    if budget < 1:
        raise ValueError("an exhausted ledger needs a budget of at least 1")
    before = ledger(store, cluster_id, job_id)
    document = {
        "mode": "seed",
        "cluster_id": cluster_id,
        "job_id": job_id,
        "budget": budget,
        "workflow_id": workflow_id,
        "ledger": {"before": before},
    }
    if before is not None:
        document["refused"] = "preexisting ledger: an earlier attempt left the test row"
        return document
    reservations = []
    for index in range(budget):
        reservation = f"{workflow_id}/{index}/RESTART_WORKLOAD"
        state, reserved = store.reserve_job_restart(cluster_id, job_id, budget, reservation)
        reservations.append({
            "reservation_id": reservation,
            "reserved": reserved,
            "restart_count": state.restart_count,
            "reservation_ids": list(state.reservation_ids),
        })
    state, reserved = store.reserve_job_restart(
        cluster_id, job_id, budget, f"{workflow_id}/exhausted-check/RESTART_WORKLOAD"
    )
    document["reservations"] = reservations
    document["exhausted_check"] = {
        "reserved": reserved,
        "restart_count": state.restart_count,
        "reservation_ids": list(state.reservation_ids),
    }
    document["ledger"]["after"] = ledger(store, cluster_id, job_id)
    return document


def main(argv):
    mode, cluster_id, job_id, budget, workflow_id = argv
    if mode != "seed":
        raise ValueError(f"unknown seed probe mode {mode!r}")
    store = ApplicationContext.from_environment().store
    return seed(store, cluster_id, job_id, int(budget), workflow_id)


print(json.dumps(main(sys.argv[1:]), sort_keys=True, default=str))
"""

#: Read-only: the notifications the product saved for one incident, with
#: their delivery status (the budget mail does not name the node, so the node
#: activity probe's text filter would miss it).
INCIDENT_NOTIFICATIONS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, incident_id = sys.argv[1:3]
store = ApplicationContext.from_environment().store
items = []
for notification in store.list_notifications(limit=1000, newest_first=True):
    if notification.incident_id != incident_id:
        continue
    result = store.get_notification_result(notification.notification_id)
    items.append({
        "notification_id": notification.notification_id,
        "incident_id": notification.incident_id,
        "category": notification.category,
        "deduplication_key": notification.deduplication_key,
        "subject": notification.subject,
        "created_at": notification.created_at.isoformat(),
        "status": result.status.value if result is not None else None,
    })
print(json.dumps(
    {"cluster_id": cluster_id, "incident_id": incident_id, "notifications": items},
    sort_keys=True,
    default=str,
))
"""


# --------------------------------------------------------------------------- #
# The pinned catalog: which XIDs may drive the chain
# --------------------------------------------------------------------------- #
def reset_grade_errors(
    entry: CatalogEntry | None, *, xid: int, family: str = SITE_FAMILY
) -> list[str]:
    """Why ``xid`` may not drive the live hardware chain (empty: it may)."""

    if entry is None:
        return [
            f"XID {xid} is absent from the pinned catalog (it would be SITE_SAFETY)"
        ]
    errors = []
    if entry.immediate_action not in RESET_GRADE_ACTIONS:
        errors.append(
            f"XID {xid} ({entry.mnemonic}) Immediate Action is "
            f"{entry.immediate_action}; the hardware chain needs "
            f"{sorted(RESET_GRADE_ACTIONS)}"
        )
    if family not in entry.products:
        errors.append(
            f"XID {xid} does not apply to the {family} family ({list(entry.products)});"
            " the product gate would quarantine the node"
        )
    if xid in LINK_XIDS:
        errors.append(f"XID {xid} joins the PCIe replay composite; not a plain reset")
    if xid in WORKFLOW_XIDS:
        errors.append(f"XID {xid} has its own workflow or companion resolver")
    if xid in PRE_ACTION_XIDS:
        errors.append(
            f"XID {xid} carries engine pre-actions (ALL_APPLICATIONS containment); "
            "not the single-GPU reset chain"
        )
    return errors


def require_reset_grade(xid: int) -> CatalogEntry:
    """The catalog entry of ``xid``, or NOT_RUN before any mutation."""

    entry = catalog_entry(xid)
    errors = reset_grade_errors(entry, xid=xid)
    if errors or entry is None:
        raise VariantNotRun("FIXTURE_NOT_READY", "; ".join(errors))
    return entry


def target_gpu(survey: GpuSurvey, entry: CatalogEntry) -> tuple[str, str]:
    """The highest-index GPU a kernel line can name, in the catalog's family.

    Returns ``(gpu_uuid, kernel PCI BDF)``. The managed job holds every GPU of
    the node, so any of them is the job's; the highest index mirrors the DCGM
    lanes' ``GpuSurvey.target``.
    """

    for gpu in reversed(survey.gpus):
        pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
        if pci_bdf is None:
            continue
        family = product_family(survey.model(gpu))
        if family not in entry.products:
            raise VariantNotRun(
                "CAPABILITY_ABSENT",
                f"{gpu} is {survey.model(gpu)!r} (family {family}); XID {entry.xid} "
                f"applies to {list(entry.products)}",
            )
        return gpu, pci_bdf
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no GPU on the node reports a PCI BDF; a kernel XID cannot name one",
    )


# --------------------------------------------------------------------------- #
# The exhausted ledger
# --------------------------------------------------------------------------- #
class ExhaustedRestartBudget(SeededRestartBudget):
    """The managed job's restart ledger, seeded to exhausted through the product API.

    Reuses the test job identity (``gl29hc-<digest>-a<attempt>``; the managed
    job is created under it, so the row is the job's) and the proven cleanup
    (release this run's reservations, remove the emptied row). The seed takes
    ``budget`` reservations under this run's own workflow id -- the same
    ``reserve_job_restart`` the executor's claim preflight calls -- and one
    more that the product must refuse. ``register`` puts the cleanup on the
    LIFO stack before the managed job is started (so the ledger is released
    only after the job is gone); ``exhaust`` registers too, before its one
    write. The cleanup is a no-op while nothing was seeded -- it reads no
    store record -- and raises when a seeded row is not provably gone, so the
    runner's teardown records a failed undo.
    """

    def __init__(
        self, ctx: VariantContext, *, tag: str = JOB_TAG, budget: int = BUDGET
    ) -> None:
        super().__init__(ctx, tag=tag, budget=budget)
        if self.budget < 1:
            raise ValueError("an exhausted ledger needs a budget of at least 1")
        self._cleanup_armed = False
        self._seeded = False

    @property
    def seed_reservation_ids(self) -> list[str]:
        return [
            f"{self.workflow_id}/{index}/RESTART_WORKLOAD"
            for index in range(self.budget)
        ]

    @property
    def seeded(self) -> bool:
        return self._seeded

    def register(self) -> None:
        """Put the ledger cleanup on the LIFO stack once (before the job)."""

        if not self._cleanup_armed:
            self.ctx.cleanup.push(self.cleanup_name, self.cleanup)
            self._cleanup_armed = True

    def exhaust(self) -> dict[str, Any]:
        """Seed the ledger to ``budget``/``budget``; NOT_RUN on a preexisting row."""

        self.register()
        self._seeded = True
        value = self.ctx.regional.cpu_python(
            SEED_PROBE,
            "seed",
            self.cluster_id,
            self.job_id,
            str(self.budget),
            self.workflow_id,
            attempts=1,
            timeout=240,
        )
        self.ctx.evidence.write("budget-seed", value)
        refused = value.get("refused")
        if refused:
            raise VariantNotRun(
                "FIXTURE_NOT_READY",
                f"{refused} ({self.cluster_id}/{self.job_id}); the cleanup removes it",
            )
        return value

    def cleanup(self) -> dict[str, Any]:
        if not self._seeded:
            # Registered ahead of the job, never written: nothing to release,
            # and no store read in a teardown that may run with no API Pod.
            return {"mode": "cleanup", "skipped": "nothing seeded", "clean": True}
        value = super().cleanup()
        errors = cleanup_errors(value)
        if errors:
            raise RuntimeError(
                f"restart ledger cleanup of {self.cluster_id}/{self.job_id}: "
                + "; ".join(errors)
            )
        return value


def seed_errors(
    document: dict[str, Any], *, budget: int, reservation_ids: list[str]
) -> list[str]:
    """Why the seed did not leave the product's ledger exhausted (empty: it did)."""

    if document.get("refused"):
        return [f"the seed probe refused: {document['refused']}"]
    errors: list[str] = []
    results = document.get("reservations") or []
    if len(results) != budget:
        errors.append(f"{len(results)} seed reservations, expected {budget}")
    for item in results:
        if item.get("reserved") is not True:
            errors.append(f"seed reservation {item.get('reservation_id')} was refused")
    after = (document.get("ledger") or {}).get("after")
    if not isinstance(after, dict):
        errors.append("no ledger row after the seed")
    else:
        if after.get("restart_count") != budget or after.get("budget") != budget:
            errors.append(
                f"ledger after the seed is {after.get('restart_count')}/"
                f"{after.get('budget')}, expected {budget}/{budget}"
            )
        if after.get("reservation_ids") != list(reservation_ids):
            errors.append(
                f"ledger holds {after.get('reservation_ids')}, expected the seed "
                f"ids {reservation_ids}"
            )
    check = document.get("exhausted_check") or {}
    if check.get("reserved") is not False:
        errors.append("the product granted a reservation past the seeded budget")
    elif check.get("restart_count") != budget:
        errors.append("the refused check changed the ledger")
    return errors


# --------------------------------------------------------------------------- #
# The managed training job (real GPU load, restart budget 1)
# --------------------------------------------------------------------------- #
class ManagedTrainingJob:
    """A managed single-node training job that really holds the node's GPUs.

    The warm-spare trainer manifest (torchrun on ``gpu_count`` GPUs, fp16
    load at ``GPU_DUTY_PERCENT``, heartbeats), node-pinned and registered with
    ``restart_budget`` through ``gpu-training-submit``; ``start`` returns once
    the Pod is Running and heartbeating, so the kernel XID meets an ACTIVE
    attempt whose stop leaves real CUDA clients for VERIFY_NO_GPU_CLIENTS to
    see gone. The product's STOP_WORKLOADS ends it (SIGTERM, exit 143).
    """

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        restart_budget: int = BUDGET,
        gpu_count: int = GPU_COUNT,
    ) -> None:
        from scripts.e2e.regional.acceptance_gpu_load import set_gpu_duty_percent
        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
            render_node_pinned_manifest,
        )
        from scripts.e2e.regional.training_topology import fit_gpu_count

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed training job needs --site-file"
            )
        source = (
            Path(__file__).with_name("manifests") / "training" / SINGLE_NODE_MANIFEST
        )
        rendered = render_node_pinned_manifest(
            source, ctx.case_dir / f"{name}.source.yaml", node=ctx.settings.node
        )
        document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        name_managed_workload(document, name)
        set_gpu_duty_percent(document, GPU_DUTY_PERCENT)
        fit_gpu_count(document, gpu_count)
        document.setdefault("metadata", {}).setdefault("labels", {}).update(
            {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": f"{name}-a001"}
        )
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.name = name
        self.node = ctx.settings.node
        self.attempt_id = f"{name}-a001"
        self.restart_budget = restart_budget
        self.gpu_count = gpu_count
        self.prewarm = ImagePrewarmFixture(
            ctx.regional,
            case_id=ctx.spec.key,
            run_id=f"{ctx.run_id}-{name}",
            state_path=ctx.case_dir / f"{name}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            ctx.regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=ctx.settings.site_file,
                job_id=name,
                attempt_id=self.attempt_id,
                restart_budget=restart_budget,
                expected_pods=1,
                expected_gpu_count=gpu_count,
            ),
            state_path=ctx.case_dir / f"{name}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        # The node the manifest is pinned to, never the cluster id (003's
        # prewarm lesson, live 2026-10-24).
        self.prewarm.create([self.node])
        self.fixture.submit()
        return self.fixture.wait_running(timeout_seconds=RUNNING_BUDGET_SECONDS)

    def identity(self) -> dict[str, Any]:
        pods = self.fixture.pods()
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "restart_budget": self.restart_budget,
            "pods": pods,
            "pod_uids": sorted(str(pod.get("uid")) for pod in pods),
        }

    def cleanup(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)


def start_job(ctx: VariantContext, *, name: str, restart_budget: int) -> Any:
    """Create, register on the LIFO stack, start and wait for the job."""

    job = ManagedTrainingJob(ctx, name=name, restart_budget=restart_budget)
    ctx.cleanup.push(f"job-{name}", job.cleanup)
    running = job.start()
    ctx.evidence.write("job-running", {"snapshot": running, **job.identity()})
    return job


# --------------------------------------------------------------------------- #
# Grading: the product's own records
# --------------------------------------------------------------------------- #
def decision_errors(
    decisions: list[dict[str, Any]], *, entry: CatalogEntry, gpu_uuid: str
) -> tuple[list[str], dict[str, Any]]:
    """Why the XID did not earn one fresh EXECUTABLE RESET_GPU decision.

    Returns the errors and the decision graded (empty when none qualifies).
    """

    xid = entry.xid
    graded = [item for item in decisions if item.get("decision")]
    if not graded:
        return [f"XID {xid} was ingested but has no FaultPolicyDecision"], {}
    fresh = [item for item in graded if not item.get("duplicate")]
    if not fresh:
        return (
            [
                f"every decision for XID {xid} is a duplicate of an earlier event "
                "(the reviewed line re-used an event id; re-run on the same boot?)"
            ],
            graded[0],
        )
    errors: list[str] = []
    if len(fresh) > 1:
        errors.append(
            f"{len(fresh)} non-duplicate decisions for XID {xid}; the reviewed "
            "record's copies did not dedup to one event"
        )
    item = fresh[0]
    if item.get("event_source") not in (None, "KERNEL_LOG"):
        errors.append(
            f"XID {xid} arrived as {item.get('event_source')}, not the trusted "
            "KERNEL_LOG source"
        )
    if item.get("official_action") != entry.immediate_action:
        errors.append(
            f"XID {xid} official action is {item.get('official_action')}, the pinned "
            f"catalog {entry.catalog_version} says {entry.immediate_action}"
        )
    if item.get("disposition") != "EXECUTABLE":
        errors.append(
            f"XID {xid} disposition is {item.get('disposition')}, expected EXECUTABLE "
            f"for catalog action {entry.immediate_action}"
        )
    if item.get("action") != "RESET_GPU":
        errors.append(
            f"XID {xid} earned recovery action {item.get('action')}, expected RESET_GPU"
        )
    if item.get("safety_action"):
        errors.append(
            f"XID {xid} earned safety action {item.get('safety_action')}: the node "
            "would be isolated, not repaired"
        )
    if item.get("correlated_event_id"):
        errors.append(
            f"XID {xid} was correlated into event {item.get('correlated_event_id')}"
        )
    if gpu_uuid and item.get("gpu_uuid") != gpu_uuid:
        errors.append(
            f"XID {xid} resolved to GPU {item.get('gpu_uuid')}, expected the target "
            f"GPU {gpu_uuid} named by its PCI BDF"
        )
    if not item.get("incident_id"):
        errors.append(f"XID {xid} decision names no incident")
    if not item.get("workflow_request_id"):
        errors.append(
            f"XID {xid} decision minted no workflow; the hardware chain was skipped"
        )
    return errors, item


def chain_receipt_errors(workflow: dict[str, Any]) -> list[str]:
    """The hardware chain ran with real receipts and ended FAILED on the budget."""

    if not workflow:
        return ["the RESET_GPU decision's workflow is not in the product's records"]
    errors: list[str] = []
    steps = operations(workflow)
    if not subsequence(HARDWARE_CHAIN, steps):
        errors.append(
            f"workflow steps {steps} lack the hardware chain {list(HARDWARE_CHAIN)}"
        )
    forbidden = sorted(set(steps) & FORBIDDEN_OPERATIONS)
    if forbidden:
        errors.append(f"workflow {workflow.get('request_id')} plans {forbidden}")
    for operation in REAL_RECEIPTS:
        execution = final_execution(workflow, operation)
        if execution is None:
            errors.append(
                f"{operation} has no receipt: the hardware chain was skipped or "
                "never reached it"
            )
        elif execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{operation} ended {execution.get('status')}: "
                f"{str(execution.get('error') or '')[:200]}"
            )
    for operation in REPAIR_RECEIPTS:
        if operation not in steps:
            continue
        execution = final_execution(workflow, operation)
        if execution is None or execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{operation} was planned but ended "
                f"{(execution or {}).get('status')}: "
                f"{str((execution or {}).get('error') or '')[:200]}"
            )
    status = str(workflow.get("status"))
    if status not in TERMINAL_WORKFLOWS:
        errors.append(
            f"workflow is still {status}; the chain did not settle within the budget"
        )
    elif status != "FAILED":
        errors.append(
            f"workflow ended {status}; a withheld restart ends it FAILED (the job "
            "was stopped and not restarted; someone has to resubmit it)"
        )
    reason = str(workflow.get("terminal_failure_reason") or "")
    if "budget" not in reason.lower():
        errors.append(
            f"terminal failure reason {reason!r} does not name the restart budget"
        )
    return errors


def withheld_errors(
    workflow: dict[str, Any], commands: list[dict[str, Any]], *, budget: int
) -> list[str]:
    """RESTART_WORKLOAD was withheld with the budget reason, and only withheld."""

    if not workflow:
        return []
    if "RESTART_WORKLOAD" not in operations(workflow):
        return [
            "the chain compiled no RESTART_WORKLOAD step (the job's restart context "
            "did not reach the plan)"
        ]
    records = executions(workflow, "RESTART_WORKLOAD")
    if not records:
        return ["RESTART_WORKLOAD has no record: it was neither withheld nor run"]
    errors: list[str] = []
    if len(records) != 1:
        errors.append(
            f"{len(records)} RESTART_WORKLOAD records (expected the one withheld record)"
        )
    for item in records:
        status = item.get("status")
        details = item.get("details") or {}
        if status == "SUCCEEDED":
            errors.append("RESTART_WORKLOAD succeeded despite the exhausted budget")
            continue
        if status != "FAILED":
            errors.append(
                f"RESTART_WORKLOAD is {status}; a withheld restart is recorded FAILED "
                "before any adapter runs"
            )
            continue
        if details.get("reason") != WITHHELD_REASON:
            errors.append(
                f"RESTART_WORKLOAD failed with {details.get('reason')!r}, expected "
                f"{WITHHELD_REASON}"
            )
        if (
            details.get("restart_count") != budget
            or details.get("restart_budget") != budget
        ):
            errors.append(
                f"the withheld record says {details.get('restart_count')}/"
                f"{details.get('restart_budget')}, expected {budget}/{budget}"
            )
        if not details.get("notification_id"):
            errors.append("the withheld record names no BUDGET_EXHAUSTED notification")
    superseded = workflow.get("superseded_step_indexes") or []
    withheld = [
        item.get("step_index") for item in records if item.get("status") == "FAILED"
    ]
    if withheld and not any(index in superseded for index in withheld):
        errors.append(
            "the withheld step index is not superseded; the step loops could still "
            "run it"
        )
    for command in commands or []:
        steps = [
            command.get("step") or {},
            *[entry.get("step") or {} for entry in command.get("batched_steps") or []],
        ]
        if any(step.get("operation") == "RESTART_WORKLOAD" for step in steps):
            errors.append(
                f"remote command {command.get('command_id')} carries "
                "RESTART_WORKLOAD despite the exhausted budget"
            )
    return errors


def contrast_errors(workflow: dict[str, Any]) -> list[str]:
    """The spec'd difference from GF-REGIONAL-ISO-007, asserted explicitly.

    ISO-007 (software XID 11, budget 0, never run live) grades a whole workflow
    FAILED on RESTART_BUDGET_EXHAUSTED with that one FAILED step and no remote
    restart. Here the fault is hardware-grade: the workflow still ends FAILED on
    the budget and RESTART_WORKLOAD is still the only FAILED step, but RESET_GPU
    must have executed -- a chain that failed on the budget without resetting
    is ISO-007's shape, not this variant's.
    """

    if not workflow:
        return []
    errors: list[str] = []
    records = workflow.get("step_executions") or []
    failed = sorted(
        {
            str(item.get("operation"))
            for item in records
            if item.get("status") == "FAILED"
        }
    )
    if failed != ["RESTART_WORKLOAD"]:
        errors.append(
            f"FAILED steps are {failed}; as in ISO-007 the budget may fail "
            "RESTART_WORKLOAD alone, and unlike ISO-007 every repair step must carry "
            "a SUCCEEDED receipt"
        )
    executed = {str(item.get("operation")) for item in records}
    if "RESET_GPU" not in executed:
        errors.append(
            "RESET_GPU never executed: this is ISO-007's shape (the workflow FAILED "
            "on the budget) rather than a hardware chain that continued"
        )
    return errors


def incident_errors(
    activity_document: dict[str, Any],
    decision: dict[str, Any],
    node_after: dict[str, Any],
    *,
    xid: int,
) -> list[str]:
    """One incident for the XID, ESCALATED, with the node no longer isolated."""

    incident_id = decision.get("incident_id")
    incidents = activity_document.get("incidents") or []
    mine = [item for item in incidents if item.get("incident_id") == incident_id]
    errors: list[str] = []
    if len(mine) != 1:
        errors.append(
            f"incident {incident_id} of the XID decision is not among the node's "
            "incidents since the injection"
        )
    others = [
        str(item.get("incident_id"))
        for item in incidents
        if item.get("incident_id") != incident_id
        and f"xid-{xid}" in str(item.get("event_id") or "")
    ]
    if others:
        errors.append(
            f"{len(others)} further incident(s) for XID {xid}: {others}; one trusted "
            "event is one incident"
        )
    state = mine[0].get("state") if mine else None
    isolated = node_isolated(node_after)
    if state == "ESCALATED":
        if isolated:
            errors.append(
                "incident ESCALATED but the node is still isolated (cordon, "
                "ownership annotation or gpu-fault taint): RESTORE_SCHEDULING's "
                "receipt is contradicted"
            )
    elif state == "QUARANTINED":
        errors.append(
            "incident QUARANTINED: the chain left the node isolated instead of "
            "restoring scheduling and escalating the unrestarted job"
        )
    else:
        errors.append(
            f"incident is {state}, expected ESCALATED (GPU repaired, job stopped and "
            "not restarted, an operator must resubmit)"
        )
    return errors


def notification_errors(
    notifications: list[dict[str, Any]],
    *,
    notification_id: str | None,
    incident_id: str | None,
) -> list[str]:
    """Exactly one saved BUDGET_EXHAUSTED notification, the one the record names."""

    budget_mails = [
        item
        for item in notifications
        if "restart-budget-exhausted" in str(item.get("deduplication_key") or "")
    ]
    errors: list[str] = []
    if len(budget_mails) != 1:
        errors.append(
            f"{len(budget_mails)} restart-budget-exhausted notifications for incident "
            f"{incident_id} (expected exactly one)"
        )
    if notification_id and not any(
        item.get("notification_id") == notification_id for item in notifications
    ):
        errors.append(
            f"the withheld record's notification {notification_id} is not saved for "
            f"incident {incident_id}"
        )
    for item in budget_mails:
        if item.get("status") == "FAILED":
            errors.append(
                f"budget notification {item.get('notification_id')} delivery FAILED"
            )
    return errors


def attempt_errors(
    records: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    budget: int,
    reservation_ids: list[str],
) -> list[str]:
    """The attempt ended STOPPED by the incident, NO_ACTION, ledger untouched."""

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    errors: list[str] = []
    terminal = (entry.get("event") or {}).get("terminal_status")
    if terminal != "STOPPED":
        errors.append(
            f"the attempt's terminal is {terminal}, expected STOPPED (stopped by the "
            "hardware incident's STOP_WORKLOADS)"
        )
    decision = entry.get("decision") or {}
    if decision and decision.get("status") != "NO_ACTION":
        errors.append(
            f"the completion decision is {decision.get('status')}; an "
            "incident-initiated stop is NO_ACTION (no second recovery)"
        )
    if entry.get("containment_incident"):
        errors.append(
            "a passive TrainingAttemptFailureDetected containment incident was "
            "opened; the stop was read as an application failure"
        )
    for item in entry.get("recovery") or []:
        workflow = item.get("workflow") or {}
        incident = item.get("incident") or {}
        if incident.get("job_id") not in (None, job_id):
            errors.append(
                f"a recovery incident names another job {incident.get('job_id')}"
            )
        for execution in executions(workflow, "RESTART_WORKLOAD"):
            if execution.get("status") == "SUCCEEDED":
                errors.append(
                    f"RESTART_WORKLOAD succeeded in workflow {workflow.get('request_id')}"
                    " despite the exhausted budget"
                )
    ledger = records.get("restart_budget")
    if not isinstance(ledger, dict):
        errors.append("the job's restart ledger row is gone before cleanup")
    elif ledger.get("restart_count") != budget or ledger.get("reservation_ids") != list(
        reservation_ids
    ):
        errors.append(
            f"ledger is {ledger.get('restart_count')} {ledger.get('reservation_ids')}, "
            f"expected the seeded {budget} {reservation_ids}: a withheld restart "
            "reserves nothing"
        )
    return errors


def gpu_health_errors(
    state: dict[str, Any], *, reset_execution: dict[str, Any] | None
) -> list[str]:
    """The GPU is back: no active finding, fresh accepted telemetry after the reset."""

    errors: list[str] = []
    active = sorted(
        str(item.get("canonical_name")) for item in state.get("active_findings") or []
    )
    if active:
        errors.append(f"active GPU findings after the chain: {active}")
    reset_at = parse_time((reset_execution or {}).get("updated_at"))
    if reset_at is None:
        return errors
    fresh = [
        item
        for item in state.get("batches") or []
        if (moment := parse_time(item.get("observed_at"))) is not None
        and moment > reset_at
    ]
    if not fresh:
        errors.append(
            "no accepted GPU_METRICS batch after the RESET_GPU receipt; the GPU's "
            "telemetry did not come back"
        )
    return errors


def display_stack_errors(reset_execution: dict[str, Any] | None) -> list[str]:
    """A reset that released the idle display stack must have brought it back.

    The node agent unloads idle ``nvidia_drm``/``nvidia_modeset`` to free a
    busy GPU and reloads them after the reset (``node_agent/operations/
    display_stack.py``); a stack still missing is the operator's remedy, and a
    node left without it is not restored.
    """

    details = (reset_execution or {}).get("details") or {}
    if details.get("display_stack_restored") is False:
        remedy = details.get("display_stack_remedy") or "reload the display stack"
        return [
            "the reset released the idle display stack and did not bring it back "
            f"(modules missing {details.get('display_modules_missing')}, DRM devices "
            f"missing {details.get('display_drm_devices_missing')}): {remedy}"
        ]
    return []


def result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The variant's result: the fault is replayed, the actions are the product's."""

    return {
        "verdict": verdict(stages),
        "proof_scope": "live-replay",
        "physical_fault_injected": False,
        "hardware_actions_real": True,
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
# The variant
# --------------------------------------------------------------------------- #
def workflow_of(
    document: dict[str, Any], *, workflow_id: str | None, incident_id: str | None
) -> dict[str, Any]:
    for item in document.get("workflows") or []:
        if workflow_id and item.get("request_id") == workflow_id:
            return item
    if incident_id:
        found = workflows_of(document, incident_id)
        if found:
            return found[0]
    return {}


def chain_budget_seconds(ctx: VariantContext) -> float:
    remaining = ctx.remaining_seconds() - CLEANUP_RESERVE_SECONDS
    return max(60.0, min(float(CHAIN_BUDGET_SECONDS), remaining))


def wait_attempt_terminal(
    ctx: VariantContext, job: Any, *, timeout_seconds: float
) -> dict[str, Any]:
    """Poll the attempt's completion records until it has a terminal event."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = attempt_records(
            ctx.regional, ctx.settings.regional.cluster_id, job.name, [job.attempt_id]
        )
        entry = (last.get("attempts") or {}).get(job.attempt_id) or {}
        return last if (entry.get("event") or {}).get("terminal_status") else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=20,
        name="attempt-terminal",
    )
    return value if value is not None else last


def incident_notifications(
    ctx: VariantContext, incident_id: str
) -> list[dict[str, Any]]:
    value = ctx.regional.cpu_python(
        INCIDENT_NOTIFICATIONS_PROBE,
        ctx.settings.regional.cluster_id,
        incident_id,
        timeout=120,
    )
    ctx.evidence.write("incident-notifications", value)
    return list(value.get("notifications") or [])


def validated_restore(ctx: VariantContext, incident_id: str) -> list[str]:
    """Hand a node the chain left isolated back through the product's restore."""

    try:
        restore_incidents(
            ctx,
            [{"incident_id": incident_id}],
            reason=f"{ctx.spec.key} hardware chain acceptance restore",
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a cleanup error
        return [f"validated restore failed: {type(exc).__name__}: {exc}"]
    return []


def hardware_chain_continues(ctx: VariantContext) -> dict[str, Any]:
    entry = require_reset_grade(HARDWARE_XID)
    budget = ExhaustedRestartBudget(ctx)
    # The ledger cleanup is armed before the job (LIFO: released after the job
    # is gone); the job's create is the first cluster mutation and no store
    # record is read before it (the runner's managed-job contract).
    budget.register()
    job = start_job(ctx, name=budget.job_id, restart_budget=budget.budget)
    survey = run_gpu_survey(ctx)
    gpu, pci_bdf = target_gpu(survey, entry)
    seeded = budget.exhaust()
    seed_problems = seed_errors(
        seeded, budget=budget.budget, reservation_ids=budget.seed_reservation_ids
    )
    if seed_problems:
        raise VariantInconclusive(
            "the exhausted ledger was not established: " + "; ".join(seed_problems)
        )
    since = utc_now()
    injection = inject_kernel_xid(
        ctx,
        xid=HARDWARE_XID,
        pci_bdf=pci_bdf,
        label=LABEL,
        require=require_reset_grade,
    )
    decisions = wait_xid_decisions(
        ctx, injection.started, xid=HARDWARE_XID, name="hardware-xid"
    )
    # The real /dev/kmsg is unread while the window is open; close it as soon
    # as the product has decided, before the reset chain runs.
    closed = injection.close()
    if not decisions:
        raise VariantInconclusive(
            f"XID {HARDWARE_XID} never reached the product: the kmsg fixture was "
            "written but the store holds no KERNEL_LOG event for it"
        )
    decision_problems, decision = decision_errors(decisions, entry=entry, gpu_uuid=gpu)
    stages: dict[str, list[str]] = {"decision": decision_problems}
    incident_id = str(decision.get("incident_id") or "") or None
    workflow_id = str(decision.get("workflow_request_id") or "") or None
    settled_document: dict[str, Any] = (
        wait_settled(
            ctx,
            since,
            [incident_id],
            timeout_seconds=chain_budget_seconds(ctx),
            name="hardware-chain",
        )
        if incident_id
        else activity(ctx, since, kinds=("GPU_METRICS",))
    )
    workflow = workflow_of(
        settled_document, workflow_id=workflow_id, incident_id=incident_id
    )
    ctx.evidence.write(
        "hardware-chain",
        {
            "decision": decision,
            "workflow": workflow,
            "incidents": settled_document.get("incidents") or [],
            "commands": settled_document.get("commands") or [],
        },
    )
    node_after = ctx.regional.node_snapshot(ctx.settings.node)
    reset_execution = final_execution(workflow, "RESET_GPU") if workflow else None
    restart_execution = (
        final_execution(workflow, "RESTART_WORKLOAD") if workflow else None
    )
    stages["chain"] = chain_receipt_errors(workflow)
    stages["withheld"] = withheld_errors(
        workflow, list(settled_document.get("commands") or []), budget=budget.budget
    )
    stages["iso007_contrast"] = contrast_errors(workflow)
    stages["incident"] = incident_errors(
        settled_document, decision, node_after, xid=HARDWARE_XID
    )
    notifications = incident_notifications(ctx, incident_id) if incident_id else []
    stages["notification"] = notification_errors(
        notifications,
        notification_id=((restart_execution or {}).get("details") or {}).get(
            "notification_id"
        ),
        incident_id=incident_id,
    )
    records = wait_attempt_terminal(
        ctx,
        job,
        timeout_seconds=min(TERMINAL_BUDGET_SECONDS, chain_budget_seconds(ctx)),
    )
    ctx.evidence.write("attempt-records", records)
    stages["attempt"] = attempt_errors(
        records,
        job_id=job.name,
        attempt_id=job.attempt_id,
        budget=budget.budget,
        reservation_ids=budget.seed_reservation_ids,
    )
    state = state_now(ctx, since)
    stages["gpu_health"] = gpu_health_errors(state, reset_execution=reset_execution)
    stages["display_stack"] = display_stack_errors(reset_execution)
    if node_isolated(node_after) and incident_id:
        stages["restore"] = validated_restore(ctx, incident_id)
    reset_details = (reset_execution or {}).get("details") or {}
    return result(
        stages,
        notes=[
            f"XID {HARDWARE_XID} ({entry.mnemonic}) is RESET_GPU in catalog "
            f"{entry.catalog_version}: {CATALOG_CITATION}",
            "the ledger was seeded to exhausted through the product's reserve API "
            "(restart_count == budget), not by a prior legitimate restart "
            "(budget_consumed proves that path); the product's own refusal of one "
            "more reservation is in the seed receipt",
            "unlike GF-REGIONAL-ISO-007 (software XID 11, budget 0, whole workflow "
            "FAILED), the hardware chain ran with real receipts and only "
            "RESTART_WORKLOAD was withheld; the node is restored by the product's "
            "RESTORE_SCHEDULING and the incident is ESCALATED for the operator",
            "the kernel-source window closed as soon as the decision existed; the "
            "hardware actions are the product's own node actions, not a replay",
        ],
        extra={
            "gpu_uuid": gpu,
            "pci_bdf": pci_bdf,
            "xid": HARDWARE_XID,
            "job_id": job.name,
            "attempt_id": job.attempt_id,
            "restart_budget": budget.budget,
            "seed_reservation_ids": budget.seed_reservation_ids,
            "decision": decision,
            "workflow_request_id": workflow_id,
            "incident_id": incident_id,
            "workflow_status": workflow.get("status"),
            "terminal_failure_reason": workflow.get("terminal_failure_reason"),
            "gpu_reset_busy_attempts": reset_details.get("gpu_reset_busy_attempts"),
            "display_stack_restored": reset_details.get("display_stack_restored"),
            "kernel_replay": {**injection.evidence(), "closed": closed},
        },
    )


VARIANTS = [
    VariantSpec(
        case_id=CASE,
        variant=VARIANT,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="H",
        flags="X",
        approvals=("GPU_RESET", "KERNEL_SOURCE_REPLAY"),
        injection=(
            "seed the managed test job's restart ledger to exhausted (1/1) through "
            "the product's reserve API, run the job on the node's GPUs, then replay "
            "one reviewed kernel XID 46 (catalog RESET_GPU) naming one of its GPUs "
            "through the kernel collector's kmsg source"
        ),
        expected=(
            "EXECUTABLE RESET_GPU decision; FREEZE -> MARK_UNSCHEDULABLE -> "
            "STOP_WORKLOADS -> QUIESCE -> VERIFY_NO_GPU_CLIENTS -> RESET_GPU -> "
            "RESTORE_GPU_SERVICES -> VALIDATE_GPU -> RESTORE_SCHEDULING with SUCCEEDED "
            "receipts; RESTART_WORKLOAD alone FAILED(RESTART_BUDGET_EXHAUSTED 1/1) "
            "with one notification and no remote restart; workflow FAILED on the "
            "budget, incident ESCALATED, node uncordoned; attempt STOPPED/NO_ACTION; "
            "ledger unchanged; fresh clean GPU telemetry"
        ),
        cleanup=(
            "close the kmsg window (deadman-backed), delete the test job and its "
            "prewarm Pods, release the seeded reservations and remove the emptied "
            "test ledger row; validated restore if the node were left isolated"
        ),
        max_minutes=60,
        execute=hardware_chain_continues,
    ),
]

__all__ = [
    "BUDGET",
    "CATALOG_CITATION",
    "FORBIDDEN_OPERATIONS",
    "HARDWARE_CHAIN",
    "HARDWARE_XID",
    "INCIDENT_NOTIFICATIONS_PROBE",
    "PRE_ACTION_XIDS",
    "REAL_RECEIPTS",
    "REPAIR_RECEIPTS",
    "RESET_GRADE_ACTIONS",
    "SEED_PROBE",
    "VARIANTS",
    "WITHHELD_REASON",
    "ExhaustedRestartBudget",
    "ManagedTrainingJob",
    "attempt_errors",
    "chain_receipt_errors",
    "contrast_errors",
    "decision_errors",
    "display_stack_errors",
    "gpu_health_errors",
    "hardware_chain_continues",
    "incident_errors",
    "notification_errors",
    "require_reset_grade",
    "reset_grade_errors",
    "result",
    "seed_errors",
    "start_job",
    "target_gpu",
    "withheld_errors",
]
