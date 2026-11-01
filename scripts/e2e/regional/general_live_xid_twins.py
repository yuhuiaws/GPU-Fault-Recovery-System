"""GF-GENERAL-LIVE-024 ``xid_twin_*``: one XID reported twice -- the kernel
NVRM line and the DCGM field-230 change -- is one occurrence (batch 237).

Every real XID reaches the control plane twice: the kernel collector posts the
NVRM line at once, and DCGM field 230 (``DCGM_FI_DEV_XID_ERRORS``) changes on
the exporter's next refresh, 15-30 s later, which ``GpuMetricsService.
_xid_event`` turns into a second ``XidEvent``. NVIDIA documents the field as a
value, not an event of its own (DCGM Field Identifiers, DCGM documentation
"Version: 4.6 (Latest)", ``DCGM_FI_DEV_XID_ERROR`` 230: "XID errors." / "The
value is the specific XID error"). Batch 237 (``policy/xid_cross_source``)
makes the two one occurrence -- one incident, one remediation, in either
arrival order -- where before the second report could run a second GPU reset,
quarantine a node already repaired, or hold two job workflows against each
other. The drills' synthetic kernel lines never move field 230, so no live case
exercised it; these variants synthesise *both* reports on one node with the
two halves the harness already has: the kmsg kernel-source replay
(``general_live_kernel_replay``: the real ``KernelLogCollector`` reads a
reviewed ``NVRM: Xid`` line) and the DCGM field replay
(``tools/gpu_dcgm_field_replay.py`` profile ``xid_twin``: the gauge, absent on a
healthy GPU, synthesised from the GPU's own label set, 0 first as the product's
baseline, then the same XID). Parsing, delivery, policy, incidents, workflows,
the executor and the Node Agent stay real; the hardware actions of the X
variants are the product's own.

The variants (S = safe, X = destructive, approvals in the registry):

* ``xid_twin_kernel_first`` (S) -- XID 92 (EXCESSIVE_SBE_INTERRUPTS, catalog
  IGNORE: MONITOR_ONLY) on an idle node, the kernel line first and the change
  on the collector's first scrape 15-30 s after it (the production lag; the
  runner advances the replay right after the scrape that precedes that window,
  from the journal's own clock). Then the negative control: the same line
  again, 150 s after the change -- outside the 120 s twin window, the gauge
  unchanged (a repeated XID does not move it) -- must be a new occurrence.
* ``xid_twin_gauge_first`` (S) -- the same XID, the change first and the kernel
  line 30 s after it, inside the twin window; the same negative control.
* ``xid_twin_reset_gpu`` (X) -- XID 48 (WORKFLOW_XID_48: solo RESET_GPU) on an
  idle node: exactly one RESET_GPU (one QUIESCE_GPU_SERVICES) executed.
* ``xid_twin_unbound_job`` (X) -- XID 48 on a GPU a managed test job holding
  four of the node's GPUs does not use: exactly one STOP_WORKLOADS + RESET_GPU
  + RESTART_WORKLOAD.
* ``xid_twin_drain_p2p`` (X) -- XID 154 "Drain P2P": the change first (it
  waits up to 120 s for its line, ``PENDING_CORRELATION``), the line 30 s
  later: exactly one RECOVER_DRAIN_P2P and no QUARANTINE planned anywhere.

Order of the X variants. A RESET_GPU or DRAIN_P2P chain quiesces the node's GPU
services -- ``gpu-fault-metrics-collector`` and the exporter container on both
providers, kubelet too on HyperPod -- about 10 s after the decision (spec
section 024, live 2026-10-26 eksb: the next scrape 15 s after the minting one
was "Connection refused"). A change that trails its kernel line by the
production 15-30 s would mostly meet a stopped collector; the field tool would
then serve it to the first scrape after RESTORE_GPU_SERVICES, minutes after the
line and outside the twin window, which no product generation pairs. So the X
variants deliver the change first and the kernel line after the product has
recorded it -- at once for XID 48 (before HyperPod's quiesce takes kubelet,
the probe's path), 30 s later for XID 154 (no chain runs while the change
waits for its line). The kernel-first order of an actionable XID is the
in-memory replay's (``tests/orchestration/test_xid_cross_source_dedupe.py``);
the gauge-follows-line path it exercises is the same function the safe
kernel-first variant drives live.

Product generation: ``general_live_xid_twin_records.require_twin_generation``
asks the deployed control plane before anything is deployed; a product without
batch 237 (whose field-230 change of the XID would be a fault of its own) ends
the variant INCONCLUSIVE with the generation named and the node untouched.

Oracles read durable records only: the two XID events, their saved decisions
and the incident the store links each to; the workflows of the node with their
step executions, and the remote commands' receipts; the twin's join is judged
on ``correlated_event_id`` / ``duplicate`` / the incident link, the counts on
SUCCEEDED step executions across every workflow of the node since the run
began. Cleanup: the kmsg window (closed as soon as the line is decided on the
X variants; deadman-backed; a close the product's quiesce defers waits for
the node), the DCGM replay (closed after the chain, never inside it), the test
job (``xid_twin_unbound_job``) and a validated restore of a node the chain left
isolated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from scripts.e2e.regional import general_live_xid_twin_delivery as delivery
from scripts.e2e.regional.general_live_common import activity, restore_incidents
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    GpuSurvey,
    arm_before_repoint,
    run_gpu_survey,
)
from scripts.e2e.regional.general_live_gpu_replay_close import node_boot_id
from scripts.e2e.regional.general_live_gpu_store import gpu_state
from scripts.e2e.regional.general_live_kernel_xid import (
    SITE_FAMILY,
    catalog_entry,
    kmsg_pci_bdf,
    product_family,
)
from scripts.e2e.regional.general_live_registry import LIVE_REPLAY, VariantSpec
from scripts.e2e.regional.general_live_store import attempt_records
from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    node_isolated,
    settled,
    verdict,
)
from scripts.e2e.regional.general_live_xid_twin_delivery import (
    GAUGE_FIRST_DELAY_SECONDS,
    TWIN_HOLD_SECONDS,
    Delivery,
    close_kernel_early,
    deliver_gauge_first,
    deliver_kernel_first,
    deliver_repeat,
    gauge_key,
    kernel_never_arrived,
    open_kernel_window,
    quiet_before_twin,
    twin_plan,
    wait_gauge_baseline,
    wait_records,
)
from scripts.e2e.regional.general_live_xid_twin_records import (
    GAUGE,
    GAUGE_SOURCE,
    JOB_GPUS_PROBE,
    KERNEL_DECODED_XIDS,
    KERNEL_SOURCE,
    TWIN_WINDOW_SECONDS,
    TwinExpectation,
    chain_errors,
    decided,
    execution_errors,
    hardware_planned,
    incident_of,
    incident_state_errors,
    job_restart_errors,
    marker_associations,
    repeat_errors,
    reports,
    require_twin_generation,
    seconds_between,
    twin_errors,
    twin_pair,
    twin_records,
    watcher_gpus,
    workflow_errors,
)

CASE = "GF-GENERAL-LIVE-024"
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"
DCGM_XID_TWIN = "DCGM_XID_TWIN"
GPU_RESET = "GPU_RESET"
GPU_DRAIN_P2P = "GPU_DRAIN_P2P"
#: Budgets: both reports' final decisions (XID 154's change waits up to 120 s
#: for its line, the coordinator polls each second), the one remediation, and
#: the reserve kept for cleanup.
DECISION_BUDGET_SECONDS = 300
CHAIN_BUDGET_SECONDS = 1800
CLEANUP_RESERVE_SECONDS = 600
#: The replay's own deadline: the chain must end well inside it -- the probe's
#: deadman restores the collector endpoint 600 s after it and would restart a
#: collector the product's quiesce holds stopped.
REPLAY_SECONDS_CAP = 3000

#: The pinned catalog rows the variants rely on (``catalog_entry`` checks them
#: before anything is touched; products ``[A100, H100, B100, GB200]``).
CATALOG_ROWS = {
    92: ("EXCESSIVE_SBE_INTERRUPTS", "IGNORE"),
    48: ("ROBUST_CHANNEL_GPU_ECC_DBE", "WORKFLOW_XID_48"),
    154: ("GPU_RECOVERY_ACTION_CHANGED", "XID_154"),
}

SAFE_EXPECTATION = TwinExpectation(
    xid=92,
    disposition="MONITOR_ONLY",
    official_action="IGNORE",
    actions=frozenset({None, "NO_ACTION"}),
    absent=frozenset(
        {
            "STOP_WORKLOADS",
            "QUIESCE_GPU_SERVICES",
            "RESET_GPU",
            "RECOVER_DRAIN_P2P",
            "RESTART_WORKLOAD",
            "QUARANTINE",
        }
    ),
)
#: Plans no workflow of the node may carry for a single-GPU XID (029's set).
SINGLE_GPU_FORBIDDEN = frozenset(
    {"RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE", "RESTART_VM"}
)
RESET_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
RESET_EXPECTATION = TwinExpectation(
    xid=48,
    disposition="EXECUTABLE",
    official_action="RESET_GPU",
    actions=frozenset({"RESET_GPU"}),
    executed={"QUIESCE_GPU_SERVICES": 1, "RESET_GPU": 1},
    absent=frozenset(
        {"STOP_WORKLOADS", "RESTART_WORKLOAD", "RECOVER_DRAIN_P2P", "QUARANTINE"}
    ),
    chain=RESET_CHAIN,
    forbidden=SINGLE_GPU_FORBIDDEN | {"QUARANTINE"},
    hardware=frozenset({"QUIESCE_GPU_SERVICES", "RESET_GPU"}),
)
JOB_EXPECTATION = TwinExpectation(
    xid=48,
    disposition="EXECUTABLE",
    official_action="RESET_GPU",
    actions=frozenset({"RESET_GPU"}),
    executed={
        "STOP_WORKLOADS": 1,
        "QUIESCE_GPU_SERVICES": 1,
        "RESET_GPU": 1,
        "RESTART_WORKLOAD": 1,
    },
    absent=frozenset({"RECOVER_DRAIN_P2P", "QUARANTINE"}),
    chain=(
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
    ),
    forbidden=SINGLE_GPU_FORBIDDEN | {"QUARANTINE"},
    hardware=frozenset({"QUIESCE_GPU_SERVICES", "RESET_GPU"}),
)
DRAIN_EXPECTATION = TwinExpectation(
    xid=154,
    disposition="EXECUTABLE",
    official_action="DRAIN_P2P",
    actions=frozenset({"STOP_WORKLOAD"}),
    executed={"QUIESCE_GPU_SERVICES": 1, "RECOVER_DRAIN_P2P": 1},
    absent=frozenset({"STOP_WORKLOADS", "RESTART_WORKLOAD", "RESET_GPU", "QUARANTINE"}),
    chain=(
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RECOVER_DRAIN_P2P",
        "RESTORE_GPU_SERVICES",
        "VALIDATE_GPU",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ),
    forbidden=SINGLE_GPU_FORBIDDEN | {"QUARANTINE"},
    hardware=frozenset({"QUIESCE_GPU_SERVICES", "RECOVER_DRAIN_P2P"}),
)


@dataclass(frozen=True)
class TwinCase:
    """One variant: the XID, which report goes first, and what it must prove."""

    variant: str
    expectation: TwinExpectation
    first: str
    #: Gauge-first: how long after the product recorded the change the line is
    #: written (0: at once).
    kernel_delay_seconds: int = 0
    repeat: bool = False
    job: bool = False
    #: What a product without batch 237 would do with this twin (gate text).
    consequence: str = ""
    #: The Node Agent operations the chain needs (refused before mutation when
    #: the node's Agent does not allow them).
    agent_operations: tuple[str, ...] = ()

    @property
    def xid(self) -> int:
        return self.expectation.xid

    @property
    def label(self) -> str:
        return f"024-{self.variant.replace('_', '-')}"


def kernel_tail(case: TwinCase, run_id: str) -> str:
    """The reviewed kernel text of ``case`` after ``<xid>,``."""

    return delivery.kernel_tail(case.xid, case.label, run_id)


def kernel_line(case: TwinCase, pci_bdf: str, run_id: str) -> str:
    return delivery.kernel_line(case.xid, case.label, pci_bdf, run_id)


def kernel_records(case: TwinCase, pci_bdf: str, run_id: str) -> list[dict[str, Any]]:
    return delivery.kernel_records(
        case.xid, case.label, pci_bdf, run_id, repeat=case.repeat
    )


# --------------------------------------------------------------------------- #
# Before anything is touched: catalog, product generation, node capability
# --------------------------------------------------------------------------- #
def require_catalog(case: TwinCase) -> None:
    """The runner's pinned catalog still has the rows the variant relies on."""

    mnemonic, action = CATALOG_ROWS[case.xid]
    entry = catalog_entry(case.xid)
    errors = []
    if entry is None:
        errors.append(f"XID {case.xid} is absent from the pinned catalog")
    else:
        if entry.mnemonic.split()[0] != mnemonic or entry.immediate_action != action:
            errors.append(
                f"XID {case.xid} is {entry.mnemonic} {entry.immediate_action} in "
                f"catalog {entry.catalog_version}, not {mnemonic} {action}"
            )
        if SITE_FAMILY not in entry.products:
            errors.append(
                f"XID {case.xid} does not apply to the {SITE_FAMILY} family "
                f"({list(entry.products)})"
            )
    if errors:
        raise VariantNotRun("FIXTURE_NOT_READY", "; ".join(errors))


def require_agent(
    ctx: VariantContext, case: TwinCase, since: datetime
) -> dict[str, Any]:
    """The node's Agent allows the chain's node operations, and its profile
    does not observe GPU reset (read-only; NOT_RUN before any mutation)."""

    state = gpu_state(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since,
        names=[GAUGE],
    )
    if not case.agent_operations:
        return state
    agent = state.get("agent") or {}
    allowed = set(agent.get("allowed_operations") or [])
    missing = [item for item in case.agent_operations if item not in allowed]
    mode = ((state.get("capabilities") or {}).get("gpuReset") or {}).get("mode")
    problems = []
    if missing:
        problems.append(f"the node's Agent does not allow {missing}")
    if mode in ("OBSERVE", "DISABLED"):
        problems.append(f"the node's Runtime Profile has gpuReset {mode}")
    if problems:
        raise VariantNotRun("CAPABILITY_ABSENT", "; ".join(problems))
    return state


def choose_gpu(
    survey: GpuSurvey, case: TwinCase, *, avoid: set[str] = frozenset()
) -> tuple[str, str]:
    """The highest-index GPU outside ``avoid`` a kernel line can name, of the
    catalog's family, whose exporter shows no real last XID (a twin plan
    would mask it): ``(UUID, kernel PCI BDF)``."""

    entry = catalog_entry(case.xid)
    products = entry.products if entry is not None else ()
    skipped = []
    for gpu in reversed(survey.gpus):
        if gpu in avoid:
            continue
        pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
        if pci_bdf is None:
            skipped.append(f"{gpu}: no PCI BDF")
            continue
        family = product_family(survey.model(gpu))
        if family not in products:
            skipped.append(f"{gpu}: {survey.model(gpu)!r} is family {family}")
            continue
        real = survey.value(gpu, GAUGE)
        if real not in (None, 0.0):
            skipped.append(f"{gpu}: the exporter publishes a real last XID {real:g}")
            continue
        return gpu, pci_bdf
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no GPU the twin can name: " + ("; ".join(skipped) or "no GPU outside the job"),
    )


# --------------------------------------------------------------------------- #
# The two replays
# --------------------------------------------------------------------------- #
def replay_seconds(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, REPLAY_SECONDS_CAP)


# --------------------------------------------------------------------------- #
# The partial-GPU managed job
# --------------------------------------------------------------------------- #
JOB_GPU_COUNT = 4
JOB_RESTART_BUDGET = 2
JOB_CONTAINER = "pytorch"
WATCHER_BUDGET_SECONDS = 180
RESTART_RUNNING_BUDGET_SECONDS = 900


def start_job(ctx: VariantContext) -> Any:
    """A managed job on four of the node's GPUs (real CUDA load), restart
    budget 2 -- a second restart would show in the ledger, not be withheld."""

    from scripts.e2e.regional.general_live_attempt_hardware import ManagedTrainingJob

    digest = hashlib.sha256(ctx.run_id.encode()).hexdigest()[:8]
    name = ctx.identity(f"gl24tw-{digest}-a{ctx.attempt}")
    job = ManagedTrainingJob(
        ctx, name=name, restart_budget=JOB_RESTART_BUDGET, gpu_count=JOB_GPU_COUNT
    )
    ctx.cleanup.push(f"job-{name}", job.cleanup)
    running = job.start()
    ctx.evidence.write("job-running", {"snapshot": running, **job.identity()})
    return job


def pod_gpus(ctx: VariantContext, job: Any) -> set[str]:
    """The GPU UUIDs the job's container sees (its own nvidia-smi)."""

    found: set[str] = set()
    for pod in job.fixture.pods():
        output = ctx.regional.kubectl(
            "gpu",
            "exec",
            str(pod["name"]),
            "-c",
            JOB_CONTAINER,
            "--",
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
            timeout=90,
        )
        found.update(
            line.strip()
            for line in str(output).splitlines()
            if line.strip().startswith("GPU-")
        )
    return found


def job_gpus(ctx: VariantContext, job: Any) -> dict[str, Any]:
    """The job's GPUs by its container and by the product's watcher."""

    seen = pod_gpus(ctx, job)
    last: dict[str, Any] = {}

    def mapped() -> dict[str, Any] | None:
        nonlocal last
        last = ctx.regional.cpu_python(
            JOB_GPUS_PROBE, ctx.settings.regional.cluster_id, job.name, timeout=120
        )
        return last if watcher_gpus(last, attempt_id=job.attempt_id) else None

    ctx.wait_until(
        mapped,
        timeout_seconds=WATCHER_BUDGET_SECONDS,
        poll_seconds=15,
        name="job-gpus",
    )
    watcher = watcher_gpus(last, attempt_id=job.attempt_id)
    document = {
        "container_gpus": sorted(seen),
        "watcher_gpus": sorted(watcher),
        "observations": last,
    }
    ctx.evidence.write("job-gpus", document)
    if not seen and not watcher:
        raise VariantInconclusive(
            "neither the job's container nor the product's watcher names the GPUs it "
            "holds; a GPU the job does not use cannot be chosen"
        )
    if seen and watcher and seen != watcher:
        raise VariantInconclusive(
            f"the job's container sees {sorted(seen)}, the watcher maps "
            f"{sorted(watcher)}; the job's GPUs are not established"
        )
    if len(seen | watcher) != JOB_GPU_COUNT:
        raise VariantInconclusive(
            f"the job holds {len(seen | watcher)} GPUs, not the {JOB_GPU_COUNT} it asked for"
        )
    return document


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def result(
    stages: dict[str, list[str]],
    *,
    inconclusive: list[str],
    notes: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages, inconclusive=inconclusive),
        "proof_scope": "live-replay",
        "physical_fault_injected": False,
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ]
        + [f"inconclusive: {item}" for item in inconclusive],
        "notes": notes,
        **(
            {"inconclusive": inconclusive, "detail": "; ".join(inconclusive)}
            if inconclusive
            else {}
        ),
        **extra,
    }


def grade(
    records: dict[str, Any],
    document: dict[str, Any],
    case: TwinCase,
    *,
    gpu: str,
    delivery: Delivery,
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    """Stages, premise failures (INCONCLUSIVE) and the facts graded."""

    expectation = case.expectation
    inconclusive = list(delivery.problems)
    pair, missing = twin_pair(records, gpu=gpu, first_source=case.first, xid=case.xid)
    inconclusive.extend(missing)
    stages: dict[str, list[str]] = {}
    facts: dict[str, Any] = {"pair": None}
    incident_id = None
    if pair is not None:
        facts["pair"] = {
            "first": pair.first.get("event_id"),
            "second": pair.second.get("event_id"),
            "primary": pair.primary.get("event_id"),
            "joiner": pair.joiner.get("event_id"),
            "joiner_source": pair.joiner.get("event_source"),
            "gap_seconds": pair.gap_seconds,
        }
        if not decided(pair.first) or not decided(pair.second):
            inconclusive.append(
                "a report of the pair has no final decision within the budget"
            )
        stages.update(twin_errors(records, pair, expectation))
        incident_id = (pair.primary.get("decision") or {}).get("incident_id")
        facts["incident_id"] = incident_id
        facts["marker_associations"] = sorted(
            {
                item
                for event in (pair.primary, pair.joiner)
                for item in marker_associations(event.get("decision") or {})
            }
        )
    errors, workflow = workflow_errors(document, incident_id, expectation)
    stages["workflows"] = errors
    executed, counts = execution_errors(document, expectation)
    stages["executions"] = executed
    facts["executions"] = counts
    stages["hardware"] = hardware_planned(document, allowed=expectation.hardware)
    if expectation.remediates:
        stages["chain"] = chain_errors(workflow, expectation)
        stages["incident"] = incident_state_errors(records, incident_id)
        facts["workflow_request_id"] = workflow.get("request_id")
        facts["workflow_status"] = workflow.get("status")
    elif incident_id:
        state = incident_of(records, incident_id).get("state")
        stages["incident"] = (
            []
            if state == "RECOVERED"
            else [f"the occurrence's incident {incident_id} is {state}, not RECOVERED"]
        )
    if case.repeat:
        stages["negative_control"], repeat_inconclusive, repeat_id = grade_repeat(
            records, case, gpu=gpu, delivery=delivery, incident_id=incident_id
        )
        inconclusive.extend(repeat_inconclusive)
        facts["repeat_event_id"] = repeat_id
    return stages, inconclusive, facts


def grade_repeat(
    records: dict[str, Any],
    case: TwinCase,
    *,
    gpu: str,
    delivery: Delivery,
    incident_id: str | None,
) -> tuple[list[str], list[str], str | None]:
    """The repeat past the window: its premise and its own occurrence."""

    kernels = reports(records, gpu=gpu, source=KERNEL_SOURCE)
    gauges = reports(records, gpu=gpu, source=GAUGE_SOURCE)
    if delivery.repeat_written_at is None:
        return [], ["the repeated kernel line was not written"], None
    if len(kernels) < 2:
        return [], ["the repeated kernel line never reached the product"], None
    repeat = kernels[-1]
    after = (
        seconds_between(repeat.get("observed_at"), gauges[0].get("observed_at"))
        if gauges
        else None
    )
    if after is None or after <= TWIN_WINDOW_SECONDS:
        return (
            [],
            [
                f"the repeat is {after} s after the change by the product's clock, not "
                f"outside the {TWIN_WINDOW_SECONDS} s twin window"
            ],
            str(repeat.get("event_id")),
        )
    errors = repeat_errors(
        records,
        str(repeat.get("event_id")),
        occurrence_incident_id=incident_id,
        expectation=case.expectation,
    )
    if len(gauges) != 1:
        errors.append(
            f"{len(gauges)} field-230 changes: the gauge moved again for the repeat"
        )
    return errors, [], str(repeat.get("event_id"))


# --------------------------------------------------------------------------- #
# The variants
# --------------------------------------------------------------------------- #
def run_twin(ctx: VariantContext, case: TwinCase) -> dict[str, Any]:
    """Gate, place both reports, judge the product's records, clean up."""

    require_catalog(case)
    probe_tail = kernel_tail(case, ctx.run_id)
    generation = require_twin_generation(
        ctx, case.expectation, probe_tail, consequence=case.consequence
    )
    since = utc_now()
    require_agent(ctx, case, since)
    job = start_job(ctx) if case.job else None
    held = job_gpus(ctx, job) if job is not None else {}
    if job is not None:
        held["pod_uids"] = sorted(
            str(item) for item in job.identity().get("pod_uids") or []
        )
    survey = run_gpu_survey(ctx)
    avoid = set(held.get("container_gpus") or []) | set(held.get("watcher_gpus") or [])
    gpu, pci_bdf = choose_gpu(survey, case, avoid=avoid)
    quiet = quiet_before_twin(ctx, case.xid, gpu)
    boot_id = survey.boot_id or node_boot_id(ctx)
    replay = DcgmReplay(
        ctx,
        twin_plan(
            gpu,
            case.xid,
            interval=survey.interval,
            hold_seconds=TWIN_HOLD_SECONDS if not case.expectation.remediates else 300,
        ),
        replay_seconds(ctx),
        node_boot_id=boot_id,
    )
    arm_before_repoint(replay, gauge_key(gpu))
    baseline = wait_gauge_baseline(ctx, since, gpu)
    kernel = open_kernel_window(
        ctx, kernel_records(case, pci_bdf, ctx.run_id), boot_id=boot_id
    )
    if case.first == KERNEL_SOURCE:
        delivery = deliver_kernel_first(ctx, replay, kernel, gpu=gpu, xid=case.xid)
    else:
        delivery = deliver_gauge_first(
            ctx,
            replay,
            kernel,
            since=since,
            gpu=gpu,
            xid=case.xid,
            delay_seconds=case.kernel_delay_seconds,
        )
    kernel_seen = wait_records(
        ctx,
        since,
        case.xid,
        lambda records: bool(reports(records, gpu=gpu, source=KERNEL_SOURCE)),
        timeout_seconds=90,
        name="twin-kernel-recorded",
    )
    if not reports(kernel_seen, gpu=gpu, source=KERNEL_SOURCE):
        raise kernel_never_arrived(kernel, f"the kernel line of XID {case.xid}")
    if case.expectation.remediates:
        # Before the chain's quiesce takes the node's GPU services.
        close_kernel_early(ctx, kernel, boot_id)
    records = wait_records(
        ctx,
        since,
        case.xid,
        lambda current: pair_decided(current, case, gpu=gpu),
        timeout_seconds=DECISION_BUDGET_SECONDS,
        name="twin-decisions",
    )
    if case.repeat:
        deliver_repeat(ctx, kernel, delivery)
        records = wait_records(
            ctx,
            since,
            case.xid,
            lambda current: len(
                [
                    item
                    for item in reports(current, gpu=gpu, source=KERNEL_SOURCE)
                    if decided(item)
                ]
            )
            >= 2,
            timeout_seconds=DECISION_BUDGET_SECONDS,
            name="twin-repeat-decision",
        )
        close_kernel_early(ctx, kernel, boot_id)
    document, chain_settled = settle(ctx, since, records, case)
    records = twin_records(ctx, since, case.xid)
    ctx.evidence.write("xid-twin-records", records)
    ctx.evidence.write("xid-twin-delivery", delivery.record())
    stages, inconclusive, facts = grade(
        records, document, case, gpu=gpu, delivery=delivery
    )
    facts["chain_settled"] = chain_settled
    if job is not None:
        stages["job"] = judge_job(ctx, job, document, facts, held, gpu=gpu)
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["isolation"] = restore_if_left_isolated(ctx, records, facts)
    return result(
        stages,
        inconclusive=inconclusive,
        notes=notes(case, generation.describe(), delivery),
        extra={
            "xid": case.xid,
            "gpu_uuid": gpu,
            "pci_bdf": pci_bdf,
            "order": delivery.order,
            "delivery": delivery.record(),
            "gauge_baseline": baseline,
            "product_generation": generation.record(),
            "kernel_replay": kernel.evidence(),
            "quiet_before_twin": quiet,
            **facts,
            **({"job": held} if job is not None else {}),
        },
    )


def pair_decided(records: dict[str, Any], case: TwinCase, *, gpu: str) -> bool:
    """Both reports of the pair are in the store with final decisions."""

    kernels = reports(records, gpu=gpu, source=KERNEL_SOURCE)
    gauges = reports(records, gpu=gpu, source=GAUGE_SOURCE)
    return bool(kernels and gauges) and decided(kernels[0]) and decided(gauges[0])


def settle(
    ctx: VariantContext, since: datetime, records: dict[str, Any], case: TwinCase
) -> tuple[dict[str, Any], bool]:
    """The node's activity once the occurrence's remediation is terminal."""

    incident_ids = sorted(
        {
            str((item.get("decision") or {}).get("incident_id"))
            for item in records.get("events") or []
            if (item.get("decision") or {}).get("incident_id")
        }
    )
    workflow_ids = {
        str((item.get("decision") or {}).get("workflow_request_id"))
        for item in records.get("events") or []
        if (item.get("decision") or {}).get("workflow_request_id")
    }
    if not case.expectation.remediates or not workflow_ids:
        # Nothing to settle: a monitor-only twin plans none, and a remediating
        # one that planned none fails its workflow stage on this one read.
        document = activity(ctx, since, metrics=(GAUGE,), kinds=("GPU_METRICS",))
        ctx.evidence.write("xid-twin-activity", document)
        return document, True
    budget = max(
        60.0,
        min(
            float(CHAIN_BUDGET_SECONDS),
            ctx.remaining_seconds() - CLEANUP_RESERVE_SECONDS,
        ),
    )
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=(GAUGE,), kinds=("GPU_METRICS",))
        return last if settled(last, incident_ids) else None

    value = ctx.wait_until(
        accept, timeout_seconds=budget, poll_seconds=20, name="twin-chain"
    )
    document = value if value is not None else last
    ctx.evidence.write("xid-twin-activity", document)
    return document, value is not None


def judge_job(
    ctx: VariantContext,
    job: Any,
    document: dict[str, Any],
    facts: dict[str, Any],
    held: dict[str, Any],
    *,
    gpu: str,
) -> list[str]:
    """The XID was on a GPU the job does not use; the job stopped once and
    came back once."""

    errors = []
    if gpu in set(held.get("container_gpus") or []) | set(
        held.get("watcher_gpus") or []
    ):
        errors.append(f"the XID GPU {gpu} is one of the job's")
    workflow = next(
        (
            item
            for item in document.get("workflows") or []
            if item.get("request_id") == facts.get("workflow_request_id")
        ),
        {},
    )
    restart = final_execution(workflow, "RESTART_WORKLOAD") if workflow else None
    restart_attempt = ((restart or {}).get("details") or {}).get("restart_attempt_id")
    records = attempt_records(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        job.name,
        [job.attempt_id],
    )
    ctx.evidence.write("job-attempt-records", records)
    errors.extend(
        job_restart_errors(
            records, attempt_id=job.attempt_id, restart_attempt_id=restart_attempt
        )
    )
    old = {str(item) for item in held.get("pod_uids") or []}
    try:
        restarted = job.fixture.wait_restarted(
            old, timeout_seconds=RESTART_RUNNING_BUDGET_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - the restart's effect is the stage
        errors.append(f"the restarted job did not run: {type(exc).__name__}: {exc}")
    else:
        ctx.evidence.write("job-restarted", restarted)
    facts["restart_attempt_id"] = restart_attempt
    return errors


def restore_if_left_isolated(
    ctx: VariantContext, records: dict[str, Any], facts: dict[str, Any]
) -> list[str]:
    """Hand back a node the chain left isolated (validated restore)."""

    node = ctx.regional.node_snapshot(ctx.settings.node)
    if not node_isolated(node):
        return []
    incidents = [
        {"incident_id": item["incident_id"]}
        for item in records.get("incidents") or []
        if item.get("incident_id") and item.get("state") != "RECOVERED"
    ]
    if not incidents:
        return ["the node is isolated and no incident of the twin owns it"]
    try:
        restore_incidents(ctx, incidents, reason=f"{ctx.spec.key} acceptance restore")
    except Exception as exc:  # noqa: BLE001 - recorded as the stage's error
        return [f"validated restore failed: {type(exc).__name__}: {exc}"]
    facts["restored_incidents"] = [item["incident_id"] for item in incidents]
    return []


def notes(case: TwinCase, generation: str, delivery: Delivery) -> list[str]:
    timing = delivery.record()
    lines = [
        generation,
        f"order {delivery.order}: the change was served "
        f"{timing.get('gauge_minus_kernel_seconds')} s after the kernel line "
        "(node clock; negative = the change first)",
        "both reports are replays (live-replay); the incidents, workflows and node "
        "actions are the product's own",
    ]
    if case.repeat:
        lines.append(
            "negative control: the same kernel line again "
            f"{timing.get('repeat_after_gauge_seconds')} s after the change, the "
            "gauge unchanged -- a repeated XID does not move field 230"
        )
    if case.xid in KERNEL_DECODED_XIDS:
        lines.append(
            "XID 154: the change carries no recovery-action label; it waits up to "
            f"{TWIN_WINDOW_SECONDS} s for its kernel line and always names it"
        )
    if case.expectation.remediates and case.first == GAUGE_SOURCE:
        lines.append(
            "the change goes first: a reset or drain chain quiesces the metrics "
            "collector ~10 s after its decision, before a change trailing its line "
            "by the production 15-30 s could be scraped"
        )
    return lines


def kernel_first(ctx: VariantContext) -> dict[str, Any]:
    return run_twin(ctx, CASES["xid_twin_kernel_first"])


def gauge_first(ctx: VariantContext) -> dict[str, Any]:
    return run_twin(ctx, CASES["xid_twin_gauge_first"])


def reset_gpu(ctx: VariantContext) -> dict[str, Any]:
    return run_twin(ctx, CASES["xid_twin_reset_gpu"])


def unbound_job(ctx: VariantContext) -> dict[str, Any]:
    return run_twin(ctx, CASES["xid_twin_unbound_job"])


def drain_p2p(ctx: VariantContext) -> dict[str, Any]:
    return run_twin(ctx, CASES["xid_twin_drain_p2p"])


CASES = {
    item.variant: item
    for item in (
        TwinCase(
            "xid_twin_kernel_first",
            SAFE_EXPECTATION,
            KERNEL_SOURCE,
            repeat=True,
            consequence="a second MONITOR_ONLY incident for one XID",
        ),
        TwinCase(
            "xid_twin_gauge_first",
            SAFE_EXPECTATION,
            GAUGE_SOURCE,
            kernel_delay_seconds=GAUGE_FIRST_DELAY_SECONDS,
            repeat=True,
            consequence="a second MONITOR_ONLY incident for one XID",
        ),
        TwinCase(
            "xid_twin_reset_gpu",
            RESET_EXPECTATION,
            GAUGE_SOURCE,
            consequence=(
                "a second RESET_GPU of the GPU once the first chain has finished"
            ),
            agent_operations=("QUIESCE_GPU_SERVICES", "RESET_GPU"),
        ),
        TwinCase(
            "xid_twin_unbound_job",
            JOB_EXPECTATION,
            GAUGE_SOURCE,
            job=True,
            consequence=(
                "a second STOP_WORKLOADS + RESET_GPU + RESTART_WORKLOAD, or two job "
                "workflows holding each other into a bare stop"
            ),
            agent_operations=("QUIESCE_GPU_SERVICES", "RESET_GPU"),
        ),
        TwinCase(
            "xid_twin_drain_p2p",
            DRAIN_EXPECTATION,
            GAUGE_SOURCE,
            kernel_delay_seconds=GAUGE_FIRST_DELAY_SECONDS,
            consequence=(
                "a BLOCKED_MISSING_EVIDENCE quarantine of the node the line's "
                "DRAIN_P2P repairs"
            ),
            agent_operations=("QUIESCE_GPU_SERVICES", "RECOVER_DRAIN_P2P"),
        ),
    )
}

CLEANUP = (
    "close the kmsg window (deadman-backed; on the X variants as soon as the line "
    "is decided), disarm the field replay and restore the metrics collector "
    "endpoint after the chain, remove every /run fixture; validated restore of a "
    "node left isolated"
)


def _spec(
    variant: str,
    execute: Callable[[VariantContext], dict[str, Any]],
    *,
    injection: str,
    expected: str,
    risk: str,
    flags: str,
    approvals: tuple[str, ...],
    max_minutes: int,
    cleanup: str = CLEANUP,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk=risk,
        flags=flags,
        approvals=approvals,
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        execute=execute,
    )


TWIN_APPROVALS = (KERNEL_SOURCE_REPLAY, DCGM_XID_TWIN)

VARIANTS = [
    _spec(
        "xid_twin_kernel_first",
        kernel_first,
        injection=(
            "on an idle node, a reviewed kernel XID 92 line (EXCESSIVE_SBE_INTERRUPTS, "
            "catalog IGNORE) through the kmsg source, then the DCGM last-XID gauge "
            "of the same GPU 0 -> 92 on the collector's first scrape 15-30 s later; "
            "150 s after the change the same line again, the gauge unchanged"
        ),
        expected=(
            "one MONITOR_ONLY incident: the change's decision duplicate with "
            "correlated_event_id naming the line, linked to the line's RECOVERED "
            "incident, no workflow; the repeat outside the 120 s twin window is a new "
            "occurrence (its own RECOVERED incident, no correlation)"
        ),
        risk="M",
        flags="S",
        approvals=TWIN_APPROVALS,
        max_minutes=25,
    ),
    _spec(
        "xid_twin_gauge_first",
        gauge_first,
        injection=(
            "on an idle node, the DCGM last-XID gauge of one GPU 0 -> 92, then a "
            "reviewed kernel XID 92 line for that GPU 30 s later (inside the 120 s "
            "twin window); 150 s after the change the same line again"
        ),
        expected=(
            "one MONITOR_ONLY incident: the line's decision duplicate with "
            "correlated_event_id naming the change, linked to the change's RECOVERED "
            "incident, no workflow; the repeat is a new occurrence"
        ),
        risk="M",
        flags="S",
        approvals=TWIN_APPROVALS,
        max_minutes=25,
    ),
    _spec(
        "xid_twin_reset_gpu",
        reset_gpu,
        injection=(
            "on an idle node, the DCGM last-XID gauge of one GPU 0 -> 48 (WORKFLOW_XID_"
            "48: solo RESET_GPU), then a reviewed kernel XID 48 line for that GPU as "
            "soon as the product recorded the change, inside the twin window"
        ),
        expected=(
            "one incident, one RESET_GPU chain (FREEZE_EVIDENCE .. RESET_GPU .. "
            "VALIDATE_GPU, RESTORE_SCHEDULING) SUCCEEDED and RECOVERED: exactly one "
            "QUIESCE_GPU_SERVICES and one RESET_GPU executed on the node; the line's "
            "decision duplicate, naming the change, no workflow of its own"
        ),
        risk="H",
        flags="X",
        approvals=(GPU_RESET, *TWIN_APPROVALS),
        max_minutes=50,
    ),
    _spec(
        "xid_twin_unbound_job",
        unbound_job,
        injection=(
            "a managed test job on four of the node's GPUs (restart budget 2); the "
            "DCGM last-XID gauge 0 -> 48 on a GPU the job does not use, then the "
            "reviewed kernel XID 48 line for it as soon as the change is recorded"
        ),
        expected=(
            "one incident, one chain: exactly one STOP_WORKLOADS, one RESET_GPU and "
            "one RESTART_WORKLOAD executed (the job stopped once and restarted once, "
            "ledger 1), RECOVERED; the line's decision duplicate, no workflow of its own"
        ),
        risk="H",
        flags="X",
        approvals=(GPU_RESET, *TWIN_APPROVALS),
        max_minutes=60,
        cleanup=CLEANUP + "; delete the test job, its restarted copy and prewarm Pods",
    ),
    _spec(
        "xid_twin_drain_p2p",
        drain_p2p,
        injection=(
            "on an idle node, the DCGM last-XID gauge of one GPU 0 -> 154, then a "
            "reviewed kernel XID 154 'GPU recovery action changed ... (Drain P2P)' "
            "line for that GPU 30 s later, inside the 120 s window the change waits"
        ),
        expected=(
            "the change waits for its line (no quarantine beside it); one incident, "
            "one DRAIN_P2P chain SUCCEEDED and RECOVERED: exactly one "
            "RECOVER_DRAIN_P2P executed, no QUARANTINE planned on the node; the "
            "change's decision duplicate naming the line"
        ),
        risk="H",
        flags="X",
        approvals=(GPU_DRAIN_P2P, *TWIN_APPROVALS),
        max_minutes=50,
    ),
]

__all__ = [
    "CASE",
    "CASES",
    "DCGM_XID_TWIN",
    "GPU_DRAIN_P2P",
    "GPU_RESET",
    "KERNEL_SOURCE_REPLAY",
    "VARIANTS",
    "TwinCase",
    "choose_gpu",
    "grade",
    "grade_repeat",
    "kernel_line",
    "kernel_records",
    "kernel_tail",
    "require_agent",
    "require_catalog",
    "run_twin",
]
