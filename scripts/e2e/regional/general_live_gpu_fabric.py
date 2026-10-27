"""GF-GENERAL-LIVE-021 NVLink fabric not trained (live-replay).

The fabric state is DCGM field 170, graded per node over every GPU
(``gpu_fabric_manager``): a graded NVSwitch GPU (H100/H200/H800/H20, HGX
B200/B300) that holds an unsettled state (1 not started, 2 in progress, 4
failure) for >= 300 s mints one CRITICAL finding whose recovery is the full
GPU + NVSwitch reset. This module sets the state through the ``fabric`` profile
and reads the product's own grading.

Producing the finding therefore drives a real RESET_ALL_GPUS_NVSWITCHES on the
node, so every variant that crosses the window needs ``--approve
GPU_FULL_FABRIC_RESET`` and runs on the lane's exclusive node on both clusters
-- each plain-EKS lane owns its node and the ``X`` flag holds every lock, so
nothing else is on the node or the cluster while the fabric resets
(``recovery_failure`` also needs ``--approve NODE_REBOOT``). The held plan
serves SUCCESS (3) once RESET_ALL has run so the post-reset VALIDATE_GPU --
the step that grades the fabric state after the reset
(``adapters/gpu_validation._validate_nvlink_fabric`` over
``gpu_fabric_manager.fabric_validation``) -- reads a trained fabric.
``before_window`` (held under 300 s), ``ungraded_product`` (a non-NVSwitch
modelName) and ``monitor_restart`` (the held clock restarts with the
collector) never cross the window, so they isolate nothing and need no reset
approval. Their held steps are bounded by ``held_reads`` -- the hold at the
collector's interval, within the tool's 200-read step limit; live 2026-10-26
(HyperPod hpb) a 400-read bound was refused at the tool's start and
``before_window`` ended INCONCLUSIVE with the unit dead.

The reboot the product's ladder performs
---------------------------------------
A post-reset validation that reads the fabric still failed is a rung of the
product's hardware ladder, not its end: ``orchestration/escalation.next_rung``
sends a failed validation behind a reset to RESTART_NODE, and
``execution/branch_escalation`` appends that rung with its validation tail
(VALIDATE_GPU, VALIDATE_HOST, VALIDATE_FABRIC, RESTORE_SCHEDULING) to the same
record (a flat single-node plan is adopted into a one-branch DAG). RESTART_NODE
is a reboot of the same instance on both providers -- HyperPod
``BatchRebootClusterNodes``, plain EKS EC2 ``RebootInstances``
(``node_provider.restart_vm_effective_action``) -- and every replay object
lives under ``/run``, so the replay cannot survive it: after the boot the
product validates the node's real, trained fabric and restores scheduling.
Live 2026-10-27 05:04-05:26Z (HyperPod, R13g) ``recovery_failure`` did exactly
that -- VALIDATE_GPU failed on the held FAILURE, RESTART_NODE rebooted the
node (boot id aa20cf7c -> b7e017d9), the post-reboot validations passed, the
incident RECOVERED -- and the variant graded FAIL because it asked the settled
record for the ladder's END state (ESCALATED / QUARANTINED), which only a
fabric still failed AFTER the reboot would produce. That branch is not
observable with a ``/run``-based replay on either provider, and must not be
made observable: a replay that survives the boot drives the product into a
reboot/quarantine loop (024 ``wrong_scope_source``, 2026-10-26 14:54Z).

So ``recovery_failure`` grades the escalation on the record the product writes
BEFORE and AT the reboot (``held_failure_errors``): the post-reset validation
FAILED on the held fault, the node was not released while the fabric read
failed (no RESTORE_SCHEDULING before the rung), and the rung followed --
RESTART_NODE, or an operator hand-off (ESCALATED / QUARANTINED) on a site that
does not reboot. Then it waits for the record to settle and judges the exit
(``escalation_exit_errors``): exactly one reset, at most one reboot, no
REPLACE_NODE, a reboot that really changed the boot id, and a settled record
whose incident state matches it (SUCCEEDED -> RECOVERED with the node released;
FAILED/BLOCKED -> ESCALATED/QUARANTINED). The reset-only variants share the
exposure: the collector's first scrape after RESTORE_GPU_SERVICES is served the
held value until the runner advances the plan, and a FAILURE state fails
VALIDATE_GPU at once (1/2 wait as ``nvlink_fabric_training``), so a late
advance makes the product reboot the node by design. Their reset-done wait
polls faster and the advance retries through a quiesce that still has kubelet
down, and a record that nevertheless carries a RESTART_NODE rung (or a node
whose boot id changed) is INCONCLUSIVE with the reason, never PASS on a reboot
nor FAIL on the product (``reboot_exposure``).

The advance the tool refuses (plain EKS, 2026-10-27)
----------------------------------------------------
Live 2026-10-27 13:14-13:28Z (plain EKS ip-10-0-43-5, R13g, harness
int/r13c-live27) ``not_started`` -- and ``in_progress`` at 13:44-13:59Z --
ended INCONCLUSIVE "the node did not answer the probe in 13 attempts over
187 s" while the node answered every attempt: the probe's loopback ``POST
/advance`` got HTTP 409 Conflict from the replay tool (``probe-error.json``
``loopback:430``), the same answer from 13:25:19Z (8 ms after reset-done was
accepted) to 13:28:26Z. The tool's journal (the case's ``oracle.jsonl``) says
why: the EKS collector scrapes through the quiesce (journal read 27: upstream
``Connection refused`` while the exporter was down with the host engine), and
its first scrape after the exporter returned read the real
``fabric_manager_status`` 2.0 of one GPU -- Fabric Manager re-training after
the reset -- against the baseline 3.0, so ``Replay._check_baseline`` aborted
("a real reading is never masked"), ``Replay.active`` is ``None`` for good,
the real readings pass through, ``Replay.advance`` raises "the replay is not
on a step" and ``do_POST`` answers 409. On HyperPod the collector pauses with
the quiesce (27 reads in 589 s, no upstream error) and its first post-restore
scrape read the fabric already trained (3.0), so the advance landed at read 27
and the variant passed at 12:11Z. The product's grading is the same either
way -- its post-reset validation reads a trained fabric, served or real -- so a
probe exit that is not a transport failure now makes the runner read the
tool's ``/status`` and classify the refusal (``plan_release``): an abort or a
lapsed hold is the plan's release (PASS with a note), a plan already past the
hold is an advance whose receipt was lost, an ended tool is INCONCLUSIVE with
the 409's meaning, a plan still on its held step is retried; kubelet-down
transport failures are retried as before without consulting the tool.
``recovery_failure`` released this way is INCONCLUSIVE (the FAILURE cannot be
held past the real transition on such a node), never FAIL on the product. The
held step's read bound was a near miss of its own: 36 reads ((300 + 180) / 15
+ 4) lapse after 39 scrapes, 585 s at 15 s, and plain EKS's reset-done came
569 s after arm; ``reset_hold_reads`` now sizes it over the detection horizon
and the chain with ``held_reads``.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import node_residual_isolation
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    arm_and_confirm,
    baseline,
    gpu_spec,
    held_reads,
    label,
    no_hardware,
    plan,
    put,
    result,
    run_gpu_survey,
    state_now,
    step,
    utc_now,
    wait_gpu_incident,
)
from scripts.e2e.regional.general_live_gpu_replay_close import node_boot_id
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    executions_of,
    fabric_incidents,
    settled,
)
from scripts.e2e.regional.general_live_verdicts import parse_time, workflows_of
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeMissingResponseError,
    HostProbeTransportError,
)

CASE = "GF-GENERAL-LIVE-021"
WINDOW = 300
#: What a collector restart adds to ``monitor_restart``'s two holds when its
#: held step's read bound is sized: the restart, the sandbox settle and two
#: scrapes on the real endpoint.
RESTART_SECONDS = 180
STATES = {"not_started": 1, "in_progress": 2, "failed": 4}
#: The product's catalog chain for ``RESET_ALL_GPUS_AND_NVSWITCHES``
#: (``orchestration/workflow_builder.catalog_operations``): the cordon that
#: holds the node is part of it, so a record that never executed it is the
#: hollow chain ``chain_errors`` names.
FULL_RESET = (
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "VALIDATE_FABRIC",
)
REBOOT = "RESTART_NODE"
REPLACE = "REPLACE_NODE"
RELEASE = "RESTORE_SCHEDULING"
#: The validations behind the reset that read the fabric state (VALIDATE_GPU
#: grades it; VALIDATE_FABRIC also fails on the node's active finding).
POST_RESET_VALIDATIONS = ("VALIDATE_GPU", "VALIDATE_FABRIC")
#: The reset-done wait releases the held fault; the product's post-reset
#: validation reads the collector's first scrape after RESTORE_GPU_SERVICES,
#: so the runner must learn of the reset's completion quickly.
RESET_POLL_SECONDS = 10
#: ``advance`` is a probe exec; on a HyperPod node kubelet is down from the
#: quiesce until RESTORE_GPU_SERVICES starts it again, and a reset that
#: completed just before the poll leaves the probe unanswered for a while.
ADVANCE_RETRY_SECONDS = 180
ADVANCE_POLL_SECONDS = 10
#: The product's chain from the finding to RESET_ALL SUCCEEDED, allowed for in
#: the held step's read bound beyond the detection horizon (WINDOW plus the
#: health summary's latency): live 165 s (plain EKS 2026-10-27 13:22:34 ->
#: 13:25:19Z) and 202 s (HyperPod 12:05:57 -> 12:09:19Z).
RESET_CHAIN_SECONDS = 300
#: What the replay tool's HTTP 409 Conflict on ``POST /advance`` means
#: (``tools/gpu_dcgm_field_replay.py`` ``Replay.advance`` / ``do_POST``).
CONFLICT_MEANING = (
    "HTTP 409 Conflict from the replay tool means the plan is not on a held step "
    "(tools/gpu_dcgm_field_replay.py Replay.advance: the replay aborted, lapsed, "
    "ended or already moved on), never that the node did not answer"
)
#: ``plan_release`` kinds that end the replay without the runner: nothing
#: holds the fault and nothing serves the tail.
ENDED_RELEASES = frozenset({"disarmed", "unarmed"})
#: How long the product has to fail its post-reset validation and take the
#: rung (live: reset done 05:16, rung 05:17), and how long the escalated
#: record has to settle (live: reboot + validations + release, 05:17-05:25).
ESCALATION_SECONDS = 600
SETTLE_SECONDS = 900
ESCALATED_STATES = frozenset({"ESCALATED", "QUARANTINED"})


def _graded_gpus(survey) -> list[str]:
    graded = [gpu for gpu in survey.gpus if survey.has(gpu, "fabric_manager_status")]
    models = {survey.model(gpu) for gpu in graded}
    if not graded or not any(
        any(token in model for token in ("H100", "H200", "H800", "H20", "B200", "B300"))
        for model in models
    ):
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "no graded NVSwitch GPU (H100/H200/H800/H20, HGX B200/B300) reports a fabric state",
        )
    return graded


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 45 * 60)


# --------------------------------------------------------------------------- #
# detection + authorized full fabric reset
# --------------------------------------------------------------------------- #
def _unsettled_reset(
    ctx: VariantContext, status: int, *, recovery_failure: bool = False
) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    tail = put("fabric_manager_status", gpus, 4 if recovery_failure else 3)
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                reset_hold_reads(survey.interval, survey.health_summary),
                put("fabric_manager_status", gpus, status),
                hold=True,
            ),
            step(40, tail),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    detection = wait_gpu_incident(
        ctx,
        since,
        fabric_incidents,
        timeout_seconds=WINDOW + survey.health_summary + 300,
        name="detection",
    )
    incidents = fabric_incidents(detection)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label="nvlink fabric not trained",
                operations=FULL_RESET,
                actions=frozenset({"RESET_GPU"}),
                forbidden=frozenset({REPLACE}),
            ),
        ),
    }
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: [
            i
            for i in incidents
            if any(
                e.get("status") == "SUCCEEDED"
                for e in executions_of(doc, "RESET_ALL_GPUS_NVSWITCHES", [i])
            )
        ],
        timeout_seconds=WINDOW + survey.health_summary + 900,
        name="reset-done",
        poll_seconds=RESET_POLL_SECONDS,
    )
    advance = _advance_when_reachable(ctx, repl)
    notes: list[str] = release_notes(advance)
    inconclusive: list[str] = []
    if recovery_failure:
        if not advance["advanced"]:
            raise VariantInconclusive(held_failure_release(advance))
        graded, escalation = _graded_escalation(ctx, repl, since, incidents)
        stages.update(graded)
        notes += escalation
    else:
        final = wait_gpu_incident(
            ctx,
            since,
            lambda doc: incidents if settled(doc, incidents) else [],
            timeout_seconds=SETTLE_SECONDS,
            name="settled",
        )
        stages["recovery"] = (
            []
            if settled(final, incidents) and not _fabric_active(ctx, since)
            else ["the fabric reset did not settle with a trained post-reset fabric"]
        )
        boot_now = node_boot_id(ctx)
        inconclusive = reboot_exposure(
            final, incidents, boot_before=repl.node_boot_id, boot_now=boot_now
        )
        if inconclusive:
            ctx.evidence.write(
                "fabric-escalation",
                escalation_record(
                    final, incidents, boot_before=repl.node_boot_id, boot_now=boot_now
                ),
            )
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=notes,
        inconclusive=inconclusive,
        extra={"gpus": gpus, "status": status, "plan_advance": advance},
    )


def reset_hold_reads(interval: int, health_summary: int) -> int:
    """The read bound of the six reset variants' held fault.

    The hold must outlast the detection (the 300 s window plus the health
    summary's latency) and the product's chain up to RESET_ALL SUCCEEDED as
    the reset-done wait observes it; ``held_reads`` doubles the reads a
    collector makes over that horizon and refuses what the tool cannot bound.
    Live 2026-10-27 the bound was ``(300 + 180) / interval + 4`` = 36 reads:
    with the 3 baseline reads it lapses after 39 scrapes, 585 s at 15 s, and
    plain EKS's reset-done came 569 s after arm with the collector scraping
    through the quiesce (HyperPod's pauses with it: 27 reads in 589 s). A
    lapsed hold passes the real reading through and the advance is refused
    (HTTP 409) with that meaning.
    """

    return held_reads(interval, WINDOW + health_summary + RESET_CHAIN_SECONDS)


def plan_release(plan: dict[str, Any], status: Any) -> dict[str, Any] | None:
    """What the tool's ``/status`` says about an advance it refused.

    ``None`` when the status does not explain a refusal -- the plan is still
    on its held step (the probe exec failed, not the tool) or the status lacks
    the tool's fields (an older probe) -- so the caller retries. Otherwise
    ``kind`` names the release: ``aborted`` (the tool aborted on a real
    upstream change or an absent target and passes the real readings through
    -- the 2026-10-27 plain-EKS shape, Fabric Manager re-training after the
    reset read 2 against the baseline 3), ``lapsed`` (armed, no abort, on no
    step: the held step's read bound or the tool's deadline passed;
    pass-through too), ``advanced`` (the plan is already past the held step: an
    earlier advance landed and its receipt was lost), ``disarmed`` /
    ``unarmed`` (the replay ended or never started: nothing holds the fault,
    nothing serves the tail).
    """

    if not isinstance(status, dict) or "step" not in status or "armed" not in status:
        return None
    step = status.get("step")
    if status.get("aborted"):
        kind, reason = "aborted", f"the tool aborted: {status['aborted']}"
    elif status.get("disarmed"):
        kind, reason = "disarmed", "the replay is disarmed"
    elif not status.get("armed"):
        kind, reason = (
            "unarmed",
            "the replay is not armed (a restarted tool serves a fresh pass-through "
            "replay)",
        )
    elif step is None:
        kind, reason = (
            "lapsed",
            "the held step's read bound or the tool's deadline passed after "
            f"{status.get('reads')} reads; the tool passes the real readings through",
        )
    else:
        steps = plan.get("steps") or []
        if (
            not isinstance(step, int)
            or not 0 <= step < len(steps)
            or steps[step].get("until_advance")
        ):
            return None
        kind, reason = (
            "advanced",
            f"the plan is already on step {step}, which is not held: an earlier "
            "advance landed and its receipt was lost",
        )
    return {
        "kind": kind,
        "reason": reason,
        "status": status,
        "meaning": CONFLICT_MEANING,
    }


def release_notes(advance: dict[str, Any]) -> list[str]:
    """The reader's note when the plan was released without the advance."""

    release = advance.get("release")
    if not release or advance.get("advanced"):
        return []
    note = (
        "the held plan was released by the replay tool, not by the runner's "
        f"advance: {release['reason']}; the tool refused the advance with HTTP 409 "
        "Conflict (the replay is not on a held step) and passes the node's real "
        "readings through, so the product's post-reset validation graded the real "
        "fabric -- the trained state the plan's tail would have served"
    )
    if release["kind"] == "aborted":
        note += (
            " (plain-EKS shape 2026-10-27: the collector scrapes through the quiesce "
            "and its first scrape after the exporter returns catches Fabric Manager "
            "re-training, 3 -> 2, which a replay never masks)"
        )
    return [note]


def held_failure_release(advance: dict[str, Any]) -> str:
    """Why ``recovery_failure`` is INCONCLUSIVE when the tool released the hold."""

    release = advance["release"]
    return (
        "the replay released the held FAILURE before the product's post-reset "
        f"validation read it ({release['reason']}); the tool refused the advance "
        "with HTTP 409 Conflict (the replay is not on a held step) and passes the "
        "real fabric through, so the RESTART_NODE rung cannot be driven on this "
        "node: its collector scrapes through the quiesce and the real post-reset "
        "Fabric Manager transition (3 -> 2 -> 3) reaches the tool first, and a real "
        "reading is never masked -- not a product error; the rung is observable "
        "where the collector pauses with the quiesce (HyperPod, 2026-10-27 05:04Z)"
    )


def _unreachable(error: BaseException) -> bool:
    """Whether the probe never ran: the transport, not a rejected host action."""

    return isinstance(error, (HostProbeTransportError, HostProbeMissingResponseError))


def _refusal(repl: DcgmReplay) -> tuple[dict[str, Any] | None, str | None]:
    """Read the tool's status after a probe exit: ``(release, status_error)``."""

    try:
        answer = repl.status()
    except HostProbeError as error:
        return None, f"{type(error).__name__}: {error}"
    status = answer.get("status") if isinstance(answer, dict) else None
    return plan_release(repl.plan, status), None


def _never_advanced(
    attempts: list[dict[str, Any]],
    elapsed: float,
    error: BaseException,
    status_failure: str | None,
) -> str:
    count = f"{len(attempts)} attempts over {int(elapsed)} s ({attempts[-1]['error']})"
    if _unreachable(error):
        seen = f"the node did not answer the probe in {count}"
    else:
        seen = (
            f"the probe ran but the advance failed in {count} and the tool's status "
            "explained no refusal"
            + (f" (status read: {status_failure})" if status_failure else "")
            + f"; {CONFLICT_MEANING}"
        )
    return (
        f"the held plan could not be advanced after the reset: {seen}; until the "
        "plan moves, the product's post-reset validation grades the held fabric "
        "state (a RESTART_NODE rung for a FAILURE state)"
    )


def _released(
    release: dict[str, Any], attempts: list[dict[str, Any]], seconds: int
) -> dict[str, Any]:
    """The advance's result for a refusal the tool explained; an ended tool
    (disarmed / unarmed) is INCONCLUSIVE with the 409's meaning."""

    kind = release["kind"]
    if kind in ENDED_RELEASES:
        raise VariantInconclusive(
            "the held plan could not be advanced after the reset: the replay tool "
            "refused the advance with HTTP 409 Conflict -- the replay is not on a "
            f"held step: {release['reason']}; nothing holds the fault and nothing "
            "serves the plan's tail, so the product's post-reset validation read "
            "the node's real fabric; harness/node state, not a product error"
        )
    return {
        "attempts": len(attempts),
        "seconds": seconds,
        "advanced": kind == "advanced",
        "release": release,
    }


def _advance_when_reachable(ctx: VariantContext, repl: DcgmReplay) -> dict[str, Any]:
    """Leave the held step, retrying while the node cannot answer the probe.

    The reset completes inside the quiesce window; on a HyperPod node kubelet
    is still down until RESTORE_GPU_SERVICES, and the probe exec fails with a
    transport ``HostProbeError`` for those seconds: retried, bounded. A probe
    that ran and exited 1 is another matter -- live 2026-10-27 (plain EKS) the
    tool answered the loopback ``POST /advance`` with HTTP 409 Conflict for 13
    attempts because it had aborted on the real post-reset fabric transition
    (3 -> 2) and was passing the real readings through -- so the runner reads
    the tool's ``/status`` and classifies the refusal (``plan_release``): an
    abort or a lapsed hold is the plan's release (the product grades the real
    fabric), a plan past the hold is an advance whose receipt was lost, an
    ended tool is INCONCLUSIVE with the 409's meaning, a plan still held is
    retried. A node that never answers leaves the plan at the held fault,
    which the product's post-reset validation then grades (a reboot rung for
    a FAILURE state), so the variant is INCONCLUSIVE with that said rather
    than graded on it. Returns ``attempts``, ``seconds``, ``advanced`` and
    ``release`` (``None`` for a plain advance).
    """

    started = ctx.clock()
    attempts: list[dict[str, Any]] = []
    status_failure: str | None = None
    while True:
        now = ctx.clock()
        try:
            receipt = repl.advance()
        except HostProbeError as error:
            attempts.append(
                {
                    "observed_at": now.isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            release = None
            if not _unreachable(error):
                release, status_failure = _refusal(repl)
            elapsed = (now - started).total_seconds()
            if release is not None:
                ctx.evidence.write(
                    "plan-advance", {"attempts": attempts, "release": release}
                )
                return _released(release, attempts, int(elapsed))
            ctx.evidence.write("plan-advance", {"attempts": attempts})
            if elapsed >= ADVANCE_RETRY_SECONDS:
                raise VariantInconclusive(
                    _never_advanced(attempts, elapsed, error, status_failure)
                ) from error
            ctx.sleep(ADVANCE_POLL_SECONDS)
            continue
        attempts.append({"observed_at": now.isoformat(), "ok": True})
        ctx.evidence.write("plan-advance", {"attempts": attempts, "receipt": receipt})
        return {
            "attempts": len(attempts),
            "seconds": int((now - started).total_seconds()),
            "advanced": True,
            "release": None,
        }


def _graded_escalation(
    ctx: VariantContext,
    repl: DcgmReplay,
    since: Any,
    incidents: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], list[str]]:
    """``recovery_failure``'s two stages and its notes, on the product's records.

    ``recovery_failure`` is judged on the record the product writes before
    and at the rung (``held_failure_errors``); ``escalation_exit`` on the
    settled record and the node (``escalation_exit_errors``). The evidence
    file ``fabric-escalation`` carries both readings.
    """

    escalation = wait_gpu_incident(
        ctx,
        since,
        lambda doc: incidents if rung_reached(doc, incidents) else [],
        timeout_seconds=ESCALATION_SECONDS,
        name="escalation",
    )
    boot_before = repl.node_boot_id
    ctx.evidence.write(
        "fabric-escalation",
        escalation_record(
            escalation, incidents, boot_before=boot_before, boot_now=node_boot_id(ctx)
        ),
    )
    stages = {"recovery_failure": held_failure_errors(escalation, incidents)}
    final = wait_gpu_incident(
        ctx,
        since,
        lambda doc: incidents if settled(doc, incidents) else [],
        timeout_seconds=SETTLE_SECONDS,
        name="settled",
    )
    boot_now = node_boot_id(ctx)
    node = ctx.regional.node_snapshot(ctx.settings.node)
    record = escalation_record(
        final, incidents, boot_before=boot_before, boot_now=boot_now
    )
    record["node_after"] = node_residual_isolation(node)
    ctx.evidence.write("fabric-escalation", record)
    stages["escalation_exit"] = escalation_exit_errors(
        final, incidents, boot_before=boot_before, boot_now=boot_now, node=node
    )
    return stages, escalation_notes(record)


def not_started(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["not_started"])


def in_progress(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["in_progress"])


def failed(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["failed"])


def full_reset(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["in_progress"])


def every_gpu_success(ctx: VariantContext) -> dict[str, Any]:
    # The recovery tail asserts SUCCESS on every graded GPU; a single-PID
    # Fabric Manager restart is not enough, VALIDATE_GPU waits for state 3.
    return _unsettled_reset(ctx, STATES["not_started"])


def recovery_failure(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["failed"], recovery_failure=True)


# --------------------------------------------------------------------------- #
# the escalation record: the rung, the held isolation, the exit
# --------------------------------------------------------------------------- #
def _node_executions(doc: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    """Executions of ``operation`` in every workflow of the node's record
    (the activity document is scoped to the node since the run began): a rung
    the whole-workflow path hands to a successor record is found as well."""

    return [
        {**item, "workflow_request_id": workflow.get("request_id")}
        for workflow in doc.get("workflows") or []
        for item in workflow.get("step_executions") or []
        if item.get("operation") == operation
    ]


def _moment(execution: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = parse_time(execution.get(key))
        if value is not None:
            return value
    return None


def _distinct_steps(executions: list[dict[str, Any]]) -> int:
    return len(
        {
            (item.get("workflow_request_id"), item.get("step_index"))
            for item in executions
        }
    )


def _incident_state(doc: dict[str, Any], incidents: list[dict[str, Any]]) -> str:
    wanted = {str(item.get("incident_id")) for item in incidents}
    for item in doc.get("incidents") or []:
        if str(item.get("incident_id")) in wanted:
            return str(item.get("state") or "")
    return str(incidents[0].get("state") or "") if incidents else ""


def _first_workflow(
    doc: dict[str, Any], incidents: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for incident in incidents:
        workflows = workflows_of(doc, str(incident.get("incident_id")))
        if workflows:
            return workflows[0]
    return None


def _escalated(doc: dict[str, Any], incidents: list[dict[str, Any]]) -> bool:
    return _incident_state(doc, incidents) in ESCALATED_STATES


def rung_reached(doc: dict[str, Any], incidents: list[dict[str, Any]]) -> bool:
    """The record can be judged: a RESTART_NODE rung is on it, the incident is
    with an operator, or the record ended without either."""

    return bool(
        _node_executions(doc, REBOOT)
        or _escalated(doc, incidents)
        or settled(doc, incidents)
    )


def held_failure_errors(
    doc: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    """What the record must say while the replay keeps the fabric failed.

    The post-reset validation FAILED (``nvlink_fabric_not_trained`` on
    VALIDATE_GPU, or the node's active finding on VALIDATE_FABRIC); the node
    was not released on that reading -- no RESTORE_SCHEDULING SUCCEEDED
    before the rung, none at all without one; and the rung followed: a
    RESTART_NODE execution (the reboot is the escalation), or an incident
    handed to an operator (ESCALATED / QUARANTINED) where the site does not
    reboot.
    """

    if not incidents:
        return []
    errors = []
    resets = [
        item
        for item in executions_of(doc, "RESET_ALL_GPUS_NVSWITCHES", incidents)
        if item.get("status") == "SUCCEEDED"
    ]
    reset_at = max((_moment(item, "updated_at") for item in resets), default=None)
    validations = [
        item
        for operation in POST_RESET_VALIDATIONS
        for item in executions_of(doc, operation, incidents)
    ]
    failed = [
        item
        for item in validations
        if item.get("status") == "FAILED"
        and (
            reset_at is None
            or _moment(item, "updated_at") is None
            or _moment(item, "updated_at") >= reset_at
        )
    ]
    if not failed:
        seen = ", ".join(
            f"{item.get('operation')} {item.get('status')}" for item in validations
        )
        errors.append(
            "the post-reset validation did not fail on the fabric the replay kept "
            "failed: no FAILED VALIDATE_GPU/VALIDATE_FABRIC after "
            f"RESET_ALL_GPUS_NVSWITCHES ({seen or 'no validation executed'})"
        )
    rungs = _node_executions(doc, REBOOT)
    rung_at = min(
        (
            moment
            for item in rungs
            if (moment := _moment(item, "started_at", "updated_at")) is not None
        ),
        default=None,
    )
    workflow = _first_workflow(doc, incidents)
    releases = [
        item
        for item in (workflow or {}).get("step_executions") or []
        if item.get("operation") == RELEASE and item.get("status") == "SUCCEEDED"
    ]
    for item in releases:
        released_at = _moment(item, "updated_at")
        if rung_at is None or released_at is None or released_at < rung_at:
            followed = (
                f", before the RESTART_NODE rung at {rung_at.isoformat()})"
                if rung_at is not None
                else ", no rung followed)"
            )
            errors.append(
                "scheduling was restored while the fabric still read failed "
                f"(RESTORE_SCHEDULING SUCCEEDED at {item.get('updated_at')}" + followed
            )
    if not rungs and not _escalated(doc, incidents):
        state = _incident_state(doc, incidents)
        status = (workflow or {}).get("status")
        errors.append(
            "a fabric that stayed failed after the reset did not escalate/hold "
            f"isolation (incident {state or 'unknown'}, workflow {status}, no "
            "RESTART_NODE rung)"
        )
    return errors


def escalation_exit_errors(
    doc: dict[str, Any],
    incidents: list[dict[str, Any]],
    *,
    boot_before: str,
    boot_now: str,
    node: dict[str, Any] | None = None,
) -> list[str]:
    """What the settled record and the node must say after the rung.

    No blind loop: exactly one RESET_ALL_GPUS_NVSWITCHES SUCCEEDED, at most one
    RESTART_NODE rung, no REPLACE_NODE (after the reboot the replay is gone, so
    a further rung means the real fabric failed or the replay survived). A
    SUCCEEDED RESTART_NODE changed the node's boot id. The record settled; a
    SUCCEEDED record is a RECOVERED incident on a released node, a FAILED or
    BLOCKED one an incident with an operator.
    """

    if not incidents:
        return []
    errors = []
    resets = [
        item
        for item in executions_of(doc, "RESET_ALL_GPUS_NVSWITCHES", incidents)
        if item.get("status") == "SUCCEEDED"
    ]
    if _distinct_steps(resets) != 1:
        errors.append(
            f"RESET_ALL_GPUS_NVSWITCHES succeeded {_distinct_steps(resets)} times "
            "(expected exactly one): the ladder looped on the reset"
        )
    rungs = _node_executions(doc, REBOOT)
    if _distinct_steps(rungs) > 1:
        errors.append(
            f"{_distinct_steps(rungs)} RESTART_NODE rungs on the node's records: "
            "a blind reboot loop"
        )
    replacements = _node_executions(doc, REPLACE)
    if replacements:
        errors.append(
            "the ladder went on to REPLACE_NODE "
            f"({', '.join(str(item.get('status')) for item in replacements)}): the "
            "reboot did not end the failed validation (the real fabric failed, or a "
            "replay survived the boot)"
        )
    rebooted = any(item.get("status") == "SUCCEEDED" for item in rungs)
    if rebooted and boot_before and boot_now and boot_before == boot_now:
        errors.append(
            f"RESTART_NODE SUCCEEDED but the node's boot id did not change ({boot_now})"
        )
    workflow = _first_workflow(doc, incidents)
    status = str((workflow or {}).get("status") or "")
    state = _incident_state(doc, incidents)
    if not settled(doc, incidents):
        errors.append(
            f"the escalation did not settle within {SETTLE_SECONDS} s "
            f"(workflow {status or 'missing'}, incident {state or 'unknown'})"
        )
        return errors
    if status == "SUCCEEDED":
        if state != "RECOVERED":
            errors.append(
                f"the record ended SUCCEEDED but the incident is {state or 'unknown'}, "
                "not RECOVERED"
            )
        residual = node_residual_isolation(node or {})
        if any(residual.values()):
            errors.append(
                f"the record ended SUCCEEDED but the node still carries {residual}"
            )
    elif state not in ESCALATED_STATES:
        errors.append(
            f"the record ended {status} but the incident is {state or 'unknown'}: "
            "neither recovered nor handed to an operator"
        )
    return errors


def reboot_exposure(
    doc: dict[str, Any],
    incidents: list[dict[str, Any]],
    *,
    boot_before: str,
    boot_now: str,
) -> list[str]:
    """Why a reset-only variant cannot claim "reset -> trained" on this record.

    A RESTART_NODE rung means the product's post-reset validation read the
    fabric state the plan still held -- the collector's first scrape after
    RESTORE_GPU_SERVICES is served the held value until the runner advances
    the plan, and a FAILURE state fails VALIDATE_GPU at once -- and rebooted
    the node by design. The claim was not observed; the product is not wrong.
    """

    reasons = []
    rungs = _node_executions(doc, REBOOT)
    if rungs:
        statuses = ", ".join(sorted({str(item.get("status")) for item in rungs}))
        reasons.append(
            f"the product escalated to RESTART_NODE ({statuses}) after the reset: "
            "its post-reset validation read the fabric state the plan still held "
            "(the collector's first scrape after RESTORE_GPU_SERVICES is served the "
            "held value until the runner advances the plan), so the reset -> trained "
            "claim was not observed; the rung is the product's designed response to "
            "what it read, not a product error -- rerun"
        )
    if boot_before and boot_now and boot_before != boot_now:
        reasons.append(
            f"the node rebooted during the variant (boot id {boot_before} -> "
            f"{boot_now}); the replay did not survive it"
        )
    return reasons


def escalation_record(
    doc: dict[str, Any],
    incidents: list[dict[str, Any]],
    *,
    boot_before: str,
    boot_now: str,
) -> dict[str, Any]:
    """The evidence a reader needs to see the ladder: the incident's state, its
    first record's status, every reset / validation / rung / release
    execution, and the boot ids."""

    workflow = _first_workflow(doc, incidents)
    wanted = {
        "RESET_ALL_GPUS_NVSWITCHES",
        *POST_RESET_VALIDATIONS,
        "VALIDATE_HOST",
        REBOOT,
        REPLACE,
        RELEASE,
    }
    executions = [
        {
            "step_index": item.get("step_index"),
            "operation": item.get("operation"),
            "status": item.get("status"),
            "started_at": item.get("started_at"),
            "updated_at": item.get("updated_at"),
            "error": str(item.get("error") or "")[:200] or None,
        }
        for item in (workflow or {}).get("step_executions") or []
        if item.get("operation") in wanted
    ]
    rungs = [
        {
            "workflow_request_id": item.get("workflow_request_id"),
            "operation": item.get("operation"),
            "status": item.get("status"),
            "started_at": item.get("started_at"),
            "updated_at": item.get("updated_at"),
        }
        for operation in (REBOOT, REPLACE)
        for item in _node_executions(doc, operation)
    ]
    return {
        "incident_id": str(incidents[0].get("incident_id")) if incidents else None,
        "incident_state": _incident_state(doc, incidents),
        "workflow_request_id": (workflow or {}).get("request_id"),
        "workflow_status": (workflow or {}).get("status"),
        "executions": sorted(executions, key=lambda item: (item["step_index"] or 0)),
        "rungs": rungs,
        "boot_id_before": boot_before,
        "boot_id_now": boot_now,
        "rebooted": bool(boot_before and boot_now and boot_before != boot_now),
    }


def escalation_notes(record: dict[str, Any]) -> list[str]:
    """What the settled record means for the reader, beside the verdict."""

    notes = []
    rungs = [item for item in record.get("rungs") or [] if item["operation"] == REBOOT]
    if rungs:
        notes.append(
            "the product's RESTART_NODE rung is the escalation: it rebooted the node"
            + (
                f" (boot id {record['boot_id_before']} -> {record['boot_id_now']})"
                if record.get("rebooted")
                else ""
            )
            + "; the replay lives under /run and cannot survive the boot, so the "
            "post-reboot validations graded the node's real fabric and the restore "
            "is the designed outcome. A fabric still failed after the reboot "
            "(REPLACE_NODE, then the operator hand-off) is not observable with a "
            "/run-based replay on either provider, by design"
        )
    elif record.get("incident_state") in ESCALATED_STATES:
        notes.append(
            f"the site held the node for an operator ({record['incident_state']}) "
            "instead of rebooting it; the incident is left to the operator exit"
        )
    return notes


# --------------------------------------------------------------------------- #
# below-window / ungraded / monitor-restart (no finding, no reset)
# --------------------------------------------------------------------------- #
def before_window(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    held = min(WINDOW - 30, _hold(ctx))
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                held_reads(survey.interval, held),
                put("fabric_manager_status", gpus, 2),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(held, name="under-window")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="under-window-check"
    )
    stages = {
        "no_finding": []
        if not fabric_incidents(activity)
        else ["an unsettled fabric under the window minted a finding"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["the 300 s window keeps boot-time training out"],
        extra={"gpus": gpus},
    )


def ungraded_product(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)[:1]
    since = utc_now()
    held = min(WINDOW + 60, _hold(ctx))
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                held_reads(survey.interval, held),
                put("fabric_manager_status", gpus, 2),
                label("fabric_manager_status", gpus, "modelName", "NVIDIA L40S"),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(held, name="ungraded")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="ungraded-check"
    )
    stages = {
        "not_graded": []
        if not fabric_incidents(activity)
        else ["a non-NVSwitch product was graded for fabric state"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["an L40S-class (no NVSwitch) product is not fabric-graded"],
        extra={"gpus": gpus},
    )


def monitor_restart(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    held = min(WINDOW - 60, _hold(ctx))
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                held_reads(survey.interval, 2 * held + RESTART_SECONDS),
                put("fabric_manager_status", gpus, 2),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(held, name="before-restart")
    restart = repl.collector_restart()
    ctx.hold(held, name="after-restart")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="restart-check"
    )
    stages = {
        "clock_reset": []
        if not fabric_incidents(activity)
        else ["the held-time clock did not restart with the collector"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            "the held-seconds clock restarts at the collector's first post-restart observation"
        ],
        extra={"gpus": gpus, "restart": restart},
    )


def _fabric_active(ctx: VariantContext, since: Any) -> bool:
    doc = state_now(ctx, since, names=["fabric_manager_status"])
    return any(
        item.get("canonical_name") == "fabric_manager_status"
        for item in doc.get("active_findings") or []
    )


RESET_EXPOSURE = (
    "; a RESTART_NODE rung (the post-reset validation read the held state before "
    "the plan advanced) is INCONCLUSIVE, not a product error"
)

VARIANTS = [
    gpu_spec(
        CASE,
        "not_started",
        "hold DCGM fabric state 1 past the 300 s window",
        "CRITICAL full GPU+NVSwitch reset; trained after" + RESET_EXPOSURE,
        not_started,
        risk="H",
        flags="X",
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "in_progress",
        "hold DCGM fabric state 2 past the window",
        "CRITICAL full fabric reset; trained after" + RESET_EXPOSURE,
        in_progress,
        risk="H",
        flags="X",
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "failed",
        "hold DCGM fabric state 4 (failure)",
        "CRITICAL full fabric reset at once; trained after (a FAILURE state fails "
        "the post-reset VALIDATE_GPU at once, so this variant is the most exposed "
        "to a late advance)" + RESET_EXPOSURE,
        failed,
        risk="H",
        flags="X",
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "before_window",
        "hold an unsettled fabric state under 300 s",
        "no finding before the window elapses",
        before_window,
        max_minutes=12,
    ),
    gpu_spec(
        CASE,
        "ungraded_product",
        "hold an unsettled state on a non-NVSwitch modelName",
        "an ungraded product is not fabric-graded",
        ungraded_product,
        max_minutes=12,
    ),
    gpu_spec(
        CASE,
        "monitor_restart",
        "restart the metrics collector mid-hold",
        "the held-time clock restarts; no premature finding",
        monitor_restart,
        max_minutes=16,
    ),
    gpu_spec(
        CASE,
        "full_reset",
        "hold an unsettled state and let the full reset run",
        "one reset of every GPU and NVSwitch inside the quiesce window"
        + RESET_EXPOSURE,
        full_reset,
        risk="H",
        flags="X",
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "every_gpu_success",
        "hold unsettled, reset, require SUCCESS on every GPU",
        "the post-reset VALIDATE_GPU waits for state 3 on every graded GPU, not one PID"
        + RESET_EXPOSURE,
        every_gpu_success,
        risk="H",
        flags="X",
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "recovery_failure",
        "keep the fabric failed after the reset",
        "the post-reset validation FAILS on the held FAILURE, the node stays "
        "isolated and the ladder's RESTART_NODE rung (or an operator hand-off) "
        "follows; exactly one reset, at most one reboot, no REPLACE_NODE; the "
        "reboot empties /run, so the post-reboot validations grade the real fabric "
        "and the incident RECOVERS -- the still-failed-after-reboot branch is not "
        "observable with a /run replay, by design",
        recovery_failure,
        risk="H",
        flags="X",
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET", "NODE_REBOOT"),
    ),
]
