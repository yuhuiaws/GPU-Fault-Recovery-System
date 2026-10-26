"""Building blocks the GF-GENERAL-LIVE category modules share.

The host survey that decides capability, the runner's own oracle sampler on
the node, the collector-window drop-in that adds a test mount or test
interface to the host collector's lists, the bounded waits on the product's
records, and the controlled readmission: the product's validated restore
(VALIDATE_GPU/HOST/FABRIC then RESTORE_SCHEDULING) is the only way a node or
an incident this runner caused is handed back -- a taint is never deleted by
hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.adapters.common import (
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.models import BlockedKind, WorkflowStatus, workflow_is_open
from scripts.e2e.regional.drill_incident_cleanup import isolation_evidence
from scripts.e2e.regional.general_live_fixture import (
    HOST_UNIT,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_store import node_activity, signal_key
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    chain_judgeable,
    finding_incidents,
    node_isolated,
    oracle_values,
    samples,
    settled,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import (
    WORKFLOW_BY_ID,
    WarmSpareLiveFixture,
)

ORACLE_POLL_SECONDS = 15
STORE_POLL_SECONDS = 20
#: Ingest, dispatch and the +-5 % health-summary jitter on top of a budget's
#: cadence terms (the margin 003 high_page_cache settled on live, 2026-10-25).
CHAIN_MARGIN_SECONDS = 180
#: The operator identity the fixture signs its incident closes with; the same
#: one its validated restores carry (``CREATE_RESTORE_WORKFLOW``).
OPERATOR_EXIT_OPERATOR = "acceptance-fixture"
#: How long an escalation's workflow may still take to settle after detection
#: (the ESCALATE_SUPPORT step and the record's BLOCKED/NEEDS_OPERATOR parking),
#: and how long two consecutive quiet reads must be apart to call it settled.
OPERATOR_EXIT_SETTLE_SECONDS = 300
OPERATOR_EXIT_QUIET_SECONDS = 15
#: How long the fresh reads after the close may take to show the exit: the
#: closure service settles the BLOCKED record in the same call, or leaves it to
#: the dispatcher's next sweep when its write was refused.
OPERATOR_EXIT_PROOF_SECONDS = 180


def sustained_detection_budget(
    ctx: VariantContext,
    sustain_seconds: float,
    *,
    summaries: int = 2,
    margin: int = CHAIN_MARGIN_SECONDS,
) -> int:
    """How long the product may take, from the oracle's crossing, to a judgeable chain.

    The host collector ships a breach once, as the crossing's edge
    (``collectors/host/collector.py`` ``_edge_reasons``/``collect_once``: a
    ``threshold:``/``sustained:`` reason stays in ``_active_edge_reasons`` while
    the breach holds and leaves ``delivery_reasons``), then only on its health
    summaries (``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS``, +-5 % jitter), and a
    summary-only batch with no finding is evaluated but kept as no raw
    evidence. The control plane mints a finding on the first *later* active
    sample at least the rule's sustain after the first
    (``store/shared/xid.py`` ``_next_health_signal_state``): the first summary
    at or past the sustain, up to sustain + one summary after the edge. The
    chain's validation is then judged on evidence newer than itself: one more
    summary. ``summaries`` is how many the chain still needs after the edge --
    two for a sustained rule, one for a rule with no sustain, whose finding is
    the edge batch itself. Live 2026-10-25 (003 high_page_cache, release
    13daa08b2227): a 720-s budget counted from before the ramp ended 482 s
    after the edge and the fill was stopped before the summary that could have
    claimed the finding; this budget is 1095 s on that site's cadence.
    """

    return int(
        sustain_seconds + summaries * ctx.host_summary + ctx.host_interval + margin
    )


def sustained_negative_window(
    ctx: VariantContext,
    sustain_seconds: float,
    *,
    margin: int = CHAIN_MARGIN_SECONDS,
) -> int:
    """The bounded "must not trigger" watch after a burst shorter than the sustain.

    A finding the product must not mint could only come from an active sample
    a health summary carries at least the sustain after the burst's edge; the
    window covers the sustain, one summary with its jitter and the collector
    interval, so the product has seen the burst's recovered edge and at least
    one summary after it before the negative verdict is read.
    """

    return int(sustain_seconds + ctx.host_summary + ctx.host_interval + margin)


def cadence_record(
    ctx: VariantContext,
    *,
    sustain_seconds: float,
    detection_budget_seconds: int,
    ramp_budget_seconds: int | None = None,
) -> dict[str, Any]:
    """The budgets and the cadence they came from, for the injection's evidence."""

    record: dict[str, Any] = {
        "sustain_seconds": sustain_seconds,
        "detection_budget_seconds": detection_budget_seconds,
        "host_summary_seconds": ctx.host_summary,
        "host_interval_seconds": ctx.host_interval,
    }
    if ramp_budget_seconds is not None:
        record["ramp_budget_seconds"] = ramp_budget_seconds
    return record


def run_survey(ctx: VariantContext) -> dict[str, Any]:
    """The node's capability and residue survey (read-only)."""

    value = ctx.node("survey", timeout=240)
    ctx.survey = value
    ctx.evidence.write("survey", value)
    residue = value.get("residue") or {}
    if any(
        residue.get(key)
        for key in (
            "page_cache_files",
            "paused_units",
            "paused_processes",
            "pause_dropins",
            "pause_hooks",
        )
    ) or residue.get("log_collector_enabled_by_run"):
        raise VariantNotRun(
            "FIXTURE_NOT_READY", "an earlier attempt of this run left node residue"
        )
    return value


class Oracle:
    """The runner's own reading of the raw source, kept in ``oracle.jsonl``."""

    def __init__(self, ctx: VariantContext, name: str) -> None:
        self.ctx = ctx
        self.name = name
        self.offset = 0
        self.records: list[dict[str, Any]] = []

    def start(
        self,
        seconds: int,
        *,
        interval: int = 5,
        cpu: bool = False,
        loadavg: bool = False,
        meminfo: bool = False,
        snmp: bool = False,
        statvfs: Sequence[str] = (),
        diskstats: Sequence[str] = (),
        netdev: Sequence[str] = (),
    ) -> dict[str, Any]:
        arguments = [
            "oracle-start",
            "--name",
            self.name,
            "--seconds",
            str(int(seconds)),
            "--interval",
            str(int(interval)),
        ]
        for flag, enabled in (
            ("--cpu", cpu),
            ("--loadavg", loadavg),
            ("--meminfo", meminfo),
            ("--snmp", snmp),
        ):
            if enabled:
                arguments.append(flag)
        for flag, values in (
            ("--statvfs", statvfs),
            ("--diskstats", diskstats),
            ("--netdev", netdev),
        ):
            for value in values:
                arguments += [flag, value]
        started = self.ctx.node(*arguments)
        self.ctx.cleanup.push(f"oracle-{self.name}", self.stop)
        return started

    def read(self) -> list[dict[str, Any]]:
        """New records since the previous read, also appended to the evidence."""

        fresh: list[dict[str, Any]] = []
        while True:
            value = self.ctx.node(
                "oracle-read", "--name", self.name, "--offset", str(self.offset)
            )
            records = list(value.get("records") or [])
            fresh.extend(records)
            self.offset = int(value.get("next_offset") or self.offset)
            if not records or self.offset >= int(value.get("total") or 0):
                break
        if fresh:
            self.records.extend(fresh)
            self.ctx.evidence.append_oracle(fresh, source=self.name)
        return fresh

    def stop(self) -> dict[str, Any]:
        self.read()
        return self.ctx.node("oracle-stop", "--name", self.name)


def await_oracle_crossing(
    ctx: VariantContext,
    oracle: Oracle,
    getter: Callable[[dict[str, Any]], Any],
    crossed: Callable[[Sequence[float]], bool],
    *,
    timeout_seconds: float,
    name: str,
    condition: str,
) -> list[float]:
    """Wait until the runner's own oracle establishes the condition.

    Detection is timed from here, never from the load's start: a fill or an
    allocator ramps for minutes on a 2 TiB host, and a budget counted from
    before the ramp ran out before the product's second summary (live
    2026-10-25, 003 high_page_cache). ``getter`` reads one value per oracle
    record, ``crossed`` judges the values read so far. The variant is
    INCONCLUSIVE "condition not established" -- never FAIL, whatever the
    product recorded -- when the oracle does not cross within ``timeout_seconds``.
    """

    values: list[float] = []

    def accept() -> bool | None:
        nonlocal values
        oracle.read()
        values = oracle_values(oracle.records, getter)
        return True if crossed(values) else None

    reached = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=ORACLE_POLL_SECONDS,
        name=name,
    )
    if reached is None:
        raise VariantInconclusive(
            f"condition not established: {condition} within {int(timeout_seconds)} s "
            f"(last {[round(value, 1) for value in values[-6:]]})"
        )
    return values


def open_window(
    ctx: VariantContext,
    *append_env: str,
    restore_seconds: int,
) -> dict[str, Any]:
    """Extend the host collector's lists through a ``/run`` drop-in window.

    The window probe arms its own deadman before it touches the unit; the
    runner registers the close first, so a half-open window is closed too.
    """

    arguments = ["open-window", "--run-id", ctx.run_id, "--unit", HOST_UNIT]
    for item in append_env:
        arguments += ["--append-env", item]
    arguments += ["--restore-seconds", str(int(restore_seconds))]
    ctx.cleanup.push("collector-window", lambda: close_window(ctx))
    opened = ctx.window(*arguments)
    ctx.evidence.write("window-open", opened)
    return opened


def close_window(ctx: VariantContext) -> dict[str, Any]:
    closed = ctx.window("close-window", "--run-id", ctx.run_id, "--unit", HOST_UNIT)
    ctx.evidence.write("window-close", closed)
    if (closed.get("after") or {}).get("ActiveState") != "active":
        raise RegionalFixtureError("the host collector is not active after the window")
    return closed


def activity(
    ctx: VariantContext,
    since: datetime,
    *,
    metrics: Iterable[str] = (),
    signal_keys: Iterable[str] = (),
    kinds: Sequence[str] = ("HOST_TELEMETRY",),
) -> dict[str, Any]:
    return node_activity(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since,
        metrics=metrics,
        signal_keys=signal_keys,
        kinds=kinds,
    )


def _after(text: Any, moment: datetime) -> bool:
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")) >= moment
    except ValueError:
        return False


def await_metric_recovery(
    ctx: VariantContext,
    metric: str,
    threshold: float,
    *,
    device: str | None = None,
    name: str | None = None,
    since: datetime | None = None,
) -> dict[str, Any]:
    """Fresh product evidence that ``metric`` fell back below ``threshold``.

    Waited for after the injection stopped, over two health-summary periods.
    Either proof is accepted: two fresh delivered samples below the threshold,
    or one (the recovered edge) whose health signal the product has cleared
    since -- the host collector delivers an unchanged normal reading only once
    per health summary, so the cleared signal is the product's own record of
    the recovery (live 2026-10-24: no second batch arrived within 14 minutes).
    """

    # ``since`` is when the injection was told to stop: the recovered edge is
    # often delivered while the stop is still in progress, before this wait
    # starts (live 2026-10-24 001.cpu_busy: the edge came 8 s before the
    # wait). Without it, look back two collector intervals.
    started = since or (utc_now() - timedelta(seconds=ctx.host_interval * 2))
    key = signal_key(
        ctx.settings.regional.cluster_id, ctx.settings.node, metric, device
    )

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics={metric}, signal_keys=[key])
        fresh = [
            sample
            for sample in samples(current, metric, device=device)
            if isinstance(sample.get("value"), (int, float))
            and sample["value"] < threshold
        ]
        signal = (current.get("health_signals") or {}).get(key) or {}
        cleared = signal.get("active") is False and _after(
            signal.get("observed_at"), started
        )
        if len(fresh) >= 2 or (fresh and cleared):
            return {"samples": fresh, "health_signal": signal}
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary * 2 + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=name or f"{metric}-recovery",
    )
    if value is None:
        label = metric + (f" on {device}" if device is not None else "")
        return {
            "errors": [
                f"{label} did not recover: no fresh sample below threshold with a "
                "cleared health signal, nor two fresh samples"
            ]
        }
    return {
        "errors": [],
        "fresh_samples": value["samples"],
        "health_signal": value["health_signal"],
    }


def wait_for_incident(
    ctx: VariantContext,
    since: datetime,
    metric: str,
    *,
    device: str | None = None,
    rule_id: str | None = None,
    validation: str | None = None,
    timeout_seconds: float,
    metrics: Iterable[str] = (),
    name: str = "detection",
) -> dict[str, Any]:
    """Poll the product until ``metric`` minted an incident (or the deadline).

    With ``validation``, until that step of the incident's workflow can be
    judged as well: a validation waits for evidence newer than itself and is
    recorded WAITING meanwhile (``chain_judgeable``), so a verdict read off
    the first snapshot that shows the incident judges a step still running
    (live 2026-10-25, 003.gpu_idle). The returned document's incidents and
    workflows are kept in the evidence as ``<name>-chain``.
    """

    wanted = {metric, *metrics}
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=wanted)
        if validation is not None:
            ready = chain_judgeable(
                last, metric, device=device, rule_id=rule_id, validation=validation
            )
        else:
            ready = bool(
                finding_incidents(last, metric, device=device, rule_id=rule_id)
            )
        return last if ready else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    document = value if value is not None else last
    ctx.evidence.write(
        f"{name}-chain",
        {
            "accepted": value is not None,
            "validation": validation,
            "incidents": document.get("incidents") or [],
            "workflows": document.get("workflows") or [],
        },
    )
    return document


def wait_settled(
    ctx: VariantContext,
    since: datetime,
    incident_ids: Sequence[str],
    *,
    timeout_seconds: float,
    metrics: Iterable[str] = (),
    name: str = "settled",
) -> dict[str, Any]:
    """Poll until every workflow of ``incident_ids`` is terminal.

    No incident is nothing to settle: one read, not a wait. ``settled`` needs
    a workflow to judge, so an empty id list could never accept and a variant
    that had already failed "no finding reached an incident" ran out the whole
    settle budget on top (live 2026-10-25, 003 high_page_cache: 12 minutes).
    """

    wanted = [str(item) for item in incident_ids]
    if not wanted:
        return activity(ctx, since, metrics=metrics)
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=metrics)
        return last if settled(last, wanted) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    return value if value is not None else last


def restore_incidents(
    ctx: VariantContext,
    incidents: Sequence[dict[str, Any]],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    """The product's validated restore for each incident this variant caused.

    Readmission is VALIDATE_GPU/HOST/FABRIC then RESTORE_SCHEDULING on fresh
    evidence (``orchestration/validated_restore.py``); a failed restore keeps
    the node isolated and fails the variant's cleanup. The incident that owns
    the node's isolation (its annotation) goes first.
    """

    restore = WarmSpareLiveFixture(ctx.regional, ctx.settings.hyperpod_cluster)
    profile = str(
        (ctx.survey.get("agent") or {}).get("runtime_profile_version")
        or ctx.survey.get("runtime_profile_version")
        or ""
    )
    if not profile:
        profile = runtime_profile_version(ctx)
    results = []
    node = ctx.regional.node_snapshot(ctx.settings.node)
    owner = str(
        (node.get("ownership_annotations") or {}).get("gpu-fault.io/incident-id") or ""
    )
    ordered = [owner] if owner else []
    ordered += [
        str(item["incident_id"]) for item in incidents if item["incident_id"] != owner
    ]
    for incident_id in ordered:
        state = restore.wait_incident_idle(incident_id)
        if (state.get("incident") or {}).get("state") == "RECOVERED":
            continue
        created = restore.create_restore_workflow(
            incident_id=incident_id,
            node=ctx.settings.node,
            profile_version=profile,
            reason=reason,
        )
        result = restore.wait_workflow_id(str(created["workflow_request_id"]))
        results.append(
            {
                "incident_id": incident_id,
                "workflow_request_id": created["workflow_request_id"],
                "status": result.get("status"),
                "operations": [
                    step.get("operation") for step in result.get("official_steps") or []
                ],
                "step_executions": [
                    {
                        "operation": item.get("operation"),
                        "status": item.get("status"),
                        "error": str(item.get("error") or "")[:200],
                    }
                    for item in result.get("step_executions") or []
                ],
            }
        )
        if result.get("status") != "SUCCEEDED":
            raise RegionalFixtureError(
                f"validated restore of {incident_id} ended {result.get('status')}"
            )
    after = ctx.regional.node_snapshot(ctx.settings.node)
    if node_isolated(after):
        raise RegionalFixtureError(
            f"{ctx.settings.node} is still isolated after the validated restore"
        )
    ctx.evidence.write("validated-restore", {"results": results, "node_after": after})
    return results


def node_residual_isolation(node: Mapping[str, Any]) -> dict[str, Any]:
    """What of the product's isolation ``node`` (a ``node_snapshot``) still carries:
    the ``gpu-fault.io/`` ownership annotations, its ``gpu-fault.io/`` taints and
    the cordon. Empty on every count means the node is released."""

    return {
        "ownership_annotations": dict(node.get("ownership_annotations") or {}),
        "taints": [
            item
            for item in node.get("taints") or []
            if str(item.get("key") or "").startswith("gpu-fault.io/")
        ],
        "unschedulable": bool(node.get("unschedulable")),
    }


def containment_missing(node: Mapping[str, Any], incident_id: str) -> list[str]:
    """Why ``node`` (a ``node_snapshot``) does not carry the containment a DRAIN
    or QUARANTINE chain of ``incident_id`` leaves: the cordon, the product's
    ``gpu-fault.io/quarantined`` taint owned by the incident (its digest form
    or the raw id an older release wrote, as ``incident_closure`` judges it) and
    the ``gpu-fault.io/incident-id`` ownership annotation naming it. Empty when
    all three are there.

    Live 2026-10-26 (019 remap_failure): the record was BLOCKED at compile time
    and superseded 35 ms later, nothing ever reached the node, and the variant
    would have closed an escalation whose chain never ran; the exit now refuses
    to proceed on a node the chain never isolated.
    """

    missing: list[str] = []
    if not node.get("unschedulable"):
        missing.append("not cordoned")
    owned = {incident_id, quarantine_taint_value(incident_id)}
    taints = [
        item for item in node.get("taints") or [] if item.get("key") == QUARANTINE_TAINT
    ]
    if not any(str(item.get("value") or "") in owned for item in taints):
        missing.append(
            f"no {QUARANTINE_TAINT} taint of the incident"
            + (" (the taint present is another incident's)" if taints else "")
        )
    owner = (node.get("ownership_annotations") or {}).get(ANNOTATION_INCIDENT)
    if owner != incident_id:
        missing.append(f"{ANNOTATION_INCIDENT} names {owner or 'no incident'}")
    return missing


def workflow_open(workflow: Mapping[str, Any]) -> bool:
    """Whether a workflow record (``WORKFLOW_BY_ID`` shape) still gates its node,
    by the product's own rule (``models.workflow_is_open``): executable, or
    BLOCKED with an occupying kind. A status the product does not name is not a
    proven exit either."""

    try:
        status = WorkflowStatus(str(workflow.get("status")))
        raw_kind = workflow.get("blocked_kind")
        kind = BlockedKind(str(raw_kind)) if raw_kind not in (None, "") else None
    except ValueError:
        return True
    return workflow_is_open(status, kind)


def operator_exit_problems(
    incident: Mapping[str, Any],
    workflow: Mapping[str, Any] | None,
    node: Mapping[str, Any],
) -> list[str]:
    """Why fresh reads do not yet show the operator exit of ``incident``.

    Empty when the incident is RECOVERED, its owning workflow (``None`` when the
    incident names none) is no longer open and ``node`` carries no residual
    isolation; otherwise one line per failing read, naming the state, the
    workflow's status/kind, or the annotation keys, taint keys and cordon left.
    """

    problems: list[str] = []
    incident_id = str(incident.get("incident_id") or "")
    state = str(incident.get("state") or "")
    if state != "RECOVERED":
        problems.append(
            f"incident {incident_id} is {state or 'unknown'}, not RECOVERED"
        )
    if workflow is not None and workflow_open(workflow):
        status = str(workflow.get("status") or "unknown")
        kind = workflow.get("blocked_kind")
        if kind:
            status = f"{status}/{kind}"
        problems.append(
            f"workflow {workflow.get('request_id')} of {incident_id} is still open "
            f"({status})"
        )
    residue = node_residual_isolation(node)
    name = str(node.get("name") or "the node")
    if residue["ownership_annotations"]:
        keys = ", ".join(sorted(residue["ownership_annotations"]))
        problems.append(f"{name} still carries the ownership annotations {keys}")
    if residue["taints"]:
        keys = ", ".join(sorted(str(item.get("key")) for item in residue["taints"]))
        problems.append(f"{name} still carries the taints {keys}")
    if residue["unschedulable"]:
        problems.append(f"{name} is still cordoned (unschedulable)")
    return problems


def operator_close_incidents(
    ctx: VariantContext,
    incidents: Sequence[dict[str, Any]],
    *,
    reason: str,
    expect_isolation: bool = True,
) -> list[dict[str, Any]]:
    """The product's operator exit for each ESCALATED incident this variant caused.

    An RMA-class DRAIN chain (``orchestration/families/health.py``) ends in
    ESCALATE_SUPPORT with the node cordoned and tainted on purpose -- there is no
    RESTORE_SCHEDULING, the record parks BLOCKED/NEEDS_OPERATOR and the incident
    is ESCALATED, "the one state an operator may close by hand". The validated
    restore is the wrong exit for it and the product refuses it ("workflow still
    owns recovery; operator hold retained"; live 2026-10-26, 019 remap_failure on
    gpu-eks-cluster). The documented exit is the operator's close, and this is
    it, performed from inside the CPU Pod: once the incident's workflow has
    settled (``WarmSpareLiveFixture.wait_incident_idle``), the incident is closed
    through ``IncidentClosureService.close_incident`` -- the call
    ``gpu-fault-admin workflow-reconcile --close-incident`` makes -- with the
    node's isolation evidence the CLI's evidence pass would read, signed
    ``OPERATOR_EXIT_OPERATOR`` with the run id as the reference. A refusal is the
    service's own wording and is not retried. The exit is then proven on fresh
    reads, within ``OPERATOR_EXIT_PROOF_SECONDS``: the incident RECOVERED, its
    workflow no longer open, and no ownership annotation, quarantine taint or
    cordon left on the node (``operator_exit_problems``); what is still there is
    the error. An incident that is not ESCALATED when it settles was not left to
    an operator, which is an error too, never something to restore around; and
    with ``expect_isolation`` (a containment chain) the node must carry the
    chain's cordon, taint and ownership annotation before the close
    (``containment_missing``) -- a chain that never isolated the node is the
    error, and nothing is closed (live 2026-10-26: the DRAIN record was blocked
    at compile time and the node was never touched).
    """

    warm = WarmSpareLiveFixture(ctx.regional, ctx.settings.hyperpod_cluster)
    node = ctx.settings.node
    results: list[dict[str, Any]] = []
    ordered = list(dict.fromkeys(str(item["incident_id"]) for item in incidents))
    for index, incident_id in enumerate(ordered):
        settled_incident = warm.wait_incident_idle(
            incident_id,
            timeout_seconds=OPERATOR_EXIT_SETTLE_SECONDS,
            quiet_seconds=OPERATOR_EXIT_QUIET_SECONDS,
        )
        before = ctx.regional.node_snapshot(node)
        workflow_id = str(settled_incident.get("workflow_request_id") or "")
        entry: dict[str, Any] = {
            "incident_id": incident_id,
            "workflow_request_id": workflow_id or None,
            "state_before": settled_incident.get("state"),
            "node_before": node_residual_isolation(before),
            "operator": OPERATOR_EXIT_OPERATOR,
            "reference": ctx.run_id,
        }
        results.append(entry)
        ctx.evidence.write("operator-exit", {"node": node, "results": results})
        missing = containment_missing(before, incident_id) if expect_isolation else []
        if missing:
            raise RegionalFixtureError(
                f"chain never isolated the node {node} for {incident_id}: "
                + "; ".join(missing)
            )
        state = str(settled_incident.get("state") or "")
        if state != "ESCALATED":
            raise RegionalFixtureError(
                f"incident {incident_id} settled {state or 'unknown'}, not "
                "ESCALATED; the operator exit applies to an escalation left to an "
                "operator"
                + (" (nothing was left to close)" if state == "RECOVERED" else "")
            )
        outcome = warm.close_incident_with_evidence(
            incident_id,
            reason=reason,
            operator=OPERATOR_EXIT_OPERATOR,
            reference=ctx.run_id,
            evidence=[isolation_evidence(node, before)],
        )
        entry["close"] = outcome
        ctx.evidence.write("operator-exit", {"node": node, "results": results})
        if outcome.get("refusal") or not outcome.get("closed"):
            raise RegionalFixtureError(
                f"operator close of {incident_id} was refused: "
                f"{outcome.get('refusal') or 'the service closed nothing'}"
            )
        last: dict[str, Any] = {}

        def proven() -> dict[str, Any] | None:
            nonlocal last
            fresh_incident = warm.incident_by_id(incident_id)
            fresh_workflow = (
                ctx.regional.cpu_python(WORKFLOW_BY_ID, workflow_id)
                if workflow_id
                else None
            )
            after = ctx.regional.node_snapshot(node)
            problems = operator_exit_problems(fresh_incident, fresh_workflow, after)
            last = {
                "state_after": fresh_incident.get("state"),
                "workflow_status_after": (
                    fresh_workflow.get("status") if fresh_workflow else None
                ),
                "workflow_blocked_kind_after": (
                    fresh_workflow.get("blocked_kind") if fresh_workflow else None
                ),
                "node_after": node_residual_isolation(after),
                "problems": problems,
            }
            return last if not problems else None

        proof = ctx.wait_until(
            proven,
            timeout_seconds=OPERATOR_EXIT_PROOF_SECONDS,
            poll_seconds=STORE_POLL_SECONDS,
            name=f"operator-exit-{index}",
        )
        entry.update(last)
        ctx.evidence.write("operator-exit", {"node": node, "results": results})
        if proof is None:
            raise RegionalFixtureError(
                f"the operator close of {incident_id} is not proven after "
                f"{OPERATOR_EXIT_PROOF_SECONDS} s: "
                + "; ".join(last.get("problems") or [])
            )
    return results


def runtime_profile_version(ctx: VariantContext) -> str:
    """The Runtime Profile version the target node's Agent runs."""

    from scripts.e2e.regional.general_live_store import preflight_state

    state = preflight_state(
        ctx.regional, ctx.settings.regional.cluster_id, ctx.settings.node
    )
    value = str((state.get("agent") or {}).get("runtime_profile_version") or "")
    if not value:
        raise RegionalFixtureError("the target node's Agent names no Runtime Profile")
    return value


def open_incident_ids(activity_document: dict[str, Any]) -> list[str]:
    return [
        str(item["incident_id"])
        for item in activity_document.get("incidents") or []
        if item.get("state") not in {"RECOVERED"}
    ]


def unsettled_workflows(activity_document: dict[str, Any]) -> list[str]:
    return [
        str(item.get("request_id"))
        for item in activity_document.get("workflows") or []
        if str(item.get("status")) not in TERMINAL_WORKFLOWS
    ]


__all__ = [
    "CHAIN_MARGIN_SECONDS",
    "OPERATOR_EXIT_OPERATOR",
    "OPERATOR_EXIT_PROOF_SECONDS",
    "OPERATOR_EXIT_QUIET_SECONDS",
    "OPERATOR_EXIT_SETTLE_SECONDS",
    "Oracle",
    "activity",
    "await_metric_recovery",
    "await_oracle_crossing",
    "cadence_record",
    "close_window",
    "containment_missing",
    "node_residual_isolation",
    "open_incident_ids",
    "open_window",
    "operator_close_incidents",
    "operator_exit_problems",
    "restore_incidents",
    "run_survey",
    "runtime_profile_version",
    "sustained_detection_budget",
    "sustained_negative_window",
    "unsettled_workflows",
    "wait_for_incident",
    "wait_settled",
    "workflow_open",
]
