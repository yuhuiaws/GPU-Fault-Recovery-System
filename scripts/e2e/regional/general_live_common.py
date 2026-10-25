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

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    HOST_UNIT,
    VariantContext,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_store import node_activity, signal_key
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    chain_judgeable,
    finding_incidents,
    node_isolated,
    samples,
    settled,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

ORACLE_POLL_SECONDS = 15
STORE_POLL_SECONDS = 20


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
    "Oracle",
    "activity",
    "await_metric_recovery",
    "close_window",
    "open_incident_ids",
    "open_window",
    "restore_incidents",
    "run_survey",
    "runtime_profile_version",
    "unsettled_workflows",
    "wait_for_incident",
    "wait_settled",
]
