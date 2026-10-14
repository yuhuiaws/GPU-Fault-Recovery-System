"""``gpu-fault-admin aurora-failback``: the writer back beside the control plane.

A failover moves the Aurora writer to whichever member RDS promotes and nothing
moves it back (``aurora_placement``): after HA-010's injected failover the
writer ran in another zone than every control-plane Pod until an operator failed
back by hand, with an ``aws rds failover-db-cluster`` the runbook otherwise
forbids. This verb is that failback, supported and never automatic -- ``status``
names it as the remedy of its placement finding, nothing calls it.

Plan (no ``--confirm``) is read-only: the placement (the writer, every member's
zone, status and promotion tier, the zones of the nodes running gpu-fault-api-ha
and gpu-fault-control-worker) and the target, the preferred member: available,
in the first zone of bootstrap's ``writer_first_zones`` (the zone with the most
of those Pods that holds a member), lowest promotion tier first -- the instance
bootstrap created as the writer at tier 0.

Execute (``--confirm AURORA_FAILBACK`` plus an approved ``--reference``) takes
the site operation lock, so it refuses while a release or configuration
mutation runs, and refuses while work is in flight by the checks the site
applies before touching a live control plane: an open release transaction and
PENDING/LEASED/WAITING remote commands (``rotate_token.ensure_rotation_allowed``,
the guard of join-cluster, remove-cluster and rotate-token), and an active
destructive workflow (the deploy preflight's ``workflow_safety``). A failover is
a write outage of tens of seconds; it is not started under a node action. The
journal ``<state-dir>/aurora-failback/<run-id>.json`` records ``PLANNED`` before
the request, which is bootstrap's own step
(``bootstrap_aurora.ensure_writer_zone_affinity``: ``failover-db-cluster
--target-db-instance-identifier`` and a bounded wait until the cluster is
``available`` with the target as its one writer), then ``CLUSTER_AVAILABLE``,
then ``CONTROL_PLANE_HEALTHY`` once every CPU Deployment's current Pods are
Ready, each registry runtime is ready on one generation
(``wait_control_plane_ready``) and a fresh store read through the ingress Pod
answers. A writer already placed is a ``NOOP``; a rerun after an interrupted or
failed execute resumes that journal when the writer is already placed (only the
waits remain) and supersedes it otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aurora_placement import (
    CONFIRMATION as CONFIRMATION,
    READ_ERRORS,
    VERB as VERB,
    WriterPlacement,
    placement_problem,
    read_writer_placement,
)
from gpu_fault.admin.bootstrap_aurora import ensure_writer_zone_affinity
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.release_engine import build_release
from gpu_fault.admin.release_state import live_release_state
from gpu_fault.admin.rotate_token import (
    release_transaction_open,
    site_process_environment,
)
from gpu_fault.admin.site import RenderedSite, SiteConfigError, load_site
from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_control_plane_ready import (
    wait_control_plane_ready,
)
from gpu_fault_release.regional_release_state import remote_command_stats
from gpu_fault_release.regional_release_workflow_safety import (
    workflow_safety_snapshot,
)

STATE_ROOT = "aurora-failback"
JOURNAL_SCHEMA_VERSION = 1
# The registry stale window (90 s) plus a refresh, with room for slow probes.
CONTROL_PLANE_HEALTHY_TIMEOUT_SECONDS = 300.0
STORE_READ_RETRY_SECONDS = 10.0
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_NOOP = "NOOP"
STATUS_SUPERSEDED = "SUPERSEDED"
STEP_PLANNED = "PLANNED"
STEP_CLUSTER_AVAILABLE = "CLUSTER_AVAILABLE"
STEP_CONTROL_PLANE_HEALTHY = "CONTROL_PLANE_HEALTHY"
SAFETY_COUNTS = (
    "blocker_count",
    "resolved_blocked_count",
    "abandoned_generation_count",
    "compile_blocked_count",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _operator() -> str:
    return operator_identity.resolve_operator_identity(
        fallback=operator_identity.local_operator_identity()
    )


def approved_reference(reference: str | None) -> str:
    normalized = (reference or "").strip()
    if not normalized:
        raise BootstrapError(
            f"{VERB} requires --reference (an approved change or "
            "maintenance-window reference)"
        )
    if not REFERENCE_PATTERN.fullmatch(normalized):
        raise BootstrapError(f"{VERB} reference is invalid")
    return normalized


def journal_directory(site: RenderedSite) -> Path:
    return site.source.parent / STATE_ROOT


def unfinished_journals(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Executes that journaled ``PLANNED`` and never reached a healthy control
    plane (interrupted, or failed on the way), newest first."""

    found: list[tuple[Path, dict[str, Any]]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BootstrapError(f"{VERB} journal {path} is unreadable: {exc}") from exc
        if not isinstance(record, dict):
            raise BootstrapError(f"{VERB} journal {path} is not a JSON object")
        steps = record.get("steps") or {}
        if (
            record.get("status") in {STATUS_IN_PROGRESS, STATUS_FAILED}
            and STEP_PLANNED in steps
            and STEP_CONTROL_PLANE_HEALTHY not in steps
        ):
            found.append((path, record))
    return found


@dataclass
class _Journal:
    path: Path
    record: dict[str, Any]
    now: Callable[[], datetime]

    def step(self, name: str, **fields: Any) -> None:
        self.record.update(fields)
        self.record.setdefault("steps", {})[name] = self.now().isoformat()
        write_json_atomic(self.path, self.record)

    def finish(self, status: str, **fields: Any) -> None:
        self.record.update(fields, status=status, finished_at=self.now().isoformat())
        write_json_atomic(self.path, self.record)


def require_quiet_site(site: RenderedSite, release: Any) -> dict[str, Any]:
    """Refuse while work is in flight; return what each check read."""

    try:
        state = live_release_state(site)
    except SiteConfigError as exc:
        raise BootstrapError(
            f"{VERB} refused: cannot read the live release state: {exc}"
        ) from exc
    phase = release_transaction_open(state)
    if phase is not None:
        raise BootstrapError(
            f"{VERB} refused: a release transaction is open (phase {phase}); "
            "finish, resume or roll it back first"
        )
    try:
        if not release._remote_commands_are_idle():
            raise BootstrapError(
                f"{VERB} refused: remote commands are PENDING/LEASED/WAITING; a "
                "failover is a write outage of tens of seconds and is not started "
                "under a node action in flight -- rerun when the site is idle"
            )
        safety = workflow_safety_snapshot(release)
    except ReleaseError as exc:
        raise BootstrapError(f"{VERB} refused: {exc}") from exc
    return {
        "release_id": state.get("release_id"),
        "release_phase": state.get("phase"),
        "remote_commands_idle": True,
        "workflow_safety": {key: safety.get(key) for key in SAFETY_COUNTS},
    }


def wait_control_plane_healthy(
    release: Any,
    *,
    timeout_seconds: float = CONTROL_PLANE_HEALTHY_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Every CPU Deployment ready on one registry generation, then a fresh
    store read through the ingress Pod: the new writer answers the control
    plane's own connection."""

    deadline = monotonic() + timeout_seconds
    ready = wait_control_plane_ready(
        release, timeout_seconds=timeout_seconds, sleep=sleep, monotonic=monotonic
    )
    while True:
        try:
            stats = remote_command_stats(release)
            break
        except ReleaseError as exc:
            if monotonic() >= deadline:
                raise ReleaseError(
                    "the control-plane store did not answer within "
                    f"{timeout_seconds:.0f} s of the failback: {exc}"
                ) from exc
            sleep(STORE_READ_RETRY_SECONDS)
    return {**ready, "open_remote_commands": stats.get("open_by_cluster") or {}}


def plan_failback(
    site: RenderedSite, *, reference: str, runner: CommandRunner | None = None
) -> dict[str, Any]:
    """The read-only plan: placement, target and what execute would do."""

    placement = read_writer_placement(
        site, ReadOnlyProbeRunner(runner or CommandRunner())
    )
    unfinished = unfinished_journals(journal_directory(site))
    target = placement.preferred()
    problem = placement_problem(placement)
    action = "none"
    if problem is not None:
        reason = problem
    elif placement.cluster_status != "available":
        action, reason = "wait", f"Aurora cluster is {placement.cluster_status}"
    elif placement.placed:
        reason = "the writer already runs in the preferred zone" + (
            "; execute finishes the unfinished journal's waits" if unfinished else ""
        )
    elif target is None:
        reason = _no_target(placement)
    else:
        action, reason = (
            "failover",
            (
                f"the writer runs outside {placement.preferred_zone}, the zone with the "
                "most control-plane Pods"
            ),
        )
    return {
        "mode": "plan",
        "reference": reference,
        "action": action,
        "reason": reason,
        "target": target.evidence() if action == "failover" and target else None,
        "placement": placement.evidence(),
        "unfinished_journal": str(unfinished[0][0]) if unfinished else None,
        "execute": (
            f"gpu-fault-admin {VERB} --state-dir {site.source.parent} "
            f"--reference {reference} --confirm {CONFIRMATION}"
        ),
    }


def _no_target(placement: WriterPlacement) -> str:
    zone = placement.preferred_zone
    if zone is None:
        return "no Aurora member is in a zone running a control-plane Pod"
    return f"no available Aurora member is in {zone} to fail back to"


def _supersede(
    journals: list[tuple[Path, dict[str, Any]]],
    *,
    placement: WriterPlacement,
    now: Callable[[], datetime],
) -> None:
    for path, record in journals:
        _Journal(path, record, now).finish(
            STATUS_SUPERSEDED, superseded_with=placement.evidence()
        )


def _finish(
    journal: _Journal,
    site: RenderedSite,
    runner: CommandRunner,
    release: Any,
    *,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """The waits after the request: placement re-read, then the control plane."""

    try:
        after = read_writer_placement(site, runner)
        evidence = after.evidence()
        if after.cluster_status != "available" or not after.placed:
            raise BootstrapError(
                f"Aurora cluster is {after.cluster_status} with writer "
                f"{evidence['writer']} in {evidence['writer_zone']}, not in the "
                f"preferred zone {evidence['preferred_zone']}"
            )
        journal.step(STEP_CLUSTER_AVAILABLE, after=evidence)
        control_plane = wait_control_plane_healthy(release, sleep=sleep)
        journal.step(STEP_CONTROL_PLANE_HEALTHY, control_plane=control_plane)
    except READ_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
        journal.finish(STATUS_FAILED, error=error)
        raise BootstrapError(
            f"{VERB} failed: {error}; journal {journal.path} (rerun to resume)"
        ) from exc
    journal.finish(STATUS_COMPLETED)
    return {**journal.record, "journal": str(journal.path)}


def run_aurora_failback(
    site: RenderedSite,
    *,
    reference: str,
    runner: CommandRunner | None = None,
    release_factory: Callable[[RenderedSite], Any] = build_release,
    now: Callable[[], datetime] = _utc_now,
    identity: Callable[[], str] = _operator,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute: refuse, resume, NOOP or fail back (the caller holds the lock)."""

    runner = runner or CommandRunner()
    directory = journal_directory(site)
    placement = read_writer_placement(site, runner)
    problem = placement_problem(placement) or (
        None
        if placement.cluster_status == "available"
        else f"Aurora cluster {placement.cluster_id} is {placement.cluster_status}; "
        "a failover may still be settling, rerun once it is available"
    )
    if problem is not None:
        raise BootstrapError(f"{VERB} refused: {problem}")
    unfinished = unfinished_journals(directory)
    if unfinished and placement.placed:
        # Only the waits of that run remain: the writer already moved.
        _supersede(unfinished[1:], placement=placement, now=now)
        resumed = _Journal(*unfinished[0], now)
        resumed.record.setdefault("resumed_at", []).append(now().isoformat())
        return _finish(resumed, site, runner, release_factory(site), sleep=sleep)
    _supersede(unfinished, placement=placement, now=now)
    started = now()
    record: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "run_id": f"{started:%Y%m%dT%H%M%S%fZ}",
        "reference": reference,
        "operator": identity(),
        "started_at": started.isoformat(),
        "before": placement.evidence(),
    }
    path = directory / f"{record['run_id']}.json"
    if placement.placed:
        record.update(status=STATUS_NOOP, finished_at=started.isoformat())
        write_json_atomic(path, record)
        return {**record, "journal": str(path)}
    target = placement.preferred()
    if target is None:
        raise BootstrapError(f"{VERB} refused: {_no_target(placement)}")
    release = release_factory(site)
    guards = require_quiet_site(site, release)
    journal = _Journal(path, {**record, "status": STATUS_IN_PROGRESS}, now)
    journal.step(STEP_PLANNED, target=target.evidence(), guards=guards)
    try:
        affinity = ensure_writer_zone_affinity(
            runner,
            aws_region=str(site.release_config["aws_region"]),
            cluster_id=placement.cluster_id,
            instance_ids=[target.instance_id],
            control_plane_zones=[target.availability_zone],
            sleep=sleep,
        )
    except READ_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
        journal.finish(STATUS_FAILED, error=error)
        raise BootstrapError(f"{VERB} failed: {error}; journal {path}") from exc
    journal.record["failover"] = affinity
    return _finish(journal, site, runner, release, sleep=sleep)


def add_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin aurora-failback``; the CLI passes site options."""

    command = commands.add_parser(
        VERB,
        usage=(
            f"gpu-fault-admin {VERB} --state-dir STATE_DIR --reference CHANGE_ID "
            f"[--confirm {CONFIRMATION}]"
        ),
        help=(
            "fail the Aurora writer back to the preferred member (available, in "
            "the zone with the most CPU control-plane Pods, lowest promotion tier) "
            "after a failover left it elsewhere: without --confirm prints the placement and "
            f"the target and changes nothing; --confirm {CONFIRMATION} runs it "
            "under the site operation lock, refused while a release transaction "
            "is open, a remote command is in flight or a destructive workflow is "
            f"active, journaled under <state-dir>/{STATE_ROOT}/, and waits for "
            "the cluster to be available and the control plane ready. Never "
            "run automatically"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--reference",
        required=True,
        metavar="CHANGE_ID",
        help="approved change or maintenance-window reference recorded in the journal",
    )
    command.add_argument(
        "--confirm",
        default=None,
        metavar="TOKEN",
        help=(
            f"execute requires exactly {CONFIRMATION}; without it the command "
            "prints the plan and changes nothing"
        ),
    )


def run_command(
    arguments: argparse.Namespace, managed_site_file: Callable[..., Path | None]
) -> int:
    site_file = managed_site_file(arguments, command=VERB)
    if site_file is None:
        raise SiteConfigError(f"{VERB} requires --state-dir")
    site = load_site(site_file, repository_root=arguments.repo_root)
    reference = approved_reference(arguments.reference)
    confirmation = getattr(arguments, "confirm", None)
    if confirmation is None:
        plan = plan_failback(site, reference=reference)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if confirmation != CONFIRMATION:
        raise BootstrapError(f"{VERB} requires --confirm {CONFIRMATION}")
    with administrator_operation_lock(site.source.parent):
        site = reload_site_for_mutation(site)
        # The engine primitives read the site's environment, as rotate-token's do.
        with site_process_environment(site):
            result = run_aurora_failback(site, reference=reference)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
