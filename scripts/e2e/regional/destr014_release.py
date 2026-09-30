"""DESTR-014 ``--release``: the documented operator handling of a held attempt.

A PASS/BLOCKED attempt of the unknown-reboot drill leaves, by design, an
operator hold: a BLOCKED/NEEDS_OPERATOR record, isolated nodes (one of them
usually re-quarantined by the support-after incident the escalation opened),
the drill PyTorchJob, an armed device holder on the fault node and two open env
windows. This module performs the release the specification describes, in
order and idempotently, through the repository's admin CLI and fixtures:

1. ``submit-remediation --disposition confirm-node-action`` for every node
   whose boot id changed since the plan (the reboot outcome is now known); a
   rebooted node that carries no unresolved action (the fault node, whose
   branch failed on a known outcome) is refused by the product and skipped;
2. ``submit-remediation --disposition restore`` for the case incident: the
   confirmed record stays BLOCKED until the product's own restore has run
   (live 2026-09-24 a3: "the record stays BLOCKED until --disposition restore
   and workflow-reconcile"); BLOCKED is not an open workflow for this verb,
   while the fixture's idle wait counts it as active -- so the verb, not a
   wait, releases the record. Refused with "no node ... is still isolated by
   it" when the escalation's support-after incident took the node over; that
   is skipped and step 3 restores through the owner;
3. wait for the restore workflow, then a validated restore for every node the
   case (or its escalation) still left isolated, through the incident that
   owns the isolation (``destr014_cleanup._restore_isolated_nodes``:
   QUARANTINED or ESCALATED owner alike);
3b. ``workflow-reconcile --incident-id`` for the case incident (and the
   follow-up incident when it has one) -- eligible only once the incident is
   RECOVERED with a verified restore successor, hence after steps 2-3;
4. delete the drill's PyTorchJob by the identity the attempt pinned (manifest
   kind/name plus the case's ``gpu-fault.io/job-id`` label) -- the managed
   workload fixture only deletes what it submitted itself, and a later
   process never holds the run's ownership nonce (live 2026-09-24: the job
   survived two releases);
5. disarm the fault-node device holder and retire the probes;
6. close both env windows through their own records;
7. close the drill incidents that are now operator-closable.

Identity is checked first: the live release id and both Node UIDs must be the
ones the plan recorded (boot ids are expected to differ -- that is the point).
Nothing here reads a plan mode or injects anything; the case's own journal,
plan and result under ``<run-dir>/cases/GF-REGIONAL-DESTR-014`` are the input.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from scripts.e2e.regional import control_plane_env_window as control_window
from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.destr014_cleanup import (
    _restore_isolated_nodes as restore_isolated_nodes,
)
from scripts.e2e.regional.managed_workload_fixture import (
    RESOURCE_APIS,
    delete_resource,
    read_resource,
    resource_identity,
)
from scripts.e2e.regional.drill_incident_cleanup import (
    close_drill_incidents,
    residual_incidents,
)
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

if TYPE_CHECKING:
    from scripts.e2e.regional.run_destr014_branch_exhaustion import Settings

CASE_ID = "GF-REGIONAL-DESTR-014"
RELEASE_CONFIRMATION = "RELEASE_GF_REGIONAL_DESTR_014_HOLD"
RELEASE_REFERENCE = "DESTR-014-operator-release"
# Product refusals the release treats as "nothing to do here" (admin CLI texts).
NO_UNRESOLVED_ACTION = "has an unresolved node action on node"
NO_NODE_ISOLATED_BY_IT = "is still isolated by it"
RELEASE_RECORD = "destr014-release.json"
JOURNAL_FILE = "destr014-recovery-journal.json"
REPO_ROOT = Path(__file__).resolve().parents[3]

# The admin CLI from this checkout's venv, without the deploy-host binding
# redirect: argument parsing, the mutating command log, the operation lock and
# the handler are the CLI's own code path.
ADMIN_DIRECT = """
import sys
from gpu_fault.admin.cli import (
    ADMIN_LOG_KIND_MUTATING, ADMIN_LOG_KIND_READONLY, READONLY_COMMANDS,
    _run_reporting_failures, command_log, is_readonly_command, parser,
)
arguments = parser().parse_args(sys.argv[1:])
command = str(getattr(arguments, "command", "admin"))
kind = (
    ADMIN_LOG_KIND_READONLY
    if is_readonly_command(arguments, readonly_commands=READONLY_COMMANDS)
    else ADMIN_LOG_KIND_MUTATING
)
with command_log(getattr(arguments, "state_dir", None), command=command, kind=kind):
    raise SystemExit(_run_reporting_failures(arguments))
"""


@dataclass
class ReleaseInputs:
    """What the case directory says about the attempt being released."""

    case_dir: Path
    plan_identity: dict[str, Any]
    journal: dict[str, Any]
    result: dict[str, Any]
    run_id: str
    incident_id: str
    follow_up_incident_id: str
    profile_version: str
    holder_armed: bool


@dataclass
class ReleaseContext:
    settings: Settings
    inputs: ReleaseInputs
    regional: Any
    warm: Any
    state_dir: Path
    fault_probe: Any
    workload: Any | None
    admin: Callable[..., dict[str, Any]]
    report: dict[str, Any] = field(default_factory=lambda: {"errors": [], "steps": {}})

    def guard(self, label: str, action: Callable[[], Any]) -> Any:
        try:
            value = action()
        except Exception as exc:  # noqa: BLE001 - every step is recorded, none hides
            self.report["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
            self.report["steps"][label] = {"error": f"{type(exc).__name__}: {exc}"}
            return None
        self.report["steps"][label] = value
        return value


def load_release_inputs(case_dir: Path) -> ReleaseInputs:
    """Read plan, journal and result; refuse when the attempt left no record."""

    plan_path = case_dir / "plan.json"
    journal_path = case_dir / JOURNAL_FILE
    result_path = case_dir / f"{CASE_ID}.json"
    if not plan_path.is_file() or not journal_path.is_file():
        raise RegionalFixtureError(
            f"DESTR-014 release needs the attempt's plan and journal under {case_dir}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.is_file()
        else {}
    )
    run = journal.get("run") or {}
    scope = journal.get("scope") or {}
    run_id = str(scope.get("run_id") or "")
    if not run_id:
        raise RegionalFixtureError("DESTR-014 journal names no run id")
    preflight = run.get("preflight") or {}
    profile_version = str(
        ((preflight.get("store") or {}).get("profile") or {}).get("profile_version")
        or plan["details"]["preflight_identity"].get("runtime_profile_version")
        or ""
    )
    return ReleaseInputs(
        case_dir=case_dir,
        plan_identity=dict(plan["details"]["preflight_identity"]),
        journal=journal,
        result=result,
        run_id=run_id,
        incident_id=str(run.get("incident_id") or result.get("incident_id") or ""),
        follow_up_incident_id=str(run.get("follow_up_incident_id") or ""),
        profile_version=profile_version,
        holder_armed=bool(run.get("holder_armed")),
    )


def identity_errors(
    regional: Any, settings: Settings, identity: dict[str, Any]
) -> list[str]:
    """Release id and both Node UIDs must be the plan's; boot ids may differ."""

    errors: list[str] = []
    live_release = regional.release_id()
    if live_release != identity.get("release_id"):
        errors.append(
            f"release id changed: plan {identity.get('release_id')} live {live_release}"
        )
    for label, node, key in (
        ("fault", settings.fault_node, "fault_node_uid"),
        ("sibling", settings.sibling_node, "sibling_node_uid"),
    ):
        uid = regional.node_snapshot(node).get("uid")
        if not identity.get(key) or uid != identity[key]:
            errors.append(
                f"{label} node UID changed: plan {identity.get(key)} live {uid}"
            )
    return errors


def rebooted_nodes(
    regional: Any, settings: Settings, identity: dict[str, Any]
) -> list[str]:
    """Nodes whose current boot id differs from the one the plan recorded."""

    rebooted: list[str] = []
    for node, key in (
        (settings.fault_node, "fault_node_boot_id"),
        (settings.sibling_node, "sibling_node_boot_id"),
    ):
        boot_id = regional.node_snapshot(node).get("boot_id")
        if identity.get(key) and boot_id and boot_id != identity[key]:
            rebooted.append(node)
    return rebooted


def run_admin(state_dir: Path, *argv: str, timeout: int = 1800) -> dict[str, Any]:
    """One ``gpu-fault-admin`` verb from this checkout, output captured."""

    completed = subprocess.run(
        [sys.executable, "-c", ADMIN_DIRECT, *argv, "--state-dir", str(state_dir)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    record = {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    if completed.returncode != 0:
        raise RegionalFixtureError(
            f"gpu-fault-admin {argv[0]} exited {completed.returncode}: "
            f"{completed.stderr.strip()[-500:] or completed.stdout.strip()[-500:]}"
        )
    return record


class PinnedWorkload:
    """The drill job of a held attempt, deletable by identity in a later process.

    ``ManagedWorkloadFixture.delete`` returns silently unless this very object
    submitted the job (ownership nonce), so a release run cannot use it. The
    pinned manifest names the object; the live copy must carry this case's
    ``gpu-fault.io/job-id`` label or it is somebody else's workload and stays.
    """

    JOB_LABEL = "gpu-fault.io/job-id"

    def __init__(self, regional: Any, manifest: Path, job_id: str) -> None:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise RegionalFixtureError("pinned workload manifest is not a mapping")
        self.kind = str(document.get("kind") or "").lower()
        self.name = str((document.get("metadata") or {}).get("name") or "")
        if self.kind not in RESOURCE_APIS or self.kind == "pod" or not self.name:
            raise RegionalFixtureError("pinned workload manifest has no kind/name")
        self.regional = regional
        self.job_id = job_id

    def delete(self) -> dict[str, Any]:
        current = read_resource(self.regional, self.kind, self.name)
        if current is None:
            return {
                "kind": self.kind,
                "name": self.name,
                "deleted": False,
                "absent": True,
            }
        labels = (current.get("metadata") or {}).get("labels") or {}
        if labels.get(self.JOB_LABEL) != self.job_id:
            raise RegionalFixtureError(
                f"{self.kind}/{self.name} carries {self.JOB_LABEL}="
                f"{labels.get(self.JOB_LABEL)!r}, not this case's {self.job_id!r}; "
                "a foreign workload is never deleted"
            )
        _kind, _name, uid = resource_identity(self.regional, current)
        delete_resource(self.regional, current)
        return {
            "kind": self.kind,
            "name": self.name,
            "uid": uid,
            "attempt_id": labels.get("gpu-fault.io/attempt-id"),
            "deleted": True,
        }


def release_product_side(context: ReleaseContext) -> None:
    """Steps 1-3b: confirm the known reboot, restore, settle the record."""

    settings, inputs = context.settings, context.inputs
    if not inputs.incident_id:
        context.report["errors"].append("the attempt recorded no incident id")
        return
    rebooted = context.guard(
        "rebooted_nodes",
        lambda: rebooted_nodes(context.regional, settings, inputs.plan_identity),
    )
    for node in rebooted or []:
        context.guard(
            f"confirm_node_action:{node}",
            lambda node=node: _tolerated(
                lambda: context.admin(
                    context.state_dir,
                    "submit-remediation",
                    "--incident-id",
                    inputs.incident_id,
                    "--disposition",
                    "confirm-node-action",
                    "--node",
                    node,
                    "--reference",
                    RELEASE_REFERENCE,
                ),
                NO_UNRESOLVED_ACTION,
                "no unresolved node action on this node",
            ),
        )
    restored = context.guard(
        "restore_disposition",
        lambda: _tolerated(
            lambda: context.admin(
                context.state_dir,
                "submit-remediation",
                "--incident-id",
                inputs.incident_id,
                "--disposition",
                "restore",
                "--reference",
                RELEASE_REFERENCE,
            ),
            NO_NODE_ISOLATED_BY_IT,
            "no node is isolated by the case incident; owners restore in step 3",
        ),
    )
    if restored is not None and "skipped" not in restored:
        # The product's restore workflow is now PENDING/RUNNING on the incident;
        # the validated restore below refuses to race it, so wait it out first.
        context.guard(
            "restore_workflow_wait",
            lambda: context.warm.wait_incident_idle(inputs.incident_id),
        )
    context.guard(
        "isolation_restore",
        lambda: restore_isolated_nodes(
            context.regional,
            context.warm,
            settings,
            inputs.incident_id,
            inputs.profile_version,
        ),
    )
    for incident_id in dict.fromkeys(
        (inputs.incident_id, inputs.follow_up_incident_id)
    ):
        if incident_id:
            context.guard(
                f"workflow_reconcile:{incident_id}",
                lambda incident_id=incident_id: context.admin(
                    context.state_dir,
                    "workflow-reconcile",
                    "--incident-id",
                    incident_id,
                    "--reference",
                    RELEASE_REFERENCE,
                ),
            )


def _tolerated(
    action: Callable[[], dict[str, Any]], refusal: str, reason: str
) -> dict[str, Any]:
    """Run one admin verb; a documented product refusal is a skip, not an error."""

    try:
        return action()
    except RegionalFixtureError as exc:
        if refusal in str(exc):
            return {"skipped": reason, "refusal": str(exc)[-400:]}
        raise


def release_cluster_side(context: ReleaseContext) -> None:
    """Steps 4-7: the job, the holder, both env windows, the drill incidents."""

    settings, inputs = context.settings, context.inputs
    if context.workload is not None:
        context.guard("workload_delete", context.workload.delete)
    else:
        context.report["steps"]["workload_delete"] = {
            "skipped": "no pinned-workload.yaml in the case directory"
        }
    if inputs.holder_armed:

        def disarm() -> dict[str, Any]:
            probe = context.fault_probe
            probe.cleanup()
            probe.create()
            try:
                return probe.execute("disarm-holder", "--run-id", inputs.run_id)
            finally:
                probe.cleanup()

        context.guard("holder_disarm", disarm)
    else:
        context.report["steps"]["holder_disarm"] = {"skipped": "holder was not armed"}
    executor_record = inputs.case_dir / "executor-env-window.json"
    if executor_record.is_file():
        context.guard(
            "env_window_close",
            lambda: env_window.close_window(
                env_window.Settings(
                    baseline=executor_record, rollout_timeout_seconds=300
                ),
                context.regional,
                env_window.survey(context.regional),
            ),
        )
    control_record = inputs.case_dir / "control-plane-env-window.json"
    if control_record.is_file():
        context.guard(
            "control_env_window_close",
            lambda: control_window.without_survey(
                control_window.close_window(
                    control_window.Settings(
                        baseline=control_record, rollout_timeout_seconds=600
                    ),
                    context.regional,
                    control_window.survey(context.regional),
                )
            ),
        )
    reports = context.guard(
        "incident_close",
        lambda: close_drill_incidents(
            context.warm,
            context.regional,
            [inputs.incident_id, inputs.follow_up_incident_id],
            reason="DESTR-014 operator release",
            reference=RELEASE_REFERENCE,
            nodes=(settings.fault_node, settings.sibling_node),
        ),
    )
    if reports:
        residual = residual_incidents(reports)
        if residual:
            context.report["residual_incidents"] = residual
            context.report["errors"].append(
                "incidents still open: " + ", ".join(sorted(residual))
            )


def release_hold(context: ReleaseContext) -> dict[str, Any]:
    """The whole release; the report is written next to the case result."""

    report = context.report
    report.update(
        {
            "case_id": CASE_ID,
            "mode": "release",
            "run_id": context.inputs.run_id,
            "incident_id": context.inputs.incident_id,
            "follow_up_incident_id": context.inputs.follow_up_incident_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    errors = identity_errors(
        context.regional, context.settings, context.inputs.plan_identity
    )
    if errors:
        report["errors"].extend(errors)
        report["refused"] = "identity mismatch"
    else:
        release_product_side(context)
        release_cluster_side(context)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["released"] = not report["errors"]
    path = context.inputs.case_dir / RELEASE_RECORD
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if not isinstance(history, list):
        history = [history]
    history.append(report)
    write_json_atomic(path, history)
    return report
