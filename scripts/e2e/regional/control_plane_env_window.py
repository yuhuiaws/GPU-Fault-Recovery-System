#!/usr/bin/env python3
"""Open or close a temporary env window on the control-worker Deployment.

`GF-REGIONAL-DESTR-018` needs the workflow lifetime compressed for one
maintenance window and then restored exactly:

* ``GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS`` -- lowered so a node
  workflow held WAITING reaches its hard lifetime inside the window instead of
  an hour later.
* ``GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS`` -- pinned at the lifetime,
  because ``restart_budget_preflight.claim_deadlines`` stamps
  ``min(execution, lifetime)``: an execution timeout *below* the lifetime makes
  the workflow fail as a plain execution-deadline miss with
  ``details.workflow_lifetime_exceeded=false``, which is the opposite of what
  the case exists to prove, and one *above* the lifetime is truncated to it.
* ``GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS``,
  ``GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS``,
  ``GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS`` and
  ``GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS`` -- lowered *with* the lifetime.
  The control plane's boot-time timing guard (``execution/config.py``) refuses
  to construct a config whose step waiting ceilings or managed-recovery window
  sit above the node lifetime, or whose lease sits at/above the execution
  timeout, so compressing the lifetime alone CrashLoopBackOffs every worker
  replica. ``assignment_errors`` encodes the full ordering these four have to
  keep, and rejects an inconsistent ``--set`` on the command line.

All six live on the CPU-plane ``gpu-fault-control-worker`` Deployment, which is
where the workflow executor and the single-instance dispatch lease run
(``execution/dispatcher.py``: ``DISPATCH_LEASE_KEY = "workflow-dispatch"``).
Changing the Deployment template rolls every worker replica, so the lease moves
and every in-process counter resets. The window must therefore be opened
*before* the case injects, and closed only after the case reached quiescence --
a rollout in the middle of an open workflow re-elects the lease holder mid
lifetime and the timestamps the case measures stop meaning one thing.

Modeled on ``synthetic_replacement_route.py`` and the executor window it is a
sibling of: it records the Deployment's pre-window env in a baseline file
before mutating, waits until *every ready replica* reports the new values, and
``--close`` restores exactly what the baseline recorded -- a variable that was
absent goes back to absent (delete), never to ``0``. It refuses to open a
window already open under a baseline it did not write, and refuses to close one
it has no record of. The variable allow-list is compiled in; it can change
nothing else, and it can only *lower* a lifetime, never raise one above the
shipped default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    install_abort_signals,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.site_profile import (  # noqa: E402
    bind_site_profile,
    install_site_profile,
)

PLANE = "cpu"
DEPLOYMENT = "gpu-fault-control-worker"
CONTAINER = "control-worker"
LIFETIME_VARIABLE = "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS"
EXECUTION_TIMEOUT_VARIABLE = "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS"
# Compressing the node lifetime alone is not a bootable config: the control
# plane's own start-up guard (``execution/config.py``
# ``validate_timing_relationships`` / ``ProductionExecutorConfig.from_mapping``)
# refuses a lifetime that sits below the step waiting ceilings, the managed
# recovery window, or at/below the lease -- so the compressed window has to move
# those in lockstep or every worker replica CrashLoopBackOffs and the rollout
# never converges. These four are lowered with the lifetime for exactly that
# reason; ``assignment_errors`` encodes the whole ordering.
STEP_TIMEOUT_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS"
MANAGED_RECOVERY_VARIABLE = "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS"
STEP_WARNING_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS"
LEASE_DURATION_VARIABLE = "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS"
ALLOWED_VARIABLES = (
    LIFETIME_VARIABLE,
    EXECUTION_TIMEOUT_VARIABLE,
    STEP_TIMEOUT_VARIABLE,
    MANAGED_RECOVERY_VARIABLE,
    STEP_WARNING_VARIABLE,
    LEASE_DURATION_VARIABLE,
)
# The shipped default each managed variable takes when the Deployment does not
# carry it, read from ``execution/config.py``. ``assignment_errors`` merges a
# partial ``--set`` onto these so it can reject a set that would only be found
# inconsistent once the control plane tried to boot it.
SHIPPED_DEFAULTS = {
    LIFETIME_VARIABLE: 3600,
    EXECUTION_TIMEOUT_VARIABLE: 1800,
    STEP_TIMEOUT_VARIABLE: 600,
    MANAGED_RECOVERY_VARIABLE: 1800,
    STEP_WARNING_VARIABLE: 300,
    LEASE_DURATION_VARIABLE: 180,
}
# Not window-settable, but the ordering rules are judged against them. The case
# runs on a control worker whose other timing knobs are at their shipped
# defaults (the preflight refuses any window variable that is already set); a
# deployment that overrode the job lifetime or the branch rung count separately
# would need these revisited.
JOB_LIFETIME_SECONDS = 3600
BRANCH_ESCALATION_MAX_RUNGS = 2
# Reported alongside the window so a case can do its timing arithmetic against
# what the deployed control plane actually reads. Never written by this helper.
OBSERVED_VARIABLES = (
    "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS",
    "GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS",
)
SURVEYED_VARIABLES = ALLOWED_VARIABLES + OBSERVED_VARIABLES
# A lifetime under a minute cannot contain the containment steps that must run
# before the deadline; above the shipped default this stops being a compression
# window and becomes a way to make production workflows outlive their bound.
MINIMUM_SECONDS = 60
MAXIMUM_SECONDS = 3600
OPEN_CONFIRMATION = "OPEN_CONTROL_PLANE_ENV_WINDOW"
CLOSE_CONFIRMATION = "CLOSE_CONTROL_PLANE_ENV_WINDOW"


@dataclass(frozen=True)
class Settings:
    baseline: Path
    rollout_timeout_seconds: int


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_assignments(pairs: list[str]) -> dict[str, str]:
    """Validate ``NAME=VALUE`` pairs against the compiled-in allow-list.

    Values are second counts, so a non-numeric, zero or negative value is a
    typo that would silently disable the very bound it means to tighten -- and
    ``ProductionExecutorConfig.from_mapping`` raises ``workflow lifetimes must
    be positive`` on the whole worker rather than on this command line, i.e.
    the mistake surfaces as a CrashLoopBackOff of the control plane.
    """

    result: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator:
            raise RegionalFixtureError(f"--set must be NAME=VALUE, got {pair!r}")
        if name not in ALLOWED_VARIABLES:
            raise RegionalFixtureError(
                f"{name} is not in the control-plane env window allow-list"
            )
        if not value.isdigit() or not MINIMUM_SECONDS <= int(value) <= MAXIMUM_SECONDS:
            raise RegionalFixtureError(
                f"{name} must be an integer of "
                f"{MINIMUM_SECONDS}..{MAXIMUM_SECONDS} seconds, got {value!r}"
            )
        if name in result:
            raise RegionalFixtureError(f"{name} was set twice")
        result[name] = value
    errors = assignment_errors(result)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return result


def assignment_errors(assignments: dict[str, str]) -> list[str]:
    """Every cross-variable rule a ``--set`` combination has to satisfy.

    Compressing the node lifetime is not a one-variable change. The control
    plane re-derives the whole timing envelope at boot in
    ``execution/config.py``: ``ProductionExecutorConfig.from_mapping`` and
    ``validate_timing_relationships`` refuse to construct a config -- so every
    worker replica CrashLoopBackOffs and the rollout never converges -- unless
    the compressed knobs stay internally ordered. That refusal used to surface
    as an opaque rollout timeout in the middle of a live case; encoding the rules
    here rejects the bad combination on the command line instead.

    The partial ``--set`` is merged onto the shipped defaults (the preflight
    guarantees the Deployment carries none of these variables yet), then judged:

    Product-validity rules (a violation CrashLoopBackOffs the control plane):

    * step timeout and managed-recovery ceilings must not exceed the node
      lifetime -- ``validate_timing_relationships`` requires every non-ack step
      waiting ceiling ``<= node_workflow_max_lifetime``.
    * managed-recovery must not sit below the default step timeout -- a
      per-operation override below the base step timeout is rejected by
      ``from_mapping``.
    * the step warning threshold must not exceed the step timeout
      (``from_mapping``).
    * the execution timeout must not exceed the node lifetime -- above it,
      ``claim_deadlines`` silently truncates to the lifetime.
    * the lease duration must stay strictly below the execution timeout, or a
      dead executor holds the workflow record past its own deadline
      (``validate_timing_relationships``).
    * twice the managed-recovery ceiling must fit inside the job workflow
      lifetime, so a reboot-then-delegate branch is not failed by the job bound
      (``validate_timing_relationships``).

    Drill-correctness rules (only when this window is compressing the lifetime,
    i.e. the lifetime variable is being set):

    * the execution timeout must not fall below the lifetime, or
      ``claim_deadlines`` stamps ``min(execution, lifetime) = execution`` and the
      failure carries ``workflow_lifetime_exceeded=false`` -- the opposite of
      what the case proves. ``step_bounds.workflow_deadline_failure`` only sets
      the flag when ``now >= lifetime``.
    * the step timeout must not fall below the lifetime, or the WAITING step's
      own bound ends the wait before the workflow lifetime does and the run
      proves the step cap rather than the lifetime.
    """

    merged = {
        name: int(assignments.get(name, SHIPPED_DEFAULTS[name]))
        for name in ALLOWED_VARIABLES
    }
    lifetime = merged[LIFETIME_VARIABLE]
    execution = merged[EXECUTION_TIMEOUT_VARIABLE]
    step_timeout = merged[STEP_TIMEOUT_VARIABLE]
    managed = merged[MANAGED_RECOVERY_VARIABLE]
    warning = merged[STEP_WARNING_VARIABLE]
    lease = merged[LEASE_DURATION_VARIABLE]

    errors: list[str] = []

    # ---- product-validity: any violation refuses the control-plane boot ----
    if step_timeout > lifetime:
        errors.append(
            f"{STEP_TIMEOUT_VARIABLE}={step_timeout} exceeds "
            f"{LIFETIME_VARIABLE}={lifetime}; a step waiting ceiling above the "
            "node lifetime makes the control plane refuse to boot "
            "(config.validate_timing_relationships)"
        )
    if managed > lifetime:
        errors.append(
            f"{MANAGED_RECOVERY_VARIABLE}={managed} exceeds "
            f"{LIFETIME_VARIABLE}={lifetime}; the managed-recovery ceiling is a "
            "step waiting ceiling and must not exceed the node lifetime "
            "(config.validate_timing_relationships)"
        )
    if managed < step_timeout:
        errors.append(
            f"{MANAGED_RECOVERY_VARIABLE}={managed} is below "
            f"{STEP_TIMEOUT_VARIABLE}={step_timeout}; a per-operation override "
            "may not sit below the default step timeout (config.from_mapping)"
        )
    if warning > step_timeout:
        errors.append(
            f"{STEP_WARNING_VARIABLE}={warning} exceeds "
            f"{STEP_TIMEOUT_VARIABLE}={step_timeout}; the step warning threshold "
            "may not exceed the step timeout (config.from_mapping)"
        )
    if execution > lifetime:
        errors.append(
            f"{EXECUTION_TIMEOUT_VARIABLE}={execution} exceeds "
            f"{LIFETIME_VARIABLE}={lifetime}; an execution timeout above the node "
            "lifetime is silently truncated to the lifetime by claim_deadlines"
        )
    if lease >= execution:
        errors.append(
            f"{LEASE_DURATION_VARIABLE}={lease} is not below "
            f"{EXECUTION_TIMEOUT_VARIABLE}={execution}; a lease at or above the "
            "execution timeout lets a dead executor hold the record past its "
            "deadline (config.validate_timing_relationships)"
        )
    if BRANCH_ESCALATION_MAX_RUNGS * managed > JOB_LIFETIME_SECONDS:
        errors.append(
            f"{BRANCH_ESCALATION_MAX_RUNGS}x {MANAGED_RECOVERY_VARIABLE}="
            f"{managed} exceeds the job workflow lifetime {JOB_LIFETIME_SECONDS}s; "
            "a reboot-then-delegate branch would be failed by the job lifetime "
            "(config.validate_timing_relationships)"
        )

    # ---- drill-correctness: only when the lifetime is being compressed ----
    if LIFETIME_VARIABLE in assignments:
        if execution < lifetime:
            errors.append(
                f"{EXECUTION_TIMEOUT_VARIABLE}={execution} is below "
                f"{LIFETIME_VARIABLE}={lifetime}; the execution deadline would "
                "fire first and the failure would not carry "
                "workflow_lifetime_exceeded"
            )
        if step_timeout < lifetime:
            errors.append(
                f"{STEP_TIMEOUT_VARIABLE}={step_timeout} is below "
                f"{LIFETIME_VARIABLE}={lifetime}; the WAITING step's own cap "
                "could end the wait before the workflow lifetime does"
            )

    return errors


def open_arguments(assignments: dict[str, str]) -> list[str]:
    return [f"{name}={assignments[name]}" for name in sorted(assignments)]


def restore_arguments(baseline: dict[str, Any]) -> list[str]:
    """`kubectl set env` arguments that put the Deployment back to baseline.

    A variable the shipped manifest did not carry is deleted (``NAME-``), not
    set back to a literal: the regional release ships neither of these two, and
    a leftover literal is config the release does not know it has.
    """

    variables = baseline.get("variables") or {}
    arguments: list[str] = []
    for name in sorted(variables):
        record = variables[name]
        if record.get("present"):
            arguments.append(f"{name}={record['value']}")
        else:
            arguments.append(f"{name}-")
    return arguments


def converged(
    replicas: list[dict[str, Any]],
    expected: dict[str, str | None],
) -> bool:
    """Whether every ready replica reports exactly ``expected`` for each name."""

    if not replicas:
        return False
    for replica in replicas:
        values = replica.get("values") or {}
        for name, value in expected.items():
            if values.get(name) != value:
                return False
    return True


def observed_value(replicas: list[dict[str, Any]], name: str) -> str | None:
    """One value every ready replica agrees on, or ``None``.

    Disagreement is not averaged into a number a case could compute a margin
    from: mid-rollout replicas reading two different lifetimes is exactly the
    state in which the arithmetic is meaningless.
    """

    if not replicas:
        return None
    values = {(replica.get("values") or {}).get(name) for replica in replicas}
    if len(values) != 1:
        return None
    single = values.pop()
    return None if single is None else str(single)


def open_decision(
    record: dict[str, Any] | None,
    live: dict[str, Any],
    assignments: dict[str, str],
) -> str:
    """``open`` | ``resume`` | ``refuse`` for the current baseline and live env.

    * No record (or a closed one) and the live env is at baseline -> ``open``.
    * No record but the live env already carries a changed value -> ``refuse``:
      someone opened a window without recording it.
    * A record this run wrote, for the same assignments -> ``resume`` the
      convergence an interrupted open did not finish.
    * A record whose assignments differ, or a live env at baseline under an
      open record -> ``refuse``.
    """

    live_variables = live.get("variables") or {}
    live_changed = any(
        (live_variables.get(name) or {}).get("value") == value
        for name, value in assignments.items()
    )
    if record is None or record.get("closed_at"):
        return "refuse" if live_changed else "open"
    if record.get("assignments") != assignments:
        return "refuse"
    return "resume" if live_changed else "refuse"


def deployment_env(regional: RegionalLiveFixture) -> dict[str, Any]:
    value = json.loads(
        regional.kubectl(PLANE, "get", "deployment", DEPLOYMENT, "-o", "json")
    )
    containers = value["spec"]["template"]["spec"]["containers"]
    container = next(
        (item for item in containers if item.get("name") == CONTAINER), None
    )
    if container is None:
        raise RegionalFixtureError(
            f"{DEPLOYMENT} has no container named {CONTAINER}: "
            + ", ".join(sorted(str(item.get("name")) for item in containers))
        )
    entries = container.get("env") or []
    variables: dict[str, Any] = {}
    for name in ALLOWED_VARIABLES:
        present = [item for item in entries if item.get("name") == name]
        if any("valueFrom" in item for item in present):
            raise RegionalFixtureError(
                f"{name} is set from a reference, not a literal; this helper "
                "only manages literal values"
            )
        variables[name] = {
            "present": bool(present),
            "value": present[0].get("value") if present else None,
        }
    return {
        "observed_at": now(),
        "plane": PLANE,
        "deployment": DEPLOYMENT,
        "container": CONTAINER,
        "generation": value["metadata"]["generation"],
        "resource_version": value["metadata"]["resourceVersion"],
        "replicas": value["spec"].get("replicas"),
        "variables": variables,
    }


def replica_env(
    regional: RegionalLiveFixture,
    *,
    plane: str,
    deployment: str,
    names: Sequence[str],
) -> list[dict[str, Any]]:
    """What every ready replica of one Deployment reads for ``names``.

    Read-only and deliberately generic: a case that compresses a control-plane
    bound also has to know the bounds it did *not* change, and several of those
    live on the GPU-plane executor rather than here. Only the mutation path
    below is pinned to one Deployment and container.
    """

    result: list[dict[str, Any]] = []
    literal = ",".join(repr(str(name)) for name in names)
    for pod in regional.ready_pods(plane, deployment):
        output = regional.kubectl(
            plane,
            "exec",
            str(pod["name"]),
            "--",
            "python3",
            "-c",
            f"import json,os; print(json.dumps({{n: os.getenv(n) for n in [{literal}]}}))",
            timeout=60,
        )
        result.append(
            {
                "pod": str(pod["name"]),
                "values": json.loads(output.splitlines()[-1]),
            }
        )
    return result


def replica_values(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    return replica_env(
        regional,
        plane=PLANE,
        deployment=DEPLOYMENT,
        names=SURVEYED_VARIABLES,
    )


def survey(regional: RegionalLiveFixture) -> dict[str, Any]:
    replicas = replica_values(regional)
    return {
        "observed_at": now(),
        "cluster_id": regional.settings.cluster_id,
        "deployment": deployment_env(regional),
        "replicas": replicas,
        "observed": {
            name: observed_value(replicas, name) for name in SURVEYED_VARIABLES
        },
    }


def wait_rollout(settings: Settings, regional: RegionalLiveFixture) -> str:
    return regional.kubectl(
        PLANE,
        "rollout",
        "status",
        f"deployment/{DEPLOYMENT}",
        f"--timeout={settings.rollout_timeout_seconds}s",
        timeout=settings.rollout_timeout_seconds + 60,
    ).strip()


def converge(
    settings: Settings,
    regional: RegionalLiveFixture,
    expected: dict[str, str | None],
    *,
    sleep: Any = time.sleep,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + settings.rollout_timeout_seconds
    while True:
        replicas = replica_values(regional)
        if converged(replicas, expected):
            return replicas
        if time.monotonic() >= deadline:
            raise RegionalFixtureError(
                "ready control-worker replicas did not all report the expected env: "
                + json.dumps(
                    {"expected": expected, "replicas": replicas}, sort_keys=True
                )
            )
        sleep(5)


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegionalFixtureError(f"no control-plane env window baseline at {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def open_window(
    settings: Settings,
    regional: RegionalLiveFixture,
    report: dict[str, Any],
    assignments: dict[str, str],
    *,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    record = read_baseline(settings.baseline) if settings.baseline.is_file() else None
    decision = open_decision(record, report["deployment"], assignments)
    if decision == "refuse":
        raise RegionalFixtureError(
            f"refusing to open the control-plane env window: baseline record and "
            f"live env disagree on {sorted(assignments)}; close the record first"
        )
    if decision == "open":
        record = {
            "opened_at": now(),
            "confirmation": OPEN_CONFIRMATION,
            "assignments": assignments,
            "baseline": report["deployment"],
            "pre_window_survey": report,
        }
        write_json_atomic(settings.baseline, record)
        regional.kubectl(
            PLANE,
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            f"--containers={CONTAINER}",
            *open_arguments(assignments),
        )
        record["rollout"] = wait_rollout(settings, regional)
    else:
        if record is None:
            raise RegionalFixtureError("resume decided without a baseline record")
        record["resumed_at"] = now()
    record["replicas"] = converge(settings, regional, dict(assignments), sleep=sleep)
    record["opened_state"] = deployment_env(regional)
    record["observed"] = {
        name: observed_value(record["replicas"], name) for name in SURVEYED_VARIABLES
    }
    write_json_atomic(settings.baseline, record)
    return record


def close_window(
    settings: Settings,
    regional: RegionalLiveFixture,
    report: dict[str, Any],
    *,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    record = read_baseline(settings.baseline)
    if record.get("closed_at"):
        raise RegionalFixtureError(
            f"{settings.baseline} was already closed at {record['closed_at']}"
        )
    baseline = record["baseline"]
    regional.kubectl(
        PLANE,
        "set",
        "env",
        f"deployment/{DEPLOYMENT}",
        f"--containers={CONTAINER}",
        *restore_arguments(baseline),
    )
    record["closed_at"] = now()
    record["close_survey"] = report
    record["rollout_after_close"] = wait_rollout(settings, regional)
    restored = deployment_env(regional)
    record["restored_state"] = restored
    write_json_atomic(settings.baseline, record)
    if restored["variables"] != baseline["variables"]:
        raise RegionalFixtureError(
            "restored control-worker env does not match the recorded baseline: "
            + json.dumps(
                {"baseline": baseline["variables"], "restored": restored["variables"]},
                sort_keys=True,
            )
        )
    expected = {
        name: (item["value"] if item["present"] else None)
        for name, item in baseline["variables"].items()
    }
    record["replicas_after_close"] = converge(settings, regional, expected, sleep=sleep)
    write_json_atomic(settings.baseline, record)
    return record


def without_survey(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.endswith("survey")}


def configure(arguments: argparse.Namespace) -> Settings:
    return Settings(
        baseline=Path(
            required(arguments.baseline, "baseline record path")
        ).expanduser(),
        rollout_timeout_seconds=int(arguments.rollout_timeout_seconds),
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Open or close the control-worker workflow-lifetime env window for "
            "GF-REGIONAL-DESTR-018. Read-only unless --open or --close is given."
        )
    )
    value.add_argument("--site-profile", default="")
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument(
        "--baseline",
        default="",
        help="file recording the Deployment's env before the window opened",
    )
    value.add_argument("--report", type=Path, default=None)
    value.add_argument("--rollout-timeout-seconds", type=int, default=600)
    value.add_argument(
        "--set",
        action="append",
        default=[],
        help=("NAME=VALUE for --open; allowed names: " + ", ".join(ALLOWED_VARIABLES)),
    )
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--open", action="store_true")
    mode.add_argument("--close", action="store_true")
    value.add_argument(
        "--confirm",
        default="",
        help=(
            f"--open requires exactly {OPEN_CONFIRMATION}; "
            f"--close requires exactly {CLOSE_CONFIRMATION}"
        ),
    )
    return bind_site_profile(value)


def main() -> int:
    install_site_profile()
    install_abort_signals()
    arguments = parser().parse_args()
    settings = configure(arguments)
    regional = RegionalLiveFixture(settings_from_arguments(arguments))
    report = survey(regional)

    if arguments.open:
        if arguments.confirm != OPEN_CONFIRMATION:
            raise RegionalFixtureError(f"--open requires --confirm {OPEN_CONFIRMATION}")
        assignments = parse_assignments(arguments.set)
        if not assignments:
            raise RegionalFixtureError("--open requires at least one --set NAME=VALUE")
        report["open"] = without_survey(
            open_window(settings, regional, report, assignments)
        )
    elif arguments.close:
        if arguments.confirm != CLOSE_CONFIRMATION:
            raise RegionalFixtureError(
                f"--close requires --confirm {CLOSE_CONFIRMATION}"
            )
        report["close"] = without_survey(close_window(settings, regional, report))
    else:
        report["mode"] = "read-only"

    if arguments.report is not None:
        write_json_atomic(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
