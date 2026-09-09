#!/usr/bin/env python3
"""Open or close a temporary env window on the control-worker Deployment.

`GF-REGIONAL-DESTR-018` needs the workflow lifetime compressed for one
maintenance window and then restored exactly. The operator chooses two values:

* ``GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS`` -- lowered so a node
  workflow held WAITING reaches its hard lifetime inside the window instead of
  an hour later.
* ``GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS`` -- pinned at the lifetime,
  because ``restart_budget_preflight.claim_deadlines`` stamps
  ``min(execution, lifetime)``: an execution timeout *below* the lifetime makes
  the workflow fail as a plain execution-deadline miss with
  ``details.workflow_lifetime_exceeded=false``, which is the opposite of what
  the case exists to prove, and one *above* the lifetime is truncated to it.

The rest of the timing set moves in lockstep, derived from those two
(``lockstep_assignments``), because ``execution.config.
validate_timing_relationships`` fails the control plane closed at boot when the
set is inconsistent -- a window that lowered the lifetime alone (step ceilings
and managed recovery left at 600 s/1800 s) CrashLoopBackOff'd every worker
replica on the live run, and the rollout never converged. An explicit ``--set``
of a derived variable wins over the derivation and is judged the same way:

* every per-step waiting ceiling must be at or below the node lifetime
  (``GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS``,
  ``GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS``,
  ``GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS``), and every override at
  or above the default step cap -- so each is set to the lifetime. Not below
  it: the drill needs the *lifetime* to be the bound that fires, and
  ``executor._execute_step`` checks the workflow deadline before the per-step
  cap, so a cap equal to the lifetime never fires first.
* ``GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS`` keeps the shipped lead (half the
  step cap) and stays at or below the cap.
* ``GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS`` takes whatever the
  lifetime leaves above the install ceiling, up to its shipped value, so the
  install execution floor (ceiling + allowance, F2) fits the lifetime.
* ``GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS`` must stay below the execution
  timeout; it is lowered to half of it, never above its shipped value.

The rendered set is checked with the control plane's own validator before the
window opens (``assignment_errors``), judged against the shipped defaults for
everything the window does not set, so an inconsistent ``--set`` is rejected on
the command line instead of surfacing as an opaque rollout timeout mid case. No
variable is ever raised above its shipped default; the two operator-chosen
values keep their historical 60..3600 s range.

All of them live on the CPU-plane ``gpu-fault-control-worker`` Deployment, which is
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
nothing else, and it can only *lower* a bound, never raise one above the
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

from gpu_fault.execution.config import (  # noqa: E402
    ProductionExecutorConfig,
    WorkflowExecutionError,
    validate_timing_from_environment,
)
from gpu_fault.models import WorkflowOperation  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    replica_vanished,
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
STEP_TIMEOUT_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS"
STEP_WARNING_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS"
MANAGED_RECOVERY_VARIABLE = "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS"
INSTALL_STEP_VARIABLE = "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS"
INSTALL_CONTAINMENT_VARIABLE = "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS"
LEASE_DURATION_VARIABLE = "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS"
_SHIPPED = ProductionExecutorConfig.from_mapping({})
# What the control plane reads when a variable is absent, taken from the config
# that reads it so the two cannot drift. Also the most a window may set: above
# it a window stops compressing and starts extending production bounds.
SHIPPED_DEFAULTS: dict[str, str] = {
    LIFETIME_VARIABLE: str(_SHIPPED.node_workflow_lifetime_seconds),
    EXECUTION_TIMEOUT_VARIABLE: str(_SHIPPED.workflow_execution_timeout_seconds),
    STEP_TIMEOUT_VARIABLE: str(_SHIPPED.step_waiting_timeout_seconds),
    STEP_WARNING_VARIABLE: str(_SHIPPED.step_waiting_warning_seconds),
    MANAGED_RECOVERY_VARIABLE: str(
        _SHIPPED.step_waiting_limit(WorkflowOperation.REPLACE_NODE)
    ),
    INSTALL_STEP_VARIABLE: str(
        _SHIPPED.step_waiting_limit(WorkflowOperation.REMEDIATE_DRIVER)
    ),
    INSTALL_CONTAINMENT_VARIABLE: str(_SHIPPED.node_install_containment_seconds),
    LEASE_DURATION_VARIABLE: str(_SHIPPED.lease_duration_seconds),
}
ALLOWED_VARIABLES = tuple(SHIPPED_DEFAULTS)
# Not window-settable; the validator judges twice the managed-recovery window
# (one escalated branch) against it.
JOB_LIFETIME_SECONDS = _SHIPPED.job_workflow_lifetime_seconds
# The two the operator chooses; the rest are derived from them.
CHOSEN_VARIABLES = (LIFETIME_VARIABLE, EXECUTION_TIMEOUT_VARIABLE)
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
# The historical range of the two chosen values; a derived one may go to 0 (the
# containment allowance) and never above its shipped default.
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
        minimum, maximum = value_bounds(name)
        if not value.isdigit() or not minimum <= int(value) <= maximum:
            raise RegionalFixtureError(
                f"{name} must be an integer of {minimum}..{maximum} seconds, "
                f"got {value!r}"
            )
        if name in result:
            raise RegionalFixtureError(f"{name} was set twice")
        result[name] = value
    errors = assignment_errors(result)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return result


def value_bounds(name: str) -> tuple[int, int]:
    """The inclusive range one variable may be set to.

    The two chosen values keep their historical range; a derived one may go as
    low as the control plane's own validation allows (it is checked next, on the
    completed set) and never above what the release ships.
    """

    if name in CHOSEN_VARIABLES:
        return MINIMUM_SECONDS, MAXIMUM_SECONDS
    return 0, int(SHIPPED_DEFAULTS[name])


def lockstep_assignments(
    lifetime_seconds: int, execution_timeout_seconds: int | None = None
) -> dict[str, str]:
    """The whole timing set for one compressed lifetime (see the module doc).

    Every per-step ceiling becomes the lifetime (a ceiling above it is refused
    at boot; one below it would end the wait before the lifetime the drill
    measures), the warning keeps the shipped half-cap lead, the install
    containment allowance is whatever the lifetime leaves above the install
    ceiling, and the lease is halved under the execution timeout. Nothing is
    raised above its shipped default, so an uncompressed lifetime renders the
    defaults themselves.
    """

    shipped = {name: int(value) for name, value in SHIPPED_DEFAULTS.items()}
    # Unless chosen, the execution timeout is the lifetime (the drill needs it at
    # or above the lifetime) but never more than the release ships.
    execution = (
        min(shipped[EXECUTION_TIMEOUT_VARIABLE], lifetime_seconds)
        if execution_timeout_seconds is None
        else execution_timeout_seconds
    )
    step = min(shipped[STEP_TIMEOUT_VARIABLE], lifetime_seconds)
    install = min(shipped[INSTALL_STEP_VARIABLE], lifetime_seconds)
    return {
        LIFETIME_VARIABLE: str(lifetime_seconds),
        EXECUTION_TIMEOUT_VARIABLE: str(execution),
        STEP_TIMEOUT_VARIABLE: str(step),
        STEP_WARNING_VARIABLE: str(min(shipped[STEP_WARNING_VARIABLE], step // 2)),
        MANAGED_RECOVERY_VARIABLE: str(
            min(shipped[MANAGED_RECOVERY_VARIABLE], lifetime_seconds)
        ),
        INSTALL_STEP_VARIABLE: str(install),
        INSTALL_CONTAINMENT_VARIABLE: str(
            min(shipped[INSTALL_CONTAINMENT_VARIABLE], lifetime_seconds - install)
        ),
        LEASE_DURATION_VARIABLE: str(
            min(shipped[LEASE_DURATION_VARIABLE], execution // 2)
        ),
    }


def complete_assignments(assignments: dict[str, str]) -> dict[str, str]:
    """``assignments`` with every derived variable filled in from the lifetime.

    An explicit value wins over the derived one; without a lifetime there is
    nothing to derive from and the request is returned as given.
    """

    lifetime = assignments.get(LIFETIME_VARIABLE)
    if lifetime is None:
        return dict(assignments)
    execution = assignments.get(EXECUTION_TIMEOUT_VARIABLE)
    derived = lockstep_assignments(
        int(lifetime), None if execution is None else int(execution)
    )
    return {**derived, **assignments}


def assignment_errors(assignments: dict[str, str]) -> list[str]:
    """What the run would prove wrongly, or the control plane would refuse.

    ``claim_deadlines`` returns ``min(now + execution_timeout, lifetime)`` as
    the execution deadline and the lifetime separately, and
    ``step_bounds.workflow_deadline_failure`` only sets
    ``workflow_lifetime_exceeded`` when ``now >= lifetime``. An execution
    timeout below the lifetime therefore fires first and reports a plain
    execution-deadline miss, so a window that compresses the lifetime without
    keeping the execution timeout at or above it silently changes which
    contract the run proves. Likewise a step waiting cap set *below* a
    compressed lifetime ends the WAITING step before the lifetime does, and the
    run proves the step cap; the derivation never does that, so this only
    catches an explicit ``--set``.

    The completed set is then given to the control plane's own boot-time
    validator (``validate_timing_from_environment``), judged against the
    shipped defaults for everything the window does not set: a set it refuses
    would CrashLoop every worker replica instead of opening a window.
    """

    errors: list[str] = []
    lifetime = assignments.get(LIFETIME_VARIABLE)
    timeout = assignments.get(EXECUTION_TIMEOUT_VARIABLE)
    step = assignments.get(STEP_TIMEOUT_VARIABLE)
    if lifetime is not None and timeout is not None and int(timeout) < int(lifetime):
        errors.append(
            f"{EXECUTION_TIMEOUT_VARIABLE}={timeout} is below "
            f"{LIFETIME_VARIABLE}={lifetime}; the execution deadline would fire "
            "first and the failure would not carry workflow_lifetime_exceeded"
        )
    if lifetime is not None and step is not None and int(step) < int(lifetime):
        errors.append(
            f"{STEP_TIMEOUT_VARIABLE}={step} is below {LIFETIME_VARIABLE}={lifetime}; "
            "the WAITING step's own cap could end the wait before the workflow "
            "lifetime does"
        )
    try:
        validate_timing_from_environment(complete_assignments(assignments))
    except (WorkflowExecutionError, RuntimeError) as exc:
        errors.append(
            "the control plane would refuse to boot with this window "
            f"(execution/config.py): {exc}"
        )
    return errors


def open_arguments(assignments: dict[str, str]) -> list[str]:
    return [f"{name}={assignments[name]}" for name in sorted(assignments)]


def restore_arguments(baseline: dict[str, Any]) -> list[str]:
    """`kubectl set env` arguments that put the Deployment back to baseline.

    A variable the shipped manifest did not carry as a literal is deleted
    (``NAME-``), not set back to a value: the regional release carries the
    timing set through ``envFrom`` ConfigMaps or not at all, and a leftover
    literal is config the release does not know it has.
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
        name = str(pod["name"])
        try:
            output = regional.kubectl(
                plane,
                "exec",
                name,
                "--",
                "python3",
                "-c",
                f"import json,os; print(json.dumps({{n: os.getenv(n) for n in [{literal}]}}))",
                timeout=60,
            )
        except RegionalFixtureError as error:
            # Every window open and close rolls the control-worker Deployment,
            # so a replica can be terminated between ``ready_pods`` listing it
            # and this exec (observed live 2026-09-08: the pod was NotFound at
            # exec time). Such a pod is no longer a ready replica; drop it and
            # let the caller (``converge``) re-poll the settling set instead of
            # failing the whole survey on a transient. Any other exec failure
            # is a real error and still propagates.
            if replica_vanished(error):
                continue
            raise
        result.append(
            {
                "pod": name,
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
    # The whole timing set is written, recorded and converged on, whatever
    # subset the caller chose; the record must name what the Deployment got.
    assignments = complete_assignments(assignments)
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
    record["replicas_after_close"] = converge(
        settings, regional, restored_expectation(record), sleep=sleep
    )
    write_json_atomic(settings.baseline, record)
    return record


def restored_expectation(record: dict[str, Any]) -> dict[str, str | None]:
    """What every replica must read once the window is closed.

    Not "None for every variable the baseline did not carry inline": two of
    the six (managed recovery, lease) reach the worker through ``envFrom``
    ConfigMaps, so a restored replica legitimately reads 1800/180 for them and
    a None expectation can never converge -- observed live 2026-09-08, where
    every close spun to its rollout timeout after the Deployment had already
    been restored. The pre-window survey recorded what the replicas actually
    read; that is the restore target. A record written without a survey falls
    back to the inline baseline.
    """

    baseline = record["baseline"]["variables"]
    replicas = (record.get("pre_window_survey") or {}).get("replicas") or []
    if replicas:
        return {name: observed_value(replicas, name) for name in baseline}
    return {
        name: (item["value"] if item["present"] else None)
        for name, item in baseline.items()
    }


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
        help=(
            "NAME=VALUE for --open; the rest of the timing set is derived from "
            f"{LIFETIME_VARIABLE} unless set explicitly. Allowed names: "
            + ", ".join(ALLOWED_VARIABLES)
        ),
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
