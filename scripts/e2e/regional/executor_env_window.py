#!/usr/bin/env python3
"""Open or close a temporary env window on the cluster executor Deployment.

`GF-REGIONAL-DESTR-014` needs two executor variables changed for one
maintenance window and then restored exactly:

* ``GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`` -- lowered so node-b's RESET_GPU
  reaches FAILED in tens of seconds rather than ~5 minutes.
* ``GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS`` -- lowered so
  node-c's RESTART_NODE reaches its bounded-waiting timeout inside the window.

Modeled on ``synthetic_replacement_route.py``: it records the Deployment's
pre-window env in a baseline file before mutating, waits until *every ready
replica* reports the new values, and ``--close`` restores exactly what the
baseline recorded -- a variable that was absent goes back to absent (delete),
never to ``false``. It refuses to open a window already open under a baseline
it did not write, and refuses to close one it has no record of. The variable
allow-list is compiled in; it can change nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

DEPLOYMENT = "gpu-fault-cluster-executor"
CONTAINER = "executor"
ALLOWED_VARIABLES = (
    "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS",
    "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS",
)
OPEN_CONFIRMATION = "OPEN_EXECUTOR_ENV_WINDOW"
CLOSE_CONFIRMATION = "CLOSE_EXECUTOR_ENV_WINDOW"


@dataclass(frozen=True)
class Settings:
    baseline: Path
    rollout_timeout_seconds: int


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_assignments(pairs: list[str]) -> dict[str, str]:
    """Validate ``NAME=VALUE`` pairs against the compiled-in allow-list.

    Values must be positive integers -- both variables are second/attempt
    counts, and the case exists to *lower* them, so a non-numeric or zero value
    is a typo that would silently disable the very timeout it means to tighten.
    """

    result: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator:
            raise RegionalFixtureError(f"--set must be NAME=VALUE, got {pair!r}")
        if name not in ALLOWED_VARIABLES:
            raise RegionalFixtureError(
                f"{name} is not in the executor env window allow-list"
            )
        if not value or not value.isdigit() or int(value) < 1:
            raise RegionalFixtureError(
                f"{name} must be set to a positive integer, got {value!r}"
            )
        result[name] = value
    return result


def open_arguments(assignments: dict[str, str]) -> list[str]:
    return [f"{name}={assignments[name]}" for name in sorted(assignments)]


def restore_arguments(baseline: dict[str, Any]) -> list[str]:
    """`kubectl set env` arguments that put the Deployment back to baseline.

    A variable the shipped manifest did not carry is deleted (``NAME-``), not
    set to ``false``: a leftover literal is config the release does not know it
    has.
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
        regional.kubectl("gpu", "get", "deployment", DEPLOYMENT, "-o", "json")
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
        "deployment": DEPLOYMENT,
        "container": CONTAINER,
        "generation": value["metadata"]["generation"],
        "resource_version": value["metadata"]["resourceVersion"],
        "replicas": value["spec"].get("replicas"),
        "variables": variables,
    }


def replica_values(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    names = ",".join(repr(name) for name in ALLOWED_VARIABLES)
    for pod in regional.ready_pods("gpu", DEPLOYMENT):
        try:
            output = regional.kubectl(
                "gpu",
                "exec",
                str(pod["name"]),
                "--",
                "python3",
                "-c",
                f"import json,os; print(json.dumps({{n: os.getenv(n) for n in [{names}]}}))",
                timeout=60,
            )
        except RegionalFixtureError as error:
            # The window rolls the executor Deployment; a replica terminated
            # between the listing and this exec is no longer a ready replica.
            # Drop it and let ``converge`` re-poll (DESTR-014, 2026-09-08).
            if replica_vanished(error):
                continue
            raise
        result.append(
            {
                "pod": str(pod["name"]),
                "values": json.loads(output.splitlines()[-1]),
            }
        )
    return result


def survey(regional: RegionalLiveFixture) -> dict[str, Any]:
    return {
        "observed_at": now(),
        "cluster_id": regional.settings.cluster_id,
        "deployment": deployment_env(regional),
        "replicas": replica_values(regional),
    }


def wait_rollout(settings: Settings, regional: RegionalLiveFixture) -> str:
    return regional.kubectl(
        "gpu",
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
                "ready executor replicas did not all report the expected env: "
                + json.dumps(
                    {"expected": expected, "replicas": replicas}, sort_keys=True
                )
            )
        sleep(5)


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegionalFixtureError(f"no executor env window baseline at {path}")
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
            f"refusing to open the executor env window: baseline record and live "
            f"env disagree on {sorted(assignments)}; close the record first"
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
            "gpu",
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            f"--containers={CONTAINER}",
            *open_arguments(assignments),
        )
        record["rollout"] = wait_rollout(settings, regional)
    else:
        assert record is not None
        record["resumed_at"] = now()
    record["replicas"] = converge(settings, regional, dict(assignments), sleep=sleep)
    record["opened_state"] = deployment_env(regional)
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
        "gpu",
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
            "restored executor env does not match the recorded baseline: "
            + json.dumps(
                {"baseline": baseline["variables"], "restored": restored["variables"]},
                sort_keys=True,
            )
        )
    expected = {
        name: (record["value"] if record["present"] else None)
        for name, record in baseline["variables"].items()
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
            "Open or close the cluster-executor env window for "
            "GF-REGIONAL-DESTR-014. Read-only unless --open or --close is given."
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
    value.add_argument("--rollout-timeout-seconds", type=int, default=300)
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
