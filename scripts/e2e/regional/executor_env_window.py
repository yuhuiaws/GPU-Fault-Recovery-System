#!/usr/bin/env python3
"""Open or close a temporary env window on the cluster executor Deployment.

`GF-REGIONAL-DESTR-014` needs one executor variable changed for one
maintenance window and then restored exactly (HA-004 lowers the lease and poll
the same way):

* ``GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`` -- lowered so node-b's RESET_GPU
  reaches FAILED in tens of seconds rather than ~5 minutes.

The managed-recovery timeout the same case compresses is *not* an executor
variable: ``execution/config.py`` reads it on the control-worker to derive the
RESTART_NODE/REPLACE_NODE waiting caps, so it goes through
``control_plane_env_window.py``. Until 2026-09-08 this window carried it too,
and seven live attempts set it where nothing read it.

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
from scripts.e2e.regional.deployment_window_guard import (  # noqa: E402
    apply_window_variables,
    complete_population,
    deployment_snapshot,
    managed_variables_match,
    require_window_record,
    retire_foreign_closed_record,
    window_scope,
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
# DESTR-014 lowers the GPU client verify attempts; HA-004 lowers the executor's
# lease and poll so a WAITING command is re-claimed inside its window. All are
# positive second/attempt counts, so the same validation applies. The
# managed-recovery timeout is a control-worker variable and is deliberately
# absent (see the module docstring).
ALLOWED_VARIABLES = (
    "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS",
    "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS",
    "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS",
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

    Values must be positive integers -- the variable is an attempt count, and
    the case exists to *lower* it, so a non-numeric or zero value is a typo that
    would silently disable the very bound it means to tighten.
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
    return {
        "observed_at": now(),
        **deployment_snapshot(
            value,
            plane="gpu",
            deployment=DEPLOYMENT,
            container=CONTAINER,
            variables=ALLOWED_VARIABLES,
        ),
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
    expected_uid: str | None = None,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + settings.rollout_timeout_seconds
    while True:
        before = deployment_env(regional)
        expected_uid = expected_uid or before["uid"]
        if before["uid"] != expected_uid:
            raise RegionalFixtureError("executor window Deployment UID changed")
        replicas = replica_values(regional)
        after = deployment_env(regional)
        if after["uid"] != expected_uid:
            raise RegionalFixtureError("executor window Deployment UID changed")
        if (
            before["generation"] == after["generation"]
            and complete_population(after, replicas)
            and converged(replicas, expected)
        ):
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
    assignments = parse_assignments(open_arguments(assignments))
    scope = window_scope(
        regional, plane="gpu", deployment=DEPLOYMENT, container=CONTAINER
    )
    record = read_baseline(settings.baseline) if settings.baseline.is_file() else None
    record = retire_foreign_closed_record(settings.baseline, record, scope)
    if record is not None:
        require_window_record(record, scope, report["deployment"], ALLOWED_VARIABLES)
    decision = open_decision(record, report["deployment"], assignments)
    if (
        record is not None
        and record.get("state") == "OPENING"
        and record.get("assignments") == assignments
        and managed_variables_match(
            report["deployment"], record["baseline"], assignments
        )
    ):
        decision = "retry"
    if decision == "refuse":
        raise RegionalFixtureError(
            f"refusing to open the executor env window: baseline record and live "
            f"env disagree on {sorted(assignments)}; close the record first"
        )
    if decision in {"open", "retry"} and (
        not isinstance(report.get("replicas"), list)
        or not complete_population(report["deployment"], report["replicas"])
    ):
        raise RegionalFixtureError(
            "executor window requires the complete healthy replicas"
        )
    if decision == "open":
        record = {
            "schema_version": 2,
            "scope": scope,
            "state": "OPENING",
            "opened_at": now(),
            "confirmation": OPEN_CONFIRMATION,
            "assignments": assignments,
            "baseline": report["deployment"],
            "pre_window_survey": report,
        }
    if decision in {"open", "retry"}:
        if record is None:
            raise RegionalFixtureError("window open has no bound baseline")
        write_json_atomic(settings.baseline, record)
        apply_window_variables(
            regional,
            plane="gpu",
            deployment=DEPLOYMENT,
            container=CONTAINER,
            expected=record["baseline"],
            desired=assignments,
        )
        record["rollout"] = wait_rollout(settings, regional)
    else:
        if record is None:
            raise RegionalFixtureError("window resume has no bound baseline")
        record["resumed_at"] = now()
    record["replicas"] = converge(
        settings,
        regional,
        dict(assignments),
        sleep=sleep,
        expected_uid=record["baseline"]["uid"],
    )
    record["opened_state"] = deployment_env(regional)
    owned = {
        "variables": {
            name: {"present": True, "value": value}
            for name, value in assignments.items()
        }
    }
    if record["opened_state"]["uid"] != record["baseline"][
        "uid"
    ] or not managed_variables_match(record["opened_state"], owned, assignments):
        raise RegionalFixtureError("executor window drifted after convergence")
    record["state"] = "OPEN"
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
    live = deployment_env(regional)
    scope = window_scope(
        regional, plane="gpu", deployment=DEPLOYMENT, container=CONTAINER
    )
    require_window_record(record, scope, live, ALLOWED_VARIABLES)
    baseline = record["baseline"]
    names = record["assignments"]
    restored_already = managed_variables_match(live, baseline, names)
    owned = {
        "uid": baseline["uid"],
        "variables": {
            name: {"present": True, "value": value} for name, value in names.items()
        },
    }
    if not restored_already and not managed_variables_match(live, owned, names):
        raise RegionalFixtureError(
            "executor window values were changed by another writer"
        )
    if record.get("closed_at") and not restored_already:
        raise RegionalFixtureError("closed executor window has drifted")
    previous_closed_at = record.pop("closed_at", None)
    record["state"] = "CLOSING"
    record["close_survey"] = report
    write_json_atomic(settings.baseline, record)
    if not restored_already:
        apply_window_variables(
            regional,
            plane="gpu",
            deployment=DEPLOYMENT,
            container=CONTAINER,
            expected=owned,
            desired={
                name: item["value"] if item["present"] else None
                for name, item in baseline["variables"].items()
                if name in names
            },
        )
    record["rollout_after_close"] = wait_rollout(settings, regional)
    restored = deployment_env(regional)
    record["restored_state"] = restored
    write_json_atomic(settings.baseline, record)
    if restored["uid"] != baseline["uid"] or not managed_variables_match(
        restored, baseline, names
    ):
        raise RegionalFixtureError(
            "restored executor env does not match the recorded baseline: "
            + json.dumps(
                {"baseline": baseline["variables"], "restored": restored["variables"]},
                sort_keys=True,
            )
        )
    expected = {
        name: (entry["value"] if entry["present"] else None)
        for name, entry in baseline["variables"].items()
        if name in names
    }
    record["replicas_after_close"] = converge(
        settings, regional, expected, sleep=sleep, expected_uid=baseline["uid"]
    )
    record["closed_at"] = previous_closed_at or now()
    record["state"] = "CLOSED"
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
