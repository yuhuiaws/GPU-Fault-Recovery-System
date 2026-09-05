#!/usr/bin/env python3
"""Open or close the synthetic node-replacement route for one test window.

`GF-REGIONAL-DESTR-003` drives a real warm-spare workflow, and the only honest
way to start one without damaging a GPU is
`POST /v1/admin/test/node-replacement`. That route is gated on
`GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS`, which the shipped regional
manifest never sets, so it answers 404 -- and the case's `preflight_errors`
refuses to run until every API replica has it enabled.

Nothing supported set it. The acceptance history told the operator to
`kubectl set env` the Deployment by hand and to "remember to remove it
afterwards", which is the shape of residual this suite exists to catch: the
route fabricates a `REPLACE_NODE` finding for any node in the cluster, so
leaving it on is a standing way to make the control plane quarantine and
fail over a healthy machine. It is protected by the workflow execution token,
which is why this is a test-window switch and not a vulnerability -- but a
switch nobody recorded turning on is one nobody turns off.

So this records the Deployment's pre-window state in a baseline file before
mutating, verifies the rollout reached every replica, and `--close` restores
exactly what the baseline recorded: a variable that was absent goes back to
absent rather than to `false`. It refuses to open a window that is already open
under a baseline it did not write, and refuses to close one it has no record of.

It touches only the API tier's own env. It does not restart the workers, does
not change the release, and cannot be used to change any other variable: the
name is compiled in.
"""

from __future__ import annotations

import argparse
import json
import sys
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

DEPLOYMENT = "gpu-fault-api-ha"
CONTAINER = "api"
ROUTE_ENV = "GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS"
OPEN_CONFIRMATION = "OPEN_SYNTHETIC_REPLACEMENT_WINDOW"
CLOSE_CONFIRMATION = "CLOSE_SYNTHETIC_REPLACEMENT_WINDOW"


@dataclass(frozen=True)
class Settings:
    baseline: Path
    rollout_timeout_seconds: int


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def configure(arguments: argparse.Namespace) -> Settings:
    return Settings(
        baseline=Path(
            required(arguments.baseline, "baseline record path")
        ).expanduser(),
        rollout_timeout_seconds=int(arguments.rollout_timeout_seconds),
    )


def deployment_env(regional: RegionalLiveFixture) -> dict[str, Any]:
    """The API container's literal env, plus the identity of what carries it."""

    value = json.loads(
        regional.kubectl("cpu", "get", "deployment", DEPLOYMENT, "-o", "json")
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
    present = [item for item in entries if item.get("name") == ROUTE_ENV]
    if any("valueFrom" in item for item in present):
        # A restore writes a literal, so a valueFrom entry would be silently
        # rewritten into a different kind of variable.
        raise RegionalFixtureError(
            f"{ROUTE_ENV} is set from a reference, not a literal; "
            "this helper only manages a literal value"
        )
    return {
        "observed_at": now(),
        "deployment": DEPLOYMENT,
        "container": CONTAINER,
        "generation": value["metadata"]["generation"],
        "resource_version": value["metadata"]["resourceVersion"],
        "replicas": value["spec"].get("replicas"),
        "route_env_present": bool(present),
        "route_env_value": present[0].get("value") if present else None,
    }


def pod_gates(regional: RegionalLiveFixture) -> list[dict[str, str | None]]:
    """What each ready replica actually has, which is what the case grades."""

    result: list[dict[str, str | None]] = []
    for pod in regional.ready_pods("cpu", DEPLOYMENT):
        output = regional.kubectl(
            "cpu",
            "exec",
            str(pod["name"]),
            "--",
            "python3",
            "-c",
            f"import json,os; print(json.dumps({{'enabled':os.getenv({ROUTE_ENV!r})}}))",
            timeout=60,
        )
        enabled = json.loads(output.splitlines()[-1]).get("enabled")
        result.append(
            {
                "pod": str(pod["name"]),
                "enabled": None if enabled is None else str(enabled),
            }
        )
    return result


def survey(regional: RegionalLiveFixture) -> dict[str, Any]:
    return {
        "observed_at": now(),
        "cluster_id": regional.settings.cluster_id,
        "deployment": deployment_env(regional),
        "pod_gates": pod_gates(regional),
    }


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegionalFixtureError(f"no synthetic-route baseline record at {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def wait_rollout(settings: Settings, regional: RegionalLiveFixture) -> str:
    return regional.kubectl(
        "cpu",
        "rollout",
        "status",
        f"deployment/{DEPLOYMENT}",
        f"--timeout={settings.rollout_timeout_seconds}s",
        timeout=settings.rollout_timeout_seconds + 60,
    ).strip()


def open_window(
    settings: Settings,
    regional: RegionalLiveFixture,
    report: dict[str, Any],
) -> dict[str, Any]:
    if settings.baseline.is_file() and not read_baseline(settings.baseline).get(
        "closed_at"
    ):
        raise RegionalFixtureError(
            f"{settings.baseline} still records an open window; close it first"
        )
    if report["deployment"]["route_env_value"] == "true":
        raise RegionalFixtureError(
            f"{ROUTE_ENV} is already true on {DEPLOYMENT} without a record here; "
            "find out who opened it before adding a second owner"
        )
    # Written before the mutation, so an interrupted open still leaves an exact
    # record of what to put back -- including "the variable was absent".
    record = {
        "opened_at": now(),
        "confirmation": OPEN_CONFIRMATION,
        "baseline": report["deployment"],
        "pre_window_survey": report,
    }
    write_json_atomic(settings.baseline, record)
    regional.kubectl(
        "cpu",
        "set",
        "env",
        f"deployment/{DEPLOYMENT}",
        f"--containers={CONTAINER}",
        f"{ROUTE_ENV}=true",
    )
    record["rollout"] = wait_rollout(settings, regional)
    gates = pod_gates(regional)
    record["pod_gates"] = gates
    if not gates or any(item["enabled"] != "true" for item in gates):
        raise RegionalFixtureError(
            "the route is not enabled on every ready replica after the rollout: "
            + json.dumps(gates, sort_keys=True)
        )
    record["opened_state"] = deployment_env(regional)
    write_json_atomic(settings.baseline, record)
    return record


def close_window(
    settings: Settings,
    regional: RegionalLiveFixture,
    report: dict[str, Any],
) -> dict[str, Any]:
    record = read_baseline(settings.baseline)
    if record.get("closed_at"):
        raise RegionalFixtureError(
            f"{settings.baseline} was already closed at {record['closed_at']}"
        )
    baseline = record["baseline"]
    if baseline.get("route_env_present"):
        regional.kubectl(
            "cpu",
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            f"--containers={CONTAINER}",
            f"{ROUTE_ENV}={baseline['route_env_value']}",
        )
    else:
        # Absent, not "false": the shipped manifest carries no such variable, and
        # a leftover `false` is config the release does not know it has.
        regional.kubectl(
            "cpu",
            "set",
            "env",
            f"deployment/{DEPLOYMENT}",
            f"--containers={CONTAINER}",
            f"{ROUTE_ENV}-",
        )
    record["closed_at"] = now()
    record["close_survey"] = report
    record["rollout_after_close"] = wait_rollout(settings, regional)
    restored = deployment_env(regional)
    record["restored_state"] = restored
    gates = pod_gates(regional)
    record["pod_gates_after_close"] = gates
    write_json_atomic(settings.baseline, record)
    if restored["route_env_present"] != baseline.get("route_env_present") or restored[
        "route_env_value"
    ] != baseline.get("route_env_value"):
        raise RegionalFixtureError(
            "the restored env does not match the recorded baseline: "
            + json.dumps({"baseline": baseline, "restored": restored}, sort_keys=True)
        )
    if any(item["enabled"] == "true" for item in gates):
        raise RegionalFixtureError(
            "a replica still has the route enabled after the close: "
            + json.dumps(gates, sort_keys=True)
        )
    return record


def without_survey(record: dict[str, Any]) -> dict[str, Any]:
    # The baseline record embeds the survey and the survey is this run's report,
    # so attaching the record back unfiltered would make the report contain
    # itself and only fail at the closing `json.dumps` -- after the mutation.
    return {key: value for key, value in record.items() if not key.endswith("_survey")}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Open or close the synthetic node-replacement route for "
            "GF-REGIONAL-DESTR-003. Read-only unless --open or --close is given."
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
    # Not a live_driver_guard runner, so it binds the profile itself.
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
        report["open"] = without_survey(open_window(settings, regional, report))
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
