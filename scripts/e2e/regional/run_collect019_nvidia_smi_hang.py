#!/usr/bin/env python3
"""GF-REGIONAL-COLLECT-019: nvidia-smi hangs; the host collector stays up.

ARCH-G5 gave the host collector one ``nvidia-smi`` call per round (memoised
across inventory, rank and GPU checks), a 15 s timeout on it, and a circuit
breaker that opens after three consecutive timeouts and skips four rounds.
Before that a hung ``nvidia-smi`` stalled the whole collector, which then
looked *silent* to the control plane -- a connectivity page for a GPU-tool
problem. This case shadows ``nvidia-smi`` on one collector unit's PATH with a
wrapper that sleeps past the timeout and then runs the real binary, and
watches the erroring/silent gauges tell the two apart.

Only the host collector unit sees the shadow; the binary, the GPUs and every
other unit are untouched. The window closes itself after
``WINDOW_RESTORE_SECONDS`` if the runner dies. Plan-only by default.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import collect019_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    collector_setting,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    open_window_or_rollback,
    CollectorWindowFixture,
    WindowSettings,
    run_case_main,
    run_window_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "the node has a business workload, ownership or a non-ACTIVE Agent",
        "the hang does not exceed the nvidia-smi timeout or the erroring window "
        "would reach the silence threshold",
        "the host collector restarts or changes MainPID during the hang",
        "the host channel is reported silent instead of erroring",
        "the window cannot be closed or the unit is not active afterwards",
        "the probe Pod or ConfigMap remains after cleanup",
    ]


def plan_details(settings: WindowSettings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "target_node": settings.node,
        "predecessor": preflight["predecessor"],
        "unit": verdicts.UNIT,
        "shadow": verdicts.SHADOW_MODE,
        "mutation": (
            f"put a nvidia-smi wrapper that sleeps {verdicts.HANG_SECONDS}s first on "
            f"the PATH of {verdicts.UNIT} only (drop-in under /run, deadman "
            f"close after {verdicts.WINDOW_RESTORE_SECONDS}s), restart that unit "
            "twice (open, close); the real binary and the GPUs are untouched"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": {
            "window_deadman_seconds": verdicts.WINDOW_RESTORE_SECONDS,
            "runner_finally_closes_the_window": True,
            "runner_finally_deletes_probe_resources": True,
        },
    }


def execute(
    settings: WindowSettings,
    fixture: CollectorWindowFixture,
    case_dir: Path,
    attempt: int,
    deadline: datetime,
) -> dict[str, Any]:
    del deadline
    run_id = f"c019-a{attempt}-{int(time.time())}"
    baseline = fixture.snapshot()
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    interval = collector_setting(
        baseline["collector_env"], "GPU_FAULT_HOST_INTERVAL_SECONDS"
    )
    timing = verdicts.timing_errors(interval_seconds=interval)
    if timing:
        raise RegionalFixtureError("timing: " + "; ".join(timing))
    stages: dict[str, list[str]] = {"timing": timing}
    metrics_before = fixture.control_plane_metrics()
    opened: dict[str, Any] = {}
    statuses: list[dict[str, Any]] = []
    window_open = False
    errors_seen: set[str] = set()
    try:
        opened = open_window_or_rollback(
            fixture,
            run_id,
            "--unit",
            verdicts.UNIT,
            "--shadow-nvidia-smi",
            verdicts.SHADOW_MODE,
            "--restore-seconds",
            str(verdicts.WINDOW_RESTORE_SECONDS),
        )
        window_open = True
        write_json_atomic(case_dir / "window-open.json", opened)
        stages["window"] = verdicts.window_errors(opened)
        # Three timed-out rounds plus the round that reports the open breaker.
        # A round pays the timeout once per distinct query (utilization and
        # inventory), so it is interval + 2 x timeout long under the hang.
        budget = (verdicts.BREAKER_ROUNDS + 2) * (
            interval + 2 * verdicts.NVIDIA_SMI_TIMEOUT_SECONDS
        ) + interval * 2

        def erroring() -> dict[str, Any] | None:
            records = fixture.collector_statuses()
            if verdicts.erroring_status_errors(records, seen=errors_seen):
                return None
            return {"records": records}

        observed = fixture.wait_until(
            erroring,
            timeout_seconds=budget,
            poll_seconds=interval,
            case_dir=case_dir,
            name="erroring",
        )
        statuses = (
            list(observed["records"])
            if observed is not None
            else fixture.collector_statuses()
        )
        during = fixture.snapshot()
        metrics_during = fixture.control_plane_metrics()
        stages["erroring_status"] = verdicts.erroring_status_errors(
            statuses, seen=errors_seen
        )
        stages["service"] = verdicts.service_errors(
            opened.get("after") or {}, during["services"][verdicts.UNIT]
        )
        stages["gauges"] = verdicts.gauge_errors(
            metrics_before, metrics_during, cluster_id=settings.regional.cluster_id
        )
    finally:
        closed: dict[str, Any] = {}
        if window_open:
            closed = fixture.execute("close-window", "--run-id", run_id, timeout=300)
            write_json_atomic(case_dir / "window-close.json", closed)
        stages["closed"] = verdicts.closed_errors(closed) if window_open else []

    def recovered() -> dict[str, Any] | None:
        records = fixture.collector_statuses()
        if verdicts.recovery_errors(records):
            return None
        return {"records": records}

    summary = collector_setting(
        baseline["collector_env"], "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS"
    )
    final = fixture.wait_until(
        recovered,
        timeout_seconds=summary + interval * 3,
        poll_seconds=interval,
        case_dir=case_dir,
        name="recovery",
    )
    final_statuses = (
        list(final["records"]) if final is not None else fixture.collector_statuses()
    )
    stages["recovery"] = verdicts.recovery_errors(final_statuses)
    after = fixture.snapshot()
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "run_id": run_id,
        "host_interval_seconds": interval,
        "window_open": opened,
        "statuses_during": statuses,
        "errors_seen_during_window": sorted(errors_seen),
        "statuses_final": final_statuses,
        "services_after": after["services"],
        "limitations": [
            "The hang is a PATH shadow on one unit; a hung driver that also stalls "
            "DCGM is not simulated.",
            "Two deliberate unit restarts (open, close) are part of the window and "
            "are not counted as crash loops; NRestarts is compared inside the window.",
        ],
    }


def main() -> int:
    return run_window_case(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser_description=(
            "Run the guarded COLLECT-019 acceptance: a hanging nvidia-smi leaves "
            "the host collector up and erroring, never silent."
        ),
        plan_details=plan_details,
        execute=execute,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
