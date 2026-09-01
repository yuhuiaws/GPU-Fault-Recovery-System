from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .acceptance_runner_common import write_json_atomic
    from .acceptance_scope import current_acceptance_scope
else:
    from acceptance_runner_common import write_json_atomic
    from acceptance_scope import current_acceptance_scope


COMMON_ENVIRONMENT_KEYS = (
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "GPU_FAULT_DATAPLANE_CONTEXT",
    "GPU_FAULT_PERF_AWS_REGION",
    "GPU_FAULT_PERF_CONTROL_NAMESPACE",
    "GPU_FAULT_PERF_DATAPLANE_NAMESPACE",
    "KUBECONFIG",
)


def add_live_arguments(
    parser: argparse.ArgumentParser,
    *,
    confirmation: str,
) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f"execute mode requires exactly {confirmation}",
    )
    parser.add_argument(
        "--maintenance-window-end",
        default=os.getenv("GPU_FAULT_ACCEPTANCE_WINDOW_END", ""),
        help="UTC ISO-8601 deadline required by execute mode",
    )


def environment_snapshot(
    keys: tuple[str, ...] = COMMON_ENVIRONMENT_KEYS,
) -> dict[str, str]:
    missing = [key for key in keys if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError(
            "required live-driver environment is missing: " + ", ".join(sorted(missing))
        )
    return {key: os.environ[key].strip() for key in keys}


def _deadline(value: str) -> datetime:
    if not value.strip():
        raise RuntimeError(
            "--maintenance-window-end or GPU_FAULT_ACCEPTANCE_WINDOW_END is required"
        )
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("maintenance-window-end must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_plan(
    *,
    run_dir: Path,
    case_id: str,
    attempt: int,
    confirmation: str,
    details: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    scope = current_acceptance_scope()
    plan = {
        "schema_version": 2,
        "case_id": case_id,
        "attempt": attempt,
        "confirmation": confirmation,
        "environment": environment or environment_snapshot(),
        **scope.plan_fields(),
        "details": details,
        "mutation_performed": False,
    }
    write_json_atomic(run_dir / "cases" / case_id / "plan.json", plan)
    return plan


def authorize_execution(
    arguments: argparse.Namespace,
    *,
    case_id: str,
    confirmation: str,
    environment: dict[str, str] | None = None,
) -> datetime:
    if not arguments.execute:
        raise RuntimeError("execute authorization requested outside --execute mode")
    if arguments.confirm != confirmation:
        raise RuntimeError(f"confirmation must be exactly {confirmation}")
    deadline = _deadline(arguments.maintenance_window_end)
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("approved maintenance window has ended")
    path = arguments.run_dir / "cases" / case_id / "plan.json"
    if not path.is_file():
        raise RuntimeError("run --plan before --execute")
    plan = json.loads(path.read_text(encoding="utf-8"))
    scope = current_acceptance_scope()
    expected = {
        "schema_version": 2,
        "case_id": case_id,
        "attempt": arguments.attempt,
        "confirmation": confirmation,
        "environment": environment or environment_snapshot(),
        **scope.plan_fields(),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"live-driver plan drifted at {key}")
    return deadline
