#!/usr/bin/env python3
"""Close a COLLECT-002 power operation the case's own restore could not close.

The case restores through the same host probe it throttled with; when that
restore fails (a reboot ended the boot the operation belongs to, the probe Pod
was lost) the node keeps ``gpu-power-owner.json``, the owner's power record and
the recovery units, and every later attempt under the same run id is refused.
This is the runner's own path to that cleanup: it runs the probe's
``restore-gpu-power-limit`` for the attempt's run id (from the case's
``power-operation-a<attempt>.json``, or ``--power-run-id``) and, with
``--after-reboot``, confirms to the probe that the node rebooted since the
operation by naming its current boot id. The probe then verifies the limits
read back at the recorded baseline before it removes the units and the owner
record; the record and the load-start receipt stay as the node's journal, so
the rerun needs a new ``--attempt``. Plan-only by default; ``--execute`` needs
``--confirm COLLECT002_POWER_CLEANUP`` and a maintenance window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic  # noqa: E402
from scripts.e2e.regional.collect002_power_journal import (  # noqa: E402
    CASE_ID,
    read_power_journal,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.collector_action_guard import action_window  # noqa: E402
from scripts.e2e.regional.live_driver_guard import add_live_arguments  # noqa: E402
from scripts.e2e.regional.plain_case_identity import (  # noqa: E402
    add_plain_identity_arguments,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    install_abort_signals,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.site_profile import install_site_profile  # noqa: E402

CONFIRMATION = "COLLECT002_POWER_CLEANUP"


def power_cleanup(fixture: Any, run_id: str, *, after_reboot: bool) -> dict[str, Any]:
    """Run the probe's restore for ``run_id`` and report what it proved.

    With ``after_reboot`` the node's current boot id is read from the probe
    snapshot and handed to the probe as the confirmed boot; the probe refuses
    it when the record still belongs to this boot (then the plain restore is
    the right call) and when the limits are not at the recorded baseline.
    """

    snapshot = fixture.snapshot()
    arguments = ["restore-gpu-power-limit", "--run-id", run_id]
    boot_id = snapshot.get("boot_id")
    if after_reboot:
        if not isinstance(boot_id, str) or not boot_id:
            raise RegionalFixtureError("host probe snapshot has no boot id")
        arguments += ["--after-reboot", boot_id]
    receipt = fixture.execute(*arguments, timeout=300)
    return {
        "case_id": CASE_ID,
        "run_id": run_id,
        "after_reboot": after_reboot,
        "boot_id": boot_id,
        "receipt": receipt,
        "cleanup_verified": receipt.get("cleanup_verified") is True,
        "gpu_power_before": snapshot.get("gpu_power"),
    }


def resolve_run_id(case_dir: Path, attempt: int, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    journal = read_power_journal(case_dir, attempt)
    run_id = str((journal or {}).get("run_id") or "")
    if not run_id:
        raise RegionalFixtureError(
            f"no power operation is journaled for attempt {attempt} under this "
            "run-dir; pass --power-run-id"
        )
    return run_id


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Close a COLLECT-002 GPU power operation through the host probe's "
            "restore, optionally confirming the node rebooted since it."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    add_plain_identity_arguments(value)
    value.add_argument("--node", required=True)
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--power-run-id", default="")
    value.add_argument(
        "--after-reboot",
        action="store_true",
        help=(
            "the node rebooted since the operation: confirm its current boot id "
            "to the probe so it may close the operation without a power write"
        ),
    )
    return value


def deadline(value: str) -> datetime:
    if not value.strip():
        raise RegionalFixtureError(
            "--maintenance-window-end or GPU_FAULT_ACCEPTANCE_WINDOW_END is required"
        )
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RegionalFixtureError("maintenance-window-end must include a timezone")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = settings_from_arguments(arguments)
    image = required(
        arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        "host probe image",
    )
    case_dir = Path(arguments.run_dir) / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = resolve_run_id(case_dir, arguments.attempt, arguments.power_run_id)
    plan = {
        "case_id": CASE_ID,
        "action": "power-cleanup",
        "attempt": arguments.attempt,
        "power_run_id": run_id,
        "node": arguments.node,
        "after_reboot": bool(arguments.after_reboot),
        "confirmation": CONFIRMATION,
        "mutations": [
            "stop the owned GPU load unit if it still runs",
            "restore the recorded power baseline per GPU (plain restore only)",
            "disable and remove the owned recovery units, remove the owner record",
        ],
        "hard_stop": (
            "the probe refuses a record of another run id, a record whose "
            "boot differs without --after-reboot, and limits that differ from "
            "the recorded baseline after a reboot; the record and load-start "
            "receipt are never removed"
        ),
    }
    if not arguments.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if arguments.confirm != CONFIRMATION:
        raise RegionalFixtureError(f"confirmation must be exactly {CONFIRMATION}")
    window_end = deadline(arguments.maintenance_window_end)
    result: dict[str, Any] = {**plan, "cleanup_verified": False}
    fixture = CollectorAcceptanceFixture(
        RegionalLiveFixture(settings),
        node=arguments.node,
        image=image,
        case_id=CASE_ID,
        run_id=f"collect002-power-cleanup-a{arguments.attempt}",
        case_dir=case_dir,
    )
    with action_window(window_end):
        try:
            fixture.create()
            result.update(
                power_cleanup(fixture, run_id, after_reboot=arguments.after_reboot)
            )
        except Exception as exc:  # noqa: BLE001 - recorded as the cleanup error
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                result["probe_residuals"] = fixture.cleanup()
            except Exception as exc:  # noqa: BLE001 - recorded, never masks
                result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    write_json_atomic(case_dir / f"power-cleanup-a{arguments.attempt}.json", result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["cleanup_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
