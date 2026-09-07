#!/usr/bin/env python3
"""GF-REGIONAL-COLLECT-020: a GPU vanishes from one inventory query.

ARCH-G6 made a changed GPU UUID set a CRITICAL ``gpu_inventory_identity_changed``
finding (DCGM diagnostic + validation, no reboot) instead of a raw-evidence
row nobody reads, and gave the inventory batch an expected GPU count derived
from the instance type when ``GPU_FAULT_EXPECTED_GPU_COUNT`` is unset. This
case proves both on a real node, plus the notification the diagnostic mails
(ARCH-E7) and the retirement fields on the incident's markers once it is
restored (ARCH-I4).

The disappearance is *never* a device change: a ``nvidia-smi`` shadow on the
metrics collector's PATH drops one UUID line from exactly one inventory query
and passes everything else through. One query is below the host collector's
``GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES`` (2), so the reboot path
cannot be reached, and the runner stops on any workflow that compiles a
reboot or reset anyway. The incident is restored through the validation-first
restore workflow (never by deleting taints). Plan-only by default.
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
# The verdicts read the instance-type table from the collector package itself.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(1, str(ROOT / "src"))

from scripts.e2e.regional import collect020_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    collector_setting,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    CollectorWindowFixture,
    WindowSettings,
    run_case_main,
    run_window_case,
    utc_now,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "the node has a business workload, ownership or a non-ACTIVE Agent",
        "the instance type is not in the expected-count table",
        "the drop would reach the host collector's mismatch threshold",
        "any workflow compiles RESTART_NODE, REPLACE_NODE or a reset",
        "the identity-changed workflow does not reach a terminal state",
        "the node cannot be restored through the validation-first workflow",
        "the probe Pod or ConfigMap remains after cleanup",
    ]


def plan_details(settings: WindowSettings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-isolation",
        "target_node": settings.node,
        "predecessor": preflight["predecessor"],
        "unit": verdicts.UNIT,
        "mutation": (
            f"on {verdicts.UNIT} only: unset GPU_FAULT_EXPECTED_GPU_COUNT and shadow "
            "nvidia-smi so ONE inventory query omits one GPU UUID (drop-uuid:<uuid>:1); "
            "the finding cordons the node for a DCGM diagnostic and validation, then "
            "the validation-first restore workflow uncordons it"
        ),
        "hard_stop": (
            "the GPU is never touched; one hidden query is below the host "
            "collector's 2-sample mismatch threshold, so REBOOT_NODE is unreachable"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": {
            "window_deadman_seconds": verdicts.WINDOW_RESTORE_SECONDS,
            "runner_finally_closes_the_window": True,
            "quarantine_is_restored_only_by_validation_workflow": True,
            "runner_finally_deletes_probe_resources": True,
        },
    }


def _restore(
    fixture: CollectorWindowFixture,
    activity: dict[str, Any],
    *,
    profile_version: str,
) -> list[dict[str, Any]]:
    restore = WarmSpareLiveFixture(fixture.regional, "")
    results = []
    for incident in activity.get("incidents") or []:
        node = fixture.regional.node_snapshot(fixture.node)
        if not node["ownership_annotations"]:
            break
        created = restore.create_restore_workflow(
            incident_id=str(incident["incident_id"]),
            node=fixture.node,
            profile_version=profile_version,
            reason="COLLECT-020 validated cleanup",
        )
        results.append(restore.wait_workflow_id(str(created["workflow_request_id"])))
    return results


def execute(
    settings: WindowSettings,
    fixture: CollectorWindowFixture,
    case_dir: Path,
    attempt: int,
    deadline: datetime,
) -> dict[str, Any]:
    del deadline
    run_id = f"c020-a{attempt}-{int(time.time())}"
    baseline = fixture.snapshot()
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    env = baseline["collector_env"]
    inventory = baseline.get("gpu_inventory") or []
    instance_type = env.get("GPU_FAULT_NODE_INSTANCE_TYPE")
    preconditions = verdicts.preconditions_errors(
        instance_type=instance_type,
        gpu_count=len(inventory),
        mismatch_samples=collector_setting(
            env, "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"
        ),
    )
    if preconditions:
        raise RegionalFixtureError("preconditions: " + "; ".join(preconditions))
    expected_count = verdicts.expected_count_for(instance_type)
    if expected_count is None:
        raise RegionalFixtureError("instance type has no expected GPU count")
    dropped_uuid = str(inventory[-1]["uuid"])
    profile_version = str(
        (fixture.regional.store_snapshot(node=fixture.node).get("profile") or {}).get(
            "profile_version"
        )
        or ""
    )
    stages: dict[str, list[str]] = {"preconditions": preconditions}
    since = utc_now()
    opened: dict[str, Any] = {}
    window_open = False
    activity: dict[str, Any] = {}
    try:
        opened = fixture.execute(
            "open-window",
            "--run-id",
            run_id,
            "--unit",
            verdicts.UNIT,
            "--unset",
            "GPU_FAULT_EXPECTED_GPU_COUNT",
            "--shadow-nvidia-smi",
            f"drop-uuid:{dropped_uuid}:{verdicts.DROP_CALLS}",
            "--restore-seconds",
            str(verdicts.WINDOW_RESTORE_SECONDS),
            timeout=300,
        )
        window_open = True
        write_json_atomic(case_dir / "window-open.json", opened)
        stages["window"] = verdicts.window_errors(opened, dropped_uuid=dropped_uuid)

        def finding_opened() -> dict[str, Any] | None:
            value = fixture.node_activity(since, evidence_kind="GPU_INVENTORY")
            if not verdicts.identity_finding_errors(
                {**value, "workflows": []}, dropped_uuid=dropped_uuid
            ) or verdicts.no_reboot_errors(
                value,
                boot_id_before=baseline["boot_id"],
                boot_id_after=baseline["boot_id"],
            ):
                return value
            return None

        activity = fixture.wait_until(
            finding_opened,
            timeout_seconds=verdicts.FINDING_TIMEOUT_SECONDS,
            poll_seconds=10,
            case_dir=case_dir,
            name="finding",
        ) or fixture.node_activity(since, evidence_kind="GPU_INVENTORY")
    finally:
        # The shadow has spent its one query by now; close early so the window
        # never overlaps the diagnostic workflow's own nvidia-smi reads.
        closed: dict[str, Any] = {}
        if window_open:
            closed = fixture.execute("close-window", "--run-id", run_id, timeout=300)
            write_json_atomic(case_dir / "window-close.json", closed)
        stages["closed"] = (
            []
            if not window_open or closed.get("dropin_removed")
            else ["drop-in remains"]
        )
    reboot_errors = verdicts.no_reboot_errors(
        activity, boot_id_before=baseline["boot_id"], boot_id_after=baseline["boot_id"]
    )
    if reboot_errors:
        # Hard stop: restore what can be restored and fail without waiting.
        stages["no_reboot"] = reboot_errors
        stages["restore"] = []
        try:
            _restore(fixture, activity, profile_version=profile_version)
        except Exception as exc:  # noqa: BLE001 - recorded on a failing path
            stages["restore"] = [f"{type(exc).__name__}: {exc}"]
        return {"verdict": "FAIL", "stages": stages, "activity": activity}

    def workflows_terminal() -> dict[str, Any] | None:
        value = fixture.node_activity(since, evidence_kind="GPU_INVENTORY")
        workflows = value.get("workflows") or []
        if workflows and all(
            item.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}
            for item in workflows
        ):
            return value
        return None

    activity = fixture.wait_until(
        workflows_terminal,
        timeout_seconds=verdicts.WORKFLOW_TIMEOUT_SECONDS,
        poll_seconds=15,
        case_dir=case_dir,
        name="workflow",
    ) or fixture.node_activity(since, evidence_kind="GPU_INVENTORY")
    write_json_atomic(case_dir / "activity.json", activity)
    stages["inventory_evidence"] = verdicts.inventory_evidence_errors(
        activity.get("evidence") or [],
        dropped_uuid=dropped_uuid,
        expected_count=expected_count,
    )
    stages["identity_finding"] = verdicts.identity_finding_errors(
        activity, dropped_uuid=dropped_uuid
    )
    stages["dcgm_notification"] = verdicts.dcgm_notification_errors(activity)
    restores = _restore(fixture, activity, profile_version=profile_version)
    write_json_atomic(case_dir / "restore.json", {"workflows": restores})
    final = fixture.node_activity(since, evidence_kind="GPU_INVENTORY")
    after = fixture.snapshot()
    stages["no_reboot"] = verdicts.no_reboot_errors(
        final, boot_id_before=baseline["boot_id"], boot_id_after=after["boot_id"]
    )
    stages["marker_retirement"] = verdicts.marker_retirement_errors(final)
    stages["node_restored"] = verdicts.restored_node_errors(
        fixture.regional.node_snapshot(fixture.node)
    )
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "run_id": run_id,
        "instance_type": instance_type,
        "expected_gpu_count": expected_count,
        "dropped_uuid": dropped_uuid,
        "window_open": opened,
        "activity": {key: value for key, value in final.items() if key != "evidence"},
        "inventory_evidence_record_ids": [
            item.get("record_id") for item in final.get("evidence") or []
        ],
        "restore_workflows": restores,
        "limitations": [
            "The lost GPU is a shadow of one nvidia-smi inventory query; the device "
            "itself never left the bus.",
            "The DCGM device-lost edge reason comes from the DCGM exporter and is "
            "not reachable through nvidia-smi shadowing; it keeps unit evidence only.",
            "The UUID's return is itself an identity change and may open a second "
            "finding; every incident of the case is restored and judged.",
        ],
    }


def main() -> int:
    return run_window_case(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser_description=(
            "Run the guarded COLLECT-020 acceptance: a GPU UUID missing from one "
            "inventory query becomes a CRITICAL identity finding with a diagnostic "
            "workflow, an instance-type expected count and retired markers."
        ),
        plan_details=plan_details,
        execute=execute,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
