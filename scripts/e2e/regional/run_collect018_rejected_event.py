#!/usr/bin/env python3
"""GF-REGIONAL-COLLECT-018: a fault-layer event rejected after the 202.

Every collector channel is ``receipt=True``: the ingress answers 202 after
decoding the JSON and the handler's real verdict lands when the processor
replays the row. Before ARCH-G1 a Pydantic 422 on that replay completed as a
success at INFO -- a payload-shape drift on the kernel XID channel looked like
a healthy pipeline whose nodes had gone quiet. This case makes the rejection
visible on a real deployment and, separately (ARCH-G7), proves an ``Xid`` line
without a code becomes an operator-review finding rather than nothing.

Phase A posts one ``NvidiaKernelLogEvent`` carrying a field the strict model
forbids, through the node's own configured sink -- a *synthetic API
injection*, labelled as such in the payload; no hardware fault. Phase B writes
one code-less ``NVRM: Xid`` line to the real ``/dev/kmsg`` from user space.
Phase C writes NET-001's monitor-only XID 63 so a genuine accepted batch
clears the erroring state. Nothing is cordoned, reset or rebooted.
Plan-only by default.
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

from scripts.e2e.regional import collect018_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
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

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "the node has a business workload, ownership or a non-ACTIVE Agent",
        "the host probe cannot post through the node's configured sink",
        "the rejection opens any workflow that isolates, resets or reboots",
        "the code-less line compiles a step outside FREEZE_EVIDENCE",
        "the kernel channel does not return to a non-erroring state",
        "the probe Pod or ConfigMap remains after cleanup",
    ]


def plan_details(settings: WindowSettings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-kernel-log-injection",
        "target_node": settings.node,
        "predecessor": preflight["predecessor"],
        "mutation": (
            "post one deliberately incompatible NvidiaKernelLogEvent through the "
            "node's own sink (synthetic API injection), write one code-less "
            "NVRM: Xid line and one monitor-only XID 63 to /dev/kmsg (user-space "
            "injection); no hardware fault is simulated"
        ),
        "expected_status_prefix": verdicts.REJECTED_PREFIX,
        "expected_finding_kind": verdicts.UNPARSED_KIND,
        "stop_conditions": stop_conditions(),
        "rollback": {
            "no_node_or_control_plane_setting_is_changed": True,
            "runner_finally_deletes_probe_resources": True,
            "findings_open_only_freeze_evidence_workflows": True,
        },
    }


def _first_bdf(snapshot: dict[str, Any]) -> str:
    inventory = snapshot.get("gpu_inventory") or []
    if not inventory:
        raise RegionalFixtureError("host probe read no GPU inventory")
    return str(inventory[0]["pci_bus_id"])


def execute(
    settings: WindowSettings,
    fixture: CollectorWindowFixture,
    case_dir: Path,
    attempt: int,
    deadline: datetime,
) -> dict[str, Any]:
    del deadline
    marker = f"c018-{int(time.time())}-a{attempt}"
    started_at = utc_now()
    baseline = fixture.snapshot()
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    metrics_before = fixture.control_plane_metrics()
    statuses_before = fixture.collector_statuses()
    stages: dict[str, list[str]] = {}
    evidence: dict[str, Any] = {
        "marker": marker,
        "host_baseline": baseline,
        "statuses_before": statuses_before,
    }

    # Phase A: the rejected payload.
    posted = fixture.execute(
        "post-rejected-event",
        "--marker",
        marker,
        "--node-id",
        settings.node,
        timeout=300,
    )
    write_json_atomic(case_dir / "posted.json", posted)
    evidence["posted"] = posted
    record_id = str(posted.get("record_id") or f"acceptance-rejected-{marker}")
    rejected_statuses = fixture.wait_until(
        lambda: (
            {"records": records}
            if not verdicts.rejected_status_errors(
                records := fixture.collector_statuses()
            )
            else None
        ),
        timeout_seconds=verdicts.REJECTION_TIMEOUT_SECONDS,
        poll_seconds=5,
        case_dir=case_dir,
        name="rejection",
    )
    statuses_after = (
        list(rejected_statuses["records"])
        if rejected_statuses is not None
        else fixture.collector_statuses()
    )
    metrics_after = fixture.control_plane_metrics()
    logs = fixture.control_plane_logs(
        int((utc_now() - started_at).total_seconds()) + 60
    )
    (case_dir / "control-plane.log").write_text(logs, encoding="utf-8")
    (case_dir / "control-plane.log").chmod(0o600)
    stages["rejected_status"] = verdicts.rejected_status_errors(statuses_after)
    stages["rejection_metrics"] = verdicts.rejection_metric_errors(
        metrics_before, metrics_after
    )
    stages["rejection_log"] = verdicts.rejection_log_errors(logs, record_id)
    stages["not_silent"] = verdicts.silence_errors(
        metrics_before,
        metrics_after,
        cluster_id=settings.regional.cluster_id,
        node=settings.node,
    )
    evidence["statuses_after_rejection"] = statuses_after

    # Phase B: the code-less Xid line.
    unparsed_since = utc_now()
    written = fixture.execute(
        "write-kmsg",
        "--kind",
        "unparsed-xid",
        "--marker",
        marker,
        "--pci-bdf",
        _first_bdf(baseline),
    )
    evidence["unparsed_write"] = written
    activity = fixture.wait_until(
        lambda: (
            value
            if (
                value := fixture.node_activity(
                    unparsed_since, evidence_kind="NVIDIA_KERNEL"
                )
            )
            and value.get("incidents")
            and value.get("notifications")
            and all(
                item.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}
                for item in value.get("workflows") or []
            )
            else None
        ),
        timeout_seconds=verdicts.FINDING_TIMEOUT_SECONDS,
        poll_seconds=5,
        case_dir=case_dir,
        name="unparsed-finding",
    ) or fixture.node_activity(unparsed_since, evidence_kind="NVIDIA_KERNEL")
    write_json_atomic(case_dir / "unparsed-activity.json", activity)
    metrics_final = fixture.control_plane_metrics()
    stages["unparsed_finding"] = verdicts.unparsed_finding_errors(
        activity,
        marker=marker,
        metrics_before=metrics_after,
        metrics_after=metrics_final,
    )
    evidence["unparsed_activity"] = {
        key: value for key, value in activity.items() if key != "evidence"
    }
    evidence["unparsed_evidence_record_ids"] = [
        item.get("record_id") for item in activity.get("evidence") or []
    ]

    # Phase C: one accepted batch clears the erroring state.
    fixture.execute(
        "write-kmsg",
        "--kind",
        "xid63",
        "--marker",
        f"{marker}-recover",
        "--pci-bdf",
        _first_bdf(baseline),
    )
    recovered = fixture.wait_until(
        lambda: (
            {"records": records}
            if not verdicts.recovery_errors(records := fixture.collector_statuses())
            else None
        ),
        timeout_seconds=verdicts.RECOVERY_TIMEOUT_SECONDS,
        poll_seconds=10,
        case_dir=case_dir,
        name="recovery",
    )
    final_statuses = (
        list(recovered["records"])
        if recovered is not None
        else fixture.collector_statuses()
    )
    stages["recovery"] = verdicts.recovery_errors(final_statuses)
    node = fixture.regional.node_snapshot(settings.node)
    stages["node_untouched"] = verdicts.node_untouched_errors(node)
    evidence["statuses_final"] = final_statuses
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        **evidence,
        "limitations": [
            "The rejected payload is a synthetic API post, not a kernel fault; the "
            "kernel collector itself cannot be made to build an incompatible payload.",
            "The code-less Xid line is a user-space /dev/kmsg write labelled as such.",
        ],
    }


def main() -> int:
    return run_window_case(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser_description=(
            "Run the guarded COLLECT-018 acceptance: a fault-layer event rejected "
            "after the ingress 202 is counted, logged and shown on the node's "
            "collector status; a code-less Xid line becomes a finding."
        ),
        plan_details=plan_details,
        execute=execute,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
