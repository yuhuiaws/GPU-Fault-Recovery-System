#!/usr/bin/env python3
"""GF-REGIONAL-PREEMPT-038: raw evidence outlives its TTL while an incident is open.

``RawEvidenceRecord`` rows expire after 24 h while incidents live forever, so
the evidence an operator opened an ESCALATED incident to look at used to be
gone before they looked. ARCH-D6 pins a record to every non-RECOVERED incident
of its cluster that names its attempt, or its node inside the incident's
window; ARCH-D7 makes every sweep log what it deleted. This case runs the
retention audit (``audit_raw_evidence_periodic_cleanup.py``) inside a
control-worker Pod against the live store and reads the sweep's log line.

The audit writes only ``audit-*`` rows under the synthetic ``audit-cluster``
and deletes them in ``finally``; the sweep it observes is the deployed
periodic runner. Plan-only by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import preempt038_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    CONTROL_WORKER_APP,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
AUDIT = Path(__file__).with_name("audit_raw_evidence_periodic_cleanup.py")
LOG_WINDOW_SECONDS = 900


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "no Ready control-worker Pod",
        "the audit leaves any audit-* row in the store",
        "the pinned row is deleted while its incident is open",
        "no cleanup log line names the deleted keys",
    ]


def control_worker_logs(regional: RegionalLiveFixture) -> str:
    chunks = []
    for pod in regional.ready_pods("cpu", CONTROL_WORKER_APP):
        chunks.append(
            regional.kubectl(
                "cpu",
                "logs",
                str(pod["name"]),
                f"--since={LOG_WINDOW_SECONDS}s",
                "--all-containers=true",
                check=False,
                timeout=120,
            )
        )
    return "\n".join(chunks)


def execute(regional: RegionalLiveFixture, case_dir: Path) -> dict[str, Any]:
    report = regional.pod_python(
        "cpu",
        CONTROL_WORKER_APP,
        AUDIT.read_text(encoding="utf-8"),
        timeout=LOG_WINDOW_SECONDS,
    )
    write_json_atomic(case_dir / "audit.json", report)
    logs = control_worker_logs(regional)
    (case_dir / "control-worker.log").write_text(logs, encoding="utf-8")
    (case_dir / "control-worker.log").chmod(0o600)
    stages = {
        "audit": verdicts.audit_errors(report),
        "cleanup_log": verdicts.log_errors(
            logs,
            unrelated_key=str(report.get("unrelated_key") or ""),
            pinned_key=str(report.get("pinned_key") or ""),
        ),
    }
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "audit": report,
        "cleanup_lines": verdicts.cleanup_lines(logs),
        "limitations": [
            "The pinned incident is a synthetic ESCALATED row under audit-cluster; "
            "the attempt-id binding is covered by unit tests only.",
            "The sweep interval is the deployed periodic cadence; the audit waits up "
            "to 180 s per phase.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded PREEMPT-038 acceptance: expired raw evidence an open "
            "incident names survives the sweep and is logged when it finally goes."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = settings_from_arguments(arguments)
    predecessor_id, path = predecessor_path(
        arguments.run_dir, CASE_ID, arguments.predecessor_evidence
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    environment = {**settings.environment(), "GPU_FAULT_EVIDENCE_CASE": CASE_ID}
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details={
                "risk": "live-non-destructive",
                "predecessor": predecessor,
                "mutation": (
                    "insert two already-expired raw_evidence rows and one ESCALATED "
                    "incident under the synthetic cluster audit-cluster, watch the "
                    "deployed sweep, then delete every audit-* row"
                ),
                "stop_conditions": stop_conditions(),
                "rollback": {"audit_rows_deleted_in_finally": True},
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if arguments.confirm != CONFIRMATION:
        raise RegionalFixtureError(f"confirmation must be exactly {CONFIRMATION}")
    authorize_execution(
        arguments, case_id=CASE_ID, confirmation=CONFIRMATION, environment=environment
    )
    if not predecessor.get("valid", False):
        raise RegionalFixtureError("formal predecessor evidence is not PASS")
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        outcome = execute(RegionalLiveFixture(settings), case_dir)
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        outcome = {"verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, CASE_ID), result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
