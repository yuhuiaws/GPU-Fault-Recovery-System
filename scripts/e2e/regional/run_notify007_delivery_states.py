#!/usr/bin/env python3
"""GF-REGIONAL-NOTIFY-007: outbox delivery states on a real control plane.

ARCH-E1 made ``result=FAILED`` a terminal verdict: the outbox writes it only
when a delivery is retired DEAD or an inline send fails with nothing to retry.
A retryable provider failure stays on the delivery row as RETRY, which the
new ``gpu_fault_notification_delivery_total{status}`` gauge reports, and the
critical page ``GpuFaultNotificationDeliveryFailing`` reads FAILED results
only. Before E1 one provider hiccup on a row the outbox retried a minute later
fired that page.

The drill (``probes/notify007_delivery_drill.py``) runs in a control-worker
Pod against an isolated in-memory store with the Pod's real SES notifier
wrapped to fail once, then to fail always -- one labelled DRILL email is sent,
no production row is written. The deployment's ``/metrics`` is read before
and after to show the E1 families and that no production FAILED appeared.
Plan-only by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import notify007_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    utc_now,
    write_json_atomic,
)
from scripts.e2e.regional.collector_window_fixture import (  # noqa: E402
    API_APP,
    API_METRICS_PORT,
    CONTROL_WORKER_APP,
    METRICS_PROBE,
    WORKER_METRICS_PORT,
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
DRILL_PROBE = Path(__file__).with_name("probes") / "notify007_delivery_drill.py"
RULES = ROOT / "deploy" / "observability" / "amp-rules.yaml"
RUNBOOK = ROOT / "docs" / "管理员日常运维.md"


def stop_conditions() -> list[str]:
    return [
        "predecessor evidence is not PASS",
        "no Ready control-worker Pod",
        "the drill's first cycle writes a result row or counts DEAD",
        "the retried delivery is not SENT with a provider message id",
        "the exhausted delivery is not DEAD with a terminal FAILED result",
        "a production FAILED result or DEAD row appears during the drill",
    ]


def control_plane_metrics(regional: RegionalLiveFixture) -> list[str]:
    texts: list[str] = []
    for app, port in (
        (CONTROL_WORKER_APP, WORKER_METRICS_PORT),
        (API_APP, API_METRICS_PORT),
    ):
        for pod in regional.ready_pods("cpu", app):
            output = regional.kubectl(
                "cpu",
                "exec",
                "-i",
                str(pod["name"]),
                "--",
                "python3",
                "-",
                str(port),
                input_text=METRICS_PROBE,
                timeout=60,
            )
            texts.append(str(json.loads(output.splitlines()[-1])["metrics"]))
    return texts


def run_drill(regional: RegionalLiveFixture, *, drill_id: str) -> dict[str, Any]:
    return regional.pod_python(
        "cpu",
        CONTROL_WORKER_APP,
        DRILL_PROBE.read_text(encoding="utf-8"),
        "--drill-id",
        drill_id,
        "--cluster-id",
        regional.settings.cluster_id,
        timeout=300,
    )


def execute(
    regional: RegionalLiveFixture, case_dir: Path, attempt: int
) -> dict[str, Any]:
    drill_id = f"notify007-{attempt}-{int(time.time())}"
    before = control_plane_metrics(regional)
    drill = run_drill(regional, drill_id=drill_id)
    write_json_atomic(case_dir / "drill.json", drill)
    after = control_plane_metrics(regional)
    stages = {
        "retry": verdicts.retry_phase_errors(drill.get("retry") or {}),
        "dead": verdicts.dead_phase_errors(drill.get("dead") or {}),
        "exported_families": verdicts.exported_family_errors(after),
        "production_untouched": verdicts.production_untouched_errors(before, after),
        "alert_rule": verdicts.alert_rule_errors(
            RULES.read_text(encoding="utf-8"), RUNBOOK.read_text(encoding="utf-8")
        ),
    }
    return {
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "drill_id": drill_id,
        "drill": drill,
        "limitations": [
            "The delivery rows live in an isolated in-memory store; the production "
            "gauges are read only to prove the families exist and did not move.",
            "The alert is judged by its rule text and runbook anchor, not by an AMP "
            "evaluation: the drill deliberately creates no production FAILED result.",
            "One labelled DRILL email is sent through SES on the retry path.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded NOTIFY-007 acceptance: RETRY is not FAILED, DEAD is."
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
    environment = {**settings.environment(), "GPU_FAULT_NOTIFICATION_CASE": CASE_ID}
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
                    "run one isolated in-memory outbox drill in a control-worker Pod; "
                    "one labelled DRILL email is sent; no production row is written"
                ),
                "stop_conditions": stop_conditions(),
                "rollback": {"drill_store_is_in_memory_and_discarded": True},
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
    regional = RegionalLiveFixture(settings)
    try:
        outcome = execute(regional, case_dir, arguments.attempt)
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
