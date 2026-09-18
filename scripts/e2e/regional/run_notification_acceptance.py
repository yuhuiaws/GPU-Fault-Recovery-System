from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.identity_acceptance_common import (  # noqa: E402
    ClusterTarget,
    IdentitySite,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
    record_focused_tests,
    reusable_focused_tests,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    TRAINING_IMAGE,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.notification_evidence import (  # noqa: E402
    NOTIFICATION_KINDS,
    NotificationAcceptanceError,
    action_completed_records_from_evidence,
    completion_checks,
    notification_kind as notification_kind,
    parse_time,
    requeue_route_errors as requeue_route_errors,
    select_live_record as select_live_record,
    validate_duplicate_evidence,
    validate_external_evidence,
)
from scripts.e2e.regional.notify005_checks import (  # noqa: E402
    low_utilization_manifest,
    node_errors,
    phase_checks,
)
from scripts.e2e.regional.notification_probe_cache import (  # noqa: E402
    cached_drill,
    drill_plan_binding,
    drill_source_digest,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_commands import run_fixture_command  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    install_abort_signals,
    predecessor_evidence,
    run_case_main,
)

CASE_IDS = tuple(f"GF-REGIONAL-NOTIFY-{number:03d}" for number in range(1, 6))
DRILL_PROBE = (Path(__file__).with_name("probes") / "notification_drill.py").read_text(
    encoding="utf-8"
)
# NOTIFY-001 replays the result handler four times in an isolated Store.
# The legacy helper remains importable; the NOTIFY-002 CLI is retired.
NOTIFY002_SUPERSEDED_BY = "GF-REGIONAL-NOTIFY-001"
YAML_ALIAS_PATTERN = re.compile(r"(?:^|\s)[&*]id\d+\b")
NOTIFY003_FOCUSED_TESTS = (
    "tests/notifications/test_notifications.py::"
    "test_first_dispatch_suppresses_the_pre_enable_backlog",
    "tests/notifications/test_notifications.py::"
    "test_suppressed_backlog_can_be_requeued_on_demand",
    "tests/notifications/test_notifications.py::"
    "test_notification_worker_reclaims_expired_lease_and_stops",
    "tests/notifications/test_acceptance_alignment_requeue.py",
)
REQUEUE_PROBE = Path(__file__).with_name("probes") / "notify003_requeue_drill.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 300,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_fixture_command(command, check=check, timeout=timeout, cwd=cwd, env=env)


def control_worker_pod(site: IdentitySite, target: ClusterTarget) -> str:
    """One Ready control-worker Pod, resolved once per case.

    Every drill used to list the Pods again; the answer does not change between
    two drills seconds apart, and each list is a kubectl round trip.
    """

    pods = site.ready_pods("cpu", "gpu-fault-control-worker", target)
    if not pods:
        raise NotificationAcceptanceError("no Ready control-worker Pod")
    return pods[0]


def drill(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    kind: str,
    drill_id: str,
    pod: str | None = None,
    maintenance_window_end: datetime | None = None,
    duplicate_delay_seconds: int = 0,
) -> dict[str, Any]:
    if (
        maintenance_window_end is not None
        and datetime.now(timezone.utc) >= maintenance_window_end
    ):
        raise NotificationAcceptanceError("maintenance window ended before drill")
    deadline_args = (
        ["--maintenance-window-end", maintenance_window_end.isoformat()]
        if maintenance_window_end is not None
        else []
    )
    result = site.pod_json(
        "cpu",
        target,
        pod or control_worker_pod(site, target),
        DRILL_PROBE,
        "--kind",
        kind,
        "--drill-id",
        drill_id,
        "--cluster-id",
        target.cluster_id,
        "--duplicate-delay-seconds",
        str(duplicate_delay_seconds),
        *deadline_args,
        timeout=180,
    )
    if (
        result.get("drill_id") != drill_id
        or parse_time(result.get("executed_at")) is None
        or parse_time(result.get("completed_at")) is None
    ):
        raise NotificationAcceptanceError(
            "drill identity or observation interval is missing"
        )
    return result


LIVE_NOTIFICATION_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext

# The current store state of notifications another case recorded. Evidence
# files say what a run saw at the time; the verdict must rest on what the
# store holds now, and the per-incident count is the deduplication proof for a
# real record (a drill proves it by re-sending, a live record by there being
# exactly one for its incident and kind).
kind_markers = json.loads(sys.argv[1])
notification_ids = sys.argv[2:]
store = ApplicationContext.from_environment().store
by_incident = {}
for item in store.list_notifications():
    if item.category != "ACTION_COMPLETED":
        continue
    by_incident.setdefault(item.incident_id, []).append(item)
records = []
for notification_id in notification_ids:
    try:
        notification = store.get_notification(notification_id)
    except Exception as exc:
        records.append({"notification_id": notification_id, "missing": True,
                        "error": type(exc).__name__})
        continue
    result = store.get_notification_result(notification_id)
    kind = next(
        (name for name, marker in kind_markers.items()
         if marker in notification.deduplication_key),
        None,
    )
    same_kind = [
        item for item in by_incident.get(notification.incident_id, [])
        if kind is not None and kind_markers[kind] in item.deduplication_key
    ]
    records.append({
        "notification_id": notification_id,
        "missing": False,
        "kind": kind,
        "incident_id": notification.incident_id,
        "cluster_name": notification.cluster_name,
        "category": notification.category,
        "deduplication_key": notification.deduplication_key,
        "created_at": notification.created_at.isoformat(),
        "drill_id": notification.drill_id,
        "status": result.status.value if result else None,
        "provider_message_id_present": bool(
            result is not None and result.provider_message_id
        ),
        "same_incident_kind_count": len(same_kind),
    })
print(json.dumps({"records": records}, sort_keys=True))
"""


def live_action_completed_records(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    run_dir: Path,
    pod: str,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the run's recorded ACTION_COMPLETED notifications against the store."""

    candidates = action_completed_records_from_evidence(
        run_dir,
        cluster_id=target.cluster_id,
        release_id=release_id,
    )
    notification_ids = sorted(
        {item["notification_id"] for items in candidates.values() for item in items}
    )
    if not notification_ids:
        return {"candidates": candidates, "records": []}
    live = site.pod_json(
        "cpu",
        target,
        pod,
        LIVE_NOTIFICATION_PROBE,
        json.dumps(NOTIFICATION_KINDS, sort_keys=True),
        *notification_ids,
        timeout=180,
    )
    records = live.get("records")
    if (
        not isinstance(records, list)
        or sorted(str(item.get("notification_id") or "") for item in records)
        != notification_ids
    ):
        raise NotificationAcceptanceError("live notification reads are incomplete")
    return {"candidates": candidates, "records": records}


def run_notify001(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    attempt: int,
    run_dir: Path,
    receipt_evidence: Path | None,
    ses_window_evidence: Path | None,
    maintenance_window_end: datetime | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    pod = control_worker_pod(site, target)
    digest = drill_source_digest() if release_id is not None else None
    plan_path = run_dir / "cases" / "GF-REGIONAL-NOTIFY-001" / "plan.json"
    plan_digest = drill_plan_binding(plan_path)

    def observed_drill(kind: str, stage: str, *, delay: int = 0) -> dict[str, Any]:
        def capture() -> dict[str, Any]:
            return drill(
                site,
                target,
                kind=kind,
                drill_id=f"notify001-{stage}-{attempt}-{int(time.time())}",
                pod=pod,
                maintenance_window_end=maintenance_window_end,
                duplicate_delay_seconds=delay,
            )

        if release_id is None:
            return capture()
        result = cached_drill(
            run_dir
            / "cases"
            / "GF-REGIONAL-NOTIFY-001"
            / f"drill-{attempt}-{stage}.json",
            {
                "drill_source_sha256": digest,
                "release_id": release_id,
                "plan_details_sha256": plan_digest,
                "cluster_id": target.cluster_id,
                "attempt": attempt,
                "stage": stage,
                "kind": kind,
                "duplicate_delay_seconds": delay,
            },
            capture,
        )
        if result.get("kind") != kind or not str(result.get("drill_id", "")).startswith(
            f"notify001-{stage}-{attempt}-"
        ):
            raise NotificationAcceptanceError(
                "cached notification drill identity changed"
            )
        return result

    live = live_action_completed_records(
        site, target, run_dir=run_dir, pod=pod, release_id=release_id
    )
    evidence: dict[str, dict[str, Any]] = {}
    drills: list[dict[str, Any]] = []
    record_times: list[datetime] = []
    for kind in NOTIFICATION_KINDS:
        candidate_ids = {
            item["notification_id"] for item in live["candidates"].get(kind, [])
        }
        unusable = [
            item
            for item in live["records"]
            if (
                item.get("kind") == kind or item.get("notification_id") in candidate_ids
            )
            and (
                item.get("missing")
                or item.get("status") != "SENT"
                or not item.get("provider_message_id_present")
            )
        ]
        if unusable:
            raise NotificationAcceptanceError(
                f"{kind} has unverified live delivery; a drill cannot replace it"
            )
        record = select_live_record(
            live["records"], kind=kind, cluster_id=target.cluster_id
        )
        if record is not None:
            evidence[kind] = {
                "source": "live",
                "sent": True,
                "deduplicated": record["same_incident_kind_count"] == 1,
                "record": record,
            }
            created_at = parse_time(record.get("created_at"))
            if created_at is None:
                raise NotificationAcceptanceError(
                    "live notification has no creation time"
                )
            record_times.append(created_at)
            continue
        # Drill fallback, as the catalog allows: the run produced no real
        # completion of this kind, so the mail path is proven with a labeled
        # drill rather than by re-executing the action.
        result = observed_drill(kind, kind)
        drills.append(result)
        executed_at = parse_time(result.get("executed_at"))
        completed_at = parse_time(result.get("completed_at"))
        if executed_at is None or completed_at is None or completed_at < executed_at:
            raise NotificationAcceptanceError("drill has no valid observation interval")
        record_times.extend((executed_at, completed_at))
        evidence[kind] = {
            "source": "drill",
            "sent": result["statuses"][0] == "SENT"
            and result["provider_message_id_present"],
            "deduplicated": (
                result["provider_message_id_stable"] is True
                and result.get("notification_count") == 1
                and result.get("notifier_calls") == 1
                and result.get("completion_calls") == 4
            ),
            "drill": result,
        }
    # Last email in this attempt. Leave a whole metric period between its
    # initial send and duplicates so a minute bucket can exclude all sends.
    dedup_drill = observed_drill("gpu-reset", "dedup", delay=65)
    drills.append(dedup_drill)
    executed_at = parse_time(dedup_drill.get("executed_at"))
    completed_at = parse_time(dedup_drill.get("completed_at"))
    if executed_at is None or completed_at is None or completed_at < executed_at:
        raise NotificationAcceptanceError("drill has no valid observation interval")
    record_times.extend((executed_at, completed_at))
    receipt = validate_external_evidence(
        receipt_evidence, "receipt", record_times=record_times
    )
    ses_window = validate_duplicate_evidence(ses_window_evidence, dedup_drill)
    checks = completion_checks(evidence, dedup_drill, receipt, ses_window)
    sources = {kind: value["source"] for kind, value in evidence.items()}
    limitations = [
        "Drill emails are explicitly labeled and are built without executing "
        "RESET_GPU or RESTART_WORKLOAD; the deduplication drill always sends "
        "one final labeled mail, waits 65 seconds and replays the same remote result "
        "three more times. Fallback delivery drills may send two earlier emails.",
        "Completed drill observations are re-judged, not re-sent, when receipts "
        "arrive later (same release/cluster/attempt/plan details/notification-runner "
        "source; operator evidence may change). An unconfirmed drill is never resent.",
        "GF-REGIONAL-NOTIFY-002 is superseded by this case: "
        "four_submissions_return_one_provider_id and the SES window evidence "
        "cover the actual result handler using an isolated Store. HTTP authorization "
        "and provider-acceptance crash ambiguity are separate boundaries.",
    ]
    if any(source == "live" for source in sources.values()):
        limitations.append(
            "Live records are this run's real ACTION_COMPLETED notifications; "
            "their deduplication proof is one notification per incident and kind."
        )
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sources": sources,
        "evidence": evidence,
        "live_candidates": live["candidates"],
        "live_records": live["records"],
        "drills": [
            {
                "kind": item.get("kind"),
                "drill_id": item.get("drill_id"),
                "executed_at": item.get("executed_at"),
                "notification_id": item.get("notification_id"),
            }
            for item in drills
        ],
        "dedup_drill": dedup_drill,
        "record_times": [item.isoformat() for item in record_times],
        "receipt_evidence": receipt,
        "ses_window_evidence": ses_window,
        "limitations": limitations,
    }


def run_notify002(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    attempt: int,
    ses_window_evidence: Path | None,
) -> dict[str, Any]:
    print(
        f"GF-REGIONAL-NOTIFY-002 is superseded by {NOTIFY002_SUPERSEDED_BY}; "
        "running it only reproduces the drill that NOTIFY-001 already records.",
        file=sys.stderr,
        flush=True,
    )
    result = drill(
        site,
        target,
        kind="gpu-reset",
        drill_id=f"notify002-{attempt}-{int(time.time())}",
    )
    external = validate_duplicate_evidence(ses_window_evidence, result)
    checks = {
        "four_submissions_return_one_provider_id": (
            len(result["statuses"]) == 4
            and result["provider_message_id_present"]
            and result["provider_message_id_stable"]
        ),
        "ses_send_count_did_not_increase_for_duplicates": external["valid"],
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "superseded_by": NOTIFY002_SUPERSEDED_BY,
        "checks": checks,
        "drill": result,
        "ses_window_evidence": external,
        "limitations": [
            f"This case is superseded by {NOTIFY002_SUPERSEDED_BY}, which records "
            "the same four-submission deduplication result; it remains runnable "
            "for reproduction only.",
            "The first call sends one labeled drill email; the next three calls "
            "reuse the same notification ID and provider message ID.",
        ],
    }


NOTIFICATION_ENV_PROBE = r"""
import json
import os

print(json.dumps({
    "service_role": os.environ.get("GPU_FAULT_SERVICE_ROLE"),
    "dispatcher_enabled": os.environ.get(
        "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED"
    ),
    "async_delivery": os.environ.get("GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY"),
}, sort_keys=True))
"""

NOTIFICATION_CONFIG_KEYS = ("service_role", "dispatcher_enabled", "async_delivery")

NOTIFICATION_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)


NOTIFICATION_ENV_NAMES = {
    "service_role": "GPU_FAULT_SERVICE_ROLE",
    "dispatcher_enabled": "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED",
    "async_delivery": "GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY",
}


def deployment_environment(site: IdentitySite, app: str) -> tuple[dict[str, str], int]:
    """Resolve a Deployment's env the way the kubelet does, ``envFrom`` included.

    Reading only ``containers[0].env`` reported ``None`` for all three settings on
    every control-plane Deployment, because the release renders them into
    ``envFrom`` ConfigMaps. That left ``only_worker_has_worker_role``
    unsatisfiable -- NOTIFY-003 could not pass on a correctly configured cluster,
    which is how it failed on 2026-09-04. Inline ``env`` is applied last because
    Kubernetes lets it override ``envFrom``.
    """

    deployment = json.loads(site.cpu("get", "deployment", app, "-o", "json"))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    environment: dict[str, str] = {}
    for source in container.get("envFrom", []):
        name = (source.get("configMapRef") or {}).get("name")
        if not name:
            continue
        value = json.loads(site.cpu("get", "configmap", name, "-o", "json"))
        environment.update(
            {str(key): str(item) for key, item in (value.get("data") or {}).items()}
        )
    for item in container.get("env", []):
        if "value" in item:
            environment[str(item["name"])] = str(item["value"])
    return environment, int(deployment["spec"].get("replicas") or 0)


def deployment_notification_config(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    """The three dispatcher settings per control-plane Deployment.

    The Deployment is the only source that answers for a Deployment scaled to
    zero -- ``gpu-fault-telemetry-spool-worker`` runs 0 replicas on this site, and
    "it has no Pod" is not evidence that its service role would refrain from
    starting the dispatcher. Where replicas do run, they are exec'd as well and
    required to agree, because the template is a desired value and a half-finished
    rollout is exactly the disagreement the case treats as worse than a wrong one.
    """

    values: dict[str, Any] = {}
    for app in NOTIFICATION_DEPLOYMENTS:
        environment, desired_replicas = deployment_environment(site, app)
        declared = {
            key: environment.get(name) for key, name in NOTIFICATION_ENV_NAMES.items()
        }
        replicas = []
        for pod in site.ready_pods("cpu", app, target):
            observed = site.pod_json("cpu", target, pod, NOTIFICATION_ENV_PROBE)
            replicas.append(
                {
                    "pod": pod,
                    **{key: observed.get(key) for key in NOTIFICATION_CONFIG_KEYS},
                }
            )
        agree = all(
            all(replica[key] == declared[key] for key in NOTIFICATION_CONFIG_KEYS)
            for replica in replicas
        )
        values[app] = {
            "desired_replicas": desired_replicas,
            "ready_replicas": len(replicas),
            "replicas": replicas,
            "replicas_agree": agree,
            **declared,
        }
    return values


def notify003_focused_tests(
    case_dir: Path | None = None,
    *,
    reuse: bool = False,
) -> dict[str, Any]:
    """Run NOTIFY-003's focused pytest, or reuse the plan's result in --execute.

    ``reuse`` consults ``reusable_focused_tests`` on the plan this case wrote:
    a passing result taken against the same source digest speaks for the tree
    now, so the suite is not paid for twice per run.
    """

    if reuse and case_dir is not None:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [sys.executable, "-m", "pytest", "-q", *NOTIFY003_FOCUSED_TESTS]
    completed = run(command, check=False, timeout=600)
    if case_dir is not None:
        log = case_dir / "focused-tests.log"
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        log.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "focused_tests_reused": False,
    }


def run_notify003(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path | None = None,
) -> dict[str, Any]:
    tests = notify003_focused_tests(case_dir, reuse=True)
    config = deployment_notification_config(site, target)
    roles = {name: value["service_role"] for name, value in config.items()}
    checks = {
        "backlog_and_requeue_contract_tests": bool(tests["passed"]),
        "enabled_worker_and_complete_replicas": all(
            value["ready_replicas"] == value["desired_replicas"]
            and (
                value["desired_replicas"] > 0
                or name == "gpu-fault-telemetry-spool-worker"
            )
            for name, value in config.items()
        ),
        "delivery_enabled": all(
            str(value[key]).strip().lower() in {"1", "true", "yes", "on"}
            for value in config.values()
            for key in ("dispatcher_enabled", "async_delivery")
        ),
        "only_worker_has_worker_role": (
            roles["gpu-fault-control-worker"] == "worker"
            and roles["gpu-fault-api-ha"] == "ingress"
            and roles["gpu-fault-telemetry-spool-worker"] == "spool-worker"
        ),
        "dispatcher_config_consistent": len(
            {
                (
                    value["dispatcher_enabled"],
                    value["async_delivery"],
                )
                for value in config.values()
            }
        )
        == 1,
        # Replica disagreement within one Deployment is worse than a wrong value
        # (docs/区域模式端到端验收测试用例.md: "副本间不一致本身即为 FAIL"), and it is
        # only visible now that the values come from the Pods rather than the
        # shared template.
        "replicas_agree_within_each_deployment": all(
            value["replicas_agree"] for value in config.values()
        ),
    }
    route_drill = (
        site.pod_json(
            "cpu",
            target,
            control_worker_pod(site, target),
            REQUEUE_PROBE.read_text(encoding="utf-8"),
            timeout=180,
        )
        if all(checks.values())
        else {}
    )
    checks["public_requeue_route_flow"] = requeue_route_errors(route_drill) == []
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "deployment_config": config,
        "focused_tests": tests,
        "focused_test_returncode": tests["returncode"],
        "public_route_drill": route_drill,
        "public_route_errors": requeue_route_errors(route_drill),
        "limitations": [
            "The installed incident router and authorization middleware exercise "
            "send/requeue/dispatch over ASGI HTTP in an isolated in-memory Store. "
            "The notifier is local; no production backlog or SES is mutated.",
            "Live Deployments separately prove the production service-role assembly.",
        ],
    }


SES_DENIAL_PROBE = r"""
import json
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    boto3.client("sesv2", config=Config(
        connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1}
    )).send_email(
        FromEmailAddress="gpu-fault-probe@example.invalid",
        Destination={"ToAddresses": ["gpu-fault-probe@example.invalid"]},
        Content={"Simple": {
            "Subject": {"Data": "gpu-fault permission probe"},
            "Body": {"Text": {"Data": "gpu-fault permission probe"}},
        }},
    )
    print(json.dumps({"result": "ALLOWED"}))
except ClientError as exc:
    print(json.dumps({
        "result": "DENIED",
        "code": exc.response.get("Error", {}).get("Code"),
    }, sort_keys=True))
"""


def run_notify004(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    maintenance_window_end: datetime | None = None,
) -> dict[str, Any]:
    pods = site.ready_pods("gpu", "gpu-fault-cluster-executor", target)
    if not pods:
        raise NotificationAcceptanceError("no Ready Executor Pod")
    results = {}
    for pod in pods:
        if (
            maintenance_window_end is not None
            and datetime.now(timezone.utc) >= maintenance_window_end
        ):
            raise NotificationAcceptanceError(
                "maintenance window ended before SES denial probe"
            )
        results[pod] = site.pod_json("gpu", target, pod, SES_DENIAL_PROBE, timeout=30)
    checks = {
        "executor_ses_denied": all(
            result.get("result") == "DENIED" for result in results.values()
        ),
        "denial_is_access_control": all(
            str(result.get("code") or "").lower()
            in {"accessdenied", "accessdeniedexception", "unauthorizedoperation"}
            for result in results.values()
        ),
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "provider_results": results,
        "limitations": [
            "The request uses invalid recipient identities and must be rejected "
            "by IAM before SES message validation."
        ],
    }


NOTIFICATION_QUERY_PROBE = r"""
import json
import re
import sys
from datetime import datetime
from gpu_fault.app import ApplicationContext

GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

# Selected by node, not by workload name. A low-utilization notification is
# host-resource scoped ("<cluster> <node> host_gpu_utilization_percent"), so
# filtering on the fixture's job name only finds the ones whose body happens to
# list the workload -- and it hides the notifications this case has to count.
cluster_id, needle, observed_after_text, *nodes = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = []
for notification in store.list_notifications():
    if notification.created_at < observed_after or notification.cluster_name != cluster_id:
        continue
    searchable = "\n".join(
        (
            notification.deduplication_key,
            notification.subject,
            notification.body_text,
        )
    )
    if "LOW_GPU_UTILIZATION" not in searchable or not any(node in searchable for node in nodes):
        continue
    incident = store.get_incident(notification.incident_id)
    if incident.cluster_id != cluster_id:
        raise RuntimeError("notification incident cluster identity differs")
    matched = sorted(set(nodes).intersection(incident.node_ids))
    if not matched:
        continue
    result = store.get_notification_result(notification.notification_id)
    records.append({
        "notification_id": notification.notification_id,
        "created_at": notification.created_at.isoformat(),
        "category": notification.category,
        "low_gpu_utilization": "LOW_GPU_UTILIZATION" in searchable,
        "matched_nodes": matched,
        # Every GPU the message speaks for. The sustained low-utilization signal
        # is tracked per device, so a node with 8 GPUs raises 8 findings; the
        # count here is what proves they were aggregated into one message
        # instead of fanned out one message per GPU.
        "gpu_devices": sorted(set(GPU_UUID.findall(searchable))),
        "names_workload": needle in searchable,
        "deduplication_key": notification.deduplication_key,
        "status": result.status.value if result else None,
        "provider_message_id_present": bool(
            result is not None and result.provider_message_id
        ),
    })
print(json.dumps({"records": records}, sort_keys=True))
"""


ATTEMPT_OBSERVATION_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext

# Only read when a phase times out. The sustained low-utilization rule is gated
# on the batch reporting an ACTIVE workload, so a node whose containers the
# control plane never observed cannot raise the signal no matter how long the
# case waits -- and this is the difference between a detector defect and a
# workload that never landed where the case thinks it did.
cluster_id, *nodes = sys.argv[1:]
store = ApplicationContext.from_environment().store
records = []
for observation in store.list_attempt_observations(cluster_id):
    containers = [
        {
            "node_id": container.node_id,
            "pod_name": container.pod_name,
            "gpu_count": container.gpu_count,
            "terminated": container.terminated,
        }
        for container in observation.containers
        if container.node_id in nodes
    ]
    if not containers:
        continue
    records.append({
        "job_id": observation.job_id,
        "attempt_id": observation.attempt_id,
        "workload_phase": observation.workload_phase.value,
        "observed_at": observation.observed_at.isoformat(),
        "workload_ids": list(observation.workload_ids),
        "containers": containers,
    })
print(json.dumps({"records": records}, sort_keys=True))
"""


LATCH_STATE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext

# The disarm state of the sustained low-utilization signal on the selected
# nodes. This is what the inter-phase quiet gap is actually waiting for, so the
# runner reads it instead of assuming a sleep was long enough: the latch is
# per GPU device, it clears only on a batch that reports no active workload, and
# an idle node delivers those batches one health summary apart.
cluster_id, *nodes = sys.argv[1:]
store = ApplicationContext.from_environment().store
records = []
for item in store._list("health_signal_state"):
    document = item if isinstance(item, dict) else item.model_dump(mode="json")
    key = str(document.get("signal_key", ""))
    parts = key.split("/")
    if len(parts) < 5:
        continue
    if parts[0] != cluster_id or parts[1] not in nodes:
        continue
    if parts[2] != "host_gpu_utilization_percent":
        continue
    records.append({
        "node_id": parts[1],
        "device": parts[3],
        "rule_id": parts[4],
        "active": bool(document.get("active")),
        "notified": bool(document.get("notified")),
        "active_since": document.get("active_since"),
    })
print(json.dumps({"records": sorted(
    records, key=lambda value: (value["node_id"], value["device"])
)}, sort_keys=True))
"""


LOW_UTILIZATION_CONFIG_PROBE = r"""
import json
import os
print(json.dumps({
    "duration_seconds": int(
        os.getenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "300")
    ),
    "threshold_percent": float(
        os.getenv("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "10")
    ),
}, sort_keys=True))
"""


def manifest_alias_errors(document: dict[str, Any]) -> list[str]:
    """Why ``document`` would not survive a per-role metadata injection.

    A PyTorchJob whose Master and Worker share one template object serializes
    with a YAML alias and is mutated twice by the submit tool; the dump is
    checked as well as the object identity because the alias is what the
    submit tool actually reads back.
    """

    errors = []
    rendered = yaml.safe_dump(document, sort_keys=False)
    if YAML_ALIAS_PATTERN.search(rendered):
        errors.append("manifest serializes with a YAML alias")
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    templates = [
        replica.get("template")
        for replica in replicas.values()
        if isinstance(replica, dict)
    ]
    if len({id(item) for item in templates}) != len(templates):
        errors.append("replica specs share one template object")
    return errors


def low_utilization_metadata_errors(
    workload: dict[str, Any],
    *,
    expected_pods: int,
) -> list[str]:
    """Role and rank-offset the submit tool injected, one per replica spec.

    The alias defect surfaced here: Master carried Worker's role and offset.
    """

    if str(workload.get("kind") or "") != "PyTorchJob":
        return []
    errors = []
    replicas = (workload.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    expected = {"Master": ("master", "0"), "Worker": ("worker", "1")}
    seen_offsets = []
    for role, (label, offset) in expected.items():
        metadata = ((replicas.get(role) or {}).get("template") or {}).get(
            "metadata"
        ) or {}
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        if labels.get("gpu-fault.io/role") != label:
            errors.append(f"{role} role label is {labels.get('gpu-fault.io/role')!r}")
        actual_offset = annotations.get("gpu-fault.io/rank-offset")
        if actual_offset != offset:
            errors.append(f"{role} rank offset is {actual_offset!r}")
        seen_offsets.append(actual_offset)
        if annotations.get("gpu-fault.io/expected-critical-ranks") != str(
            expected_pods
        ):
            errors.append(f"{role} expected critical ranks differs")
    if len(set(seen_offsets)) != len(seen_offsets):
        errors.append("replica specs share one rank offset")
    return errors


def wait_running_pods(
    fixture: ManagedWorkloadFixture,
    expected: int,
    *,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = []
    while time.monotonic() < deadline:
        last = fixture.pods()
        if (
            len(last) == expected
            and all(item["phase"] == "Running" and item["ready"] for item in last)
            and len({item["node"] for item in last}) == expected
        ):
            return {"pods": last}
        time.sleep(5)
    raise RegionalFixtureError(f"low-utilization workload did not run: {last}")


def foreign_gpu_reservations(
    regional: Any,
    nodes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Running Pods that already hold a GPU on one of the selected nodes.

    The sustained low-utilization signal latches: it fires once per activation
    episode and stays quiet until the node stops reporting an active workload.
    So an unrelated Pod that reserves a GPU and then idles keeps the node's
    signal permanently notified, and no fixture this case submits can ever
    produce a notification there. That is not a detector defect and it is not
    something a longer wait fixes, so it has to be refused up front instead of
    surfacing as an unattributable FAIL 15 minutes later.
    """

    value = json.loads(
        regional.kubectl("gpu", "get", "pod", "-o", "json", all_namespaces=True)
    )
    holders = []
    for item in value.get("items", []):
        spec = item.get("spec", {})
        if item.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        if spec.get("nodeName") not in nodes:
            continue

        def gpu_request(container: dict[str, Any]) -> int:
            resources = container.get("resources") or {}
            return int(
                (resources.get("requests") or {}).get(
                    "nvidia.com/gpu",
                    (resources.get("limits") or {}).get("nvidia.com/gpu", 0),
                )
            )

        reserved = max(
            sum(gpu_request(container) for container in spec.get("containers", [])),
            max(
                (
                    gpu_request(container)
                    for container in spec.get("initContainers", [])
                ),
                default=0,
            ),
        )
        if not reserved:
            continue
        holders.append(
            {
                "namespace": item["metadata"]["namespace"],
                "name": item["metadata"]["name"],
                "node": spec.get("nodeName"),
                "reserved_gpus": reserved,
            }
        )
    return sorted(holders, key=lambda item: (str(item["node"]), str(item["name"])))


def wait_low_utilization_latch_disarmed(
    regional: Any,
    *,
    cluster_id: str,
    nodes: tuple[str, ...],
    deadline_seconds: int,
    poll_seconds: int = 30,
) -> list[dict[str, Any]]:
    """Wait until no selected node still holds a notified low-utilization latch.

    A fixed quiet gap between the phases is the same coin flip the fixed sleep
    before the notification query was: the latch clears on the first batch that
    reports no active workload, an idle node only sends those on its health
    summary, and one live run cleared with 25 seconds to spare. So poll the
    state the gap exists to reach, and let a phase start as soon as its nodes
    are actually re-armed rather than when a timer says they should be.
    """

    deadline = time.monotonic() + deadline_seconds
    polls: list[dict[str, Any]] = []
    while True:
        records = regional.cpu_python(LATCH_STATE_PROBE, cluster_id, *nodes)["records"]
        latched = sorted(
            {item["node_id"] for item in records if item["notified"] or item["active"]}
        )
        polls.append({"observed_at": utc_now(), "latched_nodes": latched})
        if not latched:
            return polls
        if time.monotonic() >= deadline:
            raise NotificationAcceptanceError(
                "selected nodes still hold a notified low-utilization latch after "
                f"{deadline_seconds} seconds, so this phase could not raise a new "
                f"notification for them: {latched}"
            )
        time.sleep(poll_seconds)


def wait_low_utilization_notifications(
    regional: Any,
    *,
    cluster_id: str,
    needle: str,
    nodes: tuple[str, ...],
    observed_after: datetime,
    deadline_seconds: int,
    poll_seconds: int = 30,
    settle_seconds: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Poll until every selected node has reported, or the deadline passes.

    The previous fixed ``sleep(duration + 90)`` raced the detector and lost: the
    signal only becomes active once a delivered batch reports both an ACTIVE
    workload and near-zero GPU utilization, and it emits on the first batch at
    least ``duration`` seconds later. On 2026-09-04 the same fixture produced a
    notification 356 seconds after submission in one attempt and nothing at all
    within 409 seconds in the next, so the budget decided the verdict rather
    than the behaviour. Polling to a generous deadline removes that coin flip
    and records when each node actually reported.
    """

    deadline = time.monotonic() + deadline_seconds
    polls: list[dict[str, Any]] = []
    covered_since: float | None = None
    while True:
        query = regional.cpu_python(
            NOTIFICATION_QUERY_PROBE,
            cluster_id,
            needle,
            observed_after.isoformat(),
            *nodes,
        )
        records = [item for item in query["records"] if item["low_gpu_utilization"]]
        reported = {node for item in records for node in item["matched_nodes"]}
        polls.append(
            {
                "observed_at": utc_now(),
                "notification_count": len(records),
                "reported_nodes": sorted(reported),
            }
        )
        complete = set(nodes) == reported and all(
            item.get("status") == "SENT"
            and item.get("provider_message_id_present") is True
            for item in records
        )
        if complete:
            covered_since = time.monotonic() if covered_since is None else covered_since
        else:
            covered_since = None
        if (
            len(records) > len(nodes)
            or time.monotonic() >= deadline
            or (
                covered_since is not None
                and time.monotonic() - covered_since >= settle_seconds
            )
        ):
            return records, polls
        time.sleep(poll_seconds)


def delete_notification_workload(
    regional: Any, fixture: ManagedWorkloadFixture
) -> None:
    fixture.delete()
    deadline = time.monotonic() + 300
    while fixture.pods() and time.monotonic() < deadline:
        time.sleep(5)
    if (
        fixture.pods()
        or regional.kubectl(
            "gpu",
            "get",
            fixture.resource,
            fixture.name,
            "--ignore-not-found",
            "-o",
            "name",
        ).strip()
    ):
        raise NotificationAcceptanceError("test workload resources remain")


def cleanup_notification_workloads(
    regional: Any, fixtures: Sequence[ManagedWorkloadFixture]
) -> list[str]:
    errors = []
    for fixture in fixtures:
        try:
            delete_notification_workload(regional, fixture)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def run_notify005(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str, str],
    case_dir: Path,
    attempt: int,
    training_image: str,
    maintenance_window_end: datetime | None = None,
    planned_nodes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regional = site.regional(target)
    baseline_nodes = {node: regional.node_snapshot(node) for node in nodes}
    for node, value in baseline_nodes.items():
        errors = node_errors(value)
        if planned_nodes is not None and value.get("uid") != (
            planned_nodes.get(node) or {}
        ).get("uid"):
            errors.append("node UID differs from the approved plan")
        if errors:
            raise NotificationAcceptanceError(f"{node}: " + "; ".join(errors))
    worker_pods = site.ready_pods("cpu", "gpu-fault-control-worker", target)
    if not worker_pods:
        raise NotificationAcceptanceError("no Ready control-worker Pod")
    config = site.pod_json(
        "cpu",
        target,
        worker_pods[0],
        LOW_UTILIZATION_CONFIG_PROBE,
    )
    duration_seconds = int(config["duration_seconds"])
    # Three times the sustained window plus five minutes, because the wait is
    # two stages in series rather than one window. The signal needs a delivered
    # batch that reports an active workload with near-zero GPU use before the
    # window starts counting, and the host edge filter subtracts an already
    # active edge reason from its delivery reasons, so once the low-utilization
    # edge has been delivered the next batch only arrives on the periodic health
    # summary. The window therefore expires unobserved and is judged one summary
    # cycle later: interval + duration + summary, on top of however long the
    # workload takes to register as active. Live worst case was 748 seconds
    # against a 900-second budget, which is too little margin to keep.
    phase_deadline_seconds = duration_seconds * 3 + 300
    # The two phases share nodes, and the latch only clears once the node
    # reports no active workload. Deleting the fixture is not enough: an idle
    # node stops delivering on every collection interval and falls back to its
    # health summary, so the clearing batch can be several minutes out. Without
    # that quiet gap the second phase inherits phase one's notified latch and
    # the node stays silent for a reason that has nothing to do with the case.
    # The gap is polled rather than slept: a fixed `duration + 90` once cleared
    # with 25 seconds to spare, which is a flake waiting to be recorded as a
    # FAIL against the detector.
    quiesce_deadline_seconds = duration_seconds * 3 + 300
    # Record what these nodes were already reported for, over the whole history
    # rather than a recent window: the latch has no expiry, so an episode from
    # hours earlier is exactly the one that would suppress this run.
    baseline = regional.cpu_python(
        NOTIFICATION_QUERY_PROBE,
        target.cluster_id,
        "",
        datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat(),
        *nodes,
    )
    pre_existing = [item for item in baseline["records"] if item["low_gpu_utilization"]]
    fixtures = []
    observations = []
    result: dict[str, Any] = {"verdict": "FAIL"}
    try:
        for count, selected in ((1, nodes[:1]), (3, nodes)):
            if (
                maintenance_window_end is not None
                and datetime.now(timezone.utc) >= maintenance_window_end
            ):
                raise NotificationAcceptanceError(
                    "maintenance window ended before workload creation"
                )
            for node in selected:
                current = regional.node_snapshot(node)
                if (
                    node_errors(current)
                    or current["uid"] != baseline_nodes[node]["uid"]
                ):
                    raise NotificationAcceptanceError(
                        "selected node identity or readiness changed"
                    )
            # Foreign GPU holders first, because that latch never clears and
            # waiting on it would spend the whole gap to report the wrong cause.
            holders = foreign_gpu_reservations(regional, selected)
            if holders:
                raise NotificationAcceptanceError(
                    "selected nodes already hold GPUs for unrelated workloads, "
                    "whose idle reservation latches the low-utilization signal: "
                    f"{holders}"
                )
            quiesce_polls = wait_low_utilization_latch_disarmed(
                regional,
                cluster_id=target.cluster_id,
                nodes=selected,
                deadline_seconds=quiesce_deadline_seconds,
            )
            phase_started_at = datetime.now(timezone.utc)
            name = f"notify005-{count}n-{attempt}-{int(time.time())}"
            source = case_dir / f"{name}.source.yaml"
            document = low_utilization_manifest(
                name=name,
                nodes=selected,
                image=training_image,
            )
            alias_errors = manifest_alias_errors(document)
            if alias_errors:
                raise NotificationAcceptanceError(
                    "low-utilization manifest is not injectable per role: "
                    + "; ".join(alias_errors)
                )
            source.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            source.chmod(0o600)
            fixture = ManagedWorkloadFixture(
                regional,
                ManagedWorkloadSettings(
                    manifest=source,
                    site_file=site.site_file,
                    job_id=name,
                    attempt_id=f"{name}-a001",
                    restart_budget=0,
                    expected_pods=count,
                    expected_gpu_count=count,
                ),
                state_path=case_dir / f"{name}.ownership.json",
            )
            fixtures.append(fixture)
            fixture.submit()
            metadata_errors = low_utilization_metadata_errors(
                fixture.workload(), expected_pods=count
            )
            if metadata_errors:
                raise NotificationAcceptanceError("; ".join(metadata_errors))
            running = wait_running_pods(fixture, count)
            records, polls = wait_low_utilization_notifications(
                regional,
                cluster_id=target.cluster_id,
                needle=name,
                nodes=selected,
                observed_after=phase_started_at,
                deadline_seconds=phase_deadline_seconds,
            )
            observation: dict[str, Any] = {
                "node_count": count,
                "nodes": list(selected),
                "pods": running["pods"],
                "metadata_errors": metadata_errors,
                "quiesce_polls": quiesce_polls,
                "polls": polls,
                "notifications": records,
            }
            reported = {node for item in records for node in item["matched_nodes"]}
            if set(selected) - reported:
                observation["silent_nodes"] = sorted(set(selected) - reported)
                observation["attempt_observations"] = regional.cpu_python(
                    ATTEMPT_OBSERVATION_PROBE,
                    target.cluster_id,
                    *selected,
                )["records"]
            observations.append(observation)
            observation["checks"] = phase_checks(observation)
            if not all(observation["checks"].values()):
                raise NotificationAcceptanceError(
                    f"{count}-node aggregation phase failed"
                )
            delete_notification_workload(regional, fixture)
        checks = {
            f"{value['node_count']}n/{key}": passed
            for value in observations
            for key, passed in value["checks"].items()
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "policy_config": config,
            "phase_deadline_seconds": phase_deadline_seconds,
            "quiesce_deadline_seconds": quiesce_deadline_seconds,
            "pre_existing_notifications": pre_existing,
            "observations": observations,
        }
    except Exception as exc:
        result.update(
            {"error": f"{type(exc).__name__}: {exc}", "observations": observations}
        )
    finally:
        cleanup_errors = cleanup_notification_workloads(regional, fixtures)
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["verdict"] = "FAIL"
    result["limitations"] = [
        "The workloads intentionally reserve one GPU while performing CPU work; "
        "they validate notification aggregation without damaging hardware.",
        "The sustained low-utilization signal latches per GPU and re-arms only "
        "after the node reports no active workload, so the case refuses nodes "
        "that already hold a GPU for an unrelated workload and waits for each "
        "phase's nodes to be observably re-armed before submitting; "
        "quiesce_polls records that wait and pre_existing_notifications records "
        "the episodes that preceded the run.",
    ]
    return result


def case_plan(
    case_id: str,
    *,
    target: ClusterTarget,
    nodes: tuple[str, ...],
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    mutations = {
        "GF-REGIONAL-NOTIFY-001": (
            "reuse this run's real ACTION_COMPLETED notifications where the store "
            "still holds them SENT; otherwise send one GPU-reset drill and one "
            "workload-restart drill through SES; always repeat one GPU-reset "
            "notification four times for the deduplication proof"
        ),
        "GF-REGIONAL-NOTIFY-002": (
            f"superseded by {NOTIFY002_SUPERSEDED_BY}: send one GPU-reset drill "
            "and repeat the same notification ID three times"
        ),
        "GF-REGIONAL-NOTIFY-003": (
            "run isolated backlog/watermark contracts and read live role configuration"
        ),
        "GF-REGIONAL-NOTIFY-004": (
            "attempt one SES SendEmail call from the executor and require AccessDenied"
        ),
        "GF-REGIONAL-NOTIFY-005": (
            "run one-node and three-node low-GPU-utilization workloads, then delete them"
        ),
    }
    return {
        "risk": "case-defined",
        "predecessor": predecessor,
        "cluster_id": target.cluster_id,
        "context": target.context,
        "nodes": list(nodes),
        "mutation": mutations[case_id],
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "email delivery or dispatcher configuration is disabled unexpectedly",
            "a drill loses its DRILL subject/body marker",
            "a duplicate call changes the provider message ID",
            "a low-utilization workload or notification does not converge",
            "any workload remains after cleanup",
        ],
        "rollback": {
            "drills_use_in_memory_store_and_create_no production DB rows": True,
            "low_utilization_workloads_are_deleted_in_finally": True,
            "no_GPU_reset_or_workload_restart_is_executed_for_email_cases": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded regional NOTIFY-001..005 acceptance case."
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--site", type=Path, required=True)
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", action="append", default=[])
    value.add_argument("--receipt-evidence", type=Path)
    value.add_argument("--ses-window-evidence", type=Path)
    value.add_argument("--training-image", default=TRAINING_IMAGE)
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    if arguments.case == "GF-REGIONAL-NOTIFY-002":
        raise NotificationAcceptanceError(
            "NOTIFY-002 is superseded by NOTIFY-001 and cannot execute"
        )
    os.umask(0o077)
    install_abort_signals()
    site = IdentitySite(arguments.site)
    target = site.target(arguments.cluster_id)
    nodes = tuple(arguments.node)
    if arguments.case == "GF-REGIONAL-NOTIFY-005":
        if len(nodes) != 3 or len(set(nodes)) != 3:
            raise NotificationAcceptanceError(
                "NOTIFY-005 requires exactly three distinct --node values"
            )
        if "@sha256:" not in arguments.training_image:
            raise NotificationAcceptanceError(
                "NOTIFY-005 training image must use an immutable digest"
            )
    # The release and cluster this evidence is bound to: the predecessor must
    # have earned its PASS against the same pair, and the result carries it so
    # the next case can demand the same.
    identity = site.regional(target).evidence_identity()
    if not identity["release_id"].strip():
        raise NotificationAcceptanceError("the deployed release identity is missing")
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id, **identity)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    confirmation = (
        arguments.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    )
    environment = {
        "GPU_FAULT_NOTIFICATION_CASE": arguments.case,
        "GPU_FAULT_SITE_FILE": str(arguments.site.resolve()),
        "GPU_FAULT_CONTROL_KUBECONFIG": str(site.cpu_kubeconfig),
        "KUBECONFIG": str(site.gpu_kubeconfig),
        "SITE_INPUTS_SHA256": hashlib.sha256(arguments.site.read_bytes()).hexdigest(),
        "DEPLOYED_RELEASE_ID": identity["release_id"],
        "GPU_FAULT_CLUSTER_ID": target.cluster_id,
        "GPU_FAULT_TARGET_NODES": ",".join(nodes),
    }
    case_dir = arguments.run_dir / "cases" / arguments.case
    if not arguments.execute:
        details = case_plan(
            arguments.case,
            target=target,
            nodes=nodes,
            predecessor=predecessor,
        )
        if arguments.case == "GF-REGIONAL-NOTIFY-003":
            # Run the focused suite once here; --execute reuses the result
            # while the source digest still matches.
            case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            record_focused_tests(details, notify003_focused_tests(case_dir))
        preflight_errors = []
        if predecessor.get("valid") is not True:
            preflight_errors.append("predecessor evidence is not valid")
        if not site.ready_pods("cpu", "gpu-fault-control-worker", target):
            preflight_errors.append("no Ready control-worker Pod")
        if arguments.case == "GF-REGIONAL-NOTIFY-004" and not site.ready_pods(
            "gpu", "gpu-fault-cluster-executor", target
        ):
            preflight_errors.append("no Ready Executor Pod")
        if arguments.case == "GF-REGIONAL-NOTIFY-003" and (
            details["focused_tests"].get("passed") is not True
        ):
            preflight_errors.append("focused tests failed")
        if arguments.case == "GF-REGIONAL-NOTIFY-005":
            details["node_baseline"] = {
                node: site.regional(target).node_snapshot(node) for node in nodes
            }
            preflight_errors.extend(
                f"{node}: {error}"
                for node, snapshot in details["node_baseline"].items()
                for error in node_errors(snapshot)
            )
        details["preflight_errors"] = preflight_errors
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            arguments=arguments,
            preflight_passed=not preflight_errors,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight_errors else 1
    if arguments.confirm != confirmation:
        raise NotificationAcceptanceError(
            f"confirmation must be exactly {confirmation}"
        )
    deadline = authorize_execution(
        arguments,
        case_id=arguments.case,
        confirmation=confirmation,
        environment=environment,
    )
    if not predecessor.get("valid", False):
        raise NotificationAcceptanceError("formal predecessor evidence is not PASS")
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "GF-REGIONAL-NOTIFY-001": lambda: run_notify001(
                site,
                target,
                attempt=arguments.attempt,
                run_dir=arguments.run_dir,
                receipt_evidence=arguments.receipt_evidence,
                ses_window_evidence=arguments.ses_window_evidence,
                maintenance_window_end=deadline,
                release_id=identity["release_id"],
            ),
            "GF-REGIONAL-NOTIFY-002": lambda: run_notify002(
                site,
                target,
                attempt=arguments.attempt,
                ses_window_evidence=arguments.ses_window_evidence,
            ),
            "GF-REGIONAL-NOTIFY-003": lambda: run_notify003(
                site, target, case_dir=case_dir
            ),
            "GF-REGIONAL-NOTIFY-004": lambda: run_notify004(
                site, target, maintenance_window_end=deadline
            ),
            "GF-REGIONAL-NOTIFY-005": lambda: run_notify005(
                site,
                target,
                nodes=cast(tuple[str, str, str], nodes),
                case_dir=case_dir,
                attempt=arguments.attempt,
                training_image=arguments.training_image,
                maintenance_window_end=deadline,
                planned_nodes=json.loads((case_dir / "plan.json").read_text())[
                    "details"
                ].get("node_baseline", {}),
            ),
        }
        outcome = handlers[arguments.case]()
    except Exception as exc:
        outcome = {
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "limitations": [
                "The case stopped at the first failed assertion; later checks "
                "were not treated as executed."
            ],
        }
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": arguments.case,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **identity,
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
