from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
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
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
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
# NOTIFY-002's duplicate-result proof is carried by NOTIFY-001 (the drill sends
# the same notification four times and requires one provider message ID), so
# the catalog marks NOTIFY-002 superseded. The runner stays callable so an old
# evidence chain can still be reproduced, but says so in its result.
NOTIFY002_SUPERSEDED_BY = "GF-REGIONAL-NOTIFY-001"
# The two completion kinds NOTIFY-001 judges, keyed by the drill's `--kind` and
# mapped to the substring the builders put in `deduplication_key`
# (`.../gpu-reset/<operation>` and `.../workload-restarted/<operation>`).
NOTIFICATION_KINDS: dict[str, str] = {
    "gpu-reset": "/gpu-reset/",
    "workload-restart": "/workload-restarted/",
}
YAML_ALIAS_PATTERN = re.compile(r"(?:^|\s)[&*]id\d+\b")
NOTIFY003_FOCUSED_TESTS = (
    "tests/notifications/test_notifications.py::"
    "test_first_dispatch_suppresses_the_pre_enable_backlog",
    "tests/notifications/test_notifications.py::"
    "test_suppressed_backlog_can_be_requeued_on_demand",
    "tests/notifications/test_notifications.py::"
    "test_notification_worker_reclaims_expired_lease_and_stops",
)


class NotificationAcceptanceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 300,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    if check and completed.returncode:
        raise NotificationAcceptanceError(
            f"command failed ({completed.returncode}): {' '.join(command[:6])}; "
            f"stderr={completed.stderr[-1000:]}"
        )
    return completed


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
) -> dict[str, Any]:
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
        timeout=180,
    )
    # `executed_at` is written by the probe; a probe from before it existed
    # gets the runner's clock so the operator's SES window can still be
    # cross-checked against something.
    result.setdefault("executed_at", utc_now())
    result.setdefault("drill_id", drill_id)
    return result


def validate_external_evidence(
    path: Path | None,
    kind: str,
    *,
    record_times: Sequence[datetime] = (),
) -> dict[str, Any]:
    """Judge the operator's out-of-solution evidence for ``kind``.

    Both kinds must say how the fact was established (``method``), what to look
    at to re-establish it (``reference``) and the SES-side window it was read
    over (``window_start``/``window_end``); the window must cover every drill or
    record time the runner is claiming for. Without the window a `received:
    true` or a `send_count_delta: 0` cannot be tied to *this* run's messages,
    and the account has no SES configuration set, so the SES-side answer is
    necessarily a windowed CloudWatch count that only means something with its
    window attached. ``receipt`` additionally needs ``received``; ``dedup``
    needs both counters at zero.
    """

    if path is None:
        return {"valid": False, "errors": [f"{kind} evidence is required"]}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read {kind} evidence: {exc}"]}
    if not isinstance(value, dict):
        return {"valid": False, "errors": ["external evidence is not an object"]}
    errors = []
    for field in ("method", "reference"):
        if not str(value.get(field) or "").strip():
            errors.append(f"{field} is required")
    window_start = parse_time(value.get("window_start"))
    window_end = parse_time(value.get("window_end"))
    if window_start is None or window_end is None:
        errors.append("window_start and window_end must be ISO-8601 timestamps")
    elif window_start > window_end:
        errors.append("window_start is after window_end")
    else:
        uncovered = [
            item.isoformat()
            for item in record_times
            if not (window_start <= item <= window_end)
        ]
        if uncovered:
            errors.append(f"window does not cover record times: {uncovered}")
    if kind == "receipt":
        if not bool(value.get("received")):
            errors.append("received is not true")
    elif kind == "dedup":
        try:
            if int(value.get("send_count_delta", -1)) != 0:
                errors.append("send_count_delta is not 0")
            if int(value.get("duplicate_inbox_count", -1)) != 0:
                errors.append("duplicate_inbox_count is not 0")
        except (TypeError, ValueError):
            errors.append("send_count_delta/duplicate_inbox_count are not integers")
    else:
        errors.append(f"unknown evidence kind {kind}")
    return {"valid": not errors, "errors": errors, **value}


def notification_kind(deduplication_key: str) -> str | None:
    for kind, marker in NOTIFICATION_KINDS.items():
        if marker in deduplication_key:
            return kind
    return None


def _notification_entries(value: Any) -> Iterator[dict[str, Any]]:
    """Every ``{"notification": {...}, "result": ...}`` pair inside ``value``.

    That is the shape the store probe writes into a case result
    (``state.notifications[]``), wherever the runner nested the state.
    """

    if isinstance(value, dict):
        notification = value.get("notification")
        if isinstance(notification, dict) and "result" in value:
            yield value
            return
        for item in value.values():
            yield from _notification_entries(item)
    elif isinstance(value, list):
        for item in value:
            yield from _notification_entries(item)


def action_completed_records_from_evidence(
    run_dir: Path,
    *,
    cluster_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """This run's real ACTION_COMPLETED notifications, from the case evidence.

    DESTR-001, DESTR-009 and E2E-001 record the notifications their workflows
    produced; the catalog says NOTIFY-001 reuses those and drills only when the
    run has none. Only records for ``cluster_id`` count -- a notification for
    another cluster proves nothing about this target's mail path.
    """

    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in NOTIFICATION_KINDS}
    cases_dir = run_dir / "cases"
    if not cases_dir.is_dir():
        return records
    for path in sorted(cases_dir.glob("GF-REGIONAL-*/GF-REGIONAL-*.json")):
        if path.stem != path.parent.name or path.stem in CASE_IDS:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in _notification_entries(document):
            notification = entry["notification"]
            if notification.get("category") != "ACTION_COMPLETED":
                continue
            if str(notification.get("cluster_name") or "") != cluster_id:
                continue
            kind = notification_kind(str(notification.get("deduplication_key") or ""))
            if kind is None or not notification.get("notification_id"):
                continue
            records[kind].append(
                {
                    "case_id": document.get("case_id"),
                    "evidence_path": str(path),
                    "notification_id": str(notification["notification_id"]),
                    "incident_id": notification.get("incident_id"),
                    "deduplication_key": notification.get("deduplication_key"),
                    "created_at": notification.get("created_at"),
                    "recorded_result": entry.get("result"),
                }
            )
    return records


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
) -> dict[str, Any]:
    """Resolve the run's recorded ACTION_COMPLETED notifications against the store."""

    candidates = action_completed_records_from_evidence(
        run_dir, cluster_id=target.cluster_id
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
    return {"candidates": candidates, "records": live.get("records") or []}


def select_live_record(
    records: Sequence[dict[str, Any]],
    *,
    kind: str,
    cluster_id: str,
) -> dict[str, Any] | None:
    """The newest real ``kind`` record for ``cluster_id`` that is SENT with an ID."""

    matching = [
        item
        for item in records
        if not item.get("missing")
        and item.get("kind") == kind
        and item.get("cluster_name") == cluster_id
        and not item.get("drill_id")
        and item.get("status") == "SENT"
        and item.get("provider_message_id_present")
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: str(item.get("created_at") or ""))


def run_notify001(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    attempt: int,
    run_dir: Path,
    receipt_evidence: Path | None,
    ses_window_evidence: Path | None,
) -> dict[str, Any]:
    pod = control_worker_pod(site, target)
    live = live_action_completed_records(site, target, run_dir=run_dir, pod=pod)
    evidence: dict[str, dict[str, Any]] = {}
    drills: list[dict[str, Any]] = []
    record_times: list[datetime] = []
    for kind in NOTIFICATION_KINDS:
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
            if created_at is not None:
                record_times.append(created_at)
            continue
        # Drill fallback, as the catalog allows: the run produced no real
        # completion of this kind, so the mail path is proven with a labeled
        # drill rather than by re-executing the action.
        result = drill(
            site,
            target,
            kind=kind,
            drill_id=f"notify001-{kind}-{attempt}-{int(time.time())}",
            pod=pod,
        )
        drills.append(result)
        executed_at = parse_time(result.get("executed_at"))
        if executed_at is not None:
            record_times.append(executed_at)
        evidence[kind] = {
            "source": "drill",
            "sent": result["statuses"][0] == "SENT"
            and result["provider_message_id_present"],
            "deduplicated": bool(result["provider_message_id_stable"]),
            "drill": result,
        }
    # NOTIFY-002's proof lives here now: four submissions of one notification
    # must return one provider message ID. A gpu-reset drill that already ran
    # as fallback is that proof; when both kinds came from live records, one
    # dedicated drill supplies it so the check is never inferred.
    dedup_drill = next(
        (item for item in drills if item.get("kind") == "gpu-reset"),
        None,
    )
    if dedup_drill is None:
        dedup_drill = drill(
            site,
            target,
            kind="gpu-reset",
            drill_id=f"notify001-dedup-{attempt}-{int(time.time())}",
            pod=pod,
        )
        drills.append(dedup_drill)
        executed_at = parse_time(dedup_drill.get("executed_at"))
        if executed_at is not None:
            record_times.append(executed_at)
    receipt = validate_external_evidence(
        receipt_evidence, "receipt", record_times=record_times
    )
    dedup_executed_at = parse_time(dedup_drill.get("executed_at"))
    ses_window = validate_external_evidence(
        ses_window_evidence,
        "dedup",
        record_times=[dedup_executed_at] if dedup_executed_at else [],
    )
    checks = {
        "gpu_reset_sent": evidence["gpu-reset"]["sent"],
        "workload_restart_sent": evidence["workload-restart"]["sent"],
        "gpu_reset_deduplicated": evidence["gpu-reset"]["deduplicated"],
        "workload_restart_deduplicated": evidence["workload-restart"]["deduplicated"],
        "four_submissions_return_one_provider_id": (
            len(dedup_drill["statuses"]) == 4
            and bool(dedup_drill["provider_message_id_present"])
            and bool(dedup_drill["provider_message_id_stable"])
        ),
        # The gate is "delivery was confirmed by something outside this
        # solution", and an SES-side delivery record satisfies that as well as
        # a human inbox check does -- provided its window covers the records.
        "receipt_confirmed_outside_the_solution": receipt["valid"],
        "ses_send_count_did_not_increase_for_duplicates": ses_window["valid"],
    }
    sources = {kind: value["source"] for kind, value in evidence.items()}
    limitations = [
        "Drill emails are explicitly labeled and are built without executing "
        "RESET_GPU or RESTART_WORKLOAD; the deduplication drill always sends "
        "one labeled mail and repeats the same notification three more times.",
        "GF-REGIONAL-NOTIFY-002 is superseded by this case: "
        "four_submissions_return_one_provider_id and the SES window evidence "
        "carry its duplicate-result proof.",
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
    executed_at = parse_time(result.get("executed_at"))
    external = validate_external_evidence(
        ses_window_evidence,
        "dedup",
        record_times=[executed_at] if executed_at else [],
    )
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
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "deployment_config": config,
        "focused_tests": tests,
        "focused_test_returncode": tests["returncode"],
        "limitations": [
            "The backlog is exercised in an isolated in-process store while live "
            "Deployments prove the production service-role assembly."
        ],
    }


SES_DENIAL_PROBE = r"""
import json
import boto3
from botocore.exceptions import ClientError

try:
    boto3.client("sesv2").send_email(
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
) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    result = site.pod_json("gpu", target, pod, SES_DENIAL_PROBE)
    checks = {
        "executor_ses_denied": result.get("result") == "DENIED",
        "denial_is_access_control": str(result.get("code") or "").lower()
        in {"accessdenied", "accessdeniedexception", "unauthorizedoperation"},
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "provider_result": result,
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
needle, observed_after_text, *nodes = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = []
for notification in store.list_notifications():
    if notification.created_at < observed_after:
        continue
    searchable = "\n".join(
        (
            notification.deduplication_key,
            notification.subject,
            notification.body_text,
        )
    )
    matched = sorted(node for node in nodes if node in searchable)
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


def low_utilization_manifest(
    *,
    name: str,
    nodes: tuple[str, ...],
    image: str,
) -> dict[str, Any]:
    command = [
        "/bin/bash",
        "-ceu",
        (
            "python - <<'PY'\n"
            "import multiprocessing\n"
            "import time\n"
            "def burn():\n"
            "    value = 1\n"
            "    while True:\n"
            "        value = (value * 1103515245 + 12345) & 0x7fffffff\n"
            "workers = [multiprocessing.Process(target=burn, daemon=True) "
            "for _ in range(8)]\n"
            "for worker in workers: worker.start()\n"
            "time.sleep(1800)\n"
            "PY"
        ),
    ]
    container = {
        "name": "pytorch",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "resources": {
            "requests": {
                "cpu": "8",
                "memory": "2Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {
                "cpu": "16",
                "memory": "4Gi",
                "nvidia.com/gpu": "1",
            },
        },
    }
    if len(nodes) == 1:
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeName": nodes[0],
                        "containers": [container],
                    },
                }
            },
        }
    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "In",
                                "values": list(nodes),
                            }
                        ]
                    }
                ]
            }
        },
        "podAntiAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {"app": name}},
                    "topologyKey": "kubernetes.io/hostname",
                }
            ]
        },
    }
    template = {
        "metadata": {"labels": {"app": name}},
        "spec": {
            "restartPolicy": "Never",
            "affinity": affinity,
            "containers": [container],
        },
    }
    # One template object per role. Sharing the dict made yaml.safe_dump emit
    # an alias, and gpu-training-submit then injected role/rank-offset into the
    # same object twice: Master ended up `role=worker, rank-offset=1` and the
    # Completion Watcher counted ranks that did not exist.
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": name},
        "spec": {
            "runPolicy": {"cleanPodPolicy": "All"},
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "Never",
                    "template": copy.deepcopy(template),
                },
            },
        },
    }


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
        if item.get("status", {}).get("phase") != "Running":
            continue
        # A Pod already being deleted still reports phase Running while its
        # containers stop; it is releasing the GPU, not holding it (live
        # 2026-09-07: phase one's own Pod tripped this check for phase two).
        if item.get("metadata", {}).get("deletionTimestamp"):
            continue
        if spec.get("nodeName") not in nodes:
            continue
        reserved = sum(
            int(
                container.get("resources", {})
                .get("requests", {})
                .get("nvidia.com/gpu", 0)
                or 0
            )
            for container in spec.get("containers", [])
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
    needle: str,
    nodes: tuple[str, ...],
    observed_after: datetime,
    deadline_seconds: int,
    poll_seconds: int = 30,
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
    while True:
        query = regional.cpu_python(
            NOTIFICATION_QUERY_PROBE,
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
        if set(nodes) <= reported or time.monotonic() >= deadline:
            return records, polls
        time.sleep(poll_seconds)


def run_notify005(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str, str],
    case_dir: Path,
    attempt: int,
    training_image: str,
) -> dict[str, Any]:
    regional = site.regional(target)
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
            )
            fixtures.append(fixture)
            fixture.submit()
            metadata_errors = low_utilization_metadata_errors(
                fixture.workload(), expected_pods=count
            )
            running = wait_running_pods(fixture, count)
            records, polls = wait_low_utilization_notifications(
                regional,
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
            fixture.delete()
            # The PyTorchJob is gone once delete returns; its Pods may still be
            # terminating on the node, and the next phase must not see them.
            deadline = time.monotonic() + 300
            while fixture.pods() and time.monotonic() < deadline:
                time.sleep(5)
        single_count = len(observations[0]["notifications"])
        three_count = len(observations[1]["notifications"])
        observed = [item for value in observations for item in value["notifications"]]
        checks = {
            # Master and Worker must each carry their own role and rank offset;
            # a shared template object once gave Master `role=worker,
            # rank-offset=1`, and the three-node phase then counted wrong.
            "replica_metadata_injected_per_role": all(
                not value["metadata_errors"] for value in observations
            ),
            # The 判定 is an upper bound -- "通知条数按节点数或任务数增长（≤3 条），
            # 不是按 GPU 数（不应是 24 条）" -- so these three are stated as bounds
            # rather than as an exact count.
            "single_node_count_at_most_one": single_count <= 1,
            "three_node_count_at_most_node_count": three_count <= len(nodes),
            "three_node_count_not_twenty_four": three_count != 24,
            # An upper bound on its own is satisfied by a detector that never
            # fires, so every node the fixture occupied has to have reported
            # within its own phase.
            "every_selected_node_reported_in_its_phase": all(
                {
                    node
                    for item in value["notifications"]
                    for node in item["matched_nodes"]
                }
                >= set(value["nodes"])
                for value in observations
            ),
            # Each message speaks for exactly one node, which is why the count
            # tracks nodes rather than GPUs.
            "every_notification_is_node_scoped": all(
                len(item["matched_nodes"]) == 1 for item in observed
            ),
            # And the aggregation itself: the sustained signal is tracked per
            # GPU, so an 8-GPU node raises 8 findings. One message naming
            # several of them is the direct evidence that they were folded
            # together instead of mailed one by one -- that is where the 24 in
            # the 判定 would have come from.
            "notifications_aggregate_multiple_gpu_devices": bool(observed)
            and all(len(item["gpu_devices"]) > 1 for item in observed),
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
    finally:
        cleanup_errors = []
        for fixture in fixtures:
            try:
                fixture.delete()
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
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
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    if arguments.confirm != confirmation:
        raise NotificationAcceptanceError(
            f"confirmation must be exactly {confirmation}"
        )
    authorize_execution(
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
            "GF-REGIONAL-NOTIFY-004": lambda: run_notify004(site, target),
            "GF-REGIONAL-NOTIFY-005": lambda: run_notify005(
                site,
                target,
                nodes=cast(tuple[str, str, str], nodes),
                case_dir=case_dir,
                attempt=arguments.attempt,
                training_image=arguments.training_image,
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
