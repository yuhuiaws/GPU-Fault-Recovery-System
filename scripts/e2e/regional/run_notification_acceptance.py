from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    TRAINING_IMAGE,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    predecessor_evidence,
)
from scripts.e2e.regional.identity_acceptance_common import (  # noqa: E402
    ClusterTarget,
    IdentitySite,
)


CASE_IDS = tuple(f"GF-REGIONAL-NOTIFY-{number:03d}" for number in range(1, 6))
DRILL_PROBE = (Path(__file__).with_name("probes") / "notification_drill.py").read_text(
    encoding="utf-8"
)


class NotificationAcceptanceError(RuntimeError):
    pass


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


def drill(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    kind: str,
    drill_id: str,
) -> dict[str, Any]:
    pods = site.ready_pods("cpu", "gpu-fault-control-worker", target)
    if not pods:
        raise NotificationAcceptanceError("no Ready control-worker Pod")
    return site.pod_json(
        "cpu",
        target,
        pods[0],
        DRILL_PROBE,
        "--kind",
        kind,
        "--drill-id",
        drill_id,
        "--cluster-id",
        target.cluster_id,
        timeout=180,
    )


def validate_external_evidence(path: Path | None, kind: str) -> dict[str, Any]:
    if path is None:
        return {"valid": False, "error": f"{kind} evidence is required"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {"valid": False, "error": "external evidence is not an object"}
    if kind == "receipt":
        valid = bool(value.get("received")) and bool(value.get("reference"))
    else:
        valid = (
            int(value.get("send_count_delta", -1)) == 0
            and int(value.get("duplicate_inbox_count", -1)) == 0
        )
    return {"valid": valid, **value}


def run_notify001(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    attempt: int,
    receipt_evidence: Path | None,
) -> dict[str, Any]:
    reset = drill(
        site,
        target,
        kind="gpu-reset",
        drill_id=f"notify001-reset-{attempt}-{int(time.time())}",
    )
    restart = drill(
        site,
        target,
        kind="workload-restart",
        drill_id=f"notify001-restart-{attempt}-{int(time.time())}",
    )
    receipt = validate_external_evidence(receipt_evidence, "receipt")
    checks = {
        "gpu_reset_sent": reset["statuses"][0] == "SENT"
        and reset["provider_message_id_present"],
        "workload_restart_sent": restart["statuses"][0] == "SENT"
        and restart["provider_message_id_present"],
        "gpu_reset_deduplicated": reset["provider_message_id_stable"],
        "workload_restart_deduplicated": restart["provider_message_id_stable"],
        "human_receipt_confirmed": receipt["valid"],
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gpu_reset_drill": reset,
        "workload_restart_drill": restart,
        "receipt_evidence": receipt,
        "limitations": [
            "Both emails are explicitly labeled drills and are built without "
            "executing RESET_GPU or RESTART_WORKLOAD."
        ],
    }


def run_notify002(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    attempt: int,
    ses_window_evidence: Path | None,
) -> dict[str, Any]:
    result = drill(
        site,
        target,
        kind="gpu-reset",
        drill_id=f"notify002-{attempt}-{int(time.time())}",
    )
    external = validate_external_evidence(ses_window_evidence, "dedup")
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
        "checks": checks,
        "drill": result,
        "ses_window_evidence": external,
        "limitations": [
            "The first call sends one labeled drill email; the next three calls "
            "reuse the same notification ID and provider message ID."
        ],
    }


def deployment_notification_config(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    values = {}
    for app in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ):
        deployment = json.loads(site.cpu("get", "deployment", app, "-o", "json"))
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        environment = {
            item["name"]: item.get("value")
            for item in container.get("env", [])
            if "value" in item
        }
        values[app] = {
            "service_role": environment.get("GPU_FAULT_SERVICE_ROLE"),
            "dispatcher_enabled": environment.get(
                "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED"
            ),
            "async_delivery": environment.get("GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY"),
        }
    return values


def run_notify003(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/notifications/test_notifications.py::"
            "test_first_dispatch_suppresses_the_pre_enable_backlog",
            "tests/notifications/test_notifications.py::"
            "test_suppressed_backlog_can_be_requeued_on_demand",
            "tests/notifications/test_notifications.py::"
            "test_notification_worker_reclaims_expired_lease_and_stops",
        ],
        check=False,
        timeout=600,
    )
    config = deployment_notification_config(site, target)
    roles = {name: value["service_role"] for name, value in config.items()}
    checks = {
        "backlog_and_requeue_contract_tests": tests.returncode == 0,
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
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "deployment_config": config,
        "focused_test_returncode": tests.returncode,
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
import sys
from datetime import datetime
from gpu_fault.app import ApplicationContext

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
    if needle not in searchable:
        continue
    result = store.get_notification_result(notification.notification_id)
    records.append({
        "notification_id": notification.notification_id,
        "created_at": notification.created_at.isoformat(),
        "category": notification.category,
        "low_gpu_utilization": "LOW_GPU_UTILIZATION" in searchable,
        "matched_nodes": sorted(node for node in nodes if node in searchable),
        "status": result.status.value if result else None,
        "provider_message_id_present": bool(
            result is not None and result.provider_message_id
        ),
    })
print(json.dumps({"records": records}, sort_keys=True))
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
                    "template": template,
                },
                "Worker": {
                    "replicas": 2,
                    "restartPolicy": "Never",
                    "template": template,
                },
            },
        },
    }


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
    wait_seconds = int(config["duration_seconds"]) + 90
    started_at = datetime.now(timezone.utc)
    fixtures = []
    observations = []
    result: dict[str, Any] = {"verdict": "FAIL"}
    try:
        for count, selected in ((1, nodes[:1]), (3, nodes)):
            name = f"notify005-{count}n-{attempt}-{int(time.time())}"
            source = case_dir / f"{name}.source.yaml"
            source.write_text(
                yaml.safe_dump(
                    low_utilization_manifest(
                        name=name,
                        nodes=selected,
                        image=training_image,
                    ),
                    sort_keys=False,
                ),
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
            running = wait_running_pods(fixture, count)
            time.sleep(wait_seconds)
            query = regional.cpu_python(
                NOTIFICATION_QUERY_PROBE,
                name,
                started_at.isoformat(),
                *selected,
            )
            records = [item for item in query["records"] if item["low_gpu_utilization"]]
            observations.append(
                {
                    "node_count": count,
                    "nodes": list(selected),
                    "pods": running["pods"],
                    "notifications": records,
                }
            )
            fixture.delete()
        single_count = len(observations[0]["notifications"])
        three_count = len(observations[1]["notifications"])
        checks = {
            "single_node_notification_is_aggregated": 1 <= single_count <= 1,
            "three_node_notifications_do_not_scale_by_gpu": 1 <= three_count <= 3,
            "three_node_count_not_twenty_four": three_count != 24,
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "policy_config": config,
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
        "they validate notification aggregation without damaging hardware."
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
            "send one GPU-reset drill and one workload-restart drill through SES"
        ),
        "GF-REGIONAL-NOTIFY-002": (
            "send one GPU-reset drill and repeat the same notification ID three times"
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


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureError(f"received signal {signum}")


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)
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
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id)
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
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details=case_plan(
                arguments.case,
                target=target,
                nodes=nodes,
                predecessor=predecessor,
            ),
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
    case_dir = arguments.run_dir / "cases" / arguments.case
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started_at = utc_now()
    try:
        handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "GF-REGIONAL-NOTIFY-001": lambda: run_notify001(
                site,
                target,
                attempt=arguments.attempt,
                receipt_evidence=arguments.receipt_evidence,
            ),
            "GF-REGIONAL-NOTIFY-002": lambda: run_notify002(
                site,
                target,
                attempt=arguments.attempt,
                ses_window_evidence=arguments.ses_window_evidence,
            ),
            "GF-REGIONAL-NOTIFY-003": lambda: run_notify003(site, target),
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
        **{key: value for key, value in outcome.items() if key != "verdict"},
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
