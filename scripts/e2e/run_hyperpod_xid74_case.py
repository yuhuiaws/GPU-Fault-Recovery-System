from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "safe-only": {
        "test_case_id": "GF-LIVE-XID74-SAFE-20260726",
        "registers": [1 << 0, 0, 0, 0, 0, 0, 0],
        "action": "NO_ACTION",
        "disposition": "MONITOR_ONLY",
        "mode": "monitor",
    },
    "corrected-only": {
        "test_case_id": "GF-LIVE-XID74-CORRECTED-20260726",
        "registers": [0, 0, 0, 0, 1 << 20, 0, 0],
        "action": "NO_ACTION",
        "disposition": "MONITOR_ONLY",
        "mode": "monitor",
    },
    "all-zero": {
        "test_case_id": "GF-LIVE-XID74-ALLZERO-20260726",
        "registers": [0, 0, 0, 0, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "support",
    },
    "secondary-only": {
        "test_case_id": "GF-LIVE-XID74-SECONDARY-20260726",
        "registers": [1 << 1, 0, 0, 0, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "support",
    },
    "marginal-channel": {
        "test_case_id": "GF-LIVE-XID74-MARGINAL-20260726",
        "registers": [1 << 21, 0, 0, 0, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "remediation-support",
        "required_operations": [
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "RUN_NVLINK74_WORKFLOW",
            "RESET_GPU",
            "QUARANTINE",
            "ESCALATE_SUPPORT",
        ],
    },
    "unexpected-production": {
        "test_case_id": "GF-LIVE-XID74-UNEXPECTED-20260726",
        "registers": [0, 0, 1 << 13, 0, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "support",
    },
    "fabric-reset": {
        "test_case_id": "GF-LIVE-XID74-FABRIC-20260726",
        "registers": [0, 0, 0, 1 << 18, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "support",
        "required_operations": [
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "ESCALATE_SUPPORT",
        ],
    },
    "unknown-register": {
        "test_case_id": "GF-LIVE-XID74-UNKNOWN-20260726",
        "registers": [0, 1, 0, 0, 0, 0, 0],
        "action": "ESCALATE_OPERATOR",
        "disposition": "EXECUTABLE",
        "mode": "support",
    },
    "ecc-parity": {
        "test_case_id": "GF-LIVE-XID74-ECC-MONITOR",
        "registers": [1 << 4, 0, 0, 0, 0, 0, 0],
        "action": "NO_ACTION",
        "disposition": "MONITOR_ONLY",
        "mode": "monitor",
        "occurrence_key": "register1.bit4",
        "expected_occurrence_count": 1,
    },
    "report-if-repeated": {
        "test_case_id": ("GF-LIVE-XID74-REPEATEDBIT-RESET-20260819"),
        "registers": [1 << 27, 0, 0, 0, 0, 0, 0],
        "action": "RESET_GPU",
        "disposition": "EXECUTABLE",
        "mode": "reset",
        "repetitions": 2,
        "occurrence_key": "register1.bit27",
        "expected_occurrence_count": 2,
        "required_operations": [
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "RUN_NVLINK74_WORKFLOW",
            "RESET_GPU",
        ],
    },
    "ecc-third": {
        "test_case_id": ("GF-LIVE-XID74-ECC-THIRD-RESET-20260819"),
        "registers": [1 << 4, 0, 0, 0, 0, 0, 0],
        "action": "RESET_GPU",
        "disposition": "EXECUTABLE",
        "mode": "reset",
        "repetitions": 3,
        "occurrence_key": "register1.bit4",
        "expected_occurrence_count": 3,
        "required_operations": [
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "RUN_NVLINK74_WORKFLOW",
            "RESET_GPU",
        ],
    },
    "mechanical-first": {
        "test_case_id": ("GF-LIVE-XID74-MECHANICAL-RESET-20260819"),
        "registers": [1 << 8, 0, 0, 0, 0, 0, 0],
        "action": "RESET_GPU",
        "disposition": "EXECUTABLE",
        "mode": "reset",
        "occurrence_key": "register1.bit8",
        "expected_occurrence_count": 1,
        "required_operations": ["RESET_GPU"],
        "forbidden_operations": [
            "RUN_NVLINK74_WORKFLOW",
            "ESCALATE_SUPPORT",
        ],
    },
}

QUARANTINE_TAINT = "gpu-fault.io/quarantined"
XID74_SUPPORT_TEMPLATE_VERSION = "xid74-support-zh-v2"


def run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def kubectl(
    namespace: str,
    arguments: list[str],
    *,
    input_text: str | None = None,
) -> str:
    return run(
        ["kubectl", "-n", namespace, *arguments],
        input_text=input_text,
    )


def snapshot(
    namespace: str,
    node: str,
    job_id: str,
) -> dict[str, Any]:
    node_value = json.loads(run(["kubectl", "get", "node", node, "-o", "json"]))
    pods = json.loads(
        kubectl(
            namespace,
            [
                "get",
                "pods",
                "-l",
                (f"gpu-fault.io/managed=true,gpu-fault.io/job-id={job_id}"),
                "-o",
                "json",
            ],
        )
    )
    return {
        "node": {
            "ready": any(
                item.get("type") == "Ready" and item.get("status") == "True"
                for item in node_value.get("status", {}).get("conditions", [])
            ),
            "unschedulable": bool(
                node_value.get("spec", {}).get("unschedulable", False)
            ),
            "taints": node_value.get("spec", {}).get("taints", []),
            "annotations": {
                key: value
                for key, value in node_value.get("metadata", {})
                .get("annotations", {})
                .items()
                if key.startswith("gpu-fault.io/")
            },
        },
        "managed_pods": [
            {
                "name": item["metadata"]["name"],
                "uid": item["metadata"].get("uid"),
                "attempt_id": item["metadata"]
                .get("labels", {})
                .get("gpu-fault.io/attempt-id"),
                "node": item.get("spec", {}).get("nodeName"),
                "phase": item.get("status", {}).get("phase"),
            }
            for item in pods.get("items", [])
        ],
    }


def injection_manifest(
    namespace: str,
    node: str,
    case_name: str,
    pci_bdf: str,
    registers: list[int],
    suffix: str,
    link_id: int = 3,
    repetitions: int = 1,
) -> dict[str, Any]:
    register_text = " ".join(f"0x{value:x}" for value in registers)
    message = (
        f"NVRM: Xid (PCI:{pci_bdf}): 74, "
        f"pid=1234, name=python, Link {link_id}, {register_text}"
    )
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"gpu-fault-xid74-{case_name}-{suffix}",
            "namespace": namespace,
            "labels": {
                "app": "gpu-fault-xid74-injection",
                "gpu-fault.io/xid74-case": case_name,
            },
        },
        "spec": {
            "nodeName": node,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "inject",
                    "image": ("public.ecr.aws/docker/library/busybox:1.36"),
                    "command": ["/bin/sh", "-c"],
                    "args": [
                        'i=0; while [ "$i" -lt "$REPETITIONS" ]; do '
                        "printf '%s\\n' \"$MESSAGE\" > /host-dev-kmsg; "
                        "i=$((i+1)); "
                        '[ "$i" -ge "$REPETITIONS" ] || sleep 2; '
                        "done; echo written"
                    ],
                    "env": [
                        {
                            "name": "MESSAGE",
                            "value": message,
                        },
                        {
                            "name": "REPETITIONS",
                            "value": str(repetitions),
                        },
                    ],
                    "securityContext": {"privileged": True},
                    "volumeMounts": [
                        {
                            "name": "dev-kmsg",
                            "mountPath": "/host-dev-kmsg",
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "dev-kmsg",
                    "hostPath": {
                        "path": "/dev/kmsg",
                        "type": "CharDevice",
                    },
                }
            ],
        },
    }


def evidence_query(
    cluster_id: str,
    node: str,
    observed_after: str,
    registers: list[int],
) -> str:
    return f"""
import json
from datetime import datetime
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

context = ApplicationContext.from_environment()
after = datetime.fromisoformat({observed_after!r})
registers = {registers!r}
events = [
    item for item in context.store.list_xid_events(
        {cluster_id!r}, {node!r}, observed_after=after
    )
    if item.xid == 74 and item.registers == registers
]
if not events:
    print(json.dumps({{"found": False}}))
    raise SystemExit()
event = max(events, key=lambda item: item.observed_at)
decision = context.store.get_xid_policy_decision(event.event_id)
result = {{
    "found": True,
    "event": event.model_dump(mode="json"),
    "decision": decision.model_dump(mode="json"),
}}
if decision.incident_id:
    try:
        incident = context.store.get_incident(decision.incident_id)
        result["incident"] = incident.model_dump(mode="json")
    except NotFoundError:
        pass
if decision.workflow_request_id:
    try:
        workflow = context.store.get_workflow(
            decision.workflow_request_id
        )
        result["workflow"] = workflow.model_dump(mode="json")
    except NotFoundError:
        pass
notifications = [
    item for item in context.store.list_notifications()
    if item.incident_id == decision.incident_id
]
result["notifications"] = []
for notification in notifications:
    notification_result = context.store.get_notification_result(
        notification.notification_id
    )
    result["notifications"].append(
        {{
            "notification": notification.model_dump(mode="json"),
            "delivery": (
                notification_result.model_dump(mode="json")
                if notification_result is not None
                else None
            ),
        }}
    )
print(json.dumps(result))
"""


def read_evidence(
    namespace: str,
    cluster_id: str,
    node: str,
    observed_after: str,
    registers: list[int],
) -> dict[str, Any]:
    output = kubectl(
        namespace,
        [
            "exec",
            "deployment/gpu-fault-api-ha",
            "--",
            "python",
            "-c",
            evidence_query(cluster_id, node, observed_after, registers),
        ],
    )
    return json.loads(output.splitlines()[-1])


def wait_for_evidence(
    namespace: str,
    cluster_id: str,
    node: str,
    observed_after: str,
    registers: list[int],
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {"found": False}
    while time.monotonic() < deadline:
        try:
            latest = read_evidence(
                namespace,
                cluster_id,
                node,
                observed_after,
                registers,
            )
        except RuntimeError as exc:
            latest = {
                "found": False,
                "transient_query_error": str(exc),
            }
            time.sleep(3)
            continue
        if latest.get("found"):
            workflow = latest.get("workflow")
            if workflow is None or workflow.get("status") in {
                "SUCCEEDED",
                "FAILED",
                "BLOCKED",
            }:
                return latest
        time.sleep(3)
    raise TimeoutError(f"XID 74 evidence did not become terminal: {latest}")


def _attempt_ids(
    snapshot_value: dict[str, Any],
) -> list[str | None]:
    return sorted(
        {item.get("attempt_id") for item in snapshot_value["managed_pods"]},
        key=lambda value: value or "",
    )


def _pod_uids(snapshot_value: dict[str, Any]) -> set[str]:
    return {
        str(item["uid"]) for item in snapshot_value["managed_pods"] if item.get("uid")
    }


def _managed_pod_identities(
    snapshot_value: dict[str, Any],
) -> list[tuple[str | None, str | None, str | None]]:
    identities = [
        (
            item.get("uid"),
            item.get("attempt_id"),
            item.get("node"),
        )
        for item in snapshot_value["managed_pods"]
    ]
    return sorted(
        identities,
        key=lambda identity: tuple(value or "" for value in identity),
    )


def evaluate_assertions(
    *,
    case: dict[str, Any],
    evidence: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    require_email_sent: bool = False,
) -> dict[str, bool]:
    decision = evidence["decision"]
    effective_action = decision["action"]
    mode = case["mode"]
    assertions = {
        "xid_is_74": evidence["event"]["xid"] == 74,
        "registers_match": (evidence["event"]["registers"] == case["registers"]),
        "action_matches": effective_action == case["action"],
        "disposition_matches": (decision["disposition"] == case["disposition"]),
        "node_was_ready": before["node"].get("ready") is True,
        "managed_workload_was_present": bool(before["managed_pods"]),
    }
    occurrence_key = case.get("occurrence_key")
    if occurrence_key is not None:
        assertions["occurrence_count_matches"] = (
            decision.get("nvlink_occurrence_counts", {}).get(occurrence_key)
            == case["expected_occurrence_count"]
        )
    if mode == "monitor":
        assertions.update(
            {
                "workflow_not_created": (
                    decision.get("workflow_request_id") is None
                    and evidence.get("workflow") is None
                ),
                "node_remained_ready": (after["node"].get("ready") is True),
                "node_stayed_schedulable": (not after["node"]["unschedulable"]),
                "node_taints_unchanged": (
                    before["node"]["taints"] == after["node"]["taints"]
                ),
                "node_annotations_unchanged": (
                    before["node"].get("annotations", {})
                    == after["node"].get("annotations", {})
                ),
                "managed_pod_identities_unchanged": (
                    _managed_pod_identities(before) == _managed_pod_identities(after)
                ),
            }
        )
        return assertions

    workflow = evidence.get("workflow") or {}
    completed = set(workflow.get("completed_operations", []))
    required = set(
        case.get(
            "required_operations",
            ["FREEZE_EVIDENCE", "ESCALATE_SUPPORT"],
        )
    )
    forbidden = set(case.get("forbidden_operations", []))
    assertions.update(
        {
            "workflow_succeeded": (workflow.get("status") == "SUCCEEDED"),
            "required_operations_completed": required <= completed,
            "forbidden_operations_not_executed": not (forbidden & completed),
        }
    )
    if mode in {"support", "remediation-support"}:
        notifications = evidence.get("notifications", [])
        xid74_notifications = [
            item
            for item in notifications
            if XID74_SUPPORT_TEMPLATE_VERSION
            in item["notification"].get("body_text", "")
        ]
        delivery_statuses = {
            item["delivery"]["status"]
            for item in xid74_notifications
            if item.get("delivery")
        }
        assertions.update(
            {
                "fixed_support_notification_exists": bool(xid74_notifications),
                "support_notification_delivery_recorded": bool(delivery_statuses),
            }
        )
        if require_email_sent:
            assertions["support_email_sent"] = "SENT" in delivery_statuses
    if mode == "support":
        assertions.update(
            {
                "gpu_reset_not_executed": ("RESET_GPU" not in completed),
                "workload_restart_not_executed": ("RESTART_WORKLOAD" not in completed),
                "node_remained_ready": (after["node"].get("ready") is True),
                "node_stayed_schedulable": (not after["node"]["unschedulable"]),
                "node_taints_unchanged": (
                    before["node"]["taints"] == after["node"]["taints"]
                ),
                "node_annotations_unchanged": (
                    before["node"].get("annotations", {})
                    == after["node"].get("annotations", {})
                ),
                "managed_pod_identities_unchanged": (
                    _managed_pod_identities(before) == _managed_pod_identities(after)
                ),
            }
        )
    if mode == "reset":
        before_attempts = set(_attempt_ids(before))
        after_attempts = set(_attempt_ids(after))
        before_uids = _pod_uids(before)
        after_uids = _pod_uids(after)
        assertions.update(
            {
                "support_not_executed": ("ESCALATE_SUPPORT" not in completed),
                "node_recovered_ready": (after["node"].get("ready") is True),
                "node_recovered_schedulable": (not after["node"]["unschedulable"]),
                "managed_workload_recovered": bool(after["managed_pods"]),
                "managed_attempt_restarted": bool(
                    before_attempts
                    and after_attempts
                    and before_attempts.isdisjoint(after_attempts)
                ),
                "managed_pod_uids_replaced": bool(
                    before_uids and after_uids and before_uids.isdisjoint(after_uids)
                ),
            }
        )
    if mode == "remediation-support":
        before_attempts = set(_attempt_ids(before))
        after_attempts = set(_attempt_ids(after))
        assertions.update(
            {
                "node_remained_ready": (after["node"].get("ready") is True),
                "node_quarantined": (
                    after["node"]["unschedulable"]
                    or any(
                        item.get("key") == QUARANTINE_TAINT
                        for item in after["node"]["taints"]
                    )
                ),
                "managed_workload_recovered": bool(after["managed_pods"]),
                "managed_attempt_restarted": bool(
                    before_attempts
                    and after_attempts
                    and before_attempts.isdisjoint(after_attempts)
                ),
            }
        )
    return assertions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one recorded HyperPod /dev/kmsg XID 74 case."
    )
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--pci-bdf", required=True)
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--link-id", type=int, default=3)
    parser.add_argument("--allow-support", action="store_true")
    parser.add_argument(
        "--allow-reset",
        action="store_true",
        help="Allow cases that can execute a real GPU reset.",
    )
    parser.add_argument(
        "--require-email-sent",
        action="store_true",
        help=(
            "For support cases, require the fixed support email "
            "delivery result to be SENT."
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser


def validate_case_flags(
    case: dict[str, Any],
    *,
    allow_support: bool,
    allow_reset: bool,
    require_email_sent: bool,
) -> None:
    mode = case["mode"]
    if mode in {"support", "remediation-support"} and not allow_support:
        raise ValueError("case requires --allow-support")
    if mode in {"reset", "remediation-support"} and not allow_reset:
        raise ValueError("case requires --allow-reset")
    if require_email_sent and mode not in {
        "support",
        "remediation-support",
    }:
        raise ValueError("--require-email-sent is only valid for support cases")


def main() -> int:
    args = build_parser().parse_args()
    case = CASES[args.case]
    try:
        validate_case_flags(
            case,
            allow_support=args.allow_support,
            allow_reset=args.allow_reset,
            require_email_sent=args.require_email_sent,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    started = datetime.now(timezone.utc)
    suffix = started.strftime("%H%M%S")
    before = snapshot(args.namespace, args.node, args.job_id)
    manifest = injection_manifest(
        args.namespace,
        args.node,
        args.case,
        args.pci_bdf,
        case["registers"],
        suffix,
        args.link_id,
        case.get("repetitions", 1),
    )
    kubectl(
        args.namespace,
        ["apply", "-f", "-"],
        input_text=json.dumps(manifest),
    )
    pod_name = manifest["metadata"]["name"]
    kubectl(
        args.namespace,
        [
            "wait",
            "--for=jsonpath={.status.phase}=Succeeded",
            f"pod/{pod_name}",
            "--timeout=90s",
        ],
    )
    evidence = wait_for_evidence(
        args.namespace,
        args.cluster_id,
        args.node,
        started.isoformat(),
        case["registers"],
        args.timeout,
    )
    after = snapshot(args.namespace, args.node, args.job_id)
    assertions = evaluate_assertions(
        case=case,
        evidence=evidence,
        before=before,
        after=after,
        require_email_sent=args.require_email_sent,
    )
    passed = all(assertions.values())
    report = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": case["test_case_id"],
        "verdict": "PASS" if passed else "FAIL",
        "executed_at": started.isoformat(),
        "environment": {
            "cluster_id": args.cluster_id,
            "job_id": args.job_id,
            "namespace": args.namespace,
            "node_id": args.node,
            "pci_bdf": args.pci_bdf,
        },
        "injection": {
            "type": "DIRECT_DEV_KMSG_USERSPACE_WRITE",
            "physical_gpu_fault": False,
            "nvidia_driver_generated": False,
            "contains_test_id": False,
            "message": manifest["spec"]["containers"][0]["env"][0]["value"],
            "registers": case["registers"],
            "nvlink_link_id": args.link_id,
            "repetitions": case.get("repetitions", 1),
        },
        "processing_trace": evidence,
        "preconditions": before,
        "postconditions": after,
        "assertions": assertions,
        "limitations": [
            "The record was written by a privileged userspace Pod "
            "to the host /dev/kmsg character device.",
            "This validates the production software path but does "
            "not prove a physical NVLink fault occurred.",
        ],
    }
    report_path = args.report or (
        ROOT / "reports" / f"hyperpod-xid74-{args.case}-{suffix}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    print(f"REPORT {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
