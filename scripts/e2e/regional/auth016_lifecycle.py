"""AUTH016 exercises the production credential rotation, not a private replica."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.node_key_custody_crypto import read_regular
from gpu_fault.admin.node_key_custody_pods import bound_consumer_pods
from gpu_fault.admin.rotate_token import (
    DEFAULT_QUIET_SECONDS,
    ROTATION_STEPS,
    STEP_ACCEPTED,
    STEP_DATA_PLANE_ROLLED,
    STEP_NODES_ROLLED,
    STEP_OVERLAP_PUBLISHED,
    STEP_RETIRING_DROPPED,
    load_rotation_state,
    rotation_state_path,
)
from gpu_fault.admin.site import effective_environment
from gpu_fault_release.regional_deployment_inventory import (
    DEPLOYMENTS,
    GPU_EXECUTOR_DEPLOYMENT,
)
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.admin_cli import admin_command
from scripts.e2e.regional.identity_acceptance_common import (
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
    IdentitySite,
    claim,
    read_cluster_token,
    run,
)

ROTATION_TIMEOUT_SECONDS = 3600
CONSUMER_TIMEOUT_SECONDS = 600
SAMPLER_JOIN_SECONDS = 495

CONSUMER_STORE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.collector_requirements import required_collectors_for_agent
store = ApplicationContext.from_environment().store
cluster_id = sys.argv[1]
agents = store.list_agents(cluster_id)
coverage = store.get_workload_coverage_heartbeat(cluster_id)
print(json.dumps({
    "cluster_id": cluster_id,
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "agents": {
        item.node_id: {
            "generation": item.generation,
            "incarnation": item.agent_incarnation_id,
            "last_seen_at": item.last_seen_at.isoformat(),
            "lease_expires_at": item.lease_expires_at.isoformat() if item.lease_expires_at else None,
            "lifecycle": item.lifecycle_state.value,
            "required_collectors": sorted(kind.value for kind in required_collectors_for_agent(item)),
        } for item in agents
    },
    "collectors": {
        item.node_id + "/" + item.collector.value: {
            "ingested_at": item.ingested_at.isoformat(),
            "last_success_at": item.last_success_at.isoformat() if item.last_success_at else None,
            "errors": bool(item.errors),
        } for item in store.list_collector_statuses(cluster_id)
    },
    "watcher": coverage.model_dump(mode="json") if coverage else None,
}))
"""

POD_TOKEN_PROBE = r"""
import hashlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
if len(sys.argv) != 2 or sys.argv[1] not in {"executor", "standard"}:
    raise RuntimeError("AUTH016 consumer trust mode is unbound")
ca_file = os.environ.get("SSL_CERT_FILE")
if sys.argv[1] == "executor":
    ca_file = os.environ.get("GPU_FAULT_CONTROL_PLANE_CA_FILE") or ca_file
if (
    not ca_file or ca_file != ca_file.strip() or not os.path.isabs(ca_file)
    or any(ord(character) < 32 or ord(character) == 127 for character in ca_file)
):
    raise RuntimeError("AUTH016 consumer CA file path is invalid")
context = ssl.create_default_context(cafile=ca_file)
token = os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"]
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/v1/fleet/agents",
    headers={
        "Authorization": "Bearer " + token,
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    },
)
try:
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        status, body = response.status, json.loads(response.read())
except urllib.error.HTTPError as error:
    with error:
        status, body = error.code, None
print(json.dumps({
    "cluster_id": os.environ["GPU_FAULT_CLUSTER_ID"],
    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    "status": status,
    "local_scope": isinstance(body, list) and bool(body) and all(
        item.get("cluster_id") == os.environ["GPU_FAULT_CLUSTER_ID"] for item in body
    ),
}))
"""


def invoke_rotation(site: IdentitySite, target: ClusterTarget, reference: str) -> None:
    state_dir = site.site.source.parent
    run(
        [
            *admin_command(state_dir),
            "rotate-token",
            "--state-dir",
            str(state_dir),
            "--gpu-cluster-arn",
            target.eks_cluster_arn,
            "--reference",
            reference,
        ],
        cwd=site.site.repository_root,
        env=effective_environment(site.site),
        timeout=ROTATION_TIMEOUT_SECONDS,
    )


def _stamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
        return parsed
    except ValueError:
        raise IdentityAcceptanceError(
            "rotation evidence has an invalid timestamp"
        ) from None


def registry_entry_committed(entry: dict[str, Any], new_digest: str) -> bool:
    """The durable registry entry has fully committed the new token.

    Field names follow ``RegionalClusterRegistration`` (``lifecycle_state``,
    not the fake-only ``membership_state`` that hid a false verdict on the live
    site, 2026-09-20): the new digest is current, no retiring token remains,
    and the cluster is enabled and ACTIVE.
    """

    return (
        entry.get("token_sha256") == new_digest
        and not entry.get("retiring_token_sha256")
        and entry.get("enabled") is True
        and entry.get("lifecycle_state") == "ACTIVE"
    )


def rotation_journal_errors(
    state: dict[str, Any],
    *,
    reference: str,
    cluster_id: str,
    old_digest: str,
    node_names: set[str],
) -> list[str]:
    steps = state.get("steps") or {}
    if (
        state.get("schema_version") != 1
        or state.get("cluster_id") != cluster_id
        or state.get("reference") != reference
        or state.get("status") != "COMPLETED"
        or state.get("old_token_sha256") != old_digest
        or not isinstance(state.get("new_token_sha256"), str)
        or len(state["new_token_sha256"]) != 64
        or state["new_token_sha256"] == old_digest
        or state.get("pending_token_cleanup_completed") is not True
        or state.get("keep_window") is not False
        or set(steps) != set(ROTATION_STEPS)
    ):
        return ["production rotation did not commit its complete bound lifecycle"]
    errors = []
    if set(
        (steps[STEP_DATA_PLANE_ROLLED].get("evidence") or {}).get("deployments", [])
    ) != set(DEPLOYMENTS):
        errors.append("rotation did not roll all data-plane credential consumers")
    node_evidence = steps[STEP_NODES_ROLLED].get("evidence") or {}
    if (
        not node_names
        or set(node_evidence.get("reinstalled_nodes", [])) != node_names
        or set((state.get("node_rollout") or {}).get("completed_nodes", []))
        != node_names
    ):
        errors.append("rotation did not reinstall the complete bound node fleet")
    if (steps[STEP_RETIRING_DROPPED].get("evidence") or {}).get(
        "retiring_token_dropped"
    ) is not True:
        errors.append("old credential was not explicitly withdrawn")
    accepted = steps[STEP_ACCEPTED].get("evidence") or {}
    if (
        type(accepted.get("quiet_seconds")) is not int
        or accepted["quiet_seconds"] != state.get("quiet_seconds")
        or accepted["quiet_seconds"] < DEFAULT_QUIET_SECONDS
        or not isinstance(accepted.get("waited_seconds"), (int, float))
        or accepted["waited_seconds"] < accepted["quiet_seconds"]
    ):
        errors.append("production old-token quiet window was not proved")
    if accepted.get("remote_command_losses"):
        errors.append("rotation observed remote command losses")
    try:
        ordered = [_stamp(steps[name]["completed_at"]) for name in ROTATION_STEPS]
        if ordered != sorted(ordered):
            errors.append("rotation lifecycle timestamps are out of order")
    except (IdentityAcceptanceError, KeyError, TypeError):
        errors.append("rotation lifecycle timestamps are incomplete")
    return errors


def consumer_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    cluster_id: str,
    after_withdrawal: datetime,
) -> list[str]:
    previous, agents = before.get("agents"), after.get("agents")
    if (
        before.get("cluster_id") != cluster_id
        or after.get("cluster_id") != cluster_id
        or not isinstance(previous, dict)
        or not previous
        or not isinstance(agents, dict)
        or set(previous) != set(agents)
    ):
        return ["credential consumer node inventory is incomplete or changed"]
    errors = []
    try:
        for node, agent in agents.items():
            required = previous[node]["required_collectors"]
            # Activation proof (live 2026-09-20, AUTH-016 a2): the reinstall wave
            # restarts the agent and re-registers it with the wave identity, which
            # the fleet records as a generation advance; the incarnation hashes
            # cluster/node/instance/boot_id and only changes on reboot, so it must
            # exist but is not expected to change. A heartbeat after the retiring
            # token was dropped can only have carried the new credential.
            if (
                not required
                or agent["required_collectors"] != required
                or agent["lifecycle"] != "ACTIVE"
                or not agent["incarnation"]
                or int(agent["generation"]) <= int(previous[node]["generation"])
                or _stamp(agent["last_seen_at"]) <= after_withdrawal
                or _stamp(agent["lease_expires_at"]) <= _stamp(after["captured_at"])
            ):
                errors.append(
                    f"{node}: Agent activation or collector policy is unproven"
                )
            for kind in required:
                record = (after.get("collectors") or {}).get(node + "/" + kind, {})
                if (
                    record.get("errors") is not False
                    or _stamp(record.get("ingested_at")) <= after_withdrawal
                    or _stamp(record.get("last_success_at")) <= after_withdrawal
                ):
                    errors.append(
                        f"{node}/{kind}: no successful post-withdrawal delivery"
                    )
        watcher = after.get("watcher") or {}
        if (
            watcher.get("cluster_id") != cluster_id
            or not watcher.get("watcher_instance")
            or _stamp(watcher.get("observed_at")) <= after_withdrawal
        ):
            errors.append(
                "Completion Watcher has no post-withdrawal coverage heartbeat"
            )
    except (IdentityAcceptanceError, KeyError, TypeError):
        errors.append("credential consumer evidence is incomplete")
    return errors


def pod_consumers(
    site: IdentitySite, target: ClusterTarget, *, new_digest: str | None = None
) -> dict[str, Any]:
    result = {}
    for name in DEPLOYMENTS:
        deployment = json.loads(
            site.gpu(target, "get", "deployment", name, "-o", "json")
        )
        expected = deployment.get("spec", {}).get("replicas")
        pods = json.loads(
            site.gpu(target, "get", "pods", "-l", "app=" + name, "-o", "json")
        )
        replicasets = json.loads(
            site.gpu(target, "get", "replicasets", "-l", "app=" + name, "-o", "json")
        )
        ready = bound_consumer_pods(
            deployment, replicasets, pods, namespace=site.namespace
        )
        if (
            type(expected) is not int
            or expected < 1
            or len(ready) != expected
            or len(pods.get("items", [])) != expected
        ):
            raise IdentityAcceptanceError(
                "rotation consumer Deployment is not fully Ready"
            )
        rows = []
        for pod in ready:
            proof = site.pod_json(
                "gpu",
                target,
                pod["metadata"]["name"],
                POD_TOKEN_PROBE,
                "executor" if name == GPU_EXECUTOR_DEPLOYMENT else "standard",
                timeout=60,
            )
            if (
                proof.get("cluster_id") != target.cluster_id
                or proof.get("status") != 200
                or proof.get("local_scope") is not True
                or new_digest is not None
                and proof.get("token_sha256") != new_digest
            ):
                raise IdentityAcceptanceError(
                    "a deployed consumer is not using the accepted credential"
                )
            rows.append(
                {
                    "pod": pod["metadata"]["name"],
                    "uid": pod["metadata"]["uid"],
                    **proof,
                }
            )
        result[name] = rows
    return result


def wait_consumer_snapshot(
    read: Callable[[], dict[str, Any]],
    before: dict[str, Any],
    *,
    cluster_id: str,
    after_withdrawal: datetime,
) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + CONSUMER_TIMEOUT_SECONDS
    while True:
        after = read()
        errors = consumer_errors(
            before, after, cluster_id=cluster_id, after_withdrawal=after_withdrawal
        )
        if not errors or time.monotonic() >= deadline:
            return after, errors
        time.sleep(5)


def finish_sampler(
    stop: threading.Event, thread: threading.Thread, result: dict[str, Any]
) -> None:
    stop.set()
    thread.join(timeout=SAMPLER_JOIN_SECONDS)
    errors = result["sampler_errors"]
    samples = result["samples"]
    if thread.is_alive():
        errors.append("sampler supervision did not drain")
        result["cleanup_complete"] = False
    result.setdefault("checks", {})["consumer_claim_continuity"] = (
        not errors
        and bool(samples)
        and {"baseline", "overlap", "new-consumers"}
        <= {item["phase"] for item in samples}
        and all(
            type(item["status"]) is int
            and item["status"] == 200
            and type(item["command_count"]) is int
            and item["command_count"] == 0
            for item in samples
        )
    )
    result["verdict"] = (
        "PASS"
        if "failure" not in result
        and result["cleanup_complete"]
        and all(result["checks"].values())
        else "FAIL"
    )


def run_rotation_acceptance(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
    retired_probe: Callable[[str], int | str],
) -> dict[str, Any]:
    state_path = rotation_state_path(site.site, target.cluster_id)
    existing = load_rotation_state(state_path)
    if existing and (
        existing.get("status") not in {"COMPLETED", "ROLLED_BACK"}
        or existing.get("pending_token_cleanup_completed") is not True
    ):
        raise IdentityAcceptanceError(
            "an unfinished production rotation must be reconciled first"
        )
    intent = case_dir / "auth016-production-intent.json"
    if intent.exists():
        raise IdentityAcceptanceError(
            "this rotation attempt already has an intent; reconcile its journal"
        )
    old_token = read_cluster_token(site, target)
    old_digest = hashlib.sha256(old_token.encode()).hexdigest()
    primary = site.regional(target)
    before = primary.cpu_python(CONSUMER_STORE_PROBE, target.cluster_id)
    before_pods = pod_consumers(site, target, new_digest=old_digest)
    if not isinstance(before.get("agents"), dict) or not before["agents"]:
        raise IdentityAcceptanceError("rotation requires a complete Agent baseline")
    original_registry = site.registry()
    siblings = {
        item["cluster_id"]: item
        for item in original_registry
        if item["cluster_id"] != target.cluster_id
    }
    reference = "AUTH016-" + secrets.token_hex(16)
    write_json_atomic(
        intent,
        {
            "reference": reference,
            "cluster_id": target.cluster_id,
            "release": primary.evidence_identity(),
            "old_token_sha256": old_digest,
            "production_journal": str(state_path),
        },
    )
    stop = threading.Event()
    baseline_sample = claim(site, target)
    samples: list[dict[str, Any]] = [
        {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "phase": "baseline",
            "status": baseline_sample.get("status"),
            "command_count": baseline_sample.get("command_count"),
        }
    ]
    if (
        baseline_sample.get("status") != 200
        or baseline_sample.get("command_count") != 0
    ):
        raise IdentityAcceptanceError("rotation consumer baseline claim failed")
    sampler_errors: list[str] = []
    result: dict[str, Any] = {
        "verdict": "FAIL",
        "samples": samples,
        "sampler_errors": sampler_errors,
        "reference": reference,
        "production_journal": str(state_path),
        "cleanup_complete": False,
    }

    def sample() -> None:
        while not stop.is_set():
            try:
                state = load_rotation_state(state_path) or {}
                steps = state.get("steps") or {}
                phase = (
                    (
                        "new-consumers"
                        if STEP_DATA_PLANE_ROLLED in steps
                        else "overlap"
                        if STEP_OVERLAP_PUBLISHED in steps
                        else "baseline"
                    )
                    if state.get("reference") == reference
                    else "baseline"
                )
                observed = claim(site, target)
                samples.append(
                    {
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "phase": phase,
                        "status": observed.get("status"),
                        "command_count": observed.get("command_count"),
                    }
                )
            except Exception as exc:
                sampler_errors.append(type(exc).__name__)
                return
            stop.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        invoke_rotation(site, target, reference)
        state = load_rotation_state(state_path) or {}
        errors = rotation_journal_errors(
            state,
            reference=reference,
            cluster_id=target.cluster_id,
            old_digest=old_digest,
            node_names=set(before["agents"]),
        )
        if errors:
            raise IdentityAcceptanceError("; ".join(errors))
        result["cleanup_complete"] = True
        new_digest = state["new_token_sha256"]
        token_path = Path(state["token_file"])
        if (
            hashlib.sha256(read_regular(token_path, private=True).strip()).hexdigest()
            != new_digest
        ):
            raise IdentityAcceptanceError(
                "committed token file does not match production journal"
            )
        result["pod_consumers"] = pod_consumers(site, target, new_digest=new_digest)
        if any(
            {row["uid"] for row in before_pods[name]}
            & {row["uid"] for row in result["pod_consumers"][name]}
            for name in DEPLOYMENTS
        ):
            raise IdentityAcceptanceError(
                "an old credential consumer Pod survived rotation"
            )
        withdrawn = _stamp(state["steps"][STEP_RETIRING_DROPPED]["completed_at"])
        after, errors = wait_consumer_snapshot(
            lambda: primary.cpu_python(CONSUMER_STORE_PROBE, target.cluster_id),
            before,
            cluster_id=target.cluster_id,
            after_withdrawal=withdrawn,
        )
        result["consumer_errors"] = errors
        result["consumer_state"] = after
        result["old_token_status"] = retired_probe(old_token)
        entries = site.registry()
        target_entries = [
            item for item in entries if item["cluster_id"] == target.cluster_id
        ]
        result["checks"] = {
            "full_production_lifecycle": True,
            "all_consumers_use_new_credential": not errors,
            "retired_token_rejected": result["old_token_status"] == 403,
            "target_registry_committed": len(target_entries) == 1
            and registry_entry_committed(target_entries[0], new_digest),
            "sibling_registrations_unchanged": siblings
            == {
                item["cluster_id"]: item
                for item in entries
                if item["cluster_id"] != target.cluster_id
            },
        }
        result["cleanup_complete"] = True
    except Exception as exc:
        result["failure"] = type(exc).__name__
        result["reconciliation"] = (
            "Use the bound production rotate-token journal. Do not restore Secrets "
            "by hand; TOKEN_FILE_WRITTEN intent permits fail-forward only."
        )
        raise IdentityCaseFailure(
            "production rotation acceptance did not complete", details=result
        ) from exc
    finally:
        finish_sampler(stop, thread, result)
        write_json_atomic(case_dir / "auth016-details.json", result)
    return result
