#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
# ``src`` goes first on purpose: a standalone driver run from an installed
# wheel (the deploy-host bundle carries a reduced gpu_fault closure) must bind
# gpu_fault to this checkout before its first gpu_fault import, exactly as
# ``scripts.e2e.regional`` does for the drivers that import scripts first.
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS  # noqa: E402
from scripts.e2e.regional import regional_pod_inventory  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.collector_reboot_evidence import (  # noqa: E402
    partition_provider_events,
    provider_event_source,
    reboot_event_names,
)
from scripts.e2e.regional.guardrail_audit_evidence import (  # noqa: E402
    EC2_REPLACEMENT_MUTATIONS,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.regional_commands import (  # noqa: E402
    RegionalCommandFailed,
    exec_refused_before_script,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    component_python,
    predecessor_evidence,
    provider_event_actor_matches_role,
    required,
    run_case_main,
    settings_from_arguments,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-002"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-001"
CONFIRMATION = "DESTR002_REBOOT_ISOLATED_NODE"
EXPECTED_ACTION = "REBOOT_NODE"
# The HyperPod vocabulary; ``provider_reboot_evidence`` reads the EC2 one
# (RebootInstances; Run/Terminate/Stop/StartInstances) on an eks-managed cluster.
REBOOT_EVENTS = {"BatchRebootClusterNodes", "RebootClusterNodes"}
FORBIDDEN_EVENTS = {
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
}
EXECUTOR_APP = "gpu-fault-cluster-executor"
# A replacement executor on a healthy node is Ready well inside this budget.
REPLACEMENT_READY_SECONDS = 300
# The arbiter Deployments tolerate the cordon and gpu-fault.io/quarantined by
# design (an isolating control plane must not lock itself out), so the scheduler
# may bind the replacement to the very node this case reboots: idle and
# image-warm, it scores best (live round 8 and round 9 on HyperPod, 2/2). That
# Pod cannot start before the node is back, so its wait and the replay take the
# case's node-return budget instead (wait_node_ready's 1800 s).
REPLACEMENT_ON_REBOOTING_NODE_SECONDS = 1800
# How long the replacement's node must have been Ready before the replay execs:
# HyperPod restarts kubelet with its final configuration ~30 s after a boot, and
# an exec across that restart is refused -- live 2026-10-16 (round 9): Pod Ready
# 06:47:43, kubelet "Starting" again 06:47:46, exec "unable to upgrade
# connection" 06:47:46 (the rule of collector_inventory_reboot's settle wait).
REPLAY_NODE_READY_SECONDS = 45.0
# Retry budget on a healthy node for an exec kubectl refused before the probe ran.
REPLAY_EXEC_RETRY_SECONDS = 180
REPLAY_POLL_SECONDS = 5


# Finds the executor's node lifecycle adapter by the owner each provider's
# profile names for nodeReboot (gpu-fault-hyperpod-adapter: SageMaker
# BatchRebootClusterNodes; gpu-fault-eks-adapter: EC2 RebootInstances) and runs
# the same four read-only preflights through it, so both providers are audited
# under one case id. The "disabled" variant is a second adapter instance built
# from the live config with the reboot switch off; it never submits.
PREFLIGHT_PROBE = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment
from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodLifecycleAdapter,
)

node = sys.argv[1]
executor = executor_from_environment()
owners = sorted(
    str(getattr(item, "owner", "")) for item in executor.adapters if getattr(item, "owner", "")
)
providers_by_owner = {owner: name for name, owner in PROVIDER_NODE_REBOOT_OWNERS.items()}
step_adapter = next(
    (
        item
        for item in executor.adapters
        if getattr(item, "owner", "") in providers_by_owner
    ),
    None,
)
if step_adapter is None:
    print(json.dumps({"lifecycle_adapter": False, "adapter_owners": owners}))
    raise SystemExit(0)
provider = providers_by_owner[step_adapter.owner]
lifecycle = step_adapter.dispatcher.adapter
positive = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[node],
)
missing = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[],
)
wrong = lifecycle.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=["audit-other-node"],
)
disabled_config = lifecycle.config.model_copy(
    update={"execution_enabled": False, "reboot_enabled": False}
)
if provider == "eks-managed":
    from gpu_fault.eks_provider import EksManagedNodeProvider

    disabled = EksManagedNodeProvider(
        disabled_config,
        lifecycle.core,
        ec2_client=lifecycle.ec2,
        eks_client=lifecycle.eks,
        store=None,
    )
else:
    disabled = HyperPodLifecycleAdapter(disabled_config, client=lifecycle.client)
disabled_result = disabled.preflight(
    HyperPodAction.REBOOT,
    [node],
    isolation_verified_nodes=[node],
)
print(json.dumps({
    "provider": provider,
    "adapter_owner": step_adapter.owner,
    "configured_cluster_name": lifecycle.config.cluster_name,
    "positive": positive.model_dump(mode="json"),
    "missing_isolation": missing.model_dump(mode="json"),
    "wrong_isolation": wrong.model_dump(mode="json"),
    "disabled": disabled_result.model_dump(mode="json"),
}, sort_keys=True, default=str))
"""


# Runs in the *replacement* executor Pod after the submitting one was deleted,
# and reads the durable submission record through that Pod's own lifecycle
# adapter store. It never calls `submit`: the adapter's dedupe rule for a
# replay (HyperPodLifecycleAdapter._replay_durable_submission; the eks
# provider's ProviderSubmissionLedger.replay behind the same record type) is
# "same idempotency key, same request identity, state SUBMITTED, result
# present", and reading that record from the new Pod proves the same thing a
# second live submit would -- without asking the provider for a second reboot.
# The key
# is taken from the command's result_details and never recomputed: a runner
# that derives its own key can only confirm its own arithmetic.
# The probe takes only the two facts it needs (the recorded key and the step's
# node ids) -- never the whole command record, whose evidence can exceed one
# argv string -- and reports every in-Pod failure as a JSON ``error`` with the
# ``stage`` that raised, exiting 0: a non-zero exit therefore means the exec
# itself failed, and the runner names that (live 2026-10-05 the case died with
# only "command failed (1): <sensitive output redacted> (step: kubectl exec)").
STORE_REPLAY_PROBE = r"""
import json
import sys

request = json.loads(sys.argv[1])
submission_key = str(request["submission_idempotency_key"])
node_ids = [str(item) for item in request["node_ids"]]
result = {"replay_mode": "store-level", "duplicate": False, "checks": {}}
stage = "import"
try:
    from gpu_fault.cluster_executor import executor_from_environment
    from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS
    from gpu_fault.hyperpod import HyperPodAction

    stage = "executor"
    executor = executor_from_environment()
    stage = "adapter"
    lifecycle_owners = set(PROVIDER_NODE_REBOOT_OWNERS.values())
    step_adapter = next(
        (
            item
            for item in executor.adapters
            if getattr(item, "owner", "") in lifecycle_owners
        ),
        None,
    )
    if step_adapter is None:
        raise RuntimeError(
            "executor advertises no node lifecycle adapter: "
            + ", ".join(sorted(str(getattr(item, "owner", "")) for item in executor.adapters))
        )
    lifecycle = step_adapter.dispatcher.adapter
    result["adapter_owner"] = step_adapter.owner
    if lifecycle.store is None:
        raise RuntimeError("the durable provider submission store is unavailable")
    replay = getattr(lifecycle, "_replay_durable_submission", None) or (
        lambda action, ids, key: lifecycle._ledger.replay(action, ids, key)
    )
    stage = "record"
    record = lifecycle.store.get_hyperpod_submission(
        lifecycle.config.cluster_name,
        submission_key,
    )
    expected_identity = (HyperPodAction.REBOOT, tuple(sorted(node_ids)))
    result["checks"] = {
        "idempotency_key_matches": record.idempotency_key == submission_key,
        "request_identity_matches": record.request_identity == expected_identity,
        "state_submitted": record.state == "SUBMITTED",
        "result_present": record.result is not None,
    }
    result["record"] = record.model_dump(mode="json")
    if all(result["checks"].values()):
        stage = "replay"
        outcome = replay(HyperPodAction.REBOOT, node_ids, submission_key)
        result["duplicate"] = outcome is not None and outcome.duplicate is True
except Exception as exc:  # noqa: BLE001 - the runner reads the stage and type
    result["stage"] = stage
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    hyperpod_cluster: str
    executor_role_arn: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_EXECUTOR_ROLE_ARN": self.executor_role_arn,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }

    @property
    def executor_role_name(self) -> str:
        return self.executor_role_arn.rsplit("/", 1)[-1]


def configure(arguments: argparse.Namespace) -> Settings:
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        executor_role_arn=required(
            arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", ""),
            "executor role ARN",
        ),
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``.

    ``reuse`` consults ``reusable_focused_tests`` on the plan this case wrote:
    a passing result recorded against the same source digest is not re-run.
    """

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_hyperpod.py::"
        "test_read_only_preflight_reports_all_mutation_gates",
        "tests/hyperpod/test_hyperpod.py::"
        "test_provider_replace_can_be_disabled_without_disabling_reboot",
        "tests/hyperpod/test_hyperpod.py::test_reboot_submission_is_idempotent",
        "tests/execution/_node_action_cases_2.py::"
        "test_hyperpod_reboot_auto_confirms_new_ready_agent_incarnation",
        "tests/execution/_node_action_cases_2.py::"
        "test_hyperpod_reboot_waits_for_post_reboot_stabilization",
        "tests/regional/test_destructive_acceptance_fixtures.py::"
        "test_destr002_allows_transient_zero_gpu_capacity_before_validation",
        "tests/eks_provider/test_eks_provider.py::"
        "test_preflight_fails_closed_on_each_gate",
        "tests/eks_provider/test_eks_provider.py::"
        "test_durable_submission_survives_a_provider_restart",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def probe_provider(probe: dict[str, Any]) -> str:
    """The provider the preflight probe found; HyperPod when it says nothing."""

    return str(probe.get("provider") or "hyperpod")


def preflight_probe_errors(
    probe: dict[str, Any],
    hyperpod_cluster: str,
) -> list[str]:
    """The four read-only preflights, in the found adapter's own words.

    Both providers answer the same contract: a positive preflight that is safe
    to submit, two isolation gates that fail closed, a disabled switch that
    fails closed. HyperPod must also read ``NodeRecovery=None``; a plain EKS
    cluster has no NodeRecovery (its preflight reports none) and must be
    ``ACTIVE`` instead.
    """

    errors = []
    if (
        probe.get("lifecycle_adapter") is False
        or probe.get("hyperpod_adapter") is False
    ):
        owners = ", ".join(probe.get("adapter_owners") or []) or "none"
        return [
            "executor has no node lifecycle adapter owning nodeReboot (owners: "
            f"{owners}); DESTR-002 audits the provider reboot path (HyperPod "
            "BatchRebootClusterNodes or EC2 RebootInstances) and needs one"
        ]
    provider = probe_provider(probe)
    positive = probe.get("positive") or {}
    missing = probe.get("missing_isolation") or {}
    wrong = probe.get("wrong_isolation") or {}
    disabled = probe.get("disabled") or {}
    if probe.get("configured_cluster_name") != hyperpod_cluster:
        errors.append("executor provider cluster name differs from the plan")
    if not positive.get("safe_to_submit"):
        errors.append("positive reboot preflight is not safe_to_submit")
    if provider == "hyperpod":
        if positive.get("node_recovery") != "None":
            errors.append("HyperPod NodeRecovery is not None")
    else:
        if positive.get("node_recovery") is not None:
            errors.append(
                "an eks-managed preflight reports a HyperPod NodeRecovery: "
                f"{positive.get('node_recovery')!r}"
            )
        if positive.get("cluster_status") != "ACTIVE":
            errors.append(
                f"the EKS cluster is not ACTIVE: {positive.get('cluster_status')!r}"
            )
    for name, result in (("missing", missing), ("wrong", wrong)):
        failures = " ".join(result.get("gate_failures") or [])
        if result.get("safe_to_submit") or (
            "trusted scheduler isolation evidence is missing" not in failures
        ):
            errors.append(f"{name} isolation preflight did not fail closed")
    disabled_failures = " ".join(disabled.get("gate_failures") or [])
    if disabled.get("safe_to_submit") or (
        "REBOOT mutation is disabled by configuration" not in disabled_failures
    ):
        errors.append("disabled reboot preflight did not fail closed")
    return errors


def provider_reboot_evidence(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    injection_started: datetime,
    provider: str,
) -> tuple[list[str], dict[str, Any]]:
    """Exactly one accepted reboot by the executor role in the provider's own
    CloudTrail source, and no replace/delete.

    The positive claim polls until the event is visible or the delivery window
    closes; the negative claim cannot be proven inside that window and is
    recorded as provisional unless a forbidden mutation is already visible.
    Calls AWS refused (Karpenter's ``RunInstances`` dry-runs on plain EKS) are
    set aside and named in the count, never counted as the reboot.
    """

    source = provider_event_source(provider)
    reboot_names = set(reboot_event_names(provider))
    forbidden_names = (
        EC2_REPLACEMENT_MUTATIONS if provider == "eks-managed" else FORBIDDEN_EVENTS
    )
    # The waited-for events are the positive claim's own inventory (name
    # filtered, with their actor); the whole window carries the negative claim.
    waited = regional.wait_provider_events(
        injection_started,
        event_names=reboot_names,
        expected_count=1,
        event_source=source,
    )
    window_end = datetime.now(timezone.utc)
    events = regional.provider_events(
        injection_started, window_end, event_source=source
    )
    reboots = [
        dict(item)
        for item in partition_provider_events(waited)[0]
        if isinstance(item, dict) and item.get("event_name") in reboot_names
    ]
    mutations, refused = partition_provider_events(events)
    forbidden = [
        dict(item)
        for item in mutations
        if isinstance(item, dict) and item.get("event_name") in forbidden_names
    ]
    errors: list[str] = []
    if len(reboots) != 1:
        errors.append(
            "CloudTrail does not contain exactly one reboot event "
            f"({len(reboots)} accepted {sorted(reboot_names)}; {len(refused)} "
            "refused provider calls set aside)"
        )
    elif not provider_event_actor_matches_role(reboots[0], settings.executor_role_arn):
        errors.append("CloudTrail reboot actor is not the executor role")
    if forbidden:
        errors.append("CloudTrail contains replace/delete mutation")
    return errors, {
        "provider": provider,
        "event_source": source,
        "events": events,
        "reboot_events": reboots,
        "forbidden_events": forbidden,
        "refused_provider_calls": refused,
        "provider_events_provisional": (
            not forbidden and regional.provider_events_provisional(window_end)
        ),
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    fixture = RegionalLiveFixture(settings.regional)
    identity = fixture.evidence_identity()
    state = fixture.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.node)
    workloads = fixture.business_workloads(settings.node)
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    probe = fixture.executor_python(PREFLIGHT_PROBE, settings.node)
    provider = probe_provider(probe)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    errors = preflight_probe_errors(probe, settings.hyperpod_cluster)
    if not predecessor["valid"]:
        errors.append("DESTR-001 predecessor evidence is not PASS")
    if node["ready"] != "True":
        errors.append("target node is not Ready")
    if node["unschedulable"]:
        errors.append("target node is already unschedulable")
    if node["taints"]:
        errors.append("target node has pre-existing taints")
    if workloads:
        errors.append("target node has non-system running Pods")
    if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    reboot = capability(state.get("profile"), "nodeReboot")
    expected_owner = PROVIDER_NODE_REBOOT_OWNERS.get(provider)
    if reboot is None:
        errors.append("runtime profile has no nodeReboot capability")
    elif (
        reboot.get("mode") != "OWN"
        or reboot.get("owner") != expected_owner
        or reboot.get("adapter") != "regional-cluster-executor"
    ):
        errors.append(
            f"nodeReboot is not OWN by the {provider} lifecycle adapter "
            f"({expected_owner}) through the regional executor"
        )
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if state.get("event") is not None:
        errors.append("target node has a recent XID event")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "evidence_identity": identity,
        "provider": provider,
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "provider_preflight": probe,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def wait_for_submission(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    marker: str,
    observed_after: datetime,
    case_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        # A wait loop takes the cheap queue read; the drained-backlog gate is
        # for preflights (see store_snapshot).
        last = regional.store_snapshot(
            node=settings.node,
            marker=marker,
            observed_after=observed_after,
            hyperpod_cluster=settings.hyperpod_cluster,
            queue_attempts=1,
        )
        workflow = last.get("workflow") or {}
        submission = last.get("submission") or {}
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "workflow_status": workflow.get("status"),
                "completed_operations": workflow.get("completed_operations") or [],
                "submission_state": submission.get("state"),
            }
        )
        write_json_atomic(
            case_dir / "submission-timeline.json",
            {"entries": timeline},
        )
        if submission.get("state") == "SUBMITTED":
            return last
        if workflow.get("status") in {
            "BLOCKED",
            "FAILED",
            "SUCCEEDED",
            "SUPERSEDED",
        }:
            return last
        time.sleep(5)
    raise RegionalFixtureError(
        f"HyperPod reboot submission did not reach SUBMITTED: {last}"
    )


def redact_lease_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "lease_token" and child not in (None, ""):
                raw = str(child).encode()
                result["lease_token_sha256"] = hashlib.sha256(raw).hexdigest()
                result["lease_token_length"] = len(raw)
            else:
                result[key] = redact_lease_tokens(child)
        return result
    if isinstance(value, list):
        return [redact_lease_tokens(child) for child in value]
    return value


def duplicate_replay_gate(
    command: dict[str, Any],
    submission: dict[str, Any],
) -> dict[str, Any] | None:
    """Why the durable replay must not run, as a NOT_APPLIED record, or None.

    The replay reads the record under the key the executor wrote into the
    command's ``result_details``; the runner never recomputes it. Without the
    key, or without a SUBMITTED store record under that key, there is nothing
    a replay could confirm, and the case records that rather than guessing.
    """

    details = command.get("result_details") or {}
    key = details.get("submission_idempotency_key")
    if not key:
        reason = (
            "remote command result_details lacks submission_idempotency_key; "
            "the runner does not recompute it"
        )
    elif submission.get("state") != "SUBMITTED":
        reason = (
            f"HyperPod submission record is not SUBMITTED: {submission.get('state')!r}"
        )
    elif submission.get("idempotency_key") not in (None, key):
        reason = "submission record key differs from the command's recorded key"
    else:
        return None
    return {
        "status": "NOT_APPLIED",
        "replay_mode": "store-level",
        "duplicate": None,
        "reason": reason,
    }


def submission_target_errors(
    submission: dict[str, Any],
    *,
    target_aliases: set[str],
    node: str,
) -> list[str]:
    """The submission record must name exactly the target node, nothing else.

    ``requested_node_identifiers`` may carry the node's logical ID, instance ID
    or private DNS name rather than its Kubernetes name, hence the aliases from
    the provider preflight; but one identifier, inside that alias set, is the
    whole contract. An overlap test let a record naming extra nodes through.
    """

    requested = {
        str(item) for item in submission.get("requested_node_identifiers") or []
    }
    if not requested:
        return ["submission record names no node"]
    if len(requested) != 1:
        return [
            f"submission record names {len(requested)} nodes, not exactly the target"
        ]
    if not requested <= (target_aliases | {node}):
        return ["submission record does not identify the target node"]
    return []


def restart_executor(
    regional: RegionalLiveFixture,
    command: dict[str, Any],
) -> dict[str, Any]:
    pods_before = regional.ready_pods("gpu", "gpu-fault-cluster-executor")
    if len(pods_before) < 2:
        raise RegionalFixtureError("expected at least two Ready executor Pods")
    details = command.get("result_details") or {}
    source, executor_id = next(
        (
            (field, str(value))
            for field, value in (
                ("result_details.executor_id", details.get("executor_id")),
                ("last_lease_owner", command.get("last_lease_owner")),
                ("lease_owner", command.get("lease_owner")),
            )
            if value
        ),
        ("", ""),
    )
    # The case restarts *the submitting* executor; deleting an arbitrary Pod
    # when the lease owner matched none of them would test a different thing
    # and still report "executor restarted".
    matches = [
        item for item in pods_before if executor_id and str(item["name"]) in executor_id
    ]
    if len(matches) != 1:
        raise RegionalFixtureError(
            "cannot identify the submitting executor Pod from "
            f"{source or 'the command'}={executor_id!r}; Ready Pods: "
            f"{[str(item['name']) for item in pods_before]}"
        )
    target = matches[0]
    match_basis = {
        "field": source,
        "executor_id": executor_id,
        "rule": "the Pod name is a substring of the executor identity",
    }
    regional.kubectl(
        "gpu",
        "delete",
        "pod",
        str(target["name"]),
        "--wait=false",
        timeout=30,
    )
    started = time.monotonic()
    pods_after, replaced = wait_ready_replacement(
        regional,
        pods_before,
        str(target["uid"]),
        deadline=started + REPLACEMENT_READY_SECONDS,
    )
    # Only once the healthy-node budget is spent: a replacement bound to the
    # node this case reboots waits for that node, any other one has failed.
    bound = None
    if not replaced:
        bound = pending_replacement_node(
            regional,
            {str(item["uid"]) for item in pods_before},
            rebooting_nodes(command),
        )
    if not replaced and bound is not None:
        pods_after, replaced = wait_ready_replacement(
            regional,
            pods_before,
            str(target["uid"]),
            deadline=started + REPLACEMENT_ON_REBOOTING_NODE_SECONDS,
        )
    if not replaced:
        raise RegionalFixtureError(
            f"executor Deployment did not replace deleted Pod: {pods_after}"
        )
    return {
        "deleted": target,
        "match_basis": match_basis,
        "before": pods_before,
        "after": pods_after,
        "bound_to_rebooting_node": bound,
    }


def rebooting_nodes(command: dict[str, Any]) -> set[str]:
    """The nodes the reboot step names: the ones this case takes down."""

    return {str(item) for item in (command.get("step") or {}).get("node_ids") or []}


def wait_ready_replacement(
    regional: RegionalLiveFixture,
    pods_before: list[dict[str, Any]],
    deleted_uid: str,
    *,
    deadline: float,
) -> tuple[list[dict[str, Any]], bool]:
    """The last Ready executor Pods read, and whether they replace the deleted one."""

    pods_after: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        pods_after = regional.ready_pods("gpu", EXECUTOR_APP)
        if len(pods_after) >= len(pods_before) and all(
            item["uid"] != deleted_uid for item in pods_after
        ):
            return pods_after, True
        time.sleep(5)
    return pods_after, False


def pending_replacement_node(
    regional: RegionalLiveFixture,
    previous_uids: set[str],
    nodes: set[str],
) -> str | None:
    """The node in ``nodes`` a not-yet-Ready replacement executor is bound to.

    A Pod of the executor app that is not one of ``previous_uids`` and is not
    terminating is the Deployment's replacement. Best effort: an unreadable
    inventory answers None, and the healthy-node verdict stands.
    """

    try:
        items = regional_pod_inventory.pod_inventory(regional, "gpu", EXECUTOR_APP)[
            "items"
        ]
    except (RegionalFixtureError, ValueError, KeyError, TypeError):
        return None
    for item in items if isinstance(items, list) else []:
        metadata = (item.get("metadata") or {}) if isinstance(item, dict) else {}
        node = ((item.get("spec") or {}) if isinstance(item, dict) else {}).get(
            "nodeName"
        )
        if (
            metadata.get("uid") not in previous_uids
            and not metadata.get("deletionTimestamp")
            and isinstance(node, str)
            and node in nodes
        ):
            return node
    return None


def node_ready_age(snapshot: dict[str, Any]) -> float | None:
    """Seconds since the node last flipped Ready; None without ``ready_since``."""

    ready_since = snapshot.get("ready_since")
    if not isinstance(ready_since, str) or not ready_since:
        return None
    since = datetime.fromisoformat(ready_since.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - since).total_seconds()


def wait_replay_target(
    regional: RegionalLiveFixture,
    target: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    """Block until an exec into the replacement can reach it; what was seen.

    The Pod must still be the replacement ``restart_executor`` saw (same UID,
    not terminating) and its node must have been Ready for
    ``REPLAY_NODE_READY_SECONDS``. A node record without ``ready_since`` (an
    older projection) and a Pod record without a node are not waited on.
    """

    while True:
        current = json.loads(
            regional.kubectl("gpu", "get", "pod", str(target["name"]), "-o", "json")
        )
        metadata = current.get("metadata") or {}
        if metadata.get("uid") != target["uid"] or metadata.get("deletionTimestamp"):
            raise RegionalFixtureError(
                "replacement executor Pod UID changed before replay"
            )
        node = str(target.get("node") or "")
        if not node:
            return {"node": None, "node_ready_seconds": None}
        try:
            snapshot = regional.node_snapshot(node)
        except (RegionalFixtureError, ValueError, KeyError):
            snapshot = {}
        age = node_ready_age(snapshot)
        if snapshot.get("ready") == "True" and (
            age is None or age >= REPLAY_NODE_READY_SECONDS
        ):
            return {"node": node, "node_ready_seconds": age}
        if time.monotonic() >= deadline:
            raise RegionalFixtureError(
                f"replacement executor node {node} did not settle before the "
                f"replay: ready={snapshot.get('ready')} ready_age={age}"
            )
        time.sleep(REPLAY_POLL_SECONDS)


def replay_from_replacement(
    regional: RegionalLiveFixture,
    command: dict[str, Any],
    restart: dict[str, Any],
) -> dict[str, Any]:
    previous = {pod["uid"] for pod in restart["before"]}
    candidates = [pod for pod in restart["after"] if pod["uid"] not in previous]
    if len(candidates) != 1:
        raise RegionalFixtureError("replacement executor Pod identity is not unique")
    target = candidates[0]
    request = {
        "submission_idempotency_key": str(
            (command.get("result_details") or {}).get("submission_idempotency_key")
        ),
        "node_ids": [
            str(item) for item in (command.get("step") or {}).get("node_ids") or []
        ],
    }
    # On the node this case reboots the exec can also meet the reboot itself
    # (a replacement Ready before the provider took the node down), so the
    # budget is the node-return one there.
    on_rebooting_node = str(target.get("node") or "") in rebooting_nodes(command)
    deadline = time.monotonic() + (
        REPLACEMENT_ON_REBOOTING_NODE_SECONDS
        if on_rebooting_node
        else REPLAY_EXEC_RETRY_SECONDS
    )
    refused: list[str] = []
    while True:
        reach = wait_replay_target(regional, target, deadline=deadline)
        try:
            output = regional.kubectl(
                "gpu",
                "exec",
                "-i",
                str(target["name"]),
                "--",
                component_python("gpu"),
                "-",
                json.dumps(request, sort_keys=True),
                input_text=STORE_REPLAY_PROBE,
                timeout=180,
            )
            break
        except RegionalCommandFailed as exc:
            # The probe reports its own failures as JSON and exits 0, so this
            # is the exec itself. One kubectl refused before the probe ran (the
            # kubelet restarting under it, the container not attached yet)
            # repeated nothing and is tried again on the same Pod; anything
            # else -- the interpreter missing -- is final. Output stays redacted.
            if not exec_refused_before_script(exc) or time.monotonic() >= deadline:
                raise RegionalFixtureError(
                    "DESTR-002 store-level replay could not run in replacement "
                    f"executor Pod {target['name']} (uid {target['uid']}): {exc}"
                    + (f"; refused before: {refused}" if refused else "")
                ) from exc
            refused.append(exc.failure_class or "replica gone")
            time.sleep(REPLAY_POLL_SECONDS)
    lines = [line for line in output.splitlines() if line.strip()]
    value = json.loads(lines[-1]) if lines else None
    if not isinstance(value, dict):
        raise RegionalFixtureError(
            f"replacement executor {target['name']} replay returned no object"
        )
    return {
        **value,
        "executor_pod": target["name"],
        "executor_pod_uid": target["uid"],
        "executor_node": reach["node"],
        "replacement_on_rebooting_node": on_rebooting_node,
        "node_ready_seconds": reach["node_ready_seconds"],
        "refused_execs": refused,
    }


def workflow_errors(
    state: dict[str, Any],
    *,
    expected_artifact: str | None,
    expected_boot_id: str | None,
) -> list[str]:
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    submission = state.get("submission") or {}
    agent = state.get("agent") or {}
    if event.get("xid") != 79:
        errors.append("matched event is not XID 79")
    if decision.get("action") != EXPECTED_ACTION:
        errors.append("policy did not resolve XID 79 to REBOOT_NODE")
    if workflow.get("status") != "SUCCEEDED":
        errors.append("reboot workflow is not SUCCEEDED")
    operations = [item.get("operation") for item in workflow.get("official_steps", [])]
    required_operations = (
        "MARK_UNSCHEDULABLE",
        "RESTART_NODE",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    )
    for operation in required_operations:
        if operations.count(operation) != 1:
            errors.append(f"reboot workflow is missing {operation}")
        succeeded = [
            item
            for item in workflow.get("step_executions") or []
            if item.get("operation") == operation and item.get("status") == "SUCCEEDED"
        ]
        if len(succeeded) != 1:
            errors.append(f"reboot workflow lacks one successful {operation} execution")
    if [item for item in operations if item in required_operations] != list(
        required_operations
    ):
        errors.append("reboot validation and scheduling order is not exact")
    if {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "REPLACE_NODE"} & set(operations):
        errors.append("reboot workflow contains an unexpected physical operation")
    if submission.get("state") != "SUBMITTED":
        errors.append("HyperPod submission record is not SUBMITTED")
    if submission.get("action") != "REBOOT":
        errors.append("HyperPod submission action is not REBOOT")
    if expected_artifact and agent.get("artifact_sha256") != expected_artifact:
        errors.append("Node Agent artifact changed across reboot")
    if expected_boot_id and (
        not agent.get("boot_id") or agent.get("boot_id") == expected_boot_id
    ):
        errors.append("fleet Agent boot ID did not provably change")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("Node Agent did not return ACTIVE")
    return errors


def node_recovery_errors(
    baseline: dict[str, Any],
    first_ready: dict[str, Any],
    final: dict[str, Any],
) -> list[str]:
    errors = []
    if first_ready.get("boot_id") == baseline.get("boot_id"):
        errors.append("Kubernetes Node boot ID did not change")
    if first_ready.get("uid") != baseline.get("uid") or final.get(
        "uid"
    ) != baseline.get("uid"):
        errors.append("Kubernetes Node UID changed across reboot")
    if final.get("gpu_allocatable") != baseline.get("gpu_allocatable"):
        errors.append("target GPU capacity did not return to baseline after validation")
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    targets = (preflight["provider_preflight"].get("positive") or {}).get(
        "targets"
    ) or []
    details = {
        "risk": "destructive-provider-reboot",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "hyperpod_cluster": settings.hyperpod_cluster,
        "provider": preflight.get("provider"),
        "provider_targets": targets,
        "mutation": (
            "write one synthetic XID 79 to /dev/kmsg, allow the executor role "
            "to submit one provider reboot (HyperPod BatchRebootClusterNodes, or "
            "EC2 RebootInstances on eks-managed), then restart the submitting "
            "executor Pod and verify the durable submission record from the "
            "replacement Pod (store-level replay; no second submit)"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "node_boot_id": preflight["node"]["boot_id"],
            "agent_generation": (state.get("agent") or {}).get("generation"),
            "agent_artifact_sha256": (state.get("agent") or {}).get("artifact_sha256"),
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
        },
        "stop_conditions": [
            "DESTR-001 has not passed in formal sequence",
            "preflight or focused regression failure",
            "HyperPod NodeRecovery is not None (eks-managed: the EKS cluster is "
            "not ACTIVE)",
            "target node is not Ready/schedulable/idle",
            "node, Agent, Profile, cluster name or release drift",
            "submission record does not reach SUBMITTED",
            "CloudTrail shows more than one reboot or any replace/delete",
            "node does not return before the timeout",
            "GPU/host/fabric validation does not pass",
        ],
        "rollback": {
            "provider_reboot_does_not_destroy_the_instance": True,
            "runner_waits_for_the_original_node_to_return": True,
            "failed_return_keeps_the_node_out_of_service_for_operator_action": True,
            "provider_replace_is_never_attempted": True,
            "runner_never_submits_a_second_provider_reboot": True,
            "runner_finally_deletes_probe_resources": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "node_boot_id": preflight["node"]["boot_id"],
        "agent_generation": (preflight["store"].get("agent") or {}).get("generation"),
        "agent_artifact_sha256": (preflight["store"].get("agent") or {}).get(
            "artifact_sha256"
        ),
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-002 plan drifted: {planned} != {current}")


def host_probe(settings: Settings, run_id: str, case_dir: Path) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )


def verify_duplicate_replay(
    regional: RegionalLiveFixture,
    command: dict[str, Any],
    submission: dict[str, Any],
    executor_restart: dict[str, Any],
    case_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    replay_errors: list[str] = []
    replay_gate = duplicate_replay_gate(command, submission)
    if replay_gate is not None:
        duplicate = replay_gate
        replay_errors.append(
            f"durable HyperPod replay NOT_APPLIED: {replay_gate['reason']}"
        )
    else:
        duplicate = replay_from_replacement(regional, command, executor_restart)
        if duplicate.get("error"):
            replay_errors.append(
                "durable HyperPod replay probe failed in replacement executor "
                f"{duplicate.get('executor_pod')} at stage {duplicate.get('stage')}: "
                f"{duplicate['error']}"
            )
        elif duplicate.get("duplicate") is not True:
            replay_errors.append(
                "durable HyperPod submission record is not replayable from "
                f"the replacement executor: {duplicate.get('checks')}"
            )
    write_json_atomic(case_dir / "duplicate-replay.json", duplicate)
    return duplicate, replay_errors


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)

    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr002-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr002-{int(time.time())}-a{attempt}"
    host = host_probe(settings, run_id, case_dir)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        **preflight["evidence_identity"],
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "focused_tests_reused": bool(
            preflight["focused_tests"].get("focused_tests_reused")
        ),
    }
    injection_started: datetime | None = None
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("maintenance window ended before probe creation")
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA compute clients")
        if not baseline_host["kmsg_writable"]:
            raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
        target_bdf = str(baseline_host["gpu_inventory"][0]["pci_bdf"])
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        injection_started = datetime.now(timezone.utc)
        injection = host.execute(
            "write-xid79",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            target_bdf,
        )
        write_json_atomic(case_dir / "injection.json", injection)
        submitted = wait_for_submission(
            regional,
            settings,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=600,
        )
        write_json_atomic(
            case_dir / "submitted-state.json",
            redact_lease_tokens(submitted),
        )
        if (submitted.get("submission") or {}).get("state") != "SUBMITTED":
            raise RegionalFixtureError("reboot submission did not reach SUBMITTED")
        reboot_commands = [
            item
            for item in submitted.get("commands") or []
            if item.get("step", {}).get("operation") == "RESTART_NODE"
        ]
        if len(reboot_commands) != 1:
            raise RegionalFixtureError("reboot remote command is not unique")
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "maintenance window ended before executor restart"
            )
        executor_restart = restart_executor(regional, reboot_commands[0])
        write_json_atomic(case_dir / "executor-restart.json", executor_restart)
        duplicate, replay_errors = verify_duplicate_replay(
            regional,
            reboot_commands[0],
            submitted.get("submission") or {},
            executor_restart,
            case_dir,
        )
        submitted_workflow_id = str(
            (submitted.get("workflow") or {}).get("request_id") or ""
        )

        node_after_boot = regional.wait_node_ready(
            settings.node,
            timeout_seconds=1800,
            expected_boot_id=preflight["node"]["boot_id"],
        )
        write_json_atomic(case_dir / "node-after-boot.json", node_after_boot)
        state = regional.wait_for_workflow(
            node=settings.node,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=1800,
            hyperpod_cluster=settings.hyperpod_cluster,
            workflow_request_ids=(
                [submitted_workflow_id] if submitted_workflow_id else None
            ),
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        errors = workflow_errors(
            state,
            expected_artifact=(preflight["store"].get("agent") or {}).get(
                "artifact_sha256"
            ),
            expected_boot_id=(preflight["store"].get("agent") or {}).get("boot_id"),
        )
        errors.extend(replay_errors)
        provider_name = str(preflight.get("provider") or "hyperpod")
        provider_failures, provider_evidence = provider_reboot_evidence(
            regional,
            settings,
            injection_started=injection_started,
            provider=provider_name,
        )
        write_json_atomic(case_dir / "provider-events.json", provider_evidence)
        errors.extend(provider_failures)
        submission = state.get("submission") or {}
        target_aliases = {
            str(item)
            for target in (preflight["provider_preflight"].get("positive") or {}).get(
                "targets", []
            )
            for item in (
                target.get("node_logical_id"),
                target.get("instance_id"),
                target.get("private_dns_hostname"),
            )
            if item
        }
        errors.extend(
            submission_target_errors(
                submission,
                target_aliases=target_aliases,
                node=settings.node,
            )
        )
        final_node = regional.node_snapshot(settings.node)
        write_json_atomic(case_dir / "node-final.json", final_node)
        errors.extend(
            node_recovery_errors(
                preflight["node"],
                node_after_boot,
                final_node,
            )
        )
        if final_node["ready"] != "True":
            errors.append("target node is not Ready after reboot")
        if final_node["unschedulable"] or final_node["ownership_annotations"]:
            errors.append("target node scheduling ownership was not restored")
        cpu_after = regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "submission": submission,
                "duplicate_replay": duplicate,
                "provider": provider_name,
                "provider_events": provider_evidence["events"],
                "reboot_events": provider_evidence["reboot_events"],
                "refused_provider_calls": provider_evidence["refused_provider_calls"],
                "provider_events_provisional": provider_evidence[
                    "provider_events_provisional"
                ],
                "executor_restart": executor_restart,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            residuals = host.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
        try:
            final_node = regional.node_snapshot(settings.node)
            result["final_node"] = final_node
            if final_node["ready"] != "True":
                result["verdict"] = "FAIL"
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-002 HyperPod reboot acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", default="")
    value.add_argument("--region", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
