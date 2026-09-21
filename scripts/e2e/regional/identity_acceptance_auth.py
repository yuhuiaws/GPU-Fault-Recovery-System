from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from gpu_fault.regional_compatibility import (
    CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
)
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.auth015_context import (
    Auth015Context as Auth015Context,
)
from scripts.e2e.regional.auth015_context import (
    cleanup_auth015 as cleanup_auth015,
)
from scripts.e2e.regional.auth015_context import (
    scan_auth015_hosts as scan_auth015_hosts,
)
from scripts.e2e.regional.auth015_custody import run_custody_acceptance
from scripts.e2e.regional.auth015_custody_inputs import Auth015CustodyInputs
from scripts.e2e.regional.auth015_deployed import (
    prove_deployed_protocol as prove_deployed_protocol,
)
from scripts.e2e.regional.auth015_release import Auth015ReleaseInputs
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.identity_acceptance_common import (
    ACCEPTANCE_PROBE_OWNER,
    EXECUTOR_APP,
    WRITE_METHODS,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
    IdentitySite,
    claim,
    read_cluster_token,
    rollout_executor,
    run,
    run_cleanup_steps,
    secret_digest,
    write_cluster_token,
)
from scripts.e2e.regional.identity_acceptance_common import (
    secret_document as secret_document,
)
from scripts.e2e.regional.identity_auth_backlog import (
    auth008_claim as auth008_claim,
)
from scripts.e2e.regional.identity_auth_backlog import (
    run_auth008 as run_auth008,
)
from scripts.e2e.regional.identity_auth_checks import (
    AUTH015_PROOF_GAPS,
    auth015_scan_checks as auth015_scan_checks,
    valid_master_scan as valid_master_scan,
    validated_node_key_rotation as validated_node_key_rotation,
)
from scripts.e2e.regional.identity_auth_checks import (
    HIGH_RISK_ROUTE_BUCKETS as HIGH_RISK_ROUTE_BUCKETS,
)
from scripts.e2e.regional.identity_auth_checks import (
    INSTALLER_SECRET as INSTALLER_SECRET,
)
from scripts.e2e.regional.identity_auth_checks import (
    certificate_alert_checks as certificate_alert_checks,
)
from scripts.e2e.regional.identity_auth_checks import (
    commands_not_misterminated as commands_not_misterminated,
)
from scripts.e2e.regional.identity_auth_checks import (
    execution_token_hits as execution_token_hits,
)
from scripts.e2e.regional.identity_auth_checks import (
    failure_details as failure_details,
)
from scripts.e2e.regional.identity_auth_checks import (
    heartbeat_advanced as heartbeat_advanced,
)
from scripts.e2e.regional.identity_auth_checks import (
    high_risk_route_errors as high_risk_route_errors,
)
from scripts.e2e.regional.identity_auth_checks import (
    master_reference_scan as master_reference_scan,
)
from scripts.e2e.regional.identity_auth_checks import (
    node_key_digests as node_key_digests,
)
from scripts.e2e.regional.identity_auth_checks import (
    registry_token_digests as registry_token_digests,
)
from scripts.e2e.regional.identity_auth_checks import (
    update_registry_token as update_registry_token,
)
from scripts.e2e.regional.identity_auth_checks import (
    verdict as verdict,
)
from scripts.e2e.regional.identity_auth_checks import (
    world_open_rules as world_open_rules,
)
from scripts.e2e.regional.identity_auth_probes import (
    AGENT_SNAPSHOT_PROBE as AGENT_SNAPSHOT_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    ANONYMOUS_ROUTE_PROBE as ANONYMOUS_ROUTE_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    AUTH008_BACKLOG_PROBE as AUTH008_BACKLOG_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    AUTH008_CLAIM_PROBE as AUTH008_CLAIM_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    AUTH008_EXECUTOR_IDENTITY_PROBE as AUTH008_EXECUTOR_IDENTITY_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    DIRECT_CLAIM_PROBE_TEMPLATE as DIRECT_CLAIM_PROBE_TEMPLATE,
)
from scripts.e2e.regional.identity_auth_probes import (
    EXECUTION_TOKEN_DIGEST_PROBE as EXECUTION_TOKEN_DIGEST_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    REMOTE_STATUS_PROBE as REMOTE_STATUS_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    ROUTE_INVENTORY_PROBE as ROUTE_INVENTORY_PROBE,
)
from scripts.e2e.regional.identity_auth_probes import (
    TLS_BOUNDARY_PROBE as TLS_BOUNDARY_PROBE,
)
from scripts.e2e.regional.identity_auth_sampling import (
    TokenRotationSampler as TokenRotationSampler,
)
from scripts.e2e.regional.identity_fleet_scope import authenticated_fleet_isolation

AUTH015_PROBE = Path(__file__).with_name("probes") / "auth015_node_probe.py"
AUTH013_PROBE = Path(__file__).with_name("probes") / "auth013_certificate_probe.py"
NODE_ACTION_KEYS_SECRET = "gpu-fault-node-action-keys"
# How long AUTH-016 waits for its sampler thread after asking it to stop. A
# sample is one ``direct_claim``: the executor Pod lookup (``kubectl get``,
# default 300 s bound) plus one exec bounded at 60 s. The old 15 s join let a
# sampler mid-claim outlive the restore and read the restored token as a
# "completed"-phase 200.
DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS = 60
DIRECT_CLAIM_JOIN_SECONDS = 300 + DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS + 15


# A saved, leased NO_ACTION workflow prevents the dispatcher from owning the
# candidate. No node/workload targets, adapters, registry writes or SQL deletion.


def run_auth007(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    original = site.registry()
    if primary.cluster_id == secondary.cluster_id:
        raise IdentityAcceptanceError("isolation requires distinct cluster identities")
    if site.registry_generation() is None:
        raise IdentityAcceptanceError(
            "registration isolation requires a durable registry"
        )
    updated = [dict(item) for item in original]
    found = False
    for item in updated:
        if item.get("cluster_id") == secondary.cluster_id:
            if item.get("enabled") is not True:
                raise IdentityAcceptanceError("secondary registration is not enabled")
            item["enabled"] = False
            found = True
    if not found:
        raise IdentityAcceptanceError("secondary cluster is absent from registry")
    disabled_latency: float | None = None
    restored_latency: float | None = None
    secondary_disabled: dict[str, Any] = {}
    primary_healthy: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        site.write_registry(updated, expected_entries=original)
        # The revision is applied once every member acked it; that wait is the
        # isolation delay, and write_registry measured it from the POST.
        disabled_latency = site.last_registry_ready_seconds or site.rollout_control()
        secondary_disabled = claim(site, secondary)
        primary_healthy = claim(site, primary)
    except Exception as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"
        raise IdentityCaseFailure(
            str(exc),
            details=failure_details(
                checks=checks,
                cleanup_errors=cleanup_errors,
                secondary_disabled=secondary_disabled,
                primary_healthy=primary_healthy,
            ),
        ) from exc
    finally:
        # Update in place: an IdentityCaseFailure raised above already holds
        # these two containers, and a rebinding here would leave it the empty
        # ones.
        step_outcomes, step_errors = run_cleanup_steps(
            [
                ("restore_registry", lambda: site.restore_registry(original)),
                ("rollout_control", site.rollout_control),
                ("secondary_claim", lambda: claim(site, secondary)),
            ]
        )
        cleanup.update(step_outcomes)
        cleanup_errors.extend(step_errors)
        restored_latency = site.last_registry_ready_seconds or cleanup.get(
            "rollout_control"
        )
        write_json_atomic(
            case_dir / "auth007-details.json",
            {
                "cleanup": cleanup,
                "cleanup_errors": cleanup_errors,
                "secondary_disabled": secondary_disabled,
                "primary_healthy": primary_healthy,
            },
        )
    secondary_recovered = cleanup.get("secondary_claim") or {}
    try:
        disabled_detail = json.loads(secondary_disabled.get("detail") or "{}")
    except (TypeError, ValueError):
        disabled_detail = None
    checks = {
        "secondary_disabled_403": (
            secondary_disabled.get("status") == 403
            and disabled_detail == {"detail": "regional cluster authentication failed"}
        ),
        "primary_remains_200": (
            primary_healthy.get("status") == 200
            and primary_healthy.get("command_count") == 0
        ),
        "secondary_recovers_200": (
            secondary_recovered.get("status") == 200
            and secondary_recovered.get("command_count") == 0
        ),
        "registry_restored": site.registry() == original,
        "cleanup_completed": not cleanup_errors,
    }
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "isolation_latency_seconds": disabled_latency,
        "restore_latency_seconds": restored_latency,
        "cleanup_errors": cleanup_errors,
        "limitations": [
            "The case disables the explicitly selected secondary test cluster's "
            "registration; it does not disable a production training cluster. "
            "Latencies are measured from the revision POST to the last member "
            "ack, not from a control-plane rollout."
        ],
    }


def route_inventory(site: IdentitySite, target: ClusterTarget) -> list[dict[str, Any]]:
    cpu_pods = site.ready_pods("cpu", "gpu-fault-api-ha", target)
    if not cpu_pods:
        raise IdentityAcceptanceError("no Ready control-plane API Pod")
    value = site.pod_json(
        "cpu",
        target,
        cpu_pods[0],
        ROUTE_INVENTORY_PROBE,
    )
    return cast(list[dict[str, Any]], value["routes"])


def anonymous_routes(
    site: IdentitySite,
    target: ClusterTarget,
    routes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    executor = site.any_executor_pod(target)
    requests = [
        {
            "path": item["path"],
            "method": method,
            "bucket": item["bucket"],
        }
        for item in routes
        for method in item["methods"]
        if method not in {"HEAD", "OPTIONS"}
    ]
    value = site.pod_json(
        "gpu",
        target,
        executor,
        ANONYMOUS_ROUTE_PROBE,
        json.dumps(requests, separators=(",", ":")),
        timeout=600,
    )
    return cast(list[dict[str, Any]], value["results"])


def data_plane_execution_token_hits(
    site: IdentitySite,
    target: ClusterTarget,
) -> list[dict[str, str]]:
    """Data-plane Secret values and Pod env compared to the real token digest.

    The digest is computed inside the API Pod (the only place the token is
    allowed to exist) and only the digest crosses to the runner.
    """

    reference = site.api_pod_json(EXECUTION_TOKEN_DIGEST_PROBE)
    digests = {str(reference["sha256"]), str(reference["stripped_sha256"])}
    secrets_document = json.loads(
        site.gpu(target, "get", "secret", "-o", "json", all_namespaces=True)
    )
    pods_document = json.loads(
        site.gpu(target, "get", "pod", "-o", "json", all_namespaces=True)
    )
    return execution_token_hits(secrets_document, pods_document, digests=digests)


def run_auth010(site: IdentitySite, target: ClusterTarget) -> dict[str, Any]:
    """AUTH-010, kept runnable; its content is a subset of AUTH-014's audit."""
    routes = route_inventory(site, target)
    cluster_routes = [
        {
            **item,
            "methods": [
                method
                for method in item["methods"]
                if method not in {"HEAD", "OPTIONS"}
            ],
        }
        for item in routes
        if item["bucket"] == "cluster-token"
    ]
    results = anonymous_routes(site, target, cluster_routes)
    unsafe_writes = [
        {
            "path": item["path"],
            "methods": sorted(set(item["methods"]) & WRITE_METHODS),
            "bucket": item["bucket"],
        }
        for item in routes
        if set(item["methods"]) & WRITE_METHODS
        and item["bucket"] in {"public", "metrics", None}
    ]
    token_hits = data_plane_execution_token_hits(site, target)
    checks = {
        "cluster_token_routes_present": bool(results),
        "cluster_token_routes_anonymous_401": all(
            item.get("status") == 401 for item in results
        ),
        "write_routes_explicitly_protected": not unsafe_writes,
        "execution_token_absent_from_data_plane": not token_hits,
    }
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "status": "superseded",
        "superseded_by": "GF-REGIONAL-AUTH-014",
        "route_count": len(routes),
        "cluster_token_results": results,
        "unsafe_write_routes": unsafe_writes,
        "execution_token_hits": token_hits,
        "limitations": [
            "Anonymous requests validate the deployed route registry and "
            "middleware; they do not exercise every authorized success payload.",
            "AUTH-010 is superseded by GF-REGIONAL-AUTH-014, which audits every "
            "bucket and the outside-VPC denial; this run is kept for the chain.",
        ],
    }


def executor_claim_identity(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    deployment = json.loads(
        site.gpu(
            target,
            "get",
            "deployment",
            EXECUTOR_APP,
            "-o",
            "json",
        )
    )
    environment = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    pod = site.any_executor_pod(target)
    owners = site.pod_json(
        "gpu",
        target,
        pod,
        (
            "import json;"
            "from gpu_fault.cluster_executor import executor_from_environment;"
            "print(json.dumps({'owners':sorted("
            "executor_from_environment().execution_owners)}))"
        ),
    )["owners"]
    return {
        "artifact": environment["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"],
        "compatibility": environment["GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"],
        "owners": owners,
    }


def direct_claim(
    primary: Any,
    target: ClusterTarget,
    *,
    token: str,
    identity: dict[str, Any],
) -> int | str:
    """One executor claim with ``token``, made from inside the GPU data plane.

    The control plane is only reachable from the GPU VPC (private hosted zone,
    NLB allowlist -- AUTH-014 proves exactly that), so a claim issued from the
    driver host can only ever be URLError, which is what every AUTH-016 sample
    read live. The probe runs in a Ready executor Pod, uses the Pod's own
    control-plane URL and CA, and carries the token in the script body on
    stdin -- never on argv.
    """

    # The probe owner never has commands, so the claim authenticates exactly
    # like the executor's own and leases nothing (identity["owners"] leased the
    # real executor's work for 60 s per sample, every 2 s, for eight minutes).
    # The claim route answers 503 to a protocol version the control plane no
    # longer accepts; a probe pinned to the old number read 503 in every phase.
    payload = {
        "executor_id": "token-rotation-probe",
        "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        "executor_artifact_sha256": identity["artifact"],
        "executor_compatibility_digest": identity["compatibility"],
        "execution_owners": [ACCEPTANCE_PROBE_OWNER],
        "max_commands": 1,
        "lease_seconds": 60,
    }
    script = (
        DIRECT_CLAIM_PROBE_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__TOKEN__", json.dumps(token))
        .replace("__CLUSTER_ID__", json.dumps(target.cluster_id))
    )
    try:
        # One attempt: a retried claim is a second authenticated request and
        # would be read as the sample's answer for the phase it lands in.
        value = primary.executor_python(
            script, timeout=DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS, attempts=1
        )
    except Exception as exc:
        # No Ready executor to run the probe from (mid-rollout): an
        # infrastructure gap, not an authentication outcome.
        return f"probe-unavailable:{type(exc).__name__}"
    status = value.get("status")
    return int(status) if isinstance(status, int) else str(status)


def auth016_result(
    site: IdentitySite,
    primary: Any,
    *,
    samples: list[dict[str, Any]],
    before_commands: dict[str, Any],
    new_token_during_overlap: int | str | None,
    old_token_after_completion: int | str | None,
    restored: bool,
    registry_generation: int | None,
    control_rollout: float | None,
    executor_rollout: float | None,
    old_token: str,
    new_token: str,
) -> dict[str, Any]:
    """Turn the AUTH-016 samples and probes into checks and evidence."""

    by_phase: dict[str, list[int | str]] = {}
    for item in samples:
        by_phase.setdefault(str(item["phase"]), []).append(item["status"])
    # Missing observations cannot prove continuous availability. A transport
    # failure or unavailable probe is a failed sample, not an omitted sample.
    uninterrupted = [
        phase
        for phase in ("baseline", "overlap", "cutover", "completed")
        if by_phase.get(phase)
        and all(type(value) is int and value == 200 for value in by_phase[phase])
    ]
    before_ids = [str(item["command_id"]) for item in before_commands["commands"]]
    after_commands = primary.cpu_python(REMOTE_STATUS_PROBE, *before_ids)
    before_status = {
        item["command_id"]: item["status"] for item in before_commands["commands"]
    }
    after_status = {
        item["command_id"]: item["status"] for item in after_commands["commands"]
    }
    checks = {
        "baseline_only_200": "baseline" in uninterrupted,
        "overlap_only_200": "overlap" in uninterrupted,
        "cutover_only_200": "cutover" in uninterrupted,
        "completed_only_200": "completed" in uninterrupted,
        "both_credentials_sampled_during_overlap": (
            {
                item.get("credential_slot")
                for item in samples
                if item.get("phase") == "overlap"
            }
            == {"old", "new"}
        ),
        "new_token_accepted_during_overlap": new_token_during_overlap == 200,
        "old_token_rejected_after_completion": old_token_after_completion == 403,
        "remote_commands_not_misterminated": commands_not_misterminated(
            before_status, after_status
        ),
        "original_credentials_restored": restored,
    }
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "remote_command_baseline": {
            "open_before": sorted(before_status),
            "status_after": {
                command_id: after_status.get(command_id) for command_id in before_status
            },
        },
        "statuses_by_phase": {
            phase: sorted({str(value) for value in statuses})
            for phase, statuses in sorted(by_phase.items())
        },
        "rotation_window_seconds": 1800,
        "registry_mode": (
            "durable-revision" if registry_generation is not None else "clusters.json"
        ),
        "registry_generation_before": registry_generation,
        "registry_generation_after": site.registry_generation(),
        "control_rollout_seconds": control_rollout,
        "executor_rollout_seconds": executor_rollout,
        "samples": [
            {
                "observed_at": item["observed_at"],
                "phase": item["phase"],
                "status": item["status"],
                "credential_slot": item.get("credential_slot"),
            }
            for item in samples
        ],
        "old_token_sha256": secret_digest(old_token),
        "new_token_sha256": secret_digest(new_token),
        "limitations": [
            "The retiring token stays valid for the whole overlap window, so this "
            "case proves there is no interruption, not that the old credential is "
            "revoked instantly; both original Secrets are restored.",
            "remote_commands_not_misterminated requires a nonempty command baseline "
            "in the target cluster; an unrelated synthetic cluster is not proof "
            "that the rotating cluster's commands survived.",
        ],
    }


def restore_auth016(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    thread: threading.Thread,
    original_registry: list[dict[str, Any]],
    old_token: str,
    new_token: str,
    connection_uid: str,
    registry_started: bool,
    token_started: bool,
) -> tuple[dict[str, Any], list[str]]:
    def verify_restored() -> bool:
        return registry_token_digests(
            site.registry(), target.cluster_id
        ) == registry_token_digests(
            original_registry, target.cluster_id
        ) and secret_digest(read_cluster_token(site, target)) == secret_digest(
            old_token
        )

    # In place: the IdentityCaseFailure raised above holds these containers.
    def restore_token() -> None:
        if registry_token_digests(
            site.registry(), target.cluster_id
        ) != registry_token_digests(original_registry, target.cluster_id):
            raise IdentityAcceptanceError(
                "registry restoration is unproven; token restore deferred"
            )
        current = read_cluster_token(site, target)
        if current == old_token:
            return
        write_cluster_token(
            site,
            target,
            old_token,
            expected_token=new_token,
            expected_uid=connection_uid,
        )

    def restore_executor() -> float:
        if read_cluster_token(site, target) != old_token:
            raise IdentityAcceptanceError(
                "token restoration is unproven; executor rollout deferred"
            )
        return rollout_executor(site, target)

    steps: list[tuple[str, Any]] = []
    if not thread.is_alive() and registry_started:
        steps.extend(
            [
                (
                    "restore_registry",
                    lambda: site.restore_registry(
                        original_registry,
                        reason=(
                            "GF-REGIONAL-AUTH-016 restore: original token republished"
                        ),
                    ),
                ),
                ("rollout_control", site.rollout_control),
            ]
        )
    if not thread.is_alive() and token_started:
        steps.extend(
            [
                ("restore_cluster_token", restore_token),
                ("rollout_executor", restore_executor),
            ]
        )
    if not thread.is_alive():
        steps.append(("verify_restored", verify_restored))
    return run_cleanup_steps(steps)


def run_auth016(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    """Run the full production rotation and independently observe its consumers."""
    from scripts.e2e.regional.auth016_lifecycle import run_rotation_acceptance

    identity = executor_claim_identity(site, target)
    primary = site.regional(target)
    return run_rotation_acceptance(
        site,
        target,
        case_dir=case_dir,
        retired_probe=lambda token: direct_claim(
            primary, target, token=token, identity=identity
        ),
    )


CERTIFICATE_CHECK_TIMER = "gpu-fault-certificate-check.timer"


def run_auth013(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
    node: str = "",
    host_probe_image: str = "",
) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    result = site.pod_json("gpu", target, pod, TLS_BOUNDARY_PROBE)
    threshold = max(30, int(site.config["health"]["certificate_min_validity_days"]))
    alert: dict[str, Any] | None = None
    alert_residuals: dict[str, Any] = {}
    if node and host_probe_image:
        probe = HostProbeFixture(
            HostProbeSettings(
                state_directory=case_dir / "host-probes",
                kubeconfig=site.gpu_kubeconfig,
                context=target.context,
                namespace=site.namespace,
                node=node,
                image=host_probe_image,
                case_id="GF-REGIONAL-AUTH-013",
                run_id=f"auth013-{int(time.time())}",
                probe_script=AUTH013_PROBE,
                active_deadline_seconds=900,
            )
        )
        try:
            probe.create()
            alert = probe.execute("--timer", CERTIFICATE_CHECK_TIMER)
        finally:
            try:
                alert_residuals = probe.cleanup()
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                alert_residuals = {"cleanup_error": f"{type(exc).__name__}: {exc}"}
    alert_checks = certificate_alert_checks(alert, threshold_days=threshold)
    remaining = result.get("remaining_days")
    checks: dict[str, Any] = {
        "scoped_private_ca": (
            result["ca_exists"]
            and not result["ssl_cert_file_set"]
            and not result["requests_ca_bundle_set"]
        ),
        "default_sni_handshake_succeeds": result["default_handshake_ok"] is True,
        "certificate_san_matches_nlb": result["hostname_in_san"],
        "empty_ca_rejected": result["empty_ca_rejected"],
        "wrong_hostname_rejected": result["wrong_hostname_rejected"],
        "remaining_validity_exceeds_threshold": (
            isinstance(remaining, (int, float)) and remaining > threshold
        ),
    }
    not_evaluated: dict[str, str] = {}
    for name, value in alert_checks.items():
        if value == "NOT_EVALUATED":
            not_evaluated[name] = (
                "no --node/--host-probe-image given: the per-node expiry timer "
                "was not read, so the alert cannot be claimed as configured"
            )
            checks[name] = False
        else:
            checks[name] = value
    if node and host_probe_image:
        checks["alert_probe_removed"] = bool(alert_residuals) and not any(
            alert_residuals.values()
        )
    outcome = {
        "verdict": verdict(checks),
        "checks": checks,
        "not_evaluated": not_evaluated,
        "certificate": {
            "host": result["host"],
            "sans": result["sans"],
            "not_after": result["not_after"],
            "remaining_days": remaining,
            "configured_minimum_days": threshold,
            "default_handshake_error": result.get("default_handshake_error"),
            "wrong_hostname_error": result.get("wrong_hostname_error"),
        },
        "expiry_alert": alert,
        "alert_probe_residuals": alert_residuals,
        "limitations": [
            "The negative probes perform TLS handshakes only and never disable "
            "verification in the running executor.",
            "The expiry alert is the per-node gpu-fault-certificate-check timer; "
            "it is read on the one node named by --node.",
        ],
    }
    if case_dir is not None:
        write_json_atomic(case_dir / "auth013-details.json", outcome)
    return outcome


def describe_all_load_balancers(region: str) -> list[dict[str, Any]]:
    """Every ELBv2 load balancer in ``region``, following ``NextMarker``.

    ``describe-load-balancers`` pages at 400; an account whose NLB is past the
    first page made the single call select nothing.
    """

    result: list[dict[str, Any]] = []
    marker: str | None = None
    while True:
        command = [
            "aws",
            "elbv2",
            "describe-load-balancers",
            "--region",
            region,
            "--output",
            "json",
        ]
        if marker:
            command.extend(["--marker", marker])
        page = json.loads(run(command, timeout=120).stdout)
        result.extend(page.get("LoadBalancers") or [])
        marker = page.get("NextMarker")
        if not marker:
            return result


def select_load_balancer(
    load_balancers: list[dict[str, Any]],
    hostname: str,
) -> dict[str, Any]:
    for item in load_balancers:
        if str(item.get("DNSName") or "").lower() == hostname.lower():
            return item
    raise IdentityAcceptanceError(
        f"no ELBv2 load balancer has DNS name {hostname!r} "
        f"({len(load_balancers)} load balancers listed)"
    )


def nlb_security_groups(site: IdentitySite) -> dict[str, Any]:
    service = json.loads(
        site.cpu(
            "get",
            "service",
            "gpu-fault-api-nlb",
            "-o",
            "json",
        )
    )
    hostname = str(
        service.get("status", {})
        .get("loadBalancer", {})
        .get("ingress", [{}])[0]
        .get("hostname", "")
    )
    if not hostname:
        raise IdentityAcceptanceError("control-plane NLB has no hostname")
    selected = select_load_balancer(describe_all_load_balancers(site.region), hostname)
    group_ids = list(selected.get("SecurityGroups") or [])
    groups = (
        json.loads(
            run(
                [
                    "aws",
                    "ec2",
                    "describe-security-groups",
                    "--region",
                    site.region,
                    "--group-ids",
                    *group_ids,
                    "--output",
                    "json",
                ],
                timeout=120,
            ).stdout
        )["SecurityGroups"]
        if group_ids
        else []
    )
    broad = world_open_rules(groups)
    return {
        "hostname": hostname,
        "scheme": selected.get("Scheme"),
        "security_group_count": len(group_ids),
        "broad_ipv4_rules": [item for item in broad if "0.0.0.0/0" in item["sources"]],
        "broad_ipv6_rules": [item for item in broad if "::/0" in item["sources"]],
        "world_open_rules": broad,
    }


OUTSIDE_PROBE_MAX_AGE = timedelta(hours=24)


def outside_probe(
    path: Path | None,
    *,
    nlb_hostname: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate the outside-VPC connection-denial evidence.

    The file is produced by a separately executed probe; before it can stand
    for "the NLB is unreachable from outside", it has to name this site's NLB
    (``target_host``), be recent (``observed_at`` within 24 h) and be
    fingerprinted (``sha256``) so the auditor can tell which file was judged.
    """

    if path is None:
        return {
            "valid": False,
            "error": "outside-VPC probe evidence is required",
        }
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"valid": False, "error": f"outside probe is not JSON: {exc}"}
    if not isinstance(value, dict):
        return {"valid": False, "error": "outside probe is not an object"}
    errors: list[str] = []
    connection_blocked = value.get("connection_blocked") is True
    if not connection_blocked:
        errors.append("connection_blocked is not true")
    target_host = str(value.get("target_host") or "")
    if not nlb_hostname:
        errors.append("NLB hostname unknown; target_host not verified")
    elif target_host.lower() != nlb_hostname.lower():
        errors.append("target_host does not name this site's NLB")
    observed_at = str(value.get("observed_at") or "")
    observed: datetime | None = None
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        errors.append("observed_at is not ISO-8601")
    if observed is not None:
        if observed.tzinfo is None:
            errors.append("observed_at has no timezone")
        else:
            current = now or datetime.now(timezone.utc)
            age = current - observed
            if age > OUTSIDE_PROBE_MAX_AGE or age < -timedelta(minutes=5):
                errors.append("observed_at is not within the last 24 hours")
    return {
        "valid": not errors,
        "errors": errors,
        "connection_blocked": connection_blocked,
        "target_host": target_host,
        "observed_at": observed_at,
        "probe_location": str(value.get("probe_location") or "external"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
    }


def run_auth014(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    outside_probe_path: Path | None,
    secondary: ClusterTarget | None = None,
) -> dict[str, Any]:
    routes = route_inventory(site, target)
    results = anonymous_routes(site, target, routes)
    nlb = nlb_security_groups(site)
    external = outside_probe(outside_probe_path, nlb_hostname=str(nlb["hostname"]))
    token_hits = data_plane_execution_token_hits(site, target)
    # ``secondary`` is the unregistered peer a one-cluster site names for the
    # fleet-read scope proof; a multi-cluster site's registered peers suffice.
    authenticated_fleet = authenticated_fleet_isolation(site, target, peer=secondary)
    expected = {
        "cluster-token": {401},
        "dual-credential": {403},
        "execution-token": {403},
        "metrics": {403},
        "public": {200},
    }
    # The OpenAPI surface is recorded, not judged: it has no bucket.
    documented = [item for item in results if item["bucket"] != "public-undocumented"]
    openapi_surface = {
        f"{item['method']} {item['path']}": item.get("status", item.get("error"))
        for item in results
        if item["bucket"] == "public-undocumented"
    }
    mismatches = [
        item
        for item in documented
        if item.get("status") not in expected.get(item["bucket"], set())
    ]
    anonymous_write_success = [
        item
        for item in results
        if item["method"] in WRITE_METHODS
        and isinstance(item.get("status"), int)
        and 200 <= int(item["status"]) < 300
    ]
    # Keyed by "METHOD path": the evidence file is JSON and a tuple key made
    # write_json_atomic raise TypeError after every probe had passed (live).
    high_risk = {
        f"{item['method']} {item['path']}": item.get("status")
        for item in results
        if item["path"] in HIGH_RISK_ROUTE_BUCKETS
    }
    bucket_errors = high_risk_route_errors(routes)
    checks = {
        "route_matrix_matches_buckets": not mismatches,
        "route_matrix_complete": {
            (item["path"], method, item["bucket"])
            for item in routes
            for method in item["methods"]
            if method not in {"HEAD", "OPTIONS"}
        }
        == {(item["path"], item["method"], item["bucket"]) for item in results},
        "write_routes_explicitly_protected": all(
            item.get("bucket")
            in {"cluster-token", "dual-credential", "execution-token"}
            for item in routes
            if set(item.get("methods") or []) & WRITE_METHODS
        ),
        "anonymous_write_routes_never_succeed": not anonymous_write_success,
        "nlb_has_security_groups": nlb["security_group_count"] > 0,
        "nlb_has_no_world_open_ipv4_rule": not nlb["broad_ipv4_rules"],
        "nlb_has_no_world_open_ipv6_rule": not nlb["broad_ipv6_rules"],
        "outside_vpc_connection_blocked": external["valid"],
        "high_risk_routes_in_declared_buckets": not bucket_errors,
        "openapi_surface_recorded": len(openapi_surface) >= 3,
        "execution_token_absent_from_data_plane": not token_hits,
        "authenticated_fleet_isolation": authenticated_fleet["passed"] is True,
    }
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "nlb": nlb,
        "outside_probe": external,
        "route_results": results,
        "mismatches": mismatches,
        "anonymous_write_success": anonymous_write_success,
        "high_risk_routes": high_risk,
        "high_risk_bucket_errors": bucket_errors,
        "openapi_surface": openapi_surface,
        "execution_token_hits": token_hits,
        "authenticated_fleet": authenticated_fleet,
        "limitations": [
            "The network denial is supplied by a separately executed probe "
            "outside the NLB allowlist; this runner validates its structured "
            "evidence (target host, age, digest).",
            "The execution-token sweep compares SHA-256 digests of every "
            "data-plane Secret value and literal Pod env value against the "
            "token as the API Pod holds it (AUTH-010 folded in here).",
        ],
    }


def restore_secret(
    site: IdentitySite,
    plane: str,
    target: ClusterTarget,
    value: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> None:
    current = secret_document(site, plane, target, value["metadata"]["name"])
    metadata = current.get("metadata") or {}
    if (
        not value["metadata"].get("uid")
        or metadata.get("uid") != value["metadata"]["uid"]
        or metadata.get("namespace") != value["metadata"].get("namespace")
        or not metadata.get("resourceVersion")
    ):
        raise IdentityAcceptanceError(
            "node-key Secret identity changed; restore refused"
        )
    if current.get("data") == value.get("data"):
        return
    if expected is None or current.get("data") != expected.get("data"):
        raise IdentityAcceptanceError(
            "node-key rotation ownership is unproven; restore deferred"
        )
    site.regional(target).kubectl(
        plane,
        "patch",
        "secret",
        metadata["name"],
        "--type=json",
        "--patch-file=/dev/stdin",
        input_text=json.dumps(
            [
                {"op": "test", "path": "/metadata/uid", "value": metadata["uid"]},
                {
                    "op": "test",
                    "path": "/metadata/resourceVersion",
                    "value": metadata["resourceVersion"],
                },
                {"op": "test", "path": "/data", "value": current["data"]},
                {"op": "replace", "path": "/data", "value": value["data"]},
            ]
        ),
    )


def gpu_master_reference_scan(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    resources = json.loads(
        site.gpu(
            target,
            "get",
            "job,pod,deployment",
            "-o",
            "json",
        )
    )
    return master_reference_scan(resources)


def auth015_focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/fleet/test_fleet.py::test_derived_node_key_cannot_sign_for_another_node",
        "tests/fleet/test_fleet.py::"
        "test_node_specific_key_can_rotate_without_changing_peer",
        "tests/admin/test_node_key_custody_openssl.py::"
        "test_real_openssl_verifies_an_independently_signed_receipt_and_rejects_forgery",
        "tests/deploy/test_node_key_custody_provisioning.py::"
        "test_real_provisioning_captures_signed_prospective_custody_and_rotation",
        "tests/regional/test_auth015_custody_activation.py::"
        "test_prospective_install_rotate_and_independent_runtime_witness_chain",
    ]
    completed = run(command, check=False, timeout=600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def prepare_auth015(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    fleet_master_file: Path,
    host_probe_image: str,
    case_dir: Path,
) -> Auth015Context:
    if len(set(nodes)) != 2 or not all(nodes):
        raise IdentityAcceptanceError("AUTH-015 requires two distinct nodes")
    if fleet_master_file.stat().st_mode & 0o077:
        raise IdentityAcceptanceError("fleet master file must be mode 0600")
    master = fleet_master_file.read_text(encoding="utf-8").strip()
    if len(master) < 32:
        raise IdentityAcceptanceError("staging fleet master is too short")
    master_sha256 = secret_digest(master)
    regional = site.regional(target)
    original_gpu = secret_document(site, "gpu", target, NODE_ACTION_KEYS_SECRET)
    original_cpu = secret_document(site, "cpu", target, NODE_ACTION_KEYS_SECRET)
    before_keys = node_key_digests(original_gpu)
    before_cpu_keys = node_key_digests(original_cpu)
    if any(before_cpu_keys.get(node) != digest for node, digest in before_keys.items()):
        raise IdentityAcceptanceError("CPU/GPU node-action keys do not agree")
    reference_before = gpu_master_reference_scan(site, target)
    if not reference_before["installer_resources"] or reference_before["hits"]:
        raise IdentityAcceptanceError(
            "installer master-reference proof is missing or unsafe"
        )
    # provision-node-action-keys.sh derives a key for every GPU node missing
    # from the Secret. If the Secret does not already cover the node set, the
    # run adds keys for nodes the case never named and the "only node A
    # changed" reading is wrong before it starts.
    gpu_node_names = sorted(str(item["name"]) for item in regional.gpu_nodes())
    if sorted(before_keys) != gpu_node_names:
        raise IdentityAcceptanceError(
            "node-action-keys Secret does not cover exactly the GPU node set: "
            f"secret={sorted(before_keys)} nodes={gpu_node_names}"
        )
    if any(node not in before_keys for node in nodes):
        raise IdentityAcceptanceError("a target node has no key in the Secret")
    before_agents = regional.cpu_python(AGENT_SNAPSHOT_PROBE, target.cluster_id, *nodes)
    probes = [
        HostProbeFixture(
            HostProbeSettings(
                state_directory=case_dir / "host-probes",
                kubeconfig=site.gpu_kubeconfig,
                context=target.context,
                namespace=site.namespace,
                node=node,
                image=host_probe_image,
                case_id="GF-REGIONAL-AUTH-015",
                run_id=f"auth015-{index}-{int(time.time())}",
                probe_script=AUTH015_PROBE,
                active_deadline_seconds=1800,
            )
        )
        for index, node in enumerate(nodes)
    ]
    return Auth015Context(
        master_sha256=master_sha256,
        regional=regional,
        original_gpu=original_gpu,
        original_cpu=original_cpu,
        before_keys=before_keys,
        before_cpu_keys=before_cpu_keys,
        gpu_node_names=gpu_node_names,
        before_agents=before_agents,
        probes=probes,
    )


def run_auth015(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    fleet_master_file: Path | None = None,
    host_probe_image: str = "",
    case_dir: Path,
    focused_tests: dict[str, Any] | None = None,
    release_inputs: Auth015ReleaseInputs | None = None,
    custody_inputs: Auth015CustodyInputs | None = None,
) -> dict[str, Any]:
    # Compatibility arguments are never opened. A current master/Secret snapshot
    # cannot authorize a historical claim or a new test-only key mutation.
    del fleet_master_file, host_probe_image
    if custody_inputs is None or release_inputs is None:
        return {
            "verdict": "FAIL",
            "checks": {"no_node_or_secret_mutation": True},
            "not_evaluated": dict(AUTH015_PROOF_GAPS),
            "requires_new_authorized_evidence": True,
            "cleanup_errors": [],
        }
    if focused_tests is not None and focused_tests.get("passed") is not True:
        raise IdentityAcceptanceError("AUTH-015 focused signature tests failed")
    return run_custody_acceptance(
        site,
        target,
        nodes=nodes,
        release_inputs=release_inputs,
        custody_inputs=custody_inputs,
        case_dir=case_dir,
    )
