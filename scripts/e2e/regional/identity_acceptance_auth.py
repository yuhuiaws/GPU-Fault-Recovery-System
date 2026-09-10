from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings
from scripts.e2e.regional.identity_acceptance_common import (
    ACCEPTANCE_PROBE_OWNER,
    EXECUTOR_APP,
    ROOT,
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
    utc_now,
    write_cluster_token,
)

AUTH015_PROBE = Path(__file__).with_name("probes") / "auth015_node_probe.py"
AUTH013_PROBE = Path(__file__).with_name("probes") / "auth013_certificate_probe.py"
NODE_ACTION_KEYS_SECRET = "gpu-fault-node-action-keys"
INSTALLER_SECRET = "gpu-fault-control-plane-active"
# How long AUTH-016 waits for its sampler thread after asking it to stop. A
# sample is one ``direct_claim``: the executor Pod lookup (``kubectl get``,
# default 300 s bound) plus one exec bounded at 60 s. The old 15 s join let a
# sampler mid-claim outlive the restore and read the restored token as a
# "completed"-phase 200.
DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS = 60
DIRECT_CLAIM_JOIN_SECONDS = 300 + DIRECT_CLAIM_EXEC_TIMEOUT_SECONDS + 15


def verdict(checks: dict[str, Any]) -> str:
    """PASS only when every check is literally ``True``.

    ``all(checks.values())`` accepted any truthy value, so a check recorded as
    the string ``"NOT_EVALUATED"`` passed. Checks that could not be evaluated
    belong in a separate ``not_evaluated`` mapping, not here.
    """

    return "PASS" if all(value is True for value in checks.values()) else "FAIL"


def failure_details(
    *,
    checks: dict[str, Any],
    cleanup_errors: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {"checks": checks, "cleanup_errors": cleanup_errors, **extra}


def run_auth007(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    original = site.registry()
    updated = [dict(item) for item in original]
    found = False
    for item in updated:
        if item.get("cluster_id") == secondary.cluster_id:
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
        site.write_registry(updated)
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
                ("restore_registry", lambda: site.write_registry(original)),
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
    checks = {
        "secondary_disabled_403": secondary_disabled.get("status") == 403,
        "primary_remains_200": primary_healthy.get("status") == 200,
        "secondary_recovers_200": secondary_recovered.get("status") == 200,
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


ROUTE_INVENTORY_PROBE = r"""
import json
from gpu_fault.app import create_app
from gpu_fault.app.authorization import (
    UNDOCUMENTED_PUBLIC_PATHS,
    ExplicitAuthorizationRegistry,
    iter_api_routes,
)

app = create_app()
registry = ExplicitAuthorizationRegistry()
registry.load(app.routes)
routes = []
for route in iter_api_routes(app.routes):
    if not (
        route.path.startswith("/v1/")
        or route.path in {"/healthz", "/metrics"}
    ):
        continue
    routes.append({
        "path": route.path,
        "methods": sorted(route.methods or []),
        "bucket": registry.inventory[route.path],
    })
# The OpenAPI surface has no bucket: it is public read-only information that
# AUTH-014 records (anonymous status) rather than judges against a bucket.
for path in sorted(UNDOCUMENTED_PUBLIC_PATHS):
    routes.append({"path": path, "methods": ["GET"], "bucket": "public-undocumented"})
print(json.dumps({"routes": routes}, sort_keys=True))
"""

EXECUTION_TOKEN_DIGEST_PROBE = r"""
import hashlib
import json
import os
token = os.environ["GPU_FAULT_EXECUTION_TOKEN"]
print(json.dumps({
    "sha256": hashlib.sha256(token.encode()).hexdigest(),
    "stripped_sha256": hashlib.sha256(token.strip().encode()).hexdigest(),
}))
"""

ANONYMOUS_ROUTE_PROBE = r"""
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

routes = json.loads(sys.argv[1])
base = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
results = []
for item in routes:
    path = re.sub(r"\{[^}]+\}", "probe", item["path"])
    method = item["method"]
    request = urllib.request.Request(
        base + path,
        data=(b"{}" if method in {"POST", "PUT", "PATCH", "DELETE"} else None),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, context=context, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:
        results.append({
            "path": item["path"],
            "method": method,
            "bucket": item["bucket"],
            "error": type(exc).__name__,
        })
        continue
    results.append({
        "path": item["path"],
        "method": method,
        "bucket": item["bucket"],
        "status": status,
    })
print(json.dumps({"results": results}, sort_keys=True))
"""


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


def execution_token_hits(
    secrets_document: dict[str, Any],
    pods_document: dict[str, Any],
    *,
    digests: set[str],
) -> list[dict[str, str]]:
    """Where the data plane holds the execution token, by name or by value.

    ``digests`` are the SHA-256 of the token as the API Pod holds it (raw and
    stripped); every Secret value and every literal Pod env value is digested
    and compared, so a token stored under an innocent key is found too. Only
    names and keys are returned, never values.
    """

    matches = []
    pattern = re.compile(r"execution[_.-]?token", re.IGNORECASE)

    def value_digests(raw: bytes) -> set[str]:
        return {
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.strip()).hexdigest(),
        }

    for item in secrets_document.get("items", []):
        name = str(item.get("metadata", {}).get("name", ""))
        for key, encoded in (item.get("data") or {}).items():
            try:
                raw = base64.b64decode(str(encoded))
            except (ValueError, TypeError):
                raw = b""
            by_value = bool(value_digests(raw) & digests) if raw else False
            by_name = pattern.search(str(key)) is not None
            if by_value or by_name:
                matches.append(
                    {
                        "kind": "Secret",
                        "name": name,
                        "key": str(key),
                        "match": "value" if by_value else "name",
                    }
                )
    for item in pods_document.get("items", []):
        name = str(item.get("metadata", {}).get("name", ""))
        spec = item.get("spec", {})
        containers = [*spec.get("initContainers", []), *spec.get("containers", [])]
        for container in containers:
            for entry in container.get("env", []):
                literal = entry.get("value")
                by_value = isinstance(literal, str) and bool(
                    value_digests(literal.encode()) & digests
                )
                by_name = pattern.search(str(entry.get("name", ""))) is not None
                if by_value or by_name:
                    matches.append(
                        {
                            "kind": "Pod",
                            "name": name,
                            "key": str(entry.get("name")),
                            "match": "value" if by_value else "name",
                        }
                    )
    return matches


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
    secrets_document = json.loads(site.gpu(target, "get", "secret", "-o", "json"))
    pods_document = json.loads(site.gpu(target, "get", "pod", "-o", "json"))
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


REMOTE_STATUS_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
# argv: command IDs whose status must be reported even once terminal, so the
# "after" reading can tell SUCCEEDED (not interrupted) from a vanished row.
tracked = set(sys.argv[1:])
commands = ApplicationContext.from_environment().store.list_remote_commands()
print(json.dumps({
    "commands": [
        {
            "command_id": item.command_id,
            "status": item.status.value,
            "lease_owner_present": item.lease_owner is not None,
        }
        for item in commands
        if item.status.value in {"PENDING", "WAITING", "LEASED"}
        or item.command_id in tracked
    ]
}, sort_keys=True))
"""


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


DIRECT_CLAIM_PROBE_TEMPLATE = r"""
import json
import os
import ssl
import urllib.error
import urllib.request
payload = __PAYLOAD__
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/v1/regional/executors/claim",
    data=json.dumps(payload, separators=(",", ":")).encode(),
    headers={
        "Authorization": "Bearer " + __TOKEN__,
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": __CLUSTER_ID__,
    },
    method="POST",
)
context = ssl.create_default_context(cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"])
try:
    with urllib.request.urlopen(request, context=context, timeout=10) as response:
        print(json.dumps({"status": int(response.status)}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"status": exc.code}))
except Exception as exc:
    print(json.dumps({"status": type(exc).__name__}))
"""


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


def registry_token_digests(
    entries: list[dict[str, Any]],
    cluster_id: str,
) -> tuple[str | None, str | None, str | None]:
    """(token digest, retiring digest, deadline) for one cluster.

    The bootstrap Secret carries plaintext tokens and a durable revision carries
    digests; comparing digests lets a restore be checked on either kind of site
    without handling the plaintext again.
    """

    for item in entries:
        if item.get("cluster_id") != cluster_id:
            continue
        token = item.get("token")
        retiring = item.get("retiring_token")
        return (
            secret_digest(str(token)) if token else item.get("token_sha256"),
            (
                secret_digest(str(retiring))
                if retiring
                else item.get("retiring_token_sha256")
            ),
            item.get("token_rotation_expires_at"),
        )
    raise IdentityAcceptanceError("target cluster is absent from registry")


def update_registry_token(
    entries: list[dict[str, Any]],
    cluster_id: str,
    token: str,
    *,
    retiring_token: str | None = None,
    rotation_expires_at: str | None = None,
) -> list[dict[str, Any]]:
    """Return the registry with one cluster's token, and rotation slot, replaced.

    Both plaintext fields are handed to the control plane, which stores only the
    digests; the driver never writes a token into evidence.
    """

    result = [dict(item) for item in entries]
    for item in result:
        if item.get("cluster_id") != cluster_id:
            continue
        item["token"] = token
        item.pop("token_sha256", None)
        item.pop("retiring_token", None)
        item.pop("retiring_token_sha256", None)
        item.pop("token_rotation_expires_at", None)
        if retiring_token is not None:
            item["retiring_token"] = retiring_token
            item["token_rotation_expires_at"] = rotation_expires_at
        return result
    raise IdentityAcceptanceError("target cluster is absent from registry")


def commands_not_misterminated(
    before_status: dict[str, str],
    after_status: dict[str, str | None],
) -> bool:
    """Whether every command open before the rotation survived it.

    An empty baseline used to pass vacuously (``all`` over nothing), so the
    check said "no command was misterminated" on a site that had no command to
    misterminate. It now needs at least one open command before the rotation.
    A command that ran to SUCCEEDED during the window was not interrupted; one
    that vanished (the status probe lists open commands only) or FAILED was.
    """

    if not before_status:
        return False
    return all(
        after_status.get(command_id) in {"PENDING", "WAITING", "LEASED", "SUCCEEDED"}
        for command_id in before_status
    )


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
    # Only HTTP answers speak to authentication; a sample the probe could not
    # take (no Ready executor while the data plane rolls) is recorded but does
    # not decide a phase. A phase still needs at least one real answer.
    http_by_phase = {
        phase: [value for value in values if isinstance(value, int)]
        for phase, values in by_phase.items()
    }
    uninterrupted = [
        phase
        for phase in ("baseline", "overlap", "cutover")
        if http_by_phase.get(phase) and set(http_by_phase[phase]) == {200}
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
            }
            for item in samples
        ],
        "old_token_sha256": secret_digest(old_token),
        "new_token_sha256": secret_digest(new_token),
        "limitations": [
            "The retiring token stays valid for the whole overlap window, so this "
            "case proves there is no interruption, not that the old credential is "
            "revoked instantly; both original Secrets are restored.",
            "remote_commands_not_misterminated needs at least one command open "
            "before the rotation; on an idle site the check fails rather than "
            "passing over an empty baseline.",
        ],
    }


def run_auth016(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    """Prove the overlap window rotates a cluster token without any 403.

    The old token stays valid until an explicit deadline, so the control plane
    can accept the new credential before the data plane presents it. What has to
    be proven is both halves of that: no rejection while both slots are live,
    and an immediate rejection once the retiring slot is dropped.
    """

    original_registry = site.registry()
    old_token = read_cluster_token(site, target)
    new_token = secrets.token_urlsafe(48)
    identity = executor_claim_identity(site, target)
    primary = site.regional(target)
    before_commands = primary.cpu_python(REMOTE_STATUS_PROBE)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=30)
    token_lock = threading.Lock()
    token_box = [old_token]
    phase_box = ["baseline"]
    stop = threading.Event()
    samples: list[dict[str, Any]] = []

    def sampler() -> None:
        while not stop.is_set():
            with token_lock:
                token = token_box[0]
                phase = phase_box[0]
            # Re-check after the wait: a stop that arrived during the pause
            # must not be answered with one more claim, which would run while
            # the restore is republishing the original token.
            if stop.is_set():
                return
            samples.append(
                {
                    "observed_at": utc_now(),
                    "phase": phase,
                    "status": direct_claim(
                        primary, target, token=token, identity=identity
                    ),
                }
            )
            stop.wait(2)

    def enter(phase: str, *, token: str | None = None) -> None:
        with token_lock:
            phase_box[0] = phase
            if token is not None:
                token_box[0] = token

    thread = threading.Thread(target=sampler, daemon=True)
    restored = False
    control_rollout: float | None = None
    executor_rollout = None
    new_token_during_overlap: int | str | None = None
    old_token_after_completion: int | str | None = None
    registry_generation = site.registry_generation()
    cleanup_errors: list[str] = []
    cleanup: dict[str, Any] = {}

    def partial() -> dict[str, Any]:
        return {
            "samples": samples,
            "new_token_during_overlap": new_token_during_overlap,
            "old_token_after_completion": old_token_after_completion,
            "cleanup": cleanup,
            "cleanup_errors": cleanup_errors,
            "restored": restored,
        }

    try:
        thread.start()
        time.sleep(60)
        # Control plane first: both slots live, data plane untouched.
        site.write_registry(
            update_registry_token(
                original_registry,
                target.cluster_id,
                new_token,
                retiring_token=old_token,
                rotation_expires_at=deadline.isoformat().replace("+00:00", "Z"),
            ),
            reason="GF-REGIONAL-AUTH-016 overlap: new token live, old token retiring",
        )
        # POST-to-last-ack propagation time; rollout_control is a no-op on a
        # durable-revision site and would have measured ~1 s.
        control_rollout = site.last_registry_ready_seconds or site.rollout_control()
        enter("overlap")
        time.sleep(30)
        new_token_during_overlap = direct_claim(
            primary,
            target,
            token=new_token,
            identity=identity,
        )
        write_cluster_token(site, target, new_token)
        executor_rollout = rollout_executor(site, target)
        enter("cutover", token=new_token)
        time.sleep(30)
        # Finishing the rotation must withdraw the old credential at once
        # instead of leaving it live until the deadline lapses.
        site.write_registry(
            update_registry_token(original_registry, target.cluster_id, new_token),
            reason="GF-REGIONAL-AUTH-016 completed: retiring slot withdrawn",
        )
        site.rollout_control()
        enter("completed")
        old_token_after_completion = direct_claim(
            primary,
            target,
            token=old_token,
            identity=identity,
        )
        time.sleep(10)
    except Exception as exc:
        raise IdentityCaseFailure(str(exc), details=partial()) from exc
    finally:
        stop.set()
        # The sampler's worst case is one whole direct_claim; a shorter join
        # let a claim in flight land after the restore below.
        thread.join(timeout=DIRECT_CLAIM_JOIN_SECONDS)
        if thread.is_alive():
            cleanup_errors.append("sampler thread did not stop before restore")

        def verify_restored() -> bool:
            return registry_token_digests(
                site.registry(), target.cluster_id
            ) == registry_token_digests(
                original_registry, target.cluster_id
            ) and secret_digest(read_cluster_token(site, target)) == secret_digest(
                old_token
            )

        # In place: the IdentityCaseFailure raised above holds these containers.
        step_outcomes, step_errors = run_cleanup_steps(
            [
                (
                    "restore_registry",
                    lambda: site.write_registry(
                        original_registry,
                        reason=(
                            "GF-REGIONAL-AUTH-016 restore: original token republished"
                        ),
                    ),
                ),
                ("rollout_control", site.rollout_control),
                (
                    "restore_cluster_token",
                    lambda: write_cluster_token(site, target, old_token),
                ),
                ("rollout_executor", lambda: rollout_executor(site, target)),
                ("verify_restored", verify_restored),
            ]
        )
        cleanup.update(step_outcomes)
        cleanup_errors.extend(step_errors)
        restored = cleanup.get("verify_restored") is True and not cleanup_errors
        write_json_atomic(case_dir / "auth016-details.json", partial())
    if thread.is_alive():
        raise IdentityCaseFailure(
            "sampler thread outlived the restore; samples cannot be attributed",
            details=partial(),
        )
    return auth016_result(
        site,
        primary,
        samples=samples,
        before_commands=before_commands,
        new_token_during_overlap=new_token_during_overlap,
        old_token_after_completion=old_token_after_completion,
        restored=restored,
        registry_generation=registry_generation,
        control_rollout=control_rollout,
        executor_rollout=executor_rollout,
        old_token=old_token,
        new_token=new_token,
    )


TLS_BOUNDARY_PROBE = r"""
import json
import os
import socket
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

url = urlsplit(os.environ["GPU_FAULT_CONTROL_PLANE_URL"])
host = url.hostname
ca_file = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
context = ssl.create_default_context(cafile=ca_file)
certificate = {}
default_handshake_ok = False
default_handshake_error = None
try:
    with socket.create_connection((host, url.port or 443), timeout=15) as raw:
        with context.wrap_socket(raw, server_hostname=host) as tls:
            certificate = tls.getpeercert() or {}
            default_handshake_ok = True
except (ssl.SSLError, OSError) as exc:
    default_handshake_error = type(exc).__name__

empty_ca_rejected = False
Path("/tmp/auth013-empty-ca.pem").write_text("", encoding="ascii")
try:
    ssl.create_default_context(cafile="/tmp/auth013-empty-ca.pem")
except ssl.SSLError:
    empty_ca_rejected = True

# Any TLS or socket failure is a rejection of the wrong name; the server may
# close the connection (OSError) instead of completing a handshake that then
# fails hostname verification (SSLCertVerificationError). What must not
# happen is a completed handshake.
wrong_hostname_rejected = False
wrong_hostname_error = None
try:
    with socket.create_connection((host, url.port or 443), timeout=15) as raw:
        with context.wrap_socket(raw, server_hostname="wrong.invalid"):
            pass
except (ssl.SSLError, OSError) as exc:
    wrong_hostname_rejected = True
    wrong_hostname_error = type(exc).__name__

not_after = certificate.get("notAfter")
expires = (
    datetime.fromtimestamp(ssl.cert_time_to_seconds(not_after), tz=timezone.utc)
    if not_after else None
)
sans = [
    value
    for kind, value in certificate.get("subjectAltName", [])
    if kind == "DNS"
]
print(json.dumps({
    "host": host,
    "ca_file": ca_file,
    "ca_exists": Path(ca_file).is_file(),
    "ssl_cert_file_set": bool(os.getenv("SSL_CERT_FILE")),
    "requests_ca_bundle_set": bool(os.getenv("REQUESTS_CA_BUNDLE")),
    "sans": sans,
    "hostname_in_san": host in sans,
    "default_handshake_ok": default_handshake_ok,
    "default_handshake_error": default_handshake_error,
    "empty_ca_rejected": empty_ca_rejected,
    "wrong_hostname_rejected": wrong_hostname_rejected,
    "wrong_hostname_error": wrong_hostname_error,
    "not_after": expires.isoformat() if expires else None,
    "remaining_days": (
        (expires - datetime.now(timezone.utc)).total_seconds() / 86400
        if expires else None
    ),
}, sort_keys=True))
"""

CERTIFICATE_CHECK_TIMER = "gpu-fault-certificate-check.timer"


def certificate_alert_checks(
    alert: dict[str, Any] | None,
    *,
    threshold_days: int,
) -> dict[str, Any]:
    """Whether the expiry alert the deploy ships is armed on a GPU node.

    The only certificate-expiry alerting in ``deploy/`` is the per-node
    ``gpu-fault-certificate-check.timer`` running
    ``check-control-plane-certificate`` with
    ``GPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS`` from ``collector.env``; there
    is no PrometheusRule for it. ``threshold >= 30`` read the site file, not the
    node, so it passed with the timer disabled. ``alert`` is the host probe's
    reading; ``None`` means no node was given and the check is not evaluated.
    """

    if alert is None:
        return {"expiry_threshold_configured": "NOT_EVALUATED"}
    seconds = alert.get("min_validity_seconds")
    configured = (
        isinstance(seconds, int)
        and seconds >= threshold_days * 86400
        and alert.get("timer_enabled") is True
        and alert.get("timer_active") is True
    )
    return {"expiry_threshold_configured": configured}


def run_auth013(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    node: str = "",
    host_probe_image: str = "",
    case_dir: Path | None = None,
) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    result = site.pod_json("gpu", target, pod, TLS_BOUNDARY_PROBE)
    threshold = int(site.config["health"]["certificate_min_validity_days"])
    alert: dict[str, Any] | None = None
    alert_residuals: dict[str, Any] = {}
    if node and host_probe_image:
        probe = HostProbeFixture(
            HostProbeSettings(
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
    if alert_residuals and any(
        value for key, value in alert_residuals.items() if key != "cleanup_error"
    ):
        checks["alert_probe_removed"] = False
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


def world_open_rules(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ingress permissions open to every IPv4 or IPv6 source."""

    broad = []
    for group in groups:
        for permission in group.get("IpPermissions") or []:
            sources = [
                item.get("CidrIp")
                for item in permission.get("IpRanges") or []
                if item.get("CidrIp") == "0.0.0.0/0"
            ] + [
                item.get("CidrIpv6")
                for item in permission.get("Ipv6Ranges") or []
                if item.get("CidrIpv6") == "::/0"
            ]
            if sources:
                broad.append(
                    {
                        "group_id": group["GroupId"],
                        "from_port": permission.get("FromPort"),
                        "to_port": permission.get("ToPort"),
                        "sources": sources,
                    }
                )
    return broad


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


HIGH_RISK_ROUTE_BUCKETS = {
    "/v1/runtime-profiles": "execution-token",
    "/v1/advisory-notifications/{notification_id}/send": "execution-token",
    "/v1/fleet/agents": "cluster-token",
}


def high_risk_route_errors(routes: list[dict[str, Any]]) -> list[str]:
    """The three routes whose bucket the catalog names, checked, not counted."""

    buckets = {str(item["path"]): item.get("bucket") for item in routes}
    errors = []
    for path, expected in HIGH_RISK_ROUTE_BUCKETS.items():
        actual = buckets.get(path)
        if actual is None:
            errors.append(f"{path} is not in the route inventory")
        elif actual != expected:
            errors.append(f"{path} is {actual}, expected {expected}")
    return errors


def run_auth014(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    outside_probe_path: Path | None,
) -> dict[str, Any]:
    routes = route_inventory(site, target)
    results = anonymous_routes(site, target, routes)
    nlb = nlb_security_groups(site)
    external = outside_probe(outside_probe_path, nlb_hostname=str(nlb["hostname"]))
    token_hits = data_plane_execution_token_hits(site, target)
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
        "anonymous_write_routes_never_succeed": not anonymous_write_success,
        "nlb_has_no_world_open_ipv4_rule": not nlb["broad_ipv4_rules"],
        "nlb_has_no_world_open_ipv6_rule": not nlb["broad_ipv6_rules"],
        "outside_vpc_connection_blocked": external["valid"],
        "high_risk_routes_in_declared_buckets": not bucket_errors,
        "openapi_surface_recorded": len(openapi_surface) >= 3,
        "execution_token_absent_from_data_plane": not token_hits,
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
        "limitations": [
            "The network denial is supplied by a separately executed probe "
            "outside the NLB allowlist; this runner validates its structured "
            "evidence (target host, age, digest).",
            "The execution-token sweep compares SHA-256 digests of every "
            "data-plane Secret value and literal Pod env value against the "
            "token as the API Pod holds it (AUTH-010 folded in here).",
        ],
    }


AGENT_SNAPSHOT_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
cluster_id, *nodes = sys.argv[1:]
store = ApplicationContext.from_environment().store
result = {}
for node in nodes:
    agent = store.get_agent(cluster_id, node)
    result[node] = {
        "generation": agent.generation,
        "lifecycle_state": agent.lifecycle_state.value,
        # AgentRecord renamed the heartbeat field to last_seen_at; the evidence
        # key stays for readers of earlier reports.
        "last_heartbeat_at": agent.last_seen_at.isoformat(),
        "node_action_key_version": agent.node_action_key_version,
    }
print(json.dumps({"agents": result}, sort_keys=True))
"""


def secret_document(
    site: IdentitySite,
    plane: str,
    target: ClusterTarget,
    name: str,
) -> dict[str, Any]:
    value = json.loads(
        site.regional(target).kubectl(
            plane,
            "get",
            "secret",
            name,
            "-o",
            "json",
        )
    )
    return cast(dict[str, Any], value)


def restore_secret(
    site: IdentitySite,
    plane: str,
    target: ClusterTarget,
    value: dict[str, Any],
) -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": value["metadata"]["name"],
            "namespace": value["metadata"]["namespace"],
        },
        "type": value.get("type", "Opaque"),
        "data": value.get("data") or {},
    }
    site.regional(target).kubectl(
        plane,
        "apply",
        "-f",
        "-",
        input_text=json.dumps(manifest),
    )


def node_key_digests(secret: dict[str, Any]) -> dict[str, str]:
    return {
        key: hashlib.sha256(base64.b64decode(value)).hexdigest()
        for key, value in (secret.get("data") or {}).items()
    }


def master_reference_scan(resources: dict[str, Any]) -> dict[str, Any]:
    """Which GPU-plane resources reference the fleet-master Secret.

    The check is only meaningful if an installer resource was in the scan:
    the installer Job is transient, and a scan of an idle namespace finds no
    reference because it finds no installer. ``installer_resources`` names the
    Jobs/Pods that looked like an installer so the caller can tell "clean" from
    "nothing to look at".
    """

    hits: list[dict[str, str]] = []
    installer_resources: list[str] = []
    for item in resources.get("items", []):
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        kind = str(item.get("kind") or "")
        labels = metadata.get("labels") or {}
        if "installer" in name or any("installer" in str(v) for v in labels.values()):
            installer_resources.append(f"{kind}/{name}")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            reference = value.get("secretKeyRef")
            if isinstance(reference, dict) and (
                reference.get("name") == INSTALLER_SECRET
            ):
                hits.append(
                    {
                        "path": path,
                        "secret": str(reference.get("name") or ""),
                        "key": str(reference.get("key") or ""),
                    }
                )
            secret_volume = value.get("secret")
            if isinstance(secret_volume, dict) and (
                secret_volume.get("secretName") == INSTALLER_SECRET
            ):
                hits.append(
                    {
                        "path": path,
                        "secret": str(secret_volume.get("secretName") or ""),
                        "key": "",
                    }
                )
            for key, child in value.items():
                visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(resources, "")
    return {"hits": hits, "installer_resources": sorted(installer_resources)}


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
    ]
    completed = run(command, check=False, timeout=600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def heartbeat_advanced(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Node B kept heartbeating: its last_heartbeat_at moved forward."""

    try:
        earlier = datetime.fromisoformat(str(before["last_heartbeat_at"]))
        later = datetime.fromisoformat(str(after["last_heartbeat_at"]))
    except (KeyError, ValueError, TypeError):
        return False
    return later > earlier


def run_auth015(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    fleet_master_file: Path,
    host_probe_image: str,
    case_dir: Path,
    focused_tests: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    scans_before: dict[str, Any] = {}
    scans_after: dict[str, Any] = {}
    rotation = None
    residuals: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    checks: dict[str, Any] = {}
    not_evaluated: dict[str, str] = {}
    tests = focused_tests if focused_tests is not None else auth015_focused_tests()

    def create_and_scan(probe: HostProbeFixture) -> tuple[str, dict[str, Any]]:
        probe.create()
        return probe.settings.node, probe.execute("--master-sha256", master_sha256)

    try:
        # Two privileged Pods on two nodes: creating them one after the other
        # doubled the slowest step for nothing they share.
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            scans_before = dict(pool.map(create_and_scan, probes))
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "KUBECONFIG": str(site.gpu_kubeconfig),
            "GPU_FAULT_KUBECTL_CONTEXT": target.context,
            "GPU_FAULT_NAMESPACE": site.namespace,
            "GPU_FAULT_CLUSTER_ID": target.cluster_id,
            "GPU_FAULT_HYPERPOD_CLUSTER": target.hyperpod_cluster_name,
            "GPU_FAULT_FLEET_MASTER_FILE": str(fleet_master_file.resolve()),
            "GPU_FAULT_ROTATE_NODE_ACTION_KEY": nodes[0],
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(site.cpu_kubeconfig),
            "GPU_FAULT_CONTROL_PLANE_NAMESPACE": site.namespace,
        }
        rotation = run(
            ["bash", "deploy/node/provision-node-action-keys.sh"],
            check=False,
            timeout=600,
            env=environment,
        )
        # Scan while the rotation's resources are freshest; the installer Job
        # is transient, and the scan says so when it saw none.
        reference_scan = gpu_master_reference_scan(site, target)
        after_gpu = secret_document(site, "gpu", target, NODE_ACTION_KEYS_SECRET)
        after_keys = node_key_digests(after_gpu)
        time.sleep(30)
        after_agents = regional.cpu_python(
            AGENT_SNAPSHOT_PROBE,
            target.cluster_id,
            *nodes,
        )
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            scans_after = dict(
                pool.map(
                    lambda probe: (
                        probe.settings.node,
                        probe.execute("--master-sha256", master_sha256),
                    ),
                    probes,
                )
            )
        checks = {
            "host_scan_before_zero_matches": all(
                not item["master_matches"] for item in scans_before.values()
            ),
            "host_scan_after_zero_matches": all(
                not item["master_matches"] for item in scans_after.values()
            ),
            "cross_node_signature_tests": tests.get("passed") is True,
            "rotation_command_succeeded": rotation.returncode == 0,
            "node_a_key_changed": before_keys.get(nodes[0]) != after_keys.get(nodes[0]),
            "node_b_key_unchanged": before_keys.get(nodes[1])
            == after_keys.get(nodes[1]),
            "no_other_node_key_changed": all(
                before_keys.get(node) == after_keys.get(node)
                for node in before_keys
                if node != nodes[0]
            ),
            "node_b_agent_continues": (
                before_agents["agents"][nodes[1]]["generation"]
                == after_agents["agents"][nodes[1]]["generation"]
                and after_agents["agents"][nodes[1]]["lifecycle_state"] == "ACTIVE"
            ),
            "node_b_heartbeat_advanced": heartbeat_advanced(
                before_agents["agents"][nodes[1]], after_agents["agents"][nodes[1]]
            ),
        }
        if reference_scan["installer_resources"]:
            checks["installer_references_no_fleet_master"] = not reference_scan["hits"]
        else:
            not_evaluated["installer_references_no_fleet_master"] = (
                "no installer Job/Pod was present during the scan; the "
                "reference check had nothing to inspect"
            )
    except Exception as exc:
        raise IdentityCaseFailure(
            str(exc),
            details=failure_details(
                checks=checks,
                cleanup_errors=cleanup_errors,
                scan_before=scans_before,
                scan_after=scans_after,
                rotation_returncode=(
                    rotation.returncode if rotation is not None else None
                ),
            ),
        ) from exc
    finally:
        steps: list[tuple[str, Any]] = [
            (
                "restore_gpu_secret",
                lambda: restore_secret(site, "gpu", target, original_gpu),
            ),
            (
                "restore_cpu_secret",
                lambda: restore_secret(site, "cpu", target, original_cpu),
            ),
        ]
        for probe in probes:
            steps.append((f"cleanup_probe:{probe.settings.node}", probe.cleanup))
        # In place: the IdentityCaseFailure raised above holds cleanup_errors.
        cleanup, step_errors = run_cleanup_steps(steps)
        cleanup_errors.extend(step_errors)
        for probe in probes:
            key = f"cleanup_probe:{probe.settings.node}"
            residuals[probe.settings.node] = cleanup.get(key) or {"cleanup_error": True}
        write_json_atomic(
            case_dir / "auth015-details.json",
            {
                "checks": checks,
                "cleanup_errors": cleanup_errors,
                "scan_before": scans_before,
                "scan_after": scans_after,
                "probe_residuals": residuals,
            },
        )
    checks["gpu_secret_restored"] = (
        node_key_digests(secret_document(site, "gpu", target, NODE_ACTION_KEYS_SECRET))
        == before_keys
    )
    checks["cpu_secret_restored"] = (
        node_key_digests(secret_document(site, "cpu", target, NODE_ACTION_KEYS_SECRET))
        == before_cpu_keys
    )
    checks["probe_resources_removed"] = all(
        not any(value.values()) for value in residuals.values()
    )
    checks["cleanup_completed"] = not cleanup_errors
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "not_evaluated": not_evaluated,
        "master_sha256": master_sha256,
        "gpu_node_set": gpu_node_names,
        "installer_scan": {
            "installer_resources": reference_scan["installer_resources"],
            "hits": reference_scan["hits"],
        },
        "focused_tests": tests,
        "scan_before": scans_before,
        "scan_after": scans_after,
        "probe_residuals": residuals,
        "cleanup_errors": cleanup_errors,
        "limitations": [
            "The master is supplied from a trusted-host mode-0600 file. Only "
            "its SHA-256 enters evidence; the runner restores both key Secrets.",
            "The installer-reference scan is only evaluated when an installer "
            "Job or Pod exists during the case; otherwise it is recorded under "
            "not_evaluated.",
        ],
    }
