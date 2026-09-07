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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from scripts.e2e.regional.host_probe_fixture import (
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.identity_acceptance_common import (
    EXECUTOR_APP,
    ROOT,
    WRITE_METHODS,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentitySite,
    claim,
    read_cluster_token,
    rollout_executor,
    run,
    secret_digest,
    utc_now,
    write_cluster_token,
)

AUTH015_PROBE = Path(__file__).with_name("probes") / "auth015_node_probe.py"


def run_auth007(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
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
    disabled_latency = None
    restored_latency = None
    secondary_disabled: dict[str, Any] = {}
    primary_healthy: dict[str, Any] = {}
    secondary_recovered: dict[str, Any] = {}
    try:
        site.write_registry(updated)
        disabled_latency = site.rollout_control()
        secondary_disabled = claim(site, secondary)
        primary_healthy = claim(site, primary)
    finally:
        site.write_registry(original)
        restored_latency = site.rollout_control()
        secondary_recovered = claim(site, secondary)
    checks = {
        "secondary_disabled_403": secondary_disabled.get("status") == 403,
        "primary_remains_200": primary_healthy.get("status") == 200,
        "secondary_recovers_200": secondary_recovered.get("status") == 200,
        "registry_restored": site.registry() == original,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "isolation_latency_seconds": disabled_latency,
        "restore_latency_seconds": restored_latency,
        "limitations": [
            "The case rolls ingress and control-worker for the explicitly "
            "selected secondary test cluster; it does not disable a production "
            "training cluster."
        ],
    }


ROUTE_INVENTORY_PROBE = r"""
import json
from gpu_fault.app import create_app
from gpu_fault.app.authorization import ExplicitAuthorizationRegistry, iter_api_routes

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
print(json.dumps({"routes": routes}, sort_keys=True))
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


def data_plane_execution_token_names(
    site: IdentitySite,
    target: ClusterTarget,
) -> list[dict[str, str]]:
    secret = json.loads(site.gpu(target, "get", "secret", "-o", "json"))
    pods = json.loads(site.gpu(target, "get", "pod", "-o", "json"))
    matches = []
    pattern = re.compile(r"execution[_.-]?token", re.IGNORECASE)
    for item in secret.get("items", []):
        for key in item.get("data") or {}:
            if pattern.search(str(key)):
                matches.append(
                    {
                        "kind": "Secret",
                        "name": str(item["metadata"]["name"]),
                        "key": str(key),
                    }
                )
    for item in pods.get("items", []):
        for container in item.get("spec", {}).get("containers", []):
            for entry in container.get("env", []):
                if pattern.search(str(entry.get("name", ""))):
                    matches.append(
                        {
                            "kind": "Pod",
                            "name": str(item["metadata"]["name"]),
                            "key": str(entry.get("name")),
                        }
                    )
    return matches


def run_auth010(site: IdentitySite, target: ClusterTarget) -> dict[str, Any]:
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
    token_names = data_plane_execution_token_names(site, target)
    checks = {
        "cluster_token_routes_present": bool(results),
        "cluster_token_routes_anonymous_401": all(
            item.get("status") == 401 for item in results
        ),
        "write_routes_explicitly_protected": not unsafe_writes,
        "execution_token_absent_from_data_plane": not token_names,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "route_count": len(routes),
        "cluster_token_results": results,
        "unsafe_write_routes": unsafe_writes,
        "execution_token_name_hits": token_names,
        "limitations": [
            "Anonymous requests validate the deployed route registry and "
            "middleware; they do not exercise every authorized success payload."
        ],
    }


REMOTE_STATUS_PROBE = r"""
import json
from gpu_fault.app import ApplicationContext
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

    payload = {
        "executor_id": "token-rotation-probe",
        "executor_protocol_version": 2,
        "executor_artifact_sha256": identity["artifact"],
        "executor_compatibility_digest": identity["compatibility"],
        "execution_owners": identity["owners"],
        "max_commands": 1,
        "lease_seconds": 60,
    }
    script = (
        DIRECT_CLAIM_PROBE_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__TOKEN__", json.dumps(token))
        .replace("__CLUSTER_ID__", json.dumps(target.cluster_id))
    )
    try:
        value = primary.executor_python(script, timeout=60)
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
    after_commands = primary.cpu_python(REMOTE_STATUS_PROBE)
    before_status = {
        item["command_id"]: item["status"] for item in before_commands["commands"]
    }
    after_status = {
        item["command_id"]: item["status"] for item in after_commands["commands"]
    }
    commands_stable = all(
        after_status.get(command_id) in {"PENDING", "WAITING", "LEASED"}
        for command_id in before_status
    )
    checks = {
        "baseline_only_200": "baseline" in uninterrupted,
        "overlap_only_200": "overlap" in uninterrupted,
        "cutover_only_200": "cutover" in uninterrupted,
        "new_token_accepted_during_overlap": new_token_during_overlap == 200,
        "old_token_rejected_after_completion": old_token_after_completion == 403,
        "remote_commands_not_misterminated": commands_stable,
        "original_credentials_restored": restored,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
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
            "revoked instantly; both original Secrets are restored."
        ],
    }


def run_auth016(
    site: IdentitySite,
    target: ClusterTarget,
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
    control_rollout = None
    executor_rollout = None
    new_token_during_overlap: int | str | None = None
    old_token_after_completion: int | str | None = None
    registry_generation = site.registry_generation()

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
        control_rollout = site.rollout_control()
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
    finally:
        stop.set()
        thread.join(timeout=15)
        site.write_registry(
            original_registry,
            reason="GF-REGIONAL-AUTH-016 restore: original token republished",
        )
        site.rollout_control()
        write_cluster_token(site, target, old_token)
        rollout_executor(site, target)
        restored = registry_token_digests(
            site.registry(), target.cluster_id
        ) == registry_token_digests(
            original_registry, target.cluster_id
        ) and secret_digest(read_cluster_token(site, target)) == secret_digest(
            old_token
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
with socket.create_connection((host, url.port or 443), timeout=15) as raw:
    with context.wrap_socket(raw, server_hostname=host) as tls:
        certificate = tls.getpeercert()

empty_ca_rejected = False
Path("/tmp/auth013-empty-ca.pem").write_text("", encoding="ascii")
try:
    ssl.create_default_context(cafile="/tmp/auth013-empty-ca.pem")
except ssl.SSLError:
    empty_ca_rejected = True

wrong_hostname_rejected = False
try:
    with socket.create_connection((host, url.port or 443), timeout=15) as raw:
        with context.wrap_socket(raw, server_hostname="wrong.invalid"):
            pass
except ssl.SSLCertVerificationError:
    wrong_hostname_rejected = True

not_after = certificate.get("notAfter")
expires = datetime.fromtimestamp(
    ssl.cert_time_to_seconds(not_after),
    tz=timezone.utc,
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
    "empty_ca_rejected": empty_ca_rejected,
    "wrong_hostname_rejected": wrong_hostname_rejected,
    "not_after": expires.isoformat(),
    "remaining_days": (expires - datetime.now(timezone.utc)).total_seconds() / 86400,
}, sort_keys=True))
"""


def run_auth013(site: IdentitySite, target: ClusterTarget) -> dict[str, Any]:
    pod = site.any_executor_pod(target)
    result = site.pod_json("gpu", target, pod, TLS_BOUNDARY_PROBE)
    threshold = int(site.config["health"]["certificate_min_validity_days"])
    checks = {
        "scoped_private_ca": (
            result["ca_exists"]
            and not result["ssl_cert_file_set"]
            and not result["requests_ca_bundle_set"]
        ),
        "certificate_san_matches_nlb": result["hostname_in_san"],
        "empty_ca_rejected": result["empty_ca_rejected"],
        "wrong_hostname_rejected": result["wrong_hostname_rejected"],
        "remaining_validity_exceeds_threshold": result["remaining_days"] > threshold,
        "expiry_threshold_configured": threshold >= 30,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "certificate": {
            "host": result["host"],
            "sans": result["sans"],
            "not_after": result["not_after"],
            "remaining_days": result["remaining_days"],
            "configured_minimum_days": threshold,
        },
        "limitations": [
            "The negative probes perform TLS handshakes only and never disable "
            "verification in the running executor."
        ],
    }


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
    load_balancers = json.loads(
        run(
            [
                "aws",
                "elbv2",
                "describe-load-balancers",
                "--region",
                site.region,
                "--output",
                "json",
            ],
            timeout=120,
        ).stdout
    )["LoadBalancers"]
    selected = next(item for item in load_balancers if item.get("DNSName") == hostname)
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
    broad = []
    for group in groups:
        for permission in group.get("IpPermissions") or []:
            if any(
                item.get("CidrIp") == "0.0.0.0/0"
                for item in permission.get("IpRanges") or []
            ):
                broad.append(
                    {
                        "group_id": group["GroupId"],
                        "from_port": permission.get("FromPort"),
                        "to_port": permission.get("ToPort"),
                    }
                )
    return {
        "scheme": selected.get("Scheme"),
        "security_group_count": len(group_ids),
        "broad_ipv4_rules": broad,
    }


def outside_probe(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "valid": False,
            "error": "outside-VPC probe evidence is required",
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return {"valid": False, "error": "outside probe is not an object"}
    connection_blocked = bool(value.get("connection_blocked"))
    observed_at = str(value.get("observed_at") or "")
    return {
        "valid": connection_blocked and bool(observed_at),
        "connection_blocked": connection_blocked,
        "observed_at": observed_at,
        "probe_location": str(value.get("probe_location") or "external"),
    }


def run_auth014(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    outside_probe_path: Path | None,
) -> dict[str, Any]:
    routes = route_inventory(site, target)
    results = anonymous_routes(site, target, routes)
    nlb = nlb_security_groups(site)
    external = outside_probe(outside_probe_path)
    expected = {
        "cluster-token": {401},
        "dual-credential": {403},
        "execution-token": {403},
        "metrics": {403},
        "public": {200},
    }
    mismatches = [
        item
        for item in results
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
        if item["path"]
        in {
            "/v1/fleet/agents",
            "/v1/runtime-profiles",
            "/v1/advisory-notifications/{notification_id}/send",
        }
    }
    checks = {
        "route_matrix_matches_buckets": not mismatches,
        "anonymous_write_routes_never_succeed": not anonymous_write_success,
        "nlb_has_no_world_open_ipv4_rule": not nlb["broad_ipv4_rules"],
        "outside_vpc_connection_blocked": external["valid"],
        "high_risk_routes_present": len(high_risk) >= 3,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "nlb": nlb,
        "outside_probe": external,
        "route_results": results,
        "mismatches": mismatches,
        "anonymous_write_success": anonymous_write_success,
        "high_risk_routes": high_risk,
        "limitations": [
            "The network denial is supplied by a separately executed probe "
            "outside the NLB allowlist; this runner validates its structured evidence."
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


def gpu_master_reference_hits(
    site: IdentitySite,
    target: ClusterTarget,
) -> list[dict[str, str]]:
    resources = json.loads(
        site.gpu(
            target,
            "get",
            "job,pod,deployment",
            "-o",
            "json",
        )
    )
    hits = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            reference = value.get("secretKeyRef")
            if isinstance(reference, dict) and (
                reference.get("name") == "gpu-fault-control-plane-active"
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
                secret_volume.get("secretName") == "gpu-fault-control-plane-active"
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
    return hits


def run_auth015(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    nodes: tuple[str, str],
    fleet_master_file: Path,
    host_probe_image: str,
    case_dir: Path,
) -> dict[str, Any]:
    if fleet_master_file.stat().st_mode & 0o077:
        raise IdentityAcceptanceError("fleet master file must be mode 0600")
    master = fleet_master_file.read_text(encoding="utf-8").strip()
    if len(master) < 32:
        raise IdentityAcceptanceError("staging fleet master is too short")
    master_sha256 = secret_digest(master)
    original_gpu = secret_document(
        site,
        "gpu",
        target,
        "gpu-fault-node-action-keys",
    )
    original_cpu = secret_document(
        site,
        "cpu",
        target,
        "gpu-fault-node-action-keys",
    )
    before_keys = node_key_digests(original_gpu)
    before_agents = site.regional(target).cpu_python(
        AGENT_SNAPSHOT_PROBE,
        target.cluster_id,
        *nodes,
    )
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
    tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/fleet/test_fleet.py::"
            "test_derived_node_key_cannot_sign_for_another_node",
            "tests/fleet/test_fleet.py::"
            "test_node_specific_key_can_rotate_without_changing_peer",
        ],
        check=False,
        timeout=600,
    )
    try:
        for probe in probes:
            probe.create()
            scans_before[probe.settings.node] = probe.execute(
                "--master-sha256",
                master_sha256,
            )
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
        after_gpu = secret_document(
            site,
            "gpu",
            target,
            "gpu-fault-node-action-keys",
        )
        after_keys = node_key_digests(after_gpu)
        time.sleep(30)
        after_agents = site.regional(target).cpu_python(
            AGENT_SNAPSHOT_PROBE,
            target.cluster_id,
            *nodes,
        )
        for probe in probes:
            scans_after[probe.settings.node] = probe.execute(
                "--master-sha256",
                master_sha256,
            )
        checks = {
            "installer_references_no_fleet_master": not gpu_master_reference_hits(
                site, target
            ),
            "host_scan_before_zero_matches": all(
                not item["master_matches"] for item in scans_before.values()
            ),
            "host_scan_after_zero_matches": all(
                not item["master_matches"] for item in scans_after.values()
            ),
            "cross_node_signature_tests": tests.returncode == 0,
            "rotation_command_succeeded": rotation.returncode == 0,
            "node_a_key_changed": before_keys.get(nodes[0]) != after_keys.get(nodes[0]),
            "node_b_key_unchanged": before_keys.get(nodes[1])
            == after_keys.get(nodes[1]),
            "node_b_agent_continues": (
                before_agents["agents"][nodes[1]]["generation"]
                == after_agents["agents"][nodes[1]]["generation"]
                and after_agents["agents"][nodes[1]]["lifecycle_state"] == "ACTIVE"
            ),
        }
    finally:
        restore_secret(site, "gpu", target, original_gpu)
        restore_secret(site, "cpu", target, original_cpu)
        for probe in probes:
            try:
                residuals[probe.settings.node] = probe.cleanup()
            except Exception as exc:
                residuals[probe.settings.node] = {
                    "cleanup_error": f"{type(exc).__name__}: {exc}"
                }
    checks["secrets_restored"] = (
        node_key_digests(
            secret_document(
                site,
                "gpu",
                target,
                "gpu-fault-node-action-keys",
            )
        )
        == before_keys
    )
    checks["probe_resources_removed"] = all(
        not any(value.values()) for value in residuals.values()
    )
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "master_sha256": master_sha256,
        "scan_before": scans_before,
        "scan_after": scans_after,
        "probe_residuals": residuals,
        "limitations": [
            "The master is supplied from a trusted-host mode-0600 file. Only "
            "its SHA-256 enters evidence; the runner restores both key Secrets."
        ],
    }
