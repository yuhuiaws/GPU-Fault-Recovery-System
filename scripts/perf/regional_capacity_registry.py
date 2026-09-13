from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONTROL_NAMESPACE = "gpu-fault-system"
CONTROL_NAMESPACE = os.getenv(
    "GPU_FAULT_PERF_CONTROL_NAMESPACE",
    "gpu-fault-perf-system",
)
NAMESPACE = os.getenv(
    "GPU_FAULT_PERF_DATAPLANE_NAMESPACE",
    "gpu-fault-perf-system",
)
IDENTITY_NAMESPACE = os.getenv(
    "GPU_FAULT_PERF_IDENTITY_NAMESPACE",
    "gpu-fault-system",
)
CONTROL_KUBECONFIG = os.getenv(
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "/tmp/gpu-fault-control-plane.kubeconfig",
)
DATAPLANE_CONTEXT = os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
AWS_REGION = os.getenv("GPU_FAULT_PERF_AWS_REGION", "us-west-2")

# Scripts exec'd inside a control-plane Pod that open psycopg themselves must
# connect with the DSN the product uses: ``GPU_FAULT_STORE_URL_FILE``, re-read
# on every connect so it follows a master-password rotation. The
# ``GPU_FAULT_STORE_URL`` env var is the value at Pod start and is stale the
# moment HA-009 rotates the password (attempt 4, every post-rotation read
# failed with "password authentication failed"). Prepend this to such a script
# and call ``store_dsn()``; it has no braces, so it also fits an f-string, and
# it falls back to the env var where no file is mounted (a deploy host).
STORE_DSN_SNIPPET = """\
def store_dsn():
    import os
    path = os.environ.get("GPU_FAULT_STORE_URL_FILE") or "/etc/gpu-fault/aurora/postgres-url"
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]
"""
REGISTRY_SECRET = os.getenv(
    "GPU_FAULT_PERF_REGISTRY_SECRET",
    "gpu-fault-regional-clusters",
)
CONNECTION_SECRET = os.getenv(
    "GPU_FAULT_PERF_CONNECTION_SECRET",
    "gpu-fault-regional-connection",
)
TOKEN_SECRET = "gpu-fault-perf-clusters"
PERF_CLUSTER_PREFIX = "perf-cap-"
LIVE_REGISTRY_CONFIRMATION = "ALLOW_PERF_CAPACITY_LIVE_REGISTRY"
REGISTRY_API_CLIENT = r"""
import json
import os
import sys
import urllib.request

method, path = sys.argv[1:]
body = sys.stdin.buffer.read()
request = urllib.request.Request(
    "http://127.0.0.1:8080" + path,
    data=body or None,
    method=method,
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run(
    argv: list[str],
    *,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        argv,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{' '.join(argv)}\n{result.stderr.decode()}"
        )
    return result


def control(
    *args: str,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    result = run(
        [
            "kubectl",
            "--kubeconfig",
            CONTROL_KUBECONFIG,
            "-n",
            CONTROL_NAMESPACE,
            *args,
        ],
        stdin=stdin,
        check=check,
        timeout=timeout,
    )
    return result.stdout.decode()


def dataplane(
    *args: str,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    result = run(
        [
            "kubectl",
            "--context",
            DATAPLANE_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        check=check,
        timeout=timeout,
    )
    return result.stdout.decode()


def dataplane_identity(
    *args: str,
    check: bool = True,
    timeout: int = 300,
) -> str:
    result = run(
        [
            "kubectl",
            "--context",
            DATAPLANE_CONTEXT,
            "-n",
            IDENTITY_NAMESPACE,
            *args,
        ],
        check=check,
        timeout=timeout,
    )
    return result.stdout.decode()


def sync_dataplane_connection_secret() -> dict:
    """Copy the live connection Secret into the perf namespace; report the change.

    The load generators and simulated executors read the control-plane URL,
    CA and cluster token from ``CONNECTION_SECRET`` in the perf namespace. That
    namespace outlives the site: on 2026-09-13 it still held the Secret of a
    site uninstalled and rebuilt twelve days later, every load Pod failed TLS
    verification against the old CA within a second and the round aborted with
    no log to explain it. The identity namespace's Secret is the truth; it is
    mirrored before anything is registered, and its absence fails closed.
    """

    raw = dataplane_identity(
        "get", "secret", CONNECTION_SECRET, "-o", "json", check=False
    ).strip()
    if not raw:
        raise RuntimeError(
            f"connection Secret {CONNECTION_SECRET} is missing from the identity "
            f"namespace {IDENTITY_NAMESPACE}; the data plane is not joined"
        )
    source = json.loads(raw)
    data = dict(source.get("data") or {})
    if not data:
        raise RuntimeError(f"connection Secret {CONNECTION_SECRET} carries no data")
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    previous_raw = dataplane(
        "get", "secret", CONNECTION_SECRET, "-o", "json", check=False
    ).strip()
    previous_digest = None
    if previous_raw:
        previous = json.loads(previous_raw).get("data") or {}
        previous_digest = hashlib.sha256(
            json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    if previous_digest != digest:
        document = {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": source.get("type", "Opaque"),
            "metadata": {
                "name": CONNECTION_SECRET,
                "namespace": NAMESPACE,
                "labels": {
                    "gpu-fault.io/perf-synced-from": IDENTITY_NAMESPACE,
                },
            },
            "data": data,
        }
        dataplane("apply", "-f", "-", stdin=json.dumps(document).encode())
    return {
        "secret": CONNECTION_SECRET,
        "source_namespace": IDENTITY_NAMESPACE,
        "target_namespace": NAMESPACE,
        "keys": sorted(data),
        "data_sha256": digest,
        "previous_sha256": previous_digest,
        "changed": previous_digest != digest,
    }


def load_registry() -> list[dict]:
    raw = control(
        "get",
        "secret",
        REGISTRY_SECRET,
        "-o",
        "jsonpath={.data.clusters\\.json}",
    )
    return json.loads(base64.b64decode(raw))


def validate_registry(entries: list[dict]) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from gpu_fault.regional import (  # noqa: PLC0415
            RegionalClusterRegistration,
            cluster_token_sha256,
        )
    except ImportError as exc:
        log(f"registry pre-validation skipped: {exc}")
        return
    for entry in entries:
        item = dict(entry)
        token = item.pop("token", None)
        if token:
            item["token_sha256"] = cluster_token_sha256(str(token))
        RegionalClusterRegistration(**item)


def write_registry(entries: list[dict]) -> None:
    validate_registry(entries)
    payload = base64.b64encode(json.dumps(entries, indent=1).encode()).decode()
    patch = json.dumps({"data": {"clusters.json": payload}})
    control("patch", "secret", REGISTRY_SECRET, "-p", patch)


def control_pods() -> list[str]:
    raw = control(
        "get",
        "pod",
        "-l",
        (
            "app in (gpu-fault-api-ha,gpu-fault-control-worker,"
            "gpu-fault-telemetry-spool-worker)"
        ),
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} {end}",
    )
    names = [name for name in raw.split() if name]
    if names:
        return names
    raw = control(
        "get",
        "pod",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} {end}",
    )
    return [
        name
        for name in raw.split()
        if name.startswith(
            (
                "gpu-fault-api-ha-",
                "gpu-fault-control-worker-",
                "gpu-fault-telemetry-spool-worker-",
            )
        )
    ]


def validate_notification_safety() -> None:
    for pod in control_pods():
        value = (
            control(
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                (
                    "import os; print("
                    "os.environ.get('GPU_FAULT_NOTIFICATION_DELIVER_DRILLS','false')"
                    ")"
                ),
            )
            .strip()
            .lower()
        )
        if value not in {"", "false", "0", "no", "off"}:
            raise RuntimeError(
                "capacity runs must not deliver drill notifications: "
                f"{pod} has GPU_FAULT_NOTIFICATION_DELIVER_DRILLS={value}"
            )


AMP_WORKSPACE_ID = os.getenv("GPU_FAULT_PERF_AMP_WORKSPACE_ID", "")


def _amp_workspace_id() -> str:
    """The site's AMP workspace: the configured one, else the single tagged one."""

    if AMP_WORKSPACE_ID:
        return AMP_WORKSPACE_ID
    result = run(
        [
            "aws",
            "amp",
            "list-workspaces",
            "--region",
            AWS_REGION,
            "--output",
            "json",
        ],
        check=True,
    )
    workspaces = [
        item
        for item in json.loads(result.stdout.decode()).get("workspaces", [])
        if (item.get("status") or {}).get("statusCode") == "ACTIVE"
        and "gpu-fault:site-id" in (item.get("tags") or {})
    ]
    if len(workspaces) != 1:
        raise RuntimeError(
            "cannot identify the site's AMP workspace: "
            f"{len(workspaces)} ACTIVE workspaces carry gpu-fault:site-id; set "
            "GPU_FAULT_PERF_AMP_WORKSPACE_ID"
        )
    return str(workspaces[0]["workspaceId"])


def validate_alertmanager_drill_route() -> dict:
    """Fail closed unless the live Alertmanager sinks perf-cap- alerts.

    The Store marks every notification of a `perf-cap-` cluster as a drill and
    the dispatcher suppresses it, but AMP alerts never pass through the Store:
    Alertmanager delivers them to the SNS topic itself. Live 2026-09-13, one
    afternoon of capacity rounds fired GpuFaultCollectorSilent and
    GpuFaultStaleAttemptObservation for 32 synthetic clusters and mailed the
    administrator ~580 times. The route that sends `cluster_id=~"perf-cap-.*"`
    to a receiver with no delivery configuration is therefore a precondition.
    """

    import base64

    import yaml

    workspace_id = _amp_workspace_id()
    result = run(
        [
            "aws",
            "amp",
            "describe-alert-manager-definition",
            "--workspace-id",
            workspace_id,
            "--region",
            AWS_REGION,
            "--output",
            "json",
        ],
        check=True,
    )
    definition = json.loads(result.stdout.decode())["alertManagerDefinition"]
    outer = yaml.safe_load(base64.b64decode(definition["data"]).decode()) or {}
    config = outer.get("alertmanager_config", outer)
    if isinstance(config, str):
        config = yaml.safe_load(config) or {}
    receivers = {str(item.get("name")): item for item in config.get("receivers") or []}
    sink_routes = []
    for route in (config.get("route") or {}).get("routes") or []:
        matchers = [str(m).replace(" ", "") for m in route.get("matchers") or []]
        if not any(
            m.startswith("cluster_id=~") and PERF_CLUSTER_PREFIX in m for m in matchers
        ):
            continue
        receiver = receivers.get(str(route.get("receiver", "")))
        if receiver is not None and not any(
            str(key).endswith("_configs") for key in receiver
        ):
            sink_routes.append(
                {"receiver": route.get("receiver"), "matchers": matchers}
            )
    if not sink_routes:
        raise RuntimeError(
            "capacity runs must not mail drill alerts: the live Alertmanager "
            f"(workspace {workspace_id}) has no route sending "
            f'cluster_id=~"{PERF_CLUSTER_PREFIX}.*" to a receiver without delivery; '
            "deploy the release that carries the drill sink first"
        )
    return {
        "workspace_id": workspace_id,
        "definition_status": (definition.get("status") or {}).get("statusCode"),
        "drill_sink_routes": sink_routes,
    }


def validate_registry_target(
    *,
    allow_live_registry: bool,
    confirmation: str | None,
) -> str:
    live = CONTROL_NAMESPACE == PRODUCTION_CONTROL_NAMESPACE
    if live and (not allow_live_registry or confirmation != LIVE_REGISTRY_CONFIRMATION):
        raise RuntimeError(
            "capacity registry mutation targets the production control plane; "
            "use an isolated control namespace, or provide both "
            "--allow-live-registry and the exact live-registry confirmation"
        )
    return "live" if live else "isolated"


def registry_api(
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    if not pod:
        raise RuntimeError("no Running gpu-fault-api-ha Pod for registry update")
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-c",
        REGISTRY_API_CLIENT,
        method,
        path,
        stdin=json.dumps(payload or {}, separators=(",", ":")).encode(),
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise RuntimeError("regional registry API returned a non-object")
    return value


def publish_registry_revision(
    entries: list[dict],
    *,
    reason: str,
    timeout_seconds: int = 300,
) -> dict:
    current = registry_api("GET", "/v1/regional/registry/status")
    published = registry_api(
        "POST",
        "/v1/regional/registry/revisions",
        {
            "expected_generation": int(current["generation"]),
            "registrations": redacted_registry_entries(entries),
            "reason": reason,
        },
    )
    generation = int(published["generation"])
    digest = str(published["content_sha256"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = registry_api("GET", "/v1/regional/registry/status")
        if (
            int(status["generation"]) == generation
            and str(status["content_sha256"]) == digest
            and status.get("converged") is True
        ):
            return status
        time.sleep(1)
    raise RuntimeError(f"regional registry generation {generation} did not converge")


def perf_cluster_entries(
    count: int,
    *,
    run_id: str,
    expires_at: datetime,
) -> list[dict]:
    account = "000000000000"
    entries = []
    for index in range(count):
        cluster_id = f"{PERF_CLUSTER_PREFIX}{index:03d}"
        entries.append(
            {
                "cluster_id": cluster_id,
                "region": AWS_REGION,
                "hyperpod_cluster_name": cluster_id,
                "eks_cluster_arn": (
                    f"arn:aws:eks:{AWS_REGION}:{account}:cluster/{cluster_id}"
                ),
                "token": secrets.token_urlsafe(48),
                "synthetic": True,
                "synthetic_run_id": run_id,
                "synthetic_expires_at": expires_at.isoformat(),
                "allowed_namespaces": [
                    "default",
                    NAMESPACE,
                    "kubeflow",
                    "training",
                ],
                "agent_endpoint_allowed_cidrs": ["127.0.0.1/32"],
            }
        )
    return entries


def redacted_registry_entries(entries: list[dict]) -> list[dict]:
    redacted = []
    for entry in entries:
        item = dict(entry)
        token = item.pop("token", None)
        if token is not None:
            item["token_sha256"] = hashlib.sha256(str(token).encode()).hexdigest()
        redacted.append(item)
    return redacted


def _synthetic_entry(entry: dict) -> bool:
    return bool(entry.get("synthetic")) or str(entry.get("cluster_id", "")).startswith(
        PERF_CLUSTER_PREFIX
    )


def _synthetic_expiration(entry: dict) -> datetime | None:
    raw = entry.get("synthetic_expires_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo is not None else None


def _registry_audit_entries(entries: list[dict]) -> list[dict]:
    return [
        {
            "cluster_id": str(entry.get("cluster_id") or ""),
            "synthetic": bool(entry.get("synthetic")),
            "synthetic_run_id": entry.get("synthetic_run_id"),
            "synthetic_expires_at": entry.get("synthetic_expires_at"),
            "legacy_prefix_only": (
                str(entry.get("cluster_id", "")).startswith(PERF_CLUSTER_PREFIX)
                and not bool(entry.get("synthetic"))
            ),
        }
        for entry in entries
        if _synthetic_entry(entry)
    ]


def _write_registry_audit(
    artifacts: Path | None,
    *,
    phase: str,
    scope: str,
    entries: list[dict],
    removed: int,
) -> None:
    if artifacts is None:
        return
    (artifacts / f"registry-{phase}.json").write_text(
        json.dumps(
            {
                "scope": scope,
                "control_namespace": CONTROL_NAMESPACE,
                "registry_secret": REGISTRY_SECRET,
                "removed": removed,
                "synthetic_entries": _registry_audit_entries(entries),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def cleanup_registry_residuals(
    *,
    scope: str,
    artifacts: Path | None,
    phase: str,
    force: bool,
    run_id: str | None = None,
    attempts: int = 3,
    now: datetime | None = None,
) -> int:
    observed = now or datetime.now(timezone.utc)
    initial = load_registry()
    synthetic = [
        entry
        for entry in initial
        if _synthetic_entry(entry)
        and (run_id is None or str(entry.get("synthetic_run_id") or "") == run_id)
    ]
    active = [
        entry
        for entry in synthetic
        if (expires := _synthetic_expiration(entry)) is not None and expires > observed
    ]
    if active and not force:
        run_ids = sorted(
            {str(entry.get("synthetic_run_id") or "<unknown>") for entry in active}
        )
        raise RuntimeError(
            "active synthetic registry entries already exist for run(s): "
            + ", ".join(run_ids)
        )
    if not synthetic:
        foreign = [entry for entry in initial if _synthetic_entry(entry)]
        if run_id is not None and foreign:
            raise RuntimeError("registry contains synthetic entries from another run")
        _write_registry_audit(
            artifacts,
            phase=phase,
            scope=scope,
            entries=initial,
            removed=0,
        )
        return 0
    baseline = [entry for entry in initial if entry not in synthetic]
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            write_registry(baseline)
            publish_registry_revision(
                baseline,
                reason=f"capacity cleanup {run_id or 'all-synthetic'}",
            )
            persisted = load_registry()
            remaining = [
                entry
                for entry in persisted
                if _synthetic_entry(entry)
                and (
                    run_id is None or str(entry.get("synthetic_run_id") or "") == run_id
                )
            ]
            if remaining:
                raise RuntimeError(
                    f"registry still contains {len(remaining)} synthetic entries"
                )
            foreign = [entry for entry in persisted if _synthetic_entry(entry)]
            if run_id is not None and foreign:
                raise RuntimeError(
                    "registry contains synthetic entries from another run"
                )
            _write_registry_audit(
                artifacts,
                phase=phase,
                scope=scope,
                entries=initial,
                removed=len(synthetic),
            )
            return len(synthetic)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(
        f"synthetic registry cleanup failed after {attempts} attempts: {last_error}"
    ) from last_error


def validate_registered_synthetic_run(
    *,
    count: int,
    run_id: str,
    artifacts: Path,
    scope: str,
) -> None:
    now = datetime.now(timezone.utc)
    entries = load_registry()
    synthetic = [entry for entry in entries if _synthetic_entry(entry)]
    if len(synthetic) != count:
        raise RuntimeError(
            f"registered synthetic cluster count is {len(synthetic)}, expected {count}"
        )
    run_ids = {str(entry.get("synthetic_run_id") or "<legacy>") for entry in synthetic}
    if run_ids != {run_id}:
        raise RuntimeError(
            "registered synthetic clusters do not belong to this suite ID: "
            + ", ".join(sorted(run_ids))
        )
    expired = [
        entry
        for entry in synthetic
        if (expires := _synthetic_expiration(entry)) is None or expires <= now
    ]
    if expired:
        raise RuntimeError("registered synthetic clusters are expired")
    _write_registry_audit(
        artifacts,
        phase="preflight",
        scope=scope,
        entries=entries,
        removed=0,
    )


def upsert_secret(name: str, files: dict[str, bytes]) -> None:
    data = {key: base64.b64encode(value).decode() for key, value in files.items()}
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "type": "Opaque",
        "data": data,
    }
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )


def register(
    count: int,
    artifacts: Path,
    *,
    run_id: str,
    expires_at: datetime,
    allow_live_registry: bool,
    live_registry_confirmation: str | None,
) -> list[dict]:
    scope = validate_registry_target(
        allow_live_registry=allow_live_registry,
        confirmation=live_registry_confirmation,
    )
    validate_notification_safety()
    validate_alertmanager_drill_route()
    cleanup_registry_residuals(
        scope=scope,
        artifacts=artifacts,
        phase="preflight",
        force=False,
    )
    existing = load_registry()
    baseline = [entry for entry in existing if not _synthetic_entry(entry)]
    (artifacts / "registry-baseline.json").write_text(
        json.dumps(redacted_registry_entries(baseline), indent=1) + "\n"
    )
    perf = perf_cluster_entries(
        count,
        run_id=run_id,
        expires_at=expires_at,
    )
    log(
        f"registering {len(perf)} audit clusters "
        f"(keeping {len(baseline)} production entries)"
    )
    write_registry(baseline + perf)
    publish_registry_revision(
        baseline + perf,
        reason=f"capacity register {run_id}",
    )
    tokens = [
        {
            "cluster_id": entry["cluster_id"],
            "token": entry["token"],
        }
        for entry in perf
    ]
    upsert_secret(
        TOKEN_SECRET,
        {"clusters.json": json.dumps(tokens, indent=1).encode()},
    )
    return tokens


def deregister(
    *,
    scope: str,
    artifacts: Path | None = None,
    run_id: str | None = None,
) -> int:
    return cleanup_registry_residuals(
        scope=scope,
        artifacts=artifacts,
        phase="postflight",
        force=True,
        run_id=run_id,
    )
