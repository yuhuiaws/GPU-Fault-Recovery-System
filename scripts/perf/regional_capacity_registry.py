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
CONTROL_KUBECONFIG = os.getenv(
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "/tmp/gpu-fault-control-plane.kubeconfig",
)
DATAPLANE_CONTEXT = os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
AWS_REGION = os.getenv("GPU_FAULT_PERF_AWS_REGION", "us-west-2")
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
