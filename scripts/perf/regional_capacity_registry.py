from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .regional_registry_alignment import (
        SECRET_RESTORE_FILE,
        baseline_serialization,
        preflight_checked_at,
        record_secret_baseline,
        verify_alignment,
    )
else:
    from regional_registry_alignment import (
        SECRET_RESTORE_FILE,
        baseline_serialization,
        preflight_checked_at,
        record_secret_baseline,
        verify_alignment,
    )

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
# Only an absent default mount permits the legacy environment fallback.
STORE_DSN_SNIPPET = """\
def store_dsn():
    import os
    configured = os.environ.get("GPU_FAULT_STORE_URL_FILE")
    path = configured if configured is not None else "/etc/gpu-fault/aurora/postgres-url"
    if not path:
        raise RuntimeError("configured store DSN file path is empty")
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
    except FileNotFoundError:
        if configured is not None:
            raise
        return os.environ["GPU_FAULT_STORE_URL"]
    if not value:
        raise RuntimeError("store DSN file is empty")
    return value
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
RUN_LABEL = "gpu-fault.io/acceptance-run"
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
REGISTRY_SNAPSHOT = """
import json
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
try:
    head = store.get_regional_registry_head()
    revision = store.get_regional_registry_revision(head.generation)
    if head.content_sha256 != revision.content_sha256:
        raise RuntimeError("registry head changed during snapshot")
    print(revision.model_dump_json())
finally:
    store.close()
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


def ensure_perf_namespace() -> bool:
    """Create the perf namespace when it is absent; return whether it was created.

    The namespace is the load generators' home and normally outlives the site,
    but an operator sweeping acceptance residue (live 2026-09-22) or a fresh
    data plane starts without it, and ``kubectl apply`` of the connection
    Secret then fails with NotFound. An existing namespace is left untouched.
    """

    if dataplane("get", "namespace", NAMESPACE, "-o", "json", check=False).strip():
        return False
    document = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": NAMESPACE,
            "labels": {"gpu-fault.io/perf-namespace": "capacity-load"},
        },
    }
    dataplane("apply", "-f", "-", stdin=json.dumps(document).encode())
    print(f"perf namespace {NAMESPACE} created", file=sys.stderr, flush=True)
    return True


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
    namespace_created = ensure_perf_namespace()
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
        "namespace_created": namespace_created,
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


def _registry_entry_index(
    values: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in values:
        key = entry.get("cluster_id")
        if not isinstance(key, str) or not key or key in result:
            raise RuntimeError("registry cluster identities are unknown or duplicated")
        result[key] = entry
    return result


def registry_secret_document() -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    """``(metadata, raw clusters.json bytes, parsed entries)`` of the Secret."""

    try:
        value = json.loads(control("get", "secret", REGISTRY_SECRET, "-o", "json"))
        metadata = value["metadata"]
        raw = base64.b64decode(value["data"]["clusters.json"], validate=True)
        values = json.loads(raw)
    except Exception:
        raise RuntimeError("registry Secret could not be read safely") from None
    if (
        not isinstance(metadata, dict)
        or not metadata.get("uid")
        or not metadata.get("resourceVersion")
        or not isinstance(values, list)
    ):
        raise RuntimeError("registry Secret identity or contents are unknown")
    return metadata, raw, values


def _read_registry_secret() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, _raw, values = registry_secret_document()
    return metadata, values


def capture_secret_baseline(artifacts: Path, entries: list[dict[str, Any]]) -> None:
    """Record the Secret as it is before the run's first write."""

    _metadata, raw, observed = registry_secret_document()
    if observed != entries:
        raise RuntimeError("registry Secret changed while recording its baseline")
    record_secret_baseline(artifacts, raw, entries)


def durable_registry_revision() -> dict[str, Any]:
    """The durable head revision (``generation``, ``registrations``,
    ``content_sha256``) read inside a Running api-ha Pod."""

    from scripts.e2e.regional.regional_live_fixture import component_python

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
        raise RuntimeError("no CPU API Pod for the registry snapshot")
    raw = control(
        "exec",
        "-i",
        pod,
        "--",
        component_python("cpu"),
        "-",
        "registry-snapshot",
        stdin=REGISTRY_SNAPSHOT.encode(),
        timeout=120,
    )
    revision = json.loads(raw.splitlines()[-1])
    generation = revision.get("generation")
    if type(generation) is not int or generation < 1:
        raise RuntimeError("durable registry generation is unknown")
    return revision


def write_registry(
    entries: list[dict],
    *,
    run_id: str,
    expected_entries: list[dict],
    reason: str,
    serialization: str | None = None,
) -> dict:
    """Write ``entries`` to the Secret and publish the matching durable revision.

    ``serialization`` is the exact ``clusters.json`` text to store (teardown
    passes the pre-run bytes rebuilt from the recorded shape); it must parse
    to ``entries``. Without it the compact form is written.
    """

    from gpu_fault.regional import (
        RegionalClusterRegistration,
        regional_registry_content_sha256,
    )
    from gpu_fault.regional_registry import (
        configured_regional_registrations,
        regional_registry_config_sha256,
    )

    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("registry write requires an explicit run identity")
    exact_bytes = serialization is not None
    if serialization is None:
        serialization = json.dumps(entries, separators=(",", ":"))
    elif json.loads(serialization) != entries:
        raise RuntimeError("registry serialization does not describe the entries")

    def owned(value: dict) -> bool:
        return (
            value.get("synthetic") is True and value.get("synthetic_run_id") == run_id
        )

    before = _registry_entry_index(expected_entries)
    desired = _registry_entry_index(entries)
    for key in before.keys() | desired.keys():
        if before.get(key) != desired.get(key):
            if any(
                not owned(value)
                for value in (before.get(key), desired.get(key))
                if value is not None
            ):
                raise RuntimeError(
                    "registry write would change another run or a production registration"
                )
    registrations = configured_regional_registrations(entries)

    metadata, raw_before, observed = registry_secret_document()
    if observed != expected_entries:
        raise RuntimeError("registry Secret changed since the run inspected it")
    revision = durable_registry_revision()
    generation = revision["generation"]
    current = [
        RegionalClusterRegistration.model_validate(item)
        for item in revision["registrations"]
    ]
    if regional_registry_content_sha256(current) != revision.get("content_sha256"):
        raise RuntimeError("durable registry snapshot digest is inconsistent")
    current_by_id = {item.cluster_id: item for item in current}
    if len(current_by_id) != len(current):
        raise RuntimeError("durable registry cluster identities are duplicated")
    targets = {item.cluster_id: item for item in registrations}
    for key, original in current_by_id.items():
        target = targets.get(key)
        original_owned = owned(original.model_dump(mode="json"))
        if target is None:
            if not original_owned:
                raise RuntimeError(
                    "registry cleanup would drop an unrelated durable registration"
                )
            continue
        same_config = regional_registry_config_sha256(
            [original]
        ) == regional_registry_config_sha256([target])
        same_scope = (
            original.synthetic == target.synthetic
            and original.synthetic_run_id == target.synthetic_run_id
            and original.synthetic_expires_at == target.synthetic_expires_at
        )
        if same_config and same_scope:
            targets[key] = original
        elif not original_owned or not owned(target.model_dump(mode="json")):
            raise RuntimeError("durable registry identity changed outside this run")
    for key, target in targets.items():
        if key not in current_by_id and not owned(target.model_dump(mode="json")):
            raise RuntimeError(
                "registry write would recreate an unrelated durable registration"
            )
    payload = base64.b64encode(serialization.encode()).decode()
    if observed != entries or (exact_bytes and serialization.encode() != raw_before):
        patch = [
            {"op": "test", "path": "/metadata/uid", "value": metadata["uid"]},
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": metadata["resourceVersion"],
            },
            {"op": "replace", "path": "/data/clusters.json", "value": payload},
        ]
        try:
            control(
                "patch",
                "secret",
                REGISTRY_SECRET,
                "--type=json",
                "--patch-file=/dev/stdin",
                stdin=json.dumps(patch).encode(),
            )
        except Exception:
            after_metadata, after = _read_registry_secret()
            if after_metadata["uid"] != metadata["uid"] or after != entries:
                raise RuntimeError(
                    "registry Secret write acknowledgement is unresolved"
                ) from None
    after_metadata, after = _read_registry_secret()
    if after_metadata["uid"] != metadata["uid"] or after != entries:
        raise RuntimeError("registry Secret changed while confirming the write")
    target_values = [targets[key] for key in sorted(targets)]
    digest = regional_registry_content_sha256(target_values)
    publication_required = digest != revision["content_sha256"]
    target_generation = generation + int(publication_required)
    if publication_required:
        try:
            registry_api(
                "POST",
                "/v1/regional/registry/revisions",
                {
                    "expected_generation": generation,
                    "registrations": [
                        item.model_dump(mode="json") for item in target_values
                    ],
                    "reason": reason,
                },
            )
        except Exception:
            # Reconcile the expected head after a lost ACK, without another POST.
            status = registry_api("GET", "/v1/regional/registry/status")
            if (
                status.get("generation") != target_generation
                or status.get("content_sha256") != digest
            ):
                raise RuntimeError(
                    "registry publication acknowledgement is unresolved"
                ) from None
    deadline = time.monotonic() + 300
    while True:
        status = registry_api("GET", "/v1/regional/registry/status")
        if (
            type(status.get("generation")) is not int
            or status["generation"] != target_generation
            or status.get("content_sha256") != digest
        ):
            raise RuntimeError("registry head changed while confirming this run")
        if status.get("converged") is True:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("run-owned registry revision did not converge")
        time.sleep(1)
    final_metadata, final_entries = _read_registry_secret()
    if final_metadata["uid"] != metadata["uid"] or final_entries != entries:
        raise RuntimeError("registry Secret changed during publication")
    return {
        "expected_generation": generation,
        "expected_content_sha256": digest,
        "generation": target_generation,
        "publication_required": publication_required,
        "registrations": [item.model_dump(mode="json") for item in target_values],
        "secret_uid": metadata["uid"],
    }


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


# The EKS account every synthetic registration names. No real cluster lives
# there, so a synthetic entry can never be mistaken for -- or collide with --
# a physical cluster's ARN, whichever id it carries.
SYNTHETIC_ACCOUNT = "000000000000"


def synthetic_cluster_entry(
    cluster_id: str,
    *,
    run_id: str,
    expires_at: datetime,
    region: str | None = None,
    allowed_namespaces: list[str] | None = None,
    token: str | None = None,
) -> dict:
    """One synthetic registration as the registry takes it (plaintext token).

    The shape every synthetic registration shares: ``synthetic`` plus the run
    id and expiry that let the perf sweep and the registry's own load-time
    hygiene reap it, a fresh random token, a placeholder-account EKS ARN and a
    loopback-only agent CIDR so no Node Agent can ever bind to it. The perf
    suite mints ``perf-cap-NNN`` ids through it; the AUTH-007/008 runner mints
    its ``auth-logical-`` second logical cluster through the same builder so
    the two never drift apart in what "synthetic" means.
    """

    selected_region = region or AWS_REGION
    return {
        "cluster_id": cluster_id,
        "region": selected_region,
        "hyperpod_cluster_name": cluster_id,
        "eks_cluster_arn": (
            f"arn:aws:eks:{selected_region}:{SYNTHETIC_ACCOUNT}:cluster/{cluster_id}"
        ),
        "token": token or secrets.token_urlsafe(48),
        "synthetic": True,
        "synthetic_run_id": run_id,
        "synthetic_expires_at": expires_at.isoformat(),
        "allowed_namespaces": (
            list(allowed_namespaces)
            if allowed_namespaces is not None
            else ["default", NAMESPACE, "kubeflow", "training"]
        ),
        "agent_endpoint_allowed_cidrs": ["127.0.0.1/32"],
    }


def perf_cluster_entries(
    count: int,
    *,
    run_id: str,
    expires_at: datetime,
) -> list[dict]:
    return [
        synthetic_cluster_entry(
            f"{PERF_CLUSTER_PREFIX}{index:03d}",
            run_id=run_id,
            expires_at=expires_at,
        )
        for index in range(count)
    ]


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
    del attempts, now
    initial = load_registry()
    synthetic = [entry for entry in initial if _synthetic_entry(entry)]
    if not force:
        if synthetic:
            raise RuntimeError(
                "synthetic registry entries already exist; preflight cannot purge any run"
            )
        _write_registry_audit(
            artifacts,
            phase=phase,
            scope=scope,
            entries=initial,
            removed=0,
        )
        return 0
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("registry cleanup requires an explicit run identity")
    owned = [
        entry
        for entry in synthetic
        if entry.get("synthetic") is True and entry.get("synthetic_run_id") == run_id
    ]
    baseline = [entry for entry in initial if entry not in owned]
    # Restore the pre-run bytes when the surviving entries are exactly what
    # register() saw: the api-ha replicas compare the Secret they started on
    # with the durable head, and a re-serialised Secret is a needless change.
    serialization = (
        baseline_serialization(artifacts, baseline) if artifacts is not None else None
    )
    write_registry(
        baseline,
        run_id=run_id,
        expected_entries=initial,
        reason=f"capacity cleanup {run_id}",
        serialization=serialization,
    )
    if load_registry() != baseline:
        raise RuntimeError("registry changed while verifying scoped cleanup")
    if artifacts is not None:
        # write_registry stored exactly ``serialization`` (or the compact form).
        written = serialization or json.dumps(baseline, separators=(",", ":"))
        (artifacts / SECRET_RESTORE_FILE).write_text(
            json.dumps(
                {
                    "byte_identical": serialization is not None,
                    "sha256_after": hashlib.sha256(written.encode()).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    _write_registry_audit(
        artifacts,
        phase=phase,
        scope=scope,
        entries=initial,
        removed=len(owned),
    )
    return len(owned)


def registered_cluster_ids(artifacts: Path, run_id: str) -> list[str]:
    if __package__:
        from .regional_capacity_data import validate_scope
    else:
        from regional_capacity_data import validate_scope

    intent = json.loads((artifacts / "registry-registration-intent.json").read_text())
    if not isinstance(intent, dict) or intent.get("run_id") != run_id:
        raise RuntimeError("capacity registration intent belongs to another run")
    if intent.get("data_empty_before_registration") is not True:
        raise RuntimeError("capacity registration intent lacks its empty-data proof")
    cluster_ids = intent.get("cluster_ids")
    if not isinstance(cluster_ids, list) or any(
        not isinstance(item, str) for item in cluster_ids
    ):
        raise RuntimeError(
            "capacity registration intent has no exact cluster inventory"
        )
    validate_scope(run_id, cluster_ids)
    return cluster_ids


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


def upsert_secret(name: str, files: dict[str, bytes], *, run_id: str) -> dict:
    if (
        name != TOKEN_SECRET
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("synthetic token Secret requires an explicit run identity")
    data = {key: base64.b64encode(value).decode() for key, value in files.items()}
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {RUN_LABEL: run_id},
        },
        "type": "Opaque",
        "data": data,
    }

    def read() -> dict | None:
        try:
            raw = dataplane("get", "secret", name, "--ignore-not-found", "-o", "json")
            value = json.loads(raw) if raw.strip() else None
        except Exception:
            raise RuntimeError(
                "synthetic token Secret read was not confirmed"
            ) from None
        if value is not None and not isinstance(value, dict):
            raise RuntimeError("synthetic token Secret response is invalid")
        return value

    def verify(value: dict | None, *, expected_uid: str | None = None) -> dict:
        metadata = (value or {}).get("metadata") or {}
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("uid"), str)
            or not metadata["uid"]
            or not isinstance(metadata.get("resourceVersion"), str)
            or not metadata["resourceVersion"]
            or not isinstance(metadata.get("labels"), dict)
            or (metadata.get("labels") or {}).get(RUN_LABEL) != run_id
            or (value or {}).get("data") != data
            or (expected_uid is not None and metadata["uid"] != expected_uid)
        ):
            raise RuntimeError("synthetic token Secret ownership or contents differ")
        return {
            "uid": metadata["uid"],
            "resource_version": metadata["resourceVersion"],
            "run_id": run_id,
        }

    existing = read()
    if existing is not None:
        return verify(existing)

    def create() -> str:
        return dataplane(
            "create",
            "-f",
            "-",
            "-o",
            "jsonpath={.metadata}",
            stdin=json.dumps(manifest).encode(),
        )

    try:
        raw = create()
    except Exception:
        # A lost create ACK is resolved by the run label, contents and fresh UID.
        # Never replay the create or overwrite a same-name foreign Secret. The one
        # create that is retried is the first Secret of a run whose perf namespace
        # does not exist yet (live 2026-09-23: the namespace had been swept with
        # acceptance residue); the namespace is created and the create repeated.
        if read() is None and ensure_perf_namespace():
            raw = create()
        else:
            return verify(read())
    try:
        created = json.loads(raw)
        uid = created["uid"]
        if (
            not isinstance(uid, str)
            or not uid
            or created["labels"][RUN_LABEL] != run_id
        ):
            raise ValueError("invalid create identity")
    except Exception:
        raise RuntimeError(
            "synthetic token Secret create identity is unknown"
        ) from None
    return verify(read(), expected_uid=uid)


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
    if type(count) is not int or count <= 0:
        raise RuntimeError("synthetic cluster count must be a positive integer")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None:
        raise RuntimeError(
            "synthetic registry mutation requires an explicit run identity"
        )
    validate_notification_safety()
    validate_alertmanager_drill_route()
    # A Secret that already disagrees with the durable head (or a replica that
    # says so) is refused here, so the drift is never attributed to this run.
    verify_registry_alignment(artifacts=artifacts, phase="preflight")
    cleanup_registry_residuals(
        scope=scope,
        artifacts=artifacts,
        phase="preflight",
        force=False,
        run_id=run_id,
    )
    existing = load_registry()
    baseline = [entry for entry in existing if not _synthetic_entry(entry)]
    (artifacts / "registry-baseline.json").write_text(
        json.dumps(redacted_registry_entries(baseline), indent=1) + "\n"
    )
    capture_secret_baseline(artifacts, existing)
    perf = perf_cluster_entries(
        count,
        run_id=run_id,
        expires_at=expires_at,
    )
    log(
        f"registering {len(perf)} audit clusters "
        f"(keeping {len(baseline)} production entries)"
    )
    tokens = [
        {
            "cluster_id": entry["cluster_id"],
            "token": entry["token"],
        }
        for entry in perf
    ]
    if __package__:
        from .regional_capacity_data import invoke
    else:
        from regional_capacity_data import invoke
    cluster_ids = [item["cluster_id"] for item in perf]
    data_preflight = invoke(
        control, run_id=run_id, cluster_ids=cluster_ids, cleanup=False
    )
    if data_preflight["total"] != 0:
        raise RuntimeError(
            "synthetic cluster data already exists; registration cannot adopt it"
        )
    (artifacts / "registry-data-preflight.json").write_text(
        json.dumps(data_preflight, sort_keys=True) + "\n", encoding="utf-8"
    )
    intent_path = artifacts / "registry-registration-intent.json"
    intent_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cluster_ids": cluster_ids,
                "synthetic_expires_at": expires_at.isoformat(),
                "data_empty_before_registration": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    intent_path.chmod(0o600)
    token_proof = upsert_secret(
        TOKEN_SECRET,
        {"clusters.json": json.dumps(tokens, indent=1).encode()},
        run_id=run_id,
    )
    # Every load and executor Pod mounts the mirrored connection Secret from the
    # perf namespace; mirror it for every suite here (live 2026-09-23: only the
    # integrated suite did, and a recreated namespace left the capacity suite's
    # load Pod in ContainerCreating on "secret not found" until its wait cap).
    connection = sync_dataplane_connection_secret()
    (artifacts / "connection-secret-preflight.json").write_text(
        json.dumps(connection, sort_keys=True) + "\n", encoding="utf-8"
    )
    token_path = artifacts / "registry-token-proof.json"
    token_path.write_text(
        json.dumps(token_proof, sort_keys=True) + "\n", encoding="utf-8"
    )
    token_path.chmod(0o600)
    write_registry(
        baseline + perf,
        run_id=run_id,
        expected_entries=existing,
        reason=f"capacity register {run_id}",
    )
    return tokens


def deregister(
    *,
    scope: str,
    artifacts: Path | None = None,
    run_id: str | None = None,
    verify: bool = True,
) -> int:
    removed = cleanup_registry_residuals(
        scope=scope,
        artifacts=artifacts,
        phase="postflight",
        force=True,
        run_id=run_id,
    )
    # The run's cleanup is not done until the control plane agrees with the
    # Secret it left behind; a drifted replica fails the teardown here rather
    # than the next HA-010 preflight. Callers with more run-owned resources to
    # remove (the data-plane token Secret) pass verify=False and run the gate
    # last, so a failed gate never strands them.
    if verify:
        verify_registry_alignment(artifacts=artifacts, phase="postflight")
    return removed


def verify_registry_alignment(*, artifacts: Path | None, phase: str) -> dict[str, Any]:
    """Prove the Secret, the durable head and every api-ha replica agree
    (see ``regional_registry_alignment.verify_alignment``). The postflight may
    roll replicas that started inside the run's window (their start-up Secret
    digest is the run's transient one, not a drift the run left behind)."""

    return verify_alignment(
        control,
        secret_document=registry_secret_document,
        durable_revision=durable_registry_revision,
        artifacts=artifacts,
        phase=phase,
        restart_window_start=(
            preflight_checked_at(artifacts) if phase == "postflight" else None
        ),
    )
