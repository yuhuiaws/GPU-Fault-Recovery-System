"""The skeleton NET-002 and NET-003 share: one seeded command, one probe Pod.

``seeded_command_fixture`` generalised NET-002's shape for NET-006 and CMD-017
and deliberately left NET-002 itself alone, so NET-002 and NET-003 carried two
private copies of the same helpers (``cpu_python``, the residual queries, the
state-file polling, ``cleanup``). This module is the one place the two network
cases import from. It builds on ``seeded_command_fixture`` for everything that
is identical and adds what the network cases need on top:

- the CPU API Pod name is looked up once and cached, not once per call;
- the database residual check covers the nine tables the network cases
  already inspected, not the four the seeded fixture checks;
- the run id is derived from the run directory's path and the clock, not
  from a suffix the run directory's name may or may not carry;
- the predecessor gate and the release/cluster identity every case result
  must carry so the next case in the chain can bind to it.

Nothing here targets a real node: the synthetic cluster has no registered
Node Agents and the seeded node ids are names no cluster carries.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import seeded_command_fixture as seeded  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)
from scripts.perf.regional_capacity_registry import (  # noqa: E402
    STORE_DSN_SNIPPET,
)

NetCommandProbe = seeded.SeededCommandProbe
NetCommandError = seeded.SeededCommandError
SYNTHETIC_CLUSTER_ID = seeded.SYNTHETIC_CLUSTER_ID
RELEASE_STATE_CONFIGMAP = "gpu-fault-regional-release-state"

# One import site for the runners: the perf-suite kubectl wrappers and the
# seeded-fixture helpers they already used by these names.
AWS_REGION = seeded.AWS_REGION
CONTROL_NAMESPACE = seeded.CONTROL_NAMESPACE
DATAPLANE_CONTEXT = seeded.DATAPLANE_CONTEXT
NAMESPACE = seeded.NAMESPACE
control = seeded.control
dataplane = seeded.dataplane
require_environment = seeded.require_environment
registry_residuals = seeded.registry_residuals
kubernetes_residuals = seeded.kubernetes_residuals
register_synthetic_cluster = seeded.register_synthetic_cluster
create_probe_pod = seeded.create_probe_pod
wait_file = seeded.wait_file
file_present = seeded.file_present
read_state = seeded.read_state
touch = seeded.touch
remove = seeded.remove
pod_logs = seeded.pod_logs
pod_phase = seeded.pod_phase
wait_executor_state = seeded.wait_executor_state
residual_free = seeded.residual_free

write_json = write_json_atomic

_cpu_pod_name: str | None = None


def cpu_pod(*, refresh: bool = False) -> str:
    """The CPU API Pod every store script runs in, looked up once per process.

    ``wait_command`` polls the store every two seconds for minutes; a
    ``kubectl get pod`` before each poll doubled the API-server traffic of the
    hot path for no information the previous call did not have.
    """

    global _cpu_pod_name
    if refresh or not _cpu_pod_name:
        name = control(
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ).strip()
        if not name:
            raise NetCommandError("no Running gpu-fault-api-ha Pod")
        _cpu_pod_name = name
    return _cpu_pod_name


def reset_cpu_pod_cache() -> None:
    global _cpu_pod_name
    _cpu_pod_name = None


def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
    """Run ``script`` inside the cached CPU API Pod and return its last JSON line.

    A failed exec drops the cache and retries once against a fresh lookup, so
    a Pod that was replaced mid-case (a rollout, an eviction) costs one retry
    instead of failing every later store read.
    """

    def attempt(pod: str) -> dict[str, Any]:
        output = control(
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-",
            *arguments,
            stdin=script.encode(),
            timeout=120,
        )
        lines = output.splitlines()
        if not lines:
            raise NetCommandError("store script produced no output")
        value = json.loads(lines[-1])
        if not isinstance(value, dict):
            raise NetCommandError("store script did not print a JSON object")
        return value

    try:
        return attempt(cpu_pod())
    except RuntimeError:
        return attempt(cpu_pod(refresh=True))


def run_identity(run_dir: Path, attempt: int, prefix: str) -> str:
    """A run id that is unique per run directory, attempt and second.

    The previous form took the last ``-``-separated token of the run
    directory's *name*; a run directory called ``acceptance`` or ``net`` gave
    every run the same id, and two runs in one second under differently named
    directories could still collide. The path digest and the timestamp make
    the id unique without depending on how the operator named the directory.
    """

    if not prefix or not prefix.replace("-", "").isalnum():
        raise ValueError("run id prefix must be alphanumeric with dashes")
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:6]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}{digest}-a{attempt}"


# --------------------------------------------------------------------------- #
# Residuals
# --------------------------------------------------------------------------- #
_DATABASE_RESIDUALS = (
    STORE_DSN_SNIPPET
    + r"""
import json
import sys
import psycopg

prefix = sys.argv[1]
like = "%" + prefix + "%"
queries = {
    "objects": (
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE key LIKE %s OR payload->>'cluster_id' LIKE 'perf-cap-%%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links WHERE key LIKE %s OR value LIKE %s"
    ),
    "processor_queue": (
        "SELECT count(*) FROM gpu_fault_processor_queue "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "processor_lanes": (
        "SELECT count(*) FROM gpu_fault_processor_lanes "
        "WHERE ordering_key LIKE 'perf-cap-%%'"
    ),
    "processor_queue_counts": (
        "SELECT count(*) FROM gpu_fault_processor_queue_counts "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "gpu_metric_latest": (
        "SELECT count(*) FROM gpu_fault_gpu_metric_latest "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "gpu_metric_batches": (
        "SELECT count(*) FROM gpu_fault_gpu_metrics_batches "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "attempt_observations": (
        "SELECT count(*) FROM gpu_fault_attempt_observations "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "training_progress": (
        "SELECT count(*) FROM gpu_fault_training_progress "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
}
parameters = {"objects": (like,), "links": (like, like)}
result = {}
with psycopg.connect(store_dsn()) as connection:
    with connection.cursor() as cursor:
        for name, query in queries.items():
            cursor.execute(query, parameters.get(name, ()))
            result[name] = int(cursor.fetchone()[0])
result["total"] = sum(result.values())
print(json.dumps(result, sort_keys=True))
"""
)


def database_residuals(run_prefix: str) -> dict[str, Any]:
    """Rows the case may have left behind, across every table it can touch."""

    return cpu_python(_DATABASE_RESIDUALS, run_prefix)


def preflight_residuals(probe: NetCommandProbe, case_dir: Path) -> dict[str, Any]:
    database = database_residuals(probe.run_prefix)
    write_json(case_dir / "database-preflight.json", database)
    if database.get("total") != 0:
        raise NetCommandError(f"database preflight found residuals: {database}")
    registry = registry_residuals()
    write_json(case_dir / "registry-verification-preflight.json", registry)
    if registry.get("count") != 0:
        raise NetCommandError(f"registry preflight found residuals: {registry}")
    kubernetes = kubernetes_residuals(probe)
    write_json(case_dir / "kubernetes-preflight.json", kubernetes)
    if kubernetes.get("count") != 0:
        raise NetCommandError(f"Kubernetes preflight found residuals: {kubernetes}")
    return {"database": database, "registry": registry, "kubernetes": kubernetes}


# --------------------------------------------------------------------------- #
# Store reads
# --------------------------------------------------------------------------- #
def command_snapshot(command_id: str) -> dict[str, Any]:
    """The seeded fixture's snapshot (status, lease, ``updated_at``), cached Pod."""

    return cpu_python(seeded._COMMAND_SNAPSHOT, command_id)


def wait_command(
    command_id: str,
    status: str,
    timeout_seconds: int,
    *,
    poll_seconds: float = 2.0,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    tick = clock or time.monotonic
    pause = sleep or time.sleep
    deadline = tick() + timeout_seconds
    last: dict[str, Any] = {}
    while tick() < deadline:
        last = command_snapshot(command_id)
        if last.get("status") == status:
            return last
        pause(poll_seconds)
    raise NetCommandError(f"command did not reach {status}: {last}")


def seed_command(
    run_id: str, *, owner: str, operation: str, node_ids: list[str]
) -> dict[str, Any]:
    return seeded.seed_command(
        run_id, owner=owner, operation=operation, node_ids=node_ids
    )


def purge_seed(seed: dict[str, Any]) -> dict[str, Any]:
    return seeded.purge_seed(seed)


# --------------------------------------------------------------------------- #
# Identity and predecessor
# --------------------------------------------------------------------------- #
def release_id() -> str:
    """The release the CPU control plane runs, from its release-state ConfigMap.

    The same document ``RegionalLiveFixture.release_id`` reads; the network
    runners have no ``RegionalLiveFixture`` (their kubectl wrappers come from
    the perf suite), so they read it here and record it in the case result.
    """

    value = json.loads(
        control("get", "configmap", RELEASE_STATE_CONFIGMAP, "-o", "json")
    )
    state = json.loads(value["data"]["state.json"])
    return str(state.get("release_id") or "")


def evidence_identity(cluster_id: str | None) -> dict[str, str | None]:
    """What a case result must carry for ``predecessor_evidence`` to bind to it."""

    return {"release_id": release_id() or None, "cluster_id": cluster_id or None}


def predecessor_gate(
    run_dir: Path, case_id: str, explicit_path: str | Path | None
) -> dict[str, Any]:
    """The predecessor verdict a runner records in its plan and refuses on."""

    predecessor_id, path = predecessor_path(run_dir, case_id, explicit_path)
    if predecessor_id is None or path is None:
        return {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    return predecessor_evidence(path, predecessor_id)


def require_predecessor(predecessor: dict[str, Any]) -> None:
    if not predecessor.get("valid", False):
        raise NetCommandError("formal predecessor evidence is not PASS")


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def cleanup(
    probe: NetCommandProbe,
    case_dir: Path,
    run_id: str,
    result: dict[str, Any],
    seed: dict[str, Any],
    *,
    purge: Callable[[dict[str, Any]], dict[str, Any]] = purge_seed,
) -> None:
    """Undo everything the case created; any failure downgrades to FAIL.

    ``purge`` is pluggable because NET-003 seeds a notification alongside the
    command and has to delete it too. The residual check afterwards is this
    module's nine-table one.
    """

    if seed:
        try:
            seed_cleanup = purge(seed)
            result["seed_cleanup"] = seed_cleanup
            write_json(case_dir / "seed-cleanup.json", seed_cleanup)
        except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
            result["cleanup_error"] = f"seed cleanup: {type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    dataplane("delete", "pod", probe.pod, "--ignore-not-found", check=False)
    dataplane("delete", "configmap", probe.configmap, "--ignore-not-found", check=False)
    try:
        seeded.teardown(
            purge=True,
            deregister_clusters=True,
            allow_live_registry=True,
            live_registry_confirmation=seeded.LIVE_REGISTRY_CONFIRMATION,
            artifacts=case_dir,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
        result["cleanup_error"] = f"registry cleanup: {type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    try:
        postflight = {
            "database": database_residuals(probe.run_prefix),
            "registry": registry_residuals(),
            "kubernetes": kubernetes_residuals(probe),
        }
        names = {
            "database": "database-postflight.json",
            "registry": "registry-verification-postflight.json",
            "kubernetes": "kubernetes-postflight.json",
        }
        for name, value in postflight.items():
            result[f"{name}_postflight"] = value
            write_json(case_dir / names[name], value)
            key = "total" if name == "database" else "count"
            if value.get(key) != 0:
                raise NetCommandError(f"{name} postflight found residuals: {value}")
    except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def run_main(
    *,
    case_id: str,
    confirmation: str,
    parser: Callable[[], Any],
    plan_details: Callable[[dict[str, Any]], dict[str, Any]],
    run_case: Callable[..., int],
) -> int:
    """The ``main`` NET-002 and NET-003 share.

    Site profile first, parse, ``0o077`` umask, then the predecessor gate. A
    plan run prints the plan and exits 0 only when the predecessor evidence is
    valid; ``--execute`` must pass ``authorize_execution`` before the case body
    runs, and the case verdict is the exit code.
    """

    install_site_profile()
    args = parser().parse_args()
    os.umask(0o077)
    predecessor = predecessor_gate(args.run_dir, case_id, args.predecessor_evidence)
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=case_id,
            attempt=args.attempt,
            confirmation=confirmation,
            details=plan_details(predecessor),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid", False) else 1
    deadline = authorize_execution(args, case_id=case_id, confirmation=confirmation)
    return run_case(
        args.run_dir,
        args.attempt,
        deadline,
        predecessor=predecessor,
        cluster_id=args.cluster_id or None,
    )
