#!/usr/bin/env python3
"""Host controller of the GF-GENERAL-LIVE DCGM field replay (live-replay).

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged
probe Pod that ``chroot``s into the host and runs this file with the Agent's
venv Python), so ``gpu_fault`` below is the deployed collector code: every
reading the survey reports is parsed by the product's own parser. It drives
one reviewed loopback replay tool -- ``tools/gpu_dcgm_field_replay.py`` (the
per-category DCGM field profiles) or, for the thermal / power parts of the
composites, the environment tool ``tools/gpu_environment_replay.py`` unchanged
-- and repoints the node's single metrics collector at it. Everything it
changes is bounded and restored two ways, by the runner's cleanup and by a
``systemd-run`` deadman that needs neither the runner, the probe Pod nor the
control plane:

* **survey** -- read-only facts, including every object an earlier run left
  (``residue``): per run digest the drop-in, proxy unit, run directory and
  deadman timer (with its fire time and whether it is still pending), plus the
  drop-ins / directories that are not this fixture's. The runner refuses to
  start while anything is listed.
* **install-source** -- the tool, the reviewed plan and, for the inventory
  variants, the reviewed ``nvidia-smi`` pass-through wrapper, handed in base64
  and checked against a sha256, under ``/run/gfrs-live/<digest>/`` (tmpfs; a
  ``run.json`` marker names the owning run). The files run verbatim.
* **deploy-proxy** -- a transient unit runs the tool ``--serve`` against the
  real exporter on a loopback-only port (``IPAddressDeny=any``,
  ``IPAddressAllow=localhost``, ``RuntimeMaxSec``); it starts pass-through and
  only replays after ``arm``; a port a tool still answers on is refused.
* **repoint-collector** -- the deadman is armed *first*; then a ``/run``
  drop-in adds an ``EnvironmentFile=-/run/gfrs-live/<digest>/collector.env``
  to the one allow-listed unit (``gpu-fault-metrics-collector.service``) --
  an environment file, not ``Environment=``, because a later environment file
  is what overrides ``collector.env`` (systemd.exec), and ``-`` lets a
  vanished override fall back to the real endpoint. The effective endpoint is
  read from the running process (``/proc/<MainPID>/environ``), never from
  ``systemctl show``, and only once that process has exec'd: a ``Type=simple``
  unit is active from the fork on, and until the child has built its sandbox
  and exec'd its environ is still systemd's own (live 2026-10-25: that first
  read lacked the endpoint and the repoint was judged failed). The Node Agent
  is never in the allow-list.
* **arm / disarm / status / audit-read** -- loopback control of the tool;
  ``arm --before-repoint`` arms while the collector still reads the real
  exporter (022 ``source_boot_change``: the repoint's first scrape is the
  plan's first read); **collector-restart** -- one ``systemctl restart`` of
  the same unit.
* **close-run** -- the runner's cleanup: disarm the tool if it answers, remove
  the drop-in and override, restart the unit on the real endpoint and verify
  the process, stop the tool, remove the run directory, then disarm the
  deadman. Every step runs whatever an earlier step did (a repoint that failed
  after writing its drop-in is undone like a completed run); the answer names
  what is left and keeps the deadman armed while anything is.
* **cleanup-leftover** -- the same steps for an earlier run's objects, by
  digest, only when that run's deadman is no longer pending; a pending one is
  reported and left to fire.
* **restore-collector / stop-proxy / cleanup** -- the single steps; ``cleanup``
  is the deadman's entry point (non-zero on a failed step, so its
  ``Restart=on-failure`` unit tries again).

Every answer is one JSON object on the last stdout line; only allow-listed
environment keys are read, so secrets never leave the node. A failure prints
the exception class, a digest of its message, its ``function:line`` site and a
fixed reason token -- what the host fixture may show the runner -- plus the
message the fixture keeps, redacted, as the case's ``probe-error.json``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

STATE_ROOT = Path("/run/gfrs-live")
COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
VENV_PYTHON = Path("/opt/gpu-fault/current/venv/bin/python")
#: The one unit a window may repoint. The Node Agent and every other unit are
#: deliberately absent.
METRICS_UNIT = "gpu-fault-metrics-collector.service"
DROPIN_DIR = Path("/run/systemd/system") / f"{METRICS_UNIT}.d"
ENDPOINT_ENV = "GPU_FAULT_DCGM_METRICS_URL"
DEFAULT_UPSTREAM = "http://127.0.0.1:9400/metrics"
ACKNOWLEDGEMENT = "ISOLATED_GPU_TEST_NODE"
FIELD_TOOL = "gpu_dcgm_field_replay.py"
ENVIRONMENT_TOOL = "gpu_environment_replay.py"
INVENTORY_WRAPPER = "nvidia_smi_inventory_replay.py"
PLAN_NAME = "plan.json"
INVENTORY_PLAN = "inventory-plan.json"
#: The only files that may be installed, and what each one is.
SOURCES = {
    FIELD_TOOL: "tool",
    ENVIRONMENT_TOOL: "tool",
    PLAN_NAME: "plan",
    INVENTORY_WRAPPER: "wrapper",
    INVENTORY_PLAN: "plan",
}
UNIT_PREFIX = "gfrs-live-"
LIST_UNITS = ("systemctl", "list-units", "--all", "--plain", "--no-legend")
#: ``systemctl show`` properties a unit state carries; ``Result`` and
#: ``ExecMainStatus`` say how a dead unit died (the tool's exit code).
UNIT_PROPERTIES = (
    "LoadState ActiveState SubState MainPID NRestarts InvocationID DropInPaths "
    "ExecMainStartTimestamp NextElapseUSecRealtime Result ExecMainStatus"
).split()
#: The last lines of a dead unit's journal: the tool's own refusal or traceback.
JOURNAL_TAIL = ("journalctl", "--no-pager", "-o", "cat", "-n", "40", "-u")
#: Written into a run directory when it is created: which run owns it.
RUN_MARKER = "run.json"
#: Collector settings a plan may depend on; nothing else is ever read out.
ENV_KEYS = (
    ENDPOINT_ENV,
    "GPU_FAULT_METRICS_MODE",
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
    "GPU_FAULT_DCGM_EDGE_FILTER_ENABLED",
    "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
    "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS",
    "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS",
    "GPU_FAULT_DCGM_STARTUP_GRACE_SECONDS",
    "GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS",
    "GPU_FAULT_DCGM_CORRELATION_WINDOW_SECONDS",
    "GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE",
    "GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
)
#: Process-environment keys the repoint verification may read. The scrape
#: interval is what the runner's oracle gate paces its wait by.
PROCESS_KEYS = (
    ENDPOINT_ENV,
    "PATH",
    "GPU_FAULT_BOOT_ID_PATH",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
    "GPU_FAULT_NODE_INSTANCE_TYPE",
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
)
#: The expected-count inputs the unknown-count variant blanks (discovery.py:
#: ``expected_gpu_count_from_environment``); an unknown instance type answers
#: ``None``, never a guessed count.
UNKNOWN_INSTANCE_TYPE = "gfrs-live-unknown.48xlarge"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
#: A run's on-node objects are named by the 12-hex digest of its run id.
DIGEST = re.compile(r"^[0-9a-f]{12}$")
BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,64}$")
#: A field-tool profile name (``tools/gpu_dcgm_field_replay.py`` PROFILES):
#: lowercase words joined by single underscores. Live 2026-10-26 a
#: ``[a-z]{3,16}`` shape refused ``pcie_xid`` -- the one underscored profile --
#: and the three 022 XID-join variants ended INCONCLUSIVE at ``deploy-proxy``.
PROFILE_NAME = re.compile(r"^[a-z]{2,16}(?:_[a-z]{2,16}){0,2}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MIN_PORT, MAX_PORT = 1024, 65535
MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS = 120, 5400
MAX_TOOL_SECONDS = 3600
MAX_SOURCE_BYTES = 512 * 1024
MAX_AUDIT_LINES = 400
#: How long a restarted collector may take between systemd's fork (the unit
#: is "active" from here) and its exec (its environment is the unit's from
#: here). The sandbox setup in between (``ProtectSystem=strict`` over every
#: mount of a Kubernetes node) is what makes this a wait, not an instant.
SETTLE_SECONDS = 45
#: Unit states in which a deadman is still going to restore, or restoring.
PENDING_TIMER_STATES = frozenset({"active", "activating", "reloading"})
PENDING_SERVICE_STATES = frozenset(
    {"active", "activating", "reloading", "deactivating"}
)


class ProbeError(RuntimeError):
    def __init__(self, message: str, reason: str = "", detail: Any = None) -> None:
        super().__init__(message)
        #: ``reason``: a fixed identifier of the refusal (``profile_shape``,
        #: ``unit_down``) from this file's closed vocabulary, shown by the host
        #: fixture beside the class, digest and site. ``detail``: JSON the
        #: fixture keeps, redacted, in the case's probe-error.json (a journal).
        self.reason = reason
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {command[0]}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def checked_digest(value: str) -> str:
    if DIGEST.fullmatch(value) is None:
        raise ProbeError("unsafe run digest")
    return value


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:12]


# Every on-node object of a run derives from the run id's digest; the digest
# forms let an earlier run's leftovers be addressed when only their names are
# known (``cleanup-leftover``).
def run_dir_of(run_digest: str) -> Path:
    return STATE_ROOT / checked_digest(run_digest)


def proxy_unit_of(run_digest: str) -> str:
    return f"{UNIT_PREFIX}{checked_digest(run_digest)}-proxy"


def deadman_unit_of(run_digest: str) -> str:
    return f"{UNIT_PREFIX}{checked_digest(run_digest)}-restore"


def dropin_of(run_digest: str) -> Path:
    return DROPIN_DIR / f"{UNIT_PREFIX}{checked_digest(run_digest)}.conf"


def override_env_of(run_digest: str) -> Path:
    return run_dir_of(run_digest) / "collector.env"


def run_dir(run_id: str) -> Path:
    return run_dir_of(digest(run_id))


def proxy_unit(run_id: str) -> str:
    return proxy_unit_of(digest(run_id))


def deadman_unit(run_id: str) -> str:
    return deadman_unit_of(digest(run_id))


def dropin_path(run_id: str) -> Path:
    return dropin_of(digest(run_id))


def override_env_path(run_id: str) -> Path:
    return override_env_of(digest(run_id))


def checked_port(value: int) -> int:
    return bounded(value, MIN_PORT, MAX_PORT, "port")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def collector_env() -> dict[str, str]:
    values = parse_env_file(COLLECTOR_ENV)
    return {key: values[key] for key in ENV_KEYS if key in values}


def baseline_endpoint() -> str:
    return parse_env_file(COLLECTOR_ENV).get(ENDPOINT_ENV) or DEFAULT_UPSTREAM


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        ["systemctl", "show", unit, *(f"--property={p}" for p in UNIT_PROPERTIES)],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    values["returncode"] = str(completed.returncode)
    return values


def loaded(state: dict[str, str]) -> bool:
    return state.get("LoadState") == "loaded"


def journal(unit: str, state: dict[str, str]) -> dict[str, Any]:
    """Why a transient unit died: its exit result and its last journal lines
    (the tool's argparse refusal, a traceback), bounded; the host fixture
    redacts every string before anything is written."""

    tail = run([*JOURNAL_TAIL, f"{unit}.service"], check=False)
    lines = [line[:300] for line in tail.stdout.splitlines()[-40:]]
    return {"unit": f"{unit}.service", "service": state, "journal": lines}


def process_environment(pid: str) -> dict[str, str]:
    """The allow-listed keys of a running process's real environment."""

    if not pid or not pid.isdigit() or pid == "0":
        return {}
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        name = key.decode("utf-8", "replace")
        if separator and name in PROCESS_KEYS:
            values[name] = value.decode("utf-8", "replace")
    return values


def has_execed(pid: str) -> bool:
    """Whether ``pid`` runs its own program yet, not systemd's pre-exec child.

    systemd names the forked child ``(<exec basename>)`` until it execs, and
    its ``exe`` is still systemd's; from the exec on, ``comm`` and ``exe`` are
    the service's and ``/proc/<pid>/environ`` is the unit's environment.
    """

    try:
        comm = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace")
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return False
    return not comm.strip().startswith("(") and Path(exe).name != "systemd"


def wait_active(unit: str, *, seconds: int = 60) -> dict[str, str]:
    deadline = time.monotonic() + seconds
    state = unit_state(unit)
    while time.monotonic() < deadline:
        state = unit_state(unit)
        active = state.get("ActiveState") == "active"
        if active and state.get("MainPID") not in {"", "0"}:
            return state
        time.sleep(1)
    return state


def settled_environment(
    unit: str, *, seconds: int = SETTLE_SECONDS
) -> tuple[dict[str, str], dict[str, str]]:
    """The unit's state and its main process's environment once it has exec'd.

    Returns what it last saw when the window closes; callers check for the
    endpoint key, which every exec'd collector carries from ``collector.env``.
    """

    deadline = time.monotonic() + seconds
    state = wait_active(unit, seconds=seconds)
    while True:
        pid = state.get("MainPID", "")
        environment = process_environment(pid)
        if (
            state.get("ActiveState") == "active"
            and pid not in {"", "0"}
            and (ENDPOINT_ENV in environment or has_execed(pid))
        ):
            return state, environment
        if time.monotonic() >= deadline:
            return state, environment
        time.sleep(0.5)
        state = unit_state(unit)


def effective_endpoint(*, seconds: int = 15) -> str | None:
    """The DCGM URL the running collector process actually uses."""

    _state, environment = settled_environment(METRICS_UNIT, seconds=seconds)
    return environment.get(ENDPOINT_ENV)


# --------------------------------------------------------------------------- #
# Loopback HTTP to the tool (never a proxy environment, never a redirect)
# --------------------------------------------------------------------------- #
def loopback(url: str, *, method: str = "GET", timeout: int = 10) -> tuple[int, bytes]:
    if not url.startswith("http://127.0.0.1:"):
        raise ProbeError("only loopback http is allowed")
    request = Request(url, method=method)
    if method == "POST":
        request.add_header("X-GFRS-Replay-Ack", ACKNOWLEDGEMENT)
        request.data = b""
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return int(response.status), response.read(4 * 1024 * 1024)
    except OSError as error:
        raise ProbeError(f"loopback {method} failed: {error}") from error


def tool_status(port: int) -> dict[str, Any]:
    status, body = loopback(f"http://127.0.0.1:{checked_port(port)}/status")
    if status != 200:
        raise ProbeError(f"replay /status returned {status}")
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ProbeError("replay /status did not return an object")
    return value


def port_answers(port: int) -> bool:
    """Whether a replay tool (ours or an earlier run's) answers on ``port``."""

    try:
        tool_status(port)
    except (ProbeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# survey: read-only facts a plan is built from
# --------------------------------------------------------------------------- #
def gpu_inventory() -> list[dict[str, str]]:
    query = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name"]
    completed = run([*query, "--format=csv,noheader"], check=False, timeout=60)
    keys = ("index", "uuid", "pci_bus_id", "name")
    result = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) == 4:
            result.append(dict(zip(keys, values)))
    return result


def exporter_readings(url: str) -> dict[str, Any]:
    """The real scrape as the product's parser reads it, per GPU."""

    from gpu_fault.collectors.gpu.dcgm import parse_dcgm_samples
    from gpu_fault.dcgm_fields import missing_dcgm_metric_groups

    try:
        status, body = loopback(url, timeout=10)
    except ProbeError as error:
        return {"error": str(error)}
    if status != 200:
        return {"error": f"exporter returned {status}"}
    samples = parse_dcgm_samples(body.decode("utf-8", "replace"))
    gpus: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not sample.gpu_uuid:
            continue
        entry = gpus.setdefault(
            sample.gpu_uuid,
            {
                "gpu_index": sample.gpu_index,
                "pci_bdf": sample.pci_bdf,
                "model": sample.labels.get("modelName"),
                "values": {},
                "metric_names": {},
            },
        )
        entry["values"][sample.canonical_name] = sample.value
        entry["metric_names"].setdefault(sample.canonical_name, []).append(
            sample.metric_name
        )
    return {
        "url": url,
        "sample_count": len(samples),
        "sha256": hashlib.sha256(body).hexdigest(),
        "gpus": gpus,
        "batch_missing_groups": missing_dcgm_metric_groups(
            {sample.canonical_name for sample in samples}
        ),
        "per_gpu_missing_groups": {
            uuid: missing_dcgm_metric_groups(set(entry["values"]))
            for uuid, entry in gpus.items()
        },
    }


def recorded_run_id(run_digest: str) -> str | None:
    """The run id a leftover directory names for itself, if it checks out."""

    for name in (RUN_MARKER, "repoint.json"):
        path = run_dir_of(run_digest) / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("run_id")
        except (OSError, ValueError, AttributeError):
            continue
        if (
            isinstance(value, str)
            and SAFE_RUN_ID.fullmatch(value)
            and digest(value) == run_digest
        ):
            return value
    return None


def deadman_pending(timer: dict[str, str], service: dict[str, str]) -> bool:
    """Whether a run's deadman is still going to restore (or is restoring)."""

    return (
        timer.get("ActiveState") in PENDING_TIMER_STATES
        or service.get("ActiveState") in PENDING_SERVICE_STATES
    )


def run_state(run_digest: str) -> dict[str, Any]:
    """Every object one run (by digest) still has on the node, named."""

    run_digest = checked_digest(run_digest)
    root, dropin = run_dir_of(run_digest), dropin_of(run_digest)
    proxy_name = f"{proxy_unit_of(run_digest)}.service"
    timer_name = f"{deadman_unit_of(run_digest)}.timer"
    service_name = f"{deadman_unit_of(run_digest)}.service"
    proxy, timer, service = map(unit_state, (proxy_name, timer_name, service_name))
    run_id = recorded_run_id(run_digest)
    objects: list[str] = []
    if dropin.exists():
        objects.append(f"drop-in {dropin}")
    if loaded(proxy):
        objects.append(f"unit {proxy_name} ({proxy.get('ActiveState')})")
    if root.is_dir():
        objects.append(f"run dir {root}" + (f" (run {run_id})" if run_id else ""))
    pending = deadman_pending(timer, service)
    if loaded(timer):
        fires = timer.get("NextElapseUSecRealtime") or "n/a"
        objects.append(
            f"deadman {timer_name} ({timer.get('ActiveState')}; fires {fires})"
        )
    if loaded(service):
        objects.append(
            f"deadman {service_name} ({service.get('ActiveState')}/{service.get('SubState')})"
        )
    return {
        "digest": run_digest,
        "run_id": run_id,
        "objects": objects,
        "deadman_pending": pending,
        "reclaimable": bool(objects) and not pending,
        "dropin": str(dropin) if dropin.exists() else None,
        "run_dir": str(root) if root.is_dir() else None,
        "proxy_unit": {"name": proxy_name, **proxy} if loaded(proxy) else None,
        "deadman_timer": {"name": timer_name, **timer} if loaded(timer) else None,
        "deadman_service": {"name": service_name, **service}
        if loaded(service)
        else None,
    }


def residue(port: int | None = None) -> dict[str, Any]:
    """What is on the node from any replay run, ours by digest and foreign.

    The flat lists are kept for older readers; ``runs`` groups this fixture's
    objects by run digest with the deadman's state, ``foreign_*`` name what
    this fixture never touches, and ``port_answers`` says whether a tool
    (an earlier run's) still serves the replay port.
    """

    dropins = (
        sorted(path.name for path in DROPIN_DIR.glob("*.conf"))
        if DROPIN_DIR.is_dir()
        else []
    )
    listing = run([*LIST_UNITS, f"{UNIT_PREFIX}*"], check=False).stdout.split()
    units = sorted(item for item in listing if item.startswith(UNIT_PREFIX))
    run_dirs = (
        sorted(path.name for path in STATE_ROOT.glob("*") if path.is_dir())
        if STATE_ROOT.is_dir()
        else []
    )
    prefix = re.escape(UNIT_PREFIX)
    digests: set[str] = set()
    foreign_dropins: list[str] = []
    foreign_run_dirs: list[str] = []
    for name in dropins:
        match = re.fullmatch(rf"{prefix}([0-9a-f]{{12}})\.conf", name)
        if match:
            digests.add(match.group(1))
        else:
            foreign_dropins.append(name)
    for name in units:
        match = re.fullmatch(
            rf"{prefix}([0-9a-f]{{12}})-(?:proxy|restore)\.(?:service|timer)", name
        )
        if match:
            digests.add(match.group(1))
    for name in run_dirs:
        if DIGEST.fullmatch(name):
            digests.add(name)
        else:
            foreign_run_dirs.append(name)
    return {
        "run_dirs": run_dirs,
        "metrics_dropins": dropins,
        "units": units,
        "runs": {item: run_state(item) for item in sorted(digests)},
        "foreign_dropins": foreign_dropins,
        "foreign_run_dirs": foreign_run_dirs,
        "port": port,
        "port_answers": port_answers(port) if port is not None else None,
    }


def prior_boots() -> list[str]:
    """Earlier boot ids of this node from the journal, newest first."""

    completed = run(["journalctl", "--list-boots", "--no-pager", "-q"], check=False)
    current = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    result = []
    for line in completed.stdout.splitlines():
        for token in line.split():
            if re.fullmatch(r"[0-9a-f]{32}", token):
                dashed = "-".join(
                    (token[:8], token[8:12], token[12:16], token[16:20], token[20:])
                )
                if dashed != current and dashed not in result:
                    result.append(dashed)
    return list(reversed(result))[:5]


def temperature_limits() -> list[dict[str, Any]]:
    """Per-GPU temperature limits through the product's own XML parser."""

    from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits

    by_gpu: dict[str, dict[str, Any]] = {}
    for sample in query_nvidia_temperature_limits():
        key = sample.gpu_uuid or sample.gpu_index or "gpu"
        entry = by_gpu.setdefault(
            key,
            {
                "gpu_uuid": sample.gpu_uuid,
                "gpu_index": sample.gpu_index,
                "pci_bdf": sample.pci_bdf,
                "product": sample.labels.get("modelName"),
                "limits": {},
            },
        )
        entry["limits"][sample.canonical_name] = sample.value
    return [by_gpu[key] for key in sorted(by_gpu)]


def survey(arguments: argparse.Namespace) -> None:
    checked_run_id(arguments.run_id)
    state = unit_state(METRICS_UNIT)
    environment = collector_env()
    extra: dict[str, Any] = {}
    if arguments.limits:
        extra["temperature_limits"] = temperature_limits()
    emit(
        {
            **extra,
            "prior_boot_ids": prior_boots(),
            "captured_at": utc_now(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "collector_env": environment,
            "metrics_unit": state,
            "process_environment": process_environment(state.get("MainPID", "")),
            "baseline_endpoint": baseline_endpoint(),
            "gpu_inventory": gpu_inventory(),
            "exporter": exporter_readings(baseline_endpoint()),
            "residue": residue(checked_port(arguments.port)),
            "tools": {
                name: bool(shutil.which(name))
                for name in ("systemd-run", "systemctl", "nvidia-smi")
            },
            "venv_python": VENV_PYTHON.is_file(),
        }
    )


# --------------------------------------------------------------------------- #
# install-source: reviewed files under /run/gfrs-live/<digest>
# --------------------------------------------------------------------------- #
def install_source(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = arguments.name
    if name not in SOURCES:
        raise ProbeError(
            "only the reviewed replay tools, wrapper and plans may be installed"
        )
    if SHA256.fullmatch(arguments.sha256) is None:
        raise ProbeError("a sha256 is required for every installed source")
    try:
        content = base64.b64decode(arguments.b64, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError(f"source is not valid base64: {error}") from error
    if len(content) > MAX_SOURCE_BYTES:
        raise ProbeError("installed source exceeds its size bound")
    actual = hashlib.sha256(content).hexdigest()
    if actual != arguments.sha256:
        raise ProbeError("installed source sha256 does not match")
    root = run_dir(run_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = root / RUN_MARKER
    if not marker.exists():
        record = {"run_id": run_id, "digest": digest(run_id), "created_at": utc_now()}
        marker.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        marker.chmod(0o600)
    target = root / name
    if target.exists():
        raise ProbeError(f"{name} is already installed for this run")
    target.write_bytes(content)
    target.chmod(0o700 if SOURCES[name] == "wrapper" else 0o600)
    emit({"run_id": run_id, "name": name, "sha256": actual, "bytes": len(content)})


def installed(run_id: str, name: str) -> Path:
    path = run_dir(run_id) / name
    if not path.is_file():
        raise ProbeError(f"install {name} before using it", "source_missing")
    return path


# --------------------------------------------------------------------------- #
# deploy-proxy: the tool --serve under a bounded, loopback-only transient unit
# --------------------------------------------------------------------------- #
def tool_command(arguments: argparse.Namespace, root: Path) -> list[str]:
    tool = installed(arguments.run_id, arguments.tool)
    plan = installed(arguments.run_id, PLAN_NAME)
    audit = root / "replay.jsonl"
    if audit.exists():
        raise ProbeError("an audit file already exists; start a new audited run")
    upstream = baseline_endpoint()
    if not upstream.startswith("http://127.0.0.1:") or not upstream.endswith(
        "/metrics"
    ):
        raise ProbeError("the real exporter endpoint is not loopback /metrics")
    # ``-I`` keeps a file planted beside the tool from being imported; the
    # venv's own gpu_fault stays importable (site-packages, not user site).
    port = str(checked_port(arguments.port))
    seconds = str(bounded(arguments.max_seconds, 60, MAX_TOOL_SECONDS, "max seconds"))
    common = [str(VENV_PYTHON), "-I", str(tool), "--plan", str(plan), "--upstream"]
    common += [upstream, "--port", port, "--max-seconds", seconds, "--serve", "--ack"]
    common += [ACKNOWLEDGEMENT, "--audit", str(audit)]
    if arguments.tool == FIELD_TOOL:
        if PROFILE_NAME.fullmatch(arguments.profile or "") is None:
            raise ProbeError(
                "the field tool needs its plan's --profile", "profile_shape"
            )
        return [*common, "--profile", arguments.profile]
    if GPU_UUID.fullmatch(arguments.gpu_uuid or "") is None:
        raise ProbeError("the environment tool needs a physical --gpu-uuid")
    if int(arguments.max_seconds) > 1800:
        raise ProbeError("the environment tool runs at most 1800 s")
    return [*common, "--gpu-uuid", arguments.gpu_uuid]


def deploy_proxy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    if arguments.tool not in {FIELD_TOOL, ENVIRONMENT_TOOL}:
        raise ProbeError("only the reviewed replay tools may be served")
    root = run_dir(run_id)
    command = tool_command(arguments, root)
    port = checked_port(arguments.port)
    unit = proxy_unit(run_id)
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    if port_answers(port):
        raise ProbeError(
            "a replay tool already answers on the port; not starting a second",
            "port_busy",
        )
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            "--property=Type=simple",
            f"--property=RuntimeMaxSec={int(arguments.max_seconds) + 120}s",
            "--property=NoNewPrivileges=true",
            "--property=ProtectSystem=strict",
            "--property=ProtectHome=true",
            "--property=PrivateTmp=true",
            f"--property=ReadWritePaths={root}",
            "--property=IPAddressDeny=any",
            "--property=IPAddressAllow=localhost",
            "--property=MemoryMax=512M",
            "--property=CPUQuota=50%",
            "--property=TasksMax=32",
            "--setenv=no_proxy=*",
            "--setenv=NO_PROXY=*",
            *command,
        ]
    )
    deadline = time.monotonic() + 30
    status: dict[str, Any] = {}
    state: dict[str, str] = {}
    while time.monotonic() < deadline:
        state = unit_state(f"{unit}.service")
        if state.get("ActiveState") == "active":
            try:
                status = tool_status(port)
                break
            except ProbeError:
                pass
        if state.get("ActiveState") in {"failed", "inactive"}:
            raise ProbeError(
                "the replay unit did not stay up", "unit_down", journal(unit, state)
            )
        time.sleep(0.5)
    if not status:
        raise ProbeError(
            "the replay tool did not answer /status", "no_status", journal(unit, state)
        )
    if status.get("armed") is not False:
        raise ProbeError(
            "the replay tool came up armed; it must start pass-through",
            "armed_at_start",
        )
    emit(
        {
            "run_id": run_id,
            "unit": f"{unit}.service",
            "tool": arguments.tool,
            "endpoint": f"http://127.0.0.1:{port}/metrics",
            "upstream": baseline_endpoint(),
            "status": status,
            "service": unit_state(f"{unit}.service"),
        }
    )


# --------------------------------------------------------------------------- #
# repoint-collector: deadman first, then the /run drop-in
# --------------------------------------------------------------------------- #
def probe_copy(run_id: str) -> Path:
    root = run_dir(run_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def arm_deadman(run_id: str, seconds: int) -> dict[str, str]:
    bounded(seconds, MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS, "restore seconds")
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    timer = ["systemd-run", f"--unit={name}", f"--on-active={int(seconds)}s"]
    timer += ["--timer-property=AccuracySec=1s", "--property=Restart=on-failure"]
    timer += ["--property=RestartSec=30s", "--property=StartLimitIntervalSec=0"]
    run(
        [*timer, sys.executable, str(probe_copy(run_id)), "cleanup", "--run-id", run_id]
    )
    state = unit_state(f"{name}.timer")
    if state.get("ActiveState") != "active":
        raise ProbeError("the collector restore deadman is not armed")
    return state


def disarm_deadman_of(run_digest: str) -> None:
    name = deadman_unit_of(run_digest)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


def disarm_deadman(run_id: str) -> None:
    disarm_deadman_of(digest(run_id))


def override_lines(
    run_id: str,
    port: int,
    wrapper: bool,
    *,
    boot_id: str = "",
    unknown_expected_count: bool = False,
) -> list[str]:
    """The allow-listed settings of the unit's override environment file."""

    lines = [f"{ENDPOINT_ENV}=http://127.0.0.1:{checked_port(port)}/metrics"]
    if boot_id:
        if BOOT_ID.fullmatch(boot_id) is None:
            raise ProbeError("a replayed boot id must be a dashed boot UUID")
        current = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot_id == current:
            raise ProbeError("the replayed boot id is the node's current boot")
        path = run_dir(run_id) / "boot_id"
        path.write_text(boot_id + "\n", encoding="ascii")
        path.chmod(0o644)
        lines.append(f"GPU_FAULT_BOOT_ID_PATH={path}")
    if unknown_expected_count:
        lines.append("GPU_FAULT_EXPECTED_GPU_COUNT=")
        lines.append(f"GPU_FAULT_NODE_INSTANCE_TYPE={UNKNOWN_INSTANCE_TYPE}")
    if wrapper:
        installed(run_id, INVENTORY_WRAPPER)
        installed(run_id, INVENTORY_PLAN)
        bin_dir = run_dir(run_id) / "bin"
        # The unit's PATH, not the host's: the wrapper directory first, the
        # systemd default service PATH after it (the real binary stays where
        # the distribution put it; nothing is replaced or bind-mounted).
        lines.append(
            f"PATH={bin_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
        )
        lines.append(f"GFRS_LIVE_INVENTORY_PLAN={run_dir(run_id) / INVENTORY_PLAN}")
        lines.append(f"GFRS_LIVE_INVENTORY_AUDIT={run_dir(run_id) / 'inventory.jsonl'}")
    return lines


def install_wrapper_link(run_id: str) -> Path:
    """``bin/nvidia-smi`` -> the reviewed wrapper (inside the run directory)."""

    wrapper = installed(run_id, INVENTORY_WRAPPER)
    bin_dir = run_dir(run_id) / "bin"
    bin_dir.mkdir(mode=0o755, exist_ok=True)
    link = bin_dir / "nvidia-smi"
    if link.exists() or link.is_symlink():
        link.unlink()
    script = f'#!/bin/sh\nexec {VENV_PYTHON} -I {wrapper} "$@"\n'
    link.write_text(script, encoding="utf-8")
    link.chmod(0o755)
    return link


def repoint_collector(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    before = unit_state(METRICS_UNIT)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active at baseline")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError("a repoint drop-in for this run already exists")
    foreign = (
        [path.name for path in DROPIN_DIR.glob("*.conf") if path != dropin]
        if DROPIN_DIR.is_dir()
        else []
    )
    if foreign:
        raise ProbeError(
            "another runtime drop-in already changes the metrics collector"
        )
    original = baseline_endpoint()
    deadman = arm_deadman(run_id, int(arguments.restore_seconds))
    if arguments.inventory_wrapper:
        install_wrapper_link(run_id)
    override = override_env_path(run_id)
    lines = override_lines(
        run_id,
        port,
        arguments.inventory_wrapper,
        boot_id=arguments.boot_id,
        unknown_expected_count=arguments.unknown_expected_count,
    )
    override.write_text("\n".join(lines) + "\n", encoding="utf-8")
    override.chmod(0o600)
    record = {"run_id": run_id, "original_endpoint": original, "port": port}
    text = json.dumps(record, sort_keys=True)
    (run_dir(run_id) / "repoint.json").write_text(text, encoding="utf-8")
    DROPIN_DIR.mkdir(parents=True, exist_ok=True)
    # The wrapper journals its inventory calls into the run directory; the
    # unit's ProtectSystem=strict keeps everything else read-only.
    writable = (
        f"ReadWritePaths={run_dir(run_id)}\n" if arguments.inventory_wrapper else ""
    )
    dropin.write_text(
        "[Service]\n"
        f"# GF-GENERAL-LIVE DCGM field replay repoint (run {run_id})\n"
        f"EnvironmentFile=-{override}\n" + writable,
        encoding="utf-8",
    )
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the repoint")
    if ENDPOINT_ENV not in environment:
        raise ProbeError(
            "the restarted collector did not exec within the settle window"
        )
    wanted = f"http://127.0.0.1:{port}/metrics"
    if environment.get(ENDPOINT_ENV) != wanted:
        raise ProbeError("the running collector did not pick up the replay endpoint")
    if arguments.inventory_wrapper and not str(environment.get("PATH", "")).startswith(
        str(run_dir(run_id) / "bin") + ":"
    ):
        raise ProbeError("the running collector did not pick up the wrapper PATH")
    if arguments.boot_id and environment.get("GPU_FAULT_BOOT_ID_PATH") != str(
        run_dir(run_id) / "boot_id"
    ):
        raise ProbeError("the running collector did not pick up the replayed boot id")
    if arguments.unknown_expected_count and (
        environment.get("GPU_FAULT_NODE_INSTANCE_TYPE") != UNKNOWN_INSTANCE_TYPE
    ):
        raise ProbeError("the running collector did not pick up the unknown count")
    emit(
        {
            "run_id": run_id,
            "unit": METRICS_UNIT,
            "original_endpoint": original,
            "replay_endpoint": wanted,
            "process_environment": environment,
            "inventory_wrapper": bool(arguments.inventory_wrapper),
            "deadman_timer": deadman,
            "before": before,
            "after": after,
        }
    )


def arm_preconditions(run_id: str, port: int, before: bool) -> dict[str, Any]:
    """What ``arm``'s mode requires of the node: observed, checked, named.

    After the repoint (default) the collector must read this run's replay.
    Before it (``--before-repoint``: the runner arms, then repoints, so the
    restarted collector's first scrape is the plan's first read) the collector
    must still read the original exporter -- not this run's replay, not another
    run's proxy -- while this run's proxy unit is up, answers, is unarmed and
    has served no read. Live 2026-10-26 (022 ``source_boot_change``, both
    clusters) the one after-repoint guard refused that order: INCONCLUSIVE at
    ``arm``. A refusal names the mode and the endpoint seen; the facts travel
    in its detail and in the receipt.
    """

    replay, seen = f"http://127.0.0.1:{port}/metrics", effective_endpoint()
    facts: dict[str, Any] = {"before_repoint": before, "effective_endpoint": seen}
    facts["replay_endpoint"] = replay
    mode = "before" if before else "after"

    def refuse(reason: str, why: str) -> ProbeError:
        return ProbeError(f"refusing to arm {mode} the repoint: {why}", reason, facts)

    if not before:
        if seen != replay:
            raise refuse(
                "collector_not_repointed",
                f"the collector reads {seen}, not the replay {replay}",
            )
        return facts
    original, unit = baseline_endpoint(), f"{proxy_unit(run_id)}.service"
    state = unit_state(unit).get("ActiveState")
    facts.update(original_endpoint=original, proxy_unit=unit, proxy_unit_state=state)
    if state != "active":
        raise refuse("proxy_down", f"this run's proxy unit {unit} is {state}")
    try:
        status = tool_status(port)
    except ProbeError as error:
        raise refuse("no_status", f"this run's proxy on {port} gave no status: {error}")
    facts["reads_before_arm"] = reads = status.get("reads")
    if status.get("armed") is not False or reads != 0:
        raise refuse(
            "proxy_read_before_arm",
            f"the proxy on port {port} reports armed={status.get('armed')} after "
            f"{reads} reads; nothing may read it before the repoint",
        )
    if seen != original:
        why = " (this run's replay)" if seen == replay else ""
        raise refuse(
            "collector_already_repointed",
            f"the collector reads {seen}{why}, not the original exporter {original}",
        )
    return facts


def arm(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    facts = arm_preconditions(run_id, port, bool(arguments.before_repoint))
    status, body = loopback(f"http://127.0.0.1:{port}/arm", method="POST")
    if status != 200:
        raise ProbeError(f"arm returned {status}")
    emit(
        {
            "run_id": run_id,
            "armed": True,
            "response": body.decode().strip(),
            "status": tool_status(port),
            **facts,
        }
    )


def disarm(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    status, body = loopback(f"http://127.0.0.1:{port}/disarm", method="POST")
    if status != 200:
        raise ProbeError(f"disarm returned {status}")
    emit({"run_id": run_id, "disarmed": True, "status": tool_status(port)})


def advance(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    status, body = loopback(f"http://127.0.0.1:{port}/advance", method="POST")
    if status != 200:
        raise ProbeError(f"advance returned {status}")
    emit({"run_id": run_id, "advanced": True, "status": tool_status(port)})


def status_command(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    state = unit_state(METRICS_UNIT)
    emit(
        {
            "run_id": run_id,
            "status": tool_status(checked_port(arguments.port)),
            "effective_endpoint": process_environment(state.get("MainPID", "")).get(
                ENDPOINT_ENV
            ),
            "metrics_unit": state,
        }
    )


def audit_read(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = arguments.journal
    if name not in {"replay.jsonl", "inventory.jsonl"}:
        raise ProbeError("only the replay and inventory journals may be read")
    path = run_dir(run_id) / name
    offset = bounded(arguments.offset, 0, 10**6, "offset")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    chunk = lines[offset : offset + MAX_AUDIT_LINES]
    records = []
    for line in chunk:
        try:
            records.append(json.loads(line))
        except ValueError:
            records.append({"unparsed": line[:200]})
    emit(
        {
            "run_id": run_id,
            "journal": name,
            "records": records,
            "next_offset": offset + len(chunk),
            "total": len(lines),
        }
    )


def collector_restart(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    before = unit_state(METRICS_UNIT)
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restart")
    emit(
        {
            "run_id": run_id,
            "before": before,
            "after": after,
            "restarted": before.get("InvocationID") != after.get("InvocationID"),
            "effective_endpoint": environment.get(ENDPOINT_ENV),
        }
    )


# --------------------------------------------------------------------------- #
# restore / stop / cleanup
# --------------------------------------------------------------------------- #
def restore_of(run_digest: str) -> dict[str, Any]:
    dropin = dropin_of(run_digest)
    original = baseline_endpoint()
    repoint = run_dir_of(run_digest) / "repoint.json"
    if repoint.is_file():
        try:
            original = str(json.loads(repoint.read_text())["original_endpoint"])
        except (ValueError, KeyError):
            pass
    present = dropin.exists()
    dropin.unlink(missing_ok=True)
    override_env_of(run_digest).unlink(missing_ok=True)
    if present:
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restore")
    if present and ENDPOINT_ENV not in environment:
        raise ProbeError("the restored collector did not exec within the settle window")
    effective = environment.get(ENDPOINT_ENV)
    if effective is not None and effective != original:
        raise ProbeError("the collector endpoint is not the original after the restore")
    if str(environment.get("PATH", "")).startswith(str(STATE_ROOT)):
        raise ProbeError("the collector still runs with the replay wrapper PATH")
    if str(environment.get("GPU_FAULT_BOOT_ID_PATH", "")).startswith(str(STATE_ROOT)):
        raise ProbeError("the collector still reads the replayed boot id")
    if environment.get("GPU_FAULT_NODE_INSTANCE_TYPE") == UNKNOWN_INSTANCE_TYPE:
        raise ProbeError("the collector still runs with the unknown instance type")
    return {
        "unit": METRICS_UNIT,
        "dropin_removed": present and not dropin.exists(),
        "dropin_was_present": present,
        "restored_endpoint": effective,
        "original_endpoint": original,
        "after": after,
    }


def restore(run_id: str) -> dict[str, Any]:
    return restore_of(digest(run_id))


def restore_collector(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    result = restore(run_id)
    disarm_deadman(run_id)
    emit({"run_id": run_id, **result})


def stop_of(run_digest: str, port: int | None) -> dict[str, Any]:
    effective = effective_endpoint() or ""
    if effective.startswith("http://127.0.0.1:") and effective != baseline_endpoint():
        raise ProbeError("refusing to stop the replay while the collector reads it")
    unit = proxy_unit_of(run_digest)
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    root = run_dir_of(run_digest)
    audit_present = (root / "replay.jsonl").is_file()
    if root.is_dir():
        shutil.rmtree(root)
    state = unit_state(f"{unit}.service")
    return {
        "unit": f"{unit}.service",
        "unit_active": state.get("ActiveState") == "active",
        "audit_was_present": audit_present,
        "run_dir_removed": not root.exists(),
        "port": port,
    }


def stop(run_id: str, port: int | None) -> dict[str, Any]:
    return stop_of(digest(run_id), port)


def stop_proxy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    emit({"run_id": run_id, **stop(run_id, checked_port(arguments.port))})


def step_error(step: str, exc: BaseException) -> str:
    """One failed cleanup step: its name, the class and -- for this probe's
    own constant sentences -- the message."""

    detail = f": {exc}" if isinstance(exc, ProbeError) else ""
    return f"{step}: {type(exc).__name__}{detail}"


def cleanup_steps(
    run_digest: str, port: int | None
) -> tuple[dict[str, Any], list[str]]:
    """Restore the collector, then stop the tool; every step is attempted."""

    result: dict[str, Any] = {}
    errors: list[str] = []
    try:
        result["restore"] = restore_of(run_digest)
    except Exception as exc:  # noqa: BLE001 - recorded, the next step still runs
        errors.append(step_error("restore", exc))
    try:
        result["stop"] = stop_of(run_digest, port)
    except Exception as exc:  # noqa: BLE001
        errors.append(step_error("stop", exc))
    return result, errors


def disarm_tool(port: int) -> dict[str, Any]:
    """Best-effort ``POST /disarm`` so the tool passes through before the
    collector is restarted; no tool on the port is not an error here."""

    try:
        status, _body = loopback(
            f"http://127.0.0.1:{checked_port(port)}/disarm", method="POST", timeout=5
        )
    except ProbeError:
        return {"disarmed": False, "detail": "no replay tool answered"}
    return {"disarmed": status == 200, "http_status": status}


def cleanup(arguments: argparse.Namespace) -> None:
    """The deadman's entry point: every step is attempted, in order."""

    run_id = checked_run_id(arguments.run_id)
    run_digest = digest(run_id)
    result, errors = cleanup_steps(run_digest, None)
    disarm_deadman_of(run_digest)
    result.update(run_id=run_id, residue=residue(), errors=errors)
    emit(result)
    if errors:
        raise SystemExit(1)


def close_run(arguments: argparse.Namespace) -> None:
    """The runner's cleanup of this run: every step, then what is left.

    The deadman is disarmed only once nothing failed; while a step failed it
    stays armed as the independent restore of what the runner could not undo,
    and is listed among the leftovers with its fire time.
    """

    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    run_digest = digest(run_id)
    result: dict[str, Any] = {"run_id": run_id, "digest": run_digest}
    result["disarm"] = disarm_tool(port)
    steps, errors = cleanup_steps(run_digest, port)
    result.update(steps)
    if errors:
        result["deadman_kept"] = loaded(
            unit_state(f"{deadman_unit_of(run_digest)}.timer")
        )
    else:
        disarm_deadman_of(run_digest)
        result["deadman_kept"] = False
    state = run_state(run_digest)
    result.update(
        errors=errors,
        leftover=state["objects"],
        clean=not state["objects"] and not errors,
        run_state=state,
    )
    emit(result)


def cleanup_leftover(arguments: argparse.Namespace) -> None:
    """Reclaim an earlier run's objects by digest, unless its deadman is pending.

    Only this fixture's own objects (the ``gfrs-live-<digest>`` names) are
    ever addressed; a pending deadman means the run that armed it may still
    be live or its restore is about to run, so nothing is touched and the
    answer says so.
    """

    checked_run_id(arguments.run_id)
    run_digest = checked_digest(arguments.digest)
    before = run_state(run_digest)
    result: dict[str, Any] = {
        "digest": run_digest,
        "run_id": before.get("run_id"),
        "before": before,
    }
    if not before["objects"] or before["deadman_pending"]:
        reason = (
            "nothing of this run is left on the node"
            if not before["objects"]
            else "the run's deadman restore is still pending; left to fire"
        )
        emit(
            {
                **result,
                "reclaimed": False,
                "reason": reason,
                "clean": not before["objects"],
                "leftover": before["objects"],
                "errors": [],
            }
        )
        return
    steps, errors = cleanup_steps(run_digest, None)
    if not errors:
        disarm_deadman_of(run_digest)
    after = run_state(run_digest)
    emit(
        {
            **result,
            **steps,
            "reclaimed": True,
            "errors": errors,
            "leftover": after["objects"],
            "clean": not after["objects"] and not errors,
            "after": after,
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler, *, port: bool = False) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        if port:
            item.add_argument("--port", type=int, default=9412)
        item.set_defaults(handler=handler)
        return item

    surveyed = command("survey", survey, port=True)
    surveyed.add_argument("--limits", action="store_true")
    source = command("install-source", install_source)
    source.add_argument("--name", required=True)
    source.add_argument("--sha256", required=True)
    source.add_argument("--b64", required=True)
    deploy = command("deploy-proxy", deploy_proxy, port=True)
    deploy.add_argument("--tool", required=True)
    deploy.add_argument("--profile", default="")
    deploy.add_argument("--gpu-uuid", default="")
    deploy.add_argument("--max-seconds", type=int, default=1800)
    repoint = command("repoint-collector", repoint_collector, port=True)
    repoint.add_argument("--restore-seconds", type=int, default=1800)
    repoint.add_argument("--inventory-wrapper", action="store_true")
    repoint.add_argument("--boot-id", default="")
    repoint.add_argument("--unknown-expected-count", action="store_true")
    armed = command("arm", arm, port=True)
    armed.add_argument("--before-repoint", action="store_true")
    command("disarm", disarm, port=True)
    command("advance", advance, port=True)
    command("status", status_command, port=True)
    audit = command("audit-read", audit_read)
    audit.add_argument("--journal", default="replay.jsonl")
    audit.add_argument("--offset", type=int, default=0)
    command("collector-restart", collector_restart)
    command("restore-collector", restore_collector)
    command("stop-proxy", stop_proxy, port=True)
    command("close-run", close_run, port=True)
    leftover = command("cleanup-leftover", cleanup_leftover)
    leftover.add_argument("--digest", required=True)
    command("cleanup", cleanup)
    return value


def exception_site(exc: BaseException) -> str | None:
    """The innermost ``function:line`` of this file on the exception's path."""

    site = None
    frame = exc.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename == __file__:
            site = f"{frame.tb_frame.f_code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def error_fields(exc: BaseException, command: str) -> dict[str, Any]:
    """The failure line: class, message digest, site and reason token are what
    the host fixture may show; the message goes into the case's redacted
    ``probe-error.json`` (live 2026-10-26 it was withheld and brute-forced)."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "error_kind": "probe",
        "error_class": type(exc).__name__,
        "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
        "error_site": exception_site(exc),
        "error_reason": getattr(exc, "reason", "") or None,
        "error_message": str(exc)[:2000],
        "error_detail": getattr(exc, "detail", None),
        "command": command,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(error_fields(exc, arguments.command))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
