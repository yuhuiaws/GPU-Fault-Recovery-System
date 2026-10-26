#!/usr/bin/env python3
"""Host controller for GF-ENV-LIVE on one isolated, approved GPU test node.

Runs on the node through the privileged ``host_probe_fixture`` Pod (chroot into
``/host`` under the node's own ``gpu_fault`` venv), so every import below is the
deployed collector code. It drives the audited loopback replay proxy
(``tools/gpu_environment_replay.py``) and repoints the node's single metrics
collector at it through a bounded, reversible single-node maintenance change.
Everything it mutates is bounded and restored two ways -- the runner's cleanup
and a ``systemd-run`` deadman:

* **install-source** -- writes the replay tool and the reviewed plan, handed in
  base64 and checked against a sha256, under ``/run/gpu-fault-env-replay/<id>``
  (``/run`` is tmpfs: a reboot clears it). The tool is launched verbatim; the
  controller never re-types it.
* **deploy-proxy** -- a transient ``systemd-run`` unit runs the tool
  ``--serve`` against the real exporter (``127.0.0.1:9400``) on a loopback-only
  port (``127.0.0.1:9411``) with a hard ``RuntimeMaxSec``; it starts pass-through
  and only injects after ``arm``.
* **repoint-collector** -- the deadman is armed *first*; then a ``/run`` systemd
  drop-in on the one allow-listed unit (``gpu-fault-metrics-collector.service``)
  adds ``EnvironmentFile=-/run/gpu-fault-env-replay/<id>/collector.env``, a
  run-scoped file holding only ``GPU_FAULT_DCGM_METRICS_URL=<proxy>``. An
  environment *file*, never ``Environment=``: the installer writes the real
  endpoint into the unit's own ``EnvironmentFile=/etc/gpu-fault/collector.env``
  and systemd applies every ``EnvironmentFile=`` after every ``Environment=``,
  later-listed files winning (systemd.exec(5)) -- the drop-in
  ``Environment=`` this probe used to write restarted the collector but never
  reached its process (live 2026-10-26, GF-GENERAL-LIVE-024 ``thermal_warning``
  and ``power_performance`` on plain EKS; the same signature on every
  GF-ENV-LIVE case of the HyperPod round of 2026-10-24: "consumed 0 of N
  reads"). The unit is restarted and the effective endpoint is read from the
  running process (``/proc/<MainPID>/environ``, once it has exec'd), never
  from ``systemctl show`` (which lists ``Environment=`` directives only and
  reported the override as effective). The repoint then waits for the restarted
  collector's **first pass-through read** of the proxy (``/status`` ``reads``
  advancing) inside the collector's start-up spread plus one interval; an
  override the process did not pick up or a collector that never read the proxy
  is rolled back on the node at once and answered with ``repointed: false`` and
  a constant ``reason``, so the runner fails the fixture with that reason
  instead of waiting out a replay budget. The real exporter config and
  ``collector.env`` are untouched and no second producer of the same identity
  is started. The ``systemd-run`` deadman restores the original endpoint if the
  runner dies.
* **arm** -- one ``POST /arm`` with the acknowledgement header, refused while the
  collector process does not read the proxy. ``status`` is a read that never
  consumes a replay step.
* **pause-collector / pause-status / resume-collector** -- the GF-ENV-LIVE-009
  collection interruption: a transient waiter unit polls the proxy's
  ``/status`` on loopback and ``SIGSTOP``s the metrics unit once the collector
  has consumed the requested read (the same ``systemctl kill`` pause as
  ``general_live_node_probe.py unit-pause``, taken read-synchronously on the
  node because a runner-timed pause could land one collector tick late and let
  a third severe sample confirm). The stop lands ``PAUSE_SETTLE_SECONDS`` after
  the read was counted, so the collector's synchronous post of that read has
  reached the control plane; the next scrape is still a whole interval away. A
  ``systemd-run`` deadman resumes the unit after the bounded pause even if the
  runner dies; ``restore-collector`` resumes first as well. ``--label`` names a
  second, independent pause of the same run with its own waiter, deadman and
  record (the 009 *recount hold* right after the first post-gap read, keeping
  the product's regraded finding state readable until the runner has read it);
  ``pause-status`` lists every pause under ``pauses``.
* **restart-collector** -- one ``systemctl restart`` of the same unit with the
  drop-in in place (the 009 freshness / single-producer check).
* **restore-collector / stop-proxy** -- remove the drop-in and its environment
  file, reload, restart the unit when the drop-in was in place or the process
  still reads another endpoint, and read the real endpoint back from the new
  process (a second restore finds nothing to undo and restarts nothing); the
  proxy is stopped only after the endpoint is back on the real exporter.

Every answer is one JSON object on the last stdout line; a failure carries the
``error_kind: probe`` fields (class, digest, site, message, detail) the host
fixture keeps redacted in the case's ``probe-error.json``. The Node Agent is
never in the allow-list; no GPU, workload or exporter configuration is changed.
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
from urllib.request import Request, build_opener, ProxyHandler

COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
VENV_PYTHON = Path("/opt/gpu-fault/current/venv/bin/python")
ACCEPTANCE_ROOT = Path("/run/gpu-fault-env-replay")
#: Where a running process's environment is read from (``<pid>/environ``).
PROC_ROOT = Path("/proc")
#: The one unit a window may repoint. The Node Agent and every other collector
#: are deliberately absent.
METRICS_UNIT = "gpu-fault-metrics-collector.service"
ENDPOINT_ENV = "GPU_FAULT_DCGM_METRICS_URL"
DEFAULT_UPSTREAM = "http://127.0.0.1:9400/metrics"
#: The run-scoped environment file the repoint drop-in hands to the unit.
OVERRIDE_ENV_NAME = "collector.env"
#: The allow-listed keys read back from the collector process's environment
#: (``/proc/<MainPID>/environ``) into receipts.
INTERVAL_ENV = "GPU_FAULT_METRICS_INTERVAL_SECONDS"
PROCESS_KEYS = (ENDPOINT_ENV, INTERVAL_ENV, "GPU_FAULT_METRICS_MODE")
#: How long a restarted ``Type=simple`` unit may take to exec its program: the
#: unit is active from the fork on, and until the child has built its sandbox
#: and exec'd, ``/proc/<pid>/environ`` is still systemd's own.
SETTLE_SECONDS = 45
#: The restarted collector's first scrape follows its start-up spread
#: (``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS``, product default 15) plus a
#: random start-up jitter (``GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS``,
#: default 20); the first-read wait allows both, one interval and a margin, and
#: never more than ``MAX_FIRST_READ_SECONDS`` (the controller's exec timeout
#: bounds the whole repoint call).
STARTUP_SPREAD_ENV = "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS"
STARTUP_JITTER_ENV = "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS"
JITTER_RATIO_ENV = "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO"
#: The ``collector.env`` keys ``preflight`` reports: the endpoint and mode, the
#: cadence the runner sizes its windows from (interval, edge confirmation,
#: health summary, start-up spread, start-up jitter, cadence jitter ratio).
PREFLIGHT_ENV_KEYS = frozenset(
    {
        ENDPOINT_ENV,
        "GPU_FAULT_METRICS_MODE",
        INTERVAL_ENV,
        "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
        "GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS",
        STARTUP_SPREAD_ENV,
        STARTUP_JITTER_ENV,
        JITTER_RATIO_ENV,
    }
)
#: Every transient unit this probe creates starts with this prefix (the proxy,
#: the restore deadman, the pause waiter, the resume deadman); ``preflight``
#: and ``snapshot`` list what is still loaded so a leftover of an earlier run
#: is named before anything is deployed.
UNIT_PREFIX = "gpu-fault-env-"
DEFAULT_INTERVAL_SECONDS = 15.0
DEFAULT_STARTUP_SPREAD_SECONDS = 15.0
DEFAULT_STARTUP_JITTER_SECONDS = 20.0
FIRST_READ_MARGIN_SECONDS = 15
MAX_FIRST_READ_SECONDS = 90
FIRST_READ_POLL_SECONDS = 1.0
#: The constant reasons a repoint answers ``repointed: false`` with (the probe
#: has rolled the unit back by then).
REASON_NOT_PICKED_UP = (
    "the restarted collector process does not read the proxy endpoint; the "
    "override never reached it"
)
REASON_NO_FIRST_READ = (
    "the repointed collector made no pass-through read of the proxy inside the "
    "first-read budget"
)
ACKNOWLEDGEMENT = "ISOLATED_GPU_TEST_NODE"
REPLAY_TOOL_NAME = "gpu_environment_replay.py"
PLAN_NAME = "plan.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 1800
MIN_PORT, MAX_PORT = 1024, 65535
#: The proxy's replay window: the runner's 600 s plus the 009 recount hold
#: (``env_live_expectations.recount_hold_seconds``, 120 s at the default
#: cadence), which stops the collector a second time inside the window.
MAX_SECONDS_LIMIT = 900
#: A collection pause (009) is bounded like the node probe's ``unit-pause``:
#: the deadman that resumes the unit is at least a minute and at most 1500 s
#: out; the waiter that takes the pause gives up after ``MAX_WAIT_SECONDS``.
MIN_PAUSE_SECONDS = 60
MAX_PAUSE_SECONDS = 1500
MAX_WAIT_SECONDS = 600
WAIT_POLL_SECONDS = 0.25
#: The SIGSTOP lands this long after the target read was counted: the collector
#: posts a read synchronously inside the same tick (``collect_once``: scrape,
#: edge filter, post), so the pause read's batch has reached the control plane
#: before the unit stops, while the next scrape is still ``interval x (1 -
#: jitter ratio)`` (13.5 s at the defaults) away. Live 2026-10-26 (009, plain
#: EKS) the stop came 0.35 s after the read and the batch got through: a race.
PAUSE_SETTLE_SECONDS = 2.0
PAUSE_FILE = "pause.json"
#: A labelled pause is a second, independent pause of the same run (the 009
#: recount hold): the label becomes part of its waiter and deadman unit names
#: and of its record file (``pause-<label>.json``); the default label is the
#: collection gap (``pause.json``).
PAUSE_LABEL = re.compile(r"^[a-z][a-z0-9-]{0,15}$")


class ProbeError(RuntimeError):
    pass


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
            f"{completed.stderr.strip()}"
        )
    return completed


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]


def run_dir(run_id: str) -> Path:
    return ACCEPTANCE_ROOT / digest(run_id)


def proxy_unit(run_id: str) -> str:
    return f"gpu-fault-env-replay-{digest(run_id)}"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-env-collector-restore-{digest(run_id)}"


def safe_label(value: str | None) -> str:
    """A pause label (empty: the collection gap pause)."""

    label = (value or "").strip()
    if label and PAUSE_LABEL.fullmatch(label) is None:
        raise ProbeError("unsafe pause label")
    return label


def _label_suffix(label: str) -> str:
    return f"-{label}" if label else ""


def pause_waiter_unit(run_id: str, label: str = "") -> str:
    return f"gpu-fault-env-pause-wait-{digest(run_id)}{_label_suffix(label)}"


def resume_deadman_unit(run_id: str, label: str = "") -> str:
    return f"gpu-fault-env-collector-resume-{digest(run_id)}{_label_suffix(label)}"


def pause_path(run_id: str, label: str = "") -> Path:
    name = f"pause-{label}.json" if label else PAUSE_FILE
    return run_dir(run_id) / name


def pause_labels(run_id: str) -> list[str]:
    """Every pause record of the run: the gap pause (``""``) and the labelled ones."""

    root = run_dir(run_id)
    if not root.is_dir():
        return []
    return sorted(
        "" if path.name == PAUSE_FILE else path.name[len("pause-") : -len(".json")]
        for path in root.glob("pause*.json")
        if path.name == PAUSE_FILE or path.name.startswith("pause-")
    )


def read_pause(run_id: str, label: str = "") -> dict[str, Any]:
    path = pause_path(run_id, label)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def write_pause(run_id: str, value: dict[str, Any], label: str = "") -> None:
    pause_path(run_id, label).write_text(
        json.dumps(value, sort_keys=True), encoding="utf-8"
    )


def unit_is_stopped(unit: str) -> bool:
    """The unit's main process sits in ``T`` (SIGSTOP), as the node probe reads it."""

    pid = unit_state(unit).get("MainPID") or "0"
    if not pid.isdigit() or pid == "0":
        return False
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    fields = text.rsplit(")", 1)[-1].split()
    return bool(fields) and fields[0] == "T"


def dropin_dir() -> Path:
    return Path("/run/systemd/system") / f"{METRICS_UNIT}.d"


def dropin_path(run_id: str) -> Path:
    return dropin_dir() / f"gpu-fault-env-replay-{digest(run_id)}.conf"


def override_env_path(run_id: str) -> Path:
    """The run's environment file the drop-in lists (``EnvironmentFile=-``)."""

    return run_dir(run_id) / OVERRIDE_ENV_NAME


def checked_port(value: int) -> int:
    if not MIN_PORT <= int(value) <= MAX_PORT:
        raise ProbeError(f"port is outside {MIN_PORT}..{MAX_PORT}")
    return int(value)


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}"
        )
    return int(value)


def parse_env() -> dict[str, str]:
    if not COLLECTOR_ENV.is_file():
        raise ProbeError("collector env file does not exist")
    values: dict[str, str] = {}
    for line in COLLECTOR_ENV.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def baseline_endpoint() -> str:
    return parse_env().get(ENDPOINT_ENV, DEFAULT_UPSTREAM)


def unit_state(unit: str) -> dict[str, str]:
    """``systemctl show`` of ``unit``. ``Environment`` lists the unit's
    ``Environment=`` directives only -- never what an ``EnvironmentFile=``
    hands the process -- and is kept for the receipts, not for any decision;
    ``DropInPaths`` is what the restore reads back after its reload."""

    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--property=MainPID",
            "--property=InvocationID",
            "--property=ExecMainStartTimestamp",
            "--property=DropInPaths",
            "--property=Environment",
        ],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    values["returncode"] = str(completed.returncode)
    return values


def leftover_units(prefix: str = UNIT_PREFIX) -> list[dict[str, str]]:
    """The probe's transient units systemd still lists (any state).

    A finished transient unit is unloaded, so anything listed here -- a proxy
    still serving, a restore deadman still waiting, a pause waiter, a failed
    unit kept for its status -- is a run that did not end cleanly.
    """

    completed = run(
        [
            "systemctl",
            "list-units",
            "--all",
            "--plain",
            "--no-legend",
            "--no-pager",
            f"{prefix}*",
        ],
        check=False,
    )
    units: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].startswith(prefix):
            continue
        units.append(
            {
                "unit": fields[0],
                "load": fields[1],
                "active": fields[2],
                "sub": fields[3],
            }
        )
    return sorted(units, key=lambda item: item["unit"])


def open_windows() -> list[str]:
    """The run directories still present under ``ACCEPTANCE_ROOT``."""

    if not ACCEPTANCE_ROOT.is_dir():
        return []
    return sorted(path.name for path in ACCEPTANCE_ROOT.glob("*") if path.is_dir())


def process_environment(
    pid: str, keys: tuple[str, ...] = PROCESS_KEYS
) -> dict[str, str]:
    """The allow-listed keys of a running process's real environment."""

    if not pid or not pid.isdigit() or pid == "0":
        return {}
    try:
        raw = (PROC_ROOT / pid / "environ").read_bytes()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        name = key.decode("utf-8", "replace")
        if separator and name in keys:
            values[name] = value.decode("utf-8", "replace")
    return values


def has_execed(pid: str) -> bool:
    """Whether ``pid`` runs its own program yet, not systemd's pre-exec child.

    systemd names the forked child ``(<exec basename>)`` until it execs and its
    ``exe`` is still systemd's; from the exec on, ``comm`` and ``exe`` are the
    service's and ``/proc/<pid>/environ`` is the unit's environment.
    """

    try:
        comm = (PROC_ROOT / pid / "comm").read_text(encoding="utf-8", errors="replace")
        exe = os.readlink(str(PROC_ROOT / pid / "exe"))
    except OSError:
        return False
    return not comm.strip().startswith("(") and Path(exe).name != "systemd"


def settled_environment(
    unit: str, *, seconds: float = SETTLE_SECONDS
) -> tuple[dict[str, str], dict[str, str]]:
    """The unit's state and its main process's environment once it has exec'd.

    Returns what it last saw when the window closes; callers check for the
    endpoint key, which every exec'd collector carries from ``collector.env``.
    """

    deadline = time.monotonic() + seconds
    while True:
        state = unit_state(unit)
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


def effective_endpoint(unit: str, *, seconds: float = 15.0) -> str | None:
    """The ``GPU_FAULT_DCGM_METRICS_URL`` the unit's running process uses.

    Read from ``/proc/<MainPID>/environ`` (allowing a just-restarted unit its
    exec), never from ``systemctl show``: a drop-in ``Environment=`` shows there
    while the unit's ``EnvironmentFile=`` overrides it in the process.
    """

    _state, environment = settled_environment(unit, seconds=seconds)
    return environment.get(ENDPOINT_ENV)


def dropin_loaded(unit: str, dropin: Path) -> bool:
    """Whether systemd still lists ``dropin`` for ``unit`` (``DropInPaths``)."""

    return str(dropin) in (unit_state(unit).get("DropInPaths") or "").split()


def _positive_seconds(values: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(values.get(key, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def first_read_seconds(values: dict[str, str]) -> int:
    """The first-read budget from the node's ``collector.env`` cadence keys:
    start-up spread + start-up jitter + one interval + a margin, capped."""

    total = (
        _positive_seconds(values, STARTUP_SPREAD_ENV, DEFAULT_STARTUP_SPREAD_SECONDS)
        + _positive_seconds(values, STARTUP_JITTER_ENV, DEFAULT_STARTUP_JITTER_SECONDS)
        + _positive_seconds(values, INTERVAL_ENV, DEFAULT_INTERVAL_SECONDS)
        + FIRST_READ_MARGIN_SECONDS
    )
    return int(min(MAX_FIRST_READ_SECONDS, max(FIRST_READ_MARGIN_SECONDS, total)))


def wait_first_read(port: int, reads_before: int, seconds: int) -> dict[str, Any]:
    """Wait until the proxy's ``reads`` advanced past ``reads_before``.

    The proxy counts every ``GET /metrics`` (pass-through included), so the
    first advance after the restart is the repointed collector's first scrape
    of the proxy -- the only proof the override took effect end to end.
    """

    started = time.monotonic()
    deadline = started + seconds
    reads_after = reads_before
    while True:
        try:
            reads_after = int(proxy_status(port, timeout=2).get("reads") or 0)
        except (ProbeError, ValueError, TypeError):
            pass
        observed = reads_after > reads_before
        if observed or time.monotonic() >= deadline:
            return {
                "observed": observed,
                "reads_before": reads_before,
                "reads_after": reads_after,
                "waited_seconds": round(time.monotonic() - started, 1),
                "budget_seconds": seconds,
            }
        time.sleep(FIRST_READ_POLL_SECONDS)


# --------------------------------------------------------------------------- #
# Loopback HTTP to the proxy (never follows redirects, never uses a proxy env)
# --------------------------------------------------------------------------- #
def _loopback(url: str, *, method: str, timeout: int) -> tuple[int, str]:
    if not url.startswith("http://127.0.0.1:"):
        raise ProbeError("proxy URL must be loopback http")
    opener = build_opener(ProxyHandler({}))
    request = Request(url, method=method)
    if method == "POST":
        request.add_header("X-GFRS-Replay-Ack", ACKNOWLEDGEMENT)
        request.data = b""
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status), response.read(65536).decode("utf-8", "replace")
    except OSError as error:
        raise ProbeError(f"proxy {method} {url} failed: {error}") from error


def proxy_status(port: int, *, timeout: int = 5) -> dict[str, Any]:
    status, body = _loopback(
        f"http://127.0.0.1:{checked_port(port)}/status", method="GET", timeout=timeout
    )
    if status != 200:
        raise ProbeError(f"proxy /status returned {status}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ProbeError("proxy /status did not return an object")
    return value


# --------------------------------------------------------------------------- #
# preflight: read-only device limits, exporter, collector, unit
# --------------------------------------------------------------------------- #
def device_limits() -> list[dict[str, Any]]:
    """Per-GPU temperature limits from ``nvidia-smi -q -x`` via the product's
    own parser, so the scale (absolute vs T.Limit) is decided exactly as the
    collector decides it."""

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


def gpu_inventory() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    result = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) == 4:
            result.append(
                {
                    "index": values[0],
                    "uuid": values[1],
                    "pci_bus_id": values[2],
                    "name": values[3],
                }
            )
    return result


def current_tlimit_readings() -> dict[str, float]:
    """The live T.Limit margin per GPU, read from the exporter (field 153).

    The limit probe above caches its XML and never reports the current reading;
    the reading comes from the exporter the way the collector reads it, so a
    plan can check the device is in a safe state before arming."""

    readings: dict[str, float] = {}
    try:
        status, body = _loopback(DEFAULT_UPSTREAM, method="GET", timeout=5)
    except ProbeError:
        return readings
    if status != 200:
        return readings
    for line in body.splitlines():
        if not line.startswith("DCGM_FI_DEV_GPU_TEMP_LIMIT"):
            continue
        match = re.search(r'UUID="(GPU-[^"]+)".*?\}\s+(-?\d+(?:\.\d+)?)', line)
        if match:
            readings[match.group(1)] = float(match.group(2))
    return readings


def preflight(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "collector_env": {
                key: value
                for key, value in parse_env().items()
                if key in PREFLIGHT_ENV_KEYS
            },
            "metrics_unit": unit_state(METRICS_UNIT),
            "effective_endpoint": effective_endpoint(METRICS_UNIT),
            "baseline_endpoint": baseline_endpoint(),
            "device_limits": device_limits(),
            "gpu_inventory": gpu_inventory(),
            "current_tlimit_readings": current_tlimit_readings(),
            "open_windows": open_windows(),
            "leftover_units": leftover_units(),
        }
    )


# --------------------------------------------------------------------------- #
# install-source: write the audited tool and plan under /run
# --------------------------------------------------------------------------- #
def install_source(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    name = arguments.name
    if name not in {REPLAY_TOOL_NAME, PLAN_NAME}:
        raise ProbeError("only the replay tool and plan may be installed")
    if SHA256.fullmatch(arguments.sha256) is None:
        raise ProbeError("a sha256 is required for every installed source")
    try:
        content = base64.b64decode(arguments.b64, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError(f"source is not valid base64: {error}") from error
    actual = hashlib.sha256(content).hexdigest()
    if actual != arguments.sha256:
        raise ProbeError("installed source sha256 does not match")
    root = run_dir(run_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise ProbeError(f"{name} is already installed for this run")
    target.write_bytes(content)
    target.chmod(0o600)
    emit({"run_id": run_id, "name": name, "sha256": actual, "bytes": len(content)})


# --------------------------------------------------------------------------- #
# deploy-proxy: run the audited tool --serve under a transient systemd unit
# --------------------------------------------------------------------------- #
def deploy_proxy(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    if GPU_UUID.fullmatch(arguments.gpu_uuid) is None:
        raise ProbeError("a physical GPU UUID is required")
    port = checked_port(arguments.port)
    upstream = arguments.upstream or DEFAULT_UPSTREAM
    if not upstream.startswith("http://127.0.0.1:") or not upstream.endswith(
        "/metrics"
    ):
        raise ProbeError("upstream must be http://127.0.0.1:<port>/metrics")
    if not 1 <= int(arguments.max_seconds) <= MAX_SECONDS_LIMIT:
        raise ProbeError(f"max-seconds must be 1..{MAX_SECONDS_LIMIT}")
    if (
        not 0 < float(arguments.gpu_stop_c) <= 80
        or not 0 < float(arguments.memory_stop_c) <= 85
    ):
        raise ProbeError("safety limits may only be lowered from 80C GPU / 85C memory")
    root = run_dir(run_id)
    tool = root / REPLAY_TOOL_NAME
    plan = root / PLAN_NAME
    audit = root / "replay.jsonl"
    if not tool.is_file() or not plan.is_file():
        raise ProbeError("install the replay tool and plan before deploying the proxy")
    if audit.exists():
        raise ProbeError("an audit file already exists; start a new audited run")
    unit = proxy_unit(run_id)
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    # ``-I`` drops the script's directory and PYTHONPATH from sys.path so a file
    # planted beside the tool cannot be imported; the venv's own gpu_fault stays
    # importable (site-packages, not user site).
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            "--property=Type=simple",
            f"--property=RuntimeMaxSec={int(arguments.max_seconds) + 60}s",
            "--property=NoNewPrivileges=true",
            "--property=ProtectSystem=strict",
            f"--property=ReadWritePaths={root}",
            "--setenv=no_proxy=*",
            "--setenv=NO_PROXY=*",
            str(VENV_PYTHON),
            "-I",
            str(tool),
            "--plan",
            str(plan),
            "--gpu-uuid",
            arguments.gpu_uuid,
            "--upstream",
            upstream,
            "--port",
            str(port),
            "--max-seconds",
            str(int(arguments.max_seconds)),
            "--gpu-stop-c",
            str(float(arguments.gpu_stop_c)),
            "--memory-stop-c",
            str(float(arguments.memory_stop_c)),
            "--serve",
            "--ack",
            ACKNOWLEDGEMENT,
            "--audit",
            str(audit),
        ]
    )
    deadline = time.monotonic() + 20
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = unit_state(f"{unit}.service")
        if state.get("ActiveState") == "active":
            try:
                status = proxy_status(port)
                break
            except ProbeError:
                pass
        if state.get("ActiveState") in {"failed", "inactive"}:
            raise ProbeError(f"replay proxy unit did not stay up: {state}")
        time.sleep(0.5)
    if not status:
        raise ProbeError("replay proxy did not answer /status")
    if status.get("armed") is not False:
        raise ProbeError("replay proxy came up armed; it must start pass-through")
    emit(
        {
            "run_id": run_id,
            "unit": f"{unit}.service",
            "endpoint": f"http://127.0.0.1:{port}/metrics",
            "audit": str(audit),
            "status": status,
            "service": unit_state(f"{unit}.service"),
        }
    )


# --------------------------------------------------------------------------- #
# repoint-collector: drop-in + deadman restore
# --------------------------------------------------------------------------- #
def arm_deadman(run_id: str, restore_seconds: int) -> str:
    """Arm the restore deadman (before the drop-in exists, so a crash between
    the two still restores the real endpoint); returns the unit name."""

    deadman = deadman_unit(run_id)
    probe_copy = run_dir(run_id) / "probe.py"
    if not probe_copy.exists():
        shutil.copyfile(Path(__file__).resolve(), probe_copy)
        probe_copy.chmod(0o700)
    run(["systemctl", "stop", f"{deadman}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{deadman}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={deadman}",
            f"--on-active={restore_seconds}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            str(VENV_PYTHON),
            str(probe_copy),
            "restore-collector",
            "--run-id",
            run_id,
        ]
    )
    if unit_state(f"{deadman}.timer").get("ActiveState") != "active":
        raise ProbeError("collector restore deadman timer is not active")
    return deadman


def checked_first_read_seconds(value: int, env: dict[str, str]) -> int:
    """``--first-read-seconds`` (0 = derive from the node's ``collector.env``)."""

    if int(value) <= 0:
        return first_read_seconds(env)
    if not FIRST_READ_MARGIN_SECONDS <= int(value) <= MAX_FIRST_READ_SECONDS:
        raise ProbeError(
            "first-read seconds is outside "
            f"{FIRST_READ_MARGIN_SECONDS}..{MAX_FIRST_READ_SECONDS}"
        )
    return int(value)


def repoint_collector(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    port = checked_port(arguments.port)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    env = parse_env()
    budget = checked_first_read_seconds(arguments.first_read_seconds, env)
    before = unit_state(METRICS_UNIT)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active at baseline")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError("a repoint drop-in for this run already exists")
    override = override_env_path(run_id)
    if override.exists():
        raise ProbeError("a repoint override for this run already exists")
    original = env.get(ENDPOINT_ENV, DEFAULT_UPSTREAM)
    proxy_endpoint = f"http://127.0.0.1:{port}/metrics"
    # The proxy must be serving pass-through before the collector is sent to
    # it; its read count now is the baseline the first-read proof advances from.
    status = proxy_status(port)
    if status.get("armed") is not False:
        raise ProbeError("refusing to repoint the collector at an armed proxy")
    reads_before = int(status.get("reads") or 0)
    deadman = arm_deadman(run_id, restore_seconds)
    # Through a file, never ``Environment=``: see the module docstring.
    override.write_text(f"{ENDPOINT_ENV}={proxy_endpoint}\n", encoding="utf-8")
    override.chmod(0o600)
    (run_dir(run_id) / "repoint.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "original_endpoint": original,
                "proxy_endpoint": proxy_endpoint,
                "override": str(override),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text(
        "[Service]\n"
        f"# gpu-fault env-live acceptance repoint (run {run_id})\n"
        f"EnvironmentFile=-{override}\n",
        encoding="utf-8",
    )
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "unit": METRICS_UNIT,
        "original_endpoint": original,
        "proxy_endpoint": proxy_endpoint,
        "dropin": str(dropin),
        "override": str(override),
        "process_environment": environment,
        "deadman_timer": unit_state(f"{deadman}.timer"),
        "before": before,
        "after": after,
    }
    if after.get("ActiveState") != "active":
        restore_unit(run_id)
        raise ProbeError(f"{METRICS_UNIT} is not active after the repoint; restored")
    if ENDPOINT_ENV not in environment:
        restore_unit(run_id)
        raise ProbeError(
            "the restarted collector did not exec within the settle window; restored"
        )
    if environment.get(ENDPOINT_ENV) != proxy_endpoint:
        # The verification this probe used to make through ``systemctl show``
        # passed here while the process still read the real exporter.
        emit(
            {
                **receipt,
                "repointed": False,
                "reason": REASON_NOT_PICKED_UP,
                "rollback": restore_unit(run_id),
            }
        )
        return
    first_read = wait_first_read(port, reads_before, budget)
    receipt["first_read"] = first_read
    if not first_read["observed"]:
        emit(
            {
                **receipt,
                "repointed": False,
                "reason": REASON_NO_FIRST_READ,
                "rollback": restore_unit(run_id),
            }
        )
        return
    emit({**receipt, "repointed": True})


def arm(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    port = checked_port(arguments.port)
    if effective_endpoint(METRICS_UNIT) != f"http://127.0.0.1:{port}/metrics":
        raise ProbeError("refusing to arm: the collector does not read the proxy")
    status, body = _loopback(f"http://127.0.0.1:{port}/arm", method="POST", timeout=10)
    if status != 200:
        raise ProbeError(f"arm returned {status}: {body.strip()}")
    emit(
        {
            "run_id": run_id,
            "armed": True,
            "response": body.strip(),
            "status": proxy_status(port),
        }
    )


def status_command(arguments: argparse.Namespace) -> None:
    emit(
        {
            "run_id": safe_id(arguments.run_id, "run ID"),
            "status": proxy_status(checked_port(arguments.port)),
            "effective_endpoint": effective_endpoint(METRICS_UNIT),
        }
    )


# --------------------------------------------------------------------------- #
# pause-collector / pause-wait / pause-status / resume-collector (009)
# --------------------------------------------------------------------------- #
def checked_pause_seconds(value: int) -> int:
    if not MIN_PAUSE_SECONDS <= int(value) <= MAX_PAUSE_SECONDS:
        raise ProbeError(
            f"pause seconds is outside {MIN_PAUSE_SECONDS}..{MAX_PAUSE_SECONDS}"
        )
    return int(value)


def checked_wait_seconds(value: int) -> int:
    if not 1 <= int(value) <= MAX_WAIT_SECONDS:
        raise ProbeError(f"wait seconds is outside 1..{MAX_WAIT_SECONDS}")
    return int(value)


def pause_collector(arguments: argparse.Namespace) -> None:
    """Schedule the read-synchronous pause: a transient waiter unit on the node.

    The waiter (``pause-wait``, a copy of this file) polls the proxy's
    ``/status`` every ``WAIT_POLL_SECONDS`` and stops the metrics unit the
    moment ``reads`` reaches ``--after-reads``; the runner only has to start it
    well before that read (right after ``arm``), so a slow ``kubectl exec`` can
    never move the pause point by a collector tick.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    label = safe_label(getattr(arguments, "label", ""))
    port = checked_port(arguments.port)
    after = int(arguments.after_reads)
    if after < 1:
        raise ProbeError("after-reads must be at least 1")
    seconds = checked_pause_seconds(arguments.seconds)
    wait = checked_wait_seconds(arguments.wait_seconds)
    probe_copy = run_dir(run_id) / "probe.py"
    if not probe_copy.is_file():
        raise ProbeError("repoint the collector before scheduling a pause")
    if read_pause(run_id, label):
        raise ProbeError("a pause is already scheduled or taken for this run")
    # A SIGSTOPped unit is still ``active`` to systemd: the 009 recount hold is
    # scheduled while the gap pause holds the collector (no read slips in).
    if unit_state(METRICS_UNIT).get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active at baseline")
    status = proxy_status(port)
    if int(status.get("reads") or 0) >= after:
        raise ProbeError("the collector already consumed the pause read")
    waiter = pause_waiter_unit(run_id, label)
    run(["systemctl", "stop", f"{waiter}.service"], check=False)
    run(["systemctl", "reset-failed", f"{waiter}.service"], check=False)
    write_pause(
        run_id,
        {
            "run_id": run_id,
            "label": label,
            "target_reads": after,
            "pause_seconds": seconds,
            "wait_seconds": wait,
            "settle_seconds": PAUSE_SETTLE_SECONDS,
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "reads_at_schedule": status.get("reads"),
            "waiter_unit": f"{waiter}.service",
        },
        label,
    )
    run(
        [
            "systemd-run",
            f"--unit={waiter}",
            "--property=Type=simple",
            f"--property=RuntimeMaxSec={wait + 30}s",
            "--setenv=no_proxy=*",
            "--setenv=NO_PROXY=*",
            str(VENV_PYTHON),
            str(probe_copy),
            "pause-wait",
            "--run-id",
            run_id,
            "--after-reads",
            str(after),
            "--port",
            str(port),
            "--seconds",
            str(seconds),
            "--wait-seconds",
            str(wait),
            *(["--label", label] if label else []),
        ]
    )
    state = unit_state(f"{waiter}.service")
    if state.get("ActiveState") not in {"active", "activating"}:
        raise ProbeError(f"pause waiter unit did not start: {state}")
    emit({**read_pause(run_id, label), "waiter": state, "status": status})


def take_pause(
    run_id: str, seconds: int, reads: int, target: int, label: str = ""
) -> dict[str, Any]:
    """Arm the resume deadman, then SIGSTOP the metrics unit (in that order)."""

    probe_copy = run_dir(run_id) / "probe.py"
    deadman = resume_deadman_unit(run_id, label)
    run(["systemctl", "stop", f"{deadman}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{deadman}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={deadman}",
            f"--on-active={seconds}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            str(VENV_PYTHON),
            str(probe_copy),
            "resume-collector",
            "--run-id",
            run_id,
            *(["--label", label] if label else []),
        ]
    )
    if unit_state(f"{deadman}.timer").get("ActiveState") != "active":
        raise ProbeError("collector resume deadman timer is not active")
    run(["systemctl", "kill", "--kill-whom=all", "--signal=SIGSTOP", METRICS_UNIT])
    record = {
        **read_pause(run_id, label),
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "reads_at_pause": reads,
        "missed": reads > target,
        "deadman_unit": f"{deadman}.timer",
        "stopped": unit_is_stopped(METRICS_UNIT),
    }
    write_pause(run_id, record, label)
    return record


def pause_wait(arguments: argparse.Namespace) -> None:
    """The waiter's body: poll the proxy until the pause read, settle, pause.

    The read count is taken again after the settle: a tick that slipped in
    meanwhile is recorded as ``missed``.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    label = safe_label(getattr(arguments, "label", ""))
    port = checked_port(arguments.port)
    target = int(arguments.after_reads)
    seconds = checked_pause_seconds(arguments.seconds)
    deadline = time.monotonic() + checked_wait_seconds(arguments.wait_seconds)
    reads = -1
    while time.monotonic() < deadline:
        try:
            reads = int(proxy_status(port, timeout=2).get("reads") or 0)
        except (ProbeError, ValueError, TypeError):
            reads = -1
        if reads >= target:
            time.sleep(PAUSE_SETTLE_SECONDS)
            try:
                reads = int(proxy_status(port, timeout=2).get("reads") or reads)
            except (ProbeError, ValueError, TypeError):
                pass
            emit(take_pause(run_id, seconds, reads, target, label))
            return
        time.sleep(WAIT_POLL_SECONDS)
    write_pause(
        run_id,
        {
            **read_pause(run_id, label),
            "timed_out_at": datetime.now(timezone.utc).isoformat(),
            "reads_at_timeout": reads,
        },
        label,
    )
    raise ProbeError("the collector did not reach the pause read in time")


def pause_status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    emit(
        {
            "run_id": run_id,
            "pause": read_pause(run_id),
            "stopped": unit_is_stopped(METRICS_UNIT),
            "waiter": unit_state(f"{pause_waiter_unit(run_id)}.service"),
            # Every pause of the run by label (the gap pause under ""), each
            # with its waiter: the 009 recount hold is read from here.
            "pauses": {
                label: {
                    **read_pause(run_id, label),
                    "waiter": unit_state(f"{pause_waiter_unit(run_id, label)}.service"),
                }
                for label in pause_labels(run_id)
            },
            "metrics_unit": unit_state(METRICS_UNIT),
            "effective_endpoint": effective_endpoint(METRICS_UNIT),
        }
    )


def resume_paused(run_id: str, label: str = "") -> dict[str, Any]:
    """SIGCONT the unit, stop the waiter and the resume deadman (unless we are it)."""

    run(["systemctl", "kill", "--kill-whom=all", "--signal=SIGCONT", METRICS_UNIT])
    waiter = pause_waiter_unit(run_id, label)
    run(["systemctl", "stop", f"{waiter}.service"], check=False)
    run(["systemctl", "reset-failed", f"{waiter}.service"], check=False)
    deadman = resume_deadman_unit(run_id, label)
    run(["systemctl", "stop", f"{deadman}.timer"], check=False)
    if unit_state(f"{deadman}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{deadman}.service"], check=False)
    run(
        ["systemctl", "reset-failed", f"{deadman}.service", f"{deadman}.timer"],
        check=False,
    )
    record = read_pause(run_id, label)
    if record and "resumed_at" not in record:
        record["resumed_at"] = datetime.now(timezone.utc).isoformat()
        write_pause(run_id, record, label)
    return {
        "run_id": run_id,
        "label": label,
        "pause": record,
        "stopped": unit_is_stopped(METRICS_UNIT),
        "metrics_unit": unit_state(METRICS_UNIT),
    }


def resume_collector(arguments: argparse.Namespace) -> None:
    emit(
        resume_paused(
            safe_id(arguments.run_id, "run ID"),
            safe_label(getattr(arguments, "label", "")),
        )
    )


def restart_collector(arguments: argparse.Namespace) -> None:
    """One restart of the metrics unit with the repoint drop-in in place."""

    run_id = safe_id(arguments.run_id, "run ID")
    if not dropin_path(run_id).is_file():
        raise ProbeError("the repoint drop-in of this run is not in place")
    if unit_is_stopped(METRICS_UNIT):
        raise ProbeError("resume the paused collector before restarting it")
    expected = effective_endpoint(METRICS_UNIT)
    before = unit_state(METRICS_UNIT)
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restart")
    effective = environment.get(ENDPOINT_ENV)
    if effective != expected:
        raise ProbeError(
            f"collector endpoint {effective} changed across the restart ({expected})"
        )
    emit(
        {
            "run_id": run_id,
            "unit": METRICS_UNIT,
            "restarted_at": datetime.now(timezone.utc).isoformat(),
            "effective_endpoint": effective,
            "process_environment": environment,
            "before": before,
            "after": after,
        }
    )


def restore_unit(run_id: str) -> dict[str, Any]:
    """Undo the repoint and prove the real endpoint is read again (idempotent).

    Removes the drop-in and its environment file, reloads, restarts the unit
    when the drop-in was in place or the running process still reads another
    endpoint (a second restore finds nothing to undo and restarts nothing), and
    reads the endpoint back from the new process. The deadman is disarmed
    unless this process *is* the deadman.
    """

    dropin = dropin_path(run_id)
    override = override_env_path(run_id)
    original = DEFAULT_UPSTREAM
    repoint = run_dir(run_id) / "repoint.json"
    if repoint.is_file():
        try:
            original = str(json.loads(repoint.read_text())["original_endpoint"])
        except (ValueError, KeyError):
            pass
    # A paused collector is resumed first (and its deadman disarmed): systemd
    # would SIGCONT it for the restart anyway, but no pause record or waiter --
    # the gap pause's or a labelled pause's -- must outlive the window.
    resumed_pauses = {
        label: resume_paused(run_id, label) for label in pause_labels(run_id)
    }
    resumed = resumed_pauses.get("")
    present = dropin.exists()
    dropin.unlink(missing_ok=True)
    override_present = override.exists()
    override.unlink(missing_ok=True)
    run(["systemctl", "daemon-reload"])
    restarted = present
    if not present:
        # Nothing to unload: restart only a process that still reads another
        # endpoint (a drop-in removed without a restart), never a clean unit.
        current = effective_endpoint(METRICS_UNIT, seconds=SETTLE_SECONDS)
        restarted = current is not None and current != original
    if restarted:
        run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after, environment = settled_environment(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restore")
    if restarted and ENDPOINT_ENV not in environment:
        raise ProbeError("the restored collector did not exec within the settle window")
    effective = environment.get(ENDPOINT_ENV)
    if effective is not None and effective != original:
        raise ProbeError(
            f"collector endpoint {effective} is not the original {original}"
        )
    if dropin_loaded(METRICS_UNIT, dropin):
        raise ProbeError("systemd still lists the repoint drop-in after the reload")
    deadman = deadman_unit(run_id)
    if unit_state(f"{deadman}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{deadman}.timer"], check=False)
        run(["systemctl", "stop", f"{deadman}.service"], check=False)
    run(
        ["systemctl", "reset-failed", f"{deadman}.service", f"{deadman}.timer"],
        check=False,
    )
    return {
        "run_id": run_id,
        "unit": METRICS_UNIT,
        "dropin_removed": present and not dropin.exists(),
        "override_removed": override_present and not override.exists(),
        "restarted": restarted,
        "restored_endpoint": effective,
        "original_endpoint": original,
        "process_environment": environment,
        "after": after,
        "resumed_pause": resumed,
        "resumed_pauses": resumed_pauses,
    }


def restore_collector(arguments: argparse.Namespace) -> None:
    emit(restore_unit(safe_id(arguments.run_id, "run ID")))


def stop_proxy(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    # Never stop the proxy while the collector still points at it.
    effective = effective_endpoint(METRICS_UNIT)
    if effective and effective.endswith(f":{checked_port(arguments.port)}/metrics"):
        raise ProbeError(
            "refusing to stop the proxy while the collector still points at it"
        )
    unit = proxy_unit(run_id)
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    root = run_dir(run_id)
    audit_present = (root / "replay.jsonl").is_file()
    if root.is_dir():
        shutil.rmtree(root)
    emit(
        {
            "run_id": run_id,
            "unit": f"{unit}.service",
            "audit_was_present": audit_present,
        }
    )


def snapshot(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "metrics_unit": unit_state(METRICS_UNIT),
            "effective_endpoint": effective_endpoint(METRICS_UNIT),
            "baseline_endpoint": baseline_endpoint(),
            "open_windows": open_windows(),
            "leftover_units": leftover_units(),
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight").set_defaults(handler=preflight)
    commands.add_parser("snapshot").set_defaults(handler=snapshot)

    source = commands.add_parser("install-source")
    source.add_argument("--run-id", required=True)
    source.add_argument("--name", required=True)
    source.add_argument("--sha256", required=True)
    source.add_argument("--b64", required=True)
    source.set_defaults(handler=install_source)

    deploy = commands.add_parser("deploy-proxy")
    deploy.add_argument("--run-id", required=True)
    deploy.add_argument("--gpu-uuid", required=True)
    deploy.add_argument("--port", type=int, default=9411)
    deploy.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    deploy.add_argument("--max-seconds", type=int, default=600)
    deploy.add_argument("--gpu-stop-c", type=float, default=80)
    deploy.add_argument("--memory-stop-c", type=float, default=85)
    deploy.set_defaults(handler=deploy_proxy)

    repoint = commands.add_parser("repoint-collector")
    repoint.add_argument("--run-id", required=True)
    repoint.add_argument("--port", type=int, default=9411)
    repoint.add_argument("--restore-seconds", type=int, default=600)
    # 0: derive the first-read budget from the node's collector.env cadence.
    repoint.add_argument("--first-read-seconds", type=int, default=0)
    repoint.set_defaults(handler=repoint_collector)

    arm_parser = commands.add_parser("arm")
    arm_parser.add_argument("--run-id", required=True)
    arm_parser.add_argument("--port", type=int, default=9411)
    arm_parser.set_defaults(handler=arm)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--port", type=int, default=9411)
    status_parser.set_defaults(handler=status_command)

    pause = commands.add_parser("pause-collector")
    pause.add_argument("--run-id", required=True)
    pause.add_argument("--after-reads", type=int, required=True)
    pause.add_argument("--port", type=int, default=9411)
    pause.add_argument("--seconds", type=int, required=True)
    pause.add_argument("--wait-seconds", type=int, default=300)
    # A second, independent pause of the same run (the 009 recount hold).
    pause.add_argument("--label", default="")
    pause.set_defaults(handler=pause_collector)

    waiter = commands.add_parser("pause-wait")
    waiter.add_argument("--run-id", required=True)
    waiter.add_argument("--after-reads", type=int, required=True)
    waiter.add_argument("--port", type=int, default=9411)
    waiter.add_argument("--seconds", type=int, required=True)
    waiter.add_argument("--wait-seconds", type=int, default=300)
    waiter.add_argument("--label", default="")
    waiter.set_defaults(handler=pause_wait)

    pause_state = commands.add_parser("pause-status")
    pause_state.add_argument("--run-id", required=True)
    pause_state.set_defaults(handler=pause_status)

    resume = commands.add_parser("resume-collector")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--label", default="")
    resume.set_defaults(handler=resume_collector)

    restart = commands.add_parser("restart-collector")
    restart.add_argument("--run-id", required=True)
    restart.set_defaults(handler=restart_collector)

    restore = commands.add_parser("restore-collector")
    restore.add_argument("--run-id", required=True)
    restore.set_defaults(handler=restore_collector)

    stop = commands.add_parser("stop-proxy")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--port", type=int, default=9411)
    stop.set_defaults(handler=stop_proxy)
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


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        # The host fixture's exception text shows only the tokens it whitelists
        # for ``error_kind: probe`` (class, digest, site); the message travels
        # in the same line and is kept, redacted, as the case's probe-error.json.
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": exception_site(exc),
                "error_message": str(exc)[:2000],
                "error_detail": getattr(exc, "detail", None),
                "command": getattr(arguments, "command", None),
            }
        )
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
