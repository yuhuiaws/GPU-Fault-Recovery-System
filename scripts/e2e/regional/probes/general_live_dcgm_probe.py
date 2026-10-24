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

* **install-source** -- the tool, the reviewed plan and, for the inventory
  variants, the reviewed ``nvidia-smi`` pass-through wrapper, handed in base64
  and checked against a sha256, under ``/run/gfrs-live/<digest>/`` (tmpfs: a
  reboot clears it). The files run verbatim; nothing here re-types them.
* **deploy-proxy** -- a transient unit runs the tool ``--serve`` against the
  real exporter on a loopback-only port (``IPAddressDeny=any``,
  ``IPAddressAllow=localhost``, ``RuntimeMaxSec``); it starts pass-through and
  only replays after ``arm``.
* **repoint-collector** -- the deadman is armed *first*; then a ``/run``
  drop-in adds an ``EnvironmentFile=-/run/gfrs-live/<digest>/collector.env``
  to the one allow-listed unit (``gpu-fault-metrics-collector.service``). An
  ``EnvironmentFile`` and not an ``Environment=`` line: ``collector.env``
  already sets ``GPU_FAULT_DCGM_METRICS_URL`` and systemd lets an environment
  file override ``Environment=`` (systemd.exec), so only a later environment
  file wins. The ``-`` prefix makes a vanished override file fall back to the
  real endpoint. The effective endpoint is read from the running process
  (``/proc/<MainPID>/environ``), never from ``systemctl show``. The real
  exporter, ``collector.env`` and every other unit are untouched; the Node
  Agent is never in the allow-list.
* **arm / disarm / status / audit-read** -- loopback control of the tool.
* **collector-restart** -- one ``systemctl restart`` of the same unit (the
  monitor-restart variant), nothing else.
* **restore-collector / stop-proxy / cleanup** -- remove the drop-in and the
  override file, restart the unit on the real endpoint, verify the process,
  then stop the tool; ``cleanup`` is the deadman's idempotent entry point.

Every answer is one JSON object on the last stdout line. Secrets in
``collector.env`` or the collector's environment never leave the node: only
allow-listed keys are read.
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
#: Collector settings a plan may depend on; nothing else is ever read out.
ENV_KEYS = (
    ENDPOINT_ENV,
    "GPU_FAULT_METRICS_MODE",
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
    "GPU_FAULT_DCGM_EDGE_FILTER_ENABLED",
    "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_STARTUP_GRACE_SECONDS",
    "GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS",
    "GPU_FAULT_DCGM_CORRELATION_WINDOW_SECONDS",
    "GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE",
    "GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
)
#: Process-environment keys the repoint verification may read.
PROCESS_KEYS = (
    ENDPOINT_ENV,
    "PATH",
    "GPU_FAULT_BOOT_ID_PATH",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
    "GPU_FAULT_NODE_INSTANCE_TYPE",
)
#: The expected-count inputs the unknown-count variant blanks (discovery.py:
#: ``expected_gpu_count_from_environment``); an unknown instance type answers
#: ``None``, never a guessed count.
UNKNOWN_INSTANCE_TYPE = "gfrs-live-unknown.48xlarge"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
BOOT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MIN_PORT, MAX_PORT = 1024, 65535
MIN_RESTORE_SECONDS, MAX_RESTORE_SECONDS = 120, 5400
MAX_TOOL_SECONDS = 3600
MAX_SOURCE_BYTES = 512 * 1024
MAX_AUDIT_LINES = 400


class ProbeError(RuntimeError):
    pass


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


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:12]


def run_dir(run_id: str) -> Path:
    return STATE_ROOT / digest(run_id)


def proxy_unit(run_id: str) -> str:
    return f"{UNIT_PREFIX}{digest(run_id)}-proxy"


def deadman_unit(run_id: str) -> str:
    return f"{UNIT_PREFIX}{digest(run_id)}-restore"


def dropin_path(run_id: str) -> Path:
    return DROPIN_DIR / f"{UNIT_PREFIX}{digest(run_id)}.conf"


def override_env_path(run_id: str) -> Path:
    return run_dir(run_id) / "collector.env"


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
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=NRestarts",
            "--property=InvocationID",
            "--property=DropInPaths",
            "--property=ExecMainStartTimestamp",
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


def effective_endpoint() -> str | None:
    """The DCGM URL the running collector process actually uses."""

    state = unit_state(METRICS_UNIT)
    return process_environment(state.get("MainPID", "")).get(ENDPOINT_ENV)


def wait_active(unit: str, *, seconds: int = 60) -> dict[str, str]:
    deadline = time.monotonic() + seconds
    state = unit_state(unit)
    while time.monotonic() < deadline:
        state = unit_state(unit)
        if state.get("ActiveState") == "active" and state.get("MainPID") not in {
            "",
            "0",
        }:
            return state
        time.sleep(1)
    return state


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


# --------------------------------------------------------------------------- #
# survey: read-only facts a plan is built from
# --------------------------------------------------------------------------- #
def gpu_inventory() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        check=False,
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


def residue() -> dict[str, Any]:
    dropins = (
        sorted(path.name for path in DROPIN_DIR.glob("*.conf"))
        if DROPIN_DIR.is_dir()
        else []
    )
    units = run(
        [
            "systemctl",
            "list-units",
            "--all",
            "--plain",
            "--no-legend",
            f"{UNIT_PREFIX}*",
        ],
        check=False,
    ).stdout.split()
    return {
        "run_dirs": sorted(path.name for path in STATE_ROOT.glob("*") if path.is_dir())
        if STATE_ROOT.is_dir()
        else [],
        "metrics_dropins": dropins,
        "units": sorted(item for item in units if item.startswith(UNIT_PREFIX)),
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
            "residue": residue(),
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
    target = root / name
    if target.exists():
        raise ProbeError(f"{name} is already installed for this run")
    target.write_bytes(content)
    target.chmod(0o700 if SOURCES[name] == "wrapper" else 0o600)
    emit({"run_id": run_id, "name": name, "sha256": actual, "bytes": len(content)})


def installed(run_id: str, name: str) -> Path:
    path = run_dir(run_id) / name
    if not path.is_file():
        raise ProbeError(f"install {name} before using it")
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
    common = [
        str(VENV_PYTHON),
        "-I",
        str(tool),
        "--plan",
        str(plan),
        "--upstream",
        upstream,
        "--port",
        str(checked_port(arguments.port)),
        "--max-seconds",
        str(bounded(arguments.max_seconds, 60, MAX_TOOL_SECONDS, "max seconds")),
        "--serve",
        "--ack",
        ACKNOWLEDGEMENT,
        "--audit",
        str(audit),
    ]
    if arguments.tool == FIELD_TOOL:
        if not re.fullmatch(r"[a-z]{3,16}", arguments.profile or ""):
            raise ProbeError("the field tool needs its plan's --profile")
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
    while time.monotonic() < deadline:
        state = unit_state(f"{unit}.service")
        if state.get("ActiveState") == "active":
            try:
                status = tool_status(port)
                break
            except ProbeError:
                pass
        if state.get("ActiveState") in {"failed", "inactive"}:
            raise ProbeError("the replay unit did not stay up")
        time.sleep(0.5)
    if not status:
        raise ProbeError("the replay tool did not answer /status")
    if status.get("armed") is not False:
        raise ProbeError("the replay tool came up armed; it must start pass-through")
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
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "cleanup",
            "--run-id",
            run_id,
        ]
    )
    state = unit_state(f"{name}.timer")
    if state.get("ActiveState") != "active":
        raise ProbeError("the collector restore deadman is not armed")
    return state


def disarm_deadman(run_id: str) -> None:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


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
        [
            path.name
            for path in DROPIN_DIR.glob("*.conf")
            if DROPIN_DIR.is_dir() and path != dropin
        ]
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
    override.write_text(
        "\n".join(
            override_lines(
                run_id,
                port,
                arguments.inventory_wrapper,
                boot_id=arguments.boot_id,
                unknown_expected_count=arguments.unknown_expected_count,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    override.chmod(0o600)
    (run_dir(run_id) / "repoint.json").write_text(
        json.dumps(
            {"run_id": run_id, "original_endpoint": original, "port": port},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
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
    after = wait_active(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the repoint")
    environment = process_environment(after.get("MainPID", ""))
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


def arm(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    port = checked_port(arguments.port)
    if effective_endpoint() != f"http://127.0.0.1:{port}/metrics":
        raise ProbeError("refusing to arm: the collector does not read the replay")
    status, body = loopback(f"http://127.0.0.1:{port}/arm", method="POST")
    if status != 200:
        raise ProbeError(f"arm returned {status}")
    emit(
        {
            "run_id": run_id,
            "armed": True,
            "response": body.decode().strip(),
            "status": tool_status(port),
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
    after = wait_active(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restart")
    emit(
        {
            "run_id": run_id,
            "before": before,
            "after": after,
            "restarted": before.get("InvocationID") != after.get("InvocationID"),
            "effective_endpoint": process_environment(after.get("MainPID", "")).get(
                ENDPOINT_ENV
            ),
        }
    )


# --------------------------------------------------------------------------- #
# restore / stop / cleanup
# --------------------------------------------------------------------------- #
def restore(run_id: str) -> dict[str, Any]:
    dropin = dropin_path(run_id)
    original = baseline_endpoint()
    repoint = run_dir(run_id) / "repoint.json"
    if repoint.is_file():
        try:
            original = str(json.loads(repoint.read_text())["original_endpoint"])
        except (ValueError, KeyError):
            pass
    present = dropin.exists()
    dropin.unlink(missing_ok=True)
    override_env_path(run_id).unlink(missing_ok=True)
    if present:
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after = wait_active(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restore")
    environment = process_environment(after.get("MainPID", ""))
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


def restore_collector(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    result = restore(run_id)
    disarm_deadman(run_id)
    emit({"run_id": run_id, **result})


def stop(run_id: str, port: int | None) -> dict[str, Any]:
    effective = effective_endpoint()
    if (
        effective
        and effective.startswith("http://127.0.0.1:")
        and effective != baseline_endpoint()
    ):
        raise ProbeError("refusing to stop the replay while the collector reads it")
    unit = proxy_unit(run_id)
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    root = run_dir(run_id)
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


def stop_proxy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    emit({"run_id": run_id, **stop(run_id, checked_port(arguments.port))})


def cleanup(arguments: argparse.Namespace) -> None:
    """The deadman's entry point: every step is attempted, in order."""

    run_id = checked_run_id(arguments.run_id)
    result: dict[str, Any] = {"run_id": run_id}
    errors = []
    try:
        result["restore"] = restore(run_id)
    except Exception as exc:  # noqa: BLE001 - recorded, the next step still runs
        errors.append(f"restore: {type(exc).__name__}")
    try:
        result["stop"] = stop(run_id, None)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"stop: {type(exc).__name__}")
    disarm_deadman(run_id)
    result["residue"] = residue()
    result["errors"] = errors
    emit(result)
    if errors:
        raise SystemExit(1)


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

    surveyed = command("survey", survey)
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
    command("arm", arm, port=True)
    command("disarm", disarm, port=True)
    command("advance", advance, port=True)
    command("status", status_command, port=True)
    audit = command("audit-read", audit_read)
    audit.add_argument("--journal", default="replay.jsonl")
    audit.add_argument("--offset", type=int, default=0)
    command("collector-restart", collector_restart)
    command("restore-collector", restore_collector)
    command("stop-proxy", stop_proxy, port=True)
    command("cleanup", cleanup)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "command": arguments.command,
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
