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
* **repoint-collector** -- a systemd drop-in on the one allow-listed unit
  (``gpu-fault-metrics-collector.service``) overrides ``GPU_FAULT_DCGM_METRICS_URL``
  to the proxy. The real exporter config and ``collector.env`` are untouched and
  no second producer of the same identity is started. A ``systemd-run`` deadman
  restores the original endpoint if the runner dies.
* **arm** -- one ``POST /arm`` with the acknowledgement header. ``status`` is a
  read that never consumes a replay step.
* **restore-collector / stop-proxy** -- remove the drop-in and the unit; the
  proxy is stopped only after the endpoint is back on the real exporter.

Every answer is one JSON object on the last stdout line. The Node Agent is
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, build_opener, ProxyHandler

COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
VENV_PYTHON = Path("/opt/gpu-fault/current/venv/bin/python")
ACCEPTANCE_ROOT = Path("/run/gpu-fault-env-replay")
#: The one unit a window may repoint. The Node Agent and every other collector
#: are deliberately absent.
METRICS_UNIT = "gpu-fault-metrics-collector.service"
ENDPOINT_ENV = "GPU_FAULT_DCGM_METRICS_URL"
DEFAULT_UPSTREAM = "http://127.0.0.1:9400/metrics"
ACKNOWLEDGEMENT = "ISOLATED_GPU_TEST_NODE"
REPLAY_TOOL_NAME = "gpu_environment_replay.py"
PLAN_NAME = "plan.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 1800
MIN_PORT, MAX_PORT = 1024, 65535
MAX_SECONDS_LIMIT = 600


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


def dropin_dir() -> Path:
    return Path("/run/systemd/system") / f"{METRICS_UNIT}.d"


def dropin_path(run_id: str) -> Path:
    return dropin_dir() / f"gpu-fault-env-replay-{digest(run_id)}.conf"


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


def effective_endpoint(unit: str) -> str | None:
    """The ``GPU_FAULT_DCGM_METRICS_URL`` the unit would run with now."""

    environment = unit_state(unit).get("Environment") or ""
    found: str | None = None
    for token in environment.split():
        name, separator, value = token.partition("=")
        if separator and name == ENDPOINT_ENV:
            found = value
    return found


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
                if key
                in {
                    ENDPOINT_ENV,
                    "GPU_FAULT_METRICS_MODE",
                    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
                    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
                    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
                    "GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS",
                }
            },
            "metrics_unit": unit_state(METRICS_UNIT),
            "effective_endpoint": effective_endpoint(METRICS_UNIT),
            "device_limits": device_limits(),
            "gpu_inventory": gpu_inventory(),
            "current_tlimit_readings": current_tlimit_readings(),
            "open_windows": sorted(
                path.name for path in ACCEPTANCE_ROOT.glob("*") if path.is_dir()
            )
            if ACCEPTANCE_ROOT.is_dir()
            else [],
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
def repoint_collector(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    port = checked_port(arguments.port)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    before = unit_state(METRICS_UNIT)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active at baseline")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError("a repoint drop-in for this run already exists")
    original = baseline_endpoint()
    proxy_endpoint = f"http://127.0.0.1:{port}/metrics"
    # Arm the deadman before the drop-in exists, so a crash between the two
    # still restores the real endpoint.
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
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text(
        "[Service]\n"
        f"# gpu-fault env-live acceptance repoint (run {run_id})\n"
        f"Environment={ENDPOINT_ENV}={proxy_endpoint}\n",
        encoding="utf-8",
    )
    (run_dir(run_id) / "repoint.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "original_endpoint": original,
                "proxy_endpoint": proxy_endpoint,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after = unit_state(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the repoint")
    if effective_endpoint(METRICS_UNIT) != proxy_endpoint:
        raise ProbeError("collector did not pick up the proxy endpoint")
    emit(
        {
            "run_id": run_id,
            "unit": METRICS_UNIT,
            "original_endpoint": original,
            "proxy_endpoint": proxy_endpoint,
            "deadman_timer": unit_state(f"{deadman}.timer"),
            "before": before,
            "after": after,
        }
    )


def arm(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    port = checked_port(arguments.port)
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


def restore_collector(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    dropin = dropin_path(run_id)
    original = DEFAULT_UPSTREAM
    repoint = run_dir(run_id) / "repoint.json"
    if repoint.is_file():
        try:
            original = str(json.loads(repoint.read_text())["original_endpoint"])
        except (ValueError, KeyError):
            pass
    removed = dropin.exists()
    dropin.unlink(missing_ok=True)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", METRICS_UNIT], timeout=120)
    after = unit_state(METRICS_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{METRICS_UNIT} is not active after the restore")
    effective = effective_endpoint(METRICS_UNIT)
    if effective is not None and effective != original:
        raise ProbeError(
            f"collector endpoint {effective} is not the original {original}"
        )
    deadman = deadman_unit(run_id)
    if unit_state(f"{deadman}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{deadman}.timer"], check=False)
        run(["systemctl", "stop", f"{deadman}.service"], check=False)
    run(
        ["systemctl", "reset-failed", f"{deadman}.service", f"{deadman}.timer"],
        check=False,
    )
    emit(
        {
            "run_id": run_id,
            "unit": METRICS_UNIT,
            "dropin_removed": removed and not dropin.exists(),
            "restored_endpoint": effective,
            "original_endpoint": original,
            "after": after,
        }
    )


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
            "open_windows": sorted(
                path.name for path in ACCEPTANCE_ROOT.glob("*") if path.is_dir()
            )
            if ACCEPTANCE_ROOT.is_dir()
            else [],
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
    repoint.set_defaults(handler=repoint_collector)

    arm_parser = commands.add_parser("arm")
    arm_parser.add_argument("--run-id", required=True)
    arm_parser.add_argument("--port", type=int, default=9411)
    arm_parser.set_defaults(handler=arm)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--port", type=int, default=9411)
    status_parser.set_defaults(handler=status_command)

    restore = commands.add_parser("restore-collector")
    restore.add_argument("--run-id", required=True)
    restore.set_defaults(handler=restore_collector)

    stop = commands.add_parser("stop-proxy")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--port", type=int, default=9411)
    stop.set_defaults(handler=stop_proxy)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit({"error": f"{type(exc).__name__}: {exc}", "command": arguments.command})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
