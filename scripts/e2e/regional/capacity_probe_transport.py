"""Fleet executor pins and the supervised local forward to a capacity probe.

Two things the probe control plane inherits from the live worker make the
runner's own executors and tunnel fragile:

* The probe polls the namespace's ``gpu-fault-release-metadata`` ConfigMap
  (``gpu_fault.fleet_pins``) and enforces the fleet's executor pins, so every
  claim the runner sends must present the artifact digest and compatibility
  digest the live executor Deployment presents. They are read from that
  ConfigMap at run time, never hard-coded.
* ``kubectl port-forward`` drops the whole tunnel on the first pod-side stream
  error and prints one line per accepted connection. The forward's output goes
  to a file, a dead forward in front of the same Pod is re-established on the
  same local port, and a restarted or vanished Pod is a distinct failure whose
  restart count, last container state, events and log tail are written to the
  case directory before anything is cleaned up.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol

from scripts.e2e.regional.capacity_acceptance_common import (
    CapError,
    Probe,
    ProbeTransportError,
    utc_now,
    write_json,
)

RELEASE_METADATA_CONFIGMAP = "gpu-fault-release-metadata"
EXECUTOR_PIN_KEYS = {
    "executor_artifact_sha256": "required-regional-executor-artifact-sha256",
    "executor_compatibility_digest": (
        "required-regional-executor-compatibility-digest"
    ),
}
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
PORT_FORWARD_LOG_TAIL_LINES = 40
PROBE_LOG_TAIL_LINES = 200

_TRANSPORT_LOCK = RLock()


class ProbeHost(Protocol):
    """What the transport helpers need from the capacity harness."""

    run_dir: Path

    def kubectl(
        self, *args: str, check: bool = True, **options: Any
    ) -> subprocess.CompletedProcess[str]: ...

    def start_port_forward(
        self, service: str, local_port: int, log_path: Path
    ) -> subprocess.Popen[str]: ...

    def wait_healthz(self, url: str, port_forward: subprocess.Popen[str]) -> None: ...


def executor_pins_from_release_metadata(configmap: Any) -> dict[str, str | None]:
    """The executor pins the namespace's control planes require, or None each.

    Read from ``gpu-fault-release-metadata`` exactly as ``fleet_pins.py`` maps
    it: an absent or empty key means no pin is declared. Anything else must be
    a lowercase SHA-256, as ``RegionalExecutorCompatibilityPolicy`` demands.
    """

    if not isinstance(configmap, dict) or not isinstance(
        configmap.get("data", {}), dict
    ):
        raise CapError("release metadata ConfigMap is missing or malformed")
    data = configmap.get("data") or {}
    pins: dict[str, str | None] = {}
    for field, key in EXECUTOR_PIN_KEYS.items():
        raw = data.get(key, "")
        if not isinstance(raw, str):
            raise CapError(f"release metadata {key} is not a string")
        value = raw.strip().lower()
        if value and SHA256_HEX.fullmatch(value) is None:
            raise CapError(f"release metadata {key} is not a SHA-256 digest")
        pins[field] = value or None
    return pins


def pod_identity(item: Mapping[str, Any]) -> tuple[str, int]:
    """A Pod's uid and the sum of its container restart counts."""

    metadata = item.get("metadata") or {}
    status = item.get("status") or {}
    restarts = 0
    for key in ("containerStatuses", "initContainerStatuses"):
        for entry in status.get(key) or []:
            count = entry.get("restartCount", 0) if isinstance(entry, dict) else 0
            restarts += count if type(count) is int and count > 0 else 0
    return str(metadata.get("uid", "")), restarts


def stop_port_forward(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def probe_pod_report(host: ProbeHost, probe: Probe) -> dict[str, Any]:
    """The probe Pod as the API server sees it now, plus its recent logs.

    Read before any cleanup so a restarted or crashed probe leaves its restart
    count, last container state, events and log tail in the case directory
    instead of vanishing with the Deployment.
    """

    report: dict[str, Any] = {
        "observed_at": utc_now(),
        "pod_name": probe.pod,
        "expected_uid": probe.pod_uid,
        "expected_restart_count": probe.restart_count,
        "pod": None,
        "events": [],
        "logs": None,
        "previous_logs": None,
        "port_forward": None,
    }
    raw = host.kubectl(
        "get", "pod", probe.pod, "--ignore-not-found", "-o", "json", check=False
    ).stdout.strip()
    item = json.loads(raw) if raw else None
    if isinstance(item, dict):
        uid, restarts = pod_identity(item)
        status = item.get("status") or {}
        report["pod"] = {
            "uid": uid,
            "phase": status.get("phase"),
            "deletion_timestamp": (item.get("metadata") or {}).get("deletionTimestamp"),
            "restart_count": restarts,
            "containers": [
                {
                    "name": entry.get("name"),
                    "ready": entry.get("ready"),
                    "restart_count": entry.get("restartCount"),
                    "state": entry.get("state"),
                    "last_state": entry.get("lastState"),
                }
                for entry in status.get("containerStatuses") or []
                if isinstance(entry, dict)
            ],
        }
    events = host.kubectl(
        "get",
        "events",
        "--field-selector",
        f"involvedObject.name={probe.pod}",
        "-o",
        "json",
        check=False,
    ).stdout.strip()
    try:
        listed = json.loads(events).get("items", []) if events else []
    except ValueError:
        listed = []
    report["events"] = sorted(
        (
            {
                "time": entry.get("lastTimestamp") or entry.get("eventTime"),
                "type": entry.get("type"),
                "reason": entry.get("reason"),
                "message": str(entry.get("message", ""))[:300],
            }
            for entry in listed
            if isinstance(entry, dict)
        ),
        key=lambda entry: str(entry["time"]),
    )
    if item is not None:
        report["logs"] = host.kubectl(
            "logs", probe.pod, f"--tail={PROBE_LOG_TAIL_LINES}", check=False
        ).stdout.splitlines()[-PROBE_LOG_TAIL_LINES:]
        if report["pod"] and report["pod"]["restart_count"] > probe.restart_count:
            report["previous_logs"] = host.kubectl(
                "logs",
                probe.pod,
                "--previous",
                f"--tail={PROBE_LOG_TAIL_LINES}",
                check=False,
            ).stdout.splitlines()[-PROBE_LOG_TAIL_LINES:]
    forward = probe.port_forward
    tail: list[str] = []
    if probe.forward_log is not None and probe.forward_log.is_file():
        tail = probe.forward_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-PORT_FORWARD_LOG_TAIL_LINES:]
    report["port_forward"] = {
        "alive": forward is not None and forward.poll() is None,
        "returncode": None if forward is None else forward.poll(),
        "log_tail": tail,
    }
    return report


def ensure_probe_transport(
    host: ProbeHost, probe: Probe, *, reason: str
) -> dict[str, Any] | None:
    """Re-establish a dead local forward; refuse a restarted probe Pod.

    Returns None when the forward is alive (the caller's error was not the
    tunnel), the recorded incident when the forward was restarted, and raises
    ``ProbeTransportError`` when the Pod behind it is not the one the case has
    been measuring. Every incident, with the Pod's restart count, last
    container state, events and log tail, is written to
    ``<case>-transport-incidents.json`` before anything is cleaned up.
    """

    with _TRANSPORT_LOCK:
        forward = probe.port_forward
        if forward is None or forward.poll() is None:
            return None
        incidents: list[dict[str, Any]] = host.__dict__.setdefault(
            "transport_incidents", []
        )
        incident = {"reason": reason, **probe_pod_report(host, probe)}
        incidents.append(incident)
        evidence = host.run_dir / f"{probe.case.lower()}-transport-incidents.json"
        write_json(evidence, incidents)
        pod = incident["pod"]
        if (
            pod is None
            or pod["uid"] != probe.pod_uid
            or pod["restart_count"] != probe.restart_count
            or pod["deletion_timestamp"]
        ):
            observed = (
                "vanished"
                if pod is None
                else (
                    f"uid {pod['uid']}, restarts {pod['restart_count']}, "
                    f"phase {pod['phase']}"
                )
            )
            raise ProbeTransportError(
                f"capacity probe Pod {probe.pod} restarted or vanished while "
                f"the case was running (expected uid {probe.pod_uid}, restarts "
                f"{probe.restart_count}; observed {observed}); see {evidence.name}"
            )
        if probe.forward_log is None:
            raise CapError("capacity probe forward has no log destination")
        restarted = host.start_port_forward(
            probe.service, probe.local_port, probe.forward_log
        )
        try:
            host.wait_healthz(probe.url, restarted)
        except BaseException:
            stop_port_forward(restarted)
            raise
        probe.port_forward = restarted
        incident["recovered"] = True
        incident["recovered_at"] = utc_now()
        write_json(evidence, incidents)
        return incident
