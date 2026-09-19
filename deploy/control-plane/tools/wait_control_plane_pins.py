#!/usr/bin/env python3
"""Wait until every control-plane Pod serves the expected fleet pin digest.

The role-split apply used to stamp the release-metadata digest on the Pod
template, so a pin change rolled all three tiers (135 s + 157 s of a release)
only to make the Pods re-read a ConfigMap that had already been written. The
Pods now poll ``gpu-fault-release-metadata`` themselves (``gpu_fault.fleet_pins``)
and report the digest they serve as ``fleet_pins.content_sha256`` on
``/healthz``; this wait is the proof that replaced the roll.

For every Pod of each selected role Deployment (label ``app=<deployment>``)
the probe below is exec'd with the control plane's own interpreter: it reads
``http://127.0.0.1:<port>/healthz`` and prints the body whatever the HTTP
status, since a replica reporting itself degraded (503) still says which pins
it serves. The port is the container's ``http`` port, with the same fallbacks
``wait_for_spool_drain`` uses. Every Pod is re-read on every poll: a replica
restarted mid-wait serves its start-up snapshot until its first ConfigMap
read, so only a pass in which all of them match ends the wait.

Exit 0 once every Pod matches, 1 when the window closes first (one line per
Pod on stderr says what it served, or why it could not be read -- the apply
fails exactly like a failed rollout does), 2 on invalid arguments. Pods that
are terminating or already finished are left out; a Pod that is not Running
yet blocks until it is. A Deployment with no Pods (the spool tier scaled to
zero) has nothing to converge: the rollout gate just before this wait already
proved the replica counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

CONTROL_PLANE_PYTHON = "/opt/gpu-fault/control-plane/bin/python"
SPOOL_WORKER_DEPLOYMENT = "gpu-fault-telemetry-spool-worker"
# Fallbacks when the Pod spec carries no named http port; each matches its
# generated manifest's uvicorn --port. The spool-worker's comes from the same
# variable wait_for_spool_drain reads.
DEFAULT_HTTP_PORTS = {
    "gpu-fault-api-ha": 8080,
    "gpu-fault-control-worker": 8081,
}
SPOOL_WORKER_METRICS_PORT_VARIABLE = "GPU_FAULT_SPOOL_WORKER_METRICS_PORT"
DEFAULT_SPOOL_WORKER_METRICS_PORT = 8082
# Runs inside the Pod. A 503 still carries the JSON body, so an HTTPError's
# body is printed like a 200's; a refused connection stays an error.
HEALTHZ_PROBE = """\
import sys, urllib.error, urllib.request
try:
    body = urllib.request.urlopen("http://127.0.0.1:%d/healthz", timeout=5).read()
except urllib.error.HTTPError as error:
    body = error.read()
sys.stdout.write(body.decode("utf-8", "replace"))
"""
SHA256 = re.compile(r"^[0-9a-f]{64}$")
KUBERNETES_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MINIMUM_TIMEOUT_SECONDS = 30
KUBECTL_TIMEOUT_SECONDS = 30


class PinReadError(Exception):
    """One Pod list or Pod could not be read this poll; retried until the deadline."""


@dataclass(frozen=True)
class Options:
    namespace: str
    expected: str
    timeout_seconds: int
    deployments: tuple[str, ...]
    poll_seconds: float
    kubeconfig: str | None
    kubectl: str


@dataclass(frozen=True)
class PodReading:
    """What one Pod served this poll; ``pod`` is None for a failed Pod list."""

    deployment: str
    pod: str | None
    digest: str | None
    error: str | None

    def converged(self, expected: str) -> bool:
        return self.digest == expected

    def line(self) -> str:
        subject = (
            self.deployment if self.pod is None else f"{self.deployment}/{self.pod}"
        )
        if self.digest is not None:
            return f"{subject}: {self.digest}"
        return f"{subject}: error: {self.error}"


def sha256_digest(value: str) -> str:
    if not SHA256.match(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 hex digest")
    return value


def timeout_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a whole number of seconds: {value!r}"
        ) from None
    if seconds < MINIMUM_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"the timeout must be at least {MINIMUM_TIMEOUT_SECONDS} seconds"
        )
    return seconds


def poll_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a number of seconds: {value!r}"
        ) from None
    if not seconds > 0:
        raise argparse.ArgumentTypeError("the poll interval must be positive")
    return seconds


def kubernetes_name(value: str) -> str:
    if not KUBERNETES_NAME.match(value):
        raise argparse.ArgumentTypeError(f"not a Kubernetes resource name: {value!r}")
    return value


def deployment_names(value: str) -> tuple[str, ...]:
    names = tuple(kubernetes_name(name) for name in value.split(","))
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("a Deployment is listed twice")
    return names


def parse_arguments(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Wait until every Pod of the selected control-plane Deployments "
            "reports the expected fleet pin digest on /healthz."
        )
    )
    parser.add_argument("--namespace", required=True, type=kubernetes_name)
    parser.add_argument("--expected-sha256", required=True, type=sha256_digest)
    parser.add_argument("--timeout-seconds", required=True, type=timeout_seconds)
    parser.add_argument(
        "--deployments",
        required=True,
        type=deployment_names,
        help="comma-separated Deployment names; Pods are selected by app=<name>",
    )
    parser.add_argument("--kubeconfig", default=None)
    parser.add_argument("--poll-seconds", default=2.0, type=poll_seconds)
    parser.add_argument(
        "--kubectl",
        default="kubectl",
        help="kubectl binary to run (default: kubectl from PATH)",
    )
    arguments = parser.parse_args(argv)
    return Options(
        namespace=str(arguments.namespace),
        expected=str(arguments.expected_sha256),
        timeout_seconds=int(arguments.timeout_seconds),
        deployments=tuple(arguments.deployments),
        poll_seconds=float(arguments.poll_seconds),
        kubeconfig=str(arguments.kubeconfig) if arguments.kubeconfig else None,
        kubectl=str(arguments.kubectl),
    )


def kubectl_prefix(options: Options) -> list[str]:
    prefix = [options.kubectl]
    if options.kubeconfig:
        prefix.extend(["--kubeconfig", options.kubeconfig])
    prefix.extend(["-n", options.namespace])
    return prefix


def run_kubectl(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise PinReadError(f"kubectl {command[-1]} timed out") from None


def failure_text(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return lines[-1][:200] if lines else f"kubectl exited {result.returncode}"


def list_pods(prefix: list[str], deployment: str) -> list[dict[str, Any]]:
    result = run_kubectl(
        [*prefix, "get", "pods", "-l", f"app={deployment}", "-o", "json"]
    )
    if result.returncode:
        raise PinReadError(failure_text(result))
    try:
        document = json.loads(result.stdout)
    except ValueError:
        raise PinReadError("kubectl returned invalid JSON for the Pod list") from None
    items = document.get("items") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise PinReadError("kubectl returned no Pod list")
    return [item for item in items if isinstance(item, dict)]


def http_port(pod: dict[str, Any], deployment: str, environ: Mapping[str, str]) -> int:
    for container in (pod.get("spec") or {}).get("containers") or []:
        for port in container.get("ports") or []:
            if port.get("name") == "http" and isinstance(
                port.get("containerPort"), int
            ):
                return int(port["containerPort"])
    if deployment == SPOOL_WORKER_DEPLOYMENT:
        value = environ.get(SPOOL_WORKER_METRICS_PORT_VARIABLE, "")
        return int(value) if value.isdigit() else DEFAULT_SPOOL_WORKER_METRICS_PORT
    return DEFAULT_HTTP_PORTS.get(deployment, DEFAULT_HTTP_PORTS["gpu-fault-api-ha"])


def read_served_digest(prefix: list[str], pod: str, port: int) -> str:
    result = run_kubectl(
        [*prefix, "exec", pod, "--", CONTROL_PLANE_PYTHON, "-c", HEALTHZ_PROBE % port]
    )
    if result.returncode:
        raise PinReadError(failure_text(result))
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise PinReadError("/healthz returned invalid JSON") from None
    pins = payload.get("fleet_pins") if isinstance(payload, dict) else None
    digest = pins.get("content_sha256") if isinstance(pins, dict) else None
    if not isinstance(digest, str) or not SHA256.match(digest):
        raise PinReadError(
            "/healthz reports no fleet_pins.content_sha256 "
            "(a control plane that predates fleet pin hot reload?)"
        )
    return digest


def poll_once(
    options: Options, environ: Mapping[str, str]
) -> tuple[list[PodReading], list[str]]:
    """Read every live Pod once; also name the Deployments with nothing to read."""

    prefix = kubectl_prefix(options)
    readings: list[PodReading] = []
    empty: list[str] = []
    for deployment in options.deployments:
        try:
            pods = list_pods(prefix, deployment)
        except PinReadError as error:
            readings.append(PodReading(deployment, None, None, str(error)))
            continue
        live = 0
        for pod in pods:
            metadata = pod.get("metadata") or {}
            name = str(metadata.get("name") or "?")
            phase = str((pod.get("status") or {}).get("phase") or "Unknown")
            if metadata.get("deletionTimestamp") or phase in {"Succeeded", "Failed"}:
                continue
            live += 1
            if phase != "Running":
                readings.append(
                    PodReading(deployment, name, None, f"Pod phase {phase}")
                )
                continue
            try:
                digest = read_served_digest(
                    prefix, name, http_port(pod, deployment, environ)
                )
            except PinReadError as error:
                readings.append(PodReading(deployment, name, None, str(error)))
            else:
                readings.append(PodReading(deployment, name, digest, None))
        if not live:
            empty.append(f"{deployment}: no Pods; nothing to converge")
    return readings, empty


def wait_for_convergence(
    options: Options,
    *,
    environ: Mapping[str, str],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    out: TextIO,
    err: TextIO,
) -> int:
    deadline = monotonic() + options.timeout_seconds
    first = True
    while True:
        readings, empty = poll_once(options, environ)
        if first:
            for note in empty:
                print(note, file=err)
            first = False
        pending = [
            reading for reading in readings if not reading.converged(options.expected)
        ]
        if not pending:
            print(
                f"control-plane pins converged: {len(readings)} Pods serve "
                f"{options.expected}",
                file=out,
            )
            return 0
        remaining = deadline - monotonic()
        if remaining <= 0:
            for reading in readings:
                print(reading.line(), file=err)
            print(
                f"control-plane pins did not converge to {options.expected} "
                f"within {options.timeout_seconds}s",
                file=err,
            )
            return 1
        print(
            f"control-plane pins: {len(readings) - len(pending)}/{len(readings)} Pods "
            f"serve {options.expected}; polling again: "
            + "; ".join(reading.line() for reading in pending),
            file=err,
        )
        sleep(min(options.poll_seconds, remaining))


def main(
    argv: Sequence[str] | None = None,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    options = parse_arguments(argv)
    return wait_for_convergence(
        options,
        environ=os.environ,
        monotonic=monotonic,
        sleep=sleep,
        out=sys.stdout,
        err=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
