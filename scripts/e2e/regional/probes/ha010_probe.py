#!/usr/bin/env python3
"""In-Pod liveness/readiness sampler for GF-REGIONAL-HA-010.

Fed on stdin to ``python3 -`` inside every CPU control-plane Pod
(``kubectl exec -i <pod> -- python3 - <port> <duration> <interval>``). It
samples ``GET /livez`` and ``GET /healthz`` on ``127.0.0.1:<port>`` every
``interval`` seconds for ``duration`` seconds and prints ONE JSON document at
the end. Only the standard library is used: the Pod image ships the control
plane, nothing else may be assumed. The exec channel rides the kubelet, so
the sampler keeps running while the process it samples cannot reach Aurora.

A sample records the status code of each endpoint (``null`` when the request
could not complete at all), plus ``regional_registry.ready`` and
``secret_drift`` from the ``/healthz`` body when it parsed.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REQUEST_TIMEOUT_SECONDS = 3.0


def fetch(url: str) -> tuple[int | None, dict[str, Any] | None, str | None]:
    """Status code, parsed JSON body (when any) and an error string."""

    try:
        with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read()
            code = int(response.status)
    except HTTPError as exc:
        body = exc.read()
        code = int(exc.code)
    except (URLError, OSError, TimeoutError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    return code, parsed if isinstance(parsed, dict) else None, None


def sample(port: int) -> dict[str, Any]:
    base = f"http://127.0.0.1:{port}"
    livez, _livez_body, livez_error = fetch(base + "/livez")
    healthz, healthz_body, healthz_error = fetch(base + "/healthz")
    return summarize_sample(
        moment=time.time(),
        livez=livez,
        healthz=healthz,
        healthz_body=healthz_body,
        error=livez_error or healthz_error,
    )


def summarize_sample(
    *,
    moment: float,
    livez: int | None,
    healthz: int | None,
    healthz_body: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    """Flatten one pair of answers into the record the verdicts read."""

    registry = (healthz_body or {}).get("regional_registry") or {}
    ready = registry.get("ready") if isinstance(registry, dict) else None
    drift = registry.get("secret_drift") if isinstance(registry, dict) else None
    return {
        "t": moment,
        "livez": livez,
        "healthz": healthz,
        "registry_ready": ready if isinstance(ready, bool) else None,
        "secret_drift": drift if isinstance(drift, bool) else None,
        "status": (healthz_body or {}).get("status"),
        "error": error,
    }


def main(argv: list[str]) -> int:
    port = int(argv[1])
    duration = float(argv[2])
    interval = float(argv[3]) if len(argv) > 3 else 2.0
    if port <= 0 or duration <= 0 or interval <= 0:
        raise SystemExit("port, duration and interval must be positive")
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration
    while True:
        started = time.monotonic()
        samples.append(sample(port))
        if time.monotonic() >= deadline:
            break
        delay = interval - (time.monotonic() - started)
        if delay > 0:
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
    print(
        json.dumps(
            {
                "pod_hostname": socket.gethostname(),
                "port": port,
                "duration_seconds": duration,
                "interval_seconds": interval,
                "samples": samples,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
