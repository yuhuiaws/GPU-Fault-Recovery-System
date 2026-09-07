#!/usr/bin/env python3
"""In-Pod registry probe of GF-REGIONAL-BOOT-023.

Fed on stdin to ``python3 - <http-port>`` inside one CPU control-plane Pod.
It computes the two digests ARCH-H3 compares -- the operator-configured view
of the ``GPU_FAULT_REGIONAL_CLUSTERS_JSON`` Secret this process started from,
and the same view of the durable registry head in Aurora -- with the deployed
``regional_registry_config_sha256``, and reads the Pod's own ``/healthz`` and
``/livez`` on loopback so the reported ``secret_drift`` can be checked against
what the digests say. It prints one JSON object and never writes anything.
"""

from __future__ import annotations

import json
import socket
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def fetch(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=10) as response:
            body = response.read()
            return {
                "status": response.status,
                "payload": json.loads(body) if body else {},
            }
    except HTTPError as exc:
        body = exc.read()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", "replace")[:500]}
        return {"status": exc.code, "payload": payload}
    except (URLError, OSError, ValueError) as exc:
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    port = int(sys.argv[1])
    from gpu_fault.app import ApplicationContext
    from gpu_fault.regional_registry import regional_registry_config_sha256

    context = ApplicationContext.from_environment()
    store = context.store
    head = store.get_regional_registry_head()
    revision = store.get_regional_registry_revision(head.generation)
    report = {
        "hostname": socket.gethostname(),
        "secret_config_sha256": context.regional_registry_secret_sha256,
        "durable_config_sha256": regional_registry_config_sha256(
            revision.registrations
        ),
        "head_generation": head.generation,
        "head_content_sha256": head.content_sha256,
        "healthz": fetch(f"http://127.0.0.1:{port}/healthz"),
        "livez": fetch(f"http://127.0.0.1:{port}/livez"),
    }
    print(json.dumps(report, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
