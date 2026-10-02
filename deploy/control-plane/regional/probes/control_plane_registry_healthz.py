"""Read one control-plane Pod's own ``/healthz?verbose=1`` registry status.

Request on stdin::

    {"port": 8080}

Response: one JSON object on stdout with ``http_status`` (200 or 503 -- the
route answers 503 with the same body while the process is not ready), the
process ``status`` and ``service_role``, and ``regional_registry`` reduced to
the registry runtime's readiness fields: ``member_id``, ``service_role``,
``ready``, ``generation``, ``content_sha256``, ``target_generation``,
``secret_drift`` and ``error``. Nothing else from the health document is
forwarded, and nothing here is a credential.

The question this answers is the one a kubelet Ready condition does not: the
Ready condition is the *last* probe's verdict, up to a probe period old, while
a publish that execs into a Pod moments after a rollout needs to know that the
registry runtime inside it has loaded the durable head *now*. ``port`` is a
parameter because the three CPU roles serve their health documents on
different ports (api-ha 8080, control-worker 8081, telemetry-spool-worker
8082); the exec that runs this picks the port off the Deployment's own
readiness probe, so the probe never guesses.
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Any

REGISTRY_FIELDS = (
    "member_id",
    "service_role",
    "ready",
    "generation",
    "content_sha256",
    "target_generation",
    "secret_drift",
    "error",
)


def main() -> None:
    request = json.load(sys.stdin)
    port = int(request["port"])
    url = f"http://127.0.0.1:{port}/healthz?verbose=1"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            http_status = int(response.status)
            document: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            raise
        # The readiness route encodes the same document on its 503 branch so a
        # not-ready process can still say which generation it is on.
        http_status = 503
        document = json.load(exc)
    registry = document.get("regional_registry")
    print(
        json.dumps(
            {
                "http_status": http_status,
                "status": document.get("status"),
                "service_role": document.get("service_role"),
                "regional_registry": (
                    {key: registry.get(key) for key in REGISTRY_FIELDS}
                    if isinstance(registry, dict)
                    else None
                ),
            },
            separators=(",", ":"),
        )
    )


main()
