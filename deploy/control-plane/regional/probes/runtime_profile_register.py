"""Register a runtime profile through the Pod's own HTTP API.

Request: the runtime profile document on stdin.
Response: the API's JSON response, re-serialized compactly on stdout.

This goes over loopback HTTP rather than through ``ApplicationContext`` on
purpose: registration is a write, and the API is where the write's validation,
authorization and audit live. The execution token comes from the Pod's own
environment, so nothing secret crosses the exec boundary.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# The same budget as the registry client probe; the API answers a registration
# in well under a second when it is healthy, so a wait this long is itself
# the diagnosis the ERROR line below carries out of the Pod.
TIMEOUT_SECONDS = 30


def main() -> int:
    payload = sys.stdin.buffer.read()
    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/runtime-profiles",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            print(json.dumps(json.load(response), separators=(",", ":")))
    except urllib.error.HTTPError as exc:
        # The API's JSON detail names the rule that refused the profile; it
        # carries no key material (the token travelled in the request header).
        detail = exc.read(512).decode("utf-8", "replace").strip()
        print(
            f"ERROR: runtime profile registration refused: HTTP {exc.code} "
            f"{exc.reason}: {detail}",
            file=sys.stderr,
        )
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            "ERROR: runtime profile registration got no answer within "
            f"{TIMEOUT_SECONDS}s from the control-plane API: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


raise SystemExit(main())
