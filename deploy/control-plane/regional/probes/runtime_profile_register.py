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
import urllib.request


def main() -> None:
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
    with urllib.request.urlopen(request, timeout=15) as response:
        print(json.dumps(json.load(response), separators=(",", ":")))


main()
