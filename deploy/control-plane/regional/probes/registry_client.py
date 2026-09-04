"""Make one authenticated request to the regional registry API from inside the Pod.

Request: ``sys.argv[1:]`` is ``method path``, and the request body is read from
stdin (the exec that runs this one is given ``-i``).
Response: the response body on stdout.

The registry API is bound to loopback and authenticated with the execution
token, so the only place that can call it is a Pod that already holds the token
in its environment. Passing the method and path in argv rather than in the body
keeps the body exactly the JSON document the API expects, so nothing has to
unwrap an envelope on either side.
"""

import json
import os
import sys
import urllib.request

method, path = sys.argv[1:]
body = sys.stdin.buffer.read()
request = urllib.request.Request(
    "http://127.0.0.1:8080" + path,
    data=body or None,
    method=method,
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
