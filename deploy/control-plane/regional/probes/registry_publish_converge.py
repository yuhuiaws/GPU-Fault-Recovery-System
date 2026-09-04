"""Publish one registry revision and wait for the fleet to converge on it.

Request on stdin::

    {"path": "/v1/regional/registry/...", "payload": {...},
     "use_current_generation": false, "timeout_seconds": 120}

Response: the converged status document on stdout, or exit 1 with the generation
that did not converge.

Publish and wait are one program because they are one decision. The generation
this probe waits for is the generation *its own* POST minted, so a second exec
that only polled could not tell the difference between the fleet converging on
this revision and it converging on someone else's -- and an administrator who
saw the publish succeed and the wait fail separately would have no way to know
which revision is live.

``use_current_generation`` reads the generation immediately before the POST
instead of taking one from the deploy host. The optimistic-concurrency check
then rejects the write if anything slipped in between, which is what it is for;
a generation captured on the host minutes earlier would either fail spuriously
or, worse, be reused after a retry.
"""

import json
import os
import sys
import time
import urllib.request
from typing import Any


def request(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
    value = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(value, timeout=30) as response:
        document: dict[str, Any] = json.load(response)
        return document


def main() -> None:
    transaction = json.load(sys.stdin)
    payload = dict(transaction["payload"])
    if transaction.get("use_current_generation"):
        payload["expected_generation"] = int(
            request("GET", "/v1/regional/registry/status")["generation"]
        )
    published = request("POST", transaction["path"], payload)
    generation = int(published["generation"])
    digest = str(published["content_sha256"])
    deadline = time.monotonic() + float(transaction["timeout_seconds"])
    while time.monotonic() < deadline:
        status = request("GET", "/v1/regional/registry/status")
        # Generation *and* digest *and* the converged flag: a generation match
        # alone is satisfied by a revision with the same number and different
        # content after a rollback republished one.
        if (
            int(status["generation"]) == generation
            and str(status["content_sha256"]) == digest
            and status.get("converged") is True
        ):
            print(json.dumps(status, separators=(",", ":")))
            raise SystemExit(0)
        time.sleep(1)
    raise SystemExit(f"regional registry generation {generation} did not converge")


main()
