"""Read collector readiness for the named clusters, in one exec.

Request: one environment variable, ``ADMIN_CLUSTER_IDS_JSON`` (a JSON list of
cluster ids). Response: one JSON object on stdout with a ``clusters`` entry per
cluster id -- the control plane's ``/v1/collector-readiness/{cluster_id}``
document -- and ``collector_thresholds``, the silence threshold in seconds the
control plane applies to each collector kind (empty when the deployed
``gpu_fault`` predates ``collector_kinds``, as a rollback's probe may find).

This is the re-poll behind the verify's post-roll collector grace. The full
``control_api_inspect`` answers six questions per cluster; a cluster that only
needs its collectors asked about again should not pay for the other five, nor
for a ``kubectl get nodes`` against its GPU cluster, every ten seconds.

The ``ADMIN_`` prefix is deliberate, as in ``control_api_inspect``: the Pod's
own environment carries the execution token under the runtime's prefix, so an
input under that prefix would be indistinguishable from a variable the runtime
is entitled to read.
"""

import json
import os
import urllib.parse
import urllib.request
from typing import Any


def get(path: str) -> Any:
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        headers={
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def collector_thresholds() -> dict[str, float]:
    try:
        from gpu_fault.collector_kinds import collector_silent_thresholds
    except ImportError:
        return {}
    return {
        kind.value: seconds for kind, seconds in collector_silent_thresholds().items()
    }


def main() -> None:
    clusters = json.loads(os.environ["ADMIN_CLUSTER_IDS_JSON"])
    result: dict[str, Any] = {
        "collector_thresholds": collector_thresholds(),
        "clusters": {
            cluster_id: get(
                "/v1/collector-readiness/" + urllib.parse.quote(cluster_id, safe="")
            )
            for cluster_id in clusters
        },
    }
    print(json.dumps(result, separators=(",", ":")))


main()
