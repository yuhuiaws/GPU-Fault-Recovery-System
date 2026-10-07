"""The in-Pod spare release the warm-spare cleanups run (``release_spares``).

The release goes through the provider lifecycle adapter's own spare
coordinator -- the product path REPLACE_NODE reserved the spare through --
whichever provider owns it: ``gpu-fault-hyperpod-adapter`` on HyperPod,
``gpu-fault-eks-adapter`` on a plain EKS cluster. Live 2026-10-07
(gpu-eks-cluster DESTR-003 attempt 9): keyed on the HyperPod owner string the
probe raised StopIteration inside the executor Pod, exited 1 behind the
redaction, and the handed-over spare stayed ALLOCATED and reserved, refusing
DESTR-008, DESTR-014 and ``config spare --release`` after it. The script is
shipped as source and runs under whatever ``gpu_fault`` the executor Pod has.
"""

from __future__ import annotations

RELEASE_SPARES = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment

nodes = json.loads(sys.argv[1])
incident_id = sys.argv[2]
executor = executor_from_environment()
adapters = list(executor.adapters)
adapter = next(
    (
        item
        for item in adapters
        if getattr(item, "spare_coordinator", None) is not None
    ),
    None,
)
if adapter is None:
    owners = sorted(
        str(getattr(item, "owner", "") or type(item).__name__) for item in adapters
    )
    raise RuntimeError(
        "no lifecycle adapter with a spare coordinator among the executor's "
        "adapters: " + ", ".join(owners)
    )
adapter.spare_coordinator.release(nodes, incident_id)
print(json.dumps({
    "released_nodes": nodes,
    "incident_id": incident_id,
    "adapter_owner": str(getattr(adapter, "owner", "")),
}, sort_keys=True))
"""
