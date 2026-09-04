"""Report which nodes have a live agent, grouped by cluster.

Request: nothing on stdin.
Response: one JSON object on stdout, ``{"<cluster>": ["<node>", ...]}``.

"Live" means ACTIVE *and* holding an unexpired lease. Lifecycle alone is not
enough -- a node whose agent died stays ACTIVE until its lease lapses -- and the
engine uses this set to decide which nodes a wave may touch.
"""

import json
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext


def main() -> None:
    now = datetime.now(timezone.utc)
    result: dict[str, list[str]] = {}
    for item in ApplicationContext.from_environment().store.list_agents():
        lifecycle = getattr(item.lifecycle_state, "value", item.lifecycle_state)
        if (
            lifecycle == "ACTIVE"
            and item.lease_expires_at is not None
            and item.lease_expires_at > now
        ):
            result.setdefault(item.cluster_id, []).append(item.node_id)
    print(
        json.dumps(
            {
                cluster_id: sorted(set(node_ids))
                for cluster_id, node_ids in result.items()
            },
            sort_keys=True,
        )
    )


main()
