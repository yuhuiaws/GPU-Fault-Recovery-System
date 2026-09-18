from __future__ import annotations

import json

from tests._builders import processor_request
from tests.store._cov95_runtime_models import CLUSTER, NODE, NOW

HOST_PATH = "/v1/collector-events/host-telemetry"
FAULT_PATH = "/v1/collector-events/nvidia-kernel"


def sample(identity, *, node=NODE, cluster=CLUSTER, path=HOST_PATH, value=1):
    payload = {"node_id": node, "summary": True, "value": value}
    if path == FAULT_PATH:
        payload = {"node_id": node, "lines": []}
    return processor_request(
        path, cluster_id=cluster, body=json.dumps(payload).encode()
    ).model_copy(update={"request_id": identity, "created_at": NOW, "updated_at": NOW})


def assert_rejected_spool_duplicates_are_not_acknowledged(store, reason):
    requests = [sample("sample-first"), sample("sample-last", value=2)]
    limits = (
        {"max_depth": 0, "max_cluster_depth": 10}
        if reason == "global"
        else {"max_depth": 10, "max_cluster_depth": 0}
    )
    result = store.try_spool_telemetry_requests(requests, now=NOW, **limits)
    assert result == [(None, reason), (None, reason)], (
        "a discarded duplicate cannot be acknowledged when its winning sample was rejected"
    )
    assert store.telemetry_spool_stats(now=NOW)["depth"] == 0, (
        "the rejected batch must leave no accepted sample in storage"
    )
