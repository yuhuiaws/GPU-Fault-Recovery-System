from __future__ import annotations

import json
import threading
import time

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from scripts.e2e.regional.capacity_acceptance_executor import (
    Cap004Error,
    Cap004ThreadsRunning,
)
from scripts.e2e.regional.capacity_queued_lease import run_queued_lease_proof
from scripts.e2e.regional.probes.cap004_commands import command_snapshot
from tests.regional.test_cap004_executor_proof import RUN_ID, TOKEN, URL
from tests.regional.test_cap004_executor_proof import local_api as shared_api_fixture

local_api = shared_api_fixture


@pytest.mark.parametrize(
    "defect",
    [
        "reserve",
        "pair",
        "identity",
        "missing-expiry",
        "competitor",
        "stale-accepted",
        "late-start",
        "deadline",
    ],
)
def test_queued_proof_fails_closed_on_incomplete_authority_and_deadlines(
    local_api, tmp_path, defect
) -> None:
    command_snapshot(local_api.store, RUN_ID, "seed")
    second_id = f"command-{RUN_ID}-cap004-001"

    def transform(path, request, status, body):
        value = json.loads(body)
        owner = str(request.get("executor_id", ""))
        if path.endswith("/claim"):
            if defect == "reserve" and owner.endswith("-queue-reserve"):
                value["commands"] = []
            if owner.endswith("-queued-executor"):
                if defect == "pair":
                    value["commands"] = value["commands"][:1]
                elif defect == "identity":
                    value["commands"][0]["step"]["node_ids"] = ["foreign"]
                elif defect == "missing-expiry":
                    next(
                        item
                        for item in value["commands"]
                        if item["command_id"] == second_id
                    )["lease_expires_at"] = None
            if defect == "competitor" and owner.endswith("-queue-competitor"):
                value["commands"] = []
        if path.endswith("/renew"):
            if defect == "stale-accepted" and path.endswith(f"/{second_id}/renew"):
                status = 200
                value = local_api.store.get_remote_command(second_id).model_dump(
                    mode="json"
                )
            if defect == "late-start":
                time.sleep(0.1)
        return status, json.dumps(value).encode()

    local_api.transform = transform
    before = set(threading.enumerate())
    with pytest.raises((Cap004Error, ClusterExecutorError), match="CAP004"):
        run_queued_lease_proof(
            URL,
            TOKEN,
            RUN_ID,
            tmp_path,
            timeout_seconds=0.02 if defect in {"late-start", "deadline"} else 5,
        )
    assert not [
        thread
        for thread in threading.enumerate()
        if thread not in before
        and thread.name.startswith(
            (RUN_ID, "gpu-fault-command", "cmd-command-", "lease-command-")
        )
    ], "a failed queue proof must still stop every owned thread"


def test_interrupted_queue_join_preserves_unverified_isolation(
    local_api, tmp_path, monkeypatch
) -> None:
    command_snapshot(local_api.store, RUN_ID, "seed")
    join = threading.Thread.join
    intercepted = []

    def interrupt(thread, *args, **kwargs):
        result = join(thread, *args, **kwargs)
        if thread.name == RUN_ID + "-queued-competitor" and not intercepted:
            intercepted.append(True)
            raise KeyboardInterrupt("isolated join interruption")
        return result

    monkeypatch.setattr(threading.Thread, "join", interrupt)
    with pytest.raises(Cap004ThreadsRunning, match="did not stop"):
        run_queued_lease_proof(URL, TOKEN, RUN_ID, tmp_path, timeout_seconds=5)
    assert intercepted == [True]
