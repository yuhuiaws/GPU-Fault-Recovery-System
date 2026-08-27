from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gpu_fault.app import ApplicationContext
from gpu_fault.app.processor_factory import ProcessorFactory
from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import SqliteStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--claim-file", required=True)
    parser.add_argument("--fail-release", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    os.environ.update(
        {
            "GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE": "true",
            "GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS": "1",
            "GPU_FAULT_PROCESSOR_REQUEST_RENEW_SECONDS": "0.2",
            "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS": "1",
            "GPU_FAULT_PROCESSOR_WORKERS": "1",
            "GPU_FAULT_PROCESSOR_REPLAY_SECRET": "r" * 40,
            "POD_UID": (
                "ha008-release-fail" if args.fail_release else "ha008-release-ok"
            ),
        }
    )
    store = SqliteStore(args.database)
    context = ApplicationContext(
        store=store,
        execution_token="x" * 40,
        processor_replay_secret="r" * 40,
    )
    factory = ProcessorFactory(
        context,
        mode="active-active",
        exit_grace_seconds=0.05,
    )
    processor = factory.build()
    assert processor is not None
    store.enqueue_processor_request(
        ProcessorRequest.from_http(
            method="POST",
            path="/v1/workload-observations",
            query="",
            body=b"{}",
            content_type="application/json",
            cluster_id="ha008-cluster",
        )
    )
    claimed = store.claim_active_processor_requests(
        processor.owner_id,
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=1),
        limit=1,
    )[0]
    Path(args.claim_file).write_text(
        json.dumps(
            {
                "request_id": claimed.request_id,
                "owner_id": processor.owner_id,
                "lane_epoch": claimed.leader_epoch,
                "lease_token": claimed.lease_token,
            },
            sort_keys=True,
        )
    )
    processor._request_started(claimed)
    if args.fail_release:
        processor._release = lambda _item: (_ for _ in ()).throw(
            RuntimeError("HA008 injected release failure")
        )
    print(
        json.dumps(
            {
                "request_id": claimed.request_id,
                "status": "LEASED",
                "fail_release": args.fail_release,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    processor._mark_execution_deadline_exceeded(claimed)
    time.sleep(30)


if __name__ == "__main__":
    main()
