from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    suffix = sys.argv[2].lower() if len(sys.argv) > 2 else "manual"
    attempt_id = f"perf-p05-{suffix}-a1"
    job_id = f"perf-p05-{suffix}"
    created_at = datetime.now(timezone.utc).isoformat()
    items = []
    for rank in range(count):
        items.append(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": f"perf-p05-{suffix}-{rank:04d}",
                    "namespace": "gpu-fault-perf",
                    "labels": {
                        "app": "gpu-fault-perf-p05",
                        "gpu-fault.io/managed": "true",
                        "gpu-fault.io/job-id": job_id,
                        "gpu-fault.io/attempt-id": attempt_id,
                        "gpu-fault.io/role": "worker",
                        "gpu-fault.io/critical": "true",
                    },
                    "annotations": {
                        "gpu-fault.io/rank": str(rank),
                        "gpu-fault.io/expected-critical-ranks": str(count),
                        "gpu-fault.io/training-container": "trainer",
                        "gpu-fault.io/runtime-profile-version": ("hyperpod-v1"),
                        "gpu-fault.io/restart-budget": "0",
                        "gpu-fault.io/workload-ids": json.dumps(
                            [f"kubernetes/podset/perf-p05-{suffix}"]
                        ),
                        "gpu-fault.io/perf-created-at": created_at,
                    },
                },
                "spec": {
                    "terminationGracePeriodSeconds": 0,
                    "containers": [
                        {
                            "name": "trainer",
                            "image": ("public.ecr.aws/docker/library/busybox:1.36"),
                            "command": ["sh", "-c", "sleep 900"],
                            "resources": {
                                "requests": {
                                    "cpu": "1m",
                                    "memory": "8Mi",
                                },
                                "limits": {
                                    "cpu": "10m",
                                    "memory": "16Mi",
                                },
                            },
                        }
                    ],
                },
            }
        )
    print(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "List",
                "items": items,
            }
        )
    )


if __name__ == "__main__":
    main()
