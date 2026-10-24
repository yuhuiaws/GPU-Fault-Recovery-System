import json
from datetime import timedelta

from gpu_fault.node_agent.operations.process_samples import process_window
from gpu_fault.node_agent.operations.stack_samples import stack_summary
from tests.execution._flight_series_support import NOW


def stack_json(pid, function="compute", filename="/job/train.py", *, tid=None):
    return json.dumps(
        [
            {
                "pid": pid,
                "thread_id": pid + 10000,
                "os_thread_id": tid or pid,
                "thread_name": "MainThread",
                "active": True,
                "owns_gil": False,
                "frames": [{"name": function, "filename": filename, "line": 10}],
            }
        ]
    )


def process_signal(
    index=0,
    *,
    rank=None,
    context=False,
    state="R",
    wchan="0",
    cpu=1,
    interval=1,
    function="compute",
    group_size=5,
):
    pid = 100 + index
    roles = {str(pid): "TRAINING"} if context else {}
    samples = [
        {
            "start_ticks": 1000,
            "cpu_ticks": int(sample * interval * cpu * 100),
            "read_bytes": 0,
            "write_bytes": 0,
            "wchar": 0,
            "sample_monotonic": sample * interval,
            "observed_at": (NOW + timedelta(seconds=sample * interval)).isoformat(),
            "state": state,
            "wchan": wchan,
            "threads": {
                str(pid): {"start_ticks": 1000, "state": state, "wchan": wchan}
            },
            "thread_scan_complete": True,
        }
        for sample in range(3)
    ]
    stacks = [
        {
            **stack_summary(
                stack_json(
                    pid,
                    function,
                    "/site/torch/distributed/distributed_c10d.py"
                    if function == "all_reduce"
                    else "/job/train.py",
                ),
                pid,
                roles,
            ),
            "observed_at": sample["observed_at"],
            "returncode": 0,
        }
        for sample in samples
    ]
    return {
        "node_id": f"node-{index}",
        "pid": pid,
        "rank": rank,
        "gpu_uuid": f"GPU-{index}",
        "gpu_uuids": [f"GPU-{index}"],
        "process_identity": {"verified": True, "start_time_ticks": 1000},
        "process_series": {
            "version": "process-series/v1",
            "samples": samples,
            "windows": [
                process_window(left, right, 100)
                for left, right in zip(samples, samples[1:])
            ],
        },
        "stack_series": {"version": "stack-series/v1", "samples": stacks},
        "gpu_series": {
            "mapping_verified": True,
            "shared_with_other_processes": False,
            "samples": [
                {
                    "observed_at": sample["observed_at"],
                    "devices": {
                        f"GPU-{index}": {
                            "device_model": "H100",
                            "utilization_gpu_percent": 60,
                        }
                    },
                }
                for sample in samples
            ],
        },
        "comparison_context": {
            "verified": True,
            "source": "PROCESS_BOUND_SNAPSHOT",
            "job_id": "job-a",
            "attempt_id": "attempt-a",
            "group_name": "tp",
            "members": list(range(group_size)),
            "declared_rank": index,
            "role": "trainer",
            "pipeline_stage": "0",
            "expert_role": "none",
            "hardware_class": "p5",
            "code_version": "image-v1",
            "thread_roles": roles,
        }
        if context
        else {},
    }
