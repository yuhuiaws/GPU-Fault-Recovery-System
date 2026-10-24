from datetime import datetime, timezone


def passive_document(
    node, ranks, total, *, epoch="epoch", job="job-a", attempt="attempt-a", stamp=None
):
    at = (stamp or datetime.now(timezone.utc)).isoformat()
    return {
        "summary_version": "hung-summary/v1",
        "snapshot_epoch": epoch,
        "node_id": node,
        "job_id": job,
        "attempt_id": attempt,
        "status": "COLLECTED",
        "scan_complete": True,
        "sample_started_at": at,
        "sample_finished_at": at,
        "groups": {},
        "gpu": {},
        "rdma": {},
        "ranks": [
            {
                "pid": 100 + rank,
                "rank": rank,
                "world_size": total,
                "process_identity": {"verified": True, "start_time_ticks": 1000},
                "gpu_uuids": [f"GPU-{rank}"],
                "participant_basis": "GPU_CONTEXT",
                "proc": {"cpu_ticks_delta": 0, "state": "S"},
                "application": {"status": "UNAVAILABLE"},
                "communication": [],
            }
            for rank in ranks
        ],
    }


def add_communication(document, total):
    document["groups"] = {
        str(index): {
            "group_key": str(index) * 64,
            "name": f"pg-{index}",
            "member_count": total,
            "membership_sha256": "a" * 64,
        }
        for index in range(3)
    }
    for row in document["ranks"]:
        row["proc"].update(
            wchan="futex_wait_queue",
            write_bytes_delta=0,
            wchar_delta=0,
            read_bytes_delta=0,
            voluntary_ctxt_switches_delta=0,
        )
        row["gpu_uuids"] = [f"GPU-{row['rank']:036d}"]
        row["application"] = {
            "status": "AVAILABLE",
            "phase": "TRAINING",
            "step_before": 123456,
            "step": 123456,
            "observed_at": document["sample_started_at"],
            "advanced": False,
        }
        row["communication"] = [
            {
                "group": str(index),
                "before": [123456, 123456, 123455],
                "after": [123456, 123456, 123455],
            }
            for index in range(3)
        ]
    document["gpu"] = {
        row["gpu_uuids"][0]: {
            "utilization_gpu_percent": 100,
            "utilization_memory_percent": 0,
        }
        for row in document["ranks"]
    }
    document["rdma"] = {
        f"efa_{index}/1": {
            "valid": True,
            "source": ["rx_bytes", "tx_bytes"],
            "rx_bytes_delta": 0,
            "tx_bytes_delta": 0,
        }
        for index in range(8)
    }
    return document
