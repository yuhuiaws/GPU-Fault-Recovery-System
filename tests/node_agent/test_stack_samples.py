import json

import pytest

from gpu_fault.node_agent.operations.stack_samples import stack_summary
from tests.execution._process_evidence_support import stack_json


def test_json_thread_structure_and_roles_survive_reordered_background_threads():
    main = json.loads(
        stack_json(100, "all_reduce", "/site/torch/distributed/distributed_c10d.py")
    )[0]
    background = {
        **main,
        "thread_id": 99999,
        "os_thread_id": 101,
        "thread_name": "background",
        "frames": [{"name": "log", "filename": "/lib/logging.py", "line": 20}],
    }
    roles = {"100": "TRAINING", "101": "BACKGROUND"}
    first = stack_summary(json.dumps([background, main]), 100, roles)
    second = stack_summary(json.dumps([main, background]), 100, roles)
    assert first == second, (first, second)
    assert len(first["threads"]) == 2 and first["training_thread_declared"] is True, (
        first
    )
    assert first["threads"][0]["role"] == "TRAINING", first
    assert first["threads"][0]["distributed_call_context"] is True, first


@pytest.mark.parametrize(
    "function,filename",
    [
        ("_check_single_tensor", "/site/torch/distributed/distributed_c10d.py"),
        ("all_reduce", "/job/train.py"),
        ("nccl_logging", "/lib/logging.py"),
    ],
)
def test_broad_names_do_not_imply_a_distributed_blocking_location(function, filename):
    result = stack_summary(stack_json(100, function, filename), 100)
    assert result["collective_frames"] is False, result


@pytest.mark.parametrize(
    "text",
    [
        "Error: permission denied",
        "",
        "[]",
        "{}",
        '[{"pid":999,"thread_id":1,"frames":[]}]',
    ],
)
def test_bad_format_or_foreign_pid_is_capture_failure_not_a_signature(text):
    result = stack_summary(text, 100)
    assert result["status"] == "INVALID" and result["signature"] is None, result


def test_container_native_tid_is_bound_to_host_tid_using_namespace_identity():
    payload = json.loads(stack_json(100))
    payload[0]["os_thread_id"] = 1
    result = stack_summary(
        json.dumps(payload),
        100,
        {"1": "TRAINING"},
        {"100": {"start_ticks": 1000, "namespace_ids": [100, 1]}},
    )
    [thread] = result["threads"]
    assert thread["host_thread_id"] == 100 and thread["os_thread_id"] == 1, result
    assert thread["role"] == "TRAINING", result


def test_thread_limit_retains_explicit_training_thread_and_reports_omissions():
    payload = []
    for index in range(80):
        thread = json.loads(stack_json(100, tid=100 + index))[0]
        thread["thread_id"] = 10000 + index
        payload.append(thread)
    result = stack_summary(json.dumps(payload), 100, {"179": "TRAINING"})
    assert result["threads_truncated"] is True, result
    assert result["observed_thread_count"] == 80, result
    assert result["threads"][0]["os_thread_id"] == 179, result
