from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.responses import JSONResponse

from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded
from tests.app_services._cov95_runtime_admission import Admission
from tests.app_services._cov95_runtime_admission import (
    admission_fixture as admission_fixture,
)
from tests.app_services._cov95_runtime_dispatch import (
    ImmediateIo,
    dependencies,
    dispatch,
    request,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

REPLAY_HEADERS = {
    "X-GPU-Fault-Processor-Replay": "unit-authenticated",
    "X-GPU-Fault-Processor-Request-ID": "unit-request",
    "X-GPU-Fault-Processor-Owner-ID": "unit-owner",
    "X-GPU-Fault-Processor-Lane-Epoch": "1",
    "X-GPU-Fault-Processor-Lane-Token": "unit-lane",
    "X-GPU-Fault-Processor-Lane-Key": base64.urlsafe_b64encode(b"unit-lane").decode(),
}


@pytest.mark.parametrize(
    "defect", ["missing-id", "empty-id", "long-id", "epoch", "base64", "utf8"]
)
def test_authenticated_replay_still_rejects_malformed_binding_headers(
    admission: Admission, defect: str
) -> None:
    headers = dict(REPLAY_HEADERS)
    if defect == "missing-id":
        headers.pop("X-GPU-Fault-Processor-Request-ID")
    elif defect == "empty-id":
        headers["X-GPU-Fault-Processor-Request-ID"] = ""
    elif defect == "long-id":
        headers["X-GPU-Fault-Processor-Request-ID"] = "x" * 257
    elif defect == "epoch":
        headers["X-GPU-Fault-Processor-Lane-Epoch"] = "invalid"
    elif defect == "base64":
        headers["X-GPU-Fault-Processor-Lane-Key"] = "a"
    else:
        headers["X-GPU-Fault-Processor-Lane-Key"] = base64.urlsafe_b64encode(
            b"\xff"
        ).decode()
    deps = dependencies(admission, replay_authorized=lambda request: True)
    response = dispatch(admission, request(headers=headers), deps)
    assert response.status_code == 400
    assert deps.store_io.calls == 0
    assert deps.processor_replay_tracker.phases() == {}


@pytest.mark.parametrize("active", [False, True])
def test_replay_lost_leadership_or_lane_finishes_tracking_without_serving(
    admission: Admission, active: bool
) -> None:
    called = []

    async def handler(value: Any) -> JSONResponse:
        called.append(value)
        return JSONResponse({})

    deps = dependencies(
        admission,
        replay_authorized=lambda request: True,
        processor=SimpleNamespace(active_consumers=active, is_leader=lambda: False),
    )
    response = dispatch(
        admission, request(headers=REPLAY_HEADERS), deps, next_call=handler
    )
    assert response.status_code == 409
    assert called == []
    assert deps.processor_replay_tracker.phases() == {}
    assert deps.store_io.calls == int(active)


@pytest.mark.parametrize("raises", [False, True])
def test_authorized_legacy_replay_tracks_dispatch_and_cleans_up_after_handler(
    admission: Admission, raises: bool
) -> None:
    deps = dependencies(
        admission,
        replay_authorized=lambda request: True,
        processor=SimpleNamespace(active_consumers=False, is_leader=lambda: True),
    )
    seen = []

    async def handler(value: Any) -> JSONResponse:
        seen.append(deps.processor_replay_tracker.phases())
        if raises:
            raise RuntimeError("synthetic handler failure")
        return JSONResponse({"served": True})

    if raises:
        with pytest.raises(RuntimeError, match="handler failure"):
            dispatch(
                admission, request(headers=REPLAY_HEADERS), deps, next_call=handler
            )
    else:
        assert (
            dispatch(
                admission, request(headers=REPLAY_HEADERS), deps, next_call=handler
            ).status_code
            == 200
        )
    assert list(seen[0]) == ["handler_dispatch"]
    assert deps.processor_replay_tracker.phases() == {}


@pytest.mark.parametrize("fail_at", [1, 2])
@pytest.mark.parametrize("deadline", [False, True])
def test_decode_and_request_build_capacity_failures_are_distinguished(
    admission: Admission, fail_at: int, deadline: bool
) -> None:
    error = (
        RequestDeadlineExceeded("unit deadline")
        if deadline
        else StoreIoCapacityExceeded("unit capacity")
    )
    io = ImmediateIo(error=error, fail_at=fail_at)
    deps = dependencies(admission, decode_io=io)
    response = dispatch(admission, request(), deps)
    assert response.status_code == 503
    assert json.loads(response.body)["detail"] == (
        "request deadline exceeded" if deadline else "request decode capacity exceeded"
    )
    assert io.calls == fail_at


def test_cached_decoded_payload_does_not_bypass_the_body_size_limit(
    admission: Admission,
) -> None:
    deps = dependencies(admission, processor_max_request_bytes=2)
    response = dispatch(admission, request(body=b"{} ", cached={}), deps)
    assert response.status_code == 413
    assert deps.state.oversize_rejections == 1
    assert deps.decode_io.calls == 0


@pytest.mark.parametrize(
    "scope", ["global", "cluster", "global_reserved", "cluster_reserved"]
)
def test_queue_rejection_counts_its_scope_and_never_returns_a_receipt(
    admission: Admission, scope: str
) -> None:
    async def reject(item: Any) -> tuple[None, str]:
        return None, scope

    batcher = SimpleNamespace(submit=reject)
    deps = dependencies(
        admission,
        processor_admission_batcher=batcher,
        fault_admission_batcher=batcher,
        evidence_admission_batcher=batcher,
    )
    response = dispatch(
        admission, request(headers={"X-GPU-Fault-Cluster-ID": "cluster-a"}), deps
    )
    assert response.status_code == 429
    assert json.loads(response.body)["scope"] == scope
    assert deps.processor_admission_rejections[scope] == 1
    if scope in {"cluster", "cluster_reserved"}:
        assert deps.processor_admission_rejections[scope + "\x1fcluster-a"] == 1


@pytest.mark.parametrize("result", ["global", "cluster", "coalesced", "missing"])
def test_spool_result_keeps_capacity_coalescing_and_missing_result_distinct(
    admission: Admission, result: str
) -> None:
    async def spool(item: Any) -> tuple[Any, Any]:
        return (item if result == "coalesced" else None), (
            None if result == "missing" else result
        )

    deps = dependencies(
        admission,
        telemetry_spool_enabled=True,
        telemetry_spool_batcher=SimpleNamespace(submit=spool),
    )
    if result == "missing":
        with pytest.raises(RuntimeError, match="returned no request"):
            dispatch(admission, request(), deps)
        return
    response = dispatch(admission, request(), deps)
    assert response.status_code == (202 if result == "coalesced" else 429)
    if result == "coalesced":
        assert deps.state.telemetry_spool_coalesced == 1
    else:
        assert deps.telemetry_spool_rejections[result] == 1
