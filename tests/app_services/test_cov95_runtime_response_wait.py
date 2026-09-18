from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.middleware import dispatch as dispatch_module
from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded
from gpu_fault.processor import ProcessorRequestStatus
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
from tests.regional._cov95_runtime_support import Clock
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("mode", ["missing", "coalesced"])
def test_queue_admission_requires_a_record_and_reports_coalescing(
    admission: Admission, mode: str
) -> None:
    async def admit(item: Any) -> tuple[Any, str | None]:
        return (None, None) if mode == "missing" else (item, "coalesced")

    deps = dependencies(
        admission, processor_admission_batcher=SimpleNamespace(submit=admit)
    )
    if mode == "missing":
        with pytest.raises(RuntimeError, match="returned no queued request"):
            dispatch(admission, request(), deps)
        assert deps.state.telemetry_coalesced == 0
    else:
        response = dispatch(admission, request(), deps)
        assert response.status_code == 202
        assert json.loads(response.body)["coalesced"] is True
        assert deps.state.telemetry_coalesced == 1


@pytest.mark.parametrize(
    "mode",
    ["capacity", "deadline", "late-capacity", "slow-read", "complete", "untyped"],
)
def test_response_wait_preserves_capacity_deadline_and_result_semantics(
    admission: Admission, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    clock = Clock(step=0.001)
    monkeypatch.setattr(dispatch_module, "time", clock)
    stored = []
    reads = []

    async def admit(item: Any) -> tuple[Any, None]:
        stored.append(item)
        return item, None

    async def read(function: Any, identifier: str) -> Any:
        reads.append(identifier)
        (item,) = stored
        assert identifier == item.request_id
        if mode in {"late-capacity", "slow-read"}:
            clock.value += 10
        if mode == "deadline":
            raise RequestDeadlineExceeded("synthetic deadline")
        if "capacity" in mode:
            raise StoreIoCapacityExceeded("synthetic capacity")
        return item.model_copy(
            update={
                "status": (
                    ProcessorRequestStatus.PENDING
                    if mode == "slow-read"
                    else ProcessorRequestStatus.COMPLETED
                ),
                "response_status": None if mode == "untyped" else 201,
                "response_content_type": None
                if mode == "untyped"
                else "application/json",
                "response_body_base64": base64.b64encode(
                    b'{"committed":true}'
                ).decode(),
            }
        )

    deps = dependencies(
        admission,
        processor_admission_batcher=SimpleNamespace(submit=admit),
        store_io=SimpleNamespace(run=read),
        returns_processor_receipt=lambda path: False,
        decode_io=ImmediateIo(),
    )
    response = dispatch(admission, request(), deps)
    assert reads == [stored[0].request_id]
    if mode in {"complete", "untyped"}:
        assert response.status_code == (201 if mode == "complete" else 500)
        assert json.loads(response.body) == {"committed": True}
        assert ("content-type" in response.headers) is (mode == "complete")
    else:
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert json.loads(response.body) == {
            "detail": (
                "store I/O capacity exceeded"
                if mode == "capacity"
                else "processor response timed out"
            ),
            "processor_request_id": stored[0].request_id,
        }
