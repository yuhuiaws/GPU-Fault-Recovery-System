"""A retried request lands on the same queue row instead of a second one.

FINAL-建议汇总 F-E4 (P2-74I, P1-36A, P1-36C, P3-17E, P3-17F). ``request_id``
was a fresh ``uuid4()`` on every POST, so a collector that lost the 202 to a
timeout and retried -- with the very ``Idempotency-Key`` the data plane already
sends -- created a brand-new processor request, and both were executed. Every
store already returns the existing row for a known ``request_id``; the control
plane just never derived one from the key.

The second half of F-E5's positive finding lives here too: the synchronous
response wait for a fault-ingress request used the general Store I/O pool, so
the fault path's isolation ended at admission.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

from gpu_fault.app.middleware import dispatch
from gpu_fault.processor import ProcessorRequestStatus
from tests.processor.test_processor_response_wait import (
    Batcher,
    _dependencies,
    _request,
)

FAULT_PATH = "/v1/attempts/failure-detected"


class _DedupingBatcher(Batcher):
    """Mimics the one thing all three stores do: return the row already
    queued under ``request_id`` instead of inserting a second one."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: dict[str, Any] = {}

    async def submit(self, item):
        self.submitted.append(item)
        existing = self.rows.get(item.request_id)
        if existing is not None:
            return existing, None
        self.rows[item.request_id] = item
        return item, "queued"


def _fault_dependencies(store=None, *, batcher=None):
    batcher = batcher or _DedupingBatcher()
    return dataclasses.replace(
        _dependencies(store, timeout_seconds=0.2),
        fault_admission_batcher=batcher,
        returns_processor_receipt=lambda path: path == FAULT_PATH,
        decode_json_body=lambda body, _encoding: (body, json.loads(body)),
    )


def _post(dependencies, *, body: bytes, headers: dict[str, str]):
    request = _request(body, headers=headers)
    request.scope["path"] = FAULT_PATH
    request.scope["raw_path"] = FAULT_PATH.encode()

    async def unreachable(_request):
        raise AssertionError("the request bypassed the queue")

    return asyncio.run(
        dispatch.dispatch_processor_request(request, unreachable, dependencies)
    )


BODY = b'{"cluster_id":"cluster-a","job_id":"job-1","attempt_id":"att-1","event_id":"evt-1"}'
KEY = {"X-GPU-Fault-Cluster-ID": "cluster-a", "Idempotency-Key": "evt-1"}


def test_a_retry_with_the_same_idempotency_key_gets_the_same_receipt():
    batcher = _DedupingBatcher()
    dependencies = _fault_dependencies(batcher=batcher)

    first = _post(dependencies, body=BODY, headers=KEY)
    second = _post(dependencies, body=BODY, headers=KEY)

    assert first.status_code == second.status_code == 202
    first_id = json.loads(first.body)["processor_request_id"]
    second_id = json.loads(second.body)["processor_request_id"]
    assert first_id == second_id, "the retry became a second processor request"
    assert len(batcher.rows) == 1
    assert first_id.startswith("processor-"), (
        'expected first_id.startswith("processor-") to be true'
    )


def test_the_derived_request_id_is_scoped_to_cluster_path_and_body():
    dependencies = _fault_dependencies()

    def request_id(*, body: bytes = BODY, headers: dict[str, str] = KEY) -> str:
        return json.loads(_post(dependencies, body=body, headers=headers).body)[
            "processor_request_id"
        ]

    baseline = request_id()
    assert request_id() == baseline
    # Same key, different payload: a key reused by mistake must not hand back
    # another request's receipt.
    assert request_id(body=BODY.replace(b"att-1", b"att-2")) != baseline
    # Same key from another cluster is another cluster's request.
    assert request_id(headers={**KEY, "X-GPU-Fault-Cluster-ID": "cluster-b"}) != (
        baseline
    )


def test_without_a_key_every_post_is_still_its_own_request():
    dependencies = _fault_dependencies()
    headers = {"X-GPU-Fault-Cluster-ID": "cluster-a"}

    first = json.loads(_post(dependencies, body=BODY, headers=headers).body)
    second = json.loads(_post(dependencies, body=BODY, headers=headers).body)

    assert first["processor_request_id"] != second["processor_request_id"]


def test_a_blank_key_is_ignored_rather_than_hashed():
    dependencies = _fault_dependencies()
    headers = {**KEY, "Idempotency-Key": "   "}

    first = json.loads(_post(dependencies, body=BODY, headers=headers).body)
    second = json.loads(_post(dependencies, body=BODY, headers=headers).body)

    assert first["processor_request_id"] != second["processor_request_id"]


def test_the_fault_path_waits_for_its_response_on_the_fault_pool():
    polled: list[str] = []

    class Pool:
        def __init__(self, name: str) -> None:
            self.name = name

        async def run(self, function, *arguments, **keywords):
            polled.append(self.name)
            return function(*arguments, **keywords)

    store = type(
        "Store",
        (),
        {
            "get_processor_request": staticmethod(
                lambda _request_id: type(
                    "Row",
                    (),
                    {
                        "status": ProcessorRequestStatus.COMPLETED,
                        "response_content_type": "application/json",
                        "response_status": 200,
                        "response_body": staticmethod(lambda: b'{"ok":true}'),
                    },
                )()
            )
        },
    )()
    dependencies = dataclasses.replace(
        _fault_dependencies(store),
        store_io=Pool("general"),
        fault_store_io=Pool("fault"),
        is_fault_ingress_path=lambda path: path == FAULT_PATH,
        returns_processor_receipt=lambda _path: False,
    )

    response = _post(dependencies, body=BODY, headers=KEY)

    assert response.status_code == 200
    assert polled and set(polled) == {"fault"}, polled
