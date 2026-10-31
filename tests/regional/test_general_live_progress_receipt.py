"""027 ``old_attempt`` on the regional path: the receipt is not the decision.

Live (HyperPod lane), the foreign heartbeat's POST came back
``{"accepted": true, "coalesced": false, "processor_request_id": ...,
"status_url": "/v1/processor/requests/..."}`` and the variant FAILed
"the product accepted a heartbeat ... it does not run" -- while every record
read showed the heartbeat refused. ``/v1/training-progress`` is a
``receipt=True`` channel (``channel_registry``): with a processor the ingress
admits the POST to the queue and answers 202 at once
(``app/middleware/dispatch.py``); the handler's answer exists only once the
processor replayed the row, and ``status_url`` returns it when COMPLETED
(``app/routes/processor.py``). The module test of S59b drove
``TrainingHealthService.ingest`` synchronously and never met the receipt.

Here the shipped ``HEARTBEAT_POST`` runs against that path: its one delivery
is admitted as the ingress admits it (the dispatch middleware's 202 body over
a ``ProcessorRequest`` of the POST), the "processor" replays the row through
the product's real ``/v1/training-progress`` route and commits the handler's
status and body to the row, and the script's ``GET status_url`` is answered by
the product's real receipt route over the same row -- 202 PENDING before the
replay, the stored answer with ``X-GPU-Fault-Processor-Status: COMPLETED``
after it. The grader reads the completed answer, never the admission.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from typing import Any
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH, channel_for_path
from gpu_fault.collectors.sinks import HttpEventSink
from gpu_fault.processor import ProcessorRequest, ProcessorRequestStatus
from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_progress_probes import HEARTBEAT_POST
from scripts.e2e.regional.general_live_progress_verdicts import (
    foreign_refusal,
    ingest_decision,
)
from tests._builders import attempt_observation, container_observation
from tests.app_services._cov95_runtime_api import Api
from tests.app_services._cov95_runtime_api import api_fixture as api_fixture

CLUSTER = "cluster-a"  # the Api fixture's own cluster header
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
JOB = "gl27-old-a1-fcea983e"
LIVE = f"{JOB}-a001"
FOREIGN = JOB + progress.FOREIGN_ATTEMPT_SUFFIX
POD_UID = "ff97b4f0-2945-431a-98f8-e2b61535f796"
SIDECAR_ENV = {
    "GPU_FAULT_CONTROL_PLANE_URL": "https://gpu-fault-api.example.internal:8443",
    "GPU_FAULT_CLUSTER_ID": CLUSTER,
    "GPU_FAULT_CONTROL_PLANE_TOKEN": "t" * 64,
    "CONTAINER_NAME": "pytorch",
    "NODE_NAME": NODE,
    "POD_UID": POD_UID,
}
POST_CODE = compile(HEARTBEAT_POST, "<heartbeat-post>", "exec")
#: What the live POST answered (HyperPod lane, the admission receipt).
LIVE_ADMISSION = {
    "accepted": True,
    "coalesced": False,
    "processor_request_id": "processor-idem-3b6deaf9f9ad5d127509e8a51a3a37654da91476",
    "status_url": (
        "/v1/processor/requests/processor-idem-3b6deaf9f9ad5d127509e8a51a3a37654da91476"
    ),
}


class Reply:
    """``urlopen``'s answer for a 2xx read."""

    def __init__(self, status: int, headers: Any, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> Reply:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class RegionalPath:
    """The ingress admission, the processor's replay and the receipt route.

    ``admit`` stands in for the dispatch middleware's admission of a
    ``receipt=True`` POST (its 202 body over a ``ProcessorRequest`` of the
    request); ``replay`` runs the queued POST through the product's own route
    and commits the handler's answer to the row, as ``coordinator._execute``
    and ``replay_completion.finalize_replay_response`` do; ``urlopen`` answers
    the script's reads with the product's own receipt route over the rows. The
    processor replays a row after the first read that found it PENDING when
    ``replays`` is set.
    """

    def __init__(self, api: Api, *, replays: bool = True) -> None:
        self.api = api
        self.replays = replays
        self.rows: dict[str, ProcessorRequest] = {}
        self.reads: list[int] = []

    def admit(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == TRAINING_PROGRESS_PATH, path
        assert channel_for_path(path).receipt, (
            "the ingress answers this channel's POST with a receipt"
        )
        body = json.dumps(payload).encode()
        item = ProcessorRequest.from_http(
            method="POST",
            path=path,
            query="",
            body=body,
            content_type="application/json",
            cluster_id=payload["cluster_id"],
            execution_authorized=False,
            parsed_payload=payload,
        )
        self.rows[item.request_id] = item
        return {
            "accepted": True,
            "processor_request_id": item.request_id,
            "status_url": f"/v1/processor/requests/{item.request_id}",
            "coalesced": False,
        }

    def replay(self, request_id: str) -> None:
        item = self.rows[request_id]
        response = self.api.client.post(
            item.path, content=item.body(), headers={"Content-Type": "application/json"}
        )
        self.rows[request_id] = item.model_copy(
            update={
                "status": ProcessorRequestStatus.COMPLETED,
                "response_status": response.status_code,
                "response_content_type": response.headers.get("content-type"),
                "response_body_base64": base64.b64encode(response.content).decode(),
            }
        )

    def urlopen(self, request: Any, timeout: float | None = None) -> Reply:
        path = urlsplit(request.full_url).path
        response = self.api.client.get(path, headers=dict(request.header_items()))
        self.reads.append(response.status_code)
        if response.status_code == 202 and self.replays:
            self.replay(path.rsplit("/", 1)[-1])  # the processor runs meanwhile
        if response.status_code >= 400:
            raise HTTPError(
                request.full_url,
                response.status_code,
                "error",
                response.headers,  # type: ignore[arg-type]
                io.BytesIO(response.content),
            )
        return Reply(response.status_code, response.headers, response.content)


def live_allocation(api: Api) -> None:
    api.context.regional_mode = True
    api.context.store.save_attempt_observation(
        attempt_observation(
            JOB,
            LIVE,
            datetime.now(timezone.utc),
            cluster_id=CLUSTER,
            containers=[
                container_observation(
                    POD_UID, f"{JOB}-master-0", 0, NODE, container_name="pytorch"
                )
            ],
        )
    )


def post(
    monkeypatch: pytest.MonkeyPatch,
    path: RegionalPath,
    attempt_id: str,
    fields: dict[str, Any],
    *,
    follow_seconds: float = 30,
) -> dict[str, Any]:
    """``HEARTBEAT_POST`` as ``python -c`` runs it in the sidecar."""

    def deliver_once(
        _sink: HttpEventSink, route: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return path.admit(route, payload)

    monkeypatch.setattr(HttpEventSink, "deliver_once", deliver_once)
    monkeypatch.setattr(
        path.api.context.store, "get_processor_request", lambda key: path.rows[key]
    )
    for name, value in SIDECAR_ENV.items():
        monkeypatch.setenv(name, value)
    output = io.StringIO()
    argv = ["-c", attempt_id, "0", json.dumps(fields), str(follow_seconds)]
    with (
        mock.patch("urllib.request.urlopen", path.urlopen),
        mock.patch.object(sys, "argv", argv),
        contextlib.redirect_stdout(output),
    ):
        exec(POST_CODE, {"__name__": "__main__"})
    text = output.getvalue()
    assert SIDECAR_ENV["GPU_FAULT_CONTROL_PLANE_TOKEN"] not in text, "never printed"
    return json.loads(text.splitlines()[-1])


NAN = {"step": 10, "loss": 1.0, "numerical_error": True}


def test_the_completed_decision_behind_the_receipt_is_the_refusal(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_allocation(api)
    path = RegionalPath(api)
    answer = post(monkeypatch, path, FOREIGN, NAN)
    # The POST's answer is the admission -- "accepted" means queued.
    assert answer["body"]["accepted"] is True, answer["body"]
    assert answer["body"]["processor_request_id"] in path.rows, answer["body"]
    receipt = answer["receipt"]
    assert path.reads == [202, 200], "one PENDING read, then the completed answer"
    assert (receipt["terminal"], receipt["processor_status"]) == (True, "COMPLETED")
    assert receipt["http_status"] == 200, receipt
    decision = ingest_decision(answer, FOREIGN)
    assert decision["accepted"] is False, decision
    assert decision["heartbeat_id"] == answer["heartbeat"]["heartbeat_id"], decision
    assert foreign_refusal(answer, FOREIGN) == [], "the ingest refused it"
    assert api.context.store.list_training_progress(CLUSTER, FOREIGN) == [], (
        "nothing was applied for the attempt the product does not run"
    )


def test_the_same_path_accepts_the_live_attempts_heartbeat(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control: same Pod identity, the live attempt -- the completed
    # decision is accepted=true, which the grader FAILs.
    live_allocation(api)
    answer = post(monkeypatch, RegionalPath(api), LIVE, {"step": 11, "loss": 1.0})
    assert ingest_decision(answer, LIVE)["accepted"] is True, answer["receipt"]
    assert foreign_refusal(answer, LIVE) == [
        f"the product accepted a heartbeat for {LIVE}, an attempt it does not run "
        "(0 findings)"
    ], answer["receipt"]


def test_an_admission_the_processor_never_completes_is_inconclusive(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_allocation(api)
    path = RegionalPath(api, replays=False)
    answer = post(monkeypatch, path, FOREIGN, NAN, follow_seconds=0.6)
    receipt = answer["receipt"]
    assert receipt["terminal"] is False, receipt
    assert (receipt["http_status"], receipt["processor_status"]) == (202, "PENDING")
    assert set(path.reads) == {202} and len(path.reads) >= 2, path.reads
    with pytest.raises(VariantInconclusive, match="did not reach a terminal state"):
        foreign_refusal(answer, FOREIGN)


def test_the_live_admission_receipt_alone_is_no_decision() -> None:
    # Today's live answer: the receipt, never followed. Not a FAIL.
    answer = {
        "delivered": True,
        "heartbeat": {"heartbeat_id": "training-progress-99aa09f4"},
        "body": dict(LIVE_ADMISSION),
    }
    with pytest.raises(VariantInconclusive, match="did not reach a terminal state"):
        foreign_refusal(answer, FOREIGN)
