"""027 ``old_attempt`` against the product's own ingest, not a scripted answer.

The variant used to "post" its foreign heartbeat by rewriting the live
attempt's progress file -- the live reporter then posted a heartbeat of the
LIVE attempt -- and passed because nothing stopped. Here the heartbeat comes
from ``HEARTBEAT_POST``, the script the variant runs in the reporter sidecar,
executed as the sidecar would run it (its environment, the product's own model
and client); the client's one delivery is handed to the product's
``TrainingHealthService.ingest`` over an in-memory store holding the live
attempt's allocation, and the records the variant grades are read by the real
``TRAINING_PROBE`` over the same store. The graders then judge the product's
own answers: refused for the attempt the product does not run, accepted for the
live attempt with the same identity (the control: the refusal is about the
attempt id), no state left behind, the live attempt still heard.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.collectors.sinks import CollectorError, HttpEventSink
from gpu_fault.training_health import TrainingHealthService
from gpu_fault.training_models import TrainingProgressHeartbeat
from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_progress_probes import (
    HEARTBEAT_POST,
    RECEIPT_FOLLOW_SECONDS,
    TRAINING_PROBE,
)
from scripts.e2e.regional.general_live_progress_verdicts import (
    foreign_refusal,
    foreign_state_errors,
    live_kept_errors,
)
from tests._builders import attempt_observation, build_store, container_observation

CLUSTER = "gpu-a"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
JOB = "gl27-old-a1-deadbeef"
LIVE = f"{JOB}-a001"
FOREIGN = JOB + progress.FOREIGN_ATTEMPT_SUFFIX
POD_UID = "0f0e0d0c-pod-uid"
TOKEN = "t" * 64
#: The reporter sidecar's environment (``_reporter_container``).
SIDECAR_ENV = {
    "GPU_FAULT_CONTROL_PLANE_URL": "https://gpu-fault-api.example.internal:8443",
    "GPU_FAULT_CLUSTER_ID": CLUSTER,
    "GPU_FAULT_CONTROL_PLANE_TOKEN": TOKEN,
    "CONTAINER_NAME": "pytorch",
    "NODE_NAME": NODE,
    "POD_UID": POD_UID,
}
POST_CODE = compile(HEARTBEAT_POST, "<heartbeat-post>", "exec")
PROBE_CODE = compile(TRAINING_PROBE, "<training-probe>", "exec")


def live_store() -> Any:
    """An in-memory store with the live attempt's RUNNING allocation: rank 0's
    training container in the job's Pod on the lane node."""

    store = build_store()
    store.save_attempt_observation(
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
    return store


def post(
    monkeypatch: pytest.MonkeyPatch,
    store: Any,
    attempt_id: str,
    fields: dict[str, Any],
    *,
    deliver: Any = None,
) -> tuple[dict[str, Any], str]:
    """Run ``HEARTBEAT_POST`` as ``python -c`` would in the sidecar; its one
    delivery goes to the product's ingest over ``store`` (or ``deliver``)."""

    service = TrainingHealthService(store)

    def ingest(_sink: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == TRAINING_PROGRESS_PATH, path
        heartbeat = TrainingProgressHeartbeat.model_validate(payload)
        return service.ingest(heartbeat).model_dump(mode="json")

    monkeypatch.setattr(HttpEventSink, "deliver_once", deliver or ingest)
    for name, value in SIDECAR_ENV.items():
        monkeypatch.setenv(name, value)
    output = io.StringIO()
    argv = ["-c", attempt_id, "0", json.dumps(fields)]
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
        exec(POST_CODE, {"__name__": "__main__"})
    text = output.getvalue()
    return json.loads(text.splitlines()[-1]), text


def records(store: Any, since: datetime) -> dict[str, Any]:
    """``TRAINING_PROBE`` over ``store`` with the arguments ``_records`` sends."""

    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    argv = ["training-probe", CLUSTER, NODE, JOB, LIVE, since.isoformat(), FOREIGN]
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", argv),
        contextlib.redirect_stdout(output),
    ):
        exec(PROBE_CODE, {"__name__": "__main__"})
    return json.loads(output.getvalue())


def test_the_product_refuses_the_foreign_heartbeat_and_accepts_the_live_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = live_store()
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    foreign, text = post(
        monkeypatch, store, FOREIGN, {"step": 10, "loss": 1.0, "numerical_error": True}
    )
    assert TOKEN not in text, "the cluster token is never printed"
    sent = foreign["heartbeat"]
    assert (sent["attempt_id"], sent["rank"], sent["cluster_id"]) == (
        FOREIGN,
        0,
        CLUSTER,
    ), sent
    assert (sent["node_id"], sent["pod_uid"], sent["container_name"]) == (
        NODE,
        POD_UID,
        "pytorch",
    ), "the live Pod's identity: only the attempt id differs"
    assert sent["numerical_error"] is True, sent
    assert foreign["delivered"] is True, foreign
    assert foreign["body"]["accepted"] is False, foreign["body"]
    assert foreign_refusal(foreign, FOREIGN) == [], "the grader reads the refusal"

    # The control: the same script, the same Pod identity, the live attempt.
    live, _text = post(monkeypatch, store, LIVE, {"step": 11, "loss": 1.0})
    assert live["body"]["accepted"] is True, live["body"]
    assert foreign_refusal(live, LIVE) != [], "an accepted heartbeat is a FAIL"

    read = records(store, since)
    assert read["other_progress_states"] == {FOREIGN: []}, read["other_progress_states"]
    assert foreign_state_errors(read, FOREIGN) == [], "no state for the foreign id"
    assert [item["rank"] for item in read["progress_states"]] == [0], read
    assert live_kept_errors(read, foreign) == [], "the live attempt is still heard"


def test_a_delivery_the_ingest_never_graded_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(_sink: Any, _path: str, _payload: dict[str, Any]) -> dict[str, Any]:
        raise CollectorError("collector event rejected (401): no", status_code=401)

    answer, _text = post(
        monkeypatch,
        live_store(),
        FOREIGN,
        {"step": 10, "numerical_error": True},
        deliver=refused,
    )
    assert (answer["delivered"], answer["status"]) == (False, 401), answer
    with pytest.raises(VariantInconclusive, match="not graded by the product"):
        foreign_refusal(answer, FOREIGN)


def test_the_records_read_without_other_attempts_lists_none() -> None:
    output = io.StringIO()
    context = types.SimpleNamespace(store=live_store())
    since = datetime.now(timezone.utc).isoformat()
    argv = ["training-probe", CLUSTER, NODE, JOB, LIVE, since]
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", argv),
        contextlib.redirect_stdout(output),
    ):
        exec(PROBE_CODE, {"__name__": "__main__"})
    read = json.loads(output.getvalue())
    assert read["other_progress_states"] == {}, "the other 027 variants' reads"
    with pytest.raises(VariantInconclusive, match="listed no progress entry"):
        foreign_state_errors(read, FOREIGN)


class Kubectl:
    """The GPU-plane kubectl the real job execs through; scripted stdout."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    def kubectl(self, plane: str, *arguments: str, **options: Any) -> str:
        self.calls.append((plane, arguments, options))
        return self.stdout


def real_job(stdout: str) -> tuple[Any, Kubectl]:
    """The real ``ManagedProgressJob`` with only what ``post_heartbeat`` reads:
    the context's kubectl and the job's Pod on rank 0's node."""

    job = object.__new__(progress.ManagedProgressJob)
    regional = Kubectl(stdout)
    job.ctx = types.SimpleNamespace(regional=regional)
    job.rank_nodes = {0: NODE}
    job.fixture = types.SimpleNamespace(
        pods=lambda: [{"name": f"{JOB}-master-0", "node": NODE}]
    )
    return job, regional


def test_post_heartbeat_runs_the_script_in_the_ranks_reporter_sidecar() -> None:
    answer = {"delivered": True, "body": {"accepted": False}, "heartbeat": {}}
    job, regional = real_job("warning: noise\n" + json.dumps(answer) + "\n")
    got = job.post_heartbeat(
        0, FOREIGN, step=10, loss=1.0, numerical_error=True, checkpoint_ref=None
    )
    assert got == answer, got
    ((plane, arguments, options),) = regional.calls
    assert plane == "gpu", plane
    assert arguments == (
        "exec",
        f"{JOB}-master-0",
        "-c",
        "progress-reporter-0",
        "--",
        progress.component_python("gpu"),
        "-c",
        HEARTBEAT_POST,
        FOREIGN,
        "0",
        json.dumps({"step": 10, "loss": 1.0, "numerical_error": True}),
        str(RECEIPT_FOLLOW_SECONDS),
    ), arguments
    assert "input_text" not in options, "the script is an argument, not stdin"
    assert options["timeout"] >= RECEIPT_FOLLOW_SECONDS + 30 + 30 + 30, (
        "the exec outlives the POST, the follow bound and one last read"
    )


def test_post_heartbeat_output_that_is_no_answer_is_an_undelivered_answer() -> None:
    job, _regional = real_job("Traceback (most recent call last):\n  boom\n")
    got = job.post_heartbeat(0, FOREIGN, step=10)
    assert got["delivered"] is False, got
    assert got["error"].startswith("unparseable: "), got
    with pytest.raises(VariantInconclusive):
        foreign_refusal(got, FOREIGN)
