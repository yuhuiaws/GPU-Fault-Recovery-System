"""The reviewed DCGM field replay tool at its edges: local fixtures, loopback only.

``test_gpu_dcgm_field_replay`` pins the operations on the scrapes a live
exporter serves. These cases pin what the tool refuses or gives up on: the
plan shapes a reviewer would send back (a step list or op list of the wrong
size, an operation that is not an object, a non-finite or out-of-range
structural value), the upstream it will not follow (a redirect) or not read
(an oversized body), the scrapes it serves around (samples that are no GPU's
field, a stale step with nothing to repeat, a GPU or a field that vanished
mid-plan, an alias that is the sample's own name), an alias or a synthesised
gauge that joins a family another GPU already publishes, the HTTP verbs it
answers 404/409 to, and its ``serve``/``__main__`` entry points.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import runpy
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.test_gpu_dcgm_field_replay import (
    XID_DECLARED,
    A,
    B,
    armed,
    declared_plan,
    exporter,
    live_exporter,
    plan,
    product_view,
    serve_once,
    xid,
)
from tools import gpu_dcgm_field_replay as tool

ROOT = Path(__file__).resolve().parents[1]
ACKNOWLEDGED = {"X-GFRS-Replay-Ack": tool.ACKNOWLEDGEMENT}


def without(body: bytes, *, name: str, gpu: str | None = None) -> bytes:
    """``body`` without the ``name`` samples (of ``gpu`` only, when given)."""

    kept = [
        line
        for line in body.decode().splitlines()
        if not (line.startswith(name + "{") and (gpu is None or gpu in line))
    ]
    return ("\n".join(kept) + "\n").encode()


# --------------------------------------------------------------------------- #
# plan review
# --------------------------------------------------------------------------- #
def duplicate(value: Any) -> dict[str, Any]:
    return {"op": "duplicate", "field": "power_limit_w", "gpus": [A], "value": value}


def alias(value: Any, name: str = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS") -> dict:
    return {
        "op": "alias",
        "field": "clock_throttle_reasons",
        "gpus": [A],
        "name": name,
        "value": value,
    }


@pytest.mark.parametrize(
    "operation, message",
    [
        (duplicate(math.nan), "duplicate value must be a finite number"),
        (duplicate(math.inf), "duplicate value must be a finite number"),
        (duplicate("650"), "duplicate value must be a finite number"),
        (duplicate(2**52), "duplicate value is out of range"),
        (alias(True), "alias value must be a finite number"),
        (alias(-1), "alias value must be non-negative"),
        ("omit A", "each operation must be an object"),
    ],
    ids=["nan", "infinity", "text", "too-large", "bool", "negative", "not-an-object"],
)
def test_structural_values_must_be_finite_in_range_objects(
    operation: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan("completeness", (1, [operation]))


@pytest.mark.parametrize(
    "steps",
    [[], {"reads": 1, "ops": []}, [{"reads": 1, "ops": []}] * (tool.MAX_STEPS + 1)],
    ids=["empty", "not-a-list", "too-many"],
)
def test_a_plan_needs_one_to_max_steps(steps: Any) -> None:
    with pytest.raises(ValueError, match=f"plan requires 1..{tool.MAX_STEPS} steps"):
        tool.validate_plan({"profile": "memory", "steps": steps})


@pytest.mark.parametrize(
    "operations",
    [{"op": "omit_gpu", "gpus": [A]}, [{"op": "omit_gpu", "gpus": [A]}] * 33],
    ids=["not-a-list", "too-many"],
)
def test_a_step_holds_a_list_of_at_most_32_operations(operations: Any) -> None:
    with pytest.raises(ValueError, match="ops must be a list of at most 32"):
        tool.validate_plan(
            {"profile": "completeness", "steps": [{"reads": 1, "ops": operations}]}
        )


# --------------------------------------------------------------------------- #
# what an armed read serves around
# --------------------------------------------------------------------------- #
FOREIGN = (
    "# HELP process_cpu_seconds_total Total user and system CPU time\n"
    "# TYPE process_cpu_seconds_total counter\n"
    "process_cpu_seconds_total 12.5\n"
    'DCGM_FI_DEV_GPU_TEMP{gpu="9",modelName="NVIDIA H200"} 41\n'
)


def test_samples_that_are_no_gpus_field_pass_an_armed_read_untouched() -> None:
    replay = armed(
        "memory",
        (1, [{"op": "set", "field": "row_remap_pending", "gpus": [A], "value": 1}]),
    )
    body = exporter() + FOREIGN.encode()

    output, receipt = serve_once(replay, body)

    assert receipt["aborted"] is None, receipt
    assert receipt["served"] == {f"{A}/row_remap_pending": 1.0}
    assert product_view(output)[(A, "row_remap_pending")] == [1.0]
    text = output.decode()
    assert "process_cpu_seconds_total 12.5" in text, "a process metric is kept"
    assert 'DCGM_FI_DEV_GPU_TEMP{gpu="9",modelName="NVIDIA H200"} 41.0' in text, (
        "a sample naming no GPU UUID is served as it came"
    )


def test_an_oversized_upstream_body_is_refused_before_anything_is_served() -> None:
    replay = armed("memory", (1, []))

    with pytest.raises(ValueError, match="upstream response exceeds byte bound"):
        replay.render(b"#" * (tool.MAX_BYTES + 1))
    assert (replay.reads, list(replay.outputs)) == (0, []), "nothing was served"


def test_a_stale_step_with_nothing_to_repeat_aborts_into_pass_through() -> None:
    replay = armed("nvlink", (2, [{"op": "stale", "age": 2}]))

    output, receipt = serve_once(replay, exporter())

    assert output == exporter(), "nothing was served instead of the scrape"
    assert receipt["aborted"] == "no scrape 2 reads old to serve as stale"
    assert serve_once(replay, exporter(crc=5))[0] == exporter(crc=5), (
        "an aborted replay passes fresh upstream data through"
    )


def test_a_gpu_that_vanishes_mid_plan_leaves_no_label_set_to_clone() -> None:
    replay = tool.Replay(declared_plan((2, [xid(A, 32)])), 600)
    replay.arm()
    serve_once(replay, live_exporter())
    alone = without(
        without(live_exporter(), name="DCGM_FI_DEV_PCIE_REPLAY_COUNTER", gpu=A),
        name="DCGM_FI_DEV_GPU_TEMP",
        gpu=A,
    )

    output, receipt = serve_once(replay, alone)

    assert output == alone, "the scrape passes through unchanged"
    assert receipt["aborted"] == (f"no real sample of {A} to clone a label set from")
    assert receipt["served"] == {}


def test_a_duplicated_field_that_vanishes_mid_plan_aborts() -> None:
    replay = armed("completeness", (2, [duplicate(650)]))
    serve_once(replay, exporter())
    gone = without(exporter(), name="DCGM_FI_DEV_POWER_MGMT_LIMIT", gpu=A)

    output, receipt = serve_once(replay, gone)

    assert output == gone
    assert receipt["aborted"] == f"no real {A}/power_limit_w sample to derive from"


def test_an_alias_named_like_the_sample_itself_aborts() -> None:
    replay = armed("completeness", (1, [alias(1, "DCGM_FI_DEV_CLOCKS_EVENT_REASONS")]))

    output, receipt = serve_once(replay, exporter())

    assert output == exporter()
    assert receipt["aborted"] == (
        "DCGM_FI_DEV_CLOCKS_EVENT_REASONS is not a distinct alias of "
        "DCGM_FI_DEV_CLOCKS_EVENT_REASONS"
    )


def test_an_alias_joins_the_family_another_gpu_already_publishes() -> None:
    body = exporter().replace(
        f'DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{gpu="1",UUID="{B}"'.encode(),
        f'DCGM_FI_DEV_CLOCK_THROTTLE_REASONS{{gpu="1",UUID="{B}"'.encode(),
    )
    replay = armed("completeness", (1, [alias(1)]))

    output, receipt = serve_once(replay, body)

    assert receipt["aborted"] is None, receipt
    text = output.decode()
    assert text.count("# TYPE DCGM_FI_DEV_CLOCK_THROTTLE_REASONS") == 1, (
        "one family, not a second header for the same name"
    )
    view = product_view(output)
    assert sorted(view[(A, "clock_throttle_reasons")]) == [0.0, 1.0], view
    assert view[(B, "clock_throttle_reasons")] == [0.0], view
    assert receipt["served"] == {
        f"{A}/clock_throttle_reasons#alias:DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 1
    }


def test_a_synthesised_gauge_joins_the_xid_family_another_gpu_publishes() -> None:
    replay = tool.Replay(declared_plan((1, [xid(A, 32)]), fields=XID_DECLARED), 600)
    replay.arm()
    body = live_exporter(xid={B: 0})

    output, receipt = serve_once(replay, body)

    assert receipt["aborted"] is None, receipt
    view = product_view(output)
    assert view[(A, "xid_last_error")] == [32.0], view
    assert view[(B, "xid_last_error")] == [0.0], "the real gauge is kept"
    assert output.decode().count("# TYPE DCGM_FI_DEV_XID_ERRORS") == 1
    assert receipt["served"][f"{A}/xid_last_error#synthesised"] == (
        "DCGM_FI_DEV_XID_ERRORS"
    )


# --------------------------------------------------------------------------- #
# the loopback endpoint
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def running(server: HTTPServer) -> Iterator[str]:
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


class Redirecting(BaseHTTPRequestHandler):
    """An upstream that answers every scrape with a redirect elsewhere."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/metrics")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def journal(audit: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in audit.getvalue().splitlines()]


def test_an_upstream_redirect_is_never_followed() -> None:
    upstream = HTTPServer(("127.0.0.1", 0), Redirecting)
    with running(upstream) as upstream_base:
        replay = tool.Replay(plan("memory", (1, [])), 600)
        audit = io.StringIO()
        server = tool.make_server(upstream_base + "/metrics", 0, replay, audit)
        with running(server) as base:
            with pytest.raises(HTTPError) as refused:
                urlopen(base + "/metrics", timeout=5)
            assert refused.value.code == 503
            assert refused.value.read() == (
                b"upstream/replay unavailable; no healthy data fabricated\n"
            )
    (record,) = journal(audit)
    assert record["error"] == "upstream redirects are forbidden"
    assert record["read"] == 1
    assert replay.reads == 0, "a refused upstream read consumes no read"


def test_the_endpoint_answers_unknown_paths_and_refused_verbs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool, "open_upstream", lambda *_a, **_k: io.BytesIO(exporter()))
    replay = tool.Replay(
        plan(
            "memory",
            (2, [{"op": "set", "field": "row_remap_pending", "gpus": [A], "value": 1}]),
        ),
        600,
    )
    audit = io.StringIO()
    server = tool.make_server("http://127.0.0.1:9400/metrics", 0, replay, audit)

    def post(base: str, path: str) -> tuple[int, bytes]:
        try:
            with urlopen(
                Request(base + path, method="POST", headers=ACKNOWLEDGED), timeout=5
            ) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    with running(server) as base:
        with pytest.raises(HTTPError) as missing:
            urlopen(base + "/metrics/extra", timeout=5)
        assert (missing.value.code, missing.value.read()) == (404, b"not found\n")
        assert post(base, "/advance") == (409, b"the replay is not on a step")
        assert post(base, "/arm") == (200, b"armed\n")
        assert post(base, "/arm") == (
            409,
            b"replay already armed or ended; start a new audited run",
        )
        assert post(base, "/advance") == (
            409,
            b"the current step is not held for an advance",
        )
    events = [item.get("event") for item in journal(audit)]
    assert events == ["ARM"], "a refused verb journals nothing"
    assert replay.reads == 0, "no scrape was served"


class FakeServer:
    def __init__(self) -> None:
        self.entered = self.exited = False
        self.polls: list[float] = []

    def __enter__(self) -> FakeServer:
        self.entered = True
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.exited = True

    def serve_forever(self, poll_interval: float) -> None:
        self.polls.append(poll_interval)
        raise KeyboardInterrupt


def test_serve_announces_the_endpoint_and_stops_on_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeServer()
    made: list[tuple[Any, ...]] = []

    def make_server(*arguments: Any) -> FakeServer:
        made.append(arguments)
        return fake

    monkeypatch.setattr(tool, "make_server", make_server)
    replay = tool.Replay(plan("pcie", (1, [])), 600)
    audit = io.StringIO()

    tool.serve("http://127.0.0.1:9400/metrics", 9412, replay, audit)

    assert made == [("http://127.0.0.1:9400/metrics", 9412, replay, audit)]
    assert (fake.entered, fake.exited, fake.polls) == (True, True, [0.5])
    assert capsys.readouterr().out.splitlines() == [
        "DCGM field replay http://127.0.0.1:9412/metrics (profile pcie); "
        "/status does not consume steps",
        "Pass-through until POST /arm with X-GFRS-Replay-Ack.",
    ]


def test_the_tool_entry_point_prints_its_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = ROOT / "tools" / "gpu_dcgm_field_replay.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")

    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert "Opt-in loopback DCGM *field* replay" in usage
    assert "--profile" in usage and "--serve" in usage, usage
