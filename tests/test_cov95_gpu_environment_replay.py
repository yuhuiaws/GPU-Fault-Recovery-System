"""GPU environment replay tool: plan bounds, safety scrape, HTTP routes and the CLI."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tools import gpu_environment_replay as replay_tool

RAW = (
    b'DCGM_FI_DEV_GPU_TEMP{UUID="GPU-a"} 40\n'
    b'DCGM_FI_DEV_CLOCKS_EVENT_REASONS{UUID="GPU-a"} 0\n'
)
STEP = {"reads": 3, "values": {"gpu_temperature_c": 95}}


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ([STEP], "plan must contain only steps"),
        ({"steps": [STEP], "notes": "x"}, "plan must contain only steps"),
        ({"steps": STEP}, "plan requires 1..100 steps"),
        ({"steps": []}, "plan requires 1..100 steps"),
        ({"steps": [STEP] * 101}, "plan requires 1..100 steps"),
        ({"steps": ["three reads"]}, "each step requires reads and values"),
        ({"steps": [{"reads": 3}]}, "each step requires reads and values"),
        (
            {"steps": [{"reads": 100, "values": {}}] * 4},
            "plan exceeds 300 collector reads",
        ),
    ],
)
def test_plan_bounds_name_the_violated_rule(plan: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replay_tool.validate_plan(plan)


def test_a_plan_of_exactly_three_hundred_reads_is_accepted() -> None:
    steps = [{"reads": 100, "values": {}}] * 3

    assert replay_tool.validate_plan({"steps": steps}) == steps


def test_the_safety_scrape_ignores_samples_that_are_not_dcgm_fields() -> None:
    body = b"node_cpu_seconds_total 5\n" + RAW

    assert replay_tool.unsafe_upstream(body, 80, 85) is None
    hot = b"node_cpu_seconds_total 5\n" + RAW.replace(b"} 40", b"} 81")
    assert replay_tool.unsafe_upstream(hot, 80, 85) == (
        "live safety stop: DCGM_FI_DEV_GPU_TEMP"
    )


def test_an_oversized_upstream_response_is_refused_before_parsing() -> None:
    replay = replay_tool.Replay("GPU-a", [STEP], 600)

    with pytest.raises(ValueError, match="upstream response exceeds byte bound"):
        replay.render(b"#" * (replay_tool.MAX_BYTES + 1))

    assert replay.reads == 0, "a refused body consumes no read"


def test_the_endpoint_answers_404_off_its_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_reads: list[Any] = []
    monkeypatch.setattr(
        replay_tool,
        "open_upstream",
        lambda *args, **kwargs: upstream_reads.append(args) or io.BytesIO(RAW),
    )
    replay = replay_tool.Replay("GPU-a", [STEP], 600)
    server = replay_tool.make_server(
        "http://127.0.0.1:9400/metrics", 0, replay, io.StringIO()
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as missing:
            urlopen(f"http://127.0.0.1:{server.server_address[1]}/other", timeout=5)
        assert missing.value.code == 404
        assert missing.value.read() == b"not found\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)

    assert upstream_reads == [], "an unknown route never reads the exporter"
    assert replay.reads == 0


class StoppedServer:
    """``make_server``: serves until the operator's Ctrl-C."""

    def __init__(self) -> None:
        self.polls: list[float] = []

    def __enter__(self) -> StoppedServer:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def serve_forever(self, *, poll_interval: float) -> None:
        self.polls.append(poll_interval)
        raise KeyboardInterrupt


def test_serve_announces_the_routes_and_stops_quietly_on_ctrl_c(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = StoppedServer()
    made: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        replay_tool, "make_server", lambda *args: made.append(args) or server
    )
    replay = replay_tool.Replay("GPU-a", [STEP], 600)
    audit = io.StringIO()

    replay_tool.serve("http://127.0.0.1:9400/metrics", 9411, replay, audit)

    assert made == [("http://127.0.0.1:9400/metrics", 9411, replay, audit)]
    assert server.polls == [0.5]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "Replay endpoint http://127.0.0.1:9411/metrics; /status does not consume steps"
    )
    assert len(lines) == 3


def plan_file(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"steps": [STEP]}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--port", "0"], "invalid port or max-seconds (1..1800)"),
        (["--max-seconds", "1801"], "invalid port or max-seconds (1..1800)"),
        (["--gpu-uuid", "MIG-a"], "a physical GPU UUID is required"),
        (["--gpu-stop-c", "81"], "safety limits may only be lowered"),
        (["--memory-stop-c", "0"], "safety limits may only be lowered"),
    ],
)
def test_the_cli_refuses_unsafe_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as stop:
        replay_tool.main(
            ["--plan", str(plan_file(tmp_path)), "--gpu-uuid", "GPU-a", *extra]
        )

    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_the_cli_reports_an_unreadable_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        replay_tool.main(
            ["--plan", str(tmp_path / "absent.json"), "--gpu-uuid", "GPU-a"]
        )

    assert stop.value.code == 2
    assert "absent.json" in capsys.readouterr().err


def test_an_acknowledged_serve_opens_a_new_audit_and_serves_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[tuple[Any, ...]] = []

    def serve(upstream: str, port: int, replay: Any, audit: Any) -> None:
        served.append((upstream, port, replay.gpu_uuid, replay.steps, audit.name))
        audit.write("{}\n")

    monkeypatch.setattr(replay_tool, "serve", serve)
    audit = tmp_path / "audit.jsonl"
    arguments = [
        "--plan",
        str(plan_file(tmp_path)),
        "--gpu-uuid",
        "GPU-a",
        "--serve",
        "--ack",
        replay_tool.ACKNOWLEDGEMENT,
        "--audit",
        str(audit),
    ]

    assert replay_tool.main(arguments) == 0

    assert served == [
        ("http://127.0.0.1:9400/metrics", 9411, "GPU-a", [STEP], str(audit))
    ]
    assert audit.read_text(encoding="utf-8") == "{}\n"
    # The audit of an earlier run is never appended to.
    with pytest.raises(FileExistsError):
        replay_tool.main(arguments)
    assert len(served) == 1
