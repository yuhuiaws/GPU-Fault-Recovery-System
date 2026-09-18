from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/node/bounded-preflight.sh"


def run_preflight(
    tmp_path: Path, callback: str, *, limit: str = "2"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'source "$1"\n'
            + callback
            + '\nrun_bounded_preflight host "$LIMIT" "$OUTPUT" task 0 1 2 3 4 5 6 7 8\n',
            "preflight",
            str(SCRIPT),
        ],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "OUTPUT": str(tmp_path),
            "LIMIT": limit,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_slow_first_node_does_not_block_the_sliding_window(tmp_path: Path) -> None:
    result = run_preflight(
        tmp_path,
        """
task() {
    local phase="$1" node="$2"
    printf 'start:%s\\n' "$node" >>"$OUTPUT/events"
    if [[ "$node" == 0 ]]; then
        for _ in {1..100}; do
            [[ -f "$OUTPUT/last-finished" ]] && break
            sleep 0.02
        done
        [[ -f "$OUTPUT/last-finished" ]]
    elif [[ "$node" == 8 ]]; then
        touch "$OUTPUT/last-finished"
    fi
    printf 'done:%s\\n' "$node" >>"$OUTPUT/events"
    printf 'output:%s\\n' "$node"
}
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = (tmp_path / "events").read_text().splitlines()
    assert events.index("done:8") < events.index("done:0")
    active = 0
    for event in events:
        active += 1 if event.startswith("start:") else -1
        assert 0 <= active <= 2
    assert active == 0
    assert result.stdout.splitlines() == [f"output:{node}" for node in range(9)]


def test_failure_stops_new_nodes_and_joins_started_children(tmp_path: Path) -> None:
    result = run_preflight(
        tmp_path,
        """
task() {
    local phase="$1" node="$2"
    printf '%s\\n' "$node" >>"$OUTPUT/started"
    if [[ "$node" == 1 ]]; then
        false
        touch "$OUTPUT/must-not-run"
    fi
    sleep 0.2
    touch "$OUTPUT/joined"
}
""",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert sorted((tmp_path / "started").read_text().splitlines()) == ["0", "1"]
    assert (tmp_path / "joined").exists(), (
        "preflight returned before its started child finished"
    )
    assert not (tmp_path / "must-not-run").exists(), "callback lost errexit"
    assert "failed on 1" in result.stderr


def test_killed_worker_cannot_leave_the_completion_wait_blocked(tmp_path: Path) -> None:
    result = run_preflight(
        tmp_path,
        """
task() {
    printf '%s\\n' "$2" >>"$OUTPUT/started"
    if [[ "$2" == 1 ]]; then
        kill -KILL "$BASHPID"
    fi
    sleep 0.2
    touch "$OUTPUT/joined"
}
""",
    )
    assert result.returncode == 1, result.stderr
    assert (tmp_path / "joined").exists(), "the surviving worker was not joined"
    assert "exit 137" in result.stderr
    assert sorted((tmp_path / "started").read_text().splitlines()) == ["0", "1"]


@pytest.mark.parametrize("limit", ["0", "-1", "9", "invalid"])
def test_invalid_budget_cannot_silently_skip_preflight(
    tmp_path: Path, limit: str
) -> None:
    result = run_preflight(
        tmp_path, 'task() { touch "$OUTPUT/must-not-run"; }', limit=limit
    )
    assert result.returncode == 2
    assert not (tmp_path / "must-not-run").exists(), (
        "invalid scheduling entered a node callback"
    )
