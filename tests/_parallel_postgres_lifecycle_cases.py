"""Small explicit pytest workload for the eight-worker lifetime regression."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic


@pytest.mark.parametrize("number", range(40))
def test_lifetime(number: int) -> None:
    del number
    state = Path(os.environ["PARALLEL_POSTGRES_STATE"])
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    write_json_atomic(
        state / "test-running.json", {"pytest": os.getpid(), "child": child.pid}
    )
    until = time.monotonic() + 15
    for index in range(8):
        while not (state.parent / f"shard-{index}/test-running.json").exists():
            assert time.monotonic() < until, "eight test processes did not start"
            time.sleep(0.02)
    if state.name == "shard-3":
        if os.environ["PARALLEL_POSTGRES_MODE"] == "failure":
            pytest.fail("example private failure details")
        if os.environ["PARALLEL_POSTGRES_MODE"] == "skip":
            pytest.skip("example private skip details")
    time.sleep(30)
