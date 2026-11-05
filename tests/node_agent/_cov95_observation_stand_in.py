"""A stand-in observation worker whose misbehaviour the request picks.

Spawned by the real ``ObservationSupervisor`` in place of ``observation_guard``
(``-m tests.node_agent._cov95_observation_stand_in``); ``request["stand_in"]``
picks what it does:

* ``oversize``: writes ``request["size"]`` bytes without a line end, more than
  the supervisor's result budget, and exits;
* ``not_a_result``: writes one JSON line that is no ``{"result": {...}}`` and
  exits;
* ``deaf``: ignores SIGTERM -- the supervisor's ``terminate`` and a
  preemption's ``killpg`` alike, the way a tree stuck in kernel I/O does not
  act on them -- writes its pid to ``request["started"]`` once it does, and
  then waits for a signal it cannot ignore (the test's SIGKILL).

It imports nothing from the package, so it starts in the time an interpreter
takes and needs no ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path


def main() -> None:
    request = json.loads(sys.stdin.buffer.read() or b"{}")
    mode = request.get("stand_in")
    output = sys.stdout.buffer
    if mode == "oversize":
        output.write(b"x" * int(request["size"]))
        output.flush()
    elif mode == "not_a_result":
        output.write(json.dumps({"final": True, "ranks": []}).encode() + b"\n")
        output.flush()
    elif mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(request["started"]).write_text(str(os.getpid()))
        while True:
            signal.pause()
    else:
        raise SystemExit(f"unknown stand-in mode {mode!r}")


if __name__ == "__main__":
    main()
