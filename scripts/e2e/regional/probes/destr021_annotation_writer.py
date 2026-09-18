#!/usr/bin/env python3
"""Bounded concurrent annotation writer for GF-REGIONAL-DESTR-021.

Runs on the operator host, not on the node: it keeps patching one harmless
annotation (``gpu-fault-acceptance.io/destr021-tick``) on the target node
every ``interval_seconds`` so the deployed executor's ``MARK_UNSCHEDULABLE`` and
``RESTORE_SCHEDULING`` patches race real ``409 Conflict`` responses and have to
go through ``patch_node_with_retry`` (ARCH-A2).

The writer is bounded twice: ``stop()`` from the runner's ``finally`` and a hard
``max_seconds`` after which the loop ends on its own, so a runner that dies
mid-case cannot leave a process hammering the API server. ``clear()`` nulls
the annotation; the node ends exactly as it started. The kubectl runner is
injectable so the loop is unit-testable without a cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import regional_commands  # noqa: E402

TICK_ANNOTATION = "gpu-fault-acceptance.io/destr021-tick"
DEFAULT_INTERVAL_SECONDS = 0.5
DEFAULT_MAX_SECONDS = 900.0
MINIMUM_INTERVAL_SECONDS = 0.2
MAXIMUM_MAX_SECONDS = 3600.0

Runner = Callable[[list[str]], int]


def subprocess_runner(command: list[str]) -> int:
    completed = regional_commands.run_fixture_command(command, check=False, timeout=60)
    return int(completed.returncode)


def patch_command(
    kubectl_prefix: list[str], node: str, value: str | None, *, node_uid: str = ""
) -> list[str]:
    """The merge patch that sets (or, with ``None``, removes) the tick."""

    metadata: dict[str, Any] = {"annotations": {TICK_ANNOTATION: value}}
    if node_uid:
        metadata["uid"] = node_uid
    body = {"metadata": metadata}
    return [
        *kubectl_prefix,
        "patch",
        "node",
        node,
        "--type=merge",
        "-p",
        json.dumps(body, sort_keys=True),
    ]


class AnnotationWriter:
    def __init__(
        self,
        kubectl_prefix: list[str],
        node: str,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        node_uid: str = "",
        runner: Runner = subprocess_runner,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not node:
            raise ValueError("annotation writer needs a node name")
        if not math.isfinite(interval_seconds) or (
            not MINIMUM_INTERVAL_SECONDS <= interval_seconds <= MAXIMUM_MAX_SECONDS
        ):
            raise ValueError(
                f"interval below {MINIMUM_INTERVAL_SECONDS}s would be an API flood"
            )
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= MAXIMUM_MAX_SECONDS:
            raise ValueError(f"max_seconds must be within (0, {MAXIMUM_MAX_SECONDS}]")
        self.kubectl_prefix = list(kubectl_prefix)
        self.node = node
        self.node_uid = node_uid
        self.interval_seconds = interval_seconds
        self.max_seconds = max_seconds
        self._runner = runner
        self._clock = clock
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop_active = False
        self.patches = 0
        self.conflicts = 0
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.exceeded_max_seconds = False
        self.cleared = False
        self.clear_returncode: int | None = None
        self.error: str | None = None

    # ------------------------------------------------------------------ loop
    def run_loop(self) -> None:
        """One patch per interval until stopped or the hard bound is hit."""

        self._loop_active = True
        self.started_at = self._clock()
        deadline = self.started_at + self.max_seconds
        tick = 0
        try:
            while not self._stop.is_set():
                if self._clock() >= deadline:
                    self.exceeded_max_seconds = True
                    break
                tick += 1
                returncode = self._runner(
                    patch_command(
                        self.kubectl_prefix,
                        self.node,
                        str(tick),
                        node_uid=self.node_uid,
                    )
                )
                if returncode == 0:
                    self.patches += 1
                else:
                    self.conflicts += 1
                if self._sleep is time.sleep:
                    self._stop.wait(self.interval_seconds)
                else:
                    self._sleep(self.interval_seconds)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.ended_at = self._clock()
            self._loop_active = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("annotation writer already started")
        self._thread = threading.Thread(
            target=self.run_loop, name="destr021-annotation-writer", daemon=True
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 90.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    @property
    def running(self) -> bool:
        return self._loop_active or (
            self._thread is not None and self._thread.is_alive()
        )

    # --------------------------------------------------------------- cleanup
    def clear(self) -> int:
        """Remove the tick annotation; idempotent, safe to call twice."""

        if self.running:
            raise RuntimeError("cannot clear the tick while its writer is running")
        self.clear_returncode = self._runner(
            patch_command(self.kubectl_prefix, self.node, None, node_uid=self.node_uid)
        )
        self.cleared = self.clear_returncode == 0
        return self.clear_returncode

    def report(self) -> dict[str, Any]:
        elapsed = (
            None
            if self.started_at is None
            else round(
                (
                    (self.ended_at if self.ended_at is not None else self._clock())
                    - self.started_at
                ),
                3,
            )
        )
        return {
            "node": self.node,
            "node_uid": self.node_uid,
            "annotation": TICK_ANNOTATION,
            "interval_seconds": self.interval_seconds,
            "max_seconds": self.max_seconds,
            "patches": self.patches,
            "conflicts": self.conflicts,
            "elapsed_seconds": elapsed,
            "stopped": not self.running and self.ended_at is not None,
            "running": self.running,
            "exceeded_max_seconds": self.exceeded_max_seconds,
            "cleared": self.cleared,
            "clear_returncode": self.clear_returncode,
            "error": self.error,
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Patch a harmless tick annotation on one node for a bounded time, "
            "then remove it. Implementation support for GF-REGIONAL-DESTR-021."
        )
    )
    value.add_argument("--kubeconfig", required=True)
    value.add_argument("--context", required=True)
    value.add_argument("--node", required=True)
    value.add_argument("--node-uid", required=True)
    value.add_argument("--seconds", type=float, required=True)
    value.add_argument(
        "--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS
    )
    return value


def main() -> int:
    arguments = parser().parse_args()
    writer = AnnotationWriter(
        [
            "kubectl",
            "--kubeconfig",
            arguments.kubeconfig,
            "--context",
            arguments.context,
        ],
        arguments.node,
        node_uid=arguments.node_uid,
        interval_seconds=arguments.interval_seconds,
        max_seconds=arguments.seconds,
    )
    writer.start()
    try:
        while writer.running:
            time.sleep(0.5)
    finally:
        writer.stop()
        writer.clear()
    print(json.dumps(writer.report(), sort_keys=True))
    return 0 if writer.cleared else 1


if __name__ == "__main__":
    sys.exit(main())
