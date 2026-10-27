"""Starting the observation guard costs an interpreter, not the worker's imports.

Every summary, triage, bundle and 30-second passive sample on a node starts two
fresh interpreters: the subreaper guard (``observation_guard``) and, under it,
the read-only worker (``observation_worker``). The guard imported the worker
only to read its ``__name__`` for the ``python -m`` command line, and so loaded
the worker's whole closure -- pydantic, ``gpu_fault.models``, every operation
mixin -- that it never runs: 0.25 s of CPU idle per observation, and from spawn
to the worker's first line 0.54 s idle and 5.3 s with eight busy loops on its
CPU against 0.30 s and 3.3 s once the guard names the worker by string (b214,
measured on the release-gate host). On a GPU node a training job saturates the
CPU, and that start runs on the command's deadline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from gpu_fault.node_agent import observation_guard, observation_worker
from tests.node_agent.test_communication_monitor import use_imported_package

#: What the worker's closure pulls in and the guard never needs.
WORKER_ONLY_MODULES = frozenset(
    {
        observation_worker.__name__,
        "gpu_fault.models",
        "gpu_fault.node_agent.operations",
        "gpu_fault.node_agent.protocol",
        "pydantic",
    }
)


def test_starting_the_guard_does_not_import_the_worker(monkeypatch) -> None:
    """Beyond the package ``__init__`` files ``python -m`` loads first, the
    guard adds itself and standard-library modules, nothing of the worker's."""

    use_imported_package(monkeypatch)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "import gpu_fault.node_agent\n"
            "package = set(sys.modules)\n"
            f"import {observation_guard.__name__}\n"
            "print(json.dumps([sorted(package), sorted(set(sys.modules) - package)]))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ),
        timeout=120,
    )
    package, guard = (set(names) for names in json.loads(completed.stdout))

    assert (package | guard) & WORKER_ONLY_MODULES == set(), sorted(
        (package | guard) & WORKER_ONLY_MODULES
    )
    assert {name for name in guard if name.startswith("gpu_fault")} == {
        observation_guard.__name__
    }, sorted(guard)


def test_the_guard_still_starts_the_worker_module() -> None:
    """The string is the worker's real name and entry point: a rename of the
    module or of ``main`` fails here, not on a node."""

    module, _, attribute = observation_guard.WORKER_ENTRY_POINT.partition(":")
    assert module == observation_guard.WORKER_MODULE == observation_worker.__name__
    assert getattr(observation_worker, attribute) is observation_worker.main
