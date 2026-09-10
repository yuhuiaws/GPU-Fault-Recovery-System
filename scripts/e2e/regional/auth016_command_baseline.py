"""One remote command held open through AUTH-016's token rotation.

``commands_not_misterminated`` refuses an empty baseline, so on an idle site the
case could only fail: there was no command whose survival it could prove. The
fixture NET-002/NET-006/CMD-017 share seeds one for the synthetic cluster
``perf-cap-000`` on a node id no cluster carries, under a workflow lease held by
a seed identity nothing claims, so it sits open (never executed) through the
rotation, both rollouts and the restore. It is purged when the ``with`` block
exits -- after the verdict has read its status -- and on any failure inside.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.e2e.regional import seeded_command_fixture as seeded

OWNER = "auth016-baseline"
# Never executed: the seeded node exists in no cluster and the lease belongs to
# a seed identity, so the operation only has to be a valid, node-scoped one.
OPERATION = "TRIGGER_HEALTH_SNAPSHOT"
NODE_IDS = ["auth016-synthetic-node"]
# The rotation waits about three minutes, the rollouts and the restore a few
# more; the lease must outlive all of it or the dispatcher claims the workflow.
LEASE_SECONDS = 45 * 60
SEED_FILE = "seeded-baseline-command.json"
PURGE_FILE = "seeded-baseline-purge.json"

CpuPython = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SeededBaseline:
    """The seeded command as a context manager: open before the baseline
    probe, purge after the verdict."""

    cpu_python: CpuPython
    seed: dict[str, Any]
    case_dir: Path

    @classmethod
    def open(cls, cpu_python: CpuPython, case_dir: Path) -> SeededBaseline:
        run_id = f"auth016-{secrets.token_hex(4)}"
        seed = seeded.seed_command(
            run_id,
            owner=OWNER,
            operation=OPERATION,
            node_ids=NODE_IDS,
            lease_seconds=LEASE_SECONDS,
            run=cpu_python,
        )
        seeded.write_json(case_dir / SEED_FILE, seed)
        baseline = cls(cpu_python, seed, case_dir)
        if seed.get("registered_agents"):
            # A Node Agent registered under the synthetic cluster could act on
            # the command; the fixture's hard stop is that none ever is.
            baseline.purge()
            raise seeded.SeededCommandError(
                f"synthetic cluster has registered agents: {seed['registered_agents']}"
            )
        return baseline

    @property
    def command_id(self) -> str:
        return str(self.seed["command_id"])

    def purge(self) -> dict[str, Any]:
        purged = seeded.purge_seed(self.seed, run=self.cpu_python)
        seeded.write_json(self.case_dir / PURGE_FILE, purged)
        return purged

    def __enter__(self) -> SeededBaseline:
        return self

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        try:
            self.purge()
        except Exception as exc:
            # Residue is a failure in its own right, but not one that may hide
            # the case's own exception: record it and let the original out.
            if exc_type is None:
                raise
            seeded.write_json(self.case_dir / PURGE_FILE, {"error": str(exc)})
