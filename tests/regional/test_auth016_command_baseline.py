"""AUTH-016's seeded baseline command: opened before the baseline probe, never
executable, purged after the verdict has read it."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional import auth016_command_baseline as baseline
from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional import seeded_command_fixture as seeded

_SEED = {
    "command_id": "remote-auth016-1",
    "workflow_id": "workflow-actionperf-auth016-1",
    "incident_id": "incident-auth016-1",
    "event_id": "event-auth016-1",
    "registered_agents": [],
}


class ControlPlane:
    """Plays the control-plane Pod: records each script run and answers the
    seed and the purge the way the fixture's scripts do."""

    def __init__(self, *, agents: list[str] = (), purge_fails: bool = False):
        self.scripts: list[tuple[str, tuple[str, ...]]] = []
        self.agents = list(agents)
        self.purge_fails = purge_fails

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        self.scripts.append((script, arguments))
        if "store.ensure_remote_command(command)" in script:
            return {**_SEED, "registered_agents": self.agents}
        if self.purge_fails:
            raise RuntimeError("purge exploded")
        return {"deleted": {}, "remaining": [], "remaining_links": 0}

    def operations(self) -> list[str]:
        return [
            "seed" if "ensure_remote_command" in s else "purge" for s, _ in self.scripts
        ]


def test_the_seed_is_a_valid_operation_on_a_node_no_cluster_carries() -> None:
    assert WorkflowOperation(baseline.OPERATION) is not None, "not a WorkflowOperation"
    assert baseline.LEASE_SECONDS >= seeded.SEED_LEASE_SECONDS, (
        "a lease shorter than the fixture's own lets the dispatcher claim the seed"
    )
    for node_id in baseline.NODE_IDS:
        assert not re.match(r"(hyperpod-)?i-[0-9a-f]{8,}", node_id), node_id


def test_the_seed_is_run_through_the_given_control_plane_and_purged_on_exit(
    tmp_path: Path,
) -> None:
    plane = ControlPlane()

    with baseline.SeededBaseline.open(plane, tmp_path) as seed:
        assert seed.command_id == "remote-auth016-1"
        assert plane.operations() == ["seed"], "the seed was purged before the body"
        seed_arguments = plane.scripts[0][1]
        assert seed_arguments[1] == seeded.SYNTHETIC_CLUSTER_ID
        assert seed_arguments[2] == baseline.OWNER
        assert seed_arguments[3] == baseline.OPERATION
        assert seed_arguments[5] == str(baseline.LEASE_SECONDS)

    assert plane.operations() == ["seed", "purge"]
    assert json.loads((tmp_path / baseline.SEED_FILE).read_text())["command_id"] == (
        "remote-auth016-1"
    )
    assert (tmp_path / baseline.PURGE_FILE).exists(), (
        "the purge result was not recorded"
    )


def test_a_synthetic_cluster_with_agents_is_refused_and_the_seed_purged(
    tmp_path: Path,
) -> None:
    """A Node Agent registered under perf-cap-000 could act on the command; the
    fixture's hard stop is that none ever is."""

    plane = ControlPlane(agents=["some-node"])

    with pytest.raises(seeded.SeededCommandError, match="registered agents"):
        baseline.SeededBaseline.open(plane, tmp_path)

    assert plane.operations() == ["seed", "purge"]


def test_a_purge_failure_does_not_hide_the_case_failure(tmp_path: Path) -> None:
    plane = ControlPlane(purge_fails=True)

    with pytest.raises(ValueError, match="the case itself"):
        with baseline.SeededBaseline.open(plane, tmp_path):
            raise ValueError("the case itself")

    assert "purge exploded" in (tmp_path / baseline.PURGE_FILE).read_text()

    with pytest.raises(RuntimeError, match="purge exploded"):
        with baseline.SeededBaseline.open(plane, tmp_path):
            pass


def test_run_auth016_opens_the_seed_before_the_baseline_probe_and_verdict() -> None:
    """The seed must exist when REMOTE_STATUS_PROBE takes the baseline and still
    when the verdict re-reads it, so both sit inside the ``with``."""

    source = Path(auth.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_auth016(") : source.index("TLS_BOUNDARY_PROBE")]
    opened = body.index("with SeededBaseline.open(primary.cpu_python, case_dir):")
    assert opened < body.index(
        "before_commands = primary.cpu_python(REMOTE_STATUS_PROBE)"
    )
    verdict_line = next(
        line for line in body.splitlines() if "return auth016_result(" in line
    )
    assert verdict_line.startswith("        return"), (
        "the verdict is computed outside the with, after the seed was purged"
    )
