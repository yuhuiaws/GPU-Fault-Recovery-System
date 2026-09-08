"""The cluster-executor env window survives a replica rolling away mid-survey."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional.acceptance_runner_common import replica_vanished

ATTEMPTS, RECOVERY = env_window.ALLOWED_VARIABLES


class _RollingRegional:
    def __init__(self, vanished: str, values: dict[str, str], *, stderr: str) -> None:
        self._vanished = vanished
        self._values = values
        self._stderr = stderr
        self.exec_targets: list[str] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": self._vanished}, {"name": "survivor"}]

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        assert arguments[0] == "exec", arguments
        target = arguments[1]
        self.exec_targets.append(target)
        if target == self._vanished:
            raise env_window.RegionalFixtureError(
                f"command failed (1): kubectl ... exec {target} ...; stderr={self._stderr}"
            )
        return json.dumps(self._values)


@pytest.mark.parametrize(
    "stderr",
    [
        'Error from server (NotFound): pods "gone" not found',
        "error: cannot exec into a container in a completed pod; current phase is Succeeded",
    ],
)
def test_replica_values_skip_a_replica_that_rolled_away(stderr: str) -> None:
    """DESTR-014 attempt 3 (2026-09-08) died inside the open window's converge
    on exactly this NotFound and left the executor running the compressed
    timing set until the window was closed by hand."""
    values = {ATTEMPTS: "6", RECOVERY: "300"}
    regional = _RollingRegional("gone-68rp8", values, stderr=stderr)

    replicas = env_window.replica_values(regional)

    assert regional.exec_targets == ["gone-68rp8", "survivor"], regional.exec_targets
    assert replicas == [{"pod": "survivor", "values": values}], replicas


def test_a_real_exec_failure_still_raises() -> None:
    regional = _RollingRegional(
        "a", {ATTEMPTS: "6", RECOVERY: "300"}, stderr="OCI runtime exec failed"
    )
    with pytest.raises(env_window.RegionalFixtureError, match="OCI runtime"):
        env_window.replica_values(regional)


def test_the_shared_marker_list_names_every_kubelet_wording() -> None:
    for text in (
        'pods "x" not found',
        "cannot exec into a container in a completed pod; current phase is Succeeded",
        'unable to upgrade connection: container not found ("executor")',
        "container is not running",
    ):
        assert replica_vanished(RuntimeError(text)), text
    assert not replica_vanished(RuntimeError("permission denied")), "a real error"
