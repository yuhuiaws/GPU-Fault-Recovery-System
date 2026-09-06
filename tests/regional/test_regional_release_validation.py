"""Reusing a deploy's own read-only verification instead of repeating it.

What `next_deploy` says about a deploy that finished but has not committed.

`rollout deploy` leaves the release at `phase=complete,
transaction_committed=False`: every component is applied and quick validation
passed, and the only step left is the commit the driver runs after verify and
the stability window. `next_deploy` correctly calls that resumable -- an
administrator who walked away here must be told to resume -- but "resume" alone
does not say *which* resume, so the release driver could not tell this state
apart from a half-applied transaction and threw away the read-only verifier
evidence the deploy had just produced. The last case covers what that evidence
then buys: the named checks are marked reused and their scripts do not run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ADMIN = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_admin_commands.py"
)
EVIDENCE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_validation_evidence.py"
)


def _release() -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(site_name="test-site"))


def _persisted_diff(module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the retry diff so these cases only exercise the labelling."""

    monkeypatch.setattr(
        module,
        "retry_release_diff",
        lambda _release, _state: module.diff_from_changed({"control_plane_wheel"}),
    )


def test_next_deploy_labels_a_completed_transaction_pending_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ADMIN.load()
    _persisted_diff(module, monkeypatch)

    result = module.next_deploy(
        _release(),
        {
            "phase": "complete",
            "transaction_committed": False,
            "release_id": "release-a",
            "release_diff": {
                "kind": "CONTROL_PLANE_ONLY",
                "changed": ["control_plane_wheel"],
            },
        },
    )

    assert result["action"] == "upgrade"
    assert result["resume"] is True, (
        "an uncommitted transaction is still resumable; only its reason is new"
    )
    assert result["pending_commit"] is True
    assert result["release_id"] == "release-a"


@pytest.mark.parametrize(
    ("phase", "committed"),
    [("cpu-finalized", False), ("data-plane-progress", False), ("rolled-back", None)],
)
def test_next_deploy_withholds_pending_commit_from_every_other_resume(
    monkeypatch: pytest.MonkeyPatch, phase: str, committed: bool | None
) -> None:
    """Only `complete` may claim it. A half-applied transaction may not.

    The release driver reuses the deploy's verifier evidence when it sees
    `pending_commit`, so any state where components are still unapplied has to
    stay unlabelled or the reuse would vouch for probes that never ran against
    the finished release.
    """

    module = ADMIN.load()
    _persisted_diff(module, monkeypatch)
    state: dict[str, object] = {"phase": phase, "release_id": "release-a"}
    if committed is not None:
        state["transaction_committed"] = committed

    result = module.next_deploy(_release(), state)

    assert "pending_commit" not in result
    assert result["action"] == "upgrade"


def test_next_deploy_withholds_pending_commit_from_a_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ADMIN.load()
    _persisted_diff(module, monkeypatch)

    result = module.next_deploy(
        _release(),
        {
            "phase": "rollback-data-progress",
            "transaction_committed": False,
            "release_id": "release-a",
        },
    )

    assert result["action"] == "rollback"
    assert "pending_commit" not in result


def test_reused_checks_replace_the_verifier_scripts_they_name() -> None:
    """The evidence names checks, and only those checks are skipped.

    This is the payoff of labelling a pending commit: the health report marks the
    control-plane role split as `reused` instead of spending ~26 s re-running the
    two read-only verifier scripts the deploy proved a minute earlier. A cluster
    the evidence does not name still gets its probe, so partial evidence buys
    exactly the part it covers.
    """

    module = EVIDENCE.load()
    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        commands.append(list(arguments))
        return {"healthy": True}

    release = SimpleNamespace(
        runner=SimpleNamespace(run=run),
        runtime_image="registry.example/runtime:v1",
        executor_wheel_cm="gpu-fault-executor-wheel",
        config=SimpleNamespace(
            cpu_kubeconfig="/secure/cpu.kubeconfig",
            namespace="gpu-fault-system",
            clusters=(
                SimpleNamespace(cluster_id="gpu-a", context="gpu-a"),
                SimpleNamespace(cluster_id="gpu-b", context="gpu-b"),
            ),
        ),
    )

    details = module.read_only_verifier_details(
        release, reused_checks={"control_plane_role_split", "data_plane_executor:gpu-a"}
    )

    assert details["cpu"] == {"reused": True}
    assert details["clusters"]["gpu-a"] == {"reused": True}
    assert details["clusters"]["gpu-b"] == {"healthy": True}, "gpu-b must be healthy"
    assert len(commands) == 1, "a reused check still ran its verifier script"
    assert commands[0][1].endswith("verify_dataplane_executor.py"), (
        "the data-plane verifier must still run when only the CPU check is reused"
    )
