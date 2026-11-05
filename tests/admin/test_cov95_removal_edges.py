"""``remove-cluster`` barriers that only a failure between two steps reaches.

Driven through the real removal pipeline over the in-memory transports of
``RemovalTransport``: a retried discovery reuses the registry snapshot the
failed attempt exported; a cleanup attempt that did not complete is rerun in
place (and its evidence named when the rerun fails); cleanup evidence that is
not an object, a CPU node-key Secret that vanished after the unregister, a
bootstrap state of another site, a namespace or a registry member that came
back before the final verification and an incomplete durable registry
identity each stop the removal with their own reason and leave it resumable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin._cov95_removal_support import RemovalTransport

STATE_DIR = "remove-cluster/gpu-a"


class ScriptedRemoval(RemovalTransport):
    """``RemovalTransport`` plus one scripted change at a chosen boundary.

    ``after[mode]`` runs once right after the lifecycle driver ``mode``
    answered; ``cleanup_document`` is what the cleanup tool writes instead of
    its evidence; ``cpu_keys_gone`` makes the CPU node-key Secret read empty.
    """

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.after: dict[str, Callable[[], None]] = {}
        self.cleanup_document: str | None = None
        self.cpu_keys_gone = False
        self.exported: list[Path | None] = []
        super().__init__(directory, monkeypatch)

    def fetch(self, site, *, output=None):
        self.exported.append(output)
        return super().fetch(site, output=output)

    def driver(self, arguments, **options):
        cleanup = Path(arguments[0]).name == "prepare-clean-redeploy.sh"
        mode = "cleanup" if cleanup else arguments[1]
        if cleanup and self.cleanup_document is not None and self.failure != mode:
            self.events.append(mode)
            state_file = Path(arguments[arguments.index("--state-file") + 1])
            state_file.write_text(self.cleanup_document, encoding="utf-8")
            return self.result(arguments, {})
        result = super().driver(arguments, **options)
        hook = self.after.pop(mode, None)
        if hook is not None:
            hook()
        return result

    def command(self, arguments, **options):
        arguments = list(arguments)
        if (
            self.cpu_keys_gone
            and "get" in arguments
            and "gpu-fault-node-action-keys" in arguments
            and "--context" not in arguments
        ):
            return self.result(arguments, "")
        return super().command(arguments, **options)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ScriptedRemoval:
    return ScriptedRemoval(tmp_path, monkeypatch)


def completed_steps(world: ScriptedRemoval) -> list[str]:
    return list(world.state()["completed_steps"])


def attempt_dir(world: ScriptedRemoval) -> Path:
    """The evidence directory of the removal attempt the journal resumes."""

    return world.directory / STATE_DIR / "attempts" / world.state()["attempt_id"]


def test_a_registry_exported_for_another_site_is_refused(
    world: ScriptedRemoval,
) -> None:
    request = world.request()
    foreign = InstallationResourceSnapshot(
        site_id="another-site",
        resources=[
            item.model_copy(update={"site_id": "another-site"})
            for item in world.snapshot.resources
        ],
    )
    foreign = foreign.model_copy(update={"source_sha256": foreign.digest()})

    with pytest.raises(BootstrapError, match="registry belongs to another site"):
        removal.validate_target_registry(
            request,
            request.site.release_config["clusters"][0],
            foreign,
            {"hyperpod_arn": world.hyperpod},
        )

    removal.validate_target_registry(
        request,
        request.site.release_config["clusters"][0],
        world.snapshot,
        {"hyperpod_arn": world.hyperpod},
    )


def test_a_retried_discovery_reuses_the_registry_snapshot_it_exported(
    world: ScriptedRemoval,
) -> None:
    world.endpoint_override = "https://foreign.example.invalid"
    with pytest.raises(BootstrapError, match="context does not match"):
        world.remove()
    snapshot = attempt_dir(world) / "installation-resources-before.json"
    assert world.exported == [snapshot], "the first attempt did not export once"
    assert snapshot.is_file(), "the failed attempt kept no registry snapshot"

    world.endpoint_override = None
    assert world.remove()["phase"] == "COMPLETED"

    assert snapshot not in world.exported[1:], (
        "the retry exported a newer registry instead of the snapshot it bound"
    )


def test_an_unfinished_cleanup_attempt_is_rerun_in_place(
    world: ScriptedRemoval,
) -> None:
    world.cleanup_override = {"status": "FAILED", "phase": "CLEANUP_FAILED"}
    with pytest.raises(BootstrapError, match="no completed, bound evidence"):
        world.remove()
    attempt = attempt_dir(world) / "kubernetes-cleanup-001.json"
    assert json.loads(attempt.read_text())["status"] == "FAILED"

    world.cleanup_override = None
    world.failure = "cleanup"
    with pytest.raises(
        BootstrapError,
        match="cleanup failed; evidence retained in " + re.escape(str(attempt)),
    ):
        world.remove()
    assert "delete-namespace" not in world.events

    world.failure = None
    assert world.remove()["phase"] == "COMPLETED"
    assert sorted(path.name for path in attempt.parent.glob("kubernetes-*")) == [
        attempt.name
    ], "an unfinished attempt must be rerun in place, not beside a new one"
    assert json.loads(attempt.read_text())["status"] == "COMPLETED"
    assert world.events.count("cleanup") == 3


@pytest.mark.parametrize("document", ["[]", '"done"', "7"])
def test_cleanup_evidence_that_is_not_an_object_never_deletes_the_namespace(
    world: ScriptedRemoval, document: str
) -> None:
    world.cleanup_document = document

    with pytest.raises(BootstrapError, match="^invalid GPU cleanup evidence$"):
        world.remove()

    assert "delete-namespace" not in world.events
    assert "KUBERNETES_QUIESCED" not in completed_steps(world)
    assert world.namespace, "the target namespace was touched without evidence"


def test_a_cpu_node_key_secret_gone_after_unregister_keeps_the_step_open(
    world: ScriptedRemoval,
) -> None:
    world.after["remove-cluster"] = lambda: setattr(world, "cpu_keys_gone", True)

    with pytest.raises(
        BootstrapError,
        match="CONTROL_REGISTRY_REMOVED: CPU node-key Secret identity is unavailable",
    ):
        world.remove()

    assert "CONTROL_REGISTRY_REMOVED" not in completed_steps(world)
    assert "remove-keys" not in world.events
    world.cpu_keys_gone = False
    assert world.remove()["phase"] == "COMPLETED"
    assert world.keys == {}, "the resumed removal left the target's node keys"


def test_a_foreign_bootstrap_state_stops_the_site_commit(
    world: ScriptedRemoval,
) -> None:
    bootstrap = world.directory / "bootstrap-state.json"
    foreign: dict[str, Any] = {
        "site_id": "another-site",
        "completed_tasks": ["executor_role:gpu-a"],
        "resources": {"executor_role:gpu-a": {}},
    }
    bootstrap.write_text(json.dumps(foreign), encoding="utf-8")
    token = Path(world.gpu["token_file"])

    with pytest.raises(BootstrapError, match="bootstrap state belongs to another site"):
        world.remove()

    steps = completed_steps(world)
    assert "SITE_COMMIT_STARTED" in steps and "SITE_UPDATED" not in steps, steps
    assert json.loads(bootstrap.read_text()) == foreign, "another site's state changed"
    assert token.is_file(), "the credential went before the site commit finished"

    # The resumed commit finds the credential already gone and finishes anyway.
    bootstrap.unlink()
    token.unlink()
    assert world.remove()["phase"] == "COMPLETED"
    assert "SITE_UPDATED" in completed_steps(world)
    assert not token.exists(), "the resumed commit recreated the credential"


def test_a_namespace_back_before_verification_fails_the_final_check(
    world: ScriptedRemoval,
) -> None:
    world.after["sync-state"] = lambda: setattr(world, "namespace", True)

    with pytest.raises(
        BootstrapError,
        match="final verification failed: .*target GPU solution namespace still exists",
    ):
        world.remove()

    assert "VERIFIED" not in completed_steps(world)
    assert world.state()["phase"] != "COMPLETED"


def test_a_registry_member_back_before_verification_fails_the_final_check(
    world: ScriptedRemoval,
) -> None:
    def regress() -> None:
        world.runtime_override = {
            "generation": world.generation + 1,
            "content_sha256": "b" * 64,
            "cluster_states": {"gpu-z": "ACTIVE"},
            "required_member_ids": [],
            "acked_member_ids": [],
            "missing_member_ids": [],
            "active_member_ids": [],
            "members": [],
            "converged": True,
        }

    world.after["sync-state"] = regress

    with pytest.raises(
        BootstrapError,
        match="control registry: durable regional registry removal has not converged",
    ):
        world.remove()

    assert "VERIFIED" not in completed_steps(world)


@pytest.mark.parametrize(
    "change",
    [
        {"registry_generation": 0},
        {"registry_generation": True},
        {"registry_cluster_states": ["gpu-a"]},
        {"registry_cluster_states": {"gpu-a": 1}},
        {"registry_content_sha256": "A" * 64},
        {"live_release_identity_sha256": "b" * 63},
    ],
    ids=[
        "generation-zero",
        "generation-bool",
        "states-not-mapping",
        "state-not-text",
        "content-not-lower-hex",
        "release-identity-short",
    ],
)
def test_an_incomplete_durable_registry_identity_stops_discovery(
    world: ScriptedRemoval, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    runtime = {
        "registry_cluster_states": {"gpu-a": "ACTIVE"},
        "registry_generation": 1,
        "registry_content_sha256": "a" * 64,
        "live_release_identity_sha256": "b" * 64,
        **change,
    }
    monkeypatch.setattr(
        removal, "membership_runtime_snapshot", lambda _site: dict(runtime)
    )

    with pytest.raises(BootstrapError, match="registry identity is incomplete"):
        world.remove()

    assert "drain-cluster" not in world.events
    assert "cleanup" not in world.events
