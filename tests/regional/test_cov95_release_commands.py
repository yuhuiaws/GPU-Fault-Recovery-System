from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault_release import regional_admin_commands as commands
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import diff_from_changed
from tests.regional._cov95_release_support import (
    RecordingRunner,
    ResourceRelease,
    deployment,
    json_response,
)
from tests.regional._resource_probe_fakes import resource_probe_result


class ProbeRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__(json_response({}))
        self.present = True
        self.probes = []

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        self.probes.append(arguments)
        return resource_probe_result(arguments, present=self.present)


class CommandRelease(ResourceRelease):
    def __init__(self) -> None:
        super().__init__()
        self.runner = ProbeRunner()
        self.wheel_cm = "candidate-cpu-wheel"
        self.bundle_cm = "candidate-bundle"
        self.live_state = {
            "release_id": self.release_id,
            "phase": "complete",
            "transaction_committed": True,
        }
        self.actions: list[tuple[str, Any]] = []
        for name in inventory.CPU_DEPLOYMENTS:
            value = deployment(name, wheel=self.wheel_cm)
            value["metadata"]["generation"] = 3
            value["status"].update(
                observedGeneration=3, updatedReplicas=1, availableReplicas=1
            )
            self.documents[("cpu", "deployment", name)] = value

    def commit_release(self) -> None:
        self.actions.append(("commit", None))

    def upgrade(self, **kwargs: Any) -> None:
        self.actions.append(("upgrade", kwargs))

    def bootstrap(self) -> None:
        self.actions.append(("bootstrap", None))

    def pin_approved_manifest_plan(self, digest: str | None) -> None:
        self.actions.append(("pin", digest))


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"kind": "unknown", "changed": []},
        {"kind": "NOOP", "changed": None},
    ],
)
def test_untrusted_or_incomplete_stored_diff_is_not_a_plan(value: Any) -> None:
    assert commands.stored_release_diff({"release_diff": value}) is None


@pytest.mark.parametrize(
    "fault", [None, "metadata", "wheel", "generation", "ready", "typed-value", "read"]
)
def test_bootstrap_checkpoint_requires_matching_pins_generation_and_readiness(
    fault: str | None,
) -> None:
    release = CommandRelease()
    item = release.documents[("cpu", "deployment", inventory.CPU_INGRESS_DEPLOYMENT)]
    if fault == "metadata":
        release.metadata["required-agent-config-digest"] = "other"
    elif fault == "wheel":
        item["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = "other"
    elif fault == "generation":
        item["status"]["observedGeneration"] = 2
    elif fault == "ready":
        item["status"]["readyReplicas"] = 0
    elif fault == "typed-value":
        item["status"]["updatedReplicas"] = "unknown"
    elif fault == "read":
        release.documents[("cpu", "deployment", inventory.CPU_INGRESS_DEPLOYMENT)] = (
            ReleaseError("read unavailable")
        )
    assert commands.bootstrap_cpu_is_current(release) is (fault is None)
    assert release.actions == []


@pytest.mark.parametrize("resume", [False, True])
def test_completed_commit_cleanup_resumes_cleanup_not_rollout(
    monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    release = CommandRelease()
    release.live_state["commit_cleanup_completed"] = False
    monkeypatch.setattr(
        commands, "classify_release", lambda *_args: diff_from_changed(())
    )
    decision = commands.next_deploy(release, release.live_state)
    assert decision["action"] == "commit"
    assert decision["resume"] is True
    if resume:
        commands.run_resume(release)
    else:
        commands.run_deploy(release)
    assert release.actions == [("commit", None)]


@pytest.mark.parametrize("present", [False, True])
def test_expected_state_change_is_refused_before_deploy(
    monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    release = CommandRelease()
    release.runner.present = present
    monkeypatch.setenv(commands.EXPECTED_STATE_SHA256_ENV, "f" * 64)
    with pytest.raises(ReleaseError, match="disappeared|changed after"):
        commands.run_deploy(release)
    assert release.actions == []


def test_deploy_preflight_refuses_disappeared_or_uninitialized_empty_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CommandRelease()
    release.runner.present = False
    monkeypatch.setenv(commands.EXPECTED_STATE_SHA256_ENV, "f" * 64)
    with pytest.raises(ReleaseError, match="disappeared before deployment preflight"):
        commands.build_deploy_preflight_report(release)
    monkeypatch.delenv(commands.EXPECTED_STATE_SHA256_ENV)
    release.config.clusters = ()
    with pytest.raises(ReleaseError, match="at least one GPU cluster"):
        commands.build_deploy_preflight_report(release)
    assert release.actions == []


def test_schema_and_ca_commands_carry_only_cpu_scope_to_fake_runner() -> None:
    release = CommandRelease()
    commands.ensure_schema(release)
    commands.apply_rds_ca_bundle(release)
    assert len(release.runner.calls) == 2
    first, second = release.runner.calls
    assert first[1]["env"]["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] == "/dev/null"
    assert first[1]["env"]["GPU_FAULT_WHEEL_CONFIGMAP"] == release.wheel_cm
    assert first[1]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == release.runtime_image
    assert second[1]["env"]["KUBECONFIG"] == "/dev/null"


def test_release_diff_hashes_the_exact_read_state_and_noop_refuses_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CommandRelease()
    expected = diff_from_changed({"control_plane_wheel"})
    monkeypatch.setattr(commands, "classify_release", lambda *_args: expected)
    report = commands.build_release_diff(release)
    assert report == {
        "mode": "release-diff",
        "state_sha256": commands.release_state_sha256(release.live_state),
        "next_deploy": expected.as_dict(),
    }
    with pytest.raises(ReleaseError, match="stage-noop requires"):
        commands.stage_noop_release(release)
    assert release.actions == []


@pytest.mark.parametrize("monitoring", [None, {"adot": []}, {"adot": {}}])
def test_rollback_preflight_requires_monitoring_snapshot_before_selecting_repair(
    monkeypatch: pytest.MonkeyPatch, monitoring: Any
) -> None:
    release = CommandRelease()
    previous = {"observability": monitoring}
    release.live_state.update(
        phase="rollback-failed",
        previous=previous,
        previous_snapshot_sha256=canonical_sha256(previous),
        component_progress={
            "schema_version": 1,
            "global": {Component.OBSERVABILITY.value: {"status": "STARTED"}},
            "clusters": {},
        },
    )
    received = []

    def report(_release: Any, **kwargs: Any) -> dict[str, Any]:
        received.append(kwargs["repair_plan"])
        return {"checks": [], "healthy": True}

    monkeypatch.setattr(commands, "build_preflight_report", report)
    if monitoring != {"adot": {}}:
        with pytest.raises(ReleaseError, match="complete monitoring snapshot"):
            commands.build_deploy_preflight_report(release)
        assert received == []
    else:
        commands.build_deploy_preflight_report(release)
        assert received[0].has(Component.OBSERVABILITY), (
            "rollback preflight lost the started monitoring component"
        )
        assert not received[0].has(Component.SCHEMA), (
            "monitoring rollback must not authorize a schema change"
        )


def test_commit_cleanup_preflight_does_not_schedule_new_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CommandRelease()
    release.live_state["commit_cleanup_completed"] = False
    monkeypatch.setattr(
        commands, "classify_release", lambda *_args: diff_from_changed(())
    )
    received = []
    monkeypatch.setattr(
        commands,
        "build_preflight_report",
        lambda _release, **kwargs: received.append(kwargs) or {"healthy": True},
    )
    assert commands.build_deploy_preflight_report(release) == {"healthy": True}
    assert received == [{}]


def test_deployment_preflight_does_not_accept_an_invalid_next_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CommandRelease()
    monkeypatch.setattr(commands, "next_deploy", lambda *_args: {"action": "upgrade"})
    with pytest.raises(ReleaseError, match="no verified release plan"):
        commands.build_deploy_preflight_report(release)
    assert release.actions == []


def test_preflight_expected_state_digest_compares_canonical_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = CommandRelease()
    original = copy.deepcopy(release.live_state)
    monkeypatch.setenv(
        commands.EXPECTED_STATE_SHA256_ENV, commands.release_state_sha256(original)
    )
    monkeypatch.setattr(
        commands, "classify_release", lambda *_args: diff_from_changed(())
    )
    monkeypatch.setattr(
        commands, "build_preflight_report", lambda *_args, **_kwargs: {"healthy": True}
    )
    assert commands.build_deploy_preflight_report(release) == {"healthy": True}
    release.live_state["new-field"] = "drift"
    with pytest.raises(ReleaseError, match="state changed after"):
        commands.build_deploy_preflight_report(release)
