"""Bootstrap monitoring must not expect collectors that failed to bootstrap."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_dataplane_observability as OBSERVABILITY
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_bootstrap import bootstrap_observability
from tests.regional._release_orchestrator_support import config_file

WAIT_SECONDS = 5


def test_background_supervision_loss_dominates_an_ordinary_gpu_failure() -> None:
    class SupervisionLost(BaseException):
        pass

    joined = threading.Event()
    primary = RuntimeError("GPU bootstrap failed")
    fatal = SupervisionLost("background ownership is unproved")

    def configure() -> None:
        joined.set()
        raise fatal

    with pytest.raises(SupervisionLost) as captured:
        with bootstrap_observability(configure):
            raise primary

    assert joined.is_set(), "the background branch was not joined"
    assert captured.value is fatal, (
        "background ownership loss was replaced by an ordinary error"
    )
    assert captured.value.__cause__ is primary, "the foreground failure was lost"


def test_ordinary_background_failure_keeps_the_primary_error_and_diagnostic_note() -> (
    None
):
    primary = RuntimeError("GPU bootstrap failed")

    def configure() -> None:
        raise RuntimeError("monitoring failed")

    with pytest.raises(RuntimeError) as captured:
        with bootstrap_observability(configure):
            raise primary

    assert captured.value is primary, (
        "an ordinary secondary failure replaced the primary"
    )
    assert any("monitoring failed" in note for note in primary.__notes__), (
        "the joined background failure lost its diagnostic"
    )


class MonitoringRunner(ROLLOUT.Runner):
    """Model the installer's arguments and AMP writes without running binaries."""

    def __init__(self) -> None:
        super().__init__()
        self.expected_rules: bytes | None = None
        self.status = "ACTIVE"
        self.installs: list[list[str]] = []
        self.mutations: list[str] = []
        self.reads: list[list[str]] = []
        self.applies: list[tuple[list[str], str]] = []
        self.configure = lambda: None
        self.on_mutation = lambda: None

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        if arguments[:2] == ["bash", str(OBSERVABILITY.AMP_MONITORING_INSTALLER)]:
            self.installs.append(list(arguments))
            if "--dataplane-expected-rules" in arguments:
                path = Path(
                    arguments[arguments.index("--dataplane-expected-rules") + 1]
                )
                self.expected_rules = path.read_bytes()
                self.mutations.append("installer-put")
                self.on_mutation()
            elif "--no-dataplane-expected-rules" in arguments:
                self.expected_rules = None
                self.mutations.append("installer-delete")
                self.on_mutation()
            self.configure()
            return ""
        if arguments[:2] == ["aws", "amp"]:
            assert arguments[arguments.index("--name") + 1] == (
                OBSERVABILITY.DATAPLANE_EXPECTED_RULE_NAMESPACE
            )
            assert arguments[arguments.index("--region") + 1] == "us-east-1"
            assert arguments[arguments.index("--workspace-id") + 1] == "ws-test"
            if arguments[2] == "delete-rule-groups-namespace":
                self.expected_rules = None
            else:
                assert arguments[2] in {
                    "create-rule-groups-namespace",
                    "put-rule-groups-namespace",
                }
                path = arguments[arguments.index("--data") + 1].removeprefix("fileb://")
                self.expected_rules = Path(path).read_bytes()
            self.mutations.append(arguments[2])
            self.on_mutation()
            return ""
        assert arguments[0] == "kubectl" and "apply" in arguments
        assert kwargs.get("input_text"), (
            "only the prerequisite manifests may be applied here"
        )
        self.applies.append((list(arguments), str(kwargs["input_text"])))
        return ""

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        self.reads.append(list(arguments))
        assert arguments[:3] == ["aws", "amp", "describe-rule-groups-namespace"]
        if self.expected_rules is None:
            return 254, "", "ResourceNotFoundException"
        return (
            0,
            json.dumps(
                {
                    "ruleGroupsNamespace": {
                        "status": {"statusCode": self.status},
                        "data": base64.b64encode(self.expected_rules).decode(),
                    }
                }
            ),
            "",
        )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    path = config_file(tmp_path)
    document = json.loads(path.read_text())
    document["health"] = {
        "amp_workspace_id": "ws-test",
        "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts",
    }
    document["clusters"][0]["adot_irsa_role_arn"] = (
        "arn:aws:iam::123456789012:role/gpu-fault-adot-writer"
    )
    path.write_text(json.dumps(document))
    runner = MonitoringRunner()
    release = ROLLOUT.RegionalRelease(ROLLOUT.ReleaseConfig.load(path), runner)
    state = SimpleNamespace(
        release=release, runner=runner, calls=[], on_state=lambda _phase: None
    )

    def save(phase: str, **updates: object) -> None:
        state.on_state(phase)
        state.calls.append(phase)
        release.state.update(phase=phase, **updates)

    for name in (
        "_ensure_contexts",
        "_apply_rds_ca_bundle",
        "_require_cpu_secrets",
        "_initialize_registry",
        "_prepare_nlb",
        "_upload_release",
        "_prepare_bootstrap_workflows",
        "_ensure_schema",
        "_apply_cpu",
        "_wait_nlb",
        "_cancel_active_installer_jobs",
    ):
        monkeypatch.setattr(release, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release, "_save_state", save)
    monkeypatch.setattr(
        release, "_scale_if_present", lambda *_a, **_k: state.calls.append("scale")
    )
    monkeypatch.setattr(
        release, "_validate_release", lambda: state.calls.append("validate")
    )
    monkeypatch.setattr(ROLLOUT, "ensure_runtime_profile", lambda _release: None)
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", lambda *_args: None)
    return state


@pytest.mark.parametrize("existing", [None, b"groups: []\n"])
@pytest.mark.parametrize("collector_enabled", [False, True])
def test_gpu_failure_never_publishes_rules_and_drains_monitoring_before_cleanup(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    existing: bytes | None,
    collector_enabled: bool,
) -> None:
    if not collector_enabled:
        config = harness.release.config
        harness.release.config = replace(
            config,
            clusters=tuple(
                replace(target, adot_irsa_role_arn=None) for target in config.clusters
            ),
        )
    runner = harness.runner
    runner.expected_rules = existing
    monitoring_started = threading.Event()
    monitoring_done = threading.Event()
    gpu_done = threading.Event()
    drain = threading.Event()
    terminal_observations = []

    def configure() -> None:
        monitoring_started.set()
        assert drain.wait(WAIT_SECONDS), "test did not release monitoring configuration"
        monitoring_done.set()

    def gpu(_release, _completed) -> None:
        assert monitoring_started.wait(WAIT_SECONDS), (
            "parallel monitoring never started"
        )
        gpu_done.set()
        raise RuntimeError("GPU failed before collector creation")

    def observe_state(phase: str) -> None:
        if phase == "bootstrap-failed" or phase.startswith("bootstrap-cleanup"):
            terminal_observations.append((monitoring_done.is_set(), gpu_done.is_set()))

    runner.configure = configure
    harness.on_state = observe_state
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.release.bootstrap)
        try:
            assert gpu_done.wait(WAIT_SECONDS), "GPU failure branch never finished"
            assert not future.done(), "GPU failure returned before monitoring drained"
            assert runner.mutations == [], (
                "expected rules changed before GPU convergence"
            )
            assert runner.expected_rules == existing
            assert "bootstrap-failed" not in harness.calls
            assert "scale" not in harness.calls
        finally:
            drain.set()
        with pytest.raises(RuntimeError, match="before collector creation"):
            future.result(timeout=WAIT_SECONDS)

    assert terminal_observations and all(
        monitoring and gpu for monitoring, gpu in terminal_observations
    )
    assert len(runner.installs) == 1
    assert runner.installs[0][2:] == ["--runtime-only"]
    assert runner.mutations == []
    assert runner.reads == []
    assert runner.expected_rules == existing
    assert "scale" in harness.calls
    assert "validate" not in harness.calls
    assert harness.release.state["phase"] == "bootstrap-cleaned"


@pytest.mark.parametrize("first", ["gpu", "monitoring"])
def test_success_publishes_once_after_both_branches_join(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    started = {name: threading.Event() for name in ("gpu", "monitoring")}
    finish = {name: threading.Event() for name in started}
    done = {name: threading.Event() for name in started}
    publication_observations = []
    runner = harness.runner

    def branch(name: str) -> None:
        started[name].set()
        assert finish[name].wait(WAIT_SECONDS), (
            "test did not release the bootstrap branch"
        )
        done[name].set()

    def gpu(_release, completed: set[str]) -> None:
        branch("gpu")
        completed.add("gpu-a")

    runner.configure = lambda: branch("monitoring")
    runner.on_mutation = lambda: publication_observations.append(
        (done["gpu"].is_set(), done["monitoring"].is_set(), "validate" in harness.calls)
    )
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.release.bootstrap)
        try:
            assert all(event.wait(WAIT_SECONDS) for event in started.values()), (
                "both bootstrap branches did not start"
            )
            finish[first].set()
            assert done[first].wait(WAIT_SECONDS), (
                "selected bootstrap branch did not finish"
            )
            assert runner.mutations == []
            assert "validate" not in harness.calls
            assert not future.done(), (
                "bootstrap returned before the publication barrier"
            )
        finally:
            for event in finish.values():
                event.set()
        future.result(timeout=WAIT_SECONDS)

    assert publication_observations == [(True, True, False)]
    assert len(runner.installs) == 1, "the full installer must not be rerun serially"
    assert runner.installs[0][2:] == ["--runtime-only"]
    assert runner.mutations == ["create-rule-groups-namespace"]
    expected = OBSERVABILITY.render_dataplane_expected_rules(harness.release)
    assert expected is not None
    assert runner.expected_rules == expected.encode("utf-8")
    assert harness.calls[-2:] == ["validate", "complete"]


def test_role_less_bootstrap_deletes_existing_rules_only_after_both_workers_finish(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = harness.release
    release.config = replace(
        release.config,
        clusters=tuple(
            replace(target, adot_irsa_role_arn=None)
            for target in release.config.clusters
        ),
    )
    harness.runner.expected_rules = b"groups: []\n"
    completed = []
    observed = []
    harness.runner.configure = lambda: completed.append("monitoring")
    harness.runner.on_mutation = lambda: observed.append(sorted(completed))
    monkeypatch.setattr(
        ROLLOUT, "bootstrap_gpu_clusters", lambda *_args: completed.append("gpu")
    )

    release.bootstrap()

    assert observed == [["gpu", "monitoring"]]
    assert harness.runner.installs[0][2:] == ["--runtime-only"]
    assert harness.runner.mutations == ["delete-rule-groups-namespace"]
    assert harness.runner.expected_rules is None
    assert harness.calls[-2:] == ["validate", "complete"]


def test_bootstrap_grants_release_metadata_reads_right_after_the_prerequisites(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control-plane Pods poll ``gpu-fault-release-metadata`` themselves
    (fleet pin hot reload), so the read grant lands with the prerequisites, in
    the site's namespace, before anything can start a Pod that polls."""

    release = harness.release
    release.config = replace(release.config, namespace="isolated-cpu")
    harness.runner.expected_rules = b"groups: []\n"
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", lambda *_args: None)

    release.bootstrap()

    applies = harness.runner.applies
    arguments, manifest = applies[0]
    assert "kind: Namespace" in manifest and "kind: RoleBinding" in manifest, (
        "the read grant must ride the prerequisites apply itself"
    )
    assert not any("kind: RoleBinding" in text for _arguments, text in applies[1:]), (
        "bootstrap must not apply the grant a second time before the store ensure"
    )
    assert arguments[:3] == ["kubectl", "--kubeconfig", release.config.cpu_kubeconfig]
    assert "name: gpu-fault-control-plane-release-metadata" in manifest
    assert 'resourceNames: ["gpu-fault-release-metadata"]' in manifest
    assert "namespace: isolated-cpu" in manifest
    assert "namespace: gpu-fault-system" not in manifest
    assert "name: gpu-fault-control-plane\n" in manifest, (
        "the grant must bind the control-plane ServiceAccount"
    )


def test_parallel_configuration_refuses_static_namespace_alias_before_invocation(
    harness: SimpleNamespace,
) -> None:
    release = harness.release
    release.config = replace(
        release.config,
        health=replace(
            release.config.health,
            amp_rule_namespace=OBSERVABILITY.DATAPLANE_EXPECTED_RULE_NAMESPACE,
        ),
    )

    with pytest.raises(ROLLOUT.ReleaseError, match="must not alias"):
        OBSERVABILITY.apply_control_plane_observability(release)

    assert harness.runner.installs == []
    assert harness.runner.reads == []
    assert harness.runner.mutations == []


def test_deferred_publication_reuses_current_active_rules(
    harness: SimpleNamespace,
) -> None:
    rendered = OBSERVABILITY.render_dataplane_expected_rules(harness.release)
    assert rendered is not None
    harness.runner.expected_rules = rendered.encode("utf-8")

    harness.release.bootstrap()

    assert len(harness.runner.installs) == 1
    assert harness.runner.installs[0][2:] == ["--runtime-only"]
    assert len(harness.runner.reads) == 1
    assert harness.runner.mutations == []
    assert harness.calls[-2:] == ["validate", "complete"]


def test_deferred_publication_updates_only_the_expected_namespace(
    harness: SimpleNamespace,
) -> None:
    harness.runner.expected_rules = b"groups: []\n"

    OBSERVABILITY.apply_dataplane_expected_rules(harness.release)

    assert harness.runner.installs == [], "expected rules must not rerun monitoring"
    assert harness.runner.mutations == ["put-rule-groups-namespace"]
    rendered = OBSERVABILITY.render_dataplane_expected_rules(harness.release)
    assert rendered is not None
    assert harness.runner.expected_rules == rendered.encode("utf-8")


def test_deferred_publication_refuses_unreadable_namespace_without_writes(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness.runner,
        "probe_output",
        lambda *_args, **_kwargs: (254, "", "AccessDeniedException"),
    )

    with pytest.raises(ROLLOUT.ReleaseError, match="cannot read"):
        harness.release.bootstrap()

    assert harness.runner.mutations == []
    assert "validate" not in harness.calls
    assert "complete" not in harness.calls
    assert "scale" in harness.calls


@pytest.mark.parametrize("data", [None, "", "%%%", []])
def test_deferred_publication_refuses_invalid_current_data(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, data: object
) -> None:
    monkeypatch.setattr(
        harness.runner,
        "probe_output",
        lambda *_args, **_kwargs: (
            0,
            json.dumps(
                {
                    "ruleGroupsNamespace": {
                        "status": {"statusCode": "ACTIVE"},
                        "data": data,
                    }
                }
            ),
            "",
        ),
    )

    with pytest.raises(ROLLOUT.ReleaseError, match="cannot read"):
        OBSERVABILITY.apply_dataplane_expected_rules(harness.release)

    assert harness.runner.mutations == []


@pytest.mark.parametrize("status", ["CREATION_FAILED", "UPDATE_FAILED", "UNKNOWN"])
def test_publication_must_become_active_before_validation(
    harness: SimpleNamespace, status: str
) -> None:
    harness.runner.status = status

    with pytest.raises(ROLLOUT.ReleaseError, match="AMP rejected|unexpected status"):
        harness.release.bootstrap()

    assert harness.runner.mutations == ["create-rule-groups-namespace"]
    assert "validate" not in harness.calls
    assert "complete" not in harness.calls
    assert "scale" in harness.calls


def test_deferred_publication_timeout_prevents_validation(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OBSERVABILITY, "EXPECTED_RULES_SETTLE_ATTEMPTS", 1)
    monkeypatch.setattr(OBSERVABILITY, "EXPECTED_RULES_SETTLE_SECONDS", 0)
    harness.runner.status = "CREATING"

    with pytest.raises(ROLLOUT.ReleaseError, match="still CREATING"):
        harness.release.bootstrap()

    assert harness.runner.mutations == ["create-rule-groups-namespace"]
    assert "validate" not in harness.calls
    assert "complete" not in harness.calls


def test_expected_rule_dry_run_does_not_probe_or_mutate_amp(
    harness: SimpleNamespace,
) -> None:
    harness.runner.dry_run = True

    OBSERVABILITY.apply_dataplane_expected_rules(harness.release)

    assert harness.runner.reads == []
    assert harness.runner.mutations == []
    assert harness.runner.installs == []
