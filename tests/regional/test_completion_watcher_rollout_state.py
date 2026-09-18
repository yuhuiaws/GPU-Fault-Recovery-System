from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_gpu_rollout as ROLLOUT
from gpu_fault_release.regional_release_diff import ReleaseChangeKind, ReleaseDiff
from tests.regional._release_orchestrator_support import RELEASE_MODULE, config_file
from tests.regional._resource_probe_fakes import resource_probe_result

ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> str:
    return yaml.safe_dump_all(
        [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "gpu-fault-completion-watcher-outbox"},
                "data": {"events.json": "[]"},
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "gpu-fault-completion-watcher-outbox-active"},
                "data": {"active-attempts.json": "{}"},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "gpu-fault-completion-watcher"},
            },
        ],
        sort_keys=False,
    )


def test_existing_watcher_state_configmap_is_not_reapplied() -> None:
    release = SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=False,
            probe_output=lambda arguments, **_kwargs: resource_probe_result(arguments),
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *args: list(args),
    )
    target = SimpleNamespace(cluster_id="gpu-a")

    rendered = ROLLOUT.preserve_completion_watcher_state(release, target, _manifest())
    documents = [item for item in yaml.safe_load_all(rendered) if item]

    assert [item["kind"] for item in documents] == ["Deployment"]


def test_missing_watcher_state_configmap_is_created() -> None:
    release = SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=False,
            probe_output=lambda arguments, **_kwargs: resource_probe_result(
                arguments, present=False
            ),
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *args: list(args),
    )

    rendered = ROLLOUT.preserve_completion_watcher_state(
        release, SimpleNamespace(cluster_id="gpu-a"), _manifest()
    )

    assert rendered == _manifest()


def test_the_new_active_state_object_is_applied_while_the_outbox_is_preserved() -> None:
    """The upgrade that splits the state: one object exists, the other does not.

    Probing the pair as one object would have made this rollout either drop the
    buffered outbox records or never create the object the new watcher persists
    its attempt state to.
    """

    probes: list[str] = []

    def probe_output(command, *_args, **_kwargs):
        name = command[command.index("get") + 2]
        probes.append(name)
        return resource_probe_result(command, present=not name.endswith("-active"))

    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False, probe_output=probe_output),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *args: list(args),
    )

    rendered = ROLLOUT.preserve_completion_watcher_state(
        release, SimpleNamespace(cluster_id="gpu-a"), _manifest()
    )
    documents = [item for item in yaml.safe_load_all(rendered) if item]

    assert probes == list(ROLLOUT.COMPLETION_WATCHER_STATE_CONFIG_MAPS), (
        f"both state objects must be probed, got {probes}"
    )
    assert [
        (item["kind"], (item.get("metadata") or {}).get("name")) for item in documents
    ] == [
        ("ConfigMap", "gpu-fault-completion-watcher-outbox-active"),
        ("Deployment", "gpu-fault-completion-watcher"),
    ], f"only the missing state object may be applied, got {documents}"


class _RecordingRunner:
    """A live-looking runner: ``existing`` names the ConfigMaps the probe finds."""

    dry_run = False

    def __init__(self, existing: tuple[str, ...]) -> None:
        self.existing = existing
        self.applied: list[tuple[list[str], str]] = []

    def run(self, arguments, **kwargs):
        self.applied.append((list(arguments), str(kwargs.get("input_text") or "")))
        return ""

    def probe_output(self, arguments, **_kwargs):
        name = arguments[arguments.index("get") + 2]
        return resource_probe_result(arguments, present=name in self.existing)


def _applied_objects(runner: _RecordingRunner) -> list[tuple[str, str]]:
    assert len(runner.applied) == 1, (
        f"the re-assert must be one kubectl apply, got {runner.applied}"
    )
    arguments, text = runner.applied[0]
    assert arguments[-3:] == ["apply", "-f", "-"], arguments
    return [
        (item["kind"], item["metadata"]["name"])
        for item in yaml.safe_load_all(text)
        if isinstance(item, dict)
    ]


def test_reassert_recreates_the_missing_state_object_and_the_role_only(
    tmp_path: Path,
) -> None:
    """GpuFaultCompletionActiveStateUnavailable's sanctioned recovery.

    ``kubectl apply -f deploy/dataplane/completion-watcher.yaml`` on a
    production cluster applies the unrendered Deployment too (placeholder image
    and wheel volume) and recreates the watcher into a CrashLoop. The re-assert
    goes through the release renderer and applies only what the alert names:
    the state ConfigMap that is missing and the ClusterRole (with its binding).
    """
    config = RELEASE_MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = _RecordingRunner(existing=("gpu-fault-completion-watcher-outbox",))
    release = RELEASE_MODULE.RegionalRelease(config, runner)

    ROLLOUT.reassert_completion_watcher_state(release, config.clusters[0])

    assert _applied_objects(runner) == [
        ("ConfigMap", "gpu-fault-completion-watcher-outbox-active"),
        ("ClusterRole", "gpu-fault-completion-watcher"),
        ("ClusterRoleBinding", "gpu-fault-completion-watcher"),
    ], "the Deployment, ServiceAccount and the existing outbox must not be touched"


def test_reassert_never_reapplies_a_state_object_that_exists(tmp_path: Path) -> None:
    """Both ConfigMaps hold live records; the manifest declares them empty."""
    config = RELEASE_MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = _RecordingRunner(existing=ROLLOUT.COMPLETION_WATCHER_STATE_CONFIG_MAPS)
    release = RELEASE_MODULE.RegionalRelease(config, runner)

    ROLLOUT.reassert_completion_watcher_state(release, config.clusters[0])

    assert [kind for kind, _name in _applied_objects(runner)] == [
        "ClusterRole",
        "ClusterRoleBinding",
    ], "a ConfigMap that exists must never be re-applied empty"


def test_stage_noop_reasserts_watcher_state_on_every_cluster_before_saving(
    monkeypatch,
) -> None:
    """The admin ``deploy`` fast path (a NOOP release) is the one command an
    operator runs, so it is where the missing objects come back."""
    events: list[str] = []
    targets = [SimpleNamespace(cluster_id="gpu-a"), SimpleNamespace(cluster_id="gpu-b")]
    release = SimpleNamespace(
        _load_state=lambda: {"phase": "complete"},
        _ensure_contexts=lambda: events.append("contexts"),
        _require_cpu_secrets=lambda: events.append("secrets"),
        _save_state=lambda *_args, **_kwargs: events.append("save"),
        _reassert_completion_watcher_state=lambda target: events.append(
            f"reassert:{target.cluster_id}"
        ),
        config=SimpleNamespace(clusters=targets),
        state=None,
    )
    monkeypatch.setattr(
        ADMIN,
        "classify_release",
        lambda _release, _state: ReleaseDiff(
            ReleaseChangeKind.NOOP, frozenset({"release_delivery"})
        ),
    )

    ADMIN.stage_noop_release(release)

    assert events == [
        "contexts",
        "secrets",
        "reassert:gpu-a",
        "reassert:gpu-b",
        "save",
    ], f"a NOOP release must re-assert the watcher state on each GPU cluster: {events}"


@pytest.mark.parametrize("allow_repair", [True, False])
def test_orchestrator_noop_reasserts_watcher_state(
    tmp_path: Path, monkeypatch, allow_repair: bool
) -> None:
    config = RELEASE_MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = RELEASE_MODULE.RegionalRelease(
        config, RELEASE_MODULE.Runner(dry_run=True)
    )
    events: list[str] = []
    for name in ("_ensure_contexts", "_require_cpu_secrets", "_validate_release"):
        monkeypatch.setattr(release, name, lambda name=name: events.append(name))
    monkeypatch.setattr(
        release, "_save_state", lambda *_args, **_kwargs: events.append("save")
    )
    monkeypatch.setattr(
        release,
        "_reassert_completion_watcher_state",
        lambda target: events.append(f"reassert:{target.cluster_id}"),
    )

    release.noop(
        ReleaseDiff(ReleaseChangeKind.NOOP, frozenset()),
        allow_prerequisite_repair=allow_repair,
    )

    assert ("reassert:gpu-a" in events) is allow_repair, events
    assert "_validate_release" in events, events
    assert events[-1] == "save", events
    if allow_repair:
        assert events.index("reassert:gpu-a") < events.index("save"), events


@pytest.mark.parametrize("repair_value", [None, "false", 0])
def test_noop_rejects_ambiguous_repair_policy(tmp_path: Path, repair_value) -> None:
    release = RELEASE_MODULE.RegionalRelease(
        RELEASE_MODULE.ReleaseConfig.load(config_file(tmp_path)),
        RELEASE_MODULE.Runner(dry_run=True),
    )
    with pytest.raises(RELEASE_MODULE.ReleaseError, match="boolean"):
        release.noop(
            ReleaseDiff(ReleaseChangeKind.NOOP, frozenset()),
            allow_prerequisite_repair=repair_value,
        )


def test_noop_rejects_changed_release_before_any_resource_access(
    tmp_path: Path,
) -> None:
    release = RELEASE_MODULE.RegionalRelease(
        RELEASE_MODULE.ReleaseConfig.load(config_file(tmp_path)),
        RELEASE_MODULE.Runner(dry_run=True),
    )
    with pytest.raises(RELEASE_MODULE.ReleaseError, match="unchanged release"):
        release.noop(
            ReleaseDiff(ReleaseChangeKind.FULL, frozenset({"schema"})),
            allow_prerequisite_repair=False,
        )
