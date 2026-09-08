from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from gpu_fault_release import regional_release_gpu_rollout as ROLLOUT

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
            dry_run=False, probe_output=lambda *_args, **_kwargs: (0, "", "")
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
            probe_output=lambda *_args, **_kwargs: (
                1,
                "",
                'Error from server (NotFound): configmaps "x" not found',
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
        name = command[-1]
        probes.append(name)
        if name.endswith("-active"):
            return (1, "", 'Error from server (NotFound): configmaps "x" not found')
        return (0, "", "")

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
