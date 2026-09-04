from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_gpu_rollout.py"
)


def _manifest() -> str:
    return yaml.safe_dump_all(
        [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "gpu-fault-completion-watcher-outbox"},
                "data": {"active-attempts.json": "{}", "events.json": "[]"},
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
