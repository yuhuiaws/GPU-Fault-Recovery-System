"""The regional release's schema stage runs the three-step index method.

F-J3 / F-D2 deploy integration. The index-build and preflight Jobs were
wired only into the single-cluster deploy.sh; the regional `deploy` command
ran the ensure Job alone, whose transactional DDL would build a missing
index with a write lock on the hot tables. The schema stage now runs
index-build → ensure → preflight, and the release digest covers all three
manifests so a change to any of them is a FULL release.
"""

from __future__ import annotations

from pathlib import Path

from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
)
RENDERING_MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_rendering.py"
)
TOOL = ROOT / "deploy/control-plane/tools/ensure-postgres-schema.sh"
JOBS = (
    "gpu-fault-postgres-index-build",
    "gpu-fault-postgres-schema-ensure",
    "gpu-fault-postgres-schema-preflight",
)


def test_the_schema_tool_runs_index_build_ensure_then_preflight() -> None:
    script = TOOL.read_text(encoding="utf-8")
    invocations = [
        line for line in script.splitlines() if line.startswith("run_postgres_job ")
    ]

    assert [line.split()[1] for line in invocations] == list(JOBS)
    for manifest in RENDERING_MODULE.SCHEMA_JOB_MANIFESTS:
        assert manifest in script, manifest
        assert (ROOT / manifest).is_file(), manifest


def test_the_release_digest_covers_all_three_schema_jobs(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RecordingRunner:
        dry_run = True

        def run(self, args, **kwargs):
            return ""

    release = MODULE.RegionalRelease(config, RecordingRunner())

    payload = RENDERING_MODULE.render_release_payload(release)

    names = [
        doc["metadata"]["name"] for doc in payload["schema"] if doc.get("kind") == "Job"
    ]
    assert names == list(JOBS)
    assert all(
        doc["metadata"]["namespace"] == config.namespace for doc in payload["schema"]
    ), "every schema Job is rendered into the release namespace"
    assert RENDERING_MODULE.rendered_release_manifest_sha256(release), (
        "digest still renders"
    )
