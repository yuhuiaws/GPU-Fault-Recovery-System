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

from gpu_fault_release import regional_release_rendering as RENDERING_MODULE
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
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


def test_a_failed_schema_job_stops_the_stage_without_sitting_out_the_timeout(
    tmp_path: Path,
) -> None:
    """Live 2026-09-12: the index-build Job failed twice within seconds on a
    fresh database and `kubectl wait --for=condition=complete` then sat out its
    60-minute timeout. The stage must stop as soon as the Job reports Failed."""

    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    (fake_bin / "kubectl").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{calls}"\n'
        'case " $* " in\n'
        '  *" apply -f - "*) cat >/dev/null; echo "job.batch/x created" ;;\n'
        "  *\" get job/\"*) printf '%s\\n' '"
        '{"apiVersion":"batch/v1","kind":"Job",'
        '"metadata":{"name":"gpu-fault-postgres-index-build","uid":"schema-job-uid"},'
        '"status":{"conditions":[{"type":"Failed","status":"True",'
        '"reason":"BackoffLimitExceeded"}]}}\' ;;\n'
        '  *" logs job/"*) echo "relation does not exist" ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "kubectl").chmod(0o755)

    completed = subprocess.run(
        ["bash", str(TOOL)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(tmp_path / "kubeconfig"),
            "GPU_FAULT_WHEEL_CONFIGMAP": "gpu-fault-wheel-test",
            "GPU_FAULT_RUNTIME_IMAGE": "registry.example/runtime:test",
            "GPU_FAULT_JOB_POLL_SECONDS": "0",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "Job gpu-fault-postgres-index-build failed: BackoffLimitExceeded"
        in completed.stderr
    ), "the waiter must observe a valid failed Job, not reject malformed fixture JSON"
    assert "did not complete; its log follows" in completed.stderr
    recorded = calls.read_text(encoding="utf-8")
    assert "job/gpu-fault-postgres-schema-ensure" not in recorded, (
        "a failed index build must stop the stage before the ensure Job"
    )
    assert " wait " not in recorded, "no kubectl wait: it cannot see a failed Job"


def test_a_completed_schema_job_is_deleted_after_its_log(tmp_path: Path) -> None:
    """Live 2026-09-20: the three completed schema Jobs outlived the release and
    the fail-closed uninstall refused them as unregistered live resources. The
    release owns its transient Jobs: print the log, then delete the Job."""
    import os
    import subprocess

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    (fake_bin / "kubectl").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{calls}"\n'
        'case " $* " in\n'
        '  *" apply -f - "*) cat >/dev/null; echo "job.batch/x created" ;;\n'
        "  *\" get job/\"*) printf '%s\\n' '"
        '{"apiVersion":"batch/v1","kind":"Job",'
        '"metadata":{"name":"gpu-fault-postgres-index-build","uid":"schema-job-uid"},'
        '"status":{"succeeded":1,"conditions":[{"type":"Complete","status":"True"}]}}\' ;;\n'
        '  *" logs job/"*) echo "schema ok" ;;\n'
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "kubectl").chmod(0o755)
    completed = subprocess.run(
        ["bash", str(TOOL)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(tmp_path / "kubeconfig"),
            "GPU_FAULT_WHEEL_CONFIGMAP": "gpu-fault-wheel-test",
            "GPU_FAULT_RUNTIME_IMAGE": "registry.example/runtime:test",
            "GPU_FAULT_JOB_POLL_SECONDS": "0",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    recorded = calls.read_text(encoding="utf-8").splitlines()
    for name in (
        "gpu-fault-postgres-index-build",
        "gpu-fault-postgres-schema-ensure",
        "gpu-fault-postgres-schema-preflight",
    ):
        deletes = [i for i, line in enumerate(recorded) if f"delete job/{name}" in line]
        logs = [i for i, line in enumerate(recorded) if f"logs job/{name}" in line]
        assert logs, f"{name}: no log printed"
        assert len(deletes) == 2 and deletes[-1] > logs[-1], (
            f"{name}: the completed Job must be deleted after its log: {recorded}"
        )
