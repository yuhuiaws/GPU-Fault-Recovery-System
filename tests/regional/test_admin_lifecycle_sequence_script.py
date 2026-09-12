"""Offline contract of the GF-REGIONAL-BOOT-029 driver.

The driver runs the five admin lifecycle stages in order against fake
``gpu-fault-admin``, ``docker`` and ``aws`` binaries that log every invocation
to one shared file, so the order across tools (cache prune before each cold
deploy, no prune between the middle stages) is observable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "acceptance" / "admin-lifecycle-sequence.sh"
CPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/cpu-fixture"
GPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-fixture"
ADMIN_EMAIL = "admin@example.com"

FAKE_ADMIN = """#!/usr/bin/env bash
set -euo pipefail
printf 'gpu-fault-admin %s\\n' "$*" >> "$FAKE_TOOL_LOG"
verb="$1"; shift
state_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) state_dir="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "$verb" == "${FAKE_ADMIN_FAIL_VERB:-}" ]]; then
  echo "injected failure in $verb" >&2
  exit 7
fi
case "$verb" in
  deploy)
    mkdir -p "$state_dir"
    echo "#1 [internal] load build definition from Dockerfile"
    echo "#5 [1/6] FROM registry.access.redhat.com/ubi9/python-312-minimal@sha256:0"
    echo "#5 resolve registry.access.redhat.com/ubi9/python-312-minimal done"
    echo "#5 extracting sha256:1111111111111111111111111111111111111111111111111111111111111111"
    echo "#6 [2/6] WORKDIR /app"
    if [[ -n "${FAKE_ADMIN_CACHED:-}" ]]; then echo "#6 CACHED"; fi
    echo "#7 [3/6] COPY requirements/runtime.lock /tmp/gpu-fault-runtime.lock"
    echo "#8 [4/6] RUN python -m pip install"
    echo "#9 [5/6] COPY wheels /tmp/gpu-fault-wheels"
    echo "#10 [6/6] RUN python -m venv"
    echo "#11 exporting to image"
    printf '{"schema_version": 1, "site_id": "fixture-site", "phase": "site-ready"}' \\
      > "$state_dir/bootstrap-state.json"
    ;;
  remove-cluster)
    mkdir -p "$state_dir/remove-cluster/cluster-a"
    echo '{"phase": "COMPLETED"}' > "$state_dir/remove-cluster/cluster-a/state.json"
    ;;
  join-cluster)
    mkdir -p "$state_dir/join-cluster/abcdef012345"
    echo '{"phase": "COMPLETED"}' > "$state_dir/join-cluster/abcdef012345/state.json"
    ;;
  uninstall)
    mkdir -p "$state_dir/uninstall"
    echo '{"phase": "COMPLETED"}' > "$state_dir/uninstall/state.json"
    ;;
esac
"""

FAKE_DOCKER = """#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$FAKE_TOOL_LOG"
case "$*" in
  "buildx inspect"*) printf 'Name:   fixture-builder\\nDriver: docker-container\\n' ;;
  images*) printf 'gpu-fault-runtime-local:latest\\nubuntu:24.04\\n<none>:<none>\\n' ;;
esac
"""

FAKE_AWS = """#!/usr/bin/env bash
printf 'aws %s\\n' "$*" >> "$FAKE_TOOL_LOG"
case "$*" in
  "ecr describe-repositories"*)
    if [[ -n "${FAKE_ECR_EXISTS:-}" ]]; then echo '{"repositories": []}'; else
      echo "RepositoryNotFoundException" >&2; exit 254; fi ;;
  "ecr list-images"*) echo '[{"imageDigest": "sha256:0", "imageTag": "build-0"}]' ;;
esac
"""


def _install_fakes(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in (
        ("gpu-fault-admin", FAKE_ADMIN),
        ("docker", FAKE_DOCKER),
        ("aws", FAKE_AWS),
    ):
        path = fake_bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_TOOL_LOG": str(tmp_path / "tools.log"),
    }


def _run(
    tmp_path: Path, env: dict[str, str], *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(SCRIPT),
            "--state-dir",
            str(tmp_path / "state"),
            "--cpu-cluster-arn",
            CPU_ARN,
            "--gpu-cluster-arn",
            GPU_ARN,
            "--admin-email",
            ADMIN_EMAIL,
            "--repo",
            str(ROOT),
            *extra,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _tool_log(env: dict[str, str]) -> list[str]:
    path = Path(env["FAKE_TOOL_LOG"])
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def _admin_verbs(lines: list[str]) -> list[str]:
    return [line.split()[1] for line in lines if line.startswith("gpu-fault-admin ")]


def _record(tmp_path: Path) -> dict:
    path = tmp_path / "state" / "acceptance" / "admin-lifecycle-sequence.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_runs_the_five_stages_in_order_and_prunes_before_each_cold_deploy(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
    ], "the five stages must run exactly once each, in lifecycle order"
    deploys = [index for index, line in enumerate(lines) if "admin deploy " in line]
    prunes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("docker buildx prune")
    ]
    builder_prunes = [
        index for index, line in enumerate(lines) if line == "docker builder prune -af"
    ]
    assert len(prunes) == len(builder_prunes) == 2, "one prune pair per cold deploy"
    assert prunes[0] < builder_prunes[0] < deploys[0], (
        "stage 1 prunes the caches before its deploy"
    )
    assert deploys[0] < prunes[1] < builder_prunes[1] < deploys[1], (
        "stage 5 prunes again, after the middle stages and before its deploy"
    )
    assert "docker buildx prune --builder fixture-builder -af" in lines, (
        "the prune targets the builder buildx reports"
    )
    assert "docker image rm -f gpu-fault-runtime-local:latest" in lines, (
        "every local gpu-fault-runtime image is removed, nothing else"
    )
    assert not any("ubuntu" in line for line in lines if "image rm" in line), (
        "unrelated local images are left alone"
    )
    first_deploy = lines[deploys[0]]
    assert f"--state-dir {tmp_path / 'state'} " in first_deploy, first_deploy
    assert f"--cpu-cluster-arn {CPU_ARN}" in first_deploy, first_deploy
    assert f"--admin-email {ADMIN_EMAIL}" in first_deploy, first_deploy
    assert f"--state-dir {tmp_path / 'state-second'} " in lines[deploys[1]], (
        "stage 5 deploys into the second, empty state directory"
    )
    assert any(
        line.startswith("gpu-fault-admin remove-cluster ")
        and f"--gpu-cluster-arn {GPU_ARN} --confirm REMOVE_GPU_CLUSTER" in line
        for line in lines
    ), "remove-cluster names the GPU cluster and carries its confirmation token"
    assert any(
        "--cpu-cluster keep --reset-database --confirm UNINSTALL_GPU_FAULT" in line
        for line in lines
    ), "uninstall keeps the CPU cluster and resets the database"

    record = _record(tmp_path)
    assert record["case_id"] == "GF-REGIONAL-BOOT-029", record
    assert sorted(record["stages"]) == ["1", "2", "3", "4", "5"], record["stages"]
    for number, stage in record["stages"].items():
        assert stage["status"] == "PASS", (number, stage)
        assert isinstance(stage["wall_seconds"], int), (number, stage)
        log = tmp_path / "state" / "acceptance" / f"stage-{number}-{stage['name']}.log"
        assert stage["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest(), (
            number,
            "the record binds each stage to the digest of its log",
        )
    for number in ("1", "5"):
        cold = record["stages"][number]["cold_build"]
        assert cold == {
            "cached_layers": 0,
            "base_image_pulled": True,
            "dockerfile_steps": 6,
        }, (number, cold)
    for number in ("2", "3", "4"):
        assert record["stages"][number]["phase"] == "COMPLETED", (number, record)
    assert record["inputs"] == {
        "cpu_cluster_arn_sha256": _sha256(CPU_ARN),
        "gpu_cluster_arn_sha256": _sha256(GPU_ARN),
        "admin_email_sha256": _sha256(ADMIN_EMAIL),
        "state_dir_sha256": _sha256(str(tmp_path / "state")),
        "second_state_dir_sha256": _sha256(str(tmp_path / "state-second")),
    }, "identifiers enter the record only as digests"
    serialized = json.dumps(record)
    for secret in (CPU_ARN, GPU_ARN, ADMIN_EMAIL, "cpu-fixture", str(tmp_path)):
        assert secret not in serialized, f"{secret} leaked into the record"


def test_stops_at_the_first_failing_stage_and_resumes_from_it(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})

    assert failed.returncode != 0, failed.stdout
    assert "FAILED at stage 3 (join-cluster)" in failed.stderr, failed.stderr
    assert "--stage 3" in failed.stderr, "the resume hint names the failed stage"
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
    ], "nothing after the failing stage runs"
    record = _record(tmp_path)
    assert {number: stage["status"] for number, stage in record["stages"].items()} == {
        "1": "PASS",
        "2": "PASS",
        "3": "FAIL",
    }, record["stages"]
    assert record["stages"]["3"]["failure"], "the failing stage records a reason"
    stage_one_digest = record["stages"]["1"]["log_sha256"]

    resumed = _run(tmp_path, env, "--stage", "3")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _admin_verbs(_tool_log(env))[3:] == [
        "join-cluster",
        "uninstall",
        "deploy",
    ], "--stage 3 resumes at the failed stage without repeating 1 and 2"
    record = _record(tmp_path)
    assert record["resumed_from_stage"] == 3, record
    assert record["stages"]["1"]["log_sha256"] == stage_one_digest, (
        "earlier stage records survive a resume"
    )
    assert record["stages"]["3"]["status"] == "PASS", record["stages"]["3"]
    assert record["stages"]["3"]["failure"] is None, record["stages"]["3"]
    assert record["stages"]["5"]["status"] == "PASS", record["stages"]["5"]


def test_a_cached_layer_fails_the_cold_build_criterion_at_stage_one(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, {**env, "FAKE_ADMIN_CACHED": "1"})

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in result.stderr, result.stderr
    assert "CACHED" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy"], "stage 2 never starts"
    stage = _record(tmp_path)["stages"]["1"]
    assert stage["status"] == "FAIL", stage
    assert stage["cold_build"]["cached_layers"] == 1, stage
    assert "CACHED" in stage["failure"], stage


def test_ecr_tags_are_deleted_only_when_the_site_repositories_exist(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "bootstrap-state.json").write_text(
        json.dumps({"site_id": "fixture-site", "phase": "site-ready"}), encoding="utf-8"
    )
    digest = _sha256("fixture-site")[:12]

    absent = _run(tmp_path, env, "--stage", "5")

    assert absent.returncode == 0, absent.stdout + absent.stderr
    lines = _tool_log(env)
    assert (
        f"aws ecr describe-repositories --repository-names gpu-fault/runtime-{digest}"
        in (lines)
    ), "the runtime repository is derived from the previous site id"
    assert (
        f"aws ecr describe-repositories --repository-names gpu-fault/runtime-cache-{digest}"
        in lines
    ), "the build-cache repository is derived from the previous site id"
    assert not any("batch-delete-image" in line for line in lines), (
        "no delete is attempted when the repositories are gone"
    )
    Path(env["FAKE_TOOL_LOG"]).unlink()
    (tmp_path / "state-second").rename(tmp_path / "state-second.done")

    present = _run(tmp_path, {**env, "FAKE_ECR_EXISTS": "1"}, "--stage", "5")

    assert present.returncode == 0, present.stdout + present.stderr
    deletes = [line for line in _tool_log(env) if "batch-delete-image" in line]
    assert len(deletes) == 2, deletes
    assert any(
        f"--repository-name gpu-fault/runtime-{digest} --image-ids "
        '[{"imageDigest": "sha256:0", "imageTag": "build-0"}]' in line
        for line in deletes
    ), deletes
    assert any(
        f"--repository-name gpu-fault/runtime-cache-{digest} --image-ids "
        '[{"imageTag":"buildcache-linux-amd64"}]' in line
        for line in deletes
    ), deletes


def test_a_first_deploy_refuses_a_state_directory_that_already_bootstrapped(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "bootstrap-state.json").write_text("{}", encoding="utf-8")

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "already holds bootstrap-state.json" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == [], "no admin command runs"


def test_usage_errors_exit_two_without_touching_any_tool(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    missing = subprocess.run(
        [str(SCRIPT), "--state-dir", str(tmp_path / "state")],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    bad_stage = _run(tmp_path, env, "--stage", "6")

    assert missing.returncode == 2, missing.stderr
    assert "usage:" in missing.stderr, missing.stderr
    assert bad_stage.returncode == 2, bad_stage.stderr
    assert _tool_log(env) == [], "argument errors never reach the tools"
