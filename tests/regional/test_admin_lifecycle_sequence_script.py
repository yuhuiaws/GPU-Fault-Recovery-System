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

import pytest

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
snapshot_policy=retain
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) state_dir="$2"; shift 2 ;;
    --aurora-final-snapshot) snapshot_policy="$2"; shift 2 ;;
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
    python3 - "$state_dir" <<'PY'
import json, os, pathlib, sys
state = pathlib.Path(sys.argv[1])
dist = state / "snapshot/dist"
dist.mkdir(parents=True, exist_ok=True)
mapping = {"control_plane": "runtime", "executor": "executor", "node_dependencies": "node_dependencies"}
images = {name: {"reference": "repo@sha256:" + str(index) * 64, "deployable": True,
    "registry_reused": False, "image_input_sha256": "a" * 64,
    "source_identity_sha256": "b" * 64} for index,name in enumerate(mapping)}
(dist / "current-release.json").write_text(json.dumps({"delivery": {"images": {
    field: {"reference": images[name]["reference"]} for name,field in mapping.items()}}}))
(dist / "release-runtime-image.json").write_text(json.dumps({
    "schema_version": 3, "source_identity_sha256": "b" * 64, "images": images}))
def identity(role):
    return {"input_arn": os.environ[f"FAKE_{role}_ARN"],
        "eks_arn": os.environ.get(f"FAKE_{role}_EKS_ARN", os.environ[f"FAKE_{role}_ARN"]),
        "hyperpod_arn": os.environ.get(f"FAKE_{role}_HYPERPOD_ARN", "")}
(state / "bootstrap-state.json").write_text(json.dumps({
    "schema_version": 3, "site_id": "fixture-site",
    "phase": "aws-infrastructure-ready" if os.getenv("FAKE_ADMIN_FAIL_DEPLOY_STATE") == str(state) else "site-ready",
    "resources": {"release": {"manifest": str(dist / "current-release.json")},
    "initial_deploy_target": {"cpu": identity("CPU"),
       "gpu_clusters": [identity("GPU")]}}}))
PY
    if [[ "$state_dir" == "${FAKE_ADMIN_FAIL_DEPLOY_STATE:-}" ]]; then exit 19; fi
    ;;
  remove-cluster)
    mkdir -p "$state_dir/remove-cluster/cluster-a"
    python3 - "$state_dir" <<'PY'
import json, os, pathlib, sys
eks = os.environ.get("FAKE_GPU_EKS_ARN", os.environ["FAKE_GPU_ARN"])
path = pathlib.Path(sys.argv[1]) / "remove-cluster/cluster-a/state.json"
path.write_text(json.dumps({
    "phase": "COMPLETED", "attempt_id": "a" * 32,
    "target": {"eks_cluster_arn": eks},
    "evidence": {"DISCOVERED": {"provider_identity": {
        "eks_arn": eks, "hyperpod_arn": os.environ.get("FAKE_GPU_HYPERPOD_ARN", "")}}}}))
PY
    ;;
  join-cluster)
    mkdir -p "$state_dir/join-cluster/abcdef012345"
    printf '{"phase": "COMPLETED", "attempt": 1, "gpu_cluster_arn": "%s"}' "$FAKE_GPU_ARN" > "$state_dir/join-cluster/abcdef012345/state.json"
    ;;
  uninstall)
    mkdir -p "$state_dir/uninstall"
    python3 - "$state_dir" "$snapshot_policy" <<'PY'
import json, os, pathlib, sys
state = pathlib.Path(sys.argv[1])
value = {
    "phase": "COMPLETED", "cpu_disposition": "keep", "reset_database": True,
    "final_snapshot_policy": os.getenv("FAKE_UNINSTALL_SNAPSHOT_POLICY", sys.argv[2]),
    "site_identity": {"cpu_eks_arn": os.getenv("FAKE_CPU_EKS_ARN", os.environ["FAKE_CPU_ARN"])},
}
if os.getenv("FAKE_OMIT_UNINSTALL_SNAPSHOT_POLICY"):
    value.pop("final_snapshot_policy")
(state / "uninstall/state.json").write_text(json.dumps(value))
PY
    ;;
esac
if [[ "$verb" == "${FAKE_ADMIN_FAIL_AFTER_WRITE:-}" ]]; then exit 17; fi
"""

FAKE_DOCKER = """#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$FAKE_TOOL_LOG"
case "$*" in
  "buildx inspect"*) printf 'Name:   fixture-builder\\nDriver: docker-container\\n' ;;
  images*) printf 'gpu-fault-runtime-local:latest\\nubuntu:24.04\\n<none>:<none>\\n' ;;
esac
"""

FAKE_AWS = """#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_TOOL_LOG"])
with log.open("a") as stream:
    stream.write("aws " + " ".join(args) + "\\n")
operation = args[1]
def argument(name):
    return args[args.index(name) + 1]
if operation == "describe-repositories":
    if not os.getenv("FAKE_ECR_EXISTS"):
        print("An error occurred (RepositoryNotFoundException) when calling the DescribeRepositories operation: absent", file=sys.stderr)
        sys.exit(254)
    name = argument("--repository-names")
    print(json.dumps({"repositories": [{"repositoryName": name, "registryId": "123456789012",
        "repositoryArn": "arn:aws:ecr:us-west-2:123456789012:repository/" + name}]}))
elif operation == "list-tags-for-resource":
    print(json.dumps({"tags": [{"Key": "gpu-fault:site-id", "Value": "fixture-site"}]}))
elif operation in {"list-images", "batch-delete-image"}:
    name = argument("--repository-name")
    marker = log.parent / ("removed-" + hashlib.sha256(name.encode()).hexdigest())
    if operation == "list-images":
        print(json.dumps({"imageIds": [] if marker.exists() else [{"imageDigest": "sha256:" + "1" * 64, "imageTag": "build-component"}]}))
    else:
        marker.write_text("removed")
        print(json.dumps({"imageIds": json.loads(argument("--image-ids")), "failures": []}))
else:
    raise RuntimeError("unexpected test AWS operation")
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
        "FAKE_CPU_ARN": CPU_ARN,
        "FAKE_GPU_ARN": GPU_ARN,
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
            "--confirm-isolated-build-host",
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
        "--cpu-cluster keep --reset-database --aurora-final-snapshot retain "
        "--confirm UNINSTALL_GPU_FAULT" in line
        for line in lines
    ), "uninstall keeps the CPU cluster and resets the database"

    record = _record(tmp_path)
    assert record["case_id"] == "GF-REGIONAL-BOOT-029", record
    assert sorted(record["stages"]) == ["1", "2", "3", "4", "5"], record["stages"]
    for number, stage in record["stages"].items():
        assert stage["status"] == "PASS", (number, stage)
        assert isinstance(stage["wall_seconds"], int), (number, stage)
        log = tmp_path / "state" / "acceptance" / stage["log_name"]
        assert stage["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest(), (
            number,
            "the record binds each stage to the digest of its log",
        )
    for number in ("1", "5"):
        cold = record["stages"][number]["cold_build"]
        assert cold["cached_layers"] == 0 and cold["base_image_pulled"] is True
        assert set(cold["images"]) == {"control_plane", "executor", "node_dependencies"}
        assert all(
            image["registry_reused"] is False for image in cold["images"].values()
        ), "each cold deployment must build every current image without registry reuse"
    for number in ("2", "3", "4"):
        assert record["stages"][number]["phase"] == "COMPLETED", (number, record)
    assert record["stages"]["4"]["aurora_final_snapshot"] == "retain", (
        "the default reset keeps its final snapshot"
    )
    expected_inputs = {
        "cpu_cluster_arn_sha256": _sha256(CPU_ARN),
        "gpu_cluster_arn_sha256": _sha256(GPU_ARN),
        "admin_email_sha256": _sha256(ADMIN_EMAIL),
        "state_dir_sha256": _sha256(str(tmp_path / "state")),
        "second_state_dir_sha256": _sha256(str(tmp_path / "state-second")),
        "aurora_final_snapshot_sha256": _sha256("retain"),
    }
    assert {key: record["inputs"][key] for key in expected_inputs} == expected_inputs
    assert set(record["inputs"]) == set(expected_inputs) | {
        "repository_sha256",
        "source_identity_sha256",
        "previous_site_id_sha256",
        "email_wait_sha256",
    }, "resume must also bind the repository, source and wait/cache policy"
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
    assert stage["failure"], stage
    assert "CACHED" in (tmp_path / "state/acceptance" / stage["log_name"]).read_text()


def test_ecr_tags_are_deleted_only_when_the_site_repositories_exist(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    digest = _sha256("fixture-site")[:12]

    absent = _run(tmp_path, env)

    assert absent.returncode == 0, absent.stdout + absent.stderr
    lines = _tool_log(env)
    assert any(
        f"ecr describe-repositories --repository-names gpu-fault/runtime-{digest}"
        in line
        for line in lines
    ), "the runtime repository is derived from the previous site id"
    assert any(
        f"ecr describe-repositories --repository-names gpu-fault/runtime-cache-{digest}"
        in line
        for line in lines
    ), "the build-cache repository is derived from the previous site id"
    assert not any("batch-delete-image" in line for line in lines), (
        "no delete is attempted when the repositories are gone"
    )
    separate = tmp_path / "second-case"
    separate.mkdir()
    env = _install_fakes(separate)
    present = _run(separate, {**env, "FAKE_ECR_EXISTS": "1"})

    assert present.returncode == 0, present.stdout + present.stderr
    deletes = [line for line in _tool_log(env) if "batch-delete-image" in line]
    assert len(deletes) == 2, deletes
    assert any(
        f"--repository-name gpu-fault/runtime-{digest} --image-ids " in line
        for line in deletes
    ), deletes
    assert any(
        f"--repository-name gpu-fault/runtime-cache-{digest} --image-ids " in line
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


def test_completed_journal_cannot_override_failed_admin_exit(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)
    result = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_AFTER_WRITE": "remove-cluster"})
    assert result.returncode != 0
    assert _admin_verbs(_tool_log(env)) == ["deploy", "remove-cluster"]
    journal = tmp_path / "state/remove-cluster/cluster-a/state.json"
    assert json.loads(journal.read_text())["phase"] == "COMPLETED"
    record = _record(tmp_path)
    assert record["stages"]["2"]["status"] == "FAIL"
    assert "3" not in record["stages"]


def test_resume_rejects_changed_bound_inputs_without_rewriting_the_receipt(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})
    assert failed.returncode != 0
    path = tmp_path / "state/acceptance/admin-lifecycle-sequence.json"
    before, tools = path.read_bytes(), _tool_log(env)
    resumed = _run(
        tmp_path, env, "--stage", "3", "--admin-email", "changed@example.invalid"
    )
    assert resumed.returncode != 0
    assert "input identity" in resumed.stderr
    assert path.read_bytes() == before
    assert _tool_log(env) == tools


@pytest.mark.parametrize("stage,verb", [(2, "remove-cluster"), (4, "uninstall")])
def test_hyperpod_alias_sequence_resumes_after_failed_exit_with_completed_journal(
    tmp_path, stage, verb
):
    cpu = "arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu-fixture"
    gpu = "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu-fixture"
    env = {
        **_install_fakes(tmp_path),
        "FAKE_CPU_ARN": cpu,
        "FAKE_GPU_ARN": gpu,
        "FAKE_CPU_EKS_ARN": CPU_ARN,
        "FAKE_GPU_EKS_ARN": GPU_ARN,
        "FAKE_CPU_HYPERPOD_ARN": cpu,
        "FAKE_GPU_HYPERPOD_ARN": gpu,
    }
    arguments = ("--cpu-cluster-arn", cpu, "--gpu-cluster-arn", gpu)
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_AFTER_WRITE": verb}, *arguments)
    assert failed.returncode != 0
    before = _record(tmp_path)
    assert before["stages"][str(stage)]["status"] == "FAIL"
    assert str(stage + 1) not in before["stages"]
    assert _admin_verbs(_tool_log(env))[-1] == verb
    resumed = _run(tmp_path, env, *arguments, "--stage", str(stage))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    assert all(value["status"] == "PASS" for value in after["stages"].values()), (
        "resuming the original HyperPod aliases must complete every lifecycle stage"
    )
    assert after["inputs"] == before["inputs"]
    assert after["inputs"]["cpu_cluster_arn_sha256"] == _sha256(cpu)
    assert after["inputs"]["gpu_cluster_arn_sha256"] == _sha256(gpu)
    assert after["stages"][str(stage)]["attempts"] == 2
    assert (
        after["stages"][str(stage)]["journal_baseline"]
        == before["stages"][str(stage)]["journal_baseline"]
    )
    for number in range(1, stage):
        assert after["stages"][str(number)] == before["stages"][str(number)]


def test_explicit_snapshot_skip_is_not_a_retain_snapshot_companion_pass(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot", "skip")

    assert result.returncode == 0, result.stdout + result.stderr
    assert any(
        "--reset-database --aurora-final-snapshot skip --confirm UNINSTALL_GPU_FAULT"
        in line
        for line in _tool_log(env)
    ), "only the explicitly approved policy reaches uninstall"
    record = _record(tmp_path)
    assert record["inputs"]["aurora_final_snapshot_sha256"] == _sha256("skip"), (
        "snapshot approval must be bound before the first stage"
    )
    stage = record["stages"]["4"]
    assert stage["status"] == "PASS" and stage["aurora_final_snapshot"] == "skip", (
        "the driver records completion of the selected skip branch"
    )
    assert stage["companion_cases"] == {
        "GF-REGIONAL-BOOT-027": {
            "status": "NOT_RUN",
            "reason": "Aurora final snapshot explicitly skipped",
        }
    }, "skip cannot satisfy BOOT-027's available final snapshot requirement"
    assert "GF-REGIONAL-BOOT-027: NOT_RUN" in result.stderr, (
        "the terminal summary must preserve the same evidence distinction"
    )


@pytest.mark.parametrize("policy", ["bogus", "", "RETAIN", "retain|skip"])
def test_invalid_snapshot_policy_never_reaches_a_tool(
    tmp_path: Path, policy: str
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot", policy)

    assert result.returncode == 2 and "usage:" in result.stderr, (
        "unknown snapshot policy must fail argument validation"
    )
    assert _tool_log(env) == [], "invalid approval cannot start a lifecycle stage"
    assert not (tmp_path / "state").exists(), (
        "invalid approval cannot create acceptance state"
    )


def test_missing_snapshot_policy_value_is_a_usage_error(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot")

    assert result.returncode == 2 and "usage:" in result.stderr, (
        "the snapshot option requires an explicit value"
    )
    assert _tool_log(env) == [], "a missing approval value cannot reach any tool"


@pytest.mark.parametrize(
    "policy,stage,verb",
    [
        ("retain", 3, "join-cluster"),
        ("retain", 4, "uninstall"),
        ("skip", 4, "uninstall"),
    ],
)
def test_snapshot_approval_is_immutable_across_driver_retries(
    tmp_path: Path, policy: str, stage: int, verb: str
) -> None:
    env = _install_fakes(tmp_path)
    failed = _run(
        tmp_path,
        {**env, "FAKE_ADMIN_FAIL_VERB": verb},
        "--aurora-final-snapshot",
        policy,
    )
    assert failed.returncode != 0, "the fixture must stop at its requested stage"
    path = tmp_path / "state/acceptance/admin-lifecycle-sequence.json"
    before, tools = path.read_bytes(), _tool_log(env)

    refused = _run(
        tmp_path,
        env,
        "--stage",
        str(stage),
        "--aurora-final-snapshot",
        "skip" if policy == "retain" else "retain",
    )

    assert refused.returncode != 0 and "input identity" in refused.stderr, (
        "resume must not change the approval even before uninstall starts"
    )
    assert path.read_bytes() == before, "refusal must preserve the original receipt"
    assert _tool_log(env) == tools, "policy drift must fail before any tool invocation"

    resumed = _run(
        tmp_path, env, "--stage", str(stage), "--aurora-final-snapshot", policy
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    assert after["inputs"] == json.loads(before)["inputs"], (
        "same-policy retry must retain every original input binding"
    )
    assert after["stages"]["4"]["aurora_final_snapshot"] == policy, (
        "the completed uninstall must prove the original approval"
    )


@pytest.mark.parametrize(
    "approved,reported", [("retain", "skip"), ("skip", "retain"), ("retain", None)]
)
def test_uninstall_completion_requires_the_approved_snapshot_policy(
    tmp_path: Path, approved: str, reported: str | None
) -> None:
    env = _install_fakes(tmp_path)
    if reported is None:
        env["FAKE_OMIT_UNINSTALL_SNAPSHOT_POLICY"] = "1"
    else:
        env["FAKE_UNINSTALL_SNAPSHOT_POLICY"] = reported

    result = _run(tmp_path, env, "--aurora-final-snapshot", approved)

    assert result.returncode != 0 and "approved snapshot policy" in result.stderr, (
        "a successful command exit cannot substitute for the selected journal policy"
    )
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], "policy mismatch must prevent the second deployment"
    record = _record(tmp_path)
    assert record["stages"]["4"]["status"] == "FAIL" and "5" not in record["stages"], (
        "mismatched or missing approval evidence cannot produce a stage PASS"
    )


@pytest.mark.parametrize("stage", [1, 5])
def test_failed_initial_deploy_resumes_only_its_bound_checkpoint(
    tmp_path: Path, stage: int
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / ("state" if stage == 1 else "state-second")
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_DEPLOY_STATE": str(state)})
    assert failed.returncode != 0, "the fixture must interrupt the selected deploy"
    checkpoint = state / "bootstrap-state.json"
    before = checkpoint.read_bytes()
    assert json.loads(before)["phase"] == "aws-infrastructure-ready", (
        "the interrupted first deployment must retain its actual bootstrap checkpoint"
    )
    assert not (state / "site.yaml").exists(), "the fixture is not an installed site"
    record = _record(tmp_path)
    first = record["stages"][str(stage)]
    log = tmp_path / "state/acceptance" / first["log_name"]
    original_log = log.read_bytes()
    tools = _tool_log(env)

    resumed = _run(tmp_path, env, "--stage", str(stage))

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    current = after["stages"][str(stage)]
    assert current["status"] == "PASS" and current["attempts"] == 2, (
        "the original failed stage must be retried exactly once"
    )
    assert current["resumed_first_deploy"] is True, (
        "the receipt must distinguish an owned resume from a new cold deployment"
    )
    assert current["journal_baseline"] == first["journal_baseline"], (
        "retry cannot redefine its pre-deployment journal baseline"
    )
    assert after["inputs"] == record["inputs"], "all original inputs remain bound"
    assert log.read_bytes() == original_log, "the failed attempt log remains immutable"
    assert current["attempt_logs"][0] == {
        "name": log.name,
        "sha256": _sha256(original_log.decode()),
    }, "cold-build verification must retain the failed attempt's original proof"
    assert current["cold_build"]["cached_layers"] == 0, (
        "resuming must not bypass the uncached-build requirement"
    )
    new_tools = _tool_log(env)[len(tools) :]
    assert new_tools[0].startswith("gpu-fault-admin deploy "), (
        "a resumed deployment must not prune its already-started build or ECR tags"
    )
    assert f"--state-dir {state} " in new_tools[0], (
        "the resume must use the same state directory"
    )
    for number in range(1, stage):
        assert after["stages"][str(number)] == record["stages"][str(number)], (
            "already passed stages cannot be rewritten by a later resume"
        )


@pytest.mark.parametrize("stage", [1, 5])
def test_installed_site_without_bootstrap_proof_is_not_a_fresh_deploy(
    tmp_path: Path, stage: int
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / ("state" if stage == 1 else "state-second")
    state.mkdir()
    site = state / "site.yaml"
    site.write_text("spec: {}\n", encoding="utf-8")

    result = _run(tmp_path, env)

    assert result.returncode != 0 and "already holds site.yaml" in result.stderr, (
        "missing bootstrap proof cannot authorize treating an installed site as fresh"
    )
    tools = _tool_log(env)
    assert _admin_verbs(tools) == (
        [] if stage == 1 else ["deploy", "remove-cluster", "join-cluster", "uninstall"]
    ), "the installed target must not receive another deploy"
    assert sum(line.startswith("docker buildx prune") for line in tools) == (
        0 if stage == 1 else 1
    ), "the refused deploy must not prune its target's build caches"
    assert site.read_text(encoding="utf-8") == "spec: {}\n", (
        "the installed site must be preserved for reconciliation"
    )


def test_unfinished_second_deploy_without_predecessor_receipts_cannot_resume(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    second = tmp_path / "state-second"
    second.mkdir()
    checkpoint = second / "bootstrap-state.json"
    checkpoint.write_text(
        json.dumps({"schema_version": 3, "phase": "aws-infrastructure-ready"}),
        encoding="utf-8",
    )
    before = checkpoint.read_bytes()

    result = _run(tmp_path, env, "--stage", "5")

    assert result.returncode != 0 and "preceding PASS receipts" in result.stderr, (
        "an unfinished bootstrap alone is not a BOOT-029 resume authorization"
    )
    assert _tool_log(env) == [], "unbound resume must not reach any tool"
    assert checkpoint.read_bytes() == before, "unbound checkpoints must not be changed"


def _install_bound_admin(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    venv = state / "deployer-venv"
    (venv / "bin").mkdir(parents=True)
    binding = venv / "gpu-fault-managed-state-dir.json"
    binding.write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state.resolve())}),
        encoding="utf-8",
    )
    binding.chmod(0o600)
    admin = venv / "bin" / "gpu-fault-admin"
    body = FAKE_ADMIN.replace(
        "printf 'gpu-fault-admin %s\\n' \"$*\"", "printf 'bound-admin %s\\n' \"$*\""
    ).replace(
        'verb="$1"; shift',
        "[[ ! -v PYTHONPATH && ! -v PYTHONHOME && ! -v GPU_FAULT_REPOSITORY_ROOT "
        "&& ! -v GPU_FAULT_REPO_ROOT ]] || exit 31\n"
        '[[ "${PATH%%:*}" == "${0%/*}" ]] || exit 32\n'
        'verb="$1"; shift',
    )
    admin.write_text(body, encoding="utf-8")
    admin.chmod(0o700)
    return admin


def test_middle_stages_use_the_bound_cli_without_checkout_overrides(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    _install_bound_admin(tmp_path)
    env["GPU_FAULT_REPOSITORY_ROOT"] = str(tmp_path / "foreign-source")
    env["GPU_FAULT_REPO_ROOT"] = str(tmp_path / "foreign-source")

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert [line.split()[1] for line in lines if line.startswith("bound-admin ")] == [
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], lines
    assert _admin_verbs(lines) == ["deploy", "deploy"], (
        "cold deploys must still use the source-preparing PATH CLI"
    )
    assert {
        number: stage["status"] for number, stage in _record(tmp_path)["stages"].items()
    } == {str(number): "PASS" for number in range(1, 6)}


@pytest.mark.parametrize("damage", ["binding", "executable", "dangling-binding"])
def test_broken_bound_installation_never_falls_back_to_the_checkout(
    tmp_path: Path, damage: str
) -> None:
    env = _install_fakes(tmp_path)
    admin = _install_bound_admin(tmp_path)
    binding = admin.parent.parent / "gpu-fault-managed-state-dir.json"
    if damage == "binding":
        binding.write_text("{", encoding="utf-8")
    elif damage == "executable":
        admin.chmod(0o600)
    else:
        binding.unlink()
        binding.symlink_to(tmp_path / "missing.json")

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == ["deploy"], "no mutating stage may use the checkout"
    assert not any(line.startswith("bound-admin ") for line in lines), lines
    stages = _record(tmp_path)["stages"]
    assert {number: stage["status"] for number, stage in stages.items()} == {
        "1": "PASS",
        "2": "FAIL",
    }, stages


def test_failed_bound_command_stops_the_sequence_and_preserves_resume_guards(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    _install_bound_admin(tmp_path)
    env["FAKE_ADMIN_FAIL_VERB"] = "join-cluster"

    failed = _run(tmp_path, env)

    assert failed.returncode != 0, failed.stdout + failed.stderr
    assert {
        number: stage["status"] for number, stage in _record(tmp_path)["stages"].items()
    } == {"1": "PASS", "2": "PASS", "3": "FAIL"}
    original_inputs = _record(tmp_path)["inputs"]
    env.pop("FAKE_ADMIN_FAIL_VERB")
    resumed = _run(tmp_path, env, "--stage", "3")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _record(tmp_path)["inputs"] == original_inputs
    lines = _tool_log(env)
    assert [line.split()[1] for line in lines if line.startswith("bound-admin ")] == [
        "remove-cluster",
        "join-cluster",
        "join-cluster",
        "uninstall",
    ], lines
    assert _admin_verbs(lines) == ["deploy", "deploy"]
