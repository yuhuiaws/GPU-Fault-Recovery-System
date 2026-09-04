from __future__ import annotations

import json
import subprocess
import time

import pytest

from gpu_fault.admin import bootstrap as admin_bootstrap
from gpu_fault.admin import bootstrap_aurora as admin_bootstrap_aurora
from gpu_fault.admin import bootstrap_dependencies as admin_bootstrap_dependencies
from gpu_fault.admin import release_artifacts as admin_release_artifacts
from gpu_fault.admin import release_repositories as admin_release_repositories
from gpu_fault.admin.bootstrap import _unused_subnet_cidrs
from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    BootstrapState,
    run_parallel,
)
from gpu_fault.admin.config import AuroraCapacityConfig
from gpu_fault.admin.release_repositories import ensure_release_repositories
from tests.admin._bootstrap_support import _cluster


def _cache_lifecycle_policy(*, retention_days: int = 7) -> dict:
    return {
        "rules": [
            {
                "rulePriority": 1,
                "description": (
                    "Expire untagged BuildKit cache artifacts after "
                    f"{retention_days} days"
                ),
                "selection": {
                    "tagStatus": "untagged",
                    "countType": "sinceImagePushed",
                    "countUnit": "days",
                    "countNumber": retention_days,
                },
                "action": {"type": "expire"},
            }
        ]
    }


def test_bootstrap_dependencies_require_docker_buildx(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        admin_bootstrap_dependencies,
        "load_deploy_host_tool_manifest",
        lambda: {
            "schema_version": 1,
            "python": {"major": 3, "minor": 12},
            "tools": [
                {
                    "name": "docker-buildx",
                    "executable": "docker",
                    "command": ["docker", "buildx", "version"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        admin_bootstrap_dependencies.shutil, "which", lambda _name: "/usr/bin/tool"
    )

    def missing_buildx(arguments, **_kwargs):
        commands.append(arguments)
        return __import__("subprocess").CompletedProcess(arguments, 1, "", "missing")

    monkeypatch.setattr(admin_bootstrap_dependencies.subprocess, "run", missing_buildx)

    with pytest.raises(BootstrapError, match="docker-buildx"):
        admin_bootstrap_dependencies.validate_bootstrap_dependencies()
    assert commands == [["docker", "buildx", "version"]], (
        "bootstrap dependency validation did not probe Docker Buildx"
    )


def test_cluster_arn_parser_accepts_eks_and_hyperpod() -> None:
    eks = Arn.parse("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a")
    hyperpod = Arn.parse("arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a")

    assert eks.service == "eks"
    assert eks.resource_name == "gpu-a"
    assert hyperpod.service == "sagemaker"
    assert hyperpod.resource_name == "gpu-a"


def test_hyperpod_discovery_uses_the_complete_input_arn(monkeypatch) -> None:
    hyperpod_arn = "arn:aws:sagemaker:us-east-1:123456789012:cluster/internal-id"
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class Runner:
        def aws_json(self, region, service, operation, *arguments, **_kwargs):
            calls.append((service, operation, arguments))
            if service == "sagemaker":
                return {
                    "ClusterArn": hyperpod_arn,
                    "ClusterName": "gpu-a",
                    "NodeRecovery": "None",
                    "Orchestrator": {
                        "Eks": {
                            "ClusterArn": (
                                "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
                            )
                        }
                    },
                }
            return {
                "cluster": {
                    "status": "ACTIVE",
                    "resourcesVpcConfig": {"vpcId": "vpc-a", "subnetIds": ["subnet-a"]},
                }
            }

    monkeypatch.setattr(
        admin_bootstrap,
        "discover_subnet_cidrs",
        lambda *_args, **_kwargs: ("10.0.0.0/24",),
    )

    discovered = admin_bootstrap.discover_cluster(
        Runner(), cluster_arn=hyperpod_arn, role="gpu", context="gpu-a"
    )

    assert discovered.hyperpod_name == "gpu-a"
    assert calls[0] == (
        "sagemaker",
        "describe-cluster",
        ("--cluster-name", hyperpod_arn),
    )


def test_eks_hyperpod_inventory_is_loaded_once_per_runner(monkeypatch) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    eks_arns = {
        "gpu-a": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        "gpu-b": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
    }

    class Runner:
        def aws_json(self, region, service, operation, *arguments, **_kwargs):
            calls.append((service, operation, arguments))
            if service == "sagemaker" and operation == "list-clusters":
                return {
                    "ClusterSummaries": [
                        {"ClusterName": name} for name in sorted(eks_arns)
                    ]
                }
            if service == "sagemaker":
                name = arguments[arguments.index("--cluster-name") + 1]
                return {
                    "ClusterArn": (
                        f"arn:aws:sagemaker:{region}:123456789012:cluster/{name}"
                    ),
                    "ClusterName": name,
                    "NodeRecovery": "None",
                    "Orchestrator": {"Eks": {"ClusterArn": eks_arns[name]}},
                }
            name = arguments[arguments.index("--name") + 1]
            return {
                "cluster": {
                    "status": "ACTIVE",
                    "resourcesVpcConfig": {
                        "vpcId": "vpc-a",
                        "subnetIds": [f"subnet-{name}"],
                    },
                }
            }

    monkeypatch.setattr(
        admin_bootstrap,
        "discover_subnet_cidrs",
        lambda *_args, **_kwargs: ("10.0.0.0/24",),
    )
    admin_bootstrap.hyperpod_inventory.cache_clear()
    runner = Runner()

    for name, eks_arn in eks_arns.items():
        discovered = admin_bootstrap.discover_cluster(
            runner, cluster_arn=eks_arn, role="gpu", context=name
        )
        assert discovered.hyperpod_name == name

    assert (
        sum(operation == "list-clusters" for _service, operation, _args in calls) == 1
    )
    assert (
        sum(
            service == "sagemaker" and operation == "describe-cluster"
            for service, operation, _args in calls
        )
        == 2
    )


def test_public_subnet_allocator_avoids_existing_ranges() -> None:
    selected = _unused_subnet_cidrs(
        ["10.0.0.0/24"], ["10.0.0.0/28", "10.0.0.16/28"], count=2
    )

    assert selected == ["10.0.0.32/28", "10.0.0.48/28"]


def test_independent_bootstrap_tasks_run_in_parallel(tmp_path) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")

    def task(value: str) -> str:
        time.sleep(0.1)
        return value

    started = time.monotonic()
    result = run_parallel(
        {"a": lambda: task("a"), "b": lambda: task("b"), "c": lambda: task("c")},
        state=state,
    )
    elapsed = time.monotonic() - started

    assert result == {"a": "a", "b": "b", "c": "c"}
    assert elapsed < 0.25


def test_bootstrap_aurora_formats_admin_config_capacity() -> None:
    assert (
        admin_bootstrap_aurora.scaling_configuration(
            AuroraCapacityConfig(min_acu=8.0, max_acu=32.0)
        )
        == "MinCapacity=8,MaxCapacity=32"
    )


def test_legacy_foundation_consumes_prebuilt_release(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "dist/current-release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "deployable": True,
                "release_id": "release-a",
                "delivery": {
                    "images": {
                        name: {
                            "reference": f"registry.example/{name}@sha256:"
                            + character * 64
                        }
                        for name, character in {
                            "runtime": "1",
                            "node_installer": "2",
                            "dcgm_exporter": "3",
                            "adot": "4",
                        }.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "compute_agent_config_digest",
        lambda *_args, **_kwargs: "a" * 64,
    )

    result = admin_release_artifacts.load_prebuilt_release(
        type("Runner", (), {"dry_run": False})(),
        repository_root=tmp_path,
        runtime_profile="profile-a",
    )

    assert result["manifest"] == str(manifest)
    assert result["agent_config_digest"] == "a" * 64
    assert result["images"]["runtime"].endswith("1" * 64), (
        "legacy bootstrap did not preserve the signed runtime image digest"
    )


def test_legacy_foundation_rejects_source_only_release(tmp_path) -> None:
    manifest = tmp_path / "dist/current-release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {"schema_version": 3, "deployable": False, "release_id": "release-a"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapError, match="deployable schema v3"):
        admin_release_artifacts.load_prebuilt_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            runtime_profile="profile-a",
        )


def test_staging_release_requires_explicit_bootstrap_authorization(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "dist/current-release.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "deployable": True,
                "staging_only": True,
                "release_id": "release-a",
                "delivery": {
                    "images": {
                        name: {
                            "reference": f"registry.example/{name}@sha256:"
                            + character * 64
                        }
                        for name, character in {
                            "runtime": "1",
                            "node_installer": "2",
                            "dcgm_exporter": "3",
                            "adot": "4",
                        }.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "compute_agent_config_digest",
        lambda *_args, **_kwargs: "a" * 64,
    )

    with pytest.raises(BootstrapError, match="staging-only"):
        admin_release_artifacts.load_prebuilt_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            runtime_profile="profile-a",
        )

    result = admin_release_artifacts.load_prebuilt_release(
        type("Runner", (), {"dry_run": False})(),
        repository_root=tmp_path,
        runtime_profile="profile-a",
        allow_staging=True,
    )

    assert result["staging_only"] is True


def test_release_repositories_are_created_with_separate_mutability(monkeypatch) -> None:
    calls = []

    class Runner:
        dry_run = False

        def aws_json(self, region, service, operation, *arguments, **kwargs):
            calls.append((region, service, operation, arguments, kwargs))
            assert service == "ecr"
            if operation == "create-repository":
                name = arguments[arguments.index("--repository-name") + 1]
                mutability = arguments[arguments.index("--image-tag-mutability") + 1]
                scan = arguments[
                    arguments.index("--image-scanning-configuration") + 1
                ].endswith("true")
                return {
                    "repository": {
                        "repositoryArn": (
                            f"arn:aws:ecr:us-east-1:123456789012:repository/{name}"
                        ),
                        "repositoryUri": (
                            f"123456789012.dkr.ecr.us-east-1.amazonaws.com/{name}"
                        ),
                        "imageTagMutability": mutability,
                        "imageScanningConfiguration": {"scanOnPush": scan},
                        "encryptionConfiguration": {"encryptionType": "AES256"},
                    }
                }
            assert operation == "put-lifecycle-policy"
            policy = arguments[arguments.index("--lifecycle-policy-text") + 1]
            return {"lifecyclePolicyText": policy}

    def missing(arguments, **_kwargs):
        error = (
            "LifecyclePolicyNotFoundException"
            if "get-lifecycle-policy" in arguments
            else "RepositoryNotFoundException"
        )
        return __import__("subprocess").CompletedProcess(arguments, 254, "", error)

    monkeypatch.setattr(admin_release_repositories.subprocess, "run", missing)

    repositories = ensure_release_repositories(
        Runner(), cpu=_cluster(), site_id="site-a"
    )

    assert repositories["runtime"]["repository_name"].startswith(
        "gpu-fault/runtime-"
    ), "runtime ECR repository name is not site-content-addressed"
    assert repositories["cache"]["repository_name"].startswith(
        "gpu-fault/runtime-cache-"
    ), "cache ECR repository name is not site-content-addressed"
    create_calls = {
        call[3][call[3].index("--repository-name") + 1]: call
        for call in calls
        if call[2] == "create-repository"
    }
    runtime_call = next(
        call for name, call in create_calls.items() if "runtime-cache-" not in name
    )
    cache_call = next(
        call for name, call in create_calls.items() if "runtime-cache-" in name
    )
    assert runtime_call[3][runtime_call[3].index("--image-tag-mutability") + 1] == (
        "IMMUTABLE"
    )
    assert cache_call[3][cache_call[3].index("--image-tag-mutability") + 1] == (
        "MUTABLE"
    )
    policy_call = next(call for call in calls if call[2] == "put-lifecycle-policy")
    policy = json.loads(
        policy_call[3][policy_call[3].index("--lifecycle-policy-text") + 1]
    )
    assert policy["rules"][0]["selection"] == {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 7,
    }
    assert repositories["cache"]["untagged_retention_days"] == "7"
    assert len(repositories["cache"]["lifecycle_policy_sha256"]) == 64


def test_release_repositories_reuse_matching_site_resources(monkeypatch) -> None:
    site_id = "site-a"

    def describe(arguments, **kwargs):
        del kwargs
        if "get-lifecycle-policy" in arguments:
            return __import__("subprocess").CompletedProcess(
                arguments,
                0,
                json.dumps(
                    {"lifecyclePolicyText": json.dumps(_cache_lifecycle_policy())}
                ),
                "",
            )
        name = arguments[arguments.index("--repository-names") + 1]
        cache = "runtime-cache-" in name
        return __import__("subprocess").CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "repositories": [
                        {
                            "repositoryArn": (
                                f"arn:aws:ecr:us-east-1:123456789012:repository/{name}"
                            ),
                            "repositoryUri": (
                                "123456789012.dkr.ecr.us-east-1.amazonaws.com/" + name
                            ),
                            "imageTagMutability": ("MUTABLE" if cache else "IMMUTABLE"),
                            "imageScanningConfiguration": {"scanOnPush": not cache},
                            "encryptionConfiguration": {"encryptionType": "AES256"},
                        }
                    ]
                }
            ),
            "",
        )

    class Runner:
        dry_run = False

        def aws_json(self, _region, service, operation, *_arguments, **_kwargs):
            assert service == "ecr"
            assert operation == "list-tags-for-resource"
            return {
                "tags": [
                    {"Key": "gpu-fault:site-id", "Value": site_id},
                    {"Key": "gpu-fault:owner", "Value": "release-bootstrap"},
                ]
            }

    monkeypatch.setattr(admin_release_repositories.subprocess, "run", describe)

    first = ensure_release_repositories(Runner(), cpu=_cluster(), site_id=site_id)
    second = ensure_release_repositories(Runner(), cpu=_cluster(), site_id=site_id)

    assert first == second
    assert first["cache"]["untagged_retention_days"] == "7"


def test_cache_repository_rejects_lifecycle_policy_drift(monkeypatch) -> None:
    site_id = "site-a"

    def describe(arguments, **kwargs):
        del kwargs
        if "get-lifecycle-policy" in arguments:
            policy = _cache_lifecycle_policy(retention_days=30)
            return __import__("subprocess").CompletedProcess(
                arguments,
                0,
                json.dumps({"lifecyclePolicyText": json.dumps(policy)}),
                "",
            )
        name = arguments[arguments.index("--repository-names") + 1]
        cache = "runtime-cache-" in name
        return __import__("subprocess").CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "repositories": [
                        {
                            "repositoryArn": (
                                f"arn:aws:ecr:us-east-1:123456789012:repository/{name}"
                            ),
                            "repositoryUri": (
                                "123456789012.dkr.ecr.us-east-1.amazonaws.com/" + name
                            ),
                            "imageTagMutability": ("MUTABLE" if cache else "IMMUTABLE"),
                            "imageScanningConfiguration": {"scanOnPush": not cache},
                            "encryptionConfiguration": {"encryptionType": "AES256"},
                        }
                    ]
                }
            ),
            "",
        )

    class Runner:
        dry_run = False

        def aws_json(self, _region, service, operation, *_arguments, **_kwargs):
            assert service == "ecr"
            assert operation == "list-tags-for-resource"
            return {
                "tags": [
                    {"Key": "gpu-fault:site-id", "Value": site_id},
                    {"Key": "gpu-fault:owner", "Value": "release-bootstrap"},
                ]
            }

    monkeypatch.setattr(admin_release_repositories.subprocess, "run", describe)

    with pytest.raises(BootstrapError, match="lifecycle policy differs"):
        ensure_release_repositories(Runner(), cpu=_cluster(), site_id=site_id)


@pytest.mark.parametrize(
    ("staging_only", "target"),
    ((False, "release-build"), (True, "release-build-staging")),
)
def test_admin_release_build_uses_ecr_and_state_signing_material(
    tmp_path, monkeypatch, staging_only, target
) -> None:
    signing = tmp_path / "release-signing"
    signing.mkdir()
    for name, value in (
        ("cosign.key", "private"),
        ("cosign.pub", "public"),
        ("cosign.password", "password"),
    ):
        path = signing / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/postgres"
    )
    monkeypatch.setattr(admin_release_artifacts, "_git_output", lambda *_args: "")
    monkeypatch.setattr(
        admin_release_artifacts,
        "restore_main_ci_candidate",
        lambda *_args, **_kwargs: None,
    )
    commands = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            commands.append((list(arguments), kwargs))
            if arguments[:3] == ["aws", "ecr", "get-login-password"]:
                return "login-password"
            if any("select-affected-tests.py" in item for item in arguments):
                return json.dumps({"postgres": False})
            return ""

    monkeypatch.setattr(
        admin_release_artifacts,
        "verify_prebuilt_release",
        lambda *args, **kwargs: commands.append((["verify"], kwargs)),
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "load_prebuilt_release",
        lambda *args, **kwargs: {
            "manifest": "dist/current-release.json",
            "release_id": "release-a",
            "images": {},
            "agent_config_digest": "a" * 64,
        },
    )

    result = admin_release_artifacts.build_signed_release(
        Runner(),
        repository_root=tmp_path,
        state_dir=tmp_path,
        region="us-east-1",
        runtime_repository=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/gpu-fault/runtime-a"
        ),
        cache_repository=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/gpu-fault/runtime-cache-a"
        ),
        runtime_profile="profile-a",
        staging_only=staging_only,
        impact_base="origin/release",
    )

    make_command, make_options = next(
        item for item in commands if item[0] and item[0][0] == "make"
    )
    assert target in make_command
    assert any(item.startswith("RUNTIME_IMAGE_REPOSITORY=") for item in make_command), (
        "admin release build did not receive the created runtime ECR repository"
    )
    if staging_only:
        assert "BASE=origin/release" in make_command
        assert "IMPACT_PLAN_PREPARED=1" in make_command
        assert any(item.startswith("STAGING_IMPACT_PLAN=") for item in make_command), (
            "staging release build did not receive the prepared impact plan"
        )
    else:
        assert all(not item.startswith("BASE=") for item in make_command), (
            "production release build unexpectedly received a staging impact base"
        )
    assert any(item.startswith("RUNTIME_IMAGE_CACHE_FROM=") for item in make_command), (
        "admin release build did not receive the created cache ECR repository"
    )
    assert make_options["env"]["GPU_FAULT_TEST_POSTGRES_URL"] == (
        "" if staging_only else "postgresql://postgres@127.0.0.1:5432/postgres"
    )
    assert make_options["env"]["COSIGN_PASSWORD"] == "password"
    assert result["release_id"] == "release-a"
    assert result["release_reused"] is False


@pytest.mark.parametrize("staging_only", (False, True))
def test_admin_release_reuses_signed_release_for_same_commit(
    tmp_path, monkeypatch, staging_only
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    for name, value in (
        (
            "current-release.json",
            json.dumps(
                {"schema_version": 3, "deployable": True, "staging_only": staging_only}
            ),
        ),
        (
            "current-attestation.json",
            json.dumps(
                {
                    "source": {"git_commit": "a" * 40, "dirty": False},
                    **({"impact_base": "origin/release"} if staging_only else {}),
                }
            ),
        ),
        ("current-attestation.bundle.json", "{}"),
    ):
        (dist / name).write_text(value, encoding="utf-8")
    commands = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            commands.append((list(arguments), kwargs))
            return ""

    def git_output(_root, *arguments):
        if arguments[0] == "status":
            return ""
        return "a" * 40

    runtime_repository = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/gpu-fault/runtime-a"
    )
    monkeypatch.setattr(admin_release_artifacts, "_git_output", git_output)
    monkeypatch.setattr(
        admin_release_artifacts, "runtime_image_exists", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "verify_prebuilt_release",
        lambda *args, **kwargs: commands.append((["verify"], kwargs)),
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "load_prebuilt_release",
        lambda *args, **kwargs: {
            "manifest": "dist/current-release.json",
            "release_id": "release-a",
            "images": {"runtime": runtime_repository + "@sha256:" + "b" * 64},
            "agent_config_digest": "a" * 64,
        },
    )

    result = admin_release_artifacts.build_signed_release(
        Runner(),
        repository_root=tmp_path,
        state_dir=tmp_path,
        region="us-east-1",
        runtime_repository=runtime_repository,
        cache_repository=None,
        runtime_profile="profile-a",
        staging_only=staging_only,
        impact_base="origin/release",
    )

    assert result["release_reused"] is True
    assert [item[0] for item in commands] == [["verify"]]
    assert commands[0][1]["staging_only"] is staging_only, (
        "release verification did not preserve the requested tier"
    )


def test_admin_release_prefers_promoted_main_candidate(tmp_path, monkeypatch) -> None:
    signing = tmp_path / "release-signing"
    signing.mkdir()
    for name, value in (
        ("cosign.key", "private"),
        ("cosign.pub", "public"),
        ("cosign.password", "password"),
    ):
        path = signing / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    gate = tmp_path / "dist/ci-gate.json"
    gate.parent.mkdir()
    gate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(admin_release_artifacts, "_git_output", lambda *_args: "")
    monkeypatch.setattr(
        admin_release_artifacts,
        "load_verified_ci_candidate_receipt",
        lambda *_args, **_kwargs: {
            "available": True,
            "ci_gate": str(gate),
            "run_id": 123,
        },
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "restore_main_ci_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "verified receipt repeated GitHub candidate restore"
        ),
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "isolated_postgres_url",
        lambda *_args, **_kwargs: pytest.fail(
            "promoted candidate started local PostgreSQL"
        ),
    )
    commands = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            commands.append((list(arguments), kwargs))
            if arguments[:3] == ["aws", "ecr", "get-login-password"]:
                return "login-password"
            return ""

    monkeypatch.setattr(
        admin_release_artifacts,
        "verify_prebuilt_release",
        lambda *args, **kwargs: commands.append((["verify"], kwargs)),
    )
    monkeypatch.setattr(
        admin_release_artifacts,
        "load_prebuilt_release",
        lambda *args, **kwargs: {
            "manifest": "dist/current-release.json",
            "release_id": "release-a",
            "images": {},
            "agent_config_digest": "a" * 64,
        },
    )

    result = admin_release_artifacts.build_signed_release(
        Runner(),
        repository_root=tmp_path,
        state_dir=tmp_path,
        region="us-east-1",
        runtime_repository=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/gpu-fault/runtime-a"
        ),
        cache_repository=None,
        runtime_profile="profile-a",
    )

    make_command = next(
        command for command, _options in commands if command and command[0] == "make"
    )
    assert "release-build-promoted" in make_command
    assert f"CI_GATE={gate}" in make_command
    assert result["release_source"] == "main_ci_candidate"


def test_source_only_release_is_not_reused(tmp_path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "current-release.json").write_text(
        json.dumps({"schema_version": 3, "deployable": False}), encoding="utf-8"
    )
    (dist / "current-attestation.json").write_text(
        json.dumps({"source": {"git_commit": "a" * 40, "dirty": False}}),
        encoding="utf-8",
    )
    (dist / "current-attestation.bundle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        admin_release_artifacts,
        "verify_prebuilt_release",
        lambda *_args, **_kwargs: pytest.fail("source-only release was verified"),
    )

    assert (
        admin_release_artifacts.load_reusable_signed_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            state_dir=tmp_path,
            region="us-east-1",
            runtime_repository="repository",
            runtime_profile="profile-a",
            staging_only=False,
            impact_base="origin/main",
        )
        is None
    )


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    (
        (0, "", True),
        (254, "ImageNotFoundException", False),
        (254, "RepositoryNotFoundException", False),
    ),
)
def test_runtime_image_existence_is_checked(
    monkeypatch, returncode, stderr, expected
) -> None:
    monkeypatch.setattr(
        admin_release_artifacts.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, returncode, "", stderr
        ),
    )

    assert (
        admin_release_artifacts.runtime_image_exists(
            region="us-east-1",
            reference=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                "gpu-fault/runtime@sha256:" + "a" * 64
            ),
        )
        is expected
    )


def test_admin_release_rejects_dirty_source_before_reuse(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        admin_release_artifacts, "_git_output", lambda *_args: " M src/gpu_fault/api.py"
    )

    with pytest.raises(BootstrapError, match="clean source tree"):
        admin_release_artifacts.build_signed_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            state_dir=tmp_path,
            region="us-east-1",
            runtime_repository="repository",
            cache_repository=None,
            runtime_profile="profile-a",
        )


def test_staging_release_is_not_reused_for_a_different_impact_base(
    tmp_path, monkeypatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "current-release.json").write_text(
        json.dumps({"schema_version": 3, "deployable": True, "staging_only": True}),
        encoding="utf-8",
    )
    (dist / "current-attestation.json").write_text(
        json.dumps(
            {
                "source": {"git_commit": "a" * 40, "dirty": False},
                "impact_base": "origin/old",
            }
        ),
        encoding="utf-8",
    )
    (dist / "current-attestation.bundle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        admin_release_artifacts,
        "verify_prebuilt_release",
        lambda *_args, **_kwargs: pytest.fail(
            "release with a different impact base was verified"
        ),
    )

    assert (
        admin_release_artifacts.load_reusable_signed_release(
            type("Runner", (), {"dry_run": False})(),
            repository_root=tmp_path,
            state_dir=tmp_path,
            region="us-east-1",
            runtime_repository="repository",
            runtime_profile="profile-a",
            staging_only=True,
            impact_base="origin/new",
        )
        is None
    )


def test_legacy_bootstrap_state_revalidates_exclusive_resources(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "site_id": "test",
                "phase": "site-ready",
                "resources": {"nlb_network": {"public_subnets": ["subnet-old"]}},
                "completed_tasks": ["nlb_network"],
            }
        ),
        encoding="utf-8",
    )

    state = BootstrapState(path, site_id="test")

    assert state.value["schema_version"] == 3, "legacy state was not upgraded"
    assert state.value["completed_tasks"] == [], "ownership tasks were not invalidated"
