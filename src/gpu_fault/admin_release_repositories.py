from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any, Mapping

from gpu_fault.admin_bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    SITE_TAG_KEY,
    assert_site_tag,
    tag_map,
)
from gpu_fault.admin_release_artifacts import build_signed_release


BUILD_CACHE_UNTAGGED_RETENTION_DAYS = 7


def _release_repository_name(site_id: str, *, cache: bool) -> str:
    digest = hashlib.sha256(site_id.encode()).hexdigest()[:12]
    kind = "runtime-cache" if cache else "runtime"
    return f"gpu-fault/{kind}-{digest}"


def _build_cache_lifecycle_policy() -> dict[str, Any]:
    return {
        "rules": [
            {
                "rulePriority": 1,
                "description": (
                    "Expire untagged BuildKit cache artifacts after "
                    f"{BUILD_CACHE_UNTAGGED_RETENTION_DAYS} days"
                ),
                "selection": {
                    "tagStatus": "untagged",
                    "countType": "sinceImagePushed",
                    "countUnit": "days",
                    "countNumber": BUILD_CACHE_UNTAGGED_RETENTION_DAYS,
                },
                "action": {"type": "expire"},
            }
        ]
    }


def _canonical_policy_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lifecycle_policy_from_response(
    response: Mapping[str, Any],
    *,
    repository_name: str,
) -> dict[str, Any]:
    raw = response.get("lifecyclePolicyText")
    if not isinstance(raw, str):
        raise BootstrapError(
            f"ECR cache repository {repository_name} lifecycle policy is missing"
        )
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            f"ECR cache repository {repository_name} lifecycle policy is invalid"
        ) from exc
    if not isinstance(policy, dict):
        raise BootstrapError(
            f"ECR cache repository {repository_name} lifecycle policy is invalid"
        )
    return policy


def _ensure_build_cache_lifecycle_policy(
    runner: CommandRunner,
    *,
    region: str,
    repository_name: str,
) -> str:
    desired = _build_cache_lifecycle_policy()
    desired_text = _canonical_policy_text(desired)
    desired_sha256 = hashlib.sha256(desired_text.encode()).hexdigest()
    if runner.dry_run:
        return desired_sha256
    described = subprocess.run(
        [
            "aws",
            "ecr",
            "get-lifecycle-policy",
            "--region",
            region,
            "--repository-name",
            repository_name,
            "--output",
            "json",
        ],
        text=True,
        capture_output=True,
    )
    if described.returncode == 0:
        try:
            response = json.loads(described.stdout)
        except json.JSONDecodeError as exc:
            raise BootstrapError(
                f"ECR cache repository {repository_name} lifecycle response is invalid"
            ) from exc
        if not isinstance(response, dict):
            raise BootstrapError(
                f"ECR cache repository {repository_name} lifecycle response is invalid"
            )
        current = _lifecycle_policy_from_response(
            response,
            repository_name=repository_name,
        )
        if _canonical_policy_text(current) != desired_text:
            raise BootstrapError(
                f"ECR cache repository {repository_name} lifecycle policy differs "
                "from the required untagged-cache retention policy"
            )
        return desired_sha256
    if "LifecyclePolicyNotFoundException" not in described.stderr:
        raise BootstrapError(
            f"cannot inspect ECR cache repository {repository_name} lifecycle policy: "
            f"{described.stderr.strip()}"
        )
    response = runner.aws_json(
        region,
        "ecr",
        "put-lifecycle-policy",
        "--repository-name",
        repository_name,
        "--lifecycle-policy-text",
        desired_text,
        mutate=True,
    )
    applied = _lifecycle_policy_from_response(
        response,
        repository_name=repository_name,
    )
    if _canonical_policy_text(applied) != desired_text:
        raise BootstrapError(
            f"ECR cache repository {repository_name} lifecycle policy was not applied"
        )
    return desired_sha256


def _ensure_ecr_repository(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    cache: bool,
) -> dict[str, str]:
    name = _release_repository_name(site_id, cache=cache)
    expected_mutability = "MUTABLE" if cache else "IMMUTABLE"
    expected_scan = not cache
    if runner.dry_run:
        result = {
            "repository_name": name,
            "repository_arn": (
                f"arn:aws:ecr:{cpu.region}:{cpu.account_id}:repository/{name}"
            ),
            "repository_uri": (
                f"{cpu.account_id}.dkr.ecr.{cpu.region}.amazonaws.com/{name}"
            ),
            "ownership": "CREATED",
            "purpose": "build-cache" if cache else "runtime",
        }
        if cache:
            result.update(
                {
                    "lifecycle_policy_sha256": (
                        _ensure_build_cache_lifecycle_policy(
                            runner,
                            region=cpu.region,
                            repository_name=name,
                        )
                    ),
                    "untagged_retention_days": str(BUILD_CACHE_UNTAGGED_RETENTION_DAYS),
                }
            )
        return result
    described = subprocess.run(
        [
            "aws",
            "ecr",
            "describe-repositories",
            "--region",
            cpu.region,
            "--repository-names",
            name,
            "--output",
            "json",
        ],
        text=True,
        capture_output=True,
    )
    if described.returncode == 0:
        repositories = json.loads(described.stdout).get("repositories", [])
        if len(repositories) != 1:
            raise BootstrapError(f"ECR repository {name} did not resolve uniquely")
        repository = repositories[0]
        tags = runner.aws_json(
            cpu.region,
            "ecr",
            "list-tags-for-resource",
            "--resource-arn",
            str(repository["repositoryArn"]),
        ).get("tags")
        assert_site_tag(
            tags,
            site_id=site_id,
            description=f"ECR repository {name}",
        )
        owner = tag_map(tags).get("gpu-fault:owner")
        if owner != "release-bootstrap":
            raise BootstrapError(
                f"ECR repository {name} has unexpected owner {owner!r}"
            )
    elif "RepositoryNotFoundException" in described.stderr:
        repository = runner.aws_json(
            cpu.region,
            "ecr",
            "create-repository",
            "--repository-name",
            name,
            "--image-tag-mutability",
            expected_mutability,
            "--image-scanning-configuration",
            f"scanOnPush={str(expected_scan).lower()}",
            "--encryption-configuration",
            "encryptionType=AES256",
            "--tags",
            f"Key={SITE_TAG_KEY},Value={site_id}",
            "Key=gpu-fault:owner,Value=release-bootstrap",
            mutate=True,
        )["repository"]
    else:
        raise BootstrapError(
            f"cannot inspect ECR repository {name}: {described.stderr.strip()}"
        )
    if repository.get("imageTagMutability") != expected_mutability:
        raise BootstrapError(
            f"ECR repository {name} must use {expected_mutability} tags"
        )
    if (
        bool((repository.get("imageScanningConfiguration") or {}).get("scanOnPush"))
        is not expected_scan
    ):
        raise BootstrapError(f"ECR repository {name} scan-on-push setting is invalid")
    if (repository.get("encryptionConfiguration") or {}).get(
        "encryptionType"
    ) != "AES256":
        raise BootstrapError(f"ECR repository {name} must use AES256 encryption")
    result = {
        "repository_name": name,
        "repository_arn": str(repository["repositoryArn"]),
        "repository_uri": str(repository["repositoryUri"]),
        "ownership": "CREATED",
        "purpose": "build-cache" if cache else "runtime",
    }
    if cache:
        result.update(
            {
                "lifecycle_policy_sha256": _ensure_build_cache_lifecycle_policy(
                    runner,
                    region=cpu.region,
                    repository_name=name,
                ),
                "untagged_retention_days": str(BUILD_CACHE_UNTAGGED_RETENTION_DAYS),
            }
        )
    return result


def ensure_release_repositories(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
) -> dict[str, dict[str, str]]:
    return {
        "runtime": _ensure_ecr_repository(
            runner,
            cpu=cpu,
            site_id=site_id,
            cache=False,
        ),
        "cache": _ensure_ecr_repository(
            runner,
            cpu=cpu,
            site_id=site_id,
            cache=True,
        ),
    }


def prepare_signed_release(
    runner: CommandRunner,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    site_id: str,
    state: BootstrapState,
) -> dict[str, Any]:
    repositories = ensure_release_repositories(
        runner,
        cpu=cpu,
        site_id=site_id,
    )
    state.record("release_repositories", repositories)
    state.complete("release_repositories")
    release = build_signed_release(
        runner,
        repository_root=request.repository_root,
        state_dir=request.state_dir,
        region=cpu.region,
        runtime_repository=repositories["runtime"]["repository_uri"],
        cache_repository=repositories["cache"]["repository_uri"],
        runtime_profile="hyperpod-v1",
        cosign_signing_key=request.cosign_signing_key,
        cosign_public_key=request.cosign_public_key,
        cosign_password_file=request.cosign_password_file,
        staging_only=request.staging_only_release,
        impact_base=request.impact_base,
    )
    state.record("release", release)
    state.complete("release")
    state.phase("release-ready")
    return release
