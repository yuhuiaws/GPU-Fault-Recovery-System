from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.deploy_consent import refuse_unconsented_release
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    assert_site_tag,
    tag_map,
)
from gpu_fault.admin.release_artifacts import build_signed_release

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
    with ThreadPoolExecutor(max_workers=2) as executor:
        runtime = executor.submit(
            _ensure_ecr_repository,
            runner,
            cpu=cpu,
            site_id=site_id,
            cache=False,
        )
        cache = executor.submit(
            _ensure_ecr_repository,
            runner,
            cpu=cpu,
            site_id=site_id,
            cache=True,
        )
        return {
            "runtime": runtime.result(),
            "cache": cache.result(),
        }


class SignedReleaseBuild:
    """``prepare_signed_release`` on its own thread, joined by whoever needs it.

    The release build -- ECR repositories, the release gates, the wheels and
    the runtime image -- reads only the source snapshot; the AWS foundation
    (Aurora, the NLB, PKI, roles, notification resources) reads none of it.
    Bootstrap used to run the two back to back, so on a first deployment the
    ten-minute Aurora wait and the ten-minute cold build added up. Now the
    build starts as soon as the scope is discovered and the graph runs
    beside it; only the platform tasks that ship an image (``monitoring_install``,
    ``aurora_refresh``) block on ``result()``, from their own worker threads.

    ``result()`` re-raises the build's failure to every caller and applies the
    release-consent refusal once, so an unconsented release is refused the
    first time anything needs it. ``prepare`` is injected so the caller's
    module attribute -- what tests replace -- is what runs.
    """

    def __init__(
        self,
        prepare: Callable[..., dict[str, Any]],
        *,
        existing_site: Mapping[str, Any] | None,
        request: BootstrapRequest,
        **arguments: Any,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="release-build"
        )
        self._future = self._executor.submit(prepare, request=request, **arguments)
        self._existing_site = existing_site
        self._state_dir = request.state_dir
        self._lock = threading.Lock()
        self._checked = False

    def result(self) -> dict[str, Any]:
        release = self._future.result()
        with self._lock:
            if not self._checked:
                refuse_unconsented_release(
                    state_dir=self._state_dir,
                    manifest_path=Path(str(release["manifest"])).expanduser(),
                    existing_site=self._existing_site,
                )
                self._checked = True
        return release

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def prepare_signed_release(
    runner: CommandRunner,
    *,
    request: BootstrapRequest,
    cpu: ClusterIdentity,
    gpu_clusters: tuple[ClusterIdentity, ...],
    site_id: str,
    state: BootstrapState,
) -> dict[str, Any]:
    repositories = ensure_release_repositories(
        runner,
        cpu=cpu,
        site_id=site_id,
    )
    release = build_signed_release(
        runner,
        repository_root=request.repository_root,
        state_dir=request.state_dir,
        region=cpu.region,
        runtime_repository=repositories["runtime"]["repository_uri"],
        cache_repository=repositories["cache"]["repository_uri"],
        runtime_profile="hyperpod-v1",
        staging_only=request.staging_only_release,
        impact_base=request.impact_base,
    )
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=cpu,
        gpu_clusters=gpu_clusters,
        release=release,
    )
    state.record("release_repositories", repositories)
    state.complete("release_repositories")
    state.record("release", release)
    state.complete("release")
    state.phase("release-ready")
    return release
