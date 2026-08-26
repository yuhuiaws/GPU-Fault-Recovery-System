from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from gpu_fault.admin_site import SiteConfigError, load_site

LABEL_PATTERN = re.compile(r"^(?:[A-Za-z0-9](?:[-_.A-Za-z0-9]*[A-Za-z0-9])?)$")
MANAGED_LABEL = "gpu-fault.io/managed"
JOB_LABEL = "gpu-fault.io/job-id"
ATTEMPT_LABEL = "gpu-fault.io/attempt-id"
ROLE_LABEL = "gpu-fault.io/role"
CRITICAL_LABEL = "gpu-fault.io/critical"
EXPECTED_RANKS_ANNOTATION = "gpu-fault.io/expected-critical-ranks"
TRAINING_CONTAINER_ANNOTATION = "gpu-fault.io/training-container"
RUNTIME_PROFILE_ANNOTATION = "gpu-fault.io/runtime-profile-version"
RESTART_BUDGET_ANNOTATION = "gpu-fault.io/restart-budget"
RANK_OFFSET_ANNOTATION = "gpu-fault.io/rank-offset"
RANK_JOB_STRIDE_ANNOTATION = "gpu-fault.io/rank-job-stride"
SITE_ENV = "GPU_FAULT_SITE_FILE"


class TrainingSubmitError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedTrainingWorkload:
    job_id: str
    attempt_id: str
    manifest: str


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Inject GPU fault-management metadata and submit one "
            "Kubernetes training workload"
        )
    )
    result.add_argument("manifest", type=Path)
    result.add_argument("--job-id")
    result.add_argument("--attempt-id")
    result.add_argument("--attempt-number", type=int, default=1)
    result.add_argument(
        "--site",
        type=Path,
        help=f"RegionalSite YAML; defaults to {SITE_ENV}",
    )
    result.add_argument(
        "--runtime-profile-version",
        help="explicit override; must match --site when both are provided",
    )
    result.add_argument("--expected-critical-ranks", type=int)
    result.add_argument("--training-container")
    result.add_argument("--restart-budget", type=int)
    result.add_argument("--namespace")
    result.add_argument("--context")
    result.add_argument("--kubeconfig")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rendered manifest without invoking kubectl",
    )
    return result


def resolve_runtime_profile_version(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "runtime_profile_version", None) or "").strip()
    raw_site = getattr(args, "site", None) or os.getenv(SITE_ENV)
    if raw_site:
        try:
            site = load_site(Path(raw_site))
        except (OSError, SiteConfigError, ValueError) as exc:
            raise TrainingSubmitError(f"cannot load site Profile: {exc}") from exc
        site_version = str(site.release_config["runtime_profile"]["version"])
        if explicit and explicit != site_version:
            raise TrainingSubmitError(
                "--runtime-profile-version does not match the site Profile"
            )
        return site_version
    return explicit or os.getenv("GPU_FAULT_RUNTIME_PROFILE", "hyperpod-v1")


def _positive_int(value: Any, description: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingSubmitError(f"{description} must be an integer") from exc
    if parsed < 1:
        raise TrainingSubmitError(f"{description} must be greater than zero")
    return parsed


def _metadata(target: dict[str, Any]) -> tuple[dict, dict]:
    metadata = target.setdefault("metadata", {})
    return (
        metadata.setdefault("labels", {}),
        metadata.setdefault("annotations", {}),
    )


def _pod_spec(template: dict[str, Any]) -> dict[str, Any]:
    spec = template.get("spec")
    if not isinstance(spec, dict):
        raise TrainingSubmitError("Pod template requires spec")
    return spec


def _container_name(template: dict[str, Any], configured: str | None) -> str:
    containers = _pod_spec(template).get("containers") or []
    names = {
        item.get("name")
        for item in containers
        if isinstance(item, dict) and item.get("name")
    }
    if configured:
        if configured not in names:
            raise TrainingSubmitError(
                f"training container {configured!r} is not present "
                "in every Pod template"
            )
        return configured
    if len(names) != 1:
        raise TrainingSubmitError(
            "Pod templates with multiple containers require --training-container"
        )
    return next(iter(names))


def _inject_template(
    template: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    role: str,
    expected_ranks: int,
    runtime_profile: str,
    training_container: str | None,
    restart_budget: int | None,
    rank_offset: int = 0,
    rank_job_stride: int | None = None,
) -> None:
    labels, annotations = _metadata(template)
    labels.update(
        {
            MANAGED_LABEL: "true",
            JOB_LABEL: job_id,
            ATTEMPT_LABEL: attempt_id,
            ROLE_LABEL: role.lower(),
            CRITICAL_LABEL: "true",
        }
    )
    annotations.update(
        {
            EXPECTED_RANKS_ANNOTATION: str(expected_ranks),
            TRAINING_CONTAINER_ANNOTATION: _container_name(
                template, training_container
            ),
            RUNTIME_PROFILE_ANNOTATION: runtime_profile,
            RANK_OFFSET_ANNOTATION: str(rank_offset),
        }
    )
    if restart_budget is not None:
        annotations[RESTART_BUDGET_ANNOTATION] = str(restart_budget)
    if rank_job_stride is not None:
        annotations[RANK_JOB_STRIDE_ANNOTATION] = str(rank_job_stride)


def _job_templates(
    document: dict[str, Any],
) -> tuple[int, list[tuple[dict[str, Any], str, int, int | None]]]:
    spec = document.get("spec") or {}
    template = spec.get("template")
    if not isinstance(template, dict):
        raise TrainingSubmitError("Job requires spec.template")
    completions = _positive_int(spec.get("completions", 1), "completions")
    if completions > 1 and spec.get("completionMode") != "Indexed":
        raise TrainingSubmitError(
            "multi-rank Job requires spec.completionMode: Indexed"
        )
    return completions, [(template, "worker", 0, None)]


def _pytorch_templates(
    document: dict[str, Any],
) -> tuple[int, list[tuple[dict[str, Any], str, int, int | None]]]:
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs")
    if not isinstance(replicas, dict) or not replicas:
        raise TrainingSubmitError("PyTorchJob requires spec.pytorchReplicaSpecs")
    result = []
    offset = 0
    roles = list(replicas)
    ordered_roles = sorted(
        roles,
        key=lambda role: (
            str(role).lower() != "master",
            roles.index(role),
        ),
    )
    for role in ordered_roles:
        replica_spec = replicas[role]
        if not isinstance(replica_spec, dict):
            raise TrainingSubmitError(f"invalid PyTorch replica spec {role}")
        count = _positive_int(replica_spec.get("replicas", 1), f"{role} replicas")
        template = replica_spec.get("template")
        if not isinstance(template, dict):
            raise TrainingSubmitError(f"PyTorch replica {role} requires template")
        result.append((template, str(role), offset, None))
        offset += count
    return offset, result


def _jobset_templates(
    document: dict[str, Any],
) -> tuple[int, list[tuple[dict[str, Any], str, int, int | None]]]:
    replicated_jobs = (document.get("spec") or {}).get("replicatedJobs")
    if not isinstance(replicated_jobs, list) or not replicated_jobs:
        raise TrainingSubmitError("JobSet requires spec.replicatedJobs")
    result = []
    offset = 0
    for replicated_job in replicated_jobs:
        if not isinstance(replicated_job, dict):
            raise TrainingSubmitError("invalid replicatedJob")
        role = str(replicated_job.get("name") or "worker")
        job_count = _positive_int(
            replicated_job.get("replicas", 1),
            f"{role} Job replicas",
        )
        job_spec = (replicated_job.get("template") or {}).get("spec") or {}
        pod_template = job_spec.get("template")
        if not isinstance(pod_template, dict):
            raise TrainingSubmitError(
                f"replicatedJob {role} requires template.spec.template"
            )
        completions = _positive_int(
            job_spec.get("completions", 1),
            f"{role} completions",
        )
        if completions > 1 and job_spec.get("completionMode") != "Indexed":
            raise TrainingSubmitError(
                f"replicatedJob {role} with multiple completions "
                "requires completionMode: Indexed"
            )
        result.append((pod_template, role, offset, completions))
        offset += job_count * completions
    return offset, result


def inject_metadata(
    document: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    runtime_profile: str,
    expected_ranks: int | None = None,
    training_container: str | None = None,
    restart_budget: int | None = None,
) -> dict[str, Any]:
    kind = str(document.get("kind") or "").lower()
    resolver = {
        "job": _job_templates,
        "pytorchjob": _pytorch_templates,
        "jobset": _jobset_templates,
    }.get(kind)
    if resolver is None:
        raise TrainingSubmitError("supported kinds are Job, PyTorchJob and JobSet")
    inferred_ranks, templates = resolver(document)
    if expected_ranks is not None and expected_ranks < 1:
        raise TrainingSubmitError("--expected-critical-ranks must be positive")
    effective_ranks = expected_ranks if expected_ranks is not None else inferred_ranks
    if effective_ranks != inferred_ranks:
        raise TrainingSubmitError(
            "--expected-critical-ranks must equal the rank count "
            f"inferred from the workload ({inferred_ranks})"
        )
    labels, _ = _metadata(document)
    labels.update(
        {
            MANAGED_LABEL: "true",
            JOB_LABEL: job_id,
            ATTEMPT_LABEL: attempt_id,
        }
    )
    for template, role, offset, stride in templates:
        _inject_template(
            template,
            job_id=job_id,
            attempt_id=attempt_id,
            role=role,
            expected_ranks=effective_ranks,
            runtime_profile=runtime_profile,
            training_container=training_container,
            restart_budget=restart_budget,
            rank_offset=offset,
            rank_job_stride=stride,
        )
    return document


def _validate_id(value: str, description: str) -> None:
    if len(value) > 63 or not LABEL_PATTERN.fullmatch(value):
        raise TrainingSubmitError(
            f"{description} must be a valid Kubernetes label value "
            "of at most 63 characters"
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        documents = list(yaml.safe_load_all(path.read_text("utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingSubmitError(f"cannot read manifest {path}: {exc}") from exc
    documents = [item for item in documents if item is not None]
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise TrainingSubmitError(
            "manifest must contain exactly one Kubernetes workload"
        )
    return documents[0]


def render_workload(
    manifest: Path,
    *,
    job_id: str | None,
    attempt_id: str | None,
    attempt_number: int,
    runtime_profile_version: str,
    expected_critical_ranks: int | None,
    training_container: str | None,
    restart_budget: int | None,
    namespace: str | None = None,
) -> RenderedTrainingWorkload:
    if attempt_number < 1:
        raise TrainingSubmitError("--attempt-number must be positive")
    if restart_budget is not None and restart_budget < 0:
        raise TrainingSubmitError("--restart-budget cannot be negative")
    effective_job_id = job_id or f"train-{uuid.uuid4().hex}"
    effective_attempt_id = attempt_id or (f"{effective_job_id}-a{attempt_number:03d}")
    _validate_id(effective_job_id, "job ID")
    _validate_id(effective_attempt_id, "attempt ID")
    document = _load_manifest(manifest)
    if namespace is not None:
        if not namespace.strip():
            raise TrainingSubmitError("--namespace cannot be blank")
        document.setdefault("metadata", {})["namespace"] = namespace
    document = inject_metadata(
        document,
        job_id=effective_job_id,
        attempt_id=effective_attempt_id,
        runtime_profile=runtime_profile_version,
        expected_ranks=expected_critical_ranks,
        training_container=training_container,
        restart_budget=restart_budget,
    )
    return RenderedTrainingWorkload(
        job_id=effective_job_id,
        attempt_id=effective_attempt_id,
        manifest=yaml.safe_dump(document, sort_keys=False),
    )


def run(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    runtime_profile_version = resolve_runtime_profile_version(args)
    rendered = render_workload(
        args.manifest,
        job_id=args.job_id,
        attempt_id=args.attempt_id,
        attempt_number=args.attempt_number,
        runtime_profile_version=runtime_profile_version,
        expected_critical_ranks=args.expected_critical_ranks,
        training_container=args.training_container,
        restart_budget=args.restart_budget,
        namespace=args.namespace,
    )
    print(f"job-id: {rendered.job_id}", file=sys.stderr)
    print(f"attempt-id: {rendered.attempt_id}", file=sys.stderr)
    if args.dry_run:
        print(rendered.manifest, end="")
        return 0

    command = ["kubectl"]
    if args.kubeconfig:
        command.extend(["--kubeconfig", args.kubeconfig])
    if args.context:
        command.extend(["--context", args.context])
    if args.namespace:
        command.extend(["--namespace", args.namespace])
    command.extend(["apply", "-f", "-"])
    try:
        completed = runner(
            command,
            input=rendered.manifest,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise TrainingSubmitError(f"cannot execute kubectl: {exc}") from exc
    return completed.returncode


def main() -> None:
    try:
        raise SystemExit(run(parser().parse_args()))
    except TrainingSubmitError as exc:
        raise SystemExit(f"gpu-training-submit: {exc}") from exc


if __name__ == "__main__":
    main()
