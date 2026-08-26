from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_fault.training_submit_cli import (
    TrainingSubmitError,
    render_workload,
    resolve_runtime_profile_version,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Inject GPU fault-management metadata into one "
            "Kubernetes training workload without submitting it"
        )
    )
    result.add_argument("manifest", type=Path)
    result.add_argument("-o", "--output", type=Path)
    result.add_argument("--job-id")
    result.add_argument("--attempt-id")
    result.add_argument("--attempt-number", type=int, default=1)
    result.add_argument(
        "--site",
        type=Path,
        help="RegionalSite YAML; defaults to GPU_FAULT_SITE_FILE",
    )
    result.add_argument(
        "--runtime-profile-version",
        help="explicit override; must match --site when both are provided",
    )
    result.add_argument("--expected-critical-ranks", type=int)
    result.add_argument("--training-container")
    result.add_argument("--restart-budget", type=int)
    result.add_argument("--namespace")
    return result


def run(args: argparse.Namespace) -> int:
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
    if args.output is None:
        print(rendered.manifest, end="")
    else:
        try:
            args.output.write_text(rendered.manifest, encoding="utf-8")
        except OSError as exc:
            raise TrainingSubmitError(
                f"cannot write manifest {args.output}: {exc}"
            ) from exc
    print(f"job-id: {rendered.job_id}", file=sys.stderr)
    print(f"attempt-id: {rendered.attempt_id}", file=sys.stderr)
    if args.output is not None:
        print(f"output: {args.output}", file=sys.stderr)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parser().parse_args()))
    except TrainingSubmitError as exc:
        raise SystemExit(f"gpu-fault-workload-annotate: {exc}") from exc


if __name__ == "__main__":
    main()
