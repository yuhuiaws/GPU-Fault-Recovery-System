"""Identity-bound local receipts for the ordered BOOT-029 shell driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    Arn,
    CommandRunner,
    assert_site_tag,
    describe_or_absent,
)
from scripts.release_identity import build_release_identity

CASE_ID = "GF-REGIONAL-BOOT-029"


def clear_site_ecr_images(
    cpu_arn: str, site_id: str, *, runner: CommandRunner | None = None
) -> dict[str, Any]:
    if not site_id:
        return {"repositories": 0, "deleted": 0}
    cpu = Arn.parse(cpu_arn)
    active = runner or CommandRunner()
    suffix = digest(site_id.encode())[:12]
    deleted = repositories = 0
    for kind in ("runtime", "runtime-cache"):
        name = f"gpu-fault/{kind}-{suffix}"
        response = describe_or_absent(
            active,
            cpu.region,
            "ecr",
            "describe-repositories",
            "--repository-names",
            name,
            not_found=("RepositoryNotFoundException",),
        )
        if response is None:
            continue
        values = response.get("repositories")
        expected_arn = (
            f"arn:{cpu.partition}:ecr:{cpu.region}:{cpu.account}:repository/{name}"
        )
        if (
            not isinstance(values, list)
            or len(values) != 1
            or values[0].get("repositoryName") != name
            or values[0].get("repositoryArn") != expected_arn
            or values[0].get("registryId") != cpu.account
        ):
            raise ValueError("ECR repository ownership is incomplete")
        tags = active.aws_json(
            cpu.region, "ecr", "list-tags-for-resource", "--resource-arn", expected_arn
        )
        assert_site_tag(tags.get("tags") or [], site_id=site_id, description=name)
        response = active.aws_json(
            cpu.region, "ecr", "list-images", "--repository-name", name
        )
        images = response.get("imageIds")
        if not isinstance(images, list) or any(
            not isinstance(item, dict)
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(item.get("imageDigest")))
            for item in images
        ):
            raise ValueError("ECR image inventory is incomplete")
        for start in range(0, len(images), 100):
            batch = images[start : start + 100]
            result = active.aws_json(
                cpu.region,
                "ecr",
                "batch-delete-image",
                "--repository-name",
                name,
                "--image-ids",
                json.dumps(batch),
            )
            if (
                result.get("failures") != []
                or not isinstance(result.get("imageIds"), list)
                or len(result["imageIds"]) != len(batch)
            ):
                raise ValueError("ECR image deletion was not confirmed")
            deleted += len(batch)
        remaining = active.aws_json(
            cpu.region, "ecr", "list-images", "--repository-name", name
        )
        if remaining.get("imageIds") != []:
            raise ValueError("ECR image cache is not empty")
        repositories += 1
    return {"repositories": repositories, "deleted": deleted}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt must be an existing regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("receipt must contain an object")
    return value


def initialize(path: Path, *, inputs: dict[str, str], stage: int) -> dict[str, Any]:
    expected = {
        key + "_sha256": digest(value.encode()) for key, value in inputs.items()
    }
    if path.exists():
        record = read(path)
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 2
            or record.get("case_id") != CASE_ID
            or record.get("inputs") != expected
            or not isinstance(record.get("stages"), dict)
        ):
            raise ValueError(
                "BOOT-029 input identity changed or legacy receipt is unbound"
            )
        first = next(
            (
                number
                for number in range(1, 6)
                if record["stages"].get(str(number), {}).get("status") != "PASS"
            ),
            6,
        )
        if first != stage:
            raise ValueError("BOOT-029 must resume its first unpassed stage")
        if record["stages"].get(str(stage), {}).get("status") == "RUNNING":
            raise ValueError(
                "BOOT-029 has no prior command exit receipt; reconcile before a new run"
            )
        for number in range(1, stage):
            receipt = record["stages"][str(number)]
            log = path.parent / receipt["log_name"]
            if (
                log.parent != path.parent
                or log.is_symlink()
                or digest(log.read_bytes()) != receipt["log_sha256"]
            ):
                raise ValueError("BOOT-029 predecessor log proof changed")
    else:
        if stage != 1:
            raise ValueError("BOOT-029 resume requires bound preceding PASS receipts")
        record = {
            "case_id": CASE_ID,
            "schema_version": 2,
            "inputs": expected,
            "stages": {},
        }
    record["resumed_from_stage"] = stage
    write_json_atomic(path, record)
    return record


def journal_paths(state: Path, stage: int) -> list[Path]:
    if stage in {1, 5}:
        return [state / "bootstrap-state.json"]
    if stage == 4:
        return [state / "uninstall/state.json"]
    directory = "remove-cluster" if stage == 2 else "join-cluster"
    return sorted((state / directory).glob("*/state.json"))


def start_stage(path: Path, *, state: Path, stage: int) -> dict[str, Any]:
    record = read(path)
    receipt = record["stages"].setdefault(str(stage), {})
    if not isinstance(receipt, dict):
        raise ValueError("BOOT-029 stage receipt is invalid")
    if "journal_baseline" not in receipt:
        if stage in {1, 5} and (state / "bootstrap-state.json").exists():
            raise ValueError("first deploy already holds bootstrap-state.json")
        receipt["journal_baseline"] = {
            str(item.relative_to(state)): read(item)
            for item in journal_paths(state, stage)
            if item.exists()
        }
        # A baseline journal may contain private paths; only identities and
        # digests belong in the acceptance record.
        receipt["journal_baseline"] = {
            name: {
                "sha256": digest(json.dumps(value, sort_keys=True).encode()),
                "phase": value.get("phase"),
                "attempt": value.get("attempt_id", value.get("attempt")),
            }
            for name, value in receipt["journal_baseline"].items()
        }
    receipt["attempts"] = int(receipt.get("attempts", 0)) + 1
    receipt["status"] = "RUNNING"
    write_json_atomic(path, record)
    return receipt


def _saved_hyperpod_alias(arn: str, eks_arn: object, identity: object) -> bool:
    return (
        bool(arn)
        and isinstance(eks_arn, str)
        and bool(eks_arn)
        and isinstance(identity, dict)
        and identity.get("eks_arn") == eks_arn
        and identity.get("hyperpod_arn") == arn
    )


def _uninstall_cpu_matches(state: Path, cpu_arn: str, eks_arn: object) -> bool:
    if cpu_arn and cpu_arn == eks_arn:
        return True
    bootstrap = read(state / "bootstrap-state.json")
    initial = (bootstrap.get("resources") or {}).get("initial_deploy_target") or {}
    return _saved_hyperpod_alias(cpu_arn, eks_arn, initial.get("cpu"))


def verify_journal(
    path: Path,
    *,
    state: Path,
    stage: int,
    gpu_arn: str,
    cpu_arn: str,
    aurora_final_snapshot: str | None = None,
) -> dict[str, Any]:
    if aurora_final_snapshot is not None and (
        stage != 4 or aurora_final_snapshot not in {"retain", "skip"}
    ):
        raise ValueError(
            "snapshot policy verification requires stage 4 and retain|skip"
        )
    record = read(path)
    baseline = record["stages"][str(stage)]["journal_baseline"]
    candidates: list[dict[str, Any]] = []
    for candidate in journal_paths(state, stage):
        if not candidate.exists():
            continue
        value = read(candidate)
        if stage in {1, 5}:
            target = (value.get("resources") or {}).get("initial_deploy_target") or {}
            if (
                value.get("phase") != "site-ready"
                or cpu_arn not in (target.get("cpu") or {}).values()
                or not any(
                    gpu_arn in item.values() for item in target.get("gpu_clusters", [])
                )
            ):
                continue
        else:
            target = value.get("target") or {}
            if value.get("phase") != "COMPLETED":
                continue
            if stage == 2 and gpu_arn != target.get("eks_cluster_arn"):
                discovery = (value.get("evidence") or {}).get("DISCOVERED") or {}
                if not _saved_hyperpod_alias(
                    gpu_arn,
                    target.get("eks_cluster_arn"),
                    discovery.get("provider_identity"),
                ):
                    continue
            if stage == 3 and gpu_arn != value.get("gpu_cluster_arn"):
                continue
            if stage == 4 and (
                value.get("cpu_disposition") != "keep"
                or value.get("reset_database") is not True
                or not _uninstall_cpu_matches(
                    state,
                    cpu_arn,
                    (value.get("site_identity") or {}).get("cpu_eks_arn"),
                )
            ):
                continue
        identity = str(candidate.relative_to(state))
        old = baseline.get(identity, {})
        attempt = value.get("attempt_id", value.get("attempt"))
        if old.get("phase") == "COMPLETED" and old.get("attempt") == attempt:
            continue
        candidates.append(
            {
                "phase": value["phase"],
                "journal_path_sha256": digest(identity.encode()),
                "journal_sha256": digest(candidate.read_bytes()),
                "attempt_sha256": digest(str(attempt).encode()),
            }
        )
    if len(candidates) != 1:
        raise ValueError(
            "exactly one current, target-bound completion journal is required"
        )
    proof = candidates[0]
    if aurora_final_snapshot is not None:
        inputs = record.get("inputs")
        if not isinstance(inputs, dict) or inputs.get(
            "aurora_final_snapshot_sha256"
        ) != digest(aurora_final_snapshot.encode()):
            raise ValueError(
                "snapshot policy differs from the immutable receipt approval"
            )
        journal = (state / "uninstall/state.json").read_bytes()
        if (
            digest(journal) != proof["journal_sha256"]
            or json.loads(journal).get("final_snapshot_policy") != aurora_final_snapshot
        ):
            raise ValueError(
                "uninstall journal differs from the approved snapshot policy"
            )
        proof["aurora_final_snapshot"] = aurora_final_snapshot
        if aurora_final_snapshot == "skip":
            proof["companion_cases"] = {
                "GF-REGIONAL-BOOT-027": {
                    "status": "NOT_RUN",
                    "reason": "Aurora final snapshot explicitly skipped",
                }
            }
    return proof


def foreign_cached_layers(text: str) -> tuple[int, int]:
    """Count ``CACHED`` steps that a warm cache served, and those built in-run.

    BuildKit prints ``#N CACHED`` both for a layer imported from a pre-existing
    cache and for a layer another image of the same build run has just
    produced (the executor image shares the runtime venv steps with the
    runtime image, live 2026-09-20). Only the former breaks a cold build: a
    step is in-run reuse when the same instruction already reached ``DONE``
    earlier in the same logs.
    """

    instruction = re.compile(r"^#(\d+) (\[.*)$")
    status = re.compile(r"^#(\d+) (DONE\b.*|CACHED\s*)$")
    latest: dict[str, str] = {}
    completed: set[str] = set()
    foreign = reused = 0
    for line in text.splitlines():
        found = instruction.match(line)
        if found:
            latest[found.group(1)] = found.group(2).strip()
            continue
        found = status.match(line)
        if not found:
            continue
        step = latest.get(found.group(1))
        if found.group(2).startswith("DONE"):
            if step:
                completed.add(step)
        elif step and step in completed:
            reused += 1
        else:
            foreign += 1
    return foreign, reused


def cold_build_proof(state: Path, logs: list[Path]) -> dict[str, Any]:
    bootstrap = read(state / "bootstrap-state.json")
    manifest_path = Path(
        str((bootstrap.get("resources") or {}).get("release", {}).get("manifest", ""))
    )
    if not manifest_path.is_absolute():
        raise ValueError("cold build lacks the bootstrap-bound release manifest")
    manifest = read(manifest_path)
    descriptor_path = manifest_path.parent / "release-runtime-image.json"
    descriptor = read(descriptor_path)
    expected = {
        "control_plane": "runtime",
        "executor": "executor",
        "node_dependencies": "node_dependencies",
    }
    images = descriptor.get("images")
    if (
        descriptor.get("schema_version") != 3
        or not isinstance(images, dict)
        or set(images) != set(expected)
    ):
        raise ValueError("cold build requires all three component image descriptors")
    for name, released in expected.items():
        image = images[name]
        if (
            image.get("registry_reused") is not False
            or image.get("deployable") is not True
            or not re.fullmatch(r"[a-f0-9]{64}", str(image.get("image_input_sha256")))
            or image.get("source_identity_sha256")
            != descriptor.get("source_identity_sha256")
            or image.get("reference")
            != manifest.get("delivery", {})
            .get("images", {})
            .get(released, {})
            .get("reference")
        ):
            raise ValueError(f"cold build lacks a fresh, release-bound {name} image")
    text = "\n".join(item.read_text(encoding="utf-8") for item in logs)
    cached, reused = foreign_cached_layers(text)
    pulled = bool(re.search(r"(?m)^#\d+ extracting sha256:", text))
    if cached or not pulled:
        raise ValueError(
            "cold build used CACHED layers or has no base-image pull proof"
        )
    return {
        "cold_build": {
            "cached_layers": cached,
            "reused_within_run": reused,
            "base_image_pulled": pulled,
            "descriptor_sha256": digest(descriptor_path.read_bytes()),
            "images": {
                name: {
                    "image_input_sha256": images[name]["image_input_sha256"],
                    "reference_sha256": digest(images[name]["reference"].encode()),
                    "registry_reused": False,
                }
                for name in expected
            },
        }
    }


def finish(
    path: Path, *, stage: int, status: str, seconds: int, log: Path, name: str
) -> None:
    record = read(path)
    receipt = record["stages"][str(stage)]
    receipt.update(
        name=name,
        status=status,
        wall_seconds=seconds,
        log_name=log.name,
        log_sha256=digest(log.read_bytes()),
        failure=None if status == "PASS" else "stage failed; consult its private log",
    )
    extra = log.with_name(log.name + ".extra.json")
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            receipt.update(json.loads(line))
    receipt.setdefault("attempt_logs", []).append(
        {"name": log.name, "sha256": receipt["log_sha256"]}
    )
    write_json_atomic(path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("init", "start", "verify", "cold", "finish", "clear-ecr")
    )
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--second-state", type=Path)
    parser.add_argument("--cpu-arn", default="")
    parser.add_argument("--gpu-arn", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--previous-site-id", default="")
    parser.add_argument("--email-wait", default="0")
    parser.add_argument("--aurora-final-snapshot", choices=("retain", "skip"))
    parser.add_argument("--status", choices=("PASS", "FAIL"))
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    try:
        if args.action == "init":
            if args.repo is None or args.second_state is None:
                parser.error("init requires repo and second-state")
            inputs = {
                "cpu_cluster_arn": args.cpu_arn,
                "gpu_cluster_arn": args.gpu_arn,
                "admin_email": args.email,
                "state_dir": str(args.state.resolve()),
                "second_state_dir": str(args.second_state.resolve()),
                "repository": str(args.repo.resolve()),
                "source_identity": str(build_release_identity(args.repo)["sha256"]),
                "previous_site_id": args.previous_site_id,
                "email_wait": args.email_wait,
            }
            if args.aurora_final_snapshot is not None:
                inputs["aurora_final_snapshot"] = args.aurora_final_snapshot
            initialize(
                args.record,
                inputs=inputs,
                stage=args.stage,
            )
        elif args.action == "clear-ecr":
            print(
                json.dumps(clear_site_ecr_images(args.cpu_arn, args.previous_site_id))
            )
        elif args.action == "start":
            print(
                start_stage(args.record, state=args.state, stage=args.stage)["attempts"]
            )
        elif args.action == "verify":
            print(
                json.dumps(
                    verify_journal(
                        args.record,
                        state=args.state,
                        stage=args.stage,
                        gpu_arn=args.gpu_arn,
                        cpu_arn=args.cpu_arn,
                        aurora_final_snapshot=args.aurora_final_snapshot,
                    )
                )
            )
        elif args.action == "cold":
            if args.log is None:
                parser.error("cold requires log")
            receipt = read(args.record)["stages"][str(args.stage)]
            logs = []
            for item in receipt.get("attempt_logs", []):
                previous = args.record.parent / item["name"]
                if (
                    previous.parent != args.record.parent
                    or digest(previous.read_bytes()) != item["sha256"]
                ):
                    raise ValueError("cold build attempt log changed")
                logs.append(previous)
            print(json.dumps(cold_build_proof(args.state, [*logs, args.log])))
        else:
            if args.log is None or args.status is None:
                parser.error("finish requires log and status")
            finish(
                args.record,
                stage=args.stage,
                status=args.status,
                seconds=args.seconds,
                log=args.log,
                name=args.name,
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"BOOT-029 receipt refused: {exc}\n")


if __name__ == "__main__":
    main()
