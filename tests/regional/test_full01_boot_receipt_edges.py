"""Offline behavioral coverage for BOOT-029's identity-bound receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from scripts.e2e.regional import boot029_receipts as receipts
from tests.regional.test_acceptance_alignment_boot029_aliases import CPU_EKS, GPU_EKS
from tests.regional.test_acceptance_alignment_boot029_aliases import (
    case_documents as existing_case_documents,
)

case_documents: Callable[[Path, int], tuple[Path, Path, Path, dict[str, Any]]] = (
    existing_case_documents
)

SITE_ID = "example-boot-receipt-site"
IMAGE_ROLES = {
    "control_plane": "runtime",
    "executor": "executor",
    "node_dependencies": "node_dependencies",
}


class OwnedEcr(CommandRunner):
    """Model only the two site-owned repositories; never dispatch a command."""

    def __init__(self) -> None:
        super().__init__()
        suffix = hashlib.sha256(SITE_ID.encode()).hexdigest()[:12]
        self.names = [
            f"gpu-fault/{kind}-{suffix}" for kind in ("runtime", "runtime-cache")
        ]
        self.images: dict[str, list[dict[str, str]]] = {name: [] for name in self.names}
        self.absent: set[str] = set()
        self.overrides: dict[str, dict[str, Any]] = {}
        self.describe_error: BootstrapError | None = None
        self.keep_images = False
        self.calls: list[tuple[str, ...]] = []
        self.batches: list[tuple[str, list[dict[str, str]]]] = []

    def aws_json(
        self,
        region: str,
        *arguments: str,
        mutate: bool = False,
        sensitive: bool = False,
    ) -> dict[str, Any]:
        assert region == "us-west-2", "ECR requests escaped the CPU ARN's region"
        assert arguments[0] == "ecr", "the receipt attempted a non-ECR operation"
        self.calls.append(arguments)
        operation = arguments[1]
        if operation in self.overrides and operation != "batch-delete-image":
            return copy.deepcopy(self.overrides[operation])
        if operation == "list-tags-for-resource":
            assert arguments[2] == "--resource-arn", "tags must bind the repository ARN"
            assert arguments[3].rsplit("repository/", 1)[1] in self.names, (
                "tag discovery escaped the site-owned repositories"
            )
            return {"tags": [{"Key": "gpu-fault:site-id", "Value": SITE_ID}]}
        name = arguments[3]
        assert name in self.images, "an operation targeted an unrelated repository"
        if operation == "describe-repositories":
            if self.describe_error is not None:
                raise self.describe_error
            if name in self.absent:
                raise BootstrapError("RepositoryNotFoundException")
            return {
                "repositories": [
                    {
                        "repositoryName": name,
                        "repositoryArn": (
                            "arn:aws:ecr:us-west-2:123456789012:repository/" + name
                        ),
                        "registryId": "123456789012",
                    }
                ]
            }
        if operation == "list-images":
            return {"imageIds": copy.deepcopy(self.images[name])}
        if operation == "batch-delete-image":
            assert arguments[4] == "--image-ids", "deletion lost its explicit image IDs"
            batch = json.loads(arguments[5])
            assert batch and all(item in self.images[name] for item in batch), (
                "deletion must use the observed inventory"
            )
            self.batches.append((name, copy.deepcopy(batch)))
            if not self.keep_images:
                self.images[name] = [
                    item for item in self.images[name] if item not in batch
                ]
            return copy.deepcopy(
                self.overrides.get(operation, {"failures": [], "imageIds": batch})
            )
        pytest.fail(f"unmodeled ECR operation: {operation}")


@pytest.fixture
def ecr() -> OwnedEcr:
    return OwnedEcr()


def image_ids(count: int) -> list[dict[str, str]]:
    return [{"imageDigest": f"sha256:{number:064x}"} for number in range(count)]


def test_empty_site_does_not_parse_an_arn_or_discover_repositories(
    ecr: OwnedEcr,
) -> None:
    assert receipts.clear_site_ecr_images("not-an-arn", "", runner=ecr) == {
        "repositories": 0,
        "deleted": 0,
    }
    assert ecr.calls == [], "an absent site identity must not authorize any ECR I/O"


@pytest.mark.parametrize("absent", [False, True], ids=["owned-empty", "not-found"])
def test_empty_or_missing_repositories_need_no_delete(
    ecr: OwnedEcr, absent: bool
) -> None:
    if absent:
        ecr.absent.update(ecr.names)
    result = receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert result == {"repositories": 0 if absent else len(ecr.names), "deleted": 0}
    assert ecr.batches == [], "empty repositories must not generate delete requests"
    for name in ecr.names:
        operations = [args[1] for args in ecr.calls if args[3].endswith(name)]
        assert operations == (
            ["describe-repositories"]
            if absent
            else [
                "describe-repositories",
                "list-tags-for-resource",
                "list-images",
                "list-images",
            ]
        ), "each present repository needs an ownership check and an empty reread"


def test_image_cleanup_batches_the_observed_ids_and_confirms_both_repositories(
    ecr: OwnedEcr,
) -> None:
    inventory = {ecr.names[0]: image_ids(205), ecr.names[1]: image_ids(103)}
    ecr.images = copy.deepcopy(inventory)
    result = receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert result == {
        "repositories": len(inventory),
        "deleted": sum(map(len, inventory.values())),
    }
    for name, expected in inventory.items():
        batches = [batch for repository, batch in ecr.batches if repository == name]
        assert all(0 < len(batch) <= 100 for batch in batches), (
            "ECR deletion exceeded the service's batch limit"
        )
        assert [item for batch in batches for item in batch] == expected, (
            "batched deletion omitted, duplicated, or invented an image ID"
        )
        assert ecr.images[name] == [], "the modeled repository still contains images"
        assert [args for args in ecr.calls if args[3].endswith(name)][-1][1] == (
            "list-images"
        ), "delete acknowledgement alone must not prove an empty cache"


@pytest.mark.parametrize(
    "damage", ["not-list", "empty", "duplicate", "name", "arn", "account"]
)
def test_repository_identity_must_be_complete_before_tag_or_image_io(
    ecr: OwnedEcr, damage: str
) -> None:
    repository = ecr.aws_json(
        "us-west-2", "ecr", "describe-repositories", "--repository-names", ecr.names[0]
    )["repositories"][0]
    ecr.calls.clear()
    values: object = [repository]
    if damage == "not-list":
        values = {}
    elif damage == "empty":
        values = []
    elif damage == "duplicate":
        values = [repository, repository]
    else:
        field = {
            "name": "repositoryName",
            "arn": "repositoryArn",
            "account": "registryId",
        }[damage]
        repository[field] = "foreign"
    ecr.overrides["describe-repositories"] = {"repositories": values}
    with pytest.raises(ValueError, match="ownership is incomplete"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert [args[1] for args in ecr.calls] == ["describe-repositories"], (
        "ambiguous ownership must stop before tag discovery or deletion"
    )


@pytest.mark.parametrize(
    "tags", [None, [], [{"Key": "gpu-fault:site-id", "Value": "other"}]]
)
def test_missing_or_foreign_site_tags_refuse_image_discovery(
    ecr: OwnedEcr, tags: object
) -> None:
    ecr.overrides["list-tags-for-resource"] = {"tags": tags}
    with pytest.raises(BootstrapError, match="without|belongs to site"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert [args[1] for args in ecr.calls] == [
        "describe-repositories",
        "list-tags-for-resource",
    ], "site-tag rejection must happen before image discovery"


def test_repository_read_failure_is_not_absence(ecr: OwnedEcr) -> None:
    ecr.describe_error = BootstrapError("AccessDeniedException")
    with pytest.raises(BootstrapError, match="AccessDenied"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert ecr.batches == [], "a failed describe cannot authorize cleanup"


@pytest.mark.parametrize(
    "inventory",
    [None, {}, [None], [{}], [{"imageDigest": "sha256:" + "A" * 64}]],
    ids=["missing", "object", "non-object-image", "no-digest", "invalid-digest"],
)
def test_incomplete_image_inventory_never_reaches_delete(
    ecr: OwnedEcr, inventory: object
) -> None:
    ecr.overrides["list-images"] = {"imageIds": inventory}
    with pytest.raises(ValueError, match="inventory is incomplete"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert ecr.batches == [], "malformed image IDs must not be submitted for deletion"


@pytest.mark.parametrize(
    "response",
    [
        {"imageIds": image_ids(1)},
        {"failures": [{"failureCode": "ImageNotFound"}], "imageIds": image_ids(1)},
        {"failures": [], "imageIds": None},
        {"failures": [], "imageIds": []},
    ],
    ids=["no-failures-field", "reported-failure", "invalid-ack", "short-ack"],
)
def test_unconfirmed_delete_stops_without_claiming_cache_clear(
    ecr: OwnedEcr, response: dict[str, Any]
) -> None:
    ecr.images[ecr.names[0]] = image_ids(1)
    ecr.overrides["batch-delete-image"] = response
    with pytest.raises(ValueError, match="deletion was not confirmed"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert ecr.calls[-1][1] == "batch-delete-image", (
        "an unconfirmed delete must stop before touching the sibling repository"
    )
    assert ecr.batches == [(ecr.names[0], image_ids(1))]


def test_acknowledged_delete_is_rejected_when_the_cache_is_still_nonempty(
    ecr: OwnedEcr,
) -> None:
    ecr.images[ecr.names[0]] = image_ids(1)
    ecr.keep_images = True
    with pytest.raises(ValueError, match="cache is not empty"):
        receipts.clear_site_ecr_images(CPU_EKS, SITE_ID, runner=ecr)
    assert ecr.calls[-1][1] == "list-images", "cleanup must verify post-delete reality"
    assert ecr.images[ecr.names[0]] == image_ids(1)


def test_receipt_reader_rejects_a_json_array_without_rewriting_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        receipts.read(path)
    assert path.read_text(encoding="utf-8") == "[]"


def test_running_stage_requires_reconciliation_before_resume(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipts.initialize(path, inputs={"cpu": CPU_EKS}, stage=1)
    receipts.start_stage(path, state=tmp_path / "state", stage=1)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no prior command exit receipt"):
        receipts.initialize(path, inputs={"cpu": CPU_EKS}, stage=1)
    assert path.read_bytes() == before, "resume refusal changed the in-flight receipt"


def test_invalid_stage_receipt_cannot_be_replaced_by_a_new_attempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    write_json_atomic(path, {"stages": {"2": []}})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="stage receipt is invalid"):
        receipts.start_stage(path, state=tmp_path / "state", stage=2)
    assert path.read_bytes() == before, "malformed stage evidence was overwritten"


@pytest.mark.parametrize("stage", [1, 5])
def test_first_deploy_cannot_take_over_an_existing_bootstrap(
    tmp_path: Path, stage: int
) -> None:
    path = tmp_path / "receipt.json"
    state = tmp_path / "state"
    receipts.initialize(path, inputs={}, stage=1)
    bootstrap = state / "bootstrap-state.json"
    write_json_atomic(bootstrap, {"phase": "site-ready", "attempt": "other"})
    before = bootstrap.read_bytes()
    with pytest.raises(ValueError, match="already holds bootstrap"):
        receipts.start_stage(path, state=state, stage=stage)
    assert bootstrap.read_bytes() == before, "the pre-existing bootstrap was changed"
    assert receipts.read(path)["stages"] == {}


def test_stage_baseline_contains_only_journal_identity_and_keeps_its_first_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    state = tmp_path / "state"
    journal = state / "join-cluster/gpu/state.json"
    value = {"phase": "DISCOVERED", "attempt": 3, "private_path": "/example/private"}
    write_json_atomic(journal, value)
    receipts.initialize(path, inputs={}, stage=1)
    first = receipts.start_stage(path, state=state, stage=3)
    expected = {
        "join-cluster/gpu/state.json": {
            "sha256": hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest(),
            "phase": "DISCOVERED",
            "attempt": 3,
        }
    }
    assert first["journal_baseline"] == expected
    write_json_atomic(journal, {**value, "phase": "COMPLETED", "attempt": 4})
    second = receipts.start_stage(path, state=state, stage=3)
    assert second["journal_baseline"] == expected, (
        "retry replaced its original baseline"
    )
    assert second["attempts"] == first["attempts"] + 1
    assert "/example/private" not in path.read_text(encoding="utf-8"), (
        "private journal paths leaked into the receipt"
    )


@pytest.mark.parametrize("stage", [1, 5])
@pytest.mark.parametrize("damage", ["none", "missing", "phase", "cpu", "gpu"])
def test_bootstrap_completion_requires_the_current_cpu_and_gpu(
    tmp_path: Path, stage: int, damage: str
) -> None:
    path, state = tmp_path / "receipt.json", tmp_path / "state"
    receipts.initialize(path, inputs={}, stage=1)
    receipts.start_stage(path, state=state, stage=stage)
    value: dict[str, Any] = {
        "phase": "site-ready",
        "attempt_id": "example-deploy-attempt",
        "resources": {
            "initial_deploy_target": {
                "cpu": {"eks_arn": CPU_EKS},
                "gpu_clusters": [{"eks_arn": GPU_EKS}],
            }
        },
    }
    target = value["resources"]["initial_deploy_target"]
    if damage == "phase":
        value["phase"] = "failed"
    elif damage == "cpu":
        target["cpu"] = {"eks_arn": GPU_EKS}
    elif damage == "gpu":
        target["gpu_clusters"] = [{"eks_arn": CPU_EKS}]
    journal = state / "bootstrap-state.json"
    if damage != "missing":
        write_json_atomic(journal, value)
    if damage == "none":
        result = receipts.verify_journal(
            path, state=state, stage=stage, gpu_arn=GPU_EKS, cpu_arn=CPU_EKS
        )
        assert result == {
            "phase": "site-ready",
            "journal_path_sha256": hashlib.sha256(b"bootstrap-state.json").hexdigest(),
            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
            "attempt_sha256": hashlib.sha256(b"example-deploy-attempt").hexdigest(),
        }
    else:
        with pytest.raises(ValueError, match="exactly one current, target-bound"):
            receipts.verify_journal(
                path, state=state, stage=stage, gpu_arn=GPU_EKS, cpu_arn=CPU_EKS
            )


@pytest.mark.parametrize("stage", [2, 3, 4])
@pytest.mark.parametrize("completed", [False, True], ids=["unfinished", "completed"])
def test_lifecycle_completion_accepts_direct_eks_identity_only_after_completion(
    tmp_path: Path, stage: int, completed: bool
) -> None:
    record, state, journal, value = case_documents(tmp_path, stage)
    receipts.start_stage(record, state=state, stage=stage)
    value["phase"] = "COMPLETED" if completed else "DISCOVERED"
    if stage == 3:
        value["gpu_cluster_arn"] = GPU_EKS
    write_json_atomic(journal, value)
    if completed:
        result = receipts.verify_journal(
            record, state=state, stage=stage, gpu_arn=GPU_EKS, cpu_arn=CPU_EKS
        )
        assert (
            result["journal_sha256"] == hashlib.sha256(journal.read_bytes()).hexdigest()
        )
        assert result["phase"] == "COMPLETED"
    else:
        with pytest.raises(ValueError, match="exactly one"):
            receipts.verify_journal(
                record, state=state, stage=stage, gpu_arn=GPU_EKS, cpu_arn=CPU_EKS
            )


def test_join_completion_for_another_gpu_is_not_a_witness(tmp_path: Path) -> None:
    record, state, journal, value = case_documents(tmp_path, 3)
    receipts.start_stage(record, state=state, stage=3)
    write_json_atomic(journal, value)
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            record, state=state, stage=3, gpu_arn=CPU_EKS, cpu_arn=CPU_EKS
        )


@pytest.mark.parametrize(
    "change", [{"cpu_disposition": "delete"}, {"reset_database": False}]
)
def test_uninstall_witness_requires_keep_and_database_reset(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    record, state, journal, value = case_documents(tmp_path, 4)
    receipts.start_stage(record, state=state, stage=4)
    write_json_atomic(journal, {**value, **change})
    with pytest.raises(ValueError, match="exactly one"):
        receipts.verify_journal(
            record, state=state, stage=4, gpu_arn=GPU_EKS, cpu_arn=CPU_EKS
        )


@dataclass(frozen=True)
class ColdBuild:
    state: Path
    manifest: Path
    descriptor: Path
    log: Path


@pytest.fixture
def cold_build(tmp_path: Path) -> ColdBuild:
    files = ColdBuild(
        state=tmp_path / "state",
        manifest=tmp_path / "release/current-release.json",
        descriptor=tmp_path / "release/release-runtime-image.json",
        log=tmp_path / "build.log",
    )
    images = {
        role: {
            "registry_reused": False,
            "deployable": True,
            "image_input_sha256": hashlib.sha256(role.encode()).hexdigest(),
            "source_identity_sha256": "b" * 64,
            "reference": f"example.invalid/{role}@sha256:{index:064x}",
        }
        for index, role in enumerate(IMAGE_ROLES)
    }
    write_json_atomic(
        files.manifest,
        {
            "delivery": {
                "images": {
                    released: {"reference": images[role]["reference"]}
                    for role, released in IMAGE_ROLES.items()
                }
            }
        },
    )
    write_json_atomic(
        files.descriptor,
        {"schema_version": 3, "source_identity_sha256": "b" * 64, "images": images},
    )
    write_json_atomic(
        files.state / "bootstrap-state.json",
        {"resources": {"release": {"manifest": str(files.manifest)}}},
    )
    files.log.write_text(
        "#1 extracting sha256:abc\n#2 [2/3] COPY . .\n", encoding="utf-8"
    )
    return files


@pytest.mark.parametrize("manifest", ["", "relative/current-release.json"])
def test_cold_build_requires_an_absolute_bootstrap_manifest(
    cold_build: ColdBuild, manifest: str
) -> None:
    write_json_atomic(
        cold_build.state / "bootstrap-state.json",
        {"resources": {"release": {"manifest": manifest}}},
    )
    with pytest.raises(ValueError, match="bootstrap-bound release manifest"):
        receipts.cold_build_proof(cold_build.state, [cold_build.log])


@pytest.mark.parametrize("damage", ["schema", "shape", "missing-role", "extra-role"])
def test_cold_build_requires_the_complete_current_descriptor_shape(
    cold_build: ColdBuild, damage: str
) -> None:
    value = receipts.read(cold_build.descriptor)
    if damage == "schema":
        value["schema_version"] = 2
    elif damage == "shape":
        value["images"] = []
    elif damage == "missing-role":
        value["images"].pop("executor")
    else:
        value["images"]["unexpected"] = value["images"]["executor"]
    write_json_atomic(cold_build.descriptor, value)
    with pytest.raises(ValueError, match="all three component image descriptors"):
        receipts.cold_build_proof(cold_build.state, [cold_build.log])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registry_reused", True),
        ("deployable", False),
        ("image_input_sha256", "not-a-digest"),
        ("source_identity_sha256", "c" * 64),
        ("reference", "example.invalid/foreign@sha256:" + "c" * 64),
    ],
)
def test_cold_build_rejects_an_image_not_bound_to_the_fresh_release(
    cold_build: ColdBuild, field: str, value: object
) -> None:
    descriptor = receipts.read(cold_build.descriptor)
    descriptor["images"]["executor"][field] = value
    write_json_atomic(cold_build.descriptor, descriptor)
    with pytest.raises(ValueError, match="fresh, release-bound executor"):
        receipts.cold_build_proof(cold_build.state, [cold_build.log])


@pytest.mark.parametrize(
    "text",
    ["#1 extracting sha256:abc\n#2 CACHED\n", "#1 [1/2] FROM base\n", ""],
    ids=["cached-layer", "no-base-extraction", "empty-log"],
)
def test_cold_build_needs_a_base_pull_and_no_cached_layers(
    cold_build: ColdBuild, text: str
) -> None:
    cold_build.log.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="CACHED layers or has no base-image pull"):
        receipts.cold_build_proof(cold_build.state, [cold_build.log])


def invoke(
    monkeypatch: pytest.MonkeyPatch, record: Path, state: Path, *args: str
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["boot029", *args, "--record", str(record), "--state", str(state)]
    )
    receipts.main()


def test_cli_initializes_starts_finishes_and_verifies_owned_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record, state = tmp_path / "receipt.json", tmp_path / "state"
    repository = tmp_path / "source"
    queries: list[Path] = []

    def source_identity(path: Path) -> dict[str, str]:
        queries.append(path)
        return {"sha256": "e" * 64}

    monkeypatch.setattr(receipts, "build_release_identity", source_identity)
    invoke(
        monkeypatch,
        record,
        state,
        "init",
        "--stage",
        "1",
        "--repo",
        str(repository),
        "--second-state",
        str(tmp_path / "second"),
        "--cpu-arn",
        CPU_EKS,
        "--gpu-arn",
        GPU_EKS,
        "--email",
        "operator@example.invalid",
        "--email-wait",
        "45",
    )
    initial = receipts.read(record)
    assert queries == [repository], "CLI source identity queried a different checkout"
    assert initial["inputs"]["source_identity_sha256"] == (
        hashlib.sha256(("e" * 64).encode()).hexdigest()
    )
    assert initial["inputs"]["email_wait_sha256"] == hashlib.sha256(b"45").hexdigest()
    assert "aurora_final_snapshot_sha256" not in initial["inputs"], (
        "generic receipt callers must not acquire an implicit snapshot policy"
    )
    invoke(monkeypatch, record, state, "start", "--stage", "1")
    assert capsys.readouterr().out.strip() == "1"
    bootstrap = state / "bootstrap-state.json"
    write_json_atomic(
        bootstrap,
        {
            "phase": "site-ready",
            "attempt_id": "example-first",
            "resources": {
                "initial_deploy_target": {
                    "cpu": {"eks_arn": CPU_EKS},
                    "gpu_clusters": [{"eks_arn": GPU_EKS}],
                }
            },
        },
    )
    invoke(
        monkeypatch,
        record,
        state,
        "verify",
        "--stage",
        "1",
        "--cpu-arn",
        CPU_EKS,
        "--gpu-arn",
        GPU_EKS,
    )
    assert json.loads(capsys.readouterr().out)["journal_sha256"] == (
        hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    )
    log = tmp_path / "first.log"
    log.write_text("owned command completed\n", encoding="utf-8")
    log.with_name(log.name + ".extra.json").write_text(
        '{"build":{"fresh":true}}\n{"cleanup":{"verified":true}}\n', encoding="utf-8"
    )
    invoke(
        monkeypatch,
        record,
        state,
        "finish",
        "--stage",
        "1",
        "--status",
        "PASS",
        "--seconds",
        "17",
        "--log",
        str(log),
        "--name",
        "first",
    )
    stage = receipts.read(record)["stages"]["1"]
    assert stage["status"] == "PASS" and stage["failure"] is None
    assert stage["wall_seconds"] == 17 and stage["name"] == "first"
    assert stage["build"] == {"fresh": True}
    assert stage["cleanup"] == {"verified": True}
    assert stage["attempt_logs"] == [
        {"name": log.name, "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    ], "finish lost the actual command log's identity"


@pytest.mark.parametrize("policy", ["retain", "skip"])
def test_cli_snapshot_policy_binding_is_optional_immutable_and_not_auto_upgraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
) -> None:
    state = tmp_path / "state"
    monkeypatch.setattr(
        receipts, "build_release_identity", lambda _path: {"sha256": "e" * 64}
    )
    arguments = (
        "init",
        "--stage",
        "1",
        "--repo",
        str(tmp_path / "source"),
        "--second-state",
        str(tmp_path / "second"),
        "--cpu-arn",
        CPU_EKS,
        "--gpu-arn",
        GPU_EKS,
    )
    generic = tmp_path / "generic.json"
    invoke(monkeypatch, generic, state, *arguments)
    original = generic.read_bytes()
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch, generic, state, *arguments, "--aurora-final-snapshot", policy
        )
    assert raised.value.code == 1 and "input identity" in capsys.readouterr().err, (
        "adding a policy must not upgrade a generic receipt in place"
    )
    assert generic.read_bytes() == original, "legacy input approval must remain intact"
    invoke(monkeypatch, generic, state, *arguments)
    assert generic.read_bytes() == original, (
        "omitting the option must preserve the existing generic init behavior"
    )

    record = tmp_path / "policy.json"
    invoke(monkeypatch, record, state, *arguments, "--aurora-final-snapshot", policy)
    bound = receipts.read(record)
    assert bound["schema_version"] == 2, "optional input must not change receipt schema"
    assert bound["inputs"] == {
        **json.loads(original)["inputs"],
        "aurora_final_snapshot_sha256": hashlib.sha256(policy.encode()).hexdigest(),
    }, "policy binding must extend the single existing CLI input mapping"
    before = record.read_bytes()
    for changed in (
        (),
        ("--aurora-final-snapshot", "skip" if policy == "retain" else "retain"),
    ):
        with pytest.raises(SystemExit) as raised:
            invoke(monkeypatch, record, state, *arguments, *changed)
        assert raised.value.code == 1 and "input identity" in capsys.readouterr().err, (
            "neither omission nor a changed value may discard the original approval"
        )
        assert record.read_bytes() == before, "refusal cannot rewrite the receipt"
    invoke(monkeypatch, record, state, *arguments, "--aurora-final-snapshot", policy)
    assert record.read_bytes() == before, "same-policy init must remain repeatable"


def uninstall_policy_receipt(
    tmp_path: Path, policy: str | None
) -> tuple[Path, Path, Path]:
    record, state = tmp_path / "receipt.json", tmp_path / "state"
    inputs = {"cpu_cluster_arn": CPU_EKS, "gpu_cluster_arn": GPU_EKS}
    if policy is not None:
        inputs["aurora_final_snapshot"] = policy
    receipts.initialize(record, inputs=inputs, stage=1)
    receipts.start_stage(record, state=state, stage=4)
    journal = state / "uninstall/state.json"
    value = {
        "phase": "COMPLETED",
        "attempt_id": "a" * 32,
        "cpu_disposition": "keep",
        "reset_database": True,
        "site_identity": {"cpu_eks_arn": CPU_EKS},
    }
    if policy is not None:
        value["final_snapshot_policy"] = policy
    write_json_atomic(journal, value)
    return record, state, journal


@pytest.mark.parametrize(
    "approved,reported",
    [
        ("retain", "retain"),
        ("skip", "skip"),
        ("retain", "skip"),
        ("skip", "retain"),
        ("retain", None),
    ],
)
def test_cli_policy_aware_verify_preserves_generic_proof_and_checks_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    approved: str,
    reported: str | None,
) -> None:
    record, state, journal = uninstall_policy_receipt(tmp_path, approved)
    value = receipts.read(journal)
    if reported is None:
        value.pop("final_snapshot_policy")
    else:
        value["final_snapshot_policy"] = reported
    write_json_atomic(journal, value)
    before = record.read_bytes()
    arguments = ("verify", "--stage", "4", "--cpu-arn", CPU_EKS, "--gpu-arn", GPU_EKS)
    invoke(monkeypatch, record, state, *arguments)
    generic = json.loads(capsys.readouterr().out)
    assert set(generic) == {
        "phase",
        "journal_path_sha256",
        "journal_sha256",
        "attempt_sha256",
    }, "generic verification must keep its original evidence shape and scope"
    if approved != reported:
        with pytest.raises(SystemExit) as raised:
            invoke(
                monkeypatch,
                record,
                state,
                *arguments,
                "--aurora-final-snapshot",
                approved,
            )
        output = capsys.readouterr()
        assert raised.value.code == 1 and "approved snapshot policy" in output.err, (
            "wrong or missing journal policy must refuse explicit verification"
        )
        assert output.out == "", "a rejected policy must not emit completion evidence"
    else:
        invoke(
            monkeypatch, record, state, *arguments, "--aurora-final-snapshot", approved
        )
        expected = {**generic, "aurora_final_snapshot": approved}
        if approved == "skip":
            expected["companion_cases"] = {
                "GF-REGIONAL-BOOT-027": {
                    "status": "NOT_RUN",
                    "reason": "Aurora final snapshot explicitly skipped",
                }
            }
        assert json.loads(capsys.readouterr().out) == expected, (
            "explicit verification must preserve journal identity and companion scope"
        )
    assert record.read_bytes() == before, "verification must not alter an approval"


@pytest.mark.parametrize("bound", [None, "skip"])
def test_explicit_cli_verify_requires_matching_immutable_receipt_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bound: str | None,
) -> None:
    record, state, journal = uninstall_policy_receipt(tmp_path, bound)
    value = receipts.read(journal)
    value["final_snapshot_policy"] = "retain"
    write_json_atomic(journal, value)
    before = record.read_bytes()

    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            record,
            state,
            "verify",
            "--stage",
            "4",
            "--cpu-arn",
            CPU_EKS,
            "--gpu-arn",
            GPU_EKS,
            "--aurora-final-snapshot",
            "retain",
        )

    output = capsys.readouterr()
    assert raised.value.code == 1 and "immutable receipt approval" in output.err, (
        "the journal alone cannot supply or change the receipt's policy approval"
    )
    assert output.out == "", "unapproved policy must not produce a completion proof"
    assert record.read_bytes() == before, "verification cannot upgrade an old receipt"


@pytest.mark.parametrize(
    "stage,policy",
    [(1, "retain"), (2, "skip"), (3, "retain"), (5, "skip"), (4, "bogus")],
)
def test_snapshot_verification_rejects_wrong_stage_or_policy_before_reading(
    tmp_path: Path, stage: int, policy: str
) -> None:
    with pytest.raises(ValueError, match="requires stage 4 and retain"):
        receipts.verify_journal(
            tmp_path / "missing.json",
            state=tmp_path / "state",
            stage=stage,
            cpu_arn=CPU_EKS,
            gpu_arn=GPU_EKS,
            aurora_final_snapshot=policy,
        )
    assert not (tmp_path / "missing.json").exists(), (
        "invalid policy verification cannot initialize replacement evidence"
    )


@pytest.mark.parametrize("action", ["init", "verify"])
@pytest.mark.parametrize("policy", ["bogus", ""])
def test_cli_snapshot_policy_choices_refuse_unknown_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    policy: str,
) -> None:
    record = tmp_path / "receipt.json"
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            record,
            tmp_path / "state",
            action,
            "--stage",
            "4",
            "--aurora-final-snapshot",
            policy,
        )
    assert raised.value.code == 2 and "invalid choice" in capsys.readouterr().err, (
        "the receipt CLI must reject unknown policy values as usage errors"
    )
    assert not record.exists(), "invalid policy cannot create a receipt"


def test_cli_snapshot_policy_verification_keeps_the_exact_journal_byte_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record, state, journal = uninstall_policy_receipt(tmp_path, "retain")
    before = record.read_bytes()
    original_read = Path.read_bytes
    journal_reads = 0

    def read_bytes(path: Path) -> bytes:
        nonlocal journal_reads
        data = original_read(path)
        if path == journal:
            journal_reads += 1
            if journal_reads == 2:
                return data + b" "
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            record,
            state,
            "verify",
            "--stage",
            "4",
            "--cpu-arn",
            CPU_EKS,
            "--gpu-arn",
            GPU_EKS,
            "--aurora-final-snapshot",
            "retain",
        )
    output = capsys.readouterr()
    assert raised.value.code == 1 and "approved snapshot policy" in output.err, (
        "semantic policy equality cannot replace the original journal-byte proof"
    )
    assert journal_reads == 2 and output.out == "", (
        "changed bytes must refuse before any completion evidence is emitted"
    )
    assert record.read_bytes() == before, "failed verification must remain read-only"


@pytest.mark.parametrize(
    "arguments",
    [
        ("init",),
        ("init", "--repo", "example-source"),
        ("cold",),
        ("finish",),
        ("finish", "--status", "PASS"),
        ("finish", "--log", "example.log"),
    ],
    ids=[
        "init-no-inputs",
        "init-no-second-state",
        "cold-no-log",
        "finish-no-inputs",
        "finish-no-log",
        "finish-no-status",
    ],
)
def test_cli_missing_action_inputs_fail_before_creating_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
) -> None:
    record, state = tmp_path / "receipt.json", tmp_path / "state"
    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, record, state, *arguments, "--stage", "1")
    assert raised.value.code == 2
    assert "requires" in capsys.readouterr().err
    assert not record.exists(), "usage failure created an acceptance record"


def test_cli_refuses_missing_record_with_a_nonzero_receipt_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        invoke(
            monkeypatch,
            tmp_path / "missing.json",
            tmp_path / "state",
            "start",
            "--stage",
            "1",
        )
    assert raised.value.code == 1
    assert "BOOT-029 receipt refused:" in capsys.readouterr().err


def test_cli_ecr_cleanup_uses_the_owned_command_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ecr: OwnedEcr,
) -> None:
    ecr.images[ecr.names[0]] = image_ids(2)
    monkeypatch.setattr(receipts, "CommandRunner", lambda: ecr)
    invoke(
        monkeypatch,
        tmp_path / "record.json",
        tmp_path / "state",
        "clear-ecr",
        "--stage",
        "1",
        "--cpu-arn",
        CPU_EKS,
        "--previous-site-id",
        SITE_ID,
    )
    assert json.loads(capsys.readouterr().out) == {"repositories": 2, "deleted": 2}
    assert ecr.batches == [(ecr.names[0], image_ids(2))]
    assert not (tmp_path / "record.json").exists(), (
        "cache cleanup must not manufacture stage completion"
    )


@pytest.mark.parametrize("history", ["none", "valid", "edited", "traversal", "cached"])
def test_cli_cold_proof_includes_every_unchanged_prior_attempt_log(
    tmp_path: Path,
    cold_build: ColdBuild,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    history: str,
) -> None:
    record = tmp_path / "receipt.json"
    previous = tmp_path / "previous.log"
    previous.write_text(
        "#3 CACHED\n" if history == "cached" else "#3 previous build interrupted\n",
        encoding="utf-8",
    )
    receipts.initialize(record, inputs={}, stage=1)
    receipts.start_stage(record, state=tmp_path / "new-state", stage=1)
    if history != "none":
        receipts.finish(
            record, stage=1, status="FAIL", seconds=2, log=previous, name="first"
        )
    if history == "edited":
        previous.write_text("rewritten prior attempt\n", encoding="utf-8")
    elif history == "traversal":
        value = receipts.read(record)
        value["stages"]["1"]["attempt_logs"][0]["name"] = "../previous.log"
        write_json_atomic(record, value)
    arguments = ("cold", "--stage", "1", "--log", str(cold_build.log))
    if history in {"edited", "traversal", "cached"}:
        with pytest.raises(SystemExit) as raised:
            invoke(monkeypatch, record, cold_build.state, *arguments)
        output = capsys.readouterr()
        assert raised.value.code == 1
        assert output.out == "", "invalid history produced a cold-build proof"
        assert (
            "CACHED layers" if history == "cached" else "attempt log changed"
        ) in output.err
    else:
        invoke(monkeypatch, record, cold_build.state, *arguments)
        proof = json.loads(capsys.readouterr().out)["cold_build"]
        assert proof["cached_layers"] == 0 and proof["base_image_pulled"] is True
        assert proof["descriptor_sha256"] == (
            hashlib.sha256(cold_build.descriptor.read_bytes()).hexdigest()
        )
        assert set(proof["images"]) == set(IMAGE_ROLES)
        descriptor = receipts.read(cold_build.descriptor)
        for role, image in proof["images"].items():
            assert (
                image["reference_sha256"]
                == hashlib.sha256(
                    descriptor["images"][role]["reference"].encode()
                ).hexdigest()
            ), "cold proof did not bind the released image reference"


def test_receipt_script_entrypoint_enforces_required_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["boot029-receipts", "start"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(receipts.__file__), run_name="__main__")
    assert raised.value.code == 2
    assert "--record" in capsys.readouterr().err
