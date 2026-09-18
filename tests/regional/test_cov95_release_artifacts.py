from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_artifacts as artifacts
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import diff_from_changed
from tests.regional._cov95_release_support import RecordingRunner, ResourceRelease
from tests.regional._resource_probe_fakes import resource_probe_result


class ArtifactRunner(RecordingRunner):
    def __init__(self, release: ResourceRelease) -> None:
        super().__init__(self.execute)
        self.release = release
        self.document: dict[str, Any] | None = None
        self.probes = []
        self.files = []
        self.bad_metadata_after_create = False

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        self.probes.append(arguments)
        if self.document is None:
            return resource_probe_result(arguments, present=False)
        return 0, json.dumps(self.document), ""

    def execute(self, arguments: list[str], _kwargs: dict[str, Any]) -> str:
        if "create" not in arguments:
            raise AssertionError("unexpected artifact transport")
        name = arguments[arguments.index("configmap") + 1]
        field = next(item for item in arguments if item.startswith("--from-file="))
        key, raw_path = field.removeprefix("--from-file=").split("=", 1)
        path = Path(raw_path)
        self.files.append(path)
        self.document = {
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": self.release.config.namespace,
                "uid": "artifact-uid",
                "resourceVersion": "1",
            },
            "binaryData": {key: base64.b64encode(path.read_bytes()).decode()},
        }
        if self.bad_metadata_after_create:
            self.document["metadata"] = "invalid"
        self.release.documents[("cpu", "configmap", name)] = copy.deepcopy(
            self.document
        )
        return ""


def artifact_fixture(
    tmp_path: Path,
) -> tuple[ResourceRelease, ArtifactRunner, Path, str]:
    release = ResourceRelease()
    runner = ArtifactRunner(release)
    release.runner = runner
    path = tmp_path / "component.whl"
    path.write_bytes(b"example immutable artifact content")
    return release, runner, path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("compress", [False, True])
def test_upload_creates_and_verifies_only_the_requested_artifact(
    tmp_path: Path, compress: bool
) -> None:
    release, runner, path, digest = artifact_fixture(tmp_path)
    artifacts.upload_config_map(
        release, ["kubectl"], "artifact", path.name, path, digest, compress=compress
    )
    assert len(runner.calls) == 1
    assert release.reads == [("cpu", "configmap", "artifact")]
    if compress:
        assert runner.files[0].name.endswith(artifacts.COMPRESSED_ARTIFACT_SUFFIX), (
            "compressed upload did not use the declared compressed artifact key"
        )
        assert not runner.files[0].exists(), (
            "temporary compression file must be removed"
        )
    else:
        assert runner.files == [path]
    artifacts.upload_config_map(
        release, ["kubectl"], "artifact", path.name, path, digest, compress=compress
    )
    assert len(runner.calls) == 1, "a matching existing ConfigMap must not be rewritten"


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("binary-type", "binaryData is invalid"),
        ("value-type", "binaryData is invalid"),
        ("missing-key", "is missing"),
        ("digest", "digest mismatch"),
    ],
)
def test_existing_artifact_is_rejected_without_rewriting_on_invalid_bytes(
    tmp_path: Path, fault: str, problem: str
) -> None:
    release, runner, path, digest = artifact_fixture(tmp_path)
    runner.document = {
        "kind": "ConfigMap",
        "metadata": {
            "name": "artifact",
            "namespace": release.config.namespace,
            "uid": "artifact-uid",
        },
        "binaryData": {path.name: base64.b64encode(path.read_bytes()).decode()},
    }
    if fault == "binary-type":
        runner.document["binaryData"] = ["invalid"]
    elif fault == "value-type":
        runner.document["binaryData"][path.name] = 1
    elif fault == "missing-key":
        runner.document["binaryData"] = {}
    else:
        digest = "f" * 64
    with pytest.raises(ReleaseError, match=problem):
        artifacts.upload_config_map(
            release, ["kubectl"], "artifact", path.name, path, digest
        )
    assert runner.calls == []


def test_created_artifact_metadata_is_revalidated_on_readback(tmp_path: Path) -> None:
    release, runner, path, digest = artifact_fixture(tmp_path)
    runner.bad_metadata_after_create = True
    with pytest.raises(ReleaseError, match="metadata is invalid"):
        artifacts.upload_config_map(
            release, ["kubectl"], "artifact", path.name, path, digest
        )
    assert len(runner.calls) == 1


def test_existing_unversioned_artifact_is_verified_without_a_cached_proof(
    tmp_path: Path,
) -> None:
    release, runner, path, digest = artifact_fixture(tmp_path)
    runner.document = {
        "kind": "ConfigMap",
        "metadata": {
            "name": "artifact",
            "namespace": release.config.namespace,
            "uid": "artifact-uid",
        },
        "binaryData": {path.name: base64.b64encode(path.read_bytes()).decode()},
    }
    artifacts.upload_config_map(
        release, ["kubectl"], "artifact", path.name, path, digest
    )
    assert runner.calls == []
    runner.document["binaryData"][path.name] = base64.b64encode(b"changed").decode()
    with pytest.raises(ReleaseError, match="digest mismatch"):
        artifacts.upload_config_map(
            release, ["kubectl"], "artifact", path.name, path, digest
        )


def test_dry_run_upload_does_not_claim_readback_validation(tmp_path: Path) -> None:
    release, runner, path, digest = artifact_fixture(tmp_path)
    runner.dry_run = True
    artifacts.upload_config_map(
        release, ["kubectl"], "artifact", path.name, path, digest
    )
    assert release.reads == []


@pytest.mark.parametrize("include_registry", [False, True])
def test_cpu_secret_requirement_requests_names_only(include_registry: bool) -> None:
    release = ResourceRelease()
    release.runner.handler = lambda *_args: ""
    artifacts.require_cpu_secrets(release, include_registry=include_registry)
    arguments, kwargs = release.runner.calls[0]
    assert arguments[-2:] == ["-o", "name"]
    assert ("gpu-fault-regional-clusters" in arguments) is include_registry
    assert kwargs["capture"] is True


@pytest.mark.parametrize(
    "changed,expected",
    [
        (None, {"cpu", "executor", "node"}),
        (set(), set()),
        ({"control_plane_wheel"}, {"cpu"}),
        ({"aurora_refresh_drift"}, {"cpu"}),
        ({"executor_wheel"}, {"executor"}),
        ({"node_runtime_wheel"}, {"node"}),
        ({"node_bundle"}, {"node"}),
    ],
)
def test_upload_selection_matches_component_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed: set[str] | None,
    expected: set[str],
) -> None:
    release = ResourceRelease()
    release.wheel_cm, release.executor_wheel_cm, release.bundle_cm = (
        "cpu",
        "executor",
        "node",
    )
    release.config.wheel = tmp_path / "cpu.whl"
    release.config.executor_wheel = tmp_path / "executor.whl"
    release.config.bundle = tmp_path / "bundle.tar"
    release.wheel_sha = "a" * 64
    calls = []

    def upload(
        _release: Any,
        prefix: list[str],
        name: str,
        key: str,
        path: Path,
        sha: str,
        **kwargs: Any,
    ) -> None:
        calls.append((name, prefix, key, path, sha, kwargs))

    monkeypatch.setattr(artifacts, "upload_config_map", upload)
    artifacts.upload_release(
        release, None if changed is None else diff_from_changed(changed)
    )
    assert {call[0] for call in calls} == expected
    assert all(
        call[-1].get("compress") is True for call in calls if call[0] != "node"
    ), "component wheel uploads must retain compression"
    assert all("--context" in call[1] for call in calls if call[0] != "cpu"), (
        "GPU artifact uploads must name their cluster context"
    )
