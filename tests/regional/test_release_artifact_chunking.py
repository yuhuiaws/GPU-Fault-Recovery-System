"""A wheel past the 1 MiB ConfigMap ceiling ships as a chunked set.

Live, 2026-10-25: every release gate passed and the deploy preflight died in
its first ``kubectl create configmap`` -- the xz-compressed control-plane
wheel had reached 1,051,472 bytes and the API server answered ``Too long: may
not be more than 1048576 bytes``. The fake API server below enforces that
ceiling the way the real one does, so these tests fail against an engine that
still writes one object per wheel and pass once it splits.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.artifact_configmaps import (
    ARTIFACT_PART_BYTES,
    CONFIG_MAP_PAYLOAD_LIMIT,
    MANIFEST_KEY,
    artifact_manifest,
    part_config_map_name,
)
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_artifacts as ARTIFACTS
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_diff import diff_from_changed
from tests.regional._cov95_release_checks import CheckedRelease, finding
from tests.regional._cov95_release_support import RecordingRunner, ResourceRelease

#: The kubectl prefixes the release double routes to its CPU and GPU documents.
CPU_KUBECTL = ["kubectl", "--kubeconfig", "/dev/null"]


def gpu_kubectl(release: ResourceRelease) -> list[str]:
    return ["kubectl", "--context", release.config.clusters[0].context]


TOO_LONG = (
    'error: failed to create configmap: ConfigMap "{name}" is invalid: []: '
    "Too long: may not be more than 1048576 bytes"
)


class ApiServer(RecordingRunner):
    """kubectl against an API server that enforces the ConfigMap ceiling.

    ``create configmap --from-file`` files the bytes the way kubectl does
    (``data`` for UTF-8 text, ``binaryData`` otherwise) and refuses an object
    over ``CONFIG_MAP_PAYLOAD_LIMIT`` with the server's message. Every object
    is registered with the release double so ``_get_json`` reads it back.
    """

    def __init__(self, release: ResourceRelease) -> None:
        super().__init__(self.execute)
        self.release = release
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.created: list[tuple[str, str]] = []
        self.refused: list[str] = []

    @staticmethod
    def _plane(arguments: list[str]) -> str:
        return (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )

    def store(self, plane: str, name: str, document: dict[str, Any]) -> None:
        self.objects[(plane, name)] = document
        self.release.documents[(plane, "configmap", name)] = document

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        assert "--ignore-not-found" in arguments, "probe must tolerate absence"
        name = arguments[arguments.index("configmap") + 1]
        document = self.objects.get((self._plane(arguments), name))
        return 0, json.dumps(document) if document is not None else "", ""

    def execute(self, arguments: list[str], _kwargs: dict[str, Any]) -> str:
        if "create" not in arguments:
            raise AssertionError(f"unexpected transport: {arguments[:6]}")
        plane = self._plane(arguments)
        name = arguments[arguments.index("configmap") + 1]
        field = next(item for item in arguments if item.startswith("--from-file="))
        key, raw_path = field.removeprefix("--from-file=").split("=", 1)
        payload = Path(raw_path).read_bytes()
        if len(payload) > CONFIG_MAP_PAYLOAD_LIMIT:
            self.refused.append(name)
            raise ReleaseError(TOO_LONG.format(name=name))
        if (plane, name) in self.objects:
            raise ReleaseError(f'configmaps "{name}" already exists')
        document: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": self.release.config.namespace,
                "uid": str(uuid.uuid4()),
                "resourceVersion": str(len(self.created) + 1),
            },
        }
        try:
            document["data"] = {key: payload.decode("utf-8")}
        except UnicodeDecodeError:
            document["binaryData"] = {key: base64.b64encode(payload).decode()}
        self.store(plane, name, document)
        self.created.append((plane, name))
        return ""


def _incompressible_wheel(tmp_path: Path, name: str, size: int) -> tuple[Path, str]:
    """A wheel whose xz stream stays over the ceiling (random bytes do not shrink)."""

    wheel = tmp_path / name
    wheel.write_bytes(os.urandom(size))
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def _small_artifact(tmp_path: Path, name: str, payload: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _release() -> tuple[ResourceRelease, ApiServer]:
    release = ResourceRelease()
    server = ApiServer(release)
    release.runner = server
    return release, server


def _legacy_object(
    server: ApiServer, plane: str, name: str, key: str, payload: bytes
) -> None:
    """What an engine before chunking left behind: one object, xz under <file>.xz."""

    server.store(
        plane,
        name,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": "gpu-fault-system",
                "uid": str(uuid.uuid4()),
                "resourceVersion": "7",
            },
            "binaryData": {key: base64.b64encode(payload).decode()},
        },
    )


def test_a_wheel_over_the_ceiling_ships_as_parts_and_a_manifest(tmp_path: Path) -> None:
    release, server = _release()
    wheel, sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 65_536,
    )
    name = "gpu-fault-control-plane-wheel-0100-" + sha[:12]

    ARTIFACTS.upload_config_map(
        release,
        CPU_KUBECTL,
        name,
        wheel.name,
        wheel,
        sha,
        compress=True,
        splittable=True,
        label="control-plane wheel",
    )

    assert server.refused == [], "an object over the ceiling reached the API server"
    parts = [part_config_map_name(name, 0), part_config_map_name(name, 1)]
    assert [created for _, created in server.created] == [*parts, name], (
        "parts must exist before the manifest that names them"
    )
    for part in parts:
        document = server.objects[("cpu", part)]
        ((key, encoded),) = document["binaryData"].items()
        assert key.startswith(wheel.name + ".xz."), key
        assert len(base64.b64decode(encoded)) <= ARTIFACT_PART_BYTES, key
    primary = server.objects[("cpu", name)]
    assert "binaryData" not in primary, "the named ConfigMap holds only the manifest"
    manifest = artifact_manifest(primary)
    assert manifest is not None, "the named ConfigMap carries artifact.json"
    assert manifest["artifact"] == wheel.name
    assert manifest["sha256"] == sha
    assert [part["config_map"] for part in manifest["parts"]] == parts
    stream = b"".join(
        base64.b64decode(
            server.objects[("cpu", part["config_map"])]["binaryData"][part["key"]]
        )
        for part in manifest["parts"]
    )
    assert hashlib.sha256(stream).hexdigest() == manifest["payload_sha256"]
    assert lzma.decompress(stream) == wheel.read_bytes(), (
        "the parts reassemble the wheel"
    )
    # The same release again: everything exists, is re-read and re-hashed, nothing is rewritten.
    created_before = len(server.created)
    ARTIFACTS.upload_config_map(
        release,
        CPU_KUBECTL,
        name,
        wheel.name,
        wheel,
        sha,
        compress=True,
        splittable=True,
    )
    assert len(server.created) == created_before, (
        "an existing set must not be rewritten"
    )


def test_the_live_failure_is_reproduced_when_splitting_is_not_allowed(
    tmp_path: Path,
) -> None:
    """Without splitting the engine would hand the API server a 1.1 MB object:
    the bundle still cannot be split, so it is refused locally instead."""
    release, server = _release()
    bundle, sha = _incompressible_wheel(
        tmp_path, "gpu-fault-node-installer-0.10.0.tar.gz", CONFIG_MAP_PAYLOAD_LIMIT + 1
    )
    with pytest.raises(ReleaseError, match="cannot be split") as refused:
        ARTIFACTS.upload_config_map(
            release,
            gpu_kubectl(release),
            "gpu-fault-node-installer-0100-" + sha[:12],
            bundle.name,
            bundle,
            sha,
            label="node installer bundle",
        )
    message = str(refused.value)
    assert f"{CONFIG_MAP_PAYLOAD_LIMIT + 1:,} bytes" in message, message
    assert f"{CONFIG_MAP_PAYLOAD_LIMIT:,}" in message, message
    assert "node installer bundle" in message, message
    assert server.calls == [], "the refusal must precede every transport"
    assert server.refused == []


def test_a_tampered_part_fails_the_digest_and_is_never_overwritten(
    tmp_path: Path,
) -> None:
    release, server = _release()
    wheel, sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 4096,
    )
    name = "gpu-fault-control-plane-wheel-0100-" + sha[:12]
    ARTIFACTS.upload_config_map(
        release,
        CPU_KUBECTL,
        name,
        wheel.name,
        wheel,
        sha,
        compress=True,
        splittable=True,
    )
    part = server.objects[("cpu", part_config_map_name(name, 1))]
    ((key, encoded),) = part["binaryData"].items()
    flipped = bytearray(base64.b64decode(encoded))
    flipped[0] ^= 0xFF
    part["binaryData"][key] = base64.b64encode(bytes(flipped)).decode()

    created_before = len(server.created)
    with pytest.raises(ReleaseError, match="does not match its manifest"):
        ARTIFACTS.upload_config_map(
            release,
            CPU_KUBECTL,
            name,
            wheel.name,
            wheel,
            sha,
            compress=True,
            splittable=True,
        )
    assert len(server.created) == created_before, (
        "a bad set must not be repaired silently"
    )
    with pytest.raises(ReleaseError, match="does not match its manifest"):
        ARTIFACTS.config_map_artifact_sha(release, CPU_KUBECTL, name, wheel.name)


def test_an_interrupted_upload_resumes_from_its_existing_parts(tmp_path: Path) -> None:
    release, server = _release()
    wheel, sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 4096,
    )
    name = "gpu-fault-control-plane-wheel-0100-" + sha[:12]
    plan = ARTIFACTS.plan_artifact_config_maps(
        wheel, name=name, key=wheel.name, compress=True, splittable=True
    )
    first = plan.parts[0]
    server.store(
        "cpu",
        first.name,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": first.name,
                "namespace": "gpu-fault-system",
                "uid": "left-behind",
                "resourceVersion": "3",
            },
            "binaryData": {first.key: base64.b64encode(first.payload).decode()},
        },
    )

    ARTIFACTS.upload_config_map(
        release,
        CPU_KUBECTL,
        name,
        wheel.name,
        wheel,
        sha,
        compress=True,
        splittable=True,
    )
    assert [created for _, created in server.created] == [plan.parts[1].name, name], (
        "the part left behind is reused, the rest of the set is completed in order"
    )

    # A part under the same name with other bytes is a stale set, not a resume.
    stale, stale_sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl.other",
        CONFIG_MAP_PAYLOAD_LIMIT + 4096,
    )
    stale_name = "gpu-fault-control-plane-wheel-0100-" + stale_sha[:12]
    server.store(
        "cpu",
        part_config_map_name(stale_name, 0),
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": part_config_map_name(stale_name, 0),
                "namespace": "gpu-fault-system",
                "uid": "stale",
                "resourceVersion": "4",
            },
            "binaryData": {
                f"{stale.name}.xz.00": base64.b64encode(b"not this slice").decode()
            },
        },
    )
    created_before = len(server.created)
    with pytest.raises(ReleaseError, match="exists with content other than"):
        ARTIFACTS.upload_config_map(
            release,
            CPU_KUBECTL,
            stale_name,
            stale.name,
            stale,
            stale_sha,
            compress=True,
            splittable=True,
        )
    assert len(server.created) == created_before, (
        "a stale part set must stop the upload"
    )


def test_rollback_reads_previous_wheels_in_every_layout(tmp_path: Path) -> None:
    """The rollback pins the previous control plane by ``_config_map_sha`` and
    re-derives the Executor wheel filename from ``config_map_binary_key``; a
    release that stored one object and a release that stored a chunked set both
    have to answer, or the previous release is unrecoverable."""
    release, server = _release()
    small, small_sha = _small_artifact(
        tmp_path,
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl",
        b"PK\x03\x04" + b"old wheel " * 2048,
    )
    legacy = "gpu-fault-executor-wheel-0100-" + small_sha[:12]
    _legacy_object(
        server,
        "gpu-a",
        legacy,
        small.name + ".xz",
        lzma.compress(small.read_bytes(), preset=9 | lzma.PRESET_EXTREME),
    )
    big, big_sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 4096,
    )
    chunked = "gpu-fault-control-plane-wheel-0100-" + big_sha[:12]
    ARTIFACTS.upload_config_map(
        release,
        CPU_KUBECTL,
        chunked,
        big.name,
        big,
        big_sha,
        compress=True,
        splittable=True,
    )
    gpu = gpu_kubectl(release)

    assert (
        ARTIFACTS.config_map_artifact_sha(release, gpu, legacy, small.name) == small_sha
    )
    assert (
        ARTIFACTS.config_map_artifact_sha(release, CPU_KUBECTL, chunked, big.name)
        == big_sha
    )
    assert STATE.config_map_binary_key(release, gpu, legacy) == small.name + ".xz", (
        "a single-object layout keeps answering with its stored key"
    )
    assert STATE.config_map_binary_key(release, CPU_KUBECTL, chunked) == big.name, (
        "a chunked set answers with the artifact its manifest names"
    )
    # The legacy object still verifies through the upload path (resume of an old release).
    ARTIFACTS.upload_config_map(
        release,
        gpu,
        legacy,
        small.name,
        small,
        small_sha,
        compress=True,
        splittable=True,
    )
    assert all(created != legacy for _, created in server.created), (
        "an existing single-object wheel must not be rewritten as a set"
    )


def _full_release(tmp_path: Path) -> tuple[ResourceRelease, ApiServer, dict[str, Any]]:
    release, server = _release()
    cpu_wheel, cpu_sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 8192,
    )
    executor_wheel, executor_sha = _small_artifact(
        tmp_path,
        "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl",
        b"PK\x03\x04" + b"executor " * 4096,
    )
    bundle, bundle_sha = _small_artifact(
        tmp_path,
        "gpu-fault-node-installer-0.10.0.tar.gz",
        b"\x1f\x8b" + b"bundle" * 512,
    )
    release.config.wheel = cpu_wheel
    release.config.executor_wheel = executor_wheel
    release.config.bundle = bundle
    release.wheel_sha = cpu_sha
    release.executor_wheel_sha = executor_sha
    release.bundle_sha = bundle_sha
    release.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + cpu_sha[:12]
    release.executor_wheel_cm = "gpu-fault-executor-wheel-0100-" + executor_sha[:12]
    release.bundle_cm = "gpu-fault-node-installer-0100-" + bundle_sha[:12]
    return (
        release,
        server,
        {"cpu": cpu_sha, "executor": executor_sha, "bundle": bundle_sha},
    )


def test_upload_release_ships_each_artifact_in_its_own_layout(tmp_path: Path) -> None:
    release, server, _digests = _full_release(tmp_path)

    ARTIFACTS.upload_release(release, None)

    created = {}
    for plane, name in server.created:
        created.setdefault(plane, []).append(name)
    assert created["cpu"] == [
        part_config_map_name(release.wheel_cm, 0),
        part_config_map_name(release.wheel_cm, 1),
        release.wheel_cm,
    ], "the control-plane wheel ships as a chunked set on the CPU cluster"
    assert created["gpu-a"] == [release.executor_wheel_cm, release.bundle_cm], (
        "the Executor wheel and the bundle still fit one object each"
    )
    executor = server.objects[("gpu-a", release.executor_wheel_cm)]
    assert list(executor["binaryData"]) == [release.config.executor_wheel.name + ".xz"]
    bundle_call = next(
        arguments
        for arguments, _kwargs in server.calls
        if release.bundle_cm in arguments
    )
    assert bundle_call[-1] == (
        f"--from-file={release.config.bundle.name}={release.config.bundle}"
    ), "the bundle is handed to kubectl raw and in place"

    # A control-plane-only diff touches the CPU set and nothing on the GPU cluster.
    before = len(server.created)
    ARTIFACTS.upload_release(release, diff_from_changed({"control_plane_wheel"}))
    assert len(server.created) == before, "an existing set is verified, not recreated"
    assert server.refused == []


def test_artifact_transport_preflight_names_layouts_and_fails_on_an_unsplittable_bundle(
    tmp_path: Path,
) -> None:
    release, _server, _digests = _full_release(tmp_path)

    report = ARTIFACTS.artifact_transport_report(release)
    assert report.status == "PASS", report.summary
    assert report.details["control_plane_wheel"]["layout"] == "chunked:2"
    assert report.details["control_plane_wheel"]["objects"] == [
        part_config_map_name(release.wheel_cm, 0),
        part_config_map_name(release.wheel_cm, 1),
        release.wheel_cm,
    ]
    assert report.details["executor_wheel"]["layout"] == "xz"
    assert report.details["node_bundle"]["layout"] == "none"
    assert report.details["config_map_limit_bytes"] == CONFIG_MAP_PAYLOAD_LIMIT
    assert "control-plane wheel" in report.summary and "3 objects" in report.summary

    # The bundle close to the ceiling is a warning the operator reads before it bites.
    release.config.bundle.write_bytes(os.urandom(int(CONFIG_MAP_PAYLOAD_LIMIT * 0.95)))
    warned = ARTIFACTS.artifact_transport_report(release)
    assert warned.status == "WARN", warned.summary
    assert "95%" in warned.summary and "cannot be split" in warned.summary

    # Over it, the preflight fails before the deploy creates anything.
    release.config.bundle.write_bytes(os.urandom(CONFIG_MAP_PAYLOAD_LIMIT + 1))
    failed = ARTIFACTS.artifact_transport_report(release)
    assert failed.status == "FAIL"
    assert (
        "node installer bundle" in failed.summary
        and "cannot be split" in failed.summary
    )
    assert failed.details["node_bundle"]["error"] == failed.summary
    with pytest.raises(CheckFailure, match="cannot be split") as refused:
        CHECKS.check_artifact_config_maps(release)
    assert refused.value.details["node_bundle"]["splittable"] is False


def test_preflight_report_carries_the_artifact_transport_check(tmp_path: Path) -> None:
    release = CheckedRelease()
    cpu_wheel, cpu_sha = _incompressible_wheel(
        tmp_path,
        "gpu_fault_control_plane-0.10.0-py3-none-any.whl",
        CONFIG_MAP_PAYLOAD_LIMIT + 8192,
    )
    executor_wheel, executor_sha = _small_artifact(
        tmp_path, "gpu_fault_cluster_executor-0.10.0-py3-none-any.whl", b"executor"
    )
    bundle, bundle_sha = _small_artifact(
        tmp_path,
        "gpu-fault-node-installer-0.10.0.tar.gz",
        os.urandom(CONFIG_MAP_PAYLOAD_LIMIT + 1),
    )
    release.config.wheel, release.config.executor_wheel, release.config.bundle = (
        cpu_wheel,
        executor_wheel,
        bundle,
    )
    release.wheel_sha, release.executor_wheel_sha, release.bundle_sha = (
        cpu_sha,
        executor_sha,
        bundle_sha,
    )
    release.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + cpu_sha[:12]
    release.bundle_cm = "gpu-fault-node-installer-0100-" + bundle_sha[:12]

    report = CHECKS.build_preflight_report(release)

    result = finding(report, "artifact_config_maps")
    assert result["status"] == "FAIL", result
    assert "cannot be split" in result["summary"]
    assert result["details"]["control_plane_wheel"]["layout"] == "chunked:2"
    assert report["healthy"] is False

    bundle.write_bytes(b"\x1f\x8b" + b"small bundle" * 64)
    release.bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    result = finding(CHECKS.build_preflight_report(release), "artifact_config_maps")
    assert result["status"] == "PASS", result
    assert MANIFEST_KEY not in json.dumps(result["details"]), (
        "the report describes layouts; it never carries ConfigMap content"
    )
