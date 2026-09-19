"""Complete population and actual runtime-capability checks use no live API."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import destr008_capabilities as checks
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

IMAGE = "registry.invalid/runtime@sha256:" + "a" * 64


def population(plane: str, deployment: str, label: str) -> dict[str, Any]:
    name, uid, namespace = f"{plane}-pod", f"{plane}-uid", "gpu-fault-system"
    return {
        "deployment": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment,
                "namespace": namespace,
                "uid": f"{plane}-deployment",
                "generation": 3,
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {"containers": [{"name": "runtime", "image": IMAGE}]}
                },
            },
            "status": {
                "replicas": 1,
                "readyReplicas": 1,
                "updatedReplicas": 1,
                "availableReplicas": 1,
                "observedGeneration": 3,
            },
        },
        "pods": {
            "items": [
                {
                    "metadata": {
                        "name": name,
                        "uid": uid,
                        "namespace": namespace,
                        "labels": {"app": label},
                        "ownerReferences": [
                            {
                                "kind": "ReplicaSet",
                                "name": f"{plane}-rs",
                                "uid": f"{plane}-rs-uid",
                                "controller": True,
                            }
                        ],
                    },
                    "spec": {
                        "nodeName": f"{plane}-node",
                        "containers": [{"name": "runtime", "image": IMAGE}],
                    },
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "containerStatuses": [
                            {
                                "name": "runtime",
                                "ready": True,
                                "containerID": f"container-{plane}",
                                "imageID": "containerd://" + IMAGE,
                                "restartCount": 0,
                            }
                        ],
                    },
                }
            ]
        },
        "replicasets": {
            "items": [
                {
                    "metadata": {
                        "name": f"{plane}-rs",
                        "uid": f"{plane}-rs-uid",
                        "namespace": namespace,
                        "ownerReferences": [
                            {
                                "kind": "Deployment",
                                "name": deployment,
                                "uid": f"{plane}-deployment",
                                "controller": True,
                            }
                        ],
                    }
                }
            ]
        },
    }


def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RegionalLiveFixture, dict[str, Any], list[tuple[str, tuple[str, ...]]]]:
    paths = [tmp_path / "cpu", tmp_path / "gpu"]
    for path in paths:
        path.write_text("fixture only")
    regional = RegionalLiveFixture(
        RegionalLiveSettings(
            paths[0],
            paths[1],
            "gpu-context",
            "gpu-fault-system",
            "cluster-a",
            "us-west-2",
        )
    )
    values: dict[str, Any] = {
        plane: population(plane, name, label) for plane, name, label in checks.TARGETS
    }
    values["probe"] = {}
    calls: list[tuple[str, tuple[str, ...]]] = []

    def kube(plane: str, *args: str, **kwargs: Any) -> str:
        calls.append((plane, args))
        if args[0] == "exec":
            assert args[1] == "-i", "the pinned source must use stdin, not an exec URL"
            assert args[3:5] == ("-c", "runtime"), (
                "the exact bound container must be selected"
            )
            assert kwargs["timeout"] == 30, "capability probes need bounded execution"
            source_sha256 = hashlib.sha256(kwargs["input_text"].encode()).hexdigest()
            assert args[-4:] == ("-c", checks.SOURCE_LOADER, source_sha256, plane)
            callback = values.pop("after_exec", None)
            if callback is not None:
                callback()
            proof = {
                "capability": "synthetic-replacement-activation-inhibition",
                "capability_version": 1,
                "component": "api" if plane == "cpu" else "executor",
                "marker": "activation_forbidden",
                "minimum_executor_protocol_version": 4,
                "executor_protocol_version": 4,
                "supported": True,
                "probe_sha256": source_sha256,
                "checks": {
                    key: True for key in checks.CHECKS[plane] | checks.COMMON_CHECKS
                },
                **values["probe"],
            }
            return json.dumps(proof)
        assert args[0] == "get", "capability preflight must be read-only"
        return json.dumps(values[plane][args[1]])

    monkeypatch.setattr(regional, "kubectl", kube)
    return regional, values, calls


def test_all_ready_populations_are_probed_and_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional, _, calls = fixture(tmp_path, monkeypatch)
    result = checks.read_capabilities(regional)
    assert result["supported"] is True, "all deployed components must prove support"
    assert len(result["populations"]) == 2, "both CPU and GPU populations are required"
    assert [plane for plane, args in calls if args[0] == "exec"] == ["cpu", "gpu"], (
        "neither control-plane nor Executor support may be inferred from the other"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "deployment-kind",
        "deployment-name",
        "deployment-namespace",
        "container-count",
        "container-name",
        "mutable-image",
        "rs-uid",
        "rs-name",
    ],
)
def test_deployment_and_replicaset_identity_are_required_before_exec(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional, values, calls = fixture(tmp_path, monkeypatch)
    deployment = values["cpu"]["deployment"]
    replica = values["cpu"]["replicasets"]["items"][0]
    if defect == "deployment-kind":
        deployment["kind"] = "StatefulSet"
    elif defect == "deployment-name":
        deployment["metadata"]["name"] = "different"
    elif defect == "deployment-namespace":
        deployment["metadata"]["namespace"] = "different"
    elif defect == "container-count":
        deployment["spec"]["template"]["spec"]["containers"].append(
            {"name": "extra", "image": IMAGE}
        )
    elif defect == "container-name":
        deployment["spec"]["template"]["spec"]["containers"][0]["name"] = ""
    elif defect == "mutable-image":
        deployment["spec"]["template"]["spec"]["containers"][0]["image"] = "mutable:tag"
    elif defect == "rs-uid":
        replica["metadata"]["uid"] = ""
    else:
        replica["metadata"]["name"] = ""
    with pytest.raises(RuntimeError):
        checks.read_capabilities(regional)
    assert not any(args[0] == "exec" for _, args in calls), (
        "an unbound deployment or ReplicaSet cannot receive a capability challenge"
    )


def test_unrelated_replicaset_is_not_adopted_as_part_of_the_guard_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional, values, _ = fixture(tmp_path, monkeypatch)
    foreign = copy.deepcopy(values["cpu"]["replicasets"]["items"][0])
    foreign["metadata"]["ownerReferences"][0]["uid"] = "foreign-deployment"
    values["cpu"]["replicasets"]["items"].append(foreign)
    assert checks.read_capabilities(regional)["supported"], (
        "a foreign nonparticipant ReplicaSet is not a guard replica"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"minimum_executor_protocol_version": 3},
        {"minimum_executor_protocol_version": True},
        {"executor_protocol_version": 3},
        {"executor_protocol_version": None},
        {"marker": "other"},
        {"supported": False},
        {"supported": 1},
        {"component": "executor"},
        {"capability_version": True},
        {"capability_version": 2},
        {"capability": "other"},
        {"checks": []},
        {"checks": {}},
        {"checks": dict.fromkeys(checks.CHECKS["cpu"] | checks.COMMON_CHECKS, 1)},
        {"checks": dict.fromkeys(checks.CHECKS["cpu"] | checks.COMMON_CHECKS, False)},
    ],
)
def test_unsupported_or_untyped_capability_never_authorizes_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    regional, values, _ = fixture(tmp_path, monkeypatch)
    values["probe"].update(change)
    with pytest.raises(RuntimeError, match="cannot enforce"):
        checks.read_capabilities(regional)


@pytest.mark.parametrize(
    "boundary",
    ["partial", "uid", "owner", "missing-owner", "image", "digest", "status"],
)
def test_incomplete_foreign_or_unbound_population_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    regional, values, calls = fixture(tmp_path, monkeypatch)
    current = values["cpu"]
    pod = current["pods"]["items"][0]
    if boundary == "partial":
        current["deployment"]["spec"]["replicas"] = 2
    elif boundary == "uid":
        pod["metadata"]["uid"] = ""
    elif boundary == "owner":
        pod["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif boundary == "missing-owner":
        pod["metadata"]["ownerReferences"] = [
            {"kind": "ReplicaSet", "controller": True}
        ]
    elif boundary == "image":
        pod["spec"]["containers"][0]["image"] = "other@sha256:" + "b" * 64
    elif boundary == "digest":
        pod["status"]["containerStatuses"][0]["imageID"] = "other"
    else:
        pod["status"]["containerStatuses"][0]["restartCount"] = False
    with pytest.raises(RuntimeError):
        checks.read_capabilities(regional)
    assert not any(args[0] == "exec" for _, args in calls), (
        "unproven ownership stops before Pod exec"
    )


@pytest.mark.parametrize("change", ["uid", "container", "generation"])
def test_population_drift_during_exec_invalidates_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    regional, values, _ = fixture(tmp_path, monkeypatch)

    def alter() -> None:
        if change == "uid":
            values["cpu"]["pods"]["items"][0]["metadata"]["uid"] = "recreated"
        elif change == "container":
            values["cpu"]["pods"]["items"][0]["status"]["containerStatuses"][0][
                "containerID"
            ] = "restarted"
        else:
            values["cpu"]["deployment"]["metadata"]["generation"] = 4
            values["cpu"]["deployment"]["status"]["observedGeneration"] = 4

    values["after_exec"] = alter
    with pytest.raises(RuntimeError, match="changed during"):
        checks.read_capabilities(regional)


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_shipped_readonly_probe_exercises_real_capability_helpers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], plane: str
) -> None:
    source, digest = checks.probe_source()
    monkeypatch.setattr(sys, "argv", ["-c", digest, plane])
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(buffer=io.BytesIO(source.encode()))
    )
    exec(compile(checks.SOURCE_LOADER, "<inhibition-loader>", "exec"), {})
    result = json.loads(capsys.readouterr().out)
    assert result["probe_sha256"] == digest
    assert result["minimum_executor_protocol_version"] >= 4, (
        "the guarded command requires the new protocol"
    )
    assert result["supported"] is True and all(
        value is True for value in result["checks"].values()
    ), "the actual deployed probe must use the real protocol/activation checks"


@pytest.mark.parametrize("digest", [None, "", "f" * 64, True])
def test_missing_or_foreign_probe_digest_cannot_prove_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: object
) -> None:
    regional, values, _ = fixture(tmp_path, monkeypatch)
    values["probe"]["probe_sha256"] = digest
    with pytest.raises(RuntimeError, match="cannot enforce"):
        checks.read_capabilities(regional)


@pytest.mark.parametrize("defect", ["digest", "oversized"])
def test_loader_rejects_unpinned_bytes_before_executing_any_source(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    source = b"raise AssertionError('unapproved source executed')\n"
    expected = "0" * 64
    if defect == "oversized":
        source += b" " * checks.MAX_SOURCE_BYTES
        expected = hashlib.sha256(source).hexdigest()
    monkeypatch.setattr(sys, "argv", ["-c", expected, "cpu"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(source)))
    with pytest.raises(SystemExit, match="source differs"):
        exec(compile(checks.SOURCE_LOADER, "<inhibition-loader>", "exec"), {})


@pytest.mark.parametrize("shape", ["empty", "large", "directory", "symlink", "fifo"])
def test_probe_source_is_a_bounded_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    path = tmp_path / "probe.py"
    if shape == "directory":
        path.mkdir()
    elif shape == "symlink":
        target = tmp_path / "target.py"
        target.write_text("pass\n")
        path.symlink_to(target)
    elif shape == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(
            b"" if shape == "empty" else b"x" * (checks.MAX_SOURCE_BYTES + 1)
        )
    monkeypatch.setattr(checks, "PROBE_SOURCE", path)
    with pytest.raises((RuntimeError, OSError)):
        checks.probe_source()


@pytest.mark.parametrize("change", ["contents", "path"])
def test_probe_source_mutation_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = tmp_path / "probe.py"
    path.write_text("pass\n")
    fdopen = os.fdopen

    class Reader:
        def __init__(self, fd: int) -> None:
            self.source = fdopen(fd, "rb", closefd=False)

        def __enter__(self) -> Reader:
            return self

        def __exit__(self, *args: object) -> None:
            self.source.close()

        def read(self, size: int) -> bytes:
            value = self.source.read(size)
            if change == "contents":
                path.write_text("raise RuntimeError('changed')\n")
            else:
                replacement = tmp_path / "replacement.py"
                replacement.write_text("pass\n")
                replacement.replace(path)
            return value

    local_os = SimpleNamespace(**vars(os))
    local_os.fdopen = lambda fd, *args, **kwargs: Reader(fd)
    monkeypatch.setattr(checks, "os", local_os)
    monkeypatch.setattr(checks, "PROBE_SOURCE", path)
    with pytest.raises(RuntimeError, match="changed while reading"):
        checks.probe_source()


ROOT = Path(__file__).resolve().parents[2]
ROLE_MANIFESTS = {
    "cpu": ROOT
    / "deploy/control-plane/regional/generated/gpu-fault-api-ha-ingress.yaml",
    "gpu": ROOT / "deploy/dataplane/cluster-action-executor.yaml",
}


def test_capability_targets_name_the_deployed_deployment_objects() -> None:
    """The probe fetches ``get deployment <name>``: the name must be the object's
    ``metadata.name`` in the role manifest, not the manifest file name (the
    ingress file is ``gpu-fault-api-ha-ingress.yaml`` for a Deployment called
    ``gpu-fault-api-ha``; the live preflight answered NotFound for the file name)."""

    for plane, name, label in checks.TARGETS:
        documents = [
            item
            for item in yaml.safe_load_all(ROLE_MANIFESTS[plane].read_text())
            if isinstance(item, dict) and item.get("kind") == "Deployment"
        ]
        names = {item["metadata"]["name"] for item in documents}
        assert name in names, (plane, name, sorted(names))
        deployment = next(
            item for item in documents if item["metadata"]["name"] == name
        )
        assert (
            deployment["spec"]["template"]["metadata"]["labels"].get("app") == label
        ), (plane, label)
