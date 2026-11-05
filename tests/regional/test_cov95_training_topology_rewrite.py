"""Distributed-training topology: the rewrite and refusal edges of a manifest.

``test_training_topology`` pins the derived shapes against the checked-in
three-node fixture. These cases build small PyTorchJob documents of their own
so each rewrite rule is pinned to the manifest shape that exercises it: a
container that carries its torchrun flags in ``args``, one with no resources or
with requests that are not a mapping, a replica entry that is not a mapping, a
one-node topology that drops the Worker, the refusals for a manifest with no
Master or a malformed Worker, a projection that reports its GPU count in words
and a preflight document that records no usable topology.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import training_topology as topology

GPU = topology.GPU_RESOURCE
EFA = topology.EFA_RESOURCE
TWO_NODES = topology.TrainingTopology(2, 4, ("a", "b"), efa_per_node=8)


def container(**changes: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": "pytorch",
        "image": "unit/trainer@sha256:" + "0" * 64,
        "command": ["torchrun", "--nproc-per-node=8", "train.py"],
        "resources": {
            "requests": {GPU: "8", EFA: "16", "memory": "64Gi"},
            "limits": {GPU: "8", EFA: "16"},
        },
    }
    value.update(changes)
    return value


def manifest(tmp_path: Path, replicas: dict[str, Any], name: str = "job.yaml") -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "kubeflow.org/v1",
                "kind": "PyTorchJob",
                "metadata": {"name": "unit"},
                "spec": {"pytorchReplicaSpecs": replicas},
            }
        ),
        encoding="utf-8",
    )
    return path


def replica(*containers: Any, count: int = 1) -> dict[str, Any]:
    return {"replicas": count, "template": {"spec": {"containers": list(containers)}}}


def rendered(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["spec"][
        "pytorchReplicaSpecs"
    ]


# --------------------------------------------------------------------------- #
# render_topology_manifest
# --------------------------------------------------------------------------- #
def test_torchrun_flags_in_args_follow_the_gpus_per_node(tmp_path: Path) -> None:
    source = manifest(
        tmp_path,
        {
            "Master": replica(
                container(
                    command=None,
                    args=["--nproc-per-node=8", 8, "--standalone"],
                    resources={"limits": {GPU: "8"}},
                )
            ),
            "Worker": replica(container(), count=2),
        },
    )

    target = topology.render_topology_manifest(source, TWO_NODES, tmp_path / "out.yaml")

    replicas = rendered(target)
    (master,) = replicas["Master"]["template"]["spec"]["containers"]
    assert master["command"] is None, "a container without a command list keeps it"
    assert master["args"] == ["--nproc-per-node=4", 8, "--standalone"]
    assert master["resources"] == {"limits": {GPU: "4"}}, (
        "a limits block without an EFA request gains none"
    )
    assert replicas["Worker"]["replicas"] == 1


def test_containers_without_usable_resources_keep_them_as_written(
    tmp_path: Path,
) -> None:
    source = manifest(
        tmp_path,
        {
            "Master": replica(
                container(resources=None),
                container(
                    name="sidecar", resources={"requests": "1 GPU", "limits": {}}
                ),
                "not-a-container",
            ),
            "Worker": replica(container()),
        },
    )

    target = topology.render_topology_manifest(source, TWO_NODES, tmp_path / "out.yaml")

    first, second, third = rendered(target)["Master"]["template"]["spec"]["containers"]
    assert first["resources"] is None
    assert first["command"] == ["torchrun", "--nproc-per-node=4", "train.py"], (
        "the torchrun flag is rewritten even when the resources cannot be"
    )
    assert second["resources"] == {"requests": "1 GPU", "limits": {}}
    assert third == "not-a-container"


def test_a_one_node_topology_drops_the_worker_and_skips_odd_replica_entries(
    tmp_path: Path,
) -> None:
    source = manifest(
        tmp_path,
        {"Master": replica(container()), "Worker": replica(container()), "Notes": "x"},
    )
    single = topology.TrainingTopology(1, 8, ("a",))

    target = topology.render_topology_manifest(source, single, tmp_path / "out.yaml")

    replicas = rendered(target)
    assert sorted(replicas) == ["Master", "Notes"], "the Worker replica is dropped"
    assert replicas["Notes"] == "x", "a replica entry that is not a mapping is kept"
    (master,) = replicas["Master"]["template"]["spec"]["containers"]
    assert master["resources"]["limits"] == {GPU: "8", EFA: "16"}


@pytest.mark.parametrize(
    "replicas, message",
    [
        ({"Worker": {}}, "job.yaml has no Master replica spec"),
        ({"Master": {}, "Worker": ["not", "a", "spec"]}, "job.yaml has no Worker"),
        ({"Master": {}}, "job.yaml has no Worker replica spec"),
    ],
    ids=["no-master", "worker-list", "no-worker"],
)
def test_a_manifest_without_the_replica_specs_is_refused(
    tmp_path: Path, replicas: dict[str, Any], message: str
) -> None:
    source = manifest(tmp_path, replicas)

    with pytest.raises(topology.TrainingTopologyError, match=message):
        topology.render_topology_manifest(source, TWO_NODES, tmp_path / "out.yaml")
    assert not (tmp_path / "out.yaml").exists(), "a refused manifest writes no copy"


def test_without_a_destination_the_copy_goes_to_a_private_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    source = manifest(
        tmp_path, {"Master": replica(container()), "Worker": replica(container())}
    )

    target = topology.render_topology_manifest(source, TWO_NODES)

    assert target.name == "job.yaml"
    assert target.parent.parent == temporary
    assert target.parent.name.startswith("gpu-fault-training-topology-"), (
        f"the copy's directory {target.parent.name} is the topology's own"
    )
    assert target.stat().st_mode & 0o777 == 0o600
    assert rendered(target)["Worker"]["replicas"] == 1


# --------------------------------------------------------------------------- #
# derive / evidence
# --------------------------------------------------------------------------- #
def test_an_allocatable_count_in_words_is_not_a_gpu_count() -> None:
    nodes = [
        {"name": "a", "ready": "True", "gpu_allocatable": "eight"},
        {"name": "b", "ready": "True", "gpu_allocatable": "0"},
    ]

    derived = topology.derive_training_topology(nodes)

    assert (derived.node_count, derived.gpus_per_node) == (2, 8), (
        "unreadable or zero counts fall back to the fixture's eight GPUs"
    )


@pytest.mark.parametrize(
    "value",
    [
        {"gpus_per_node": 8},
        {"node_count": "two", "gpus_per_node": 8},
        {"node_count": 2, "gpus_per_node": 8, "candidate_nodes": 5},
    ],
    ids=["no-node-count", "words", "candidates-not-iterable"],
)
def test_a_recorded_topology_that_cannot_be_read_is_none(value: Any) -> None:
    assert topology.from_evidence(value) is None
    assert topology.execution_topology({"training_topology": value}) == (
        topology.NATIVE_TOPOLOGY
    )


# --------------------------------------------------------------------------- #
# fit_gpu_count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count", [0, 9])
def test_a_single_node_job_runs_one_to_eight_gpus(count: int) -> None:
    document: dict[str, Any] = {"spec": {"pytorchReplicaSpecs": {}}}

    with pytest.raises(ValueError, match=f"runs 1-8 GPUs, not {count}"):
        topology.fit_gpu_count(document, count)


def test_fit_keeps_an_efa_request_of_a_container_without_a_gpu_limit() -> None:
    sidecar = {
        "name": "sidecar",
        "args": ["--nproc-per-node=8"],
        "resources": {"requests": {EFA: "2"}, "limits": {EFA: "2"}},
    }
    trainer = container()
    document = {
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {"template": {"spec": {"containers": [trainer, sidecar]}}}
            }
        }
    }

    topology.fit_gpu_count(document, 4)

    assert trainer["resources"]["requests"] == {GPU: "4", EFA: "8", "memory": "64Gi"}
    assert trainer["command"] == ["torchrun", "--nproc-per-node=4", "train.py"]
    assert sidecar["resources"] == {"requests": {EFA: "2"}, "limits": {EFA: "2"}}, (
        "without a GPU limit there is no per-GPU EFA ratio to keep"
    )
    assert sidecar["args"] == ["--nproc-per-node=4"]


def test_the_manifest_digest_is_the_sha256_of_the_rendered_bytes(
    tmp_path: Path,
) -> None:
    source = manifest(
        tmp_path, {"Master": replica(container()), "Worker": replica(container())}
    )
    target = topology.render_topology_manifest(source, TWO_NODES, tmp_path / "out.yaml")

    digest = topology.manifest_digest(target)

    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest != topology.manifest_digest(source), (
        "the rewritten copy is a different manifest than its source"
    )
