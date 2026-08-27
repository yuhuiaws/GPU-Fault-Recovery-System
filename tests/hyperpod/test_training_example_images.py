from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IMAGE = (
    "public.ecr.aws/deep-learning-containers/pytorch-training"
    "@sha256:"
    "ff2c928a2e7b3b290b7c3e353085a373"
    "b3cfc7f6b165e338b9be4df439feae38"
)
OLD_VLLM_IMAGE = (
    "public.ecr.aws/deep-learning-containers/vllm:server-hyperpod-cuda-v1.1"
)


def _yaml_documents(path: Path) -> list[dict[str, Any]]:
    return [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(document, dict)
    ]


def _training_manifests() -> list[tuple[Path, dict[str, Any]]]:
    manifests = []
    for root in (
        ROOT / "examples/hyperpod",
        ROOT / "scripts/e2e/regional/manifests/training",
        ROOT / "scripts/e2e",
    ):
        for path in root.rglob("*.yaml"):
            manifests.extend(
                (path, document)
                for document in _yaml_documents(path)
                if document.get("kind") == "PyTorchJob"
            )
    return sorted(manifests, key=lambda item: str(item[0]))


def test_training_examples_use_validated_training_dlc() -> None:
    manifests = _training_manifests()

    assert manifests
    for path, job in manifests:
        text = path.read_text(encoding="utf-8")
        assert OLD_VLLM_IMAGE not in text
        assert "pytorch-training:ff2c928a" not in text
        assert "sagemaker.amazonaws.com/enable-job-auto-resume" not in text

        spec = job["spec"]
        assert spec["runPolicy"]["cleanPodPolicy"] == "None"
        replica_specs = spec["pytorchReplicaSpecs"]
        assert replica_specs
        for role, replica in replica_specs.items():
            assert replica["restartPolicy"] == "Never", (path, role)
            template = replica["template"]
            assert template["spec"]["restartPolicy"] == "Never", (path, role)
            annotations = template.get("metadata", {}).get("annotations", {})
            training_container = annotations.get(
                "gpu-fault.io/training-container", "pytorch"
            )
            containers = [
                container
                for container in template["spec"]["containers"]
                if container.get("name") == training_container
            ]
            assert len(containers) == 1, (path, role, training_container)
            assert containers[0]["image"] == EXPECTED_IMAGE, (
                path,
                role,
                containers[0]["image"],
            )


def test_hyperpod_examples_declare_managed_or_source_semantics() -> None:
    readme = (ROOT / "examples/README.md").read_text(encoding="utf-8")

    for path in sorted((ROOT / "examples/hyperpod").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        header = "\n".join(text.splitlines()[:6])
        jobs = [
            document
            for document in _yaml_documents(path)
            if document.get("kind") == "PyTorchJob"
        ]

        assert len(jobs) == 1, path
        assert path.name in readme
        assert "SOURCE MANIFEST:" in header or "ANNOTATED MANIFEST:" in header, path

        managed = (
            jobs[0].get("metadata", {}).get("labels", {}).get("gpu-fault.io/managed")
            == "true"
        )
        if "SOURCE MANIFEST:" in header:
            assert "Direct kubectl apply" in header
            assert not managed
        else:
            assert managed
            for role, replica in jobs[0]["spec"]["pytorchReplicaSpecs"].items():
                labels = replica["template"].get("metadata", {}).get("labels", {})
                assert labels.get("gpu-fault.io/managed") == "true", (path, role)


def test_p5en_sized_manifests_publish_the_resource_assumption() -> None:
    # 门禁同时覆盖 examples/hyperpod/ 和统一的区域 E2E manifest 目录。
    # 8 GPU / 16 EFA 的夹具（q118-gpu-intensive-3n、q118-log-snapshot-8gpu）。
    # 它们跑在同一批 p5en 节点上，却因为路径被 continue 掉而从来不用声明机型，
    # 于是「哪些清单需要 p5en」这件事只在一半的树上成立。
    sized: set[Path] = set()
    for path, job in _training_manifests():
        requires_p5en = False
        for replica in job["spec"]["pytorchReplicaSpecs"].values():
            for container in replica["template"]["spec"]["containers"]:
                resources = container.get("resources", {})
                requests = resources.get("requests", {})
                requires_p5en |= (
                    str(requests.get("nvidia.com/gpu")) == "8"
                    or str(requests.get("vpc.amazonaws.com/efa")) == "16"
                )
        if requires_p5en:
            sized.add(path)
            header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
            assert "p5en" in header, path

    roots = {str(path.relative_to(ROOT)).split("/", 1)[0] for path in sized}
    assert roots == {"examples", "scripts"}, sorted(roots)


def test_examples_index_points_to_related_e2e_training_fixtures() -> None:
    readme = (ROOT / "examples/README.md").read_text(encoding="utf-8")
    fixtures = [
        path
        for path, _job in _training_manifests()
        if str(path).startswith(
            (
                str(ROOT / "scripts/e2e"),
                str(ROOT / "scripts/e2e/regional/manifests/training"),
            )
        )
    ]

    assert fixtures
    assert "`scripts/e2e/regional/manifests/`" in readme
    assert "`scripts/e2e/regional/manifests/training/`" in readme


def test_current_manuals_use_the_same_training_digest() -> None:
    for relative in ("docs/部署和运维手册.md", "docs/区域模式端到端验收测试用例.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert EXPECTED_IMAGE in text
        assert OLD_VLLM_IMAGE not in text


def test_primary_training_example_performs_optimizer_updates() -> None:
    text = (ROOT / "examples/hyperpod/three-node-pytorchjob.yaml").read_text(
        encoding="utf-8"
    )

    assert "DistributedDataParallel" in text
    assert "loss.backward()" in text
    assert "optimizer.step()" in text
    assert "for step in range(3)" in text
