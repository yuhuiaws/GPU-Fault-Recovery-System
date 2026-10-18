"""One dcgm-exporter release, pinned everywhere, paired with the EKS host DCGM.

NVIDIA ("Install DCGM Exporter", "Requirements and Version Compatibility"):
the exporter's DCGM client library must be at least the version of the host
engine it connects to, and only the DCGM version an exporter release is paired
with is supported. Until batch 135 the product ran the 4.4.1 client of
``dcgm-exporter:4.4.1-4.5.2`` against host engines 4.6.0 (HyperPod AMI) and
4.7.0 (``prepare-eks-nodegroup``'s pin): the image was spelled in six files and
the EKS pin in a seventh, with nothing tying the two halves of the pair.
``gpu_fault.admin.dcgm_exporter_release`` is now the one source; every literal
spelling must equal it, and the EKS pin is derived from the exporter's DCGM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import bootstrap_common
from gpu_fault.admin import dcgm_exporter_release as PIN
from gpu_fault.admin import eks_nodegroup_user_data as USER_DATA
from gpu_fault.dcgm_versions import (
    DcgmVersionRelation,
    dcgm_version_relation,
    exporter_tag_versions,
)
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release.regional_release_config import ReleaseError

ROOT = Path(__file__).resolve().parents[2]
#: Every file outside the pin module that spells the exporter image.
LITERAL_PINS = {
    "deploy/dataplane/hyperpod-dcgm-exporter.yaml": re.compile(
        r"^\s*image: (?P<image>\S+)$", re.MULTILINE
    ),
    "deploy/hyperpod/deploy.sh": re.compile(
        r'^DEFAULT_DCGM_EXPORTER_IMAGE="(?P<image>[^"]+)"$', re.MULTILINE
    ),
    "deploy/control-plane/regional/regional-env.example.sh": re.compile(
        r"^export GPU_FAULT_DCGM_EXPORTER_IMAGE='(?P<image>[^']+)'$", re.MULTILINE
    ),
    "scripts/e2e/regional/manifests/hyperpod-canary.yaml": re.compile(
        r"^\s*image: (?P<image>nvcr\.io/nvidia/k8s/dcgm-exporter\S*)$", re.MULTILINE
    ),
}
LOCK = ROOT / "config/release-images.lock.json"
#: Where a tag of the exporter may appear in prose or code without being a pin.
SCANNED = ("src", "deploy", "scripts", "config", "docs", "COLLECTORS.md")
#: An OCI tag (letters, digits, ``_``, ``.``, ``-``); prose punctuation that
#: follows it (``;``, a sentence's ``.``) is not part of it.
EXPORTER_REFERENCE = re.compile(
    r"nvcr\.io/nvidia/k8s/dcgm-exporter:(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_])"
)


def test_every_spelling_of_the_exporter_image_is_the_pinned_release() -> None:
    assert RENDERING.DEFAULT_DCGM_EXPORTER_IMAGE == PIN.DCGM_EXPORTER_IMAGE, (
        "the renderer substitutes the release image for this text"
    )
    assert bootstrap_common.DEFAULT_DCGM_IMAGE == PIN.DCGM_EXPORTER_IMAGE
    for relative, pattern in LITERAL_PINS.items():
        images = [
            match["image"]
            for match in pattern.finditer((ROOT / relative).read_text(encoding="utf-8"))
        ]
        assert images == [PIN.DCGM_EXPORTER_IMAGE], (
            f"{relative} must name the pinned exporter {PIN.DCGM_EXPORTER_IMAGE} "
            f"exactly once, found {images}: a second spelling drifts on the next "
            "upgrade and the renderer stops substituting the release image"
        )


def test_the_image_lock_pins_the_tag_by_its_manifest_list_digest() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))["images"]["dcgm_exporter"]

    assert lock == {
        "reference": (
            f"{PIN.DCGM_EXPORTER_REPOSITORY}@{PIN.DCGM_EXPORTER_IMAGE_INDEX_DIGEST}"
        ),
        "source": PIN.DCGM_EXPORTER_IMAGE,
    }, (
        "the release deploys the lock's digest; its source tag is what names the "
        "exporter's DCGM client for the version comparison"
    )
    for digest in (
        PIN.DCGM_EXPORTER_IMAGE_INDEX_DIGEST,
        PIN.DCGM_EXPORTER_IMAGE_AMD64_DIGEST,
    ):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), digest


def test_the_tag_names_the_paired_dcgm_and_the_eks_pin_follows_it() -> None:
    """The single source: the tag's DCGM half is the EKS host packages' version."""

    versions = exporter_tag_versions(PIN.DCGM_EXPORTER_IMAGE)

    assert versions == (PIN.DCGM_EXPORTER_DCGM_VERSION, PIN.DCGM_EXPORTER_VERSION), (
        "the tag must follow NVIDIA's <dcgm-version>-<dcgm-exporter-version>-"
        "<image-variant> scheme, or nothing can tell the client's version"
    )
    assert PIN.DCGM_EXPORTER_IMAGE.endswith(f"-{PIN.DCGM_EXPORTER_IMAGE_VARIANT}"), (
        "the tag names its image variant, as NVIDIA's scheme does"
    )
    assert USER_DATA.DCGM_PACKAGE_VERSION == PIN.DCGM_HOST_PACKAGE_VERSION
    assert (
        dcgm_version_relation(
            USER_DATA.DCGM_PACKAGE_VERSION, PIN.DCGM_EXPORTER_DCGM_VERSION
        )
        is DcgmVersionRelation.PAIRED
    ), (
        "prepare-eks-nodegroup must install the DCGM version the exporter is paired "
        "with: an older engine is an unsupported pair, a newer one breaks NVIDIA's "
        "connection requirement"
    )
    USER_DATA.dcgm_pin(USER_DATA.DCGM_PACKAGE_VERSION)  # a valid version-release


def test_the_exporter_is_rendered_from_the_pin_and_fails_closed_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/dcgm@sha256:" + "a" * 64
    text = RENDERING.render_dcgm_exporter_manifest(
        namespace="gpu-fault-system", image=image
    )
    (daemonset,) = [
        item
        for item in yaml.safe_load_all(text)
        if isinstance(item, dict) and item.get("kind") == "DaemonSet"
    ]
    (container,) = daemonset["spec"]["template"]["spec"]["containers"]
    assert container["image"] == image

    # A manifest that no longer names the pinned image must fail the render: it
    # would apply its own tag instead of the release's digest.
    monkeypatch.setattr(
        RENDERING, "DEFAULT_DCGM_EXPORTER_IMAGE", "registry.example/gone:0.0.0-0.0.0"
    )
    with pytest.raises(ReleaseError, match="cannot replace it"):
        RENDERING.render_dcgm_exporter_manifest(
            namespace="gpu-fault-system", image=image
        )


def test_no_other_exporter_tag_is_named_anywhere() -> None:
    """A stale tag in a doc or script is how the 4.4.1 pin outlived its engines."""

    pinned = PIN.DCGM_EXPORTER_IMAGE.rsplit(":", 1)[1]
    stale: list[str] = []
    for entry in SCANNED:
        path = ROOT / entry
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for item in files:
            if not item.is_file() or item.suffix not in {
                ".md",
                ".py",
                ".sh",
                ".yaml",
                ".yml",
                ".json",
            }:
                continue
            text = item.read_text(encoding="utf-8", errors="replace")
            stale.extend(
                f"{item.relative_to(ROOT)}: {match['tag']}"
                for match in EXPORTER_REFERENCE.finditer(text)
                if match["tag"] != pinned
            )
    assert stale == [], (
        f"name the pinned exporter {PIN.DCGM_EXPORTER_IMAGE} (or a placeholder "
        f"such as <approved-tag>), not another release: {stale}"
    )
