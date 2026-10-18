"""The NVIDIA dcgm-exporter release this product deploys, and the host DCGM paired with it.

One source for both halves of the pair (``gpu_fault.dcgm_versions`` holds the
rule):

* :data:`DCGM_EXPORTER_IMAGE` is the exporter every GPU cluster runs:
  ``regional_release_rendering.DEFAULT_DCGM_EXPORTER_IMAGE`` and
  ``bootstrap_common.DEFAULT_DCGM_IMAGE`` are this constant; the literal in
  ``deploy/dataplane/hyperpod-dcgm-exporter.yaml`` (the renderer substitutes
  the release image for exactly this text), ``deploy/hyperpod/deploy.sh``,
  ``deploy/control-plane/regional/regional-env.example.sh``,
  ``scripts/e2e/regional/manifests/hyperpod-canary.yaml`` and the ``source``
  of ``config/release-images.lock.json`` (whose ``reference`` pins
  :data:`DCGM_EXPORTER_IMAGE_INDEX_DIGEST`) must equal it, which
  ``tests/regional/test_dcgm_exporter_release_pins.py`` enforces.
* :data:`DCGM_HOST_PACKAGE_VERSION` is the build of the DCGM 4 packages
  ``prepare-eks-nodegroup`` installs in host mode
  (``eks_nodegroup_user_data.DCGM_PACKAGE_VERSION``): the exporter's own DCGM
  version, so a host-mode EKS node runs the pair NVIDIA supports ("Run DCGM
  Exporter with the DCGM version paired with that exporter release") and the
  client is never older than the engine.

The release (registry and GitHub, read 2026-10-18). NVIDIA's container
registry (``nvcr.io/v2/nvidia/k8s/dcgm-exporter``, anonymous pull token) lists
``4.6.1-4.8.4`` and ``4.6.1-4.8.4-distroless`` -- also ``4.8.4`` and
``latest`` -- as one manifest list, :data:`DCGM_EXPORTER_IMAGE_INDEX_DIGEST`
(linux/amd64 :data:`DCGM_EXPORTER_IMAGE_AMD64_DIGEST`). Its image config
carries ``com.nvidia.dcgm.version=4.6.1`` and
``org.opencontainers.image.version=4.8.4``, base
``nvcr.io/nvidia/distroless/cc:v4.1.4``, and ships ``libdcgm.so.4.6.1``; the
GitHub release ``4.8.4`` says "Release DCGM Exporter 4.8.4 with DCGM 4.6.1 and
go-dcgm v1.4611.2". It is the newest exporter release: no exporter is paired
with DCGM 4.7.x on the registry or GitHub, although DCGM 4.7.0 packages are
published (the go-dcgm ``v1.4701.*`` bindings exist, no exporter release uses
them). ``4.6.0-4.8.3-distroless``, the release paired with the HyperPod AMI's
DCGM 4.6.0, ships no ``/etc/ld.so.cache`` (checked in its layers), and its
start-up check for ``libdcgm.so.4`` then exits with a bare "exit status 1"
unless the container runtime's NVIDIA hook writes one
(github.com/NVIDIA/dcgm-exporter#612, fixed by #724 in 4.8.4), so it is not an
option. Since 4.8.2 NVIDIA publishes the distroless variant only.

So HyperPod's AMI-managed engine (DCGM 4.6.0) meets the connection requirement
with this 4.6.1 client but is no supported pair; a plain EKS node group in host
mode installs 4.6.1 and is one. Upgrading: pick the newest
``<dcgm>-<exporter>-<variant>`` tag whose DCGM is at least every host engine of
the site, read its manifest-list digest from the registry, change this module,
the literals listed above and the lock together, and roll the EKS node groups
with ``prepare-eks-nodegroup --dcgm-version <dcgm>-<build>`` (administrator
operations §5.0.1).
"""

from __future__ import annotations

__all__ = [
    "DCGM_EXPORTER_DCGM_VERSION",
    "DCGM_EXPORTER_IMAGE",
    "DCGM_EXPORTER_IMAGE_AMD64_DIGEST",
    "DCGM_EXPORTER_IMAGE_INDEX_DIGEST",
    "DCGM_EXPORTER_IMAGE_VARIANT",
    "DCGM_EXPORTER_REPOSITORY",
    "DCGM_EXPORTER_VERSION",
    "DCGM_HOST_PACKAGE_BUILD",
    "DCGM_HOST_PACKAGE_VERSION",
]

DCGM_EXPORTER_REPOSITORY = "nvcr.io/nvidia/k8s/dcgm-exporter"
#: The DCGM release the exporter is paired with: its client library version.
DCGM_EXPORTER_DCGM_VERSION = "4.6.1"
DCGM_EXPORTER_VERSION = "4.8.4"
DCGM_EXPORTER_IMAGE_VARIANT = "distroless"
DCGM_EXPORTER_IMAGE = (
    f"{DCGM_EXPORTER_REPOSITORY}:{DCGM_EXPORTER_DCGM_VERSION}-"
    f"{DCGM_EXPORTER_VERSION}-{DCGM_EXPORTER_IMAGE_VARIANT}"
)
#: The manifest list the tag resolves to (linux/amd64 and linux/arm64).
DCGM_EXPORTER_IMAGE_INDEX_DIGEST = (
    "sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003"
)
DCGM_EXPORTER_IMAGE_AMD64_DIGEST = (
    "sha256:804c17f67ab63e8a4dd5301546beefaa104129f833dae4a68f5ff7c4819f471b"
)
#: The package release of that DCGM version in NVIDIA's CUDA repositories
#: (``datacenter-gpu-manager-4-core-4.6.1-1`` and ``-cuda13-4.6.1-1`` for
#: amzn2023 x86_64).
DCGM_HOST_PACKAGE_BUILD = "1"
DCGM_HOST_PACKAGE_VERSION = f"{DCGM_EXPORTER_DCGM_VERSION}-{DCGM_HOST_PACKAGE_BUILD}"
