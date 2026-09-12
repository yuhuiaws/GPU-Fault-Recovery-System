"""The images a release snapshot reads back from the live site.

Split from ``regional_release_state`` (file-size ratchet): the consistency
rule for a fleet's runtime / Node Installer images and the one exception a
site with no GPU cluster left needs.
"""

from __future__ import annotations

from gpu_fault_release.regional_release_config import ReleaseError


def require_consistent_images(
    description: str,
    images: dict[str, str | None],
) -> str:
    missing = sorted(name for name, image in images.items() if not image)
    if missing:
        raise ReleaseError(
            f"cannot capture previous {description} image from: " + ", ".join(missing)
        )
    distinct = {str(image) for image in images.values()}
    if len(distinct) != 1:
        raise ReleaseError(
            f"previous {description} images are inconsistent across: "
            + ", ".join(sorted(images))
        )
    return distinct.pop()


def previous_node_installer_image(
    *,
    capture_gpu: bool,
    images: dict[str, str | None],
    recorded: str,
    configured: str,
) -> str:
    """The Node Installer image the previous release ran, for the snapshot.

    Read from the GPU clusters' reconcilers when the capture covers them and
    there is at least one cluster; a site whose last GPU cluster was just
    removed has none to read, and the empty mapping used to be judged
    "inconsistent" (``remove-cluster`` of the last cluster failed at
    ``sync-state`` on 2026-09-12). Removing the last cluster is supported, so
    that site keeps the image its release state recorded, else the configured
    one.
    """

    if capture_gpu and images:
        return require_consistent_images("Node Installer", images)
    return recorded or configured
