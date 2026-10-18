"""DCGM versions: the host engine's, the exporter's client library's, and NVIDIA's rule.

A host-mode exporter (``DCGM_REMOTE_HOSTENGINE_INFO``: every HyperPod node, an
``eks-managed`` node group labelled ``gpu-fault.io/dcgm-hostengine=host``) is a
DCGM client of the node's ``nv-hostengine``. NVIDIA states two rules for it in
"Install DCGM Exporter", section "Requirements and Version Compatibility"
(:data:`NVIDIA_EXPORTER_COMPATIBILITY_URL`), quoted in
:data:`NVIDIA_PAIRING_RULE` and :data:`NVIDIA_CONNECTION_REQUIREMENT`: run the
exporter with the DCGM version its release is paired with, and the client
library must be at least the host engine's version -- meeting the second does
not make an unpaired combination supported. :func:`dcgm_version_relation`
names the relationship of two versions:

* ``paired`` -- the same DCGM version: the supported pair;
* ``client-newer`` -- the connection requirement holds, the pair is not one
  NVIDIA tests or supports;
* ``host-newer`` -- the connection requirement is broken;
* ``unknown`` -- either version could not be read.

Where the versions come from:

* the exporter's client library: the DCGM half of its image tag, which NVIDIA
  documents as ``<dcgm-version>-<dcgm-exporter-version>-<image-variant>``
  (:func:`exporter_tag_versions`);
* the host engine: what the running ``nv-hostengine`` reports about itself
  (``dcgmHostengineVersionInfo``), printed by ``dcgmi -v`` under
  ``Hostengine build info:`` (:func:`hostengine_version_from_dcgmi`). The node
  preflight puts it in its verdict; the Node Agent reports it in each heartbeat
  as one more entry of the unit map (:func:`hostengine_version_key`): the map
  takes any unit and a control plane keeps the units it knows, so an older
  control plane verifies the entry and drops it, and an older Agent sends none.

This module is shared by the Node Agent, the control plane and the deploy host
and changes only with the rule; the release the product deploys is pinned in
``gpu_fault.admin.dcgm_exporter_release``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum

from gpu_fault.cluster_scope import DCGM_HOSTENGINE_UNIT

__all__ = [
    "HOSTENGINE_VERSION_KEY_PREFIX",
    "NVIDIA_CONNECTION_REQUIREMENT",
    "NVIDIA_EXPORTER_COMPATIBILITY_URL",
    "NVIDIA_PAIRING_RULE",
    "DcgmVersionRelation",
    "dcgm_version",
    "dcgm_version_relation",
    "exporter_tag_versions",
    "hostengine_version",
    "hostengine_version_from_dcgmi",
    "hostengine_version_key",
    "is_hostengine_version_key",
]

#: NVIDIA DCGM documentation, "Install DCGM Exporter", section "Requirements and
#: Version Compatibility" (docs "Last updated on 2026-09-10" when quoted).
NVIDIA_EXPORTER_COMPATIBILITY_URL = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/installation/"
    "install-dcgm-exporter.html#requirements-and-version-compatibility"
)
#: Verbatim from that section (NVIDIA typesets the apostrophes as U+2019).
NVIDIA_PAIRING_RULE = (
    "Run DCGM Exporter with the DCGM version paired with that exporter release. "
    "Mismatched combinations might function, but they are not tested or supported."
)
NVIDIA_CONNECTION_REQUIREMENT = (
    "For a separate DCGM host engine, the exporter's DCGM client library version "
    "must also be greater than or equal to the DCGM host engine version. Meeting "
    "this connection requirement does not make a mismatched release pair supported."
)
#: The heartbeat unit-map key that carries the engine's version: no systemd
#: unit name contains ``/`` or ``=``, so the key can never shadow a real unit.
HOSTENGINE_VERSION_KEY_PREFIX = f"{DCGM_HOSTENGINE_UNIT}/version="

_VERSION = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
#: A DCGM version, optionally with a package release or build suffix
#: (``4.7.0``, ``4.7.0-2``).
_VERSION_TEXT = re.compile(r"\d+\.\d+\.\d+(?:[-+.~][0-9A-Za-z.~+-]*)?")
#: ``<dcgm-version>-<dcgm-exporter-version>[-<image-variant>]``.
_EXPORTER_TAG = re.compile(
    r"(?P<dcgm>\d+\.\d+\.\d+)-(?P<exporter>\d+\.\d+\.\d+)"
    r"(?:-(?P<variant>[A-Za-z0-9][A-Za-z0-9._-]*))?"
)
_DCGMI_SECTION = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z ]*?) build info\b.*$")
_DCGMI_VERSION = re.compile(r"^\s*Version\s*:\s*(?P<value>\S+)\s*$")


class DcgmVersionRelation(StrEnum):
    """The host engine's version against the exporter's DCGM client library."""

    PAIRED = "paired"
    CLIENT_NEWER = "client-newer"
    HOST_NEWER = "host-newer"
    UNKNOWN = "unknown"


def dcgm_version(value: object) -> str | None:
    """``major.minor.patch`` of a DCGM version text, or None.

    A package release or build suffix is dropped (``4.7.0-2`` is DCGM 4.7.0):
    the client/host-engine rule is about the DCGM version, not the package
    build.
    """

    text = str(value or "").strip()
    if not _VERSION_TEXT.fullmatch(text):
        return None
    match = _VERSION.match(text)
    return None if match is None else match.group(0)


def _key(version: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(version)
    if match is None:  # pragma: no cover - callers pass dcgm_version() results
        raise ValueError(f"not a DCGM version: {version!r}")
    return (int(match["major"]), int(match["minor"]), int(match["patch"]))


def dcgm_version_relation(host_engine: object, client: object) -> DcgmVersionRelation:
    """How the host engine's version relates to the client library's."""

    host, library = dcgm_version(host_engine), dcgm_version(client)
    if host is None or library is None:
        return DcgmVersionRelation.UNKNOWN
    if _key(host) == _key(library):
        return DcgmVersionRelation.PAIRED
    if _key(host) < _key(library):
        return DcgmVersionRelation.CLIENT_NEWER
    return DcgmVersionRelation.HOST_NEWER


def exporter_tag_versions(reference: object) -> tuple[str, str] | None:
    """``(dcgm, exporter)`` versions of a dcgm-exporter image reference's tag.

    ``nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless`` gives
    ``("4.6.1", "4.8.4")``. A digest-only reference, or a tag that does not
    follow NVIDIA's ``<dcgm-version>-<dcgm-exporter-version>`` scheme (the bare
    ``4.8.4`` or ``latest``), names no client version: None.
    """

    text = str(reference or "").strip().split("@", 1)[0]
    name = text.rsplit("/", 1)[-1]
    if ":" not in name:
        return None
    match = _EXPORTER_TAG.fullmatch(name.rsplit(":", 1)[1])
    if match is None:
        return None
    return (match["dcgm"], match["exporter"])


def hostengine_version_from_dcgmi(output: object) -> str | None:
    """The running engine's version from ``dcgmi -v``, or None.

    ``dcgmi -v`` prints the binary's own ``Local build info:`` and -- when it
    can connect -- ``Hostengine build info:``, each a block of ``Name : value``
    lines; only the engine's ``Version`` counts (DCGM 4.6 prints nothing for an
    unreachable engine, 4.7 a "not available" line).
    """

    section: str | None = None
    for line in str(output or "").splitlines():
        header = _DCGMI_SECTION.match(line)
        if header is not None:
            section = header["name"].strip().lower()
            continue
        version = _DCGMI_VERSION.match(line)
        if version is not None and section == "hostengine":
            return dcgm_version(version["value"])
    return None


def hostengine_version_key(version: object) -> str | None:
    """The heartbeat unit-map key that reports ``version``, or None."""

    normalized = dcgm_version(version)
    return None if normalized is None else HOSTENGINE_VERSION_KEY_PREFIX + normalized


def is_hostengine_version_key(key: object) -> bool:
    text = str(key or "")
    return (
        text.startswith(HOSTENGINE_VERSION_KEY_PREFIX)
        and hostengine_version_key(text[len(HOSTENGINE_VERSION_KEY_PREFIX) :]) == text
    )


def hostengine_version(services: object) -> str | None:
    """The host engine version a heartbeat unit map (or an Agent record's)
    reports: its one version entry; none, or more than one, report nothing."""

    if not isinstance(services, Mapping):
        return None
    keys = [str(key) for key in services if is_hostengine_version_key(key)]
    if len(keys) != 1:
        return None
    return keys[0][len(HOSTENGINE_VERSION_KEY_PREFIX) :]
