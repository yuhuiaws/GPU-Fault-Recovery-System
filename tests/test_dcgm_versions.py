"""NVIDIA's rule between the exporter's DCGM client and the host engine.

"For a separate DCGM host engine, the exporter's DCGM client library version
must also be greater than or equal to the DCGM host engine version. Meeting
this connection requirement does not make a mismatched release pair
supported." (NVIDIA, "Install DCGM Exporter", "Requirements and Version
Compatibility"). The product read neither version: it ran a 4.4.1 client
against newer engines (the HyperPod AMI's own, the plain EKS pin's 4.7.0)
without anything saying so.
"""

from __future__ import annotations

import pytest

from gpu_fault import dcgm_versions as VERSIONS
from gpu_fault.dcgm_versions import DcgmVersionRelation

#: ``dcgmi -v`` of DCGM 4.6.1 against a 4.6.0 engine (``dcgmi/Version.cpp``,
#: ``common/DcgmBuildInfo.cpp.in``: ``Name : value`` lines, empty values left out).
DCGMI_461_CONNECTED = """
Local build info:
Version : 4.6.1
Build ID : 12
Build Date : 2026-07-24
Build Type : Release
Commit ID : 64df9f8
Branch Name : rel_dcgm_4_6
CPU Arch : x86_64
Build Platform : Linux 4.15.0-180-generic #189-Ubuntu SMP Wed May 18 14:13:57 UTC 2022 x86_64
CRC : 1c4d2e0a

Hostengine build info:
Version : 4.6.0
Build ID : 9
Build Date : 2026-06-30
Build Type : Release
CPU Arch : x86_64
"""
#: DCGM 4.7.0 prints its own build and a line instead of the engine's block.
DCGMI_470_UNREACHABLE = """
Local build info:
Version : 4.7.0
Build ID : 3

Hostengine build info not available: could not connect to hostengine at localhost: Host engine connection invalid/disconnected
"""


@pytest.mark.parametrize(
    ("host", "client", "relation"),
    [
        ("4.6.1", "4.6.1", DcgmVersionRelation.PAIRED),
        ("4.6.1-1", "4.6.1", DcgmVersionRelation.PAIRED),
        ("4.6.0", "4.6.1", DcgmVersionRelation.CLIENT_NEWER),
        ("4.7.0", "4.6.1", DcgmVersionRelation.HOST_NEWER),
        ("4.7.0-2", "4.6.1", DcgmVersionRelation.HOST_NEWER),
        ("4.10.0", "4.9.9", DcgmVersionRelation.HOST_NEWER),
        ("4.7.0", "4.4.1", DcgmVersionRelation.HOST_NEWER),
        ("", "4.6.1", DcgmVersionRelation.UNKNOWN),
        ("4.7.0", None, DcgmVersionRelation.UNKNOWN),
        ("latest", "4.6.1", DcgmVersionRelation.UNKNOWN),
    ],
)
def test_the_relation_compares_dcgm_versions_numerically(
    host: object, client: object, relation: DcgmVersionRelation
) -> None:
    assert VERSIONS.dcgm_version_relation(host, client) is relation, (
        f"engine {host!r} against client {client!r}: the package build is not "
        "part of the DCGM version, and 4.10 is newer than 4.9"
    )


@pytest.mark.parametrize(
    ("text", "version"),
    [
        ("4.7.0", "4.7.0"),
        (" 4.7.0-2 ", "4.7.0"),
        ("4.6.1.3955-1", "4.6.1"),
        ("4.7", None),
        ("4.70-1", None),
        ("v4.7.0", None),
        (None, None),
        ("4.7.0;reboot", None),
    ],
)
def test_a_dcgm_version_is_three_numbers_with_an_optional_build(
    text: object, version: str | None
) -> None:
    assert VERSIONS.dcgm_version(text) == version, (
        f"{text!r}: a version that lands in a heartbeat key must be plain"
    )


def test_the_engine_version_is_the_hostengine_block_of_dcgmi_v() -> None:
    assert VERSIONS.hostengine_version_from_dcgmi(DCGMI_461_CONNECTED) == "4.6.0", (
        "the engine's own version, not the dcgmi binary's"
    )
    assert VERSIONS.hostengine_version_from_dcgmi(DCGMI_470_UNREACHABLE) is None, (
        "an unreachable engine reports no version"
    )
    assert VERSIONS.hostengine_version_from_dcgmi("") is None, (
        "DCGM 4.6 prints nothing at all for an unreachable engine"
    )
    local_only = DCGMI_461_CONNECTED.split("Hostengine build info:", 1)[0]
    assert VERSIONS.hostengine_version_from_dcgmi(local_only) is None, (
        "the local build's version is never taken for the engine's"
    )


@pytest.mark.parametrize(
    ("reference", "versions"),
    [
        ("nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless", ("4.6.1", "4.8.4")),
        (
            "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04",
            ("4.4.1", "4.5.2"),
        ),
        ("nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4", ("4.6.1", "4.8.4")),
        (
            "registry.example:5000/mirror/dcgm-exporter:4.6.1-4.8.4-distroless@sha256:"
            + "a" * 64,
            ("4.6.1", "4.8.4"),
        ),
        ("nvcr.io/nvidia/k8s/dcgm-exporter:4.8.4", None),
        ("nvcr.io/nvidia/k8s/dcgm-exporter:latest", None),
        ("nvcr.io/nvidia/k8s/dcgm-exporter@sha256:" + "b" * 64, None),
        ("registry.example:5000/dcgm-exporter", None),
    ],
)
def test_the_client_version_is_the_dcgm_half_of_the_exporter_tag(
    reference: str, versions: tuple[str, str] | None
) -> None:
    assert VERSIONS.exporter_tag_versions(reference) == versions, (
        f"{reference}: NVIDIA tags encode <dcgm-version>-<dcgm-exporter-version>"
        "-<image-variant>; anything else names no client version"
    )


def test_the_heartbeat_key_carries_one_plain_version() -> None:
    key = VERSIONS.hostengine_version_key("4.7.0-2")

    assert key == "nvidia-dcgm/version=4.7.0"
    assert VERSIONS.is_hostengine_version_key(key), "its own key is recognized"
    assert VERSIONS.hostengine_version_key("4.7;reboot") is None, (
        "a value that is no version never becomes a key"
    )
    for other in ("nvidia-dcgm", "nvidia-dcgm/version=4.7", "nvidia-dcgm/version="):
        assert not VERSIONS.is_hostengine_version_key(other), other
    services = {"nvidia-dcgm": {"active": "active"}, key: {"active": "active"}}
    assert VERSIONS.hostengine_version(services) == "4.7.0"
    assert VERSIONS.hostengine_version({"nvidia-dcgm": {}}) is None, (
        "an Agent that reports no version (an older one) reports none"
    )
    services[VERSIONS.hostengine_version_key("4.6.1")] = {"active": "active"}
    assert VERSIONS.hostengine_version(services) is None, (
        "two versions in one map are ambiguous: none is reported"
    )
    assert VERSIONS.hostengine_version(None) is None


def test_the_evidence_quotes_nvidia_verbatim() -> None:
    """The finding texts and the docs cite the page; the quote must stay exact."""

    assert VERSIONS.NVIDIA_EXPORTER_COMPATIBILITY_URL == (
        "https://docs.nvidia.com/datacenter/dcgm/latest/installation/"
        "install-dcgm-exporter.html#requirements-and-version-compatibility"
    )
    assert VERSIONS.NVIDIA_CONNECTION_REQUIREMENT == (
        "For a separate DCGM host engine, the exporter's DCGM client library "
        "version must also be greater than or equal to the DCGM host engine "
        "version. Meeting this connection requirement does not make a "
        "mismatched release pair supported."
    )
    assert VERSIONS.NVIDIA_PAIRING_RULE == (
        "Run DCGM Exporter with the DCGM version paired with that exporter "
        "release. Mismatched combinations might function, but they are not "
        "tested or supported."
    )
