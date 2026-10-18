"""``prepare-eks-nodegroup`` pins the DCGM the product's exporter is paired with.

NVIDIA supports an exporter only with the DCGM version its release is paired
with, and requires the exporter's client library to be at least the host
engine's version. The verb's default pin used to be 4.7.0-2 while the exporter
ran a 4.4.1 client; now the default is derived from the exporter release
(``gpu_fault.admin.dcgm_exporter_release``). A node group prepared earlier
keeps the pin of its managed part on a re-run, so a pin that is not the
exporter's version is named, with the flag that pairs it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.dcgm_exporter_release import (
    DCGM_EXPORTER_DCGM_VERSION,
    DCGM_EXPORTER_IMAGE,
    DCGM_HOST_PACKAGE_VERSION,
)
from gpu_fault.admin.eks_nodegroup_plan import dcgm_pairing_finding
from gpu_fault.admin.eks_nodegroup_user_data import (
    DCGM_HOSTENGINE_LABEL,
    dcgm_pin,
    managed_script,
    merge_managed_part,
)
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin._eks_nodegroup_world import (
    GPU_GROUP,
    LT_ID,
    SITE_ID,
    World,
    encoded,
    run_verb,
    site_user_data,
)

HOST = ("--dcgm-hostengine", "host")


def _plan(
    world: World, capsys: pytest.CaptureFixture[str], *arguments: str
) -> tuple[dict[str, Any], str]:
    try:
        code = run_verb(world, *arguments)
    except BootstrapError as exc:  # pragma: no cover - the assertion names it
        raise AssertionError(f"the plan was refused: {exc}") from exc
    captured = capsys.readouterr()
    assert code == 0, captured.err
    (group,) = json.loads(captured.out)["nodegroups"]
    return group, captured.err


def test_the_default_pin_is_the_exporters_dcgm_and_names_no_finding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    group, err = _plan(World(user_data=site_user_data()), capsys, *HOST)

    dcgm = group["dcgm_hostengine"]
    assert (dcgm["pin"], dcgm["exporter_dcgm_client"], dcgm["exporter_pairing"]) == (
        DCGM_HOST_PACKAGE_VERSION,
        DCGM_EXPORTER_DCGM_VERSION,
        "paired",
    ), "host mode installs the version the exporter's client is paired with"
    assert "DCGM pin" not in err, err


@pytest.mark.parametrize(
    ("version", "pairing", "why"),
    [
        (
            "4.7.0-2",
            "host-newer",
            "is newer than the DCGM client 4.6.1 of the product's exporter "
            f"{DCGM_EXPORTER_IMAGE}; NVIDIA requires the client to be at least the "
            "host engine version",
        ),
        (
            "4.6.0-1",
            "client-newer",
            "is not the version paired with the DCGM client 4.6.1 of the product's "
            f"exporter {DCGM_EXPORTER_IMAGE}; NVIDIA supports the paired version only",
        ),
    ],
)
def test_a_pin_off_the_exporters_version_is_a_finding_with_the_pairing_flag(
    capsys: pytest.CaptureFixture[str], version: str, pairing: str, why: str
) -> None:
    group, err = _plan(
        World(user_data=site_user_data()), capsys, *HOST, "--dcgm-version", version
    )

    assert group["dcgm_hostengine"]["exporter_pairing"] == pairing
    assert (
        f"finding: {GPU_GROUP}: the DCGM pin {version} {why}: plan with "
        f"--dcgm-version {DCGM_HOST_PACKAGE_VERSION}"
    ) in err, err


def test_a_group_prepared_with_the_old_pin_keeps_it_and_is_told_the_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live: groups were prepared with 4.7.0-2 before the exporter was paired."""

    world = World()
    text = merge_managed_part(
        site_user_data(), managed_script(dcgm=dcgm_pin("4.7.0-2"))
    ).text
    world.templates[LT_ID][3]["data"]["UserData"] = encoded(text)
    document = world.nodegroups[GPU_GROUP]
    document["labels"] = {
        **document["labels"],
        OWNERSHIP_LABEL: SITE_ID,
        DCGM_HOSTENGINE_LABEL: "host",
    }

    group, err = _plan(world, capsys)

    assert group["dcgm_hostengine"]["pin"] == "4.7.0-2", (
        "a re-run keeps the managed part's pin: changing the node DCGM is a roll "
        "the operator asks for"
    )
    assert f"plan with --dcgm-version {DCGM_HOST_PACKAGE_VERSION}" in err, err


def test_no_pin_means_no_pairing_finding() -> None:
    assert dcgm_pairing_finding(None) is None, "embedded mode installs no DCGM"
    assert dcgm_pairing_finding(dcgm_pin(DCGM_EXPORTER_DCGM_VERSION)) is None, (
        "a version-only pin of the exporter's DCGM is paired with any build"
    )
