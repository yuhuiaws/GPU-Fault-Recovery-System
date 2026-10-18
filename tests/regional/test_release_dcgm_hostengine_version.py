"""The node preflight's host engine version against the exporter's DCGM client.

NVIDIA ("Install DCGM Exporter", "Requirements and Version Compatibility"):
"the exporter's DCGM client library version must also be greater than or equal
to the DCGM host engine version". Each deploy's node preflight now reads the
running engine's version (``dcgmi -v``, the verdict's ``dcgm.version``); the
release keeps it with the node's state and, in host mode, the client version
of the exporter it deploys, names the nodes whose engine is newer, and says so
in one ``WARN`` line. Nothing refuses: the very release that brings a newer
exporter must still be deployable over the older one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks_coverage as COVERAGE
from gpu_fault_release import regional_release_dcgm as DCGM

EXPORTER = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless"
NOW = datetime(2030, 1, 2, tzinfo=UTC)


def _target(provider: str, mode: str | None = "host") -> SimpleNamespace:
    fields: dict[str, Any] = {"cluster_id": f"gpu-{provider}", "provider": provider}
    if provider == "eks-managed" and mode is not None:
        fields["dcgm_hostengine"] = mode
    return SimpleNamespace(**fields)


def _release(image: str = EXPORTER) -> SimpleNamespace:
    return SimpleNamespace(
        state={},
        dcgm_exporter_image=image,
        config=SimpleNamespace(release_delivery_identity={}),
    )


def _observed(**versions: str | None) -> dict[str, dict[str, Any]]:
    return {
        node: {
            "mode": "host",
            "state": "host-active",
            "conflict": False,
            **({"version": version} if version is not None else {}),
        }
        for node, version in versions.items()
    }


def test_host_mode_records_the_versions_and_warns_about_newer_engines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    release = _release()
    target = _target("eks-managed")

    record = DCGM.record_node_dcgm_states(
        release,
        target,
        _observed(**{"node-a": "4.7.0", "node-b": "4.6.1", "node-c": None}),
        now=NOW,
    )

    assert record is not None
    assert record["nodes"] == {
        "node-a": {"state": "host-active", "conflict": False, "version": "4.7.0"},
        "node-b": {"state": "host-active", "conflict": False, "version": "4.6.1"},
        "node-c": {"state": "host-active", "conflict": False},
    }, "a version is kept only where the preflight read one"
    assert record["exporter_client_version"] == "4.6.1"
    assert record["hostengine_newer_than_client_nodes"] == ["node-a"]
    assert release.state[DCGM.NODE_DCGM_STATE_KEY]["gpu-eks-managed"] == record
    assert capsys.readouterr().err.splitlines() == [
        "WARN  DCGM host engine newer than the exporter's DCGM client 4.6.1 on "
        "gpu-eks-managed: node-a 4.7.0; NVIDIA requires the exporter's DCGM client "
        "library to be at least the host engine version (https://docs.nvidia.com/"
        "datacenter/dcgm/latest/installation/install-dcgm-exporter.html"
        "#requirements-and-version-compatibility); install DCGM 4.6.1 on the node "
        "group (gpu-fault-admin prepare-eks-nodegroup --dcgm-hostengine host "
        "--dcgm-version 4.6.1) or deploy an exporter whose DCGM is at least the "
        "engine's (administrator operations §5.0.1)"
    ], "one advisory line naming the node, the versions, NVIDIA's rule and the exit"


def test_a_hyperpod_engine_newer_than_the_client_gets_the_ami_remedy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = DCGM.record_node_dcgm_states(
        _release(),
        _target("hyperpod"),
        _observed(**{"hyperpod-i-1": "4.7.0", "hyperpod-i-2": "4.6.0"}),
        now=NOW,
    )

    assert record is not None
    assert record["hostengine_newer_than_client_nodes"] == ["hyperpod-i-1"], (
        "4.6.0 < 4.6.1 meets the connection requirement: no warning for it"
    )
    err = capsys.readouterr().err
    assert "the AMI ships the engine: deploy an exporter whose DCGM is at least" in err
    assert "prepare-eks-nodegroup" not in err, "HyperPod has no node group pin"


def test_nothing_is_compared_without_versions_or_in_embedded_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    older = DCGM.record_node_dcgm_states(
        _release(), _target("eks-managed"), _observed(**{"node-a": None}), now=NOW
    )
    embedded = DCGM.record_node_dcgm_states(
        _release(),
        _target("eks-managed", mode=None),
        _observed(**{"node-a": "4.7.0"}),
        now=NOW,
    )

    assert older is not None and "exporter_client_version" not in older, (
        "a preflight script that reads no version (an older bundle) compares nothing"
    )
    assert embedded is not None and "exporter_client_version" not in embedded, (
        "the embedded exporter dials no host engine"
    )
    assert embedded["nodes"]["node-a"]["version"] == "4.7.0", (
        "the version is still recorded: it is evidence about the node"
    )
    assert capsys.readouterr().err == ""


def test_an_exporter_without_a_readable_client_version_warns_about_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = DCGM.record_node_dcgm_states(
        _release("mirror.example/dcgm-exporter@sha256:" + "f" * 64),
        _target("eks-managed"),
        _observed(**{"node-a": "4.7.0"}),
        now=NOW,
    )

    assert record is not None
    assert record["exporter_client_version"] == "unknown"
    assert "hostengine_newer_than_client_nodes" not in record
    assert capsys.readouterr().err == ""


def test_status_shows_the_recorded_versions_and_newer_engines() -> None:
    target = SimpleNamespace(
        cluster_id="gpu-eks",
        provider="eks-managed",
        hyperpod_cluster_name="",
        node_selector="gpu-fault.io/control-plane=site-a",
        node_selector_source="label-convention",
        dcgm_hostengine="host",
        dcgm_hostengine_source="node-group-label",
    )
    state = {
        DCGM.NODE_DCGM_STATE_KEY: {
            "gpu-eks": {
                "mode": "host",
                "observed_at": "observed",
                "nodes": {
                    "node-a": {
                        "state": "host-active",
                        "conflict": False,
                        "version": "4.7.0",
                    },
                    "node-b": {"state": "host-active", "conflict": False},
                },
                "exporter_client_version": "4.6.1",
                "hostengine_newer_than_client_nodes": ["node-a"],
            }
        }
    }
    release = SimpleNamespace(
        state=state,
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=[]),
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=lambda _arguments: {"items": []},
    )

    details = COVERAGE.node_scope_details(
        release, target, target.node_selector, {"items": []}
    )

    assert details["dcgm_hostengine_versions"] == {"node-a": "4.7.0"}
    assert details["dcgm_exporter_client_version"] == "4.6.1"
    assert details["dcgm_hostengine_newer_than_client_nodes"] == ["node-a"]
    assert details["dcgm_hostengine_conflict_nodes"] == [], (
        "a newer engine is advisory: it is no conflict and fails no check"
    )
