"""``status --full`` shows every node's host DCGM engine live.

Each Agent reports ``nvidia-dcgm`` in its heartbeat, so the Agent records the
``control_api`` probe already reads carry the engine's state as of the last
heartbeat (seconds old) -- unlike ``node_scope.dcgm_hostengine_nodes``, the
node preflight's observation of the last deploy or join. The check records
``dcgm_hostengine_live`` per cluster, judged by the cluster's mode, and the
status header names the host-mode nodes whose engine is down and the
embedded-mode nodes running one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release.regional_admin_checks_coverage import (
    annotate_dcgm_hostengine_live,
    dcgm_hostengine_lines,
    dcgm_hostengine_live,
)

SEEN = "2030-01-02T03:04:05+00:00"


def _agent(node: str, engine: dict[str, str] | None, *, state: str = "ACTIVE"):
    services: dict[str, Any] = {
        "gpu-fault-kernel-collector": {"active": "active", "enabled": "enabled"}
    }
    if engine is not None:
        services["nvidia-dcgm"] = engine
    return {
        "node_id": node,
        "lifecycle_state": state,
        "last_seen_at": SEEN,
        "collector_services": services,
    }


def _cluster() -> dict[str, Any]:
    return {
        "agents": [
            _agent("node-a", {"active": "active", "enabled": "enabled"}),
            _agent("node-b", {"active": "failed", "enabled": "enabled"}),
            _agent("node-c", None),
            _agent("node-d", {"active": "inactive", "enabled": "enabled"}),
            _agent(
                "node-e", {"active": "failed", "enabled": "enabled"}, state="REVOKED"
            ),
        ]
    }


def test_host_mode_names_every_node_whose_engine_is_down() -> None:
    live = dcgm_hostengine_live(_cluster(), "host")

    assert live["mode"] == "host" and live["unit"] == "nvidia-dcgm"
    assert live["nodes"] == {
        "node-a": {"active": "active", "enabled": "enabled", "reported_at": SEEN},
        "node-b": {"active": "failed", "enabled": "enabled", "reported_at": SEEN},
        "node-c": {"active": "not reported", "reported_at": SEEN},
        "node-d": {"active": "inactive", "enabled": "enabled", "reported_at": SEEN},
    }, "a retired Agent is no coverage; an older Agent reports nothing"
    assert live["down_nodes"] == ["node-b", "node-d"]
    assert live["not_reported_nodes"] == ["node-c"]
    assert "conflict_nodes" not in live


def test_embedded_mode_names_a_host_engine_next_to_the_exporters() -> None:
    cluster = {
        "agents": [
            _agent("node-a", {"active": "inactive", "enabled": "disabled"}),
            _agent("node-b", {"active": "active", "enabled": "enabled"}),
            _agent("node-c", {"active": "inactive", "enabled": "enabled"}),
            _agent("node-d", {"active": "inactive", "enabled": "unknown"}),
        ]
    }

    live = dcgm_hostengine_live(cluster, "embedded")

    assert live["conflict_nodes"] == ["node-b", "node-c"], (
        "running, or starting at boot, as the node preflight's conflict check"
    )
    assert "down_nodes" not in live


def test_each_cluster_is_judged_by_its_own_mode() -> None:
    report = {
        "clusters": {
            "eks-host": _cluster(),
            "eks-embedded": _cluster(),
            "hyperpod": _cluster(),
        }
    }
    targets = [
        SimpleNamespace(
            cluster_id="eks-host", provider="eks-managed", dcgm_hostengine="host"
        ),
        SimpleNamespace(cluster_id="eks-embedded", provider="eks-managed"),
        SimpleNamespace(cluster_id="hyperpod", provider="hyperpod"),
        SimpleNamespace(cluster_id="not-in-report", provider="hyperpod"),
    ]

    annotate_dcgm_hostengine_live(
        report, SimpleNamespace(config=SimpleNamespace(clusters=targets))
    )

    modes = {
        cluster_id: cluster["dcgm_hostengine_live"]["mode"]
        for cluster_id, cluster in report["clusters"].items()
    }
    assert modes == {"eks-host": "host", "eks-embedded": "embedded", "hyperpod": "host"}
    assert dcgm_hostengine_lines(report) == [
        "DCGM host engine next to the embedded one on eks-embedded (Agent "
        "heartbeats): node-a active/enabled, node-b failed/enabled, "
        "node-d inactive/enabled",
        "DCGM host engine down on eks-host (host mode, Agent heartbeats): "
        "node-b failed/enabled, node-d inactive/enabled",
        "DCGM host engine down on hyperpod (host mode, Agent heartbeats): "
        "node-b failed/enabled, node-d inactive/enabled",
    ]
    quiet = {"clusters": {"a": {"dcgm_hostengine_live": {"down_nodes": []}}}}
    assert dcgm_hostengine_lines(quiet) == [] and dcgm_hostengine_lines({}) == []


def test_the_status_header_prints_the_live_findings() -> None:
    cluster = _cluster()
    cluster["dcgm_hostengine_live"] = dcgm_hostengine_live(cluster, "host")
    report = {
        "healthy": True,
        "health_scope": "full",
        "live_release": {"release_id": "release-a", "phase": "complete"},
        "next_deploy": {"kind": "NOOP"},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [
                {
                    "name": "control_api",
                    "status": "PASS",
                    "details": {"clusters": {"gpu-a": cluster}},
                }
            ],
        },
    }

    lines = ADMIN.status_header_lines(report, json_destination="stdout")

    assert lines[3] == "failing checks: none"
    assert lines[4] == (
        "DCGM host engine down on gpu-a (host mode, Agent heartbeats): "
        "node-b failed/enabled, node-d inactive/enabled"
    )
    assert lines[-1] == "full JSON report: stdout"


def test_the_control_api_report_carries_the_live_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = {
        "metadata": {"name": "node-a"},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    probe = {"healthz": {"status": "ok"}, "clusters": {"eks-host": _cluster()}}
    asked: list[dict[str, Any]] = []
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, _target: ("selector", {"items": [ready]}),
    )

    def exec_probe(_release: Any, **options: Any) -> str:
        asked.append(options)
        return json.dumps(probe)

    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", exec_probe)
    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=[
                SimpleNamespace(
                    cluster_id="eks-host",
                    provider="eks-managed",
                    dcgm_hostengine="host",
                )
            ]
        )
    )

    report = CHECKS._control_api_report(release)

    live = report["clusters"]["eks-host"]["dcgm_hostengine_live"]
    assert (live["mode"], live["down_nodes"]) == ("host", ["node-b", "node-d"])
    assert report["clusters"]["eks-host"]["kubernetes_node_ids"] == ["node-a"]
    assert len(asked) == 1, "no extra exec: the Agent records are already read"


# --- the engine's version against the exporter's DCGM client (batch 135) ------
#
# NVIDIA requires the exporter's DCGM client library to be at least the host
# engine's version. The Agent reports the running engine's version
# (``nvidia-dcgm/version=<v>``); host mode compares it with the client of the
# release's exporter (the DCGM half of its image tag). Advisory: a header line,
# no verdict.
CLIENT_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless"
DIGEST = "sha256:" + "d" * 64


def _versioned(node: str, version: str | None) -> dict[str, Any]:
    agent = _agent(node, {"active": "active", "enabled": "enabled"})
    if version is not None:
        agent["collector_services"][f"nvidia-dcgm/version={version}"] = {
            "active": "active",
            "enabled": "enabled",
        }
    return agent


def _version_cluster() -> dict[str, Any]:
    return {
        "agents": [
            _versioned("node-a", "4.7.0"),
            _versioned("node-b", "4.6.1"),
            _versioned("node-c", "4.6.0"),
            _versioned("node-d", None),
        ]
    }


def test_host_mode_names_engines_newer_than_the_exporters_client() -> None:
    live = dcgm_hostengine_live(_version_cluster(), "host", "4.6.1")

    assert {name: entry.get("version") for name, entry in live["nodes"].items()} == {
        "node-a": "4.7.0",
        "node-b": "4.6.1",
        "node-c": "4.6.0",
        "node-d": None,
    }, "each node's running engine as its Agent reports it; an older Agent none"
    assert live["exporter_client_version"] == "4.6.1"
    assert live["newer_than_client_nodes"] == ["node-a"], (
        "4.7.0 > 4.6.1 breaks NVIDIA's connection requirement"
    )
    assert live["unpaired_nodes"] == ["node-c"], (
        "4.6.0 < 4.6.1 meets the requirement but is no supported pair"
    )
    assert live["down_nodes"] == []
    unknown = dcgm_hostengine_live(_version_cluster(), "host")
    assert unknown["exporter_client_version"] == "unknown"
    assert unknown["newer_than_client_nodes"] == [], (
        "an exporter whose client version is unknown judges no node"
    )
    embedded = dcgm_hostengine_live(_version_cluster(), "embedded", "4.6.1")
    assert "newer_than_client_nodes" not in embedded, (
        "the embedded exporter runs its own engine: no host engine is its peer"
    )


def test_the_status_header_names_engines_newer_than_the_client() -> None:
    cluster = _version_cluster()
    cluster["dcgm_hostengine_live"] = dcgm_hostengine_live(cluster, "host", "4.6.1")

    assert dcgm_hostengine_lines({"clusters": {"gpu-eks": cluster}}) == [
        "DCGM host engine newer than the exporter's DCGM client 4.6.1 on gpu-eks "
        "(host mode, Agent heartbeats): node-a 4.7.0; NVIDIA requires the client "
        "to be at least the host engine version (https://docs.nvidia.com/"
        "datacenter/dcgm/latest/installation/install-dcgm-exporter.html"
        "#requirements-and-version-compatibility; administrator operations §5.0.1)"
    ], "one advisory line; the unpaired 4.6.0 node is in the details only"


@pytest.mark.parametrize(
    ("image", "identity", "client"),
    [
        pytest.param(CLIENT_IMAGE, {}, "4.6.1", id="tag"),
        pytest.param(
            f"nvcr.io/nvidia/k8s/dcgm-exporter@{DIGEST}",
            {
                "images": {
                    "dcgm_exporter": {
                        "reference": f"nvcr.io/nvidia/k8s/dcgm-exporter@{DIGEST}",
                        "source": CLIENT_IMAGE,
                    }
                }
            },
            "4.6.1",
            id="locked-digest",
        ),
        pytest.param(
            f"mirror.example/dcgm-exporter@{DIGEST}",
            {
                "images": {
                    "dcgm_exporter": {
                        "reference": f"nvcr.io/nvidia/k8s/dcgm-exporter@{DIGEST}",
                        "source": CLIENT_IMAGE,
                    }
                }
            },
            "4.6.1",
            id="mirror-of-the-locked-digest",
        ),
        pytest.param(
            "mirror.example/dcgm-exporter@sha256:" + "e" * 64,
            {
                "images": {
                    "dcgm_exporter": {
                        "reference": f"nvcr.io/nvidia/k8s/dcgm-exporter@{DIGEST}",
                        "source": CLIENT_IMAGE,
                    }
                }
            },
            None,
            id="another-digest",
        ),
        pytest.param("nvcr.io/nvidia/k8s/dcgm-exporter:4.8.4", {}, None, id="bare"),
    ],
)
def test_the_client_is_read_from_the_release_exporter_image(
    image: str, identity: dict[str, Any], client: str | None
) -> None:
    report = {"clusters": {"eks-host": _version_cluster()}}
    release = SimpleNamespace(
        dcgm_exporter_image=image,
        config=SimpleNamespace(
            release_delivery_identity=identity,
            clusters=[
                SimpleNamespace(
                    cluster_id="eks-host",
                    provider="eks-managed",
                    dcgm_hostengine="host",
                )
            ],
        ),
    )

    annotate_dcgm_hostengine_live(report, release)

    live = report["clusters"]["eks-host"]["dcgm_hostengine_live"]
    assert live["exporter_client_version"] == (client or "unknown"), (
        "the DCGM half of the exporter's tag, through the image lock's source "
        "tag when the release pins the locked digest"
    )
    assert live["newer_than_client_nodes"] == (["node-a"] if client else []), live
