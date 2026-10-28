"""A host probe call with no response names the channel and the node.

Live 2026-10-27 (plain EKS, GF-GENERAL-LIVE-030 ``not_ready``): the window's
cleanup went over ``kubectl exec`` while the spare's kubelet was stopped -- the
API server hands an exec to the node's own kubelet -- and failed with only
"host probe response is missing (exit 1)", which said nothing about the node.
The message now names the probe Pod and node, where kubectl said the channel
broke (a fixed label, never kubectl's stderr: it carries the probe's own
stderr too) and the node's Ready condition, with a hint that follows from it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as hp
from tests.regional._host_probe_cluster import ProbeCluster
from tests.regional._host_probe_support import host_probe

#: kubectl's own words when the API server cannot dial a stopped kubelet.
KUBELET_DOWN = (
    "Error from server: error dialing backend: dial tcp 10.0.32.162:10250: "
    "connect: connection refused\n"
)


class SilentNode(ProbeCluster):
    """The cluster with a Ready condition on the node and a silent exec."""

    def __init__(self) -> None:
        super().__init__()
        self.ready = "Unknown"
        self.exec_stderr = KUBELET_DOWN

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        result = super().run(command, **kwargs)
        arguments = list(command[7:])
        if arguments[:2] == ["get", "node"]:
            value = json.loads(result.stdout)
            value["status"] = {
                "conditions": [
                    {"type": "MemoryPressure", "status": "False"},
                    {"type": "Ready", "status": self.ready},
                ]
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
        if arguments[0] == "exec" and "host-probe" in arguments and not result.stdout:
            return subprocess.CompletedProcess(command, 1, "", self.exec_stderr)
        return result


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> SilentNode:
    api = SilentNode()
    monkeypatch.setattr(hp, "run_fixture_command", api.run)
    return api


@pytest.mark.parametrize(
    ("stderr", "label"),
    [
        (KUBELET_DOWN, "the API server could not reach the node's kubelet "
         "(connection refused: kubelet is not listening)"),
        ("error dialing backend: dial tcp 10.0.0.1:10250: i/o timeout",
         "the API server could not reach the node's kubelet (timed out)"),
        ("error dialing backend: dial tcp 10.0.0.1:10250: connect: no route to host",
         "the API server could not reach the node's kubelet (no route to the node)"),
        ("Error from server: error dialing backend: EOF",
         "the API server could not reach the node's kubelet"),
        ("Traceback ... FileNotFoundError: not found\ncommand terminated with exit "
         "code 1", "the probe command ran and ended without printing a response"),
        ('error: unable to upgrade connection: container not found ("probe")',
         "the node's kubelet does not know the probe container"),
        ("error: unable to upgrade connection: Forbidden",
         "the node's kubelet refused the exec stream"),
        ("Unable to connect to the server: dial tcp: lookup api: no such host",
         "kubectl could not reach the API server"),
        ('Error from server (NotFound): pods "gpu-fault-host-probe-x" not found',
         "the probe Pod was not found"),
        ("error: read: connection reset by peer", "the exec stream was cut"),
        ("E1027 http2: stream error: stream ID 3; INTERNAL_ERROR",
         "the exec stream was cut"),
        ("error: context deadline exceeded", "the exec stream timed out"),
        ("", "kubectl reported no error"),
        ("  \n", "kubectl reported no error"),
        ("something else entirely", "an unclassified kubectl error (withheld)"),
    ],
)  # fmt: skip
def test_kubectl_exec_failures_map_to_fixed_labels(stderr: str, label: str) -> None:
    assert hp.exec_transport_failure(stderr) == label


def test_the_live_shape_names_the_pod_the_node_and_the_stopped_kubelet(
    tmp_path: Path, cluster: SilentNode
) -> None:
    probe = host_probe(tmp_path, node="ip-10-0-32-162.us-west-2.compute.internal")
    probe.create()
    cluster.probe_stdout = ""
    cluster.probe_returncode = 1

    with pytest.raises(hp.HostProbeMissingResponseError) as missing:
        probe.execute("cleanup")

    message = str(missing.value)
    assert message == (
        "host probe response is missing (exit 1): no output came back over "
        f"kubectl exec into {probe.pod} on "
        "ip-10-0-32-162.us-west-2.compute.internal; the API server could not "
        "reach the node's kubelet (connection refused: kubelet is not "
        "listening); node Ready=Unknown (kubectl exec is served by that node's "
        "own kubelet, which is not posting Ready: no response can arrive until "
        "kubelet is back)"
    ), message
    assert missing.value.transport == (
        "the API server could not reach the node's kubelet "
        "(connection refused: kubelet is not listening)"
    )
    assert missing.value.node_ready == "Unknown"
    assert "10250" not in message and "10.0.32.162" not in message, (
        "kubectl's stderr itself is never shown"
    )
    assert message.startswith("host probe response is missing (exit 1)"), (
        "the old text stays the prefix every caller matched"
    )


def test_a_ready_node_points_at_the_channel_not_at_kubelet(
    tmp_path: Path, cluster: SilentNode
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    cluster.ready = "True"
    cluster.exec_stderr = "error: read: connection reset by peer"
    cluster.probe_stdout = ""
    cluster.probe_returncode = 1

    with pytest.raises(hp.HostProbeMissingResponseError) as missing:
        probe.execute("status")

    assert str(missing.value).endswith(
        "; the exec stream was cut; node Ready=True (the node is Ready, so the "
        "exec channel itself failed)"
    ), str(missing.value)
    assert missing.value.node_ready == "True"


def test_a_node_without_a_ready_condition_is_named_unreported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = ProbeCluster()  # its node object carries no status at all
    monkeypatch.setattr(hp, "run_fixture_command", api.run)
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_stdout = ""
    api.probe_returncode = 0

    with pytest.raises(hp.HostProbeMissingResponseError) as missing:
        probe.execute("status")

    message = str(missing.value)
    assert message.startswith("host probe response is missing (exit 0): "), message
    assert "kubectl reported no error; node Ready=unreported" in message, message
