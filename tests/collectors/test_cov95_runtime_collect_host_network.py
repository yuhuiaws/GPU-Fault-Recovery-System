"""Host network and process observations from private proc/sys trees."""

from __future__ import annotations

import json

import pytest

from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case


def efa_port(case, name="efa0"):
    device = case.roots.rdma / name
    support.write_file(device / "device/uevent", "PCI_ID=1d0f:efa0\nDRIVER=efa\n")
    port = device / "ports/1"
    port.mkdir(parents=True)
    return device, port


@pytest.mark.parametrize("rx_tx", [False, True])
@pytest.mark.parametrize("link_state", ["4: ACTIVE", "1: DOWN"])
def test_rdma_reports_traffic_once_and_preserves_latest_zero_error_readings(
    host_case, rx_tx, link_state
):
    _, port = efa_port(host_case)
    support.write_file(port / "state", link_state)
    support.write_file(port / "counters/symbol_error", "2")
    support.write_file(port / "hw_counters/poll_cq_err", "bad")
    support.write_file(port / "hw_counters/packet_seq_err", "2")
    counters = ["send_bytes", "recv_bytes"] + (
        ["rx_bytes", "tx_bytes"] if rx_tx else []
    )
    for name in counters:
        support.write_file(port / "hw_counters" / name, "100")
    support.write_file(port / "hw_counters/send_wrs", "unavailable")
    support.write_file(host_case.roots.rdma / "other/device/uevent", "DRIVER=mlx5")
    collector = host_case.build("_rdma")
    collector.collect_rdma_samples()
    host_case.clock.sleep(10)
    for name in counters:
        (port / "hw_counters" / name).write_text("200")
    (port / "counters/symbol_error").write_text("5")

    batch = collector.collect_once()
    values = support.values(batch)
    assert batch.collection_errors == []
    assert values[("efa_traffic_bytes_delta", None)] == 200
    assert values[("efa_traffic_bytes_per_second", None)] == 20
    assert values[("rdma_errors_delta", "efa0/1")] == 3
    assert values[("rdma_link_down", "efa0/1")] == (
        0 if link_state.startswith("4") else 1
    )
    host_case.clock.sleep(10)
    quiet = support.values(collector.collect_once())
    assert quiet[("rdma_errors_delta", "efa0/1")] == 0
    assert quiet[("efa_traffic_bytes_per_second", None)] == 0
    assert ("rdma_symbol_error_delta", "efa0/1") not in quiet


@pytest.mark.parametrize("exit_code", [0, 1])
def test_ethtool_skips_bad_counters_without_losing_congestion_zeroes(
    host_case, exit_code
):
    support.enable_tools(host_case, "ethtool")
    device, _ = efa_port(host_case)
    (device / "device/net/ens1").mkdir(parents=True)
    support.write_file(host_case.roots.rdma / "unrelated/device/uevent", "DRIVER=other")
    argv = ("ethtool", "-S", "ens1")
    host_case.outputs[argv] = (
        "header\nrnr_bad: bad\nrx_rnr: 2\npfc: 3\nother: 5\n",
        "",
        exit_code,
    )
    collector = host_case.build("_efa_network")
    collector.collect_once()
    host_case.clock.sleep(10)
    host_case.outputs[argv] = (
        "header\nrnr_bad: bad\nrx_rnr: 9\npfc: 3\nother: 6\n",
        "",
        exit_code,
    )

    batch = collector.collect_once()
    values = support.values(batch)
    if exit_code:
        assert values == {}
    else:
        assert values[("efa_rnr_errors_delta", "ens1")] == 7
        assert values[("network_pfc_pause_delta", "ens1")] == 0
        assert values[("network_ecn_marks_delta", "ens1")] == 0
    assert all(options["timeout"] == 10 for _, options in host_case.commands), (
        "every fake ethtool request must carry the production timeout"
    )


def test_network_listing_failure_does_not_synthesize_link_down(
    host_case, monkeypatch, caplog
):
    original = type(host_case.roots.net).iterdir

    def fail(path):
        if path == host_case.roots.net:
            raise OSError("interface listing unavailable")
        return original(path)

    monkeypatch.setattr(type(host_case.roots.net), "iterdir", fail)
    batch = host_case.build("_network", required_interfaces=["ens1"]).collect_once()
    assert batch.samples == []
    assert "interface listing unavailable" in caplog.text


@pytest.mark.parametrize("present", [False, True])
def test_tcp_ignores_unrelated_sections_and_emits_delta_for_tcp_only(
    host_case, present
):
    content = "\n\nUdp: A\nUdp: 2\n"
    if present:
        content += "Tcp: RetransSegs\nTcp: 3\n"
    path = host_case.roots.proc / "net/snmp"
    support.write_file(path, content)
    collector = host_case.build("_tcp")
    assert collector.collect_once().samples == []
    host_case.clock.sleep(2)
    path.write_text(content.replace("Tcp: 3", "Tcp: 8"))
    values = support.values(collector.collect_once())
    assert values == ({("tcp_retransmits_delta", None): 5} if present else {})


@pytest.mark.parametrize(
    "result", [OSError("fake GPU process query"), ("", "driver unavailable", 1)]
)
def test_failed_rank_discovery_warns_once_and_never_invents_live_ranks(
    host_case, result, caplog
):
    argv = ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    host_case.outputs[argv] = result
    collector = host_case.build("_rank_liveness")
    assert collector.collect_once().samples == []
    host_case.clock.sleep(1)
    assert collector.collect_once().samples == []
    assert caplog.text.count("rank liveness probe cannot list compute apps") == 1


@pytest.mark.parametrize("stat", ["", "23 (name) S 0", "23 (name) " + "bad " * 20])
def test_disappearing_or_corrupt_rank_stat_is_not_a_progress_sample(host_case, stat):
    host_case.outputs[
        ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    ] = ("23\n23\nN/A\n", "", 0)
    support.write_file(host_case.roots.proc / "23/stat", stat)
    values = support.values(host_case.build("_rank_liveness").collect_once())
    assert values[("training_rank_process_count", None)] == 0
    assert values[("training_rank_advancing_count", None)] == 0
    assert ("training_rank_seconds_since_progress", None) not in values


@pytest.mark.parametrize("enabled", [False, True])
def test_rank_liveness_disabled_mode_does_not_query_gpu_processes(host_case, enabled):
    host_case.monkeypatch.setenv(
        "GPU_FAULT_RANK_LIVENESS_ENABLED", str(enabled).lower()
    )
    batch = host_case.build("_rank_liveness").collect_once()
    assert batch.samples == []
    assert len(host_case.commands) == int(enabled)


@pytest.mark.parametrize(
    ("payload", "code", "expected_error"),
    [
        ({"ports": {}}, 0, "ports array"),
        ({}, 1, "topology query failed"),
        ([None, {"switch_id": "0", "port": "2", "link_scope": "UNKNOWN"}], 0, None),
        (
            [
                {
                    "switch_id": "0",
                    "port": "2",
                    "link_scope": "ACCESS",
                    "peer_type": "GPU",
                    "gpu_uuid": "GPU-a",
                    "fabric_partition": "fabric-a",
                }
            ],
            0,
            None,
        ),
        ({"ports": [{"switch_id": "0", "port": "2", "link_scope": "TRUNK"}]}, 0, None),
    ],
)
def test_topology_command_requires_a_valid_port_before_publishing_trusted_evidence(
    host_case, payload, code, expected_error
):
    host_case.outputs[("private-topology",)] = (
        json.dumps(payload),
        "fake refusal",
        code,
    )
    collector = host_case.build(
        "_nvswitch_topology", nvswitch_topology_command=["private-topology"]
    )
    batch = collector.collect_once()
    if expected_error:
        assert batch.samples == []
        assert expected_error in batch.collection_errors[0]
    elif isinstance(payload, list) and payload[0] is None:
        assert batch.samples == []
        assert batch.collection_errors == []
    else:
        assert len(batch.samples) == 1
        assert batch.samples[0].labels["trusted"] == "true"
        assert batch.samples[0].device == "0/2"
    assert host_case.commands[0][1]["timeout"] == 15
