from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pytest

from scripts.e2e.regional import collector_acceptance_fixture
from scripts.e2e.regional import run_collect016_training_recovery as collect016
from scripts.e2e.regional import run_collect017_efa_plugin as collect017
from scripts.e2e.regional import run_collector_acceptance as collect
from scripts.e2e.regional import run_collector_destructive as collect_destructive
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e002
from scripts.e2e.regional import run_iso006_cluster_offline as iso006
from scripts.e2e.regional.collector_acceptance_fixture import collector_setting
from scripts.e2e.regional.multi_cluster_fixture import (
    ClusterTarget,
    MultiClusterSettings,
    registrations_are_distinct_physical_clusters,
)
from scripts.e2e.regional.probes import cluster_network_probe, collector_node_probe
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

ROOT = Path(__file__).resolve().parents[2]


def _kubeconfig(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("apiVersion: v1\n", encoding="utf-8")
    return path


def test_collector_case_registries_cover_the_requested_order() -> None:
    expected_low_risk = {
        "GF-REGIONAL-COLLECT-001",
        "GF-REGIONAL-COLLECT-002",
        "GF-REGIONAL-COLLECT-003",
        "GF-REGIONAL-COLLECT-005",
        "GF-REGIONAL-COLLECT-009",
        "GF-REGIONAL-COLLECT-010",
        "GF-REGIONAL-COLLECT-011",
        "GF-REGIONAL-COLLECT-012",
    }
    expected_destructive = {
        "GF-REGIONAL-COLLECT-004",
        "GF-REGIONAL-COLLECT-008",
        "GF-REGIONAL-COLLECT-013",
        "GF-REGIONAL-COLLECT-014",
        "GF-REGIONAL-COLLECT-015",
    }

    assert set(collect.CASE_IDS) == expected_low_risk, collect.CASE_IDS
    assert set(collect_destructive.CASE_IDS) == expected_destructive, (
        collect_destructive.CASE_IDS
    )
    assert collect016.CASE_ID == "GF-REGIONAL-COLLECT-016", collect016.CASE_ID
    assert collect017.CASE_ID == "GF-REGIONAL-COLLECT-017", collect017.CASE_ID


@pytest.mark.parametrize("case_id", collect.CASE_IDS)  # type: ignore[untyped-decorator]
def test_collector_runner_is_plan_only_with_case_specific_confirmation(
    tmp_path: Path, case_id: str
) -> None:
    nodes = ["node-a", "node-b"] if case_id.endswith("-011") else ["node-a"]
    arguments = collect.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--case",
            case_id,
            *[value for node in nodes for value in ("--node", node)],
        ]
    )

    assert arguments.execute is False, case_id
    assert collect.CONFIRMATIONS[case_id].endswith("_EXECUTE"), case_id


def test_collector_host_probe_has_narrow_action_allowlists() -> None:
    assert collector_node_probe.ALLOWED_XIDS == {13, 31, 46, 48, 54, 62, 63, 78, 109}, (
        collector_node_probe.ALLOWED_XIDS
    )
    assert 99999 in collector_node_probe.ALLOWED_SXIDS, (
        collector_node_probe.ALLOWED_SXIDS
    )
    assert collector_node_probe.ALLOWED_SERVICES == {
        "gpu-fault-dcgm-collector.service",
        "gpu-fault-fabric-manager-collector.service",
        "gpu-fault-host-collector.service",
        "gpu-fault-kernel-collector.service",
    }, collector_node_probe.ALLOWED_SERVICES


def test_collector_probe_arms_the_power_limit_restore_before_capping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # COLLECT-002 needs a real correlated power throttle, which means leaving the
    # node with a lowered enforced power limit for a couple of minutes. If the
    # runner dies in that window nothing else will put the limit back, so the
    # deadman timer has to exist before the cap is applied.
    calls: list[list[str]] = []
    query = (
        "0, GPU-a, 128.0, 700.0, 200.0, 700.0, 0\n"
        "1, GPU-b, 127.0, 700.0, 200.0, 700.0, 0\n"
    )

    def fake_run(command: list[str], *, check: bool = True, timeout: int = 180):
        del check, timeout
        calls.append(command)
        stdout = (
            query if command[0] == "nvidia-smi" and "--format" in command[2] else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(collector_node_probe, "run", fake_run)
    monkeypatch.setattr(
        collector_node_probe, "proftester_binary", lambda: "/usr/bin/dcgmproftester13"
    )

    collector_node_probe.throttle_gpu(
        argparse.Namespace(
            run_id="collect002-run-a",
            gpu_index=0,
            load_seconds=180,
            restore_seconds=600,
        )
    )

    timer_index = next(
        index
        for index, command in enumerate(calls)
        if command[0] == "systemd-run" and "--on-active=600s" in command
    )
    cap_index = next(
        index
        for index, command in enumerate(calls)
        if command[:2] == ["nvidia-smi", "-pl"]
    )
    assert timer_index < cap_index
    # The cap is the driver's own minimum, and the deadman restores the default.
    assert calls[cap_index] == ["nvidia-smi", "-pl", "200"]
    assert calls[timer_index][-3:] == ["/usr/bin/nvidia-smi", "-pl", "700"]


def test_collector_probe_refuses_a_load_that_outlives_its_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], *, check: bool = True, timeout: int = 180):
        del check, timeout
        return subprocess.CompletedProcess(
            command, 0, stdout="0, GPU-a, 128.0, 700.0, 200.0, 700.0, 0\n", stderr=""
        )

    monkeypatch.setattr(collector_node_probe, "run", fake_run)

    with pytest.raises(collector_node_probe.ProbeError, match="deadman"):
        collector_node_probe.throttle_gpu(
            argparse.Namespace(
                run_id="collect002-run-a",
                gpu_index=0,
                load_seconds=900,
                restore_seconds=600,
            )
        )


def test_collector_probe_only_reads_env_keys_the_installer_writes() -> None:
    # The COLLECT group's whole claim is that it judges against the values the
    # node is really running with, so every key it asks `collector.env` for has
    # to be one the installer actually writes there. `run_collector_acceptance`
    # read `GPU_FAULT_DCGM_INTERVAL_SECONDS` -- a name that appears nowhere in
    # the installer or in `collectors_cli` -- and the case died on a bare
    # KeyError against a live node instead of at review time.
    installer = (ROOT / "deploy/node/install-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )
    written = set(re.findall(r"write_env\s+(GPU_FAULT_[A-Z0-9_]+)", installer))
    missing = sorted(collector_node_probe.ENV_KEYS - written)
    assert not missing, (
        "the collector probe reads collector.env keys the node installer never "
        f"writes: {missing}"
    )


def test_collector_setting_names_the_missing_key() -> None:
    with pytest.raises(RegionalFixtureError) as caught:
        collector_setting({"GPU_FAULT_HOST_INTERVAL_SECONDS": "15"}, "GPU_FAULT_ABSENT")
    message = str(caught.value)
    assert "GPU_FAULT_ABSENT" in message, message
    # And what the node did have, because the next question after "which key"
    # is always "so what is on the node".
    assert "GPU_FAULT_HOST_INTERVAL_SECONDS" in message, message


def test_cluster_network_probe_chain_and_restore_unit_are_deterministic() -> None:
    first = cluster_network_probe.chain_name("iso006-run-a")
    second = cluster_network_probe.chain_name("iso006-run-a")

    assert first == second, (first, second)
    assert len(first) <= 28, first
    assert cluster_network_probe.restore_unit("iso006-run-a").startswith(
        "gpu-fault-network-restore-"
    ), cluster_network_probe.restore_unit("iso006-run-a")


def test_cluster_network_probe_arms_restore_before_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        del check
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_network_probe, "run", fake_run)

    cluster_network_probe.block(
        argparse.Namespace(
            run_id="iso006-run-a",
            control_plane_cidr=["10.0.0.0/24"],
            restore_seconds=180,
        )
    )

    timer_index = next(
        index for index, command in enumerate(calls) if command[0] == "systemd-run"
    )
    block_index = next(
        index
        for index, command in enumerate(calls)
        if command[:4] == ["iptables", "-I", "OUTPUT", "1"]
    )
    assert timer_index < block_index


def test_cluster_network_probe_cleans_partial_block_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        calls.append(command)
        if command[:2] == ["iptables", "-A"]:
            raise cluster_network_probe.ProbeError("synthetic rule failure")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cluster_network_probe, "run", fake_run)

    with pytest.raises(cluster_network_probe.ProbeError, match="synthetic"):
        cluster_network_probe.block(
            argparse.Namespace(
                run_id="iso006-run-a",
                control_plane_cidr=["10.0.0.0/24"],
                restore_seconds=180,
            )
        )

    assert ["iptables", "-F", cluster_network_probe.chain_name("iso006-run-a")] in calls
    assert [
        "systemctl",
        "stop",
        cluster_network_probe.restore_unit("iso006-run-a") + ".timer",
    ] in calls


def test_multi_cluster_fixture_rejects_same_physical_target(tmp_path: Path) -> None:
    cpu = _kubeconfig(tmp_path, "cpu.kubeconfig")
    gpu = _kubeconfig(tmp_path, "gpu.kubeconfig")
    target = ClusterTarget(
        cluster_id="cluster-a", gpu_kubeconfig=gpu, gpu_context="context-a"
    )

    with pytest.raises(ValueError, match="distinct cluster IDs"):
        MultiClusterSettings(
            cpu_kubeconfig=cpu,
            namespace="gpu-fault-system",
            region="us-west-2",
            cluster_a=target,
            cluster_b=target,
        )


def test_multi_cluster_registry_requires_distinct_eks_and_hyperpod_identities() -> None:
    registrations = [
        {
            "cluster_id": "cluster-a",
            "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a",
            "hyperpod_cluster_name": "hyperpod-a",
        },
        {
            "cluster_id": "cluster-b",
            "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-b",
            "hyperpod_cluster_name": "hyperpod-b",
        },
    ]

    assert registrations_are_distinct_physical_clusters(registrations), (
        "distinct EKS and HyperPod identities were rejected"
    )
    assert not registrations_are_distinct_physical_clusters(
        [
            registrations[0],
            {
                **registrations[1],
                "eks_cluster_arn": registrations[0]["eks_cluster_arn"],
            },
        ]
    ), "duplicate EKS identity was accepted as a second physical cluster"
    assert not registrations_are_distinct_physical_clusters(
        [
            registrations[0],
            {
                **registrations[1],
                "hyperpod_cluster_name": registrations[0]["hyperpod_cluster_name"],
            },
        ]
    ), "duplicate HyperPod identity was accepted as a second physical cluster"


def test_multi_cluster_runners_are_plan_only(tmp_path: Path) -> None:
    common = [
        "--run-dir",
        str(tmp_path),
        "--cluster-a",
        "cluster-a",
        "--gpu-a-kubeconfig",
        "/tmp/a.kubeconfig",
        "--gpu-a-context",
        "context-a",
        "--cluster-b",
        "cluster-b",
        "--gpu-b-kubeconfig",
        "/tmp/b.kubeconfig",
        "--gpu-b-context",
        "context-b",
    ]
    iso = iso006.parser().parse_args([*common, "--control-plane-cidr", "10.0.0.0/24"])
    e2e = e2e002.parser().parse_args(common)

    assert iso.execute is False, iso
    assert e2e.execute is False, e2e
    assert iso006.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-013", (
        iso006.PREDECESSOR_CASE_ID
    )
    # The order file names ISO-001 as E2E-002's explicit `predecessor:`; the
    # runner must read that record, not the ISO-006 verdict it used to chain to.
    assert e2e002.PREDECESSOR_CASE_ID == "GF-REGIONAL-ISO-001", (
        e2e002.PREDECESSOR_CASE_ID
    )


def test_collector_promoted_scripts_contain_no_site_specific_topology() -> None:
    paths = (
        ROOT / "scripts/e2e/regional/run_collector_acceptance.py",
        ROOT / "scripts/e2e/regional/run_collector_destructive.py",
        ROOT / "scripts/e2e/regional/run_collect016_training_recovery.py",
        ROOT / "scripts/e2e/regional/run_collect017_efa_plugin.py",
        ROOT / "scripts/e2e/regional/run_iso006_cluster_offline.py",
        ROOT / "scripts/e2e/regional/run_e2e002_multicluster_fault.py",
        # The site profile exists to hold exactly these values, so it is the
        # first place they would leak back into the repository.
        ROOT / "scripts/e2e/regional/site_profile.py",
        ROOT / "scripts/e2e/regional/live_driver_guard.py",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "/secure/gpu-fault-bootstrap" not in source, path
        assert "514385905925" not in source, path
        assert "gpu-fault-gpu-1-" not in source, path


def test_collector_probe_canonicalises_every_bdf_spelling_the_node_emits() -> None:
    """nvidia-smi, the host probe and sysfs spell one device three ways."""

    canonical = "0000:59:00.0"
    assert collector_node_probe.normalize_bdf("00000000:59:00.0") == canonical, (
        "nvidia-smi's eight-digit domain must collapse to the four-digit form"
    )
    assert collector_node_probe.normalize_bdf("0000:59:00") == canonical, (
        "a BDF without the function digit is the same device, function 0"
    )
    assert collector_node_probe.normalize_bdf("0000:59:00.0") == canonical, (
        "the canonical spelling is unchanged"
    )
    assert collector_node_probe.normalize_bdf("0000:5E:00.0") == canonical.replace(
        "59", "5e"
    ), "case is folded before validation"
    for unsafe in ("0000:59:00.0; rm -rf /", "0001:59:00.0", "59:00.0", "0000:59:00.9"):
        with pytest.raises(collector_node_probe.ProbeError, match="unsafe PCI BDF"):
            collector_node_probe.normalize_bdf(unsafe)
    assert collector_node_probe.normalized_bdf_or_raw("garbage") == "garbage", (
        "an inventory line the probe cannot canonicalise is reported as-is"
    )


def test_collector_store_probe_links_workflows_to_marked_events_and_decisions() -> None:
    """A kmsg marker never appears in the workflow record, so text matching is not enough."""

    probe = collector_acceptance_fixture.STORE_PROBE
    assert 'marked_event_ids = {item["event_id"] for item in events}' in probe
    assert "incident.event_id not in marked_event_ids" in probe
    assert "workflow.request_id not in marked_workflow_ids" in probe
    assert probe.index("marked_event_ids") < probe.index(
        "for workflow in store.list_workflows"
    ), "the marked sets must exist before the workflow scan that consults them"


def test_collect011_injects_an_nvswitch_address_that_is_not_a_gpu_slot() -> None:
    """A GPU's own BDF in the SXid line makes the reset executable; the case must not."""

    inventory = [{"pci_bdf": "0000:59:00.0"}, {"pci_bdf": "0000:ab:00.0"}]
    chosen = collect.nvswitch_pci_bdf(inventory)
    assert chosen == "0000:ac:00.0", chosen
    assert chosen.split(".")[0] not in {
        item["pci_bdf"].split(".")[0] for item in inventory
    }, "the injected switch address must never coincide with a GPU slot"
    assert collect.nvswitch_pci_bdf([{"pci_bdf": "0000:59:00.0"}]) == "0000:ab:00.0", (
        "the documented NVSwitch example address is used when it is free"
    )


def test_collector_store_probe_scopes_fabric_manager_workflows_by_injection_time() -> (
    None
):
    """SXID workflows carry neither the marker nor an XID event; the injection time links them."""

    probe = collector_acceptance_fixture.STORE_PROBE
    assert (
        'cluster_id, node_id, marker, observed_after_text = (sys.argv[1:] + [""])[:4]'
        in probe
    )
    assert "workflow.created_at >= observed_after" in probe
    assert "and not injected_since" in probe


def test_collect012_restores_the_quarantine_before_injecting_the_second_xid(
    tmp_path: Path,
) -> None:
    """XID 31 must reach BLOCKED on a node XID 13 no longer owns.

    Samples 1 and 2 write the same XID 13 line under one marker, so the kernel
    collector must give the second write its own kmsg sequence and evidence_ref
    (``kmsg-<boot_id>-<seq>``) and the runner must wait for two records before
    it judges sample 2."""

    calls: list[str] = []
    minimum_evidence_seen: list[int] = []

    class Fixture:
        boot_id = "boot-1"

        def __init__(self) -> None:
            self.sequence = 0
            self.records: dict[str, list[dict]] = {}

        def snapshot(self) -> dict:
            return {
                "gpu_inventory": [{"pci_bdf": "0000:59:00.0"}],
                "boot_id": self.boot_id,
            }

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            calls.append("inject:" + arguments[arguments.index("--xid") + 1])
            return {}

        def wait_marker(
            self, marker: str, *, minimum_evidence: int = 1, **_kwargs: object
        ) -> dict:
            # Every wait follows one more write of the marker's line; the node
            # gives it a fresh kmsg sequence even when the text is identical.
            minimum_evidence_seen.append(minimum_evidence)
            self.sequence += 1
            self.records.setdefault(marker, []).append(
                {
                    "record_id": f"kmsg-{self.boot_id}-{self.sequence}",
                    "evidence_ref": (
                        f"kmsg://hyperpod-node/{self.boot_id}/{self.sequence}"
                    ),
                }
            )
            evidence = list(self.records[marker])
            assert len(evidence) >= minimum_evidence, (marker, evidence)
            return {
                "evidence": evidence,
                "workflows": [{"status": "BLOCKED"}],
                "incidents": [{"incident_id": f"inc-{marker}"}],
            }

        def restore_incidents(self, state: dict, **_kwargs: object) -> list:
            calls.append("restore")
            return [{"status": "SUCCEEDED"}]

    result = collect.run_collect012(Fixture(), tmp_path, 1, "hyperpod-v1")

    assert result["verdict"] == "PASS", result
    assert calls == ["inject:13", "inject:13", "restore", "inject:31", "restore"], calls
    assert len(result["restore_workflows"]) == 2, result["restore_workflows"]
    # Sample 2 alone waits for the second record of the shared marker.
    assert minimum_evidence_seen == [1, 2, 1], minimum_evidence_seen
    assert result["boot_id"] == "boot-1", result
    assert len(result["record_ids"]) == 3, result["record_ids"]
    assert len(result["markers"]) == 3 and result["markers"][0] == result["markers"][1]


def test_destructive_probe_does_not_count_its_own_injected_lines_as_resets() -> None:
    from scripts.e2e.regional.probes import destructive_node_probe

    injected = (
        "gpu-fault GF-REGIONAL-DESTR-001 marker=m drill_id=d NVRM: Xid (PCI:0000:59:00): 46, "
        "GPU stopped processing, reset acceptance event"
    )
    assert (
        destructive_node_probe.counts_as_target_reset(injected, "0000:59:00") is False
    ), "the drill's own kmsg line is injection text, not a kernel reset record"
    assert (
        destructive_node_probe.counts_as_target_reset(
            "NVRM: GPU at PCI:0000:59:00: reset complete", "0000:59:00"
        )
        is True
    )
    assert (
        destructive_node_probe.counts_as_target_reset(
            "NVRM: GPU at PCI:0000:5a:00: reset complete", "0000:59:00"
        )
        is False
    )


def _collect014_harness(
    *,
    fail_status: str = "BLOCKED",
    fail_reasons=None,
    ledger_after_fail: int = 0,
    positive_raises: bool = False,
    inventory_age=None,
):
    """Stub the four collaborators run_collect014 drives and record calls.

    ``inventory_age`` is how old the store's newest GPU inventory snapshot is
    at the moment the runner reads it; the default (five minutes) leaves the
    back-dated fail-closed event a comfortable three-minute freshness margin.
    """

    from datetime import datetime, timedelta, timezone

    calls: list[str] = []
    seen: dict[str, object] = {}
    inventory_age = inventory_age or timedelta(minutes=5)

    class Settings:
        class regional:  # noqa: N801 - stub attribute holder
            cluster_id = "hp-cluster"

        node = "hyperpod-node"

    class Regional:
        def cpu_python(self, script: str, *arguments: str) -> dict:
            assert script is collect_destructive.GPU_INVENTORY_SNAPSHOT
            assert arguments == ("hp-cluster", "hyperpod-node"), arguments
            calls.append("inventory-read")
            observed_at = datetime.now(timezone.utc) - inventory_age
            return {
                "present": True,
                "observed_at": observed_at.isoformat(),
                "source_boot_id": "boot-1",
                "device_count": 1,
            }

        def executor_python(
            self, script: str, payload: str, *, attempts: int | None = None
        ) -> dict:
            import json

            assert script is collect_destructive.FABRIC_POST
            seen["post_attempts"] = attempts
            seen["fail_observed_at"] = datetime.fromisoformat(
                json.loads(payload)["observed_at"]
            )
            seen["posted_at"] = datetime.now(timezone.utc)
            calls.append("fabric-post")
            return {}

    class ResetHost:
        host_script = "/probe.py"
        snapshots = 0

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            if "start-reset-sampler" in arguments:
                calls.append("sampler-start")
                return {}
            if "stop-reset-sampler" in arguments:
                calls.append("sampler-stop")
                return {}
            # The probe returns the node's whole results ledger: an earlier
            # full reset on this node is always present, before and after.
            old_reset = {
                "operation": "RESET_ALL_GPUS_NVSWITCHES",
                "command_id": "workflow-old/5/RESET_ALL_GPUS_NVSWITCHES/node",
            }
            if "--since-epoch" in arguments:
                return {
                    "gpu_inventory": [{"pci_bdf": "0000:59:00.0"}],
                    "ledger": [
                        old_reset,
                        {
                            "operation": "RESET_ALL_GPUS_NVSWITCHES",
                            "command_id": (
                                "workflow-new/5/RESET_ALL_GPUS_NVSWITCHES/node"
                            ),
                        },
                    ],
                    "boot_id": "boot-1",
                }
            self.snapshots += 1
            ledger = [old_reset] + (
                [] if self.snapshots == 1 else [{}] * ledger_after_fail
            )
            return {
                "gpu_inventory": [{"pci_bdf": "0000:59:00.0"}],
                "ledger": ledger,
                "boot_id": "boot-1",
            }

    class Collector:
        def wait_marker(self, marker: str, **kwargs: object) -> dict:
            if marker.startswith("c014-fail-"):
                calls.append("wait-fail")
                seen["fail_kwargs"] = kwargs
                return {
                    "workflows": [
                        {
                            "status": fail_status,
                            # The BLOCKED workflow is the one that *decided*
                            # the full reset; the runner selects it by that
                            # decision, not by list position.
                            "official_action": "RESET_ALL_GPUS_AND_NVSWITCHES",
                            "blocked_reasons": (
                                fail_reasons
                                if fail_reasons is not None
                                else [
                                    "SXID 10003 requires fabric_partition and "
                                    "complete node GPU inventory"
                                ]
                            ),
                        }
                    ],
                    "incidents": [{"incident_id": "inc-fail"}],
                }
            calls.append("wait-positive")
            seen["positive_kwargs"] = kwargs
            if positive_raises:
                raise RuntimeError("store unreachable")
            return {
                "workflows": [
                    {
                        "status": "SUCCEEDED",
                        "official_steps": [{"operation": "RESET_ALL_GPUS_NVSWITCHES"}],
                        "step_executions": [
                            {
                                "operation": "RESET_ALL_GPUS_NVSWITCHES",
                                "status": "SUCCEEDED",
                            }
                        ],
                    }
                ],
                "incidents": [{"incident_id": "inc-positive"}],
            }

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            calls.append("append-sxid")
            return {}

        def restore_incidents(self, state: dict, **_kwargs: object) -> list:
            calls.append("restore")
            return [{"status": "SUCCEEDED"}]

    return Settings(), Regional(), ResetHost(), Collector(), calls, seen


def test_collect014_restores_the_node_before_the_positive_injection(
    tmp_path: Path,
) -> None:
    """The positive full-fabric SXID must run on a node the fail-closed
    incident no longer isolates, and its workflow is matched by injection
    time because FABRIC_MANAGER_LOG evidence carries no marker."""

    from datetime import timedelta

    settings, regional, host, collector, calls, seen = _collect014_harness()
    result = collect_destructive.run_collect014(
        settings, regional, host, collector, tmp_path, 1, "hyperpod-v1"
    )

    # PASS even though the ledger already held a full reset from an earlier
    # run: only rows the injection added count (2026-09-06 08:51Z FAIL).
    assert result["verdict"] == "PASS", result
    assert (
        calls.index("wait-fail") < calls.index("restore") < calls.index("append-sxid")
    ), calls
    # The store's inventory freshness is read before the post is made, the
    # post is a one-shot mutation, and both waits are scoped to their moment.
    assert calls.index("inventory-read") < calls.index("fabric-post"), calls
    assert seen["post_attempts"] == 1, seen
    assert seen["fail_kwargs"].get("observed_after") is not None, seen
    assert calls[-1] == "restore" and calls[-2] == "sampler-stop", calls
    assert len(result["restore_workflows"]) == 2, result["restore_workflows"]
    assert seen["positive_kwargs"].get("observed_after") is not None, seen
    # The fail-closed event is back-dated far enough that a stored inventory
    # sample up to 7.5 min old is still newer than event + 30s, yet it stays
    # under the 900s stale-generation fence.
    age = seen["posted_at"] - seen["fail_observed_at"]
    assert timedelta(minutes=8) <= age < timedelta(minutes=9), age
    assert collect_destructive.FAIL_CLOSED_EVENT_AGE < timedelta(seconds=900)


@pytest.mark.parametrize(
    ("fail_status", "fail_reasons", "ledger_after_fail", "expected"),
    [
        ("SUCCEEDED", None, 0, "did not fail closed"),
        ("BLOCKED", ["some other reason"], 0, "not the dual-evidence gate"),
        ("BLOCKED", None, 3, "produced node-side actions"),
    ],
)
def test_collect014_never_injects_the_positive_sxid_after_a_bad_fail_closed(
    tmp_path: Path,
    fail_status: str,
    fail_reasons: list[str] | None,
    ledger_after_fail: int,
    expected: str,
) -> None:
    """A fail-closed direction that acted on the node has already spent the
    one full fabric reset the case may perform (2026-09-06 05:11Z/05:17Z ran
    two real resets this way). The runner must FAIL without the positive."""

    settings, regional, host, collector, calls, _ = _collect014_harness(
        fail_status=fail_status,
        fail_reasons=fail_reasons,
        ledger_after_fail=ledger_after_fail,
    )
    result = collect_destructive.run_collect014(
        settings, regional, host, collector, tmp_path, 1, "hyperpod-v1"
    )

    assert result["verdict"] == "FAIL", result
    assert any(expected in error for error in result["errors"]), result["errors"]
    assert "append-sxid" not in calls, calls
    assert "sampler-start" not in calls, calls
    assert calls.count("restore") == 1, calls
    assert result["positive"] is None


def test_collect014_refuses_the_fail_closed_post_when_inventory_is_stale(
    tmp_path: Path,
) -> None:
    """The fail-closed premise is a bet that no stored inventory sample can
    vouch for the back-dated event. On 2026-09-06 the sample lagged 2-3 min,
    the bet lost, and the "no-op" SXID ran two real full-fabric resets. The
    runner must read the store first and post nothing when the margin is
    thinner than the ingest tolerance plus the safety margin."""

    from datetime import timedelta

    settings, regional, host, collector, calls, _ = _collect014_harness(
        inventory_age=timedelta(minutes=7, seconds=30)
    )
    result = collect_destructive.run_collect014(
        settings, regional, host, collector, tmp_path, 1, "hyperpod-v1"
    )

    assert result["verdict"] == "FAIL", result
    assert any("not posted" in error for error in result["errors"]), result["errors"]
    assert calls == ["inventory-read"], calls
    assert result["fail_closed"] is None and result["positive"] is None, result
    assert result["restore_workflows"] == [], result


def test_collect014_stops_the_sampler_when_the_positive_path_raises(
    tmp_path: Path,
) -> None:
    settings, regional, host, collector, calls, _ = _collect014_harness(
        positive_raises=True
    )
    with pytest.raises(RuntimeError):
        collect_destructive.run_collect014(
            settings, regional, host, collector, tmp_path, 1, "hyperpod-v1"
        )
    assert calls[-1] == "sampler-stop", calls


def test_reset_sampler_start_clears_a_stale_unit_before_systemd_run(
    tmp_path: Path,
) -> None:
    """A prior run that raised before stop-reset-sampler leaves its transient
    unit loaded; the deterministic unit name means the rerun would collide with
    "already loaded or has a fragment file". start_sampler must clear it first."""

    from argparse import Namespace

    from scripts.e2e.regional.probes import destructive_node_probe as probe

    unit = "gpu-fault-reset-sampler-deadbeefdeadbeef"
    ndjson = tmp_path / "reset-sampler.ndjson"
    commands: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> object:
        commands.append(list(argv))
        return None

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(probe, "sampler_paths", lambda _run_id: (unit, ndjson))
        monkeypatch.setattr(probe, "run", fake_run)
        monkeypatch.setattr(probe, "emit", lambda _payload: None)
        monkeypatch.setattr(
            probe, "sampler_summary", lambda _run_id: {"sample_count": 2}
        )
        probe.start_sampler(
            Namespace(
                run_id="c014-1",
                probe_script=str(Path(probe.__file__).resolve()),
                duration_seconds=1800,
                interval_seconds=0.25,
            )
        )
    finally:
        monkeypatch.undo()

    assert ["systemctl", "stop", unit + ".service"] in commands, commands
    assert ["systemctl", "reset-failed", unit + ".service"] in commands, commands
    systemd_run_index = next(
        i for i, argv in enumerate(commands) if argv and argv[0] == "systemd-run"
    )
    stop_index = commands.index(["systemctl", "stop", unit + ".service"])
    reset_index = commands.index(["systemctl", "reset-failed", unit + ".service"])
    assert stop_index < systemd_run_index, commands
    assert reset_index < systemd_run_index, commands


def test_collect015_matches_the_reboot_workflow_by_injection_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ALWAYS_FATAL SXID is FABRIC_MANAGER_LOG evidence, which never
    carries the marker into the workflow; the wait must be time-scoped."""

    seen: dict[str, object] = {}
    reboot_event = {
        "event_name": "BatchRebootClusterNodes",
        "user_identity": {"session_issuer_arn": "arn:aws:iam::1:role/executor"},
    }
    # The chain the node grows after the injection: the reboot's failed
    # validation escalated into a replace-after workflow that sits newest, so
    # picking "the latest workflow" would judge the wrong record.
    workflows = [
        {
            "request_id": "wf-replace-after",
            "status": "FAILED",
            "official_steps": [{"operation": "REPLACE_NODE"}],
        },
        {
            "request_id": "wf-reboot",
            "status": "SUCCEEDED",
            "official_steps": [{"operation": "RESTART_NODE"}],
            "step_executions": [{"operation": "RESTART_NODE", "status": "SUCCEEDED"}],
        },
    ]

    class Settings:
        node = "hyperpod-node"
        hyperpod_cluster = "hp"
        executor_role_arn = "arn:aws:iam::1:role/executor"

    class Regional:
        def node_snapshot(self, _node: str) -> dict:
            return {"boot_id": "boot-1", "unschedulable": False}

        def wait_node_ready(self, _node: str, **_kwargs: object) -> dict:
            return {"boot_id": "boot-2"}

        def cpu_python(self, script: str, *arguments: str) -> dict:
            assert script is collect_destructive.HYPERPOD_SUBMISSION
            seen["submission_query"] = arguments
            return {"submission": {"state": "SUBMITTED"}, "restart_commands": []}

        def executor_python(self, script: str, *_arguments: str) -> dict:
            assert script is collect_destructive.EXECUTOR_REPLACE_FLAG
            return {"GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "false"}

        def wait_provider_events(self, _started_at: object, **kwargs: object) -> list:
            seen["reboot_wait"] = kwargs
            return [reboot_event]

        def provider_events(self, *_args: object) -> list:
            return [reboot_event]

        def provider_events_provisional(self, _ended_at: object) -> bool:
            return True

    class Collector:
        def snapshot(self) -> dict:
            return {"gpu_inventory": [{"pci_bdf": "0000:59:00.0"}]}

        def execute(self, *_arguments: str, timeout: int = 180) -> dict:
            return {}

        def wait_marker(self, marker: str, **kwargs: object) -> dict:
            seen.update(kwargs)
            return {"workflows": workflows}

    class Provider:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def provider_inventory(self) -> dict:
            return {"nodes": ["a"]}

        def cluster_recovery(self) -> dict:
            return {"node_recovery": "None"}

    monkeypatch.setattr(collect_destructive, "WarmSpareLiveFixture", Provider)
    monkeypatch.setattr(
        collect_destructive, "provider_event_actor_matches_role", lambda *_: True
    )
    result = collect_destructive.run_collect015(
        Settings(), Regional(), Collector(), tmp_path, 1
    )

    assert result["verdict"] == "PASS", result
    assert seen.get("observed_after") is not None, seen
    assert seen.get("terminal_workflow") is True, seen
    # The RESTART_NODE workflow is judged, not the newer replace-after one,
    # and its own request_id is what the HyperPod submission is looked up by.
    assert result["workflow"]["request_id"] == "wf-reboot", result["workflow"]
    assert seen["submission_query"] == ("wf-reboot", "hp"), seen
    assert result["submission"] == {"state": "SUBMITTED"}, result
    assert seen["reboot_wait"] == {
        "event_names": set(collect_destructive.REBOOT_EVENTS),
        "expected_count": 1,
    }, seen
    assert result["provider_events_provisional"] is True, result


def _daemonset(namespace: str, name: str, desired: int) -> dict:
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": {"template": {"spec": {"affinity": {"nodeAffinity": {}}}}},
        "status": {"desiredNumberScheduled": desired},
    }


def test_device_plugin_discovery_ignores_daemonsets_that_schedule_nowhere() -> None:
    """The HyperPod chart ships `<plugin>-mps-control-daemon` next to the
    plugin; it carries the token but has desired 0. Only a DaemonSet that
    places Pods can change a node's allocatable, so it is the one and only
    candidate (2026-09-06 09:52Z: "expected one nvidia-device-plugin
    DaemonSet, found 2")."""

    import json

    items = [
        _daemonset("kube-system", "hyperpod-dependencies-nvidia-device-plugin", 4),
        _daemonset(
            "kube-system",
            "hyperpod-dependencies-nvidia-device-plugin-mps-control-daemon",
            0,
        ),
    ]

    class Regional:
        def kubectl(self, *_arguments: str, **_kwargs: object) -> str:
            return json.dumps({"items": items})

    plugin = collect_destructive.DevicePluginFixture(
        Regional(),
        token="nvidia-device-plugin",
        node="node-a",
        resource="nvidia.com/gpu",
    )
    found = plugin.discover()
    assert found["name"] == "hyperpod-dependencies-nvidia-device-plugin", found

    items[1]["status"]["desiredNumberScheduled"] = 4
    with pytest.raises(RegionalFixtureError, match="found 2"):
        plugin.discover()
