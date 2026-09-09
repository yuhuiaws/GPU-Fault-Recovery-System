"""Contracts of the systemd units and data-plane manifests the installer ships.

Covers the collector units (environment-file node id, watchdog supervision,
resource bounds, writable outbox, explicit stop budgets), the DCGM exporter
container, the node agent unit's environment file and stop timeout, the
certificate-bundle check and its timer, the GPU persistence unit, and the
tolerations that keep the data-plane Deployments on cordoned nodes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from gpu_fault.node_agent import common
from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT


def test_systemd_collectors_use_environment_node_id() -> None:
    kernel = (ROOT / "deploy/systemd/gpu-fault-kernel-collector.service").read_text()
    metrics = (ROOT / "deploy/systemd/gpu-fault-metrics-collector.service").read_text()
    fabric = (
        ROOT / "deploy/systemd/gpu-fault-fabric-manager-collector.service"
    ).read_text()

    assert "--node-id %H" not in kernel
    assert "--node-id %H" not in metrics
    assert "EnvironmentFile=/etc/gpu-fault/collector.env" in kernel
    assert "${GPU_FAULT_METRICS_MODE}" in metrics
    assert "gpu-fault-collector fabric-manager" in fabric
    assert "SupplementaryGroups=systemd-journal" in fabric


def test_host_collector_unit_is_watchdog_supervised() -> None:
    """A wedged host collector must look wedged to systemd.

    Every contributor of the host collector reads the host: ``statvfs`` on a
    hard NFS/Lustre mount and ``nvidia-smi`` inside a hung driver both block in
    uninterruptible sleep, and with ``Type=simple`` and no watchdog the unit
    stayed "active (running)" forever, so ``Restart=always`` never fired and
    the node that most needed telemetry sent none. ``Type=notify`` plus
    ``WatchdogSec=`` makes a tick that never completes a restart (``run()``
    sends ``READY=1`` once and ``WATCHDOG=1`` between contributors).

    The deadline must clear the tick's *bounded* worst case, not its typical
    one: two nvidia-smi calls, a topology dump, ethtool per EFA netdev, a
    smartctl scan plus one call per drive, ipmitool, ``statvfs`` per mount and
    the sink's retry ladder are minutes when the host is sick. A watchdog
    shorter than that kills the tick before ``sink.post`` and the node posts
    nothing, forever -- the outage the watchdog exists to end. Startup has its
    own bound because product discovery shells out before ``READY=1``, and an
    activating unit that never becomes ready fails the installer's
    ``systemctl restart`` and rolls the node install back.
    """

    unit = (ROOT / "deploy/systemd/gpu-fault-host-collector.service").read_text()

    assert "Type=notify" in unit, "the collector's watchdog needs Type=notify"
    assert "WatchdogSec=600" in unit, (
        "a watchdog below the tick's bounded worst case kills every tick before "
        "it can post"
    )
    assert "TimeoutStartSec=180" in unit, (
        "a Type=notify unit whose startup discovery hangs must fail, not be "
        "killed on systemd's default and retried forever"
    )
    assert "Restart=always" in unit, "the watchdog restart needs a restart policy"


def test_systemd_units_bound_memory_and_cpu() -> None:
    units = sorted((ROOT / "deploy/systemd").glob("*.service"))

    assert units, "expected units to be truthy"
    for unit in units:
        text = unit.read_text()
        assert "MemoryMax=" in text, unit.name
        assert "MemoryHigh=" in text, unit.name
        assert "MemorySwapMax=0" in text, unit.name
        assert "CPUQuota=" in text or "CPUWeight=" in text, unit.name
        assert "CPUWeight=" in text, unit.name


def test_every_systemd_template_is_handled_by_the_node_installer() -> None:
    installer = (ROOT / "deploy/node/install-gpu-fault-collector.sh").read_text(
        encoding="utf-8"
    )
    units = sorted((ROOT / "deploy/systemd").glob("gpu-fault-*.*"))

    assert units, "expected gpu-fault systemd templates"
    for unit in units:
        assert f"deploy/systemd/{unit.name}" in installer, (
            f"node installer does not install or render {unit.name}"
        )


def test_collector_units_can_write_the_default_outbox() -> None:
    """Every collector unit must be able to write /var/lib/gpu-fault.

    ``collectors_cli`` defaults GPU_FAULT_COLLECTOR_OUTBOX_PATH to
    /var/lib/gpu-fault/outbox/<subcommand>.ndjson, and every unit runs
    with ProtectSystem=strict, which makes /var read-only. A unit
    without StateDirectory= or ReadWritePaths= therefore silently loses
    every event it buffers -- the kernel collector shipped that way, so
    XID/SXID were dropped precisely when the control plane was
    unreachable.
    """

    collectors = {
        "gpu-fault-kernel-collector.service",
        "gpu-fault-log-collector.service",
        "gpu-fault-fabric-manager-collector.service",
        "gpu-fault-host-collector.service",
        "gpu-fault-metrics-collector.service",
    }
    for name in sorted(collectors):
        unit = ROOT / "deploy/systemd" / name
        text = unit.read_text()
        assert unit.exists(), name
        if "ProtectSystem=strict" not in text:
            continue
        assert (
            "StateDirectory=gpu-fault" in text
            or "ReadWritePaths=/var/lib/gpu-fault" in text
        ), name


def test_dcgm_exporter_container_is_resource_bounded() -> None:
    service = (ROOT / "deploy/systemd/gpu-fault-dcgm-exporter.service").read_text()

    assert "--memory 2g" in service
    assert "--cpus 2" in service
    assert "--pids-limit 512" in service


def test_node_agent_environment_contains_regional_bearer_token() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    service = (ROOT / "deploy/systemd/gpu-fault-node-agent.service").read_text()

    assert 'write_env GPU_FAULT_NODE_CONTROL_PLANE_TOKEN "${TOKEN}"' in installer
    assert "EnvironmentFile=/etc/gpu-fault/node-agent.env" in service


def test_dcgm_exporter_is_isolated_from_hma() -> None:
    service = (ROOT / "deploy/systemd/gpu-fault-dcgm-exporter.service").read_text()

    assert "--name gpu-fault-dcgm-exporter" in service
    assert "--network host" in service
    assert "-a 127.0.0.1:9400" in service
    assert "DCGM_EXPORTER_COLLECTORS=" in service
    assert "hma" not in service.lower()


def test_certificate_bundle_check_rejects_near_expiry(tmp_path) -> None:
    checker = ROOT / "deploy/node/verify-certificate-bundle.sh"

    def certificate(name: str, days: int) -> Path:
        key = tmp_path / f"{name}.key"
        cert = tmp_path / f"{name}.crt"
        result = subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-subj",
                f"/CN={name}",
                "-days",
                str(days),
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return cert

    long_lived = certificate("long-lived", 365)
    second = certificate("second-ca", 365)
    expiring = certificate("expiring", 1)
    bundle = tmp_path / "ca-bundle.pem"
    bundle.write_bytes(long_lived.read_bytes() + second.read_bytes())

    accepted = subprocess.run(
        ["bash", str(checker), str(bundle), "2592000"],
        capture_output=True,
        text=True,
        check=False,
    )
    rejected = subprocess.run(
        ["bash", str(checker), str(expiring), "2592000"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert "certificates=2" in accepted.stdout
    assert rejected.returncode != 0
    assert "expires within" in rejected.stderr


def test_certificate_timer_is_installed_and_enabled() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    bundle = (ROOT / "deploy/node/build-node-installer-bundle.sh").read_text()
    service = (ROOT / "deploy/systemd/gpu-fault-certificate-check.service").read_text()
    timer = (ROOT / "deploy/systemd/gpu-fault-certificate-check.timer").read_text()

    assert "gpu-fault-certificate-check.timer" in installer
    assert "systemctl enable gpu-fault-certificate-check.timer" in (installer)
    assert '"${REPO_DIR}/deploy/systemd/"*.timer' in bundle
    assert "check-control-plane-certificate" in service
    assert "OnUnitActiveSec=12h" in timer


def test_gpu_persistence_mode_is_installed_and_verified() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    verifier = NODE_SCRIPTS[1].read_text()
    service = (ROOT / "deploy/systemd/gpu-fault-gpu-persistence.service").read_text()

    assert "After=nvidia-persistenced.service" in service
    assert "ExecStart=@NVIDIA_SMI@ -pm 1" in service
    assert "systemctl enable gpu-fault-gpu-persistence.service" in installer
    assert "systemctl restart gpu-fault-gpu-persistence.service" in installer
    assert "--query-gpu=persistence_mode" in installer
    # Owner decision 6: a GPU that never entered persistence mode is reported,
    # not fatal -- the installer would otherwise roll back on exactly the node
    # whose GPU needs the Agent to remediate it.
    assert "not every GPU entered persistence mode" not in installer, (
        "persistence mode must no longer abort the install"
    )
    assert "WARN  GPU persistence mode (not enabled on every GPU)" in installer, (
        "the installer must report degraded persistence mode as a warning"
    )
    assert "/var/lib/gpu-fault/installer-degraded-gpu.json" in installer, (
        "the installer must record what it observed for the follow-up collector"
    )
    assert "GPU persistence mode" in verifier
    assert "gpu-fault-gpu-persistence" in common.DEFAULT_QUIESCE_SERVICES, (
        "the unit the installer enables must also be quiesced before a GPU reset"
    )


def test_data_plane_collectors_tolerate_cordoned_nodes() -> None:
    """A quarantined or cordoned node must still run the data plane.

    The reconciler is included because it used to carry a bare
    ``operator: Exists``: that tolerates ``not-ready`` and ``unreachable``
    forever, so taint-based eviction never took the singleton off a dead node
    and the Deployment -- seeing one Pod -- never rescheduled it until the Node
    object itself was deleted.
    """

    for manifest in (
        "completion-watcher.yaml",
        "kubernetes-node-resource-collector.yaml",
        "node-installer-reconciler.yaml",
    ):
        path = ROOT / "deploy/dataplane" / manifest
        content = path.read_text()
        assert "gpu-fault.io/quarantined" in content, manifest
        assert "node.kubernetes.io/unschedulable" in content, manifest
        for document in yaml.safe_load_all(content):
            if not isinstance(document, dict) or document.get("kind") != "Deployment":
                continue
            spec = document["spec"]["template"]["spec"]
            for toleration in spec.get("tolerations") or []:
                assert set(toleration) != {"operator"}, (
                    f"{manifest}: a bare `operator: Exists` tolerates every "
                    "taint, including the two that exist to evict this Pod"
                )


def test_node_agent_stop_waits_out_the_longest_handler() -> None:
    """SIGTERM must not SIGKILL a GPU reset in flight.

    ``lifespan`` shuts the action pool down without waiting, so a stop or
    restart during a handler leaves systemd's default 90 s as the only budget:
    the running ``nvidia-smi --gpu-reset`` or driver install dies with the
    cgroup and the ledger row is left INTERRUPTED, which then needs manual
    confirmation. ``KillMode=mixed`` sends the SIGTERM to the main process only
    so the handler's own child is not killed out from under it, and the timeout
    matches the installer's own drain budget -- the two must not disagree,
    because the installer waits before it stops the unit.
    """

    service = (ROOT / "deploy/systemd/gpu-fault-node-agent.service").read_text()
    installer = NODE_SCRIPTS[0].read_text()

    assert "TimeoutStopSec=1900" in service, (
        "the agent must be given at least the longest handler timeout to finish"
    )
    assert "KillMode=mixed" in service, (
        "the default control-group kill signals the handler's child process too"
    )
    assert 'NODE_AGENT_STOP_TIMEOUT_SECONDS="1900"' in installer, (
        "the installer's drain budget and the unit's TimeoutStopSec must agree"
    )


def test_kernel_collector_unit_declares_its_drain_budget() -> None:
    """The kernel collector drains its queue on SIGTERM in about ten seconds.

    Leaving the stop timeout implicit means the only documented budget is
    systemd's 90 s default, which reads as "this unit may take a minute and a
    half to stop" and hides a regression that makes the drain unbounded.
    """

    unit = (ROOT / "deploy/systemd/gpu-fault-kernel-collector.service").read_text()

    assert "TimeoutStopSec=30" in unit, (
        "the drain budget must be explicit, not systemd's 90 s default"
    )
