from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
NODE_SCRIPTS = (
    ROOT / "deploy/node/install-gpu-fault-collector.sh",
    ROOT / "deploy/node/verify-gpu-fault-collector.sh",
    ROOT / "deploy/node/uninstall-gpu-fault-collector.sh",
    ROOT / "deploy/node/build-node-installer-bundle.sh",
    ROOT / "deploy/node/run-hyperpod-installer-job.sh",
    ROOT / "deploy/node/provision-node-action-keys.sh",
    ROOT / "deploy/node/verify-certificate-bundle.sh",
    ROOT / "deploy/node/check-control-plane-certificate.sh",
)


def test_node_deployment_scripts_are_valid_bash() -> None:
    for script in NODE_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


def test_hyperpod_canary_env_values_are_complete() -> None:
    documents = list(
        yaml.safe_load_all(
            (ROOT / "scripts/e2e/manifests/hyperpod-canary.yaml").read_text()
        )
    )
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    assert all(("value" in item) != ("valueFrom" in item) for item in env), (
        'expected all(("value" in item) != ("valueFrom" in item) for item in env) to be truthy'
    )
    values = {item["name"]: item.get("value") for item in env}
    assert values["GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS"] == "true"
    assert values["GPU_FAULT_ALLOW_SINGLE_CLUSTER"] == "true"


def test_node_installer_help_does_not_require_root() -> None:
    result = subprocess.run(
        ["bash", str(NODE_SCRIPTS[0]), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dcgm-exporter existing|docker|disabled" in result.stdout
    assert "--python-command PATH" in result.stdout
    assert "--allow-service-quiesce" in result.stdout
    assert "--quiesce-failsafe-seconds SEC" in result.stdout
    assert "--fabric-manager-log-paths GLOBS" in result.stdout
    assert "--allow-driver-remediation" in result.stdout
    assert "--firmware-verify-sha256 HEX" in result.stdout
    assert "--allow-field-diagnostic" in result.stdout
    assert "--field-diagnostic-sha256 HEX" in result.stdout
    assert "--memory-field-diagnostic-command CMD" in result.stdout
    assert "--expected-gpu-count COUNT" in result.stdout
    assert "--expected-efa-device-count COUNT" in result.stdout
    assert "--inventory-mismatch-samples N" in result.stdout
    assert "--node-instance-type TYPE" in result.stdout
    assert "--enable-node-log-collector" in result.stdout
    assert "--enable-nvidia-smi-metrics-collector" in result.stdout
    assert "--certificate-min-validity-seconds N" in result.stdout


def test_hyperpod_installer_supports_regional_connection_secret() -> None:
    installer = NODE_SCRIPTS[4].read_text()

    assert "GPU_FAULT_CONNECTION_MODE:-local" in installer
    assert (
        "GPU_FAULT_REGIONAL_CONNECTION_SECRET:-gpu-fault-regional-connection"
    ) in installer
    assert "GPU_FAULT_CONNECTION_MODE must be local or regional" in installer
    assert "GPU_FAULT_KUBECTL_CONTEXT:-" in installer
    assert 'command kubectl --context "${KUBECTL_CONTEXT}" "$@"' in installer
    for key in ("control-plane-url", "cluster-token", "cluster-id", "ca.crt"):
        assert f"key: {key}" in installer
    assert "gpu-fault-node-action-keys" in installer
    assert 'NODE_ACTION_SECRET_KEY="${NODE_NAME}"' in installer
    assert 'DERIVE_NODE_ACTION_SECRET="false"' in installer
    assert "/node-secret/node-action-secret" in installer
    assert '--token "${CONTROL_PLANE_TOKEN}"' in installer
    assert '--ca-certificate "${CONTROL_PLANE_CA_CERTIFICATE}"' in installer


def test_regional_node_keys_are_derived_before_the_job() -> None:
    provision = (ROOT / "deploy/node/provision-node-action-keys.sh").read_text()
    installer = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text()
    reconciler = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text()

    assert "derive_node_action_secret" in provision
    assert "GPU_FAULT_FLEET_MASTER_FILE" in provision
    assert "GPU_FAULT_HYPERPOD_CLUSTER" in provision
    assert "cluster-name=${HYPERPOD_CLUSTER}" in provision
    assert '"${MASTER_FILE}" "${CLUSTER_ID}" "${node}"' in provision
    assert '--from-file="${key_dir}"' in provision
    assert "GPU_FAULT_FLEET_MASTER_FILE" in reconciler
    assert "GPU_FAULT_HYPERPOD_CLUSTER" in reconciler
    assert "cluster-name=${HYPERPOD_CLUSTER}" in reconciler
    assert "provision-node-action-keys.sh" in reconciler
    assert "GPU_FAULT_INSTALLER_ARTIFACT_SHA256" in installer
    assert "GPU_FAULT_INSTALLER_ARTIFACT_SHA256" in reconciler
    assert "REPLACE_WITH_INSTALLER_ARTIFACT_SHA256" in reconciler
    assert 'RUNTIME_PROFILE="${GPU_FAULT_RUNTIME_PROFILE:-hyperpod-v1}"' in reconciler
    assert 'GPU_FAULT_RUNTIME_PROFILE="${RUNTIME_PROFILE}"' in reconciler
    assert 'TEMPLATE_SHA256="$(sha256sum "${MANIFEST}"' in reconciler
    assert "gpu-fault-node-installer-template-${TEMPLATE_SHA256:0:12}" in reconciler
    manifest = (ROOT / "deploy/dataplane/node-installer-reconciler.yaml").read_text()
    assert "GPU_FAULT_INSTALLER_TEMPLATE_PATH" in manifest
    assert "REPLACE_WITH_INSTALLER_TEMPLATE_CONFIG_MAP" in manifest
    assert "REPLACE_WITH_HYPERPOD_CLUSTER" in manifest
    assert 'NODE_ACTION_SECRET_NAME="${NODE_ACTION_KEYS_SECRET}"' in installer
    assert 'NODE_ACTION_SECRET_KEY="${NODE_NAME}"' in installer
    assert 'DERIVE_NODE_ACTION_SECRET="false"' in installer
    assert "node_action_args+=(--node-action-key-version 2)" in installer
    assert "ensure_node_action_keys" in (ROOT / "deploy/hyperpod/deploy.sh").read_text()


def test_runtime_profile_placeholder_is_rendered_for_data_plane_collectors() -> None:
    resource_collector = (
        ROOT / "deploy/dataplane/kubernetes-node-resource-collector.yaml"
    ).read_text()
    hma_watcher = (ROOT / "deploy/dataplane/optional/hma-watcher.yaml").read_text()
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()

    assert "REPLACE_WITH_RUNTIME_PROFILE_VERSION" in resource_collector
    assert "REPLACE_WITH_RUNTIME_PROFILE_VERSION" in hma_watcher
    assert (
        deploy.count("s#REPLACE_WITH_RUNTIME_PROFILE_VERSION#${RUNTIME_PROFILE}#g") >= 5
    )


def test_node_installer_refuses_second_hostengine() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert "pgrep -x nv-hostengine" in installer
    assert "a DCGM exporter already answers" in installer


def test_node_agent_health_check_uses_advertised_endpoint() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    verifier = NODE_SCRIPTS[1].read_text()

    assert "NODE_AGENT_HEALTH_URL" in installer
    assert "NODE_AGENT_ADVERTISE_URL%/" in installer
    assert "http://127.0.0.1:${NODE_AGENT_PORT}/healthz" not in installer
    assert "GPU_FAULT_NODE_ADVERTISE_URL" in verifier
    assert "${node_agent_url%/}/healthz" in verifier


def test_node_installer_discovers_fabric_manager_file_logging() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert "/usr/share/nvidia/nvswitch/fabricmanager.cfg" in installer
    assert "LOG_FILE_NAME" in installer
    assert "LOG_USE_SYSLOG" in installer
    assert '[[ "${log_use_syslog}" == "0"' in installer
    assert '[[ -e "${log_file_name}" ]]' in installer
    assert "validate_fabric_manager_log_paths" in installer


def test_explicit_fabric_manager_log_paths_take_priority() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert 'FABRIC_MANAGER_LOG_PATHS_EXPLICIT="true"' in installer
    assert (
        '[[ "${FABRIC_MANAGER_LOG_PATHS_EXPLICIT}" == "false" ]] || return' in installer
    )


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


def test_verifier_does_not_source_secret_environment_file() -> None:
    verifier = NODE_SCRIPTS[1].read_text()

    assert 'source "${ENV_FILE}"' not in verifier
    assert "shlex.split" in verifier
    assert "VERIFY_STARTED_EPOCH" in verifier
    assert '"SSL_CERT_FILE": ""' in verifier
    assert 'CURL_TLS=(--cacert "${SSL_CERT_FILE}")' in verifier
    assert "verify-certificate-bundle" in verifier
    assert "check-control-plane-certificate" in verifier
    assert "for _ in {1..60}; do" in verifier


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
    node_agent = (ROOT / "src/gpu_fault/node_agent/common.py").read_text()

    assert "After=nvidia-persistenced.service" in service
    assert "ExecStart=@NVIDIA_SMI@ -pm 1" in service
    assert "systemctl enable gpu-fault-gpu-persistence.service" in installer
    assert "systemctl restart gpu-fault-gpu-persistence.service" in installer
    assert "--query-gpu=persistence_mode" in installer
    assert "not every GPU entered persistence mode" in installer
    assert "GPU persistence mode" in verifier
    assert '"gpu-fault-gpu-persistence"' in node_agent


def test_agent_pin_migration_pauses_processor_workers() -> None:
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()

    pause = deploy.index("pause_control_workers_for_agent_pin_migration")
    build = deploy.rindex("\n    build_artifacts")
    install = deploy.rindex("\n    install_node_agents")
    resume = deploy.rindex("\n    resume_control_workers_after_agent_pin_migration")

    assert pause < build < install < resume
    assert "deployment/gpu-fault-control-worker --replicas=0" in deploy
    assert '--replicas="${CONTROL_WORKER_REPLICAS}"' in deploy


def test_node_log_collector_is_disabled_by_production_deploy() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    verifier = NODE_SCRIPTS[1].read_text()
    hyperpod_installer = NODE_SCRIPTS[4].read_text()
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()
    migration = (
        ROOT / "deploy/migrations/gpu-node-collector-endpoint-migration.yaml"
    ).read_text()

    assert 'ENABLE_NODE_LOG_COLLECTOR="false"' in installer
    assert "--enable-node-log-collector" in installer
    assert "systemctl disable --now gpu-fault-log-collector.service" in installer
    assert 'if [[ "${ENABLE_NODE_LOG_COLLECTOR}" == "true" ]]' in installer
    assert "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR" in verifier
    assert "must be disabled" in verifier
    assert "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR:-false" in hyperpod_installer
    assert "node_log_args=(--enable-node-log-collector)" in hyperpod_installer
    assert "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR:-false" in deploy
    assert "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR false" in migration
    assert "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR true" not in migration
    assert "gpu_fault_node_runtime-0.10.0-py3-none-any.whl" in migration
    assert "GPU_FAULT_NODE_COMPATIBILITY_DIGEST" in migration
    assert "REPLACE_WITH_COMPATIBILITY_DIGEST" in migration


def test_training_progress_monitor_is_disabled_by_default() -> None:
    lifespan = "\n".join(
        (
            (ROOT / "src/gpu_fault/app/lifespan.py").read_text(),
            (ROOT / "src/gpu_fault/app/periodic_services.py").read_text(),
            (ROOT / "src/gpu_fault/app/lifespan_workers.py").read_text(),
        )
    )
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()

    assert '"GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", "false"' in lifespan
    assert "GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR:-false" in deploy
    assert "GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR must be true or false" in deploy
    for manifest in (
        ROOT / "deploy/control-plane/base/control-plane-deployment.yaml",
        ROOT / "scripts/e2e/manifests/xid45-correlation-canary.yaml",
    ):
        content = manifest.read_text()
        setting = content.split("- name: GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", 1)[
            1
        ].splitlines()[1]
        assert 'value: "false"' in setting


def test_kubernetes_hma_collector_is_disabled_by_default() -> None:
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()
    installer = NODE_SCRIPTS[0].read_text()
    verifier = NODE_SCRIPTS[1].read_text()

    assert "GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR:-false" in deploy
    assert "GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR must be true or false" in deploy
    assert 'if [[ "${ENABLE_KUBERNETES_HMA_COLLECTOR}" == "true" ]]' in deploy
    assert (
        "delete deployment \\\n"
        "            gpu-fault-hma-watcher --ignore-not-found --wait=true" in deploy
    )
    assert (
        "delete clusterrolebinding \\\n"
        "            gpu-fault-hma-watcher --ignore-not-found" in deploy
    )
    assert (
        "delete clusterrole \\\n"
        "            gpu-fault-hma-watcher --ignore-not-found" in deploy
    )
    assert (
        "delete serviceaccount \\\n"
        "            gpu-fault-hma-watcher --ignore-not-found" in deploy
    )
    validation = deploy.split("validate_platform() {", 1)[1]
    assert "Kubernetes HMA Node collector (disabled" in validation
    assert "systemctl enable gpu-fault-kernel-collector.service" in installer
    assert "systemctl enable gpu-fault-fabric-manager-collector.service" in installer
    assert "kernel XID/SXID collector service" in verifier
    assert "Fabric Manager SXID collector service" in verifier


def test_data_plane_collectors_tolerate_cordoned_nodes() -> None:
    for manifest in (
        "completion-watcher.yaml",
        "kubernetes-node-resource-collector.yaml",
    ):
        content = (ROOT / "deploy/dataplane" / manifest).read_text()
        assert "gpu-fault.io/quarantined" in content
        assert "node.kubernetes.io/unschedulable" in content


def test_nvidia_smi_metrics_collector_is_disabled_by_default() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    hyperpod_installer = NODE_SCRIPTS[4].read_text()
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()

    assert 'ENABLE_NVIDIA_SMI_METRICS_COLLECTOR="false"' in installer
    assert "--enable-nvidia-smi-metrics-collector" in installer
    assert "nvidia-smi metrics collector is disabled" in installer
    assert "GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR:-false" in hyperpod_installer
    assert (
        "DCGM cannot be disabled while NvidiaSmiMetricsCollector is disabled"
        in hyperpod_installer
    )
    assert "GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR:-false" in deploy


def test_production_installer_enables_failsafe_quiesce() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    hyperpod_installer = (
        ROOT / "deploy/node/run-hyperpod-installer-job.sh"
    ).read_text()
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()
    manual = (ROOT / "docs/部署和运维手册.md").read_text()

    assert "QUIESCE_GPU_SERVICES" in installer
    assert "RESTORE_GPU_SERVICES" in installer
    assert "GPU_FAULT_QUIESCE_FAILSAFE_SECONDS" in installer
    for flag in (
        "--allow-gpu-reset",
        "--allow-fabric-reset",
        "--allow-service-quiesce",
        "--allow-fabric-manager-restart",
    ):
        assert flag in hyperpod_installer
        assert flag in manual
    pin_block = manual.split('export REQUIRED_AGENT_CONFIG_DIGEST="$(', 1)[1].split(
        ')"', 1
    )[0]
    for setting in (
        "GPU_FAULT_NODE_ALLOW_GPU_RESET=true",
        "GPU_FAULT_NODE_ALLOW_FABRIC_RESET=true",
        "GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE=true",
        "GPU_FAULT_NODE_ALLOW_FABRIC_MANAGER_RESTART=true",
    ):
        assert setting in pin_block
    services = (
        "nvidia-fabricmanager,nvidia-dcgm,nvidia-persistenced,"
        "gpu-fault-gpu-persistence,gpu-fault-metrics-collector,"
        "gpu-fault-host-collector,kubelet"
    )
    assert f'QUIESCE_SERVICES="{services}"' in installer
    assert f"GPU_FAULT_QUIESCE_SERVICES:-{services}" in deploy
    assert "--allow-driver-remediation" in hyperpod_installer
    assert "--allow-firmware-update" in hyperpod_installer
    assert "--diagnostic-s3-uri" in hyperpod_installer
    assert 'DIAGNOSTIC_S3_URI="${DIAGNOSTIC_S3_URI%/}"' in deploy
    assert (
        "Agent artifact/configuration differs from control-plane requirement"
    ) in deploy
    assert "--allow-field-diagnostic" in hyperpod_installer
    assert "FIELD_DIAGNOSTIC_COMMAND_B64" in hyperpod_installer
    assert "GPU_FAULT_EXPECTED_GPU_COUNT" in hyperpod_installer
    assert "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT" in hyperpod_installer
    assert "fieldPath: metadata.uid" in hyperpod_installer
    assert "gpu-fault-node-action-secret-\\${INSTALL_RUN_ID}" in (hyperpod_installer)
    assert "gpu-fault-control-plane-ca-\\${INSTALL_RUN_ID}.crt" in (hyperpod_installer)
    for instance_type, gpu_count, efa_count in (
        ("p5.4xlarge", 1, 1),
        ("p5.48xlarge", 8, 32),
        ("p5e.48xlarge", 8, 32),
        ("p5en.48xlarge", 8, 16),
        ("p6-b200.48xlarge", 8, 8),
        ("p6-b300.48xlarge", 8, 16),
    ):
        assert f"ml.{instance_type}|{instance_type}" in hyperpod_installer
        case_body = hyperpod_installer.split(f"ml.{instance_type}|{instance_type})", 1)[
            1
        ].split(";;", 1)[0]
        assert f"EXPECTED_GPU_COUNT:-{gpu_count}" in case_body
        assert f"EXPECTED_EFA_DEVICE_COUNT:-{efa_count}" in case_body


def test_install_and_uninstall_restore_pending_quiesce_state() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    uninstaller = NODE_SCRIPTS[2].read_text()

    for script in (installer, uninstaller):
        assert "quiesce-*.json" in script
        assert "gpu-fault-restore-gpu-services" in script
    assert "upgrade aborted" in installer
    assert "uninstall aborted" in uninstaller


WHEEL_BLOCK_START = 'EXPECTED_WHEEL_SHA256=""'
WHEEL_BLOCK_END = 'does not match ${RELEASE_MANIFEST}"'


def _wheel_discovery_probe(target: Path) -> Path:
    """Run the installer's own wheel discovery, not a paraphrase of it.

    ``bash -n`` and substring assertions both pass on a ``find`` that matches
    nothing, which is exactly how ``-maxdepth 1 -type f`` survived: the node
    bundle flattens the wheel into ``dist/``, so the rule looked right, while
    a repo checkout keeps it in the content-addressed ``dist/<release_id>/``
    and every checkout install died with "collector wheel not found".
    """
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(WHEEL_BLOCK_START) == 1
    assert installer.count(WHEEL_BLOCK_END) == 1
    start = installer.index(WHEEL_BLOCK_START)
    end = installer.index(WHEEL_BLOCK_END) + len(WHEEL_BLOCK_END)
    probe = target / "probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'REPO_DIR="$1"\n'
        "PYTHON_COMMAND=python3\n"
        'WHEEL=""\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'printf \'WHEEL=%s\\nSHA=%s\\n\' "${WHEEL}" "${WHEEL_SHA256}"\n',
        encoding="utf-8",
    )
    return probe


def _run_probe(probe: Path, repo_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(repo_dir)], capture_output=True, text=True, check=False
    )


def test_installer_finds_the_wheel_in_both_release_layouts(tmp_path: Path) -> None:
    probe = _wheel_discovery_probe(tmp_path)

    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_control_plane-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"checkout-wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest = checkout / "dist/current-release.json"
    manifest.write_text(
        json.dumps(
            {"wheel": wheel.relative_to(checkout).as_posix(), "wheel_sha256": digest}
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={wheel}" in result.stdout
    assert f"SHA={digest}" in result.stdout

    bundle = tmp_path / "bundle"
    flat = bundle / "dist/gpu_fault_node_runtime-0.10.0.whl"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"bundle-wheel")

    result = _run_probe(probe, bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={flat}" in result.stdout


def test_installer_refuses_a_wheel_the_release_manifest_disowns(tmp_path: Path) -> None:
    probe = _wheel_discovery_probe(tmp_path)
    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_control_plane-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"tampered")
    (checkout / "dist/current-release.json").write_text(
        json.dumps(
            {
                "wheel": wheel.relative_to(checkout).as_posix(),
                "wheel_sha256": hashlib.sha256(b"original").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, checkout)
    assert result.returncode != 0
    assert "does not match" in result.stdout

    empty = tmp_path / "empty"
    (empty / "dist").mkdir(parents=True)
    result = _run_probe(probe, empty)
    assert result.returncode != 0
    assert "collector wheel not found" in result.stdout
