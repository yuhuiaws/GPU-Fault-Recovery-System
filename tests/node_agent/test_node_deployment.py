from __future__ import annotations

import hashlib
import json
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path

import yaml

from gpu_fault.env_validation import (
    TRAINING_HEALTH_MONITOR_ENV,
    training_health_monitor_enabled,
)
from gpu_fault.node_agent import common
from gpu_fault.node_agent.ledger import IN_PROGRESS_STATE, NodeActionLedger

ROOT = Path(__file__).parents[2]
NODE_SCRIPTS = (
    ROOT / "deploy/node/install-gpu-fault-collector.sh",
    ROOT / "deploy/node/verify-gpu-fault-collector.sh",
    ROOT / "deploy/node/uninstall-gpu-fault-collector.sh",
    ROOT / "deploy/node/build-node-installer-bundle.sh",
    ROOT / "deploy/node/run-hyperpod-installer-job.sh",
    ROOT / "deploy/node/preflight-gpu-fault-node.sh",
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
            (ROOT / "scripts/e2e/regional/manifests/hyperpod-canary.yaml").read_text()
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


def test_node_installer_exposes_config_digest_environment() -> None:
    result = subprocess.run(
        ["bash", str(NODE_SCRIPTS[0]), "--print-config-digest-environment"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "GPU_FAULT_PYTHON_STACK_TOOL": (
            "/opt/gpu-fault/tools/py-spy-0.4.1-e7c2de2dc544/venv/bin/py-spy"
        ),
        "GPU_FAULT_QUIESCE_RESTORE_COMMAND": (
            "/opt/gpu-fault/current/venv/bin/gpu-fault-restore-gpu-services"
        ),
    }


def test_node_rollout_preflight_is_read_only_and_server_validated() -> None:
    deploy = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text()
    job = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text()
    preflight = (ROOT / "deploy/node/preflight-gpu-fault-node.sh").read_text()

    assert 'PREFLIGHT_ONLY="${GPU_FAULT_RECONCILER_PREFLIGHT_ONLY:-false}"' in deploy
    assert 'if [[ "${PREFLIGHT_ONLY}" == "true" ]]' in deploy
    assert deploy.count("apply --dry-run=server") >= 3
    assert "--preflight-only --render-only" in deploy
    assert 'render_installer_job "${node}" --preflight-only' in deploy
    preflight_section = deploy.split('if [[ "${PREFLIGHT_ONLY}" == "true" ]]', 1)[1]
    assert "provision-node-action-keys.sh" not in preflight_section.split("fi", 1)[0]
    assert "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY" not in job
    assert "readOnly: ${HOST_ROOT_READ_ONLY}" in job
    assert 'HOST_ROOT_READ_ONLY="true"' in job
    assert "mountPath: /host/run/gpu-fault-preflight-artifact" in job
    assert "chroot /host /usr/bin/env" in job
    assert "preflight-gpu-fault-node.sh" in job
    assert "GPU_FAULT_PREFLIGHT_CANDIDATE_BUNDLE" in job
    assert '"${condition}" == *Failed*' in job
    assert "GPU_FAULT_REQUIRE_ROLLBACK_SLOT" in preflight
    assert "host has no rollback runtime slot" in preflight
    assert "host has an unresolved GPU service quiesce state" in preflight
    assert "host Python 3.12 venv support is unavailable" in preflight
    assert "host has less than" in preflight


def test_installer_job_wait_is_event_driven_but_the_read_still_judges() -> None:
    """The 2s between Job reads is a bounded `kubectl wait`, not a timer.

    The wait only shortens the interval: a Complete Job ends it at once, while a
    Failed Job or a transient API error just exhausts the 2s bound and the
    condition read at the top of the loop decides, exactly as before.
    """

    job = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text()
    loop = job.split("deadline=$((SECONDS + INSTALLER_ACTIVE_DEADLINE_SECONDS", 1)[1]
    loop = loop.split("done", 1)[0]

    assert "sleep 2" not in loop
    assert 'wait "job/${JOB_NAME}"' in loop
    assert "--for=condition=complete --timeout=2s" in loop
    assert "|| true" in loop, "a timed-out wait must not abort the loop"
    assert '"${condition}" == *Failed*' in loop


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
    # The bearer token reaches the installer as a 0600 file, never on argv.
    assert '--token-file "${CONTROL_PLANE_TOKEN_FILE}"' in installer
    assert '--token "' not in installer
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
    assert (
        'GPU_FAULT_INSTALLER_TEMPLATE_SHA256="${TEMPLATE_SOURCE_SHA256}"' in reconciler
    )
    assert "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP" in reconciler
    assert 'if [[ -n "${TEMPLATE_CONFIG_MAP_OVERRIDE}" ]]' in reconciler
    assert (
        "REPLACE_WITH_INSTALLER_TEMPLATE_SHA256#${TEMPLATE_SOURCE_SHA256}" in reconciler
    )
    assert 'INSTALLER_TEMPLATE_SHA256="$(' in installer
    assert "invalid GPU_FAULT_INSTALLER_TEMPLATE_SHA256" in installer
    assert 'INSTALLER_TEMPLATE_SHA256="$(sha256sum "${MANIFEST}"' not in installer
    manifest = (ROOT / "deploy/dataplane/node-installer-reconciler.yaml").read_text()
    assert "GPU_FAULT_INSTALLER_TEMPLATE_PATH" in manifest
    assert "REPLACE_WITH_INSTALLER_TEMPLATE_CONFIG_MAP" in manifest
    assert "REPLACE_WITH_HYPERPOD_CLUSTER" in manifest
    assert 'allowed-nodes: "REPLACE_WITH_INSTALLER_ALLOWED_NODES"' in manifest
    assert 'max-unavailable: "REPLACE_WITH_INSTALLER_MAX_UNAVAILABLE"' in manifest
    assert 'value: "REPLACE_WITH_INSTALLER_ACTIVE_DEADLINE_SECONDS"' in manifest
    assert "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP" in manifest
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


def test_node_runtime_uses_content_addressed_atomic_slots() -> None:
    installer = NODE_SCRIPTS[0].read_text()
    runtime_units = (
        "gpu-fault-kernel-collector.service",
        "gpu-fault-metrics-collector.service",
        "gpu-fault-host-collector.service",
        "gpu-fault-log-collector.service",
        "gpu-fault-fabric-manager-collector.service",
        "gpu-fault-node-agent.service",
    )

    assert 'RUNTIME_RELEASE_DIR="${RUNTIME_RELEASES_DIR}/${WHEEL_SHA256}"' in installer
    assert (
        'prepare_runtime_slot "${RUNTIME_RELEASE_DIR}" "${WHEEL_SHA256}"' in installer
    )
    assert (
        'atomic_symlink "${RUNTIME_RELEASE_DIR}" "${RUNTIME_CURRENT_LINK}"' in installer
    )
    assert 'touch "${release_dir}/.complete"' in installer
    assert "runtime_record_digest" in installer
    assert '"${PYTHON_COMMAND}" -m venv /opt/gpu-fault/venv' not in installer
    assert 'PREVIOUS_CURRENT_TARGET="$(readlink -f "${RUNTIME_CURRENT_LINK}")"' in (
        installer
    )
    assert "restore_install_files" in installer
    assert "restore_unit_state" in installer
    for name in runtime_units:
        unit = (ROOT / "deploy/systemd" / name).read_text()
        assert "/opt/gpu-fault/current/venv/bin/" in unit, name


def test_legacy_node_venv_is_preserved_for_first_ab_rollback() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert 'LEGACY_VENV_PATH="${RUNTIME_ROOT}/venv"' in installer
    assert (
        'if [[ ! -e "${LEGACY_VENV_PATH}" && ! -L "${LEGACY_VENV_PATH}" ]]' in installer
    )
    assert 'atomic_symlink "${RUNTIME_CURRENT_LINK}/venv" "${LEGACY_VENV_PATH}"' in (
        installer
    )
    assert 'rm -rf "${LEGACY_VENV_PATH}"' not in installer


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
    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text()

    assert training_health_monitor_enabled({}) is False, (
        "the training-health monitor reads live workloads, so it must be opt-in"
    )
    assert (
        training_health_monitor_enabled({TRAINING_HEALTH_MONITOR_ENV: "true"}) is True
    )
    # Off by default, and every value goes through the one bash boolean parser
    # (S9): an unknown token is a hard error, not a silent "off".
    assert (
        "normalize_bool GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR "
        '"${GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR:-false}"'
    ) in deploy
    for manifest in (
        ROOT / "deploy/control-plane/base/control-plane-deployment.yaml",
        ROOT / "scripts/e2e/regional/manifests/xid45-correlation-canary.yaml",
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

    assert (
        "normalize_bool GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR "
        '"${GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR:-false}"'
    ) in deploy
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


WHEEL_BLOCK_START = 'MANIFEST_WHEEL_SHA256=""'
WHEEL_BLOCK_END = (
    'sha256 ${WHEEL_SHA256} does not match expected ${EXPECTED_WHEEL_SHA256}"'
)


def _wheel_discovery_probe(target: Path) -> Path:
    """Run the installer's own wheel discovery, not a paraphrase of it.

    ``bash -n`` and substring assertions both pass on a ``find`` that matches
    nothing, which is exactly how ``-maxdepth 1 -type f`` survived: the node
    bundle flattens the wheel into ``dist/``, so the rule looked right, while
    a repo checkout keeps it in the content-addressed ``dist/<release_id>/``
    and every checkout install died with "collector wheel not found".

    The probe takes the expected wheel digest as its second argument, the
    way the Job passes ``--wheel-sha256``; the digest is a hard gate.
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
        'EXPECTED_WHEEL_SHA256="${2:-}"\n'
        "PYTHON_COMMAND=python3\n"
        'WHEEL=""\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'printf \'WHEEL=%s\\nSHA=%s\\n\' "${WHEEL}" "${WHEEL_SHA256}"\n',
        encoding="utf-8",
    )
    return probe


def _run_probe(
    probe: Path, repo_dir: Path, expected: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(repo_dir), expected],
        capture_output=True,
        text=True,
        check=False,
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

    result = _run_probe(probe, checkout, digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={wheel}" in result.stdout
    assert f"SHA={digest}" in result.stdout

    bundle = tmp_path / "bundle"
    flat = bundle / "dist/gpu_fault_node_runtime-0.10.0.whl"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"bundle-wheel")

    result = _run_probe(probe, bundle, hashlib.sha256(b"bundle-wheel").hexdigest())
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"WHEEL={flat}" in result.stdout


def test_installer_refuses_a_wheel_the_release_manifest_disowns(tmp_path: Path) -> None:
    probe = _wheel_discovery_probe(tmp_path)
    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_control_plane-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"tampered")
    original = hashlib.sha256(b"original").hexdigest()
    (checkout / "dist/current-release.json").write_text(
        json.dumps(
            {"wheel": wheel.relative_to(checkout).as_posix(), "wheel_sha256": original}
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, checkout, original)
    assert result.returncode != 0
    assert "does not match" in result.stdout

    empty = tmp_path / "empty"
    (empty / "dist").mkdir(parents=True)
    result = _run_probe(probe, empty, original)
    assert result.returncode != 0
    assert "collector wheel not found" in result.stdout


def test_deploy_scripts_accept_xz_compressed_wheel_configmaps() -> None:
    """Wheel ConfigMaps hold ``<wheel>.xz`` since the control-plane wheel
    outgrew the 1 MiB ceiling; the GPU-stage reconciler deploy refused the
    executor wheel ConfigMap live because it only looked for the raw key."""
    reconciler = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text()
    assert (
        'if sys.argv[1] not in keys and sys.argv[1] + ".xz" not in keys:' in reconciler
    ), "reconciler deploy must accept the xz-compressed executor wheel key"
    role_split = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text()
    assert 'for key in (name + ".xz", name):' in role_split, (
        "role-split apply must derive the wheel digest from either form"
    )
    assert "lzma.decompress(data)" in role_split, role_split[:0]


# GPU health is not software integrity: a node with one GPU off the bus is the
# node the Agent exists to remediate, so `verify` must report it and still let
# the install stand. These two lists are the contract owner decision 6 fixes;
# a name may appear in both when the severity depends on the observation
# (zero GPUs enumerated is fatal, a partial enumeration is a warning).
VERIFY_WARN_NAMES = (
    "GPU persistence mode",
    "DCGM exporter supported metrics",
    "NVIDIA GPU enumeration",
)
VERIFY_FATAL_NAMES = (
    "NVIDIA GPU enumeration",
    "control plane health",
    "metrics collector service",
    "host telemetry collector service",
    "Fabric Manager SXID collector service",
    "GPU persistence service",
    "signed node action agent service",
    "signed node action agent health",
    "metrics delivered to control plane",
)


def _write_stub(directory: Path, name: str, body: str) -> None:
    stub = directory / name
    stub.write_text(f"#!/usr/bin/env bash\n{body}", encoding="utf-8")
    stub.chmod(0o755)


def _run_verify(
    workspace: Path,
    *,
    persistence_modes: tuple[str, ...] = ("Enabled",),
    persistence_exit: int = 0,
    enumerated_gpus: int = 8,
    enumeration_exit: int = 0,
    dcgm_metrics: bool = True,
    inactive_units: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the real verifier against PATH stubs for nvidia-smi/systemctl/curl.

    Substring assertions cannot tell a WARN that is counted and non-fatal from
    a WARN that still leaves ``failed=1`` behind, so the contract is checked by
    executing the script and reading its exit code.
    """
    binaries = workspace / "bin"
    binaries.mkdir(parents=True)
    enumeration = "".join(
        f"GPU {index}: NVIDIA H100 80GB HBM3 (UUID: GPU-{index:012d})\n"
        for index in range(enumerated_gpus)
    )
    modes = "".join(f"{mode}\n" for mode in persistence_modes)
    detailed = "".join(
        f"{index}, GPU-{index:012d}, {mode}\n"
        for index, mode in enumerate(persistence_modes)
    )
    _write_stub(
        binaries,
        "nvidia-smi",
        'case "$*" in\n'
        f"*index,uuid,persistence_mode*) printf '%s' {shlex.quote(detailed)};;\n"
        f"*persistence_mode*) printf '%s' {shlex.quote(modes)};"
        f" exit {persistence_exit};;\n"
        f"-L) printf '%s' {shlex.quote(enumeration)}; exit {enumeration_exit};;\n"
        "*) printf '0, GPU-000000000000, 41, 300\\n';;\n"
        "esac\n"
        "exit 0\n",
    )
    down = " ".join(("gpu-fault-log-collector.service", *inactive_units))
    _write_stub(
        binaries,
        "systemctl",
        'verb="${1:-}"\n'
        "shift || true\n"
        'while [[ "${1:-}" == -* ]]; do shift; done\n'
        'unit="${1:-}"\n'
        f'down="{down}"\n'
        'for name in ${down}; do\n'
        '    if [[ "${unit}" == "${name}" ]]; then\n'
        '        case "${verb}" in\n'
        "        is-active|is-enabled) exit 3 ;;\n"
        "        esac\n"
        "    fi\n"
        "done\n"
        "exit 0\n",
    )
    dcgm_body = (
        'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 41\n'
        if dcgm_metrics
        else "# the exporter answered without a single supported metric\n"
    )
    _write_stub(
        binaries,
        "curl",
        'output=""\n'
        'url=""\n'
        "while (( $# )); do\n"
        '    case "$1" in\n'
        '    --output) output="$2"; shift 2 ;;\n'
        "    -H|--cacert) shift 2 ;;\n"
        "    -*) shift ;;\n"
        '    *) url="$1"; shift ;;\n'
        "    esac\n"
        "done\n"
        "emit() {\n"
        '    if [[ -n "${output}" ]]; then printf \'%s\' "$1" > "${output}"\n'
        "    else printf '%s' \"$1\"\n"
        "    fi\n"
        "}\n"
        'case "${url}" in\n'
        "*/healthz) exit 0 ;;\n"
        f"*9400/metrics) emit {shlex.quote(dcgm_body)}; exit 0 ;;\n"
        '*/latest) emit "[{\\"observed_at\\": \\"$(date -u +%FT%TZ)\\"}]"; exit 0 ;;\n'
        "esac\n"
        "exit 22\n",
    )
    env_file = workspace / "collector.env"
    env_file.write_text(
        "GPU_FAULT_CONTROL_PLANE_URL=https://control-plane.invalid\n"
        "GPU_FAULT_CONTROL_PLANE_TOKEN=cluster-token\n"
        "GPU_FAULT_CLUSTER_ID=cluster-a\n"
        "NODE_NAME=node-a\n"
        "GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9400/metrics\n"
        "GPU_FAULT_METRICS_MODE=dcgm\n"
        "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR=false\n"
        "GPU_FAULT_NODE_AGENT_PORT=9099\n"
        "GPU_FAULT_NODE_ADVERTISE_URL=http://127.0.0.1:9099\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(NODE_SCRIPTS[1])],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": ":".join([str(binaries), "/usr/bin", "/bin"]),
            "GPU_FAULT_ENV_FILE": str(env_file),
        },
    )


def _summary(stdout: str) -> dict[str, int]:
    line = [row for row in stdout.splitlines() if row.startswith("SUMMARY ")]
    assert len(line) == 1, f"expected exactly one SUMMARY line, got {line}"
    return {
        key: int(value)
        for key, value in (field.split("=", 1) for field in line[0].split()[1:])
    }


def test_verify_warns_on_gpu_health_and_still_fails_on_software_integrity(
    tmp_path: Path,
) -> None:
    degraded = _run_verify(
        tmp_path / "degraded",
        persistence_modes=("Enabled", "Disabled", "Enabled"),
        dcgm_metrics=False,
    )

    assert degraded.returncode == 0, (
        "a GPU that lost persistence mode must not fail the install: "
        f"{degraded.stdout}{degraded.stderr}"
    )
    assert "WARN  GPU persistence mode" in degraded.stdout, degraded.stdout
    assert "WARN  DCGM exporter supported metrics" in degraded.stdout, degraded.stdout
    assert "FAIL" not in degraded.stdout, (
        f"no software-integrity gate was broken in this scenario: {degraded.stdout}"
    )
    assert _summary(degraded.stdout)["warn"] == 2, degraded.stdout
    assert _summary(degraded.stdout)["fail"] == 0, degraded.stdout
    assert _summary(degraded.stdout)["pass"] > 0, degraded.stdout

    stopped = _run_verify(
        tmp_path / "stopped",
        persistence_modes=("Disabled",),
        inactive_units=("gpu-fault-metrics-collector.service",),
    )

    assert stopped.returncode == 1, (
        "an inactive collector unit must stay fatal: "
        f"{stopped.stdout}{stopped.stderr}"
    )
    assert "FAIL  metrics collector service" in stopped.stdout, stopped.stdout
    assert "WARN  GPU persistence mode" in stopped.stdout, stopped.stdout
    assert _summary(stopped.stdout)["fail"] == 1, stopped.stdout
    assert _summary(stopped.stdout)["warn"] == 1, stopped.stdout


def test_verify_fails_when_no_gpu_is_enumerated_at_all(tmp_path: Path) -> None:
    empty = _run_verify(tmp_path / "empty", enumerated_gpus=0, enumeration_exit=1)

    assert empty.returncode == 1, (
        f"zero enumerated GPUs must stay fatal: {empty.stdout}{empty.stderr}"
    )
    assert "FAIL  NVIDIA GPU enumeration" in empty.stdout, empty.stdout

    partial = _run_verify(
        tmp_path / "partial", enumerated_gpus=7, enumeration_exit=255
    )

    assert partial.returncode == 0, (
        "a partial enumeration is the degraded case the Agent must reach: "
        f"{partial.stdout}{partial.stderr}"
    )
    assert "WARN  NVIDIA GPU enumeration" in partial.stdout, partial.stdout


def test_verify_keeps_every_software_integrity_check_fatal() -> None:
    verifier = NODE_SCRIPTS[1].read_text()

    warned = set(re.findall(r'^\s*warn "([^"]+)"', verifier, re.MULTILINE))
    fatal = set(re.findall(r'^\s*(?:check|fail) "([^"]+)"', verifier, re.MULTILINE))

    assert warned == set(VERIFY_WARN_NAMES), (
        f"only the GPU-health gates may warn, found {sorted(warned)}"
    )
    for name in VERIFY_FATAL_NAMES:
        assert name in fatal, f"{name} must stay a fatal verifier check"
    assert "exit \"${failed}\"" in verifier, (
        "the verifier exit code must still be driven by FAILs alone"
    )


DEGRADED_MARKER_START = (
    'DEGRADED_GPU_MARKER="/var/lib/gpu-fault/installer-degraded-gpu.json"'
)
DEGRADED_MARKER_END = (
    'clear_degraded_gpu_marker() {\n    rm -f "${DEGRADED_GPU_MARKER}"\n}'
)


def _degraded_gpu_probe(target: Path) -> Path:
    """Run the installer's own marker writer, not a paraphrase of it."""
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(DEGRADED_MARKER_START) == 1, (
        "the installer must declare exactly one degraded-GPU marker path"
    )
    assert installer.count(DEGRADED_MARKER_END) == 1, (
        "the installer must clear the degraded-GPU marker from one place"
    )
    start = installer.index(DEGRADED_MARKER_START)
    end = installer.index(DEGRADED_MARKER_END) + len(DEGRADED_MARKER_END)
    probe = target / "degraded-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "PYTHON_COMMAND=python3\n"
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'DEGRADED_GPU_MARKER="$1"\n'
        'record_degraded_gpu_marker "$2"\n',
        encoding="utf-8",
    )
    return probe


def test_installer_records_the_degraded_gpu_marker_instead_of_dying(
    tmp_path: Path,
) -> None:
    probe = _degraded_gpu_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(
        binaries,
        "nvidia-smi",
        "printf '%s\\n' "
        "'0, GPU-aaaaaaaaaaaa, Enabled' "
        "'1, [N/A], [N/A]' "
        "'2, GPU-cccccccccccc, Disabled'\n"
        "exit 255\n",
    )
    marker = tmp_path / "state" / "installer-degraded-gpu.json"
    reason = "persistence mode was not Enabled after the persistence unit restart"

    result = subprocess.run(
        ["bash", str(probe), str(marker), reason],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": f"{binaries}:/usr/bin:/bin"},
    )

    assert result.returncode == 0, (
        f"a degraded GPU must not abort the install: {result.stdout}{result.stderr}"
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["observed_at"].endswith("Z"), payload
    assert payload["gpus"] == [
        {
            "index": "0",
            "uuid": "GPU-aaaaaaaaaaaa",
            "persistence_mode": "Enabled",
            "error": "",
        },
        {"index": "1", "uuid": "[N/A]", "persistence_mode": "[N/A]", "error": reason},
        {
            "index": "2",
            "uuid": "GPU-cccccccccccc",
            "persistence_mode": "Disabled",
            "error": reason,
        },
    ], payload
    assert marker.stat().st_mode & 0o777 == 0o644, oct(marker.stat().st_mode)
    assert not list(marker.parent.glob("*.tmp")), (
        "the marker must be published by tmp+mv without leaving the temporary file"
    )


LEDGER_DRAIN_START = "in_progress_node_action_ids() {"
LEDGER_DRAIN_END = (
    "drain_node_agent_before_restart() {\n"
    "    systemctl is-active --quiet gpu-fault-node-agent.service || return 0\n"
    "    wait_for_node_action_ledger_idle\n"
    "}"
)


def _ledger_drain_probe(target: Path) -> Path:
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(LEDGER_DRAIN_START) == 1, (
        "the installer must read the IN_PROGRESS ledger rows from one place"
    )
    assert installer.count(LEDGER_DRAIN_END) == 1, (
        "the drain must be guarded by the Agent unit being active"
    )
    start = installer.index(LEDGER_DRAIN_START)
    end = installer.index(LEDGER_DRAIN_END) + len(LEDGER_DRAIN_END)
    probe = target / "drain-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "PYTHON_COMMAND=python3\n"
        'NODE_ACTION_DB="$1"\n'
        'SQLITE_COMMAND="$2"\n'
        'NODE_AGENT_STOP_TIMEOUT_SECONDS="${3:-0}"\n'
        'NODE_AGENT_LEDGER_POLL_SECONDS="${4:-1}"\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        "wait_for_node_action_ledger_idle\n"
        "printf 'DRAINED\\n'\n",
        encoding="utf-8",
    )
    return probe


def _ledger_with_state(path: Path, state: str) -> None:
    NodeActionLedger(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO results (command_id, attempt, payload, state)"
            " VALUES (?, ?, ?, ?)",
            ("cmd-in-flight", 1, "{}", state),
        )


def test_installer_refuses_to_restart_the_agent_mid_operation(tmp_path: Path) -> None:
    probe = _ledger_drain_probe(tmp_path)
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)

    def drain(
        sqlite_command: str, db: Path, *, timeout: int = 0, poll: int = 1
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe), str(db), sqlite_command, str(timeout), str(poll)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin"},
        )

    blocked = drain("", database)
    assert blocked.returncode != 0, (
        f"an IN_PROGRESS row must block the restart: {blocked.stdout}{blocked.stderr}"
    )
    assert "cmd-in-flight" in blocked.stdout, blocked.stdout

    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(binaries, "sqlite3", "printf 'cmd-from-sqlite3\\n'\n")
    via_sqlite3 = drain(str(binaries / "sqlite3"), database)
    assert via_sqlite3.returncode != 0, via_sqlite3.stdout + via_sqlite3.stderr
    assert "cmd-from-sqlite3" in via_sqlite3.stdout, via_sqlite3.stdout

    finished = tmp_path / "finished.db"
    _ledger_with_state(finished, "COMPLETED")
    assert "DRAINED" in drain("", finished).stdout, "a settled ledger must not wait"
    assert "DRAINED" in drain("", tmp_path / "absent.db").stdout, (
        "a first install has no ledger and must not wait"
    )


def test_installer_drains_the_ledger_before_it_touches_the_agent_unit() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert 'NODE_AGENT_STOP_TIMEOUT_SECONDS="1900"' in installer, (
        "the installer must wait as long as the unit's TimeoutStopSec before dying"
    )
    stop_loop = installer.index('if systemctl is-active --quiet "${runtime_unit}"')
    restart = installer.index("systemctl restart gpu-fault-node-agent.service")
    calls = [
        match.start()
        for match in re.finditer(
            r"^ +drain_node_agent_before_restart$", installer, re.MULTILINE
        )
    ]
    assert len(calls) == 2, (
        f"expected a drain before the stop loop and before the restart, got {calls}"
    )
    assert calls[0] < stop_loop, "stopping the unit interrupts the same operation"
    assert calls[1] < restart, "the restart must not preempt an in-flight command"


def _drain(
    probe: Path,
    database: Path,
    *,
    sqlite_command: str = "",
    timeout: int = 0,
    poll: int = 1,
    path: str = "/usr/bin:/bin",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), str(database), sqlite_command, str(timeout), str(poll)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": path},
    )


def test_installer_treats_an_unreadable_ledger_as_in_flight(tmp_path: Path) -> None:
    """A ledger that cannot be read must not be mistaken for an idle one.

    Both readers used to swallow every error and print nothing, which the wait
    loop scored as "no in-flight commands" -- so a corrupt database, a ledger
    whose ``results`` table has not been created yet and a ``SQLITE_BUSY`` from
    the very write the guard is meant to notice all let the installer stop the
    Agent in the middle of a node action. Read failure now keeps the installer
    waiting and finally kills the install.
    """

    probe = _ledger_drain_probe(tmp_path)

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a SQLite database\n" * 64)
    corrupted = _drain(probe, corrupt)
    assert corrupted.returncode != 0, (
        f"a corrupt ledger must block the restart: {corrupted.stdout}"
    )
    assert "WARN  node action ledger unreadable" in corrupted.stdout, corrupted.stdout
    assert "DRAINED" not in corrupted.stdout, (
        "an unreadable ledger must never report the Agent as drained"
    )

    tableless = tmp_path / "tableless.db"
    with sqlite3.connect(tableless) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    without_table = _drain(probe, tableless)
    assert without_table.returncode != 0, (
        f"a ledger without the results table must block: {without_table.stdout}"
    )
    assert "DRAINED" not in without_table.stdout, without_table.stdout

    binaries = tmp_path / "bin"
    binaries.mkdir()
    argument_log = tmp_path / "sqlite3-args.txt"
    _write_stub(
        binaries,
        "sqlite3",
        f'printf \'%s\\n\' "$*" >> {shlex.quote(str(argument_log))}\n'
        f'if [[ -f {shlex.quote(str(tmp_path / "BUSY_MARKER"))} ]];'
        " then exit 0; fi\n"
        "printf 'Error: database is locked\\n' >&2\n"
        "exit 1\n",
    )
    settled = tmp_path / "settled.db"
    _ledger_with_state(settled, "COMPLETED")
    busy = _drain(probe, settled, sqlite_command=str(binaries / "sqlite3"))
    assert busy.returncode != 0, f"a locked ledger must block: {busy.stdout}"
    assert "WARN  node action ledger unreadable" in busy.stdout, busy.stdout
    assert "DRAINED" not in busy.stdout, busy.stdout
    assert ".timeout 5000" in argument_log.read_text(encoding="utf-8"), (
        "the sqlite3 CLI has no busy timeout of its own, so the install must set one"
    )

    (tmp_path / "BUSY_MARKER").write_text("readable now\n", encoding="utf-8")
    readable = _drain(
        probe,
        settled,
        sqlite_command=str(binaries / "sqlite3"),
        path=f"{binaries}:/usr/bin:/bin",
    )
    assert "DRAINED" in readable.stdout, (
        f"a ledger that becomes readable must let the install proceed: {readable.stdout}"
    )
    assert readable.returncode == 0, readable.stdout + readable.stderr


def test_installer_waits_for_an_in_flight_command_to_settle(tmp_path: Path) -> None:
    """The wait loop must actually wait, not only die at the deadline."""

    probe = _ledger_drain_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _write_stub(
        binaries,
        "sqlite3",
        f'marker={shlex.quote(str(tmp_path / "polled"))}\n'
        'if [[ -f "${marker}" ]]; then exit 0; fi\n'
        'printf served > "${marker}"\n'
        "printf 'cmd-slow\\n'\n",
    )
    database = tmp_path / "node-actions.db"
    _ledger_with_state(database, IN_PROGRESS_STATE)

    result = _drain(
        probe,
        database,
        sqlite_command=str(binaries / "sqlite3"),
        timeout=5,
        poll=1,
    )

    assert "waiting for in-flight node actions: cmd-slow" in result.stdout, (
        f"the operator must see why the upgrade is paused: {result.stdout}"
    )
    assert "DRAINED" in result.stdout, (
        f"the install must proceed once the command settles: {result.stdout}"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installer_python_ledger_reader_has_a_busy_timeout() -> None:
    installer = NODE_SCRIPTS[0].read_text()

    assert 'f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=5' in installer, (
        "the python ledger reader must wait out a writer instead of erroring at once"
    )
    assert "2>/dev/null || true\n    fi\n}" not in installer, (
        "the ledger readers must not swallow their own failures"
    )


PERSISTENCE_BLOCK_START = (
    '    persistence_modes="$(nvidia-smi --query-gpu=persistence_mode \\\n'
    '        --format=csv,noheader 2>/dev/null || true)"'
)
PERSISTENCE_BLOCK_END = (
    '    elif [[ "${DEGRADED_GPU_RECORDED}" == "false" ]]; then\n'
    "        clear_degraded_gpu_marker\n"
    "    fi"
)


def _persistence_probe(target: Path) -> Path:
    """Run the installer's own persistence verdict with the marker calls stubbed."""
    installer = NODE_SCRIPTS[0].read_text()
    assert installer.count(PERSISTENCE_BLOCK_START) == 1, (
        "the installer must capture the persistence query in exactly one place"
    )
    assert installer.count(PERSISTENCE_BLOCK_END) == 1, (
        "the installer must decide about the degraded marker in one place"
    )
    start = installer.index(PERSISTENCE_BLOCK_START)
    end = installer.index(PERSISTENCE_BLOCK_END) + len(PERSISTENCE_BLOCK_END)
    probe = target / "persistence-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'DEGRADED_GPU_RECORDED="false"\n'
        "record_degraded_gpu_marker() { printf 'RECORDED %s\\n' \"$1\"; }\n"
        "clear_degraded_gpu_marker() { printf 'CLEARED\\n'; }\n"
        f"{installer[start:end]}\n",
        encoding="utf-8",
    )
    return probe


def test_installer_does_not_read_a_dead_driver_as_healthy_persistence(
    tmp_path: Path,
) -> None:
    """An empty persistence query has no line that differs from "Enabled".

    Piping a failed ``nvidia-smi`` into ``grep -Fvx Enabled`` therefore scored a
    driver that answered nothing as "every GPU is in persistence mode", and the
    new marker logic went one step further and deleted the marker an earlier
    install had written.
    """

    probe = _persistence_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{binaries}:/usr/bin:/bin", "NO_START": "false"},
        )

    _write_stub(binaries, "nvidia-smi", "exit 255\n")
    dead = run()
    assert dead.returncode == 0, (
        f"a dead persistence query must not abort the install: {dead.stderr}"
    )
    assert "RECORDED" in dead.stdout, (
        f"a driver that reports no GPU is a degraded GPU observation: {dead.stdout}"
    )
    assert "CLEARED" not in dead.stdout, (
        "an unreadable driver must never clear a marker an earlier install wrote"
    )

    _write_stub(binaries, "nvidia-smi", "printf 'Enabled\\nEnabled\\n'\n")
    healthy = run()
    assert healthy.returncode == 0, healthy.stderr
    assert "CLEARED" in healthy.stdout, (
        f"a healthy node must drop a stale marker: {healthy.stdout}"
    )
    assert "RECORDED" not in healthy.stdout, healthy.stdout


def test_verify_warns_when_the_persistence_query_answers_nothing(
    tmp_path: Path,
) -> None:
    result = _run_verify(tmp_path, persistence_modes=(), persistence_exit=255)

    assert "WARN  GPU persistence mode (nvidia-smi reported no GPU)" in result.stdout, (
        f"an empty persistence query is not a pass: {result.stdout}"
    )
    assert "PASS  GPU persistence mode" not in result.stdout, result.stdout
    assert result.returncode == 0, (
        f"a GPU-health warning must not fail the verifier: {result.stdout}"
    )
    assert _summary(result.stdout)["warn"] >= 1, result.stdout


def test_installer_rollback_removes_the_degraded_gpu_marker() -> None:
    """A first install that warned and then died must leave no state root.

    ``rmdir`` refuses a directory that still holds the marker, so the state root
    the installer created would survive its own rollback.
    """

    installer = NODE_SCRIPTS[0].read_text()
    branch = installer.index(
        'if [[ "${status}" -ne 0 && "${STATE_ROOT_CREATED}" == "true" ]]; then'
    )
    removal = installer.index(
        "rm -f /var/lib/gpu-fault/installer-degraded-gpu.json", branch
    )
    rmdir = installer.index("rmdir /var/lib/gpu-fault", branch)
    assert removal < rmdir, (
        "the marker must be removed before the state root is reclaimed"
    )


PREFLIGHT_ENUMERATION_START = (
    'gpu_enumeration="$(\n'
    "    host_shell \"nvidia-smi -L 2>/dev/null || printf 'ENUMERATION_FAILED\\n'\"\n"
    ')"'
)
PREFLIGHT_ENUMERATION_END = (
    'if [[ "${gpu_enumeration}" == *ENUMERATION_FAILED* ]]; then\n'
    "    printf 'WARN  NVIDIA GPU enumeration (only %s GPU(s) are enumerable)\\n' \\\n"
    '        "${enumerated_gpus}"\n'
    "fi"
)


def _preflight_enumeration_probe(target: Path) -> Path:
    """Run the preflight's own enumeration gate.

    ``host_shell`` chroots into the host, which a unit test cannot do, so the
    probe replaces it with the same ``bash -ceu`` the chroot would run -- that
    is what makes the quoting of the fallback part of the contract.
    """
    preflight = (ROOT / "deploy/node/preflight-gpu-fault-node.sh").read_text()
    assert preflight.count(PREFLIGHT_ENUMERATION_START) == 1, (
        "the preflight must enumerate GPUs in exactly one place"
    )
    assert preflight.count(PREFLIGHT_ENUMERATION_END) == 1, (
        "the preflight must warn about a partial enumeration in one place"
    )
    start = preflight.index(PREFLIGHT_ENUMERATION_START)
    end = preflight.index(PREFLIGHT_ENUMERATION_END) + len(PREFLIGHT_ENUMERATION_END)
    probe = target / "preflight-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        "host_shell() { /bin/bash -ceu \"$1\"; }\n"
        f"{preflight[start:end]}\n"
        "printf 'PREFLIGHT_CONTINUED\\n'\n",
        encoding="utf-8",
    )
    return probe


def test_preflight_blocks_only_a_node_with_no_enumerable_gpu(tmp_path: Path) -> None:
    """One unreadable GPU must not disqualify the node from getting the Agent.

    ``nvidia-smi -L`` exits non-zero as soon as a single GPU is unreadable while
    it still lists the healthy ones, so the old ``die`` kept the release
    candidate off exactly the nodes that needed remediation.
    """

    probe = _preflight_enumeration_probe(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(probe)],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{binaries}:/usr/bin:/bin"},
        )

    healthy = "".join(f"GPU {index}: NVIDIA H100 80GB HBM3\n" for index in range(8))
    _write_stub(binaries, "nvidia-smi", f"printf '%s' {shlex.quote(healthy)}\n")
    intact = run()
    assert intact.returncode == 0, intact.stdout + intact.stderr
    assert "WARN" not in intact.stdout, intact.stdout
    assert "PREFLIGHT_CONTINUED" in intact.stdout, intact.stdout

    partial = "".join(f"GPU {index}: NVIDIA H100 80GB HBM3\n" for index in range(7))
    _write_stub(
        binaries,
        "nvidia-smi",
        f"printf '%s' {shlex.quote(partial)}\n"
        "printf 'Unable to determine the device handle for GPU 7\\n' >&2\n"
        "exit 255\n",
    )
    degraded = run()
    assert degraded.returncode == 0, (
        f"a partial enumeration must not block the rollout: {degraded.stdout}"
    )
    assert (
        "WARN  NVIDIA GPU enumeration (only 7 GPU(s) are enumerable)"
        in degraded.stdout
    ), degraded.stdout
    assert "PREFLIGHT_CONTINUED" in degraded.stdout, degraded.stdout

    _write_stub(binaries, "nvidia-smi", "printf 'No devices were found\\n'\nexit 9\n")
    empty = run()
    assert empty.returncode != 0, (
        f"a node with no enumerable GPU is still fatal: {empty.stdout}"
    )
    assert "DIE: NVIDIA driver enumerated zero GPUs" in empty.stdout, empty.stdout
    assert "PREFLIGHT_CONTINUED" not in empty.stdout, empty.stdout


def test_preflight_no_longer_dies_on_a_partial_enumeration() -> None:
    preflight = (ROOT / "deploy/node/preflight-gpu-fault-node.sh").read_text()

    assert "NVIDIA driver cannot enumerate GPUs" not in preflight, (
        "a non-zero nvidia-smi -L is no longer by itself a preflight failure"
    )
