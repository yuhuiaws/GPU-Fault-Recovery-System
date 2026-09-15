"""Static contracts of the node rollout path.

Covers the installer's CLI surface and config-digest environment, the
regional installer Job and its reconciler (preflight-only mode, node
action key derivation, connection Secret, event-driven Job wait), the
content-addressed runtime slots and legacy venv, ``deploy.sh`` ordering
around the agent pin migration, the default-off collector flags (node log,
training-health, Kubernetes HMA, nvidia-smi metrics) and the failsafe
quiesce configuration the production installer carries.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from gpu_fault.env_validation import (
    TRAINING_HEALTH_MONITOR_ENV,
    training_health_monitor_enabled,
)
from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT


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
    assert (
        "--allow-efa-driver-remediation  Allow EFA PCI driver rebind after "
        "workload stop; default"
    ) in result.stdout
    assert "--disable-efa-driver-remediation Do not allow EFA PCI driver rebind" in (
        result.stdout
    )
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


def test_reconciler_product_reuse_is_gated_and_reports_no_key_material() -> None:
    """The hints an earlier run of the release leaves are advisory: the live
    node set must still hash to the recorded digest, the reused template must
    still carry the content its name promises, the preflight never reuses, and
    the product report names digests and objects only. The behaviour itself is
    exercised end to end in ``test_release_installer_product_reuse``."""

    deploy = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text()

    assert (
        'REUSE_NODE_SET_SHA256="${GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256:-}"'
        in deploy
    )
    assert "invalid GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256" in deploy
    assert "invalid GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP" in deploy
    assert (
        '"${PREFLIGHT_ONLY}" != "true" && -n "${REUSE_NODE_SET_SHA256}" &&\n'
        '    "${REUSE_NODE_SET_SHA256}" == "${NODE_SET_SHA256}"'
    ) in deploy, "reuse needs the live node set and never applies to a preflight"
    assert '"gpu-fault-node-installer-template-${reused_sha256:0:12}"' in deploy, (
        "a reused template must still hash to its content-addressed name"
    )
    report = deploy.split('if [[ -n "${PRODUCTS_FILE}" ]]; then', 1)[1]
    report = report.split("\nfi\n", 1)[0]
    for field in (
        "node_set_sha256",
        "node_action_keys_provisioned",
        "template_config_map",
        "template_content_sha256",
        "template_rendered",
    ):
        assert f'"{field}"' in report
    assert "MASTER" not in report and "key_dir" not in report, (
        "the product report carries no key material"
    )
    preflight_section = deploy.split('if [[ "${PREFLIGHT_ONLY}" == "true" ]]', 1)[1]
    assert "PRODUCTS_FILE" not in preflight_section.split("exit 0", 1)[0], (
        "a preflight reports no products"
    )


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


def _shell_excerpt(script: str, pattern: str) -> str:
    matches = re.findall(pattern, script, re.M | re.S)
    assert len(matches) == 1, f"expected one match for {pattern!r}, got {len(matches)}"
    return matches[0]


def _installer_efa_switch(tmp_path: Path, *arguments: str) -> tuple[str, list[str]]:
    """Run the installer's own flag parser and allowed-operations block.

    Stitched from the script text rather than re-typed, so a renamed variable
    or a rewritten case arm fails here instead of quietly testing a copy.
    Returns the ALLOW_EFA_DRIVER_REMEDIATION value and NODE_ALLOWED_OPERATIONS.
    """

    installer = NODE_SCRIPTS[0].read_text(encoding="utf-8")
    parser = _shell_excerpt(installer, r"^while \[\[ \$# -gt 0 \]\]; do\n.*?^done\n")
    operations = _shell_excerpt(
        installer,
        r'^    NODE_ALLOWED_OPERATIONS="COLLECT_HUNG_TRIAGE.*?'
        r'UPDATE_SOFTWARE_FIRMWARE"\n    fi\n',
    )
    defaults = "".join(
        line + "\n"
        for line in re.findall(
            r"^(?:ALLOW_[A-Z_]+|MEMORY_FIELD_DIAGNOSTIC_COMMAND)=.*$", installer, re.M
        )
    )
    probe = tmp_path / "efa-switch.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'die() { printf "die: %s\\n" "$*" >&2; exit 1; }\n'
        'require_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }\n'
        "usage() { :; }\n"
        "print_config_digest_environment() { :; }\n"
        + defaults
        + parser
        + operations
        + 'printf "%s\\n%s\\n" "${ALLOW_EFA_DRIVER_REMEDIATION}" '
        '"${NODE_ALLOWED_OPERATIONS}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(probe), *arguments], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    allowed, operations_value = result.stdout.splitlines()
    return allowed, operations_value.split(",")


def test_efa_driver_remediation_is_on_by_default_and_switchable_off(
    tmp_path: Path,
) -> None:
    installer = NODE_SCRIPTS[0].read_text(encoding="utf-8")
    assert 'ALLOW_EFA_DRIVER_REMEDIATION="true"' in installer
    assert (
        '--allow-efa-driver-remediation) ALLOW_EFA_DRIVER_REMEDIATION="true"; shift ;;'
    ) in installer
    assert (
        '--disable-efa-driver-remediation) ALLOW_EFA_DRIVER_REMEDIATION="false"; '
        "shift ;;"
    ) in installer
    guard = installer.split(
        'die "node mutation options require --enable-node-agent"', 1
    )[0].rsplit("elif [[", 1)[1]
    assert "ALLOW_EFA_DRIVER_REMEDIATION" not in guard, (
        "a default-on switch listed in the --enable-node-agent guard would make "
        "every collector-only install die"
    )

    allowed, operations = _installer_efa_switch(tmp_path)
    assert allowed == "true"
    assert "REMEDIATE_EFA_DRIVER" in operations

    allowed, operations = _installer_efa_switch(
        tmp_path, "--disable-efa-driver-remediation"
    )
    assert allowed == "false"
    assert "REMEDIATE_EFA_DRIVER" not in operations
    assert "RESET_GPU" in operations, "the switch drops only its own operation"

    allowed, operations = _installer_efa_switch(
        tmp_path, "--allow-efa-driver-remediation"
    )
    assert allowed == "true"
    assert "REMEDIATE_EFA_DRIVER" in operations


def test_installer_job_forwards_the_efa_remediation_switch() -> None:
    """A false GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION must reach the node.

    The installer is default-on, so merely omitting --allow-efa-driver-remediation
    changed nothing; the Job has to say --disable-efa-driver-remediation.
    """

    job = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'ENABLE_EFA_DRIVER_REMEDIATION="${GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION:-true}"'
    ) in job
    assert "GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION must be true or false" in job
    assert (
        'ENABLE_EFA_DRIVER_REMEDIATION="\\${ENABLE_EFA_DRIVER_REMEDIATION}"' in job
    ), (
        "the Job env is not forwarded through `chroot /host /usr/bin/env` into "
        "the installer shell"
    )
    block = _shell_excerpt(
        job,
        r'^\s*if \[\[ "\\\$\{ENABLE_EFA_DRIVER_REMEDIATION\}" == "true" \]\]; then\n'
        r".*?^\s*fi\n",
    )
    for value, expected in (
        ("true", "--allow-efa-driver-remediation"),
        ("false", "--disable-efa-driver-remediation"),
    ):
        result = subprocess.run(
            [
                "bash",
                "-c",
                "set -euo pipefail\nremediation_args=()\n"
                + block.replace("\\$", "$")
                + 'printf "%s\\n" "${remediation_args[@]}"\n',
            ],
            env={"PATH": "/usr/bin:/bin", "ENABLE_EFA_DRIVER_REMEDIATION": value},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == [expected], result.stdout
