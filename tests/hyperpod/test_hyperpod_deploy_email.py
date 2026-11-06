import os
import re
import subprocess
from pathlib import Path

from gpu_fault.gpu_metrics import GpuMetricsThresholds

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "hyperpod" / "deploy.sh"
MANIFEST = ROOT / "deploy" / "control-plane" / "base" / "control-plane-deployment.yaml"
COMPLETION_MANIFEST = ROOT / "deploy" / "dataplane" / "completion-watcher.yaml"
NODE_INSTALLER = ROOT / "deploy" / "node" / "run-hyperpod-installer-job.sh"


def test_hyperpod_deploy_script_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_hyperpod_deploy_script_expands_passive_fallback() -> None:
    result = subprocess.run(
        [str(SCRIPT), "invalid"],
        env={
            **os.environ,
            "GPU_FAULT_ALLOW_LEGACY_DEPLOY": "1",
            "GPU_FAULT_PASSIVE_STOP_FALLBACK_SECONDS": "30",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "bad substitution" not in result.stderr
    assert "Usage:" in result.stderr


def test_hyperpod_deploy_refuses_to_run_without_the_legacy_opt_in() -> None:
    """The script has no regional caller; a stray invocation must stop before
    any step runs, with a pointer to the real entry point."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key != "GPU_FAULT_ALLOW_LEGACY_DEPLOY"
    }
    result = subprocess.run(
        [str(SCRIPT), "validate"], env=env, text=True, capture_output=True
    )

    assert result.returncode == 64
    assert "gpu-fault-admin deploy" in result.stderr
    assert "GPU_FAULT_ALLOW_LEGACY_DEPLOY=1" in result.stderr
    assert "Validate platform" not in result.stdout


def test_agent_consistency_ignores_revoked_and_expired_records() -> None:
    script = SCRIPT.read_text()

    assert 'item["lifecycle_state"] == "ACTIVE"' in script
    assert "for item in active" in script
    assert "print(len(active))" in script


def test_deploy_uses_content_addressed_wheel_configmap() -> None:
    script = SCRIPT.read_text()

    assert 'WHEEL_CONFIGMAP_NAME=""' in script
    assert 'EXECUTOR_WHEEL_CONFIGMAP_NAME=""' in script
    assert '"${WHEEL_SHA256:0:12}"' in script
    assert '"${EXECUTOR_WHEEL_SHA256:0:12}"' in script
    assert ("s/gpu-fault-control-plane-wheel-0100/${WHEEL_CONFIGMAP_NAME}/g") in script
    assert (
        "s/gpu-fault-executor-wheel-0100/${EXECUTOR_WHEEL_CONFIGMAP_NAME}/g"
    ) in script
    assert 'get configmap \\\n        "${WHEEL_CONFIGMAP_NAME}"' in script
    assert 'get configmap \\\n        "${EXECUTOR_WHEEL_CONFIGMAP_NAME}"' in script
    assert "get configmap \\\n        gpu-fault-control-plane-wheel-0100" not in script
    for artifact in ("control-plane-wheel", "executor-wheel", "node-runtime-wheel"):
        assert artifact in script
    assert 'GPU_FAULT_INSTALLER_ARTIFACT_SHA256="${NODE_WHEEL_SHA256}"' in script
    assert "required-agent-compatibility-digest=" in script
    assert "required-regional-executor-compatibility-digest=" in script


def test_deploy_script_configures_email_secret_ses_and_irsa() -> None:
    script = SCRIPT.read_text()

    assert 'EMAIL_SECRET_NAME="gpu-fault-email"' in script
    assert "GPU_FAULT_EMAIL_SENDER and GPU_FAULT_EMAIL_RECIPIENTS" in script
    assert "aws sesv2 get-email-identity" in script
    assert "aws sesv2 get-configuration-set" in script
    assert '"Action": "ses:SendEmail"' in script
    assert '"ses:FromAddress": os.environ["EMAIL_SENDER"]' in script
    assert "--from-literal=email-sender=" in script
    assert "--from-literal=email-recipients=" in script
    assert '"GPU_FAULT_ALLOW_EMAIL=${ALLOW_EMAIL}"' in script
    assert '"GPU_FAULT_SES_CONFIGURATION_SET=${SES_CONFIGURATION_SET}"' in script
    assert (
        "GPU_FAULT_PASSIVE_STOP_FALLBACK_SECONDS=${PASSIVE_STOP_FALLBACK_SECONDS}"
    ) in script


def test_control_plane_reads_addresses_from_optional_secret() -> None:
    manifest = MANIFEST.read_text()

    assert "name: gpu-fault-control-plane-wheel-0100" in manifest
    assert "name: GPU_FAULT_EMAIL_SENDER" in manifest
    assert "name: GPU_FAULT_EMAIL_RECIPIENTS" in manifest
    assert "name: GPU_FAULT_EMAIL_SUBJECT_PREFIX" in manifest
    assert "name: GPU_FAULT_SITE_ID" in manifest
    assert "name: GPU_FAULT_AWS_ACCOUNT_ID" in manifest
    assert manifest.count("name: gpu-fault-email") == 5
    assert "key: email-sender" in manifest
    assert "key: email-recipients" in manifest
    assert "key: email-subject-prefix" in manifest
    assert "key: site-id" in manifest
    assert "key: aws-account-id" in manifest
    assert manifest.count("optional: true") >= 5


def test_control_watchers_tolerate_solution_quarantine_taint() -> None:
    manifest = COMPLETION_MANIFEST.read_text()
    assert "key: gpu-fault.io/quarantined" in manifest
    assert "effect: NoSchedule" in manifest


def test_hyperpod_deploy_wires_pinned_field_diagnostic() -> None:
    script = SCRIPT.read_text()
    manifest = MANIFEST.read_text()

    assert "GPU_FAULT_ENABLE_FIELD_DIAGNOSTIC" in script
    assert "GPU_FAULT_FIELD_DIAGNOSTIC_COMMAND" in script
    assert "GPU_FAULT_FIELD_DIAGNOSTIC_SHA256" in script
    assert "GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_COMMAND" in script
    assert "GPU_FAULT_MEMORY_FIELD_DIAGNOSTIC_SHA256" in script
    assert 'owners["nvlinkDiagnostics"]' in script
    assert 'owners["memoryDiagnostics"]' in script
    assert "RUN_FIELD_DIAGNOSTIC" in script
    assert '"mechanicalInspection"' in script
    assert "RUN_NVLINK74_WORKFLOW" in manifest
    assert "CHECK_MECHANICALS" in manifest


def test_hyperpod_deploy_can_enable_xid_firmware_workflow() -> None:
    script = SCRIPT.read_text()

    assert "GPU_FAULT_ENABLE_FIRMWARE_UPDATE" in script
    assert "GPU_FAULT_TARGET_FIRMWARE_VERSION" in script
    assert "GPU_FAULT_ENABLE_NODE_FIRMWARE_UPDATE" in script


def test_hyperpod_deploy_rejects_invalid_dcgm_exporter_mode() -> None:
    result = subprocess.run(
        [str(SCRIPT), "validate"],
        env={
            **os.environ,
            "GPU_FAULT_ALLOW_LEGACY_DEPLOY": "1",
            "GPU_FAULT_DCGM_EXPORTER_MODE": "unexpected",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "must be auto, existing, managed, or disabled" in result.stderr


def test_hyperpod_deploy_supports_external_dcgm_exporter() -> None:
    script = SCRIPT.read_text()
    installer = NODE_INSTALLER.read_text()

    assert "GPU_FAULT_DCGM_EXPORTER_MODE:-auto" in script
    assert "GPU_FAULT_DCGM_METRICS_URL" in script
    assert "GPU_FAULT_DCGM_REQUIRED_METRICS" in script
    assert "render_dcgm_metrics_url()" in script
    assert "probe_dcgm_node()" in script
    assert "resolve_dcgm_exporter_mode()" in script
    assert "configure_dcgm_exporter()" in script
    assert "MIXED compatible=" in script
    assert "managed DCGM exporter requires free port 9400" in script
    assert 'RESOLVED_DCGM_EXPORTER_MODE="existing"' in script
    assert 'RESOLVED_DCGM_EXPORTER_MODE="managed"' in script
    assert 'RESOLVED_DCGM_EXPORTER_MODE="disabled"' in script
    assert 'node_dcgm_mode="disabled"' in script
    assert 'GPU_FAULT_DCGM_METRICS_URL="${dcgm_metrics_url}"' in script

    assert "GPU_FAULT_DCGM_EXPORTER_MODE:-existing" in installer
    assert "DCGM_METRICS_URL_B64" in installer
    assert "metrics_mode=nvidia-smi" in installer
    assert "--enable-nvidia-smi-metrics-collector" in installer
    assert '--dcgm-exporter "\\${DCGM_EXPORTER_MODE}"' in installer
    assert '--dcgm-metrics-url "\\${dcgm_metrics_url}"' in installer


#: Every GpuMetricsThresholds variable the legacy script wires, and its field.
LEGACY_THRESHOLD_FIELDS = {
    "GPU_FAULT_GPU_TEMP_WARNING_C": "gpu_temperature_warning_c",
    "GPU_FAULT_GPU_TEMP_CRITICAL_C": "gpu_temperature_critical_c",
    "GPU_FAULT_MEMORY_TEMP_WARNING_C": "memory_temperature_warning_c",
    "GPU_FAULT_MEMORY_TEMP_CRITICAL_C": "memory_temperature_critical_c",
    "GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C": "gpu_temperature_warning_margin_c",
    "GPU_FAULT_GPU_TEMP_SHUTDOWN_MARGIN_C": "gpu_temperature_shutdown_margin_c",
    "GPU_FAULT_MEMORY_TEMP_WARNING_MARGIN_C": "memory_temperature_warning_margin_c",
    "GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE": (
        "pcie_replay_rate_warning_per_minute"
    ),
    "GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL": "nvlink_error_delta_critical",
    "GPU_FAULT_POWER_VIOLATION_DELTA_WARNING_US": "power_violation_delta_warning_us",
    "GPU_FAULT_THERMAL_VIOLATION_DELTA_WARNING_US": (
        "thermal_violation_delta_warning_us"
    ),
    "GPU_FAULT_DCGM_CORRELATION_WINDOW_SECONDS": "correlation_window_seconds",
    "GPU_FAULT_DCGM_POWER_LIMIT_RATIO": "power_limit_ratio",
    "GPU_FAULT_DCGM_POWER_CORRELATION_MIN_UTILIZATION_PERCENT": (
        "power_correlation_min_utilization_percent"
    ),
    "GPU_FAULT_DCGM_ECC_SBE_DELTA_WARNING": "ecc_sbe_delta_warning",
    "GPU_FAULT_DCGM_RETIRED_PAGES_SBE_DELTA_WARNING": (
        "retired_pages_sbe_delta_warning"
    ),
    "GPU_FAULT_DCGM_RETIRED_PAGES_DBE_DELTA_CRITICAL": (
        "retired_pages_dbe_delta_critical"
    ),
    "GPU_FAULT_DCGM_ROW_REMAP_CORRECTABLE_DELTA_WARNING": (
        "row_remap_correctable_delta_warning"
    ),
    "GPU_FAULT_DCGM_CORRECTABLE_MEMORY_DRAIN_CONSECUTIVE_SAMPLES": (
        "correctable_memory_drain_consecutive_samples"
    ),
}


def test_hyperpod_deploy_threshold_defaults_are_the_application_defaults() -> None:
    """The legacy script sets every threshold it wires explicitly, so a default
    it carries replaces the application's. The violation deltas once read 1 us,
    a leftover of the old application default the hardening raised to the
    10000 us site noise bound; nothing chose 1 for the legacy path."""

    script = SCRIPT.read_text()
    defaults = GpuMetricsThresholds()

    for name, field_name in LEGACY_THRESHOLD_FIELDS.items():
        match = re.search(r"\$\{" + name + r":-([0-9.]+)\}", script)
        assert match is not None, f"{name} has no default in the legacy script"
        assert float(match.group(1)) == getattr(defaults, field_name), (
            f"{name} defaults to {match.group(1)} in the legacy script, "
            f"the application to {getattr(defaults, field_name)}"
        )


def test_hyperpod_deploy_wires_gpu_metric_thresholds() -> None:
    script = SCRIPT.read_text()

    for name in (
        "GPU_FAULT_GPU_TEMP_WARNING_C",
        "GPU_FAULT_GPU_TEMP_CRITICAL_C",
        "GPU_FAULT_MEMORY_TEMP_WARNING_C",
        "GPU_FAULT_MEMORY_TEMP_CRITICAL_C",
        "GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C",
        "GPU_FAULT_GPU_TEMP_SHUTDOWN_MARGIN_C",
        "GPU_FAULT_MEMORY_TEMP_WARNING_MARGIN_C",
        "GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE",
        "GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL",
        "GPU_FAULT_POWER_VIOLATION_DELTA_WARNING_US",
        "GPU_FAULT_THERMAL_VIOLATION_DELTA_WARNING_US",
    ):
        assert name in script
    # Retired: no grade counts thermal samples any more, so the legacy path
    # neither validates nor sets them (env_validation.RETIRED_ENVIRONMENT).
    for retired in (
        "GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES",
        "GPU_FAULT_DCGM_COMPOSITE_CONSECUTIVE_SAMPLES",
    ):
        assert retired not in script, f"{retired} is retired and must not be wired"
