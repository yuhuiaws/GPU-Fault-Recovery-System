from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def container(deployment: dict) -> dict:
    return deployment["spec"]["template"]["spec"]["containers"][0]


def environment(deployment: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for source in container(deployment).get("envFrom", []):
        reference = source.get("configMapRef")
        if not reference:
            continue
        config_map = load(GENERATED / f"{reference['name']}.yaml")
        result.update(
            {
                name: {"name": name, "value": value, "configMapRef": reference["name"]}
                for name, value in config_map.get("data", {}).items()
            }
        )
    result.update({item["name"]: item for item in container(deployment).get("env", [])})
    return result


def test_control_plane_shutdown_budget_has_real_margin() -> None:
    for filename in (
        "gpu-fault-api-ha-ingress.yaml",
        "gpu-fault-control-worker.yaml",
        "gpu-fault-telemetry-spool-worker.yaml",
    ):
        deployment = load(GENERATED / filename)
        pod_spec = deployment["spec"]["template"]["spec"]
        api = container(deployment)
        pre_stop = api["lifecycle"]["preStop"]["exec"]["command"]
        match = re.fullmatch(r"sleep (\d+)", pre_stop[-1])
        assert match is not None
        graceful = re.search(r"--timeout-graceful-shutdown (\d+)", api["args"][0])
        assert graceful is not None
        env = environment(deployment)
        lifespan = int(env["GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS"]["value"])
        used = int(match.group(1)) + int(graceful.group(1)) + lifespan
        assert pod_spec["terminationGracePeriodSeconds"] >= used + 20
        if env["GPU_FAULT_SERVICE_ROLE"]["value"] != "ingress":
            required = (
                int(env["GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS"]["value"])
                + int(env["GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS"]["value"])
                + 5
            )
            assert lifespan >= required


def test_regional_alerting_and_agent_pins_fail_closed() -> None:
    for filename in ("gpu-fault-api-ha-ingress.yaml", "gpu-fault-control-worker.yaml"):
        deployment = load(GENERATED / filename)
        env = environment(deployment)
        assert env["GPU_FAULT_ALLOW_EMAIL"]["value"] == "true"
        assert env["GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED"]["value"] == "true"
        assert env["GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"]["value"] == "false"
        assert env["GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "required-agent-artifact-sha256",
        }
        assert env["GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "compatible-agent-artifact-sha256s",
            "optional": True,
        }
        assert env["GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "required-agent-protocol-version",
        }
        assert env["GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "compatible-agent-protocol-versions",
            "optional": True,
        }
        assert env["GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION"][
            "valueFrom"
        ]["configMapKeyRef"] == {
            "name": "gpu-fault-release-metadata",
            "key": "required-regional-executor-protocol-version",
        }
        assert env["GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS"][
            "valueFrom"
        ]["configMapKeyRef"] == {
            "name": "gpu-fault-release-metadata",
            "key": "compatible-regional-executor-protocol-versions",
            "optional": True,
        }
        assert env["GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "required-agent-config-digest",
        }
        assert env["GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "compatible-agent-config-digests",
            "optional": True,
        }
        assert env["GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION"]["valueFrom"][
            "configMapKeyRef"
        ] == {
            "name": "gpu-fault-release-metadata",
            "key": "required-node-action-key-version",
        }


def test_processor_replay_secret_is_separate_and_liveness_is_http() -> None:
    ingress = load(GENERATED / "gpu-fault-api-ha-ingress.yaml")
    env = environment(ingress)
    execution = env["GPU_FAULT_EXECUTION_TOKEN"]["valueFrom"]["secretKeyRef"]
    replay = env["GPU_FAULT_PROCESSOR_REPLAY_SECRET"]["valueFrom"]["secretKeyRef"]

    assert execution["key"] == "execution-token"
    assert replay["key"] == "processor-replay-secret"
    assert execution["key"] != replay["key"]
    assert container(ingress)["livenessProbe"]["httpGet"] == {
        "path": "/healthz",
        "port": 8080,
    }


def test_control_plane_roles_and_adot_have_disruption_protection() -> None:
    ingress = load(GENERATED / "gpu-fault-api-ha-ingress.yaml")
    ingress_pdb = load(GENERATED / "gpu-fault-api-ha-pdb.yaml")
    worker = load(GENERATED / "gpu-fault-control-worker.yaml")
    worker_pdb = load(GENERATED / "gpu-fault-control-worker-pdb.yaml")
    spool = load(GENERATED / "gpu-fault-telemetry-spool-worker.yaml")
    spool_pdb = load(GENERATED / "gpu-fault-telemetry-spool-worker-pdb.yaml")
    adot_documents = list(
        yaml.safe_load_all(
            (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    adot = next(item for item in adot_documents if item.get("kind") == "Deployment")
    adot_pdb = next(
        item for item in adot_documents if item.get("kind") == "PodDisruptionBudget"
    )

    assert ingress["spec"]["replicas"] == 3, "ingress HA requires three replicas"
    assert ingress_pdb["spec"]["minAvailable"] == 2, "ingress PDB lost quorum"
    assert ingress_pdb["spec"]["selector"]["matchLabels"] == {
        "app": "gpu-fault-api-ha"
    }, "ingress PDB selects the wrong pods"
    assert worker["spec"]["replicas"] == 6
    assert worker_pdb["spec"]["maxUnavailable"] == 1
    assert worker_pdb["spec"]["selector"]["matchLabels"] == {
        "app": "gpu-fault-control-worker"
    }
    assert spool["spec"]["replicas"] == 0
    assert container(spool)["resources"] == {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "4", "memory": "4Gi"},
    }
    spool_env = environment(spool)
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES"]["value"] == str(
        4 * 1024 * 1024
    )
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES"][
        "value"
    ] == str(8 * 1024 * 1024)
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES"]["value"] == str(
        64 * 1024 * 1024
    )
    assert spool_env["GPU_FAULT_TELEMETRY_SPOOL_WORKERS"]["value"] == "8"
    assert spool_env["GPU_FAULT_POSTGRES_POOL_MAX_SIZE"]["value"] == "12"
    assert spool_pdb["spec"]["maxUnavailable"] == 1
    assert spool_pdb["spec"]["selector"]["matchLabels"] == {
        "app": "gpu-fault-telemetry-spool-worker"
    }
    assert adot["spec"]["replicas"] >= 1
    assert adot_pdb["spec"]["minAvailable"] == 1
    collector_config = next(
        item
        for item in adot_documents
        if item.get("kind") == "ConfigMap"
        and item["metadata"]["name"] == "gpu-fault-adot"
    )["data"]["collector.yaml"]
    assert "gpu_fault_telemetry_spool_.+" in collector_config
    installer = (ROOT / "deploy/observability/install-amp-monitoring.sh").read_text(
        encoding="utf-8"
    )
    assert "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION" in installer
    assert "has no confirmed subscription" in installer
    assert "KUBECONFIG_EKS_ARN" in installer
    assert installer.index("aws eks describe-cluster") < installer.index(
        "aws iam create-role"
    )


def test_ingress_has_one_nonblocking_spread_constraint() -> None:
    ingress = load(GENERATED / "gpu-fault-api-ha-ingress.yaml")
    constraints = ingress["spec"]["template"]["spec"]["topologySpreadConstraints"]

    assert len(constraints) == 1
    assert constraints[0]["topologyKey"] == ("kubernetes.io/hostname")
    assert constraints[0]["whenUnsatisfiable"] == "DoNotSchedule"
    assert constraints[0]["matchLabelKeys"] == ["pod-template-hash"]
    env = environment(ingress)
    assert env["GPU_FAULT_TELEMETRY_SPOOL"]["value"] == "false"
    assert "GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS" not in env
    assert env["GPU_FAULT_POSTGRES_POOL_MIN_SIZE"]["value"] == "2"
    assert env["GPU_FAULT_POSTGRES_POOL_MAX_SIZE"]["value"] == "40"
    assert env["GPU_FAULT_STORE_IO_WORKERS"]["value"] == "28"
    assert env["GPU_FAULT_EVIDENCE_STORE_IO_WORKERS"]["value"] == "4"
    assert env["GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS"]["value"] == "8"
    assert env["GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS"]["value"] == "8"
    assert env["GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS"]["value"] == "8"
    assert env["GPU_FAULT_FAULT_STORE_IO_WORKERS"]["value"] == "8"


def test_processor_counter_shard_jobs_are_bounded_and_explicit() -> None:
    cases = {
        "postgres-counter-shards-finalize-job.yaml": (
            "--finalize-processor-counter-shards"
        ),
        "postgres-counter-shards-rollback-job.yaml": (
            "--restore-legacy-processor-counters"
        ),
    }
    for filename, expected_flag in cases.items():
        job = load(ROOT / "deploy/migrations" / filename)
        assert job["kind"] == "Job"
        assert job["spec"]["backoffLimit"] == 1
        assert job["spec"]["activeDeadlineSeconds"] == 720
        pod = job["spec"]["template"]["spec"]
        assert pod["restartPolicy"] == "Never"
        migrate = pod["containers"][0]
        assert expected_flag in migrate["args"][0]
        assert migrate["env"] == [
            {
                "name": "GPU_FAULT_STORE_URL",
                "valueFrom": {
                    "secretKeyRef": {"name": "gpu-fault-aurora", "key": "postgres-url"}
                },
            }
        ]
        assert migrate["securityContext"] == {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }
        assert pod["volumes"][0]["configMap"]["name"] == (
            "REPLACE_WITH_WHEEL_CONFIGMAP"
        )


def test_role_split_deploy_waits_for_spool_before_ingress() -> None:
    standalone = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")
    spool_apply = standalone.index("apply_manifest gpu-fault-telemetry-spool-worker\n")
    spool_ready = standalone.index(
        "rollout status deployment/gpu-fault-telemetry-spool-worker"
    )
    ingress_apply = standalone.index("apply_manifest gpu-fault-api-ha-ingress")
    assert spool_apply < spool_ready < ingress_apply

    deploy = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")
    spool_ready = deploy.index(
        "rollout status deployment/gpu-fault-telemetry-spool-worker"
    )
    ingress_apply = deploy.index("gpu-fault-api-ha-ingress; do")
    assert spool_ready < ingress_apply


def test_deployment_scripts_support_site_image_mirrors() -> None:
    role_split = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")
    hyperpod = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")
    reconciler = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text(
        encoding="utf-8"
    )
    adot = (ROOT / "deploy/observability/install-amp-monitoring.sh").read_text(
        encoding="utf-8"
    )

    assert "GPU_FAULT_RUNTIME_IMAGE" in role_split
    assert "gpu-fault.io/runtime-image" in role_split
    assert "GPU_FAULT_RUNTIME_IMAGE" in reconciler
    assert "GPU_FAULT_NODE_INSTALLER_IMAGE" in installer
    assert "GPU_FAULT_RUNTIME_IMAGE" in hyperpod
    assert "GPU_FAULT_NODE_INSTALLER_IMAGE" in hyperpod
    assert "GPU_FAULT_DCGM_EXPORTER_IMAGE" in hyperpod
    assert "GPU_FAULT_ADOT_IMAGE" in adot


def test_regional_restart_covers_every_running_role_in_order() -> None:
    script_path = ROOT / "deploy/control-plane/tools/restart-regional-control-plane.sh"
    script = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    spool = script.index("gpu-fault-telemetry-spool-worker")
    worker = script.index("gpu-fault-control-worker")
    ingress = script.index("gpu-fault-api-ha")
    assert spool < worker < ingress
    assert "replicas == 0" in script
    assert "verify-control-plane-role-split.sh" in script
    assert "set KUBECONFIG" in script


def test_regional_environment_template_is_sourceable() -> None:
    template = ROOT / "deploy/control-plane/regional/regional-env.example.sh"
    script = template.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(template)], check=True)
    assert "regional_require()" in script
    assert "AWS_REGION='REPLACE_WITH_AWS_REGION'" in script
    assert "CPU_KUBECONFIG" in script
    assert "CPU_EKS_ARN" in script
    assert "GPU_EKS_CONTEXT" in script
    assert "regional_validate_region()" in script
    assert "regional_assert_eks_arn_region()" in script
    assert "HYPERPOD_CONFIRM_CLUSTER_NAME" in script
    assert "CONTROL_PLANE_URL" in script


def test_regional_production_assets_have_no_implicit_region() -> None:
    paths = (
        "deploy/control-plane/regional/regional-env.example.sh",
        "deploy/control-plane/regional/regional-release.example.json",
        "deploy/control-plane/regional/rollout_regional_release.py",
        "deploy/control-plane/regional/regional-control-plane-patch.yaml",
        "deploy/control-plane/regional/regional-control-plane-nlb.yaml",
        "deploy/control-plane/tools/apply-control-plane-role-split.sh",
        "deploy/dataplane/cluster-action-executor.yaml",
        "deploy/observability/adot-control-plane.yaml",
        "deploy/observability/amp-alertmanager.yaml",
        "deploy/observability/install-amp-monitoring.sh",
    )

    for relative in paths:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "us-west-2" not in text, relative


def test_role_split_apply_supports_greenfield_namespace() -> None:
    script = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")

    assert "s/namespace: gpu-fault-system/namespace: ${NAMESPACE}/g" in script
    assert "GPU_FAULT_ROLE_SPLIT_GENERATED_DIR:-" in script
    assert "GPU_FAULT_AWS_REGION" in script
    assert "REPLACE_WITH_AWS_REGION" in script
    assert "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION" in script
    assert "REPLACE_WITH_RUNTIME_PROFILE_VERSION" in script
    assert "gpu-fault.io/artifact-sha256: ${WHEEL_SHA256}" in script, (
        "apply must not toggle the pod template back to the generated source digest"
    )
    assert "gpu-fault-release-metadata is missing" in script
    assert "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST" in script
    assert "GPU_FAULT_FINALIZE_AGENT_PIN" in script
    assert "CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256" in script
    assert "compatible-agent-artifact-sha256s" in script
    assert "compatible-agent-protocol-versions" in script
    assert "compatible-agent-config-digests" in script
    assert "GPU_FAULT_FINALIZE_DATA_PLANE_PIN" in script
    assert "compatible-regional-executor-protocol-versions" in script
    assert "GPU_FAULT_LEGACY_COMPONENT_PINS" in script
    assert "remove_legacy_notification_env" in script
    assert "GPU_FAULT_ALLOW_EMAIL-" in script
    assert "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL-" in script
    assert "filter_legacy_release_env.py" in script
    assert (
        'CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST="'
        "${CURRENT_REQUIRED_AGENT_COMPATIBILITY_DIGEST:-"
        '${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}}"' in script
    )
    assert 'PIN_METADATA_CHANGED="true"' in script
    assert 'RELOAD_RELEASE_METADATA="${PIN_METADATA_CHANGED}"' in script
    assert "PIN_FINALIZATION" not in script
    assert "rollout restart" in script
    assert (
        'name="$(basename "${config}" .yaml)"\n    apply_manifest "${name}"' in script
    )
    ingress_exists = script.index(
        "get deployment \\\n    gpu-fault-api-ha >/dev/null 2>&1"
    )
    stale_env_cleanup = script.index("GPU_FAULT_PROCESSOR_WORKERS-")
    ingress_apply = script.index("apply_manifest gpu-fault-api-ha-ingress")
    assert ingress_exists < stale_env_cleanup < ingress_apply


def test_hyperpod_deploy_uses_two_phase_agent_pin_migration() -> None:
    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")

    compatibility_window = script.index(
        "--from-literal=compatible-agent-artifact-sha256s="
    )
    install = script.index(
        'step "Install collectors and signed Agent on all HyperPod nodes"'
    )
    finalize = script.index('step "Finalize the Agent compatibility window"')
    resume = script.index(
        'step "Resume processor workers after the agent pin migration"'
    )
    assert compatibility_window < install < finalize < resume
    assert "compatibility window remains open" in script
    assert (
        '--from-literal=required-agent-artifact-sha256="'
        '${STAGED_AGENT_ARTIFACT_SHA256}"' in script
    )
    assert 'COMPATIBLE_AGENT_ARTIFACT_SHA256S="${NODE_WHEEL_SHA256}"' in script
    assert (
        '--from-literal=required-agent-compatibility-digest="'
        '${STAGED_AGENT_COMPATIBILITY_DIGEST}"' in script
    )
    assert (
        '--from-literal=required-regional-executor-artifact-sha256="'
        '${STAGED_EXECUTOR_ARTIFACT_SHA256}"' in script
    )
    assert 'item["agent_protocol_version"] == 2' not in script
    assert 'item["agent_protocol_version"] == expected_protocol' in script


def test_role_split_stages_candidate_without_replacing_stable_pin() -> None:
    script = (
        ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    ).read_text(encoding="utf-8")

    assert (
        'REQUIRED_AGENT_ARTIFACT_SHA256="'
        '${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}"' in script
    )
    assert (
        '"${COMPATIBLE_AGENT_ARTIFACT_SHA256S}" \\\n'
        '                    "${TARGET_AGENT_ARTIFACT_SHA256}"' in script
    )
    assert (
        'REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="'
        '${CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}"' in script
    )
    assert 'FAST_ROLLOUT_TIMEOUT="5m"' in script
    assert '--timeout="${FAST_ROLLOUT_TIMEOUT}"' in script


def test_aurora_rotation_restarts_every_database_consumer() -> None:
    documents = list(
        yaml.safe_load_all(
            (
                ROOT / "deploy/control-plane/regional/aurora-credential-refresh.yaml"
            ).read_text(encoding="utf-8")
        )
    )
    cronjob = next(item for item in documents if item.get("kind") == "CronJob")
    role = next(item for item in documents if item.get("kind") == "Role")
    env = {
        item["name"]: item.get("value")
        for item in cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
    }
    targets = {
        item.strip() for item in env["GPU_FAULT_AURORA_RESTART_DEPLOYMENTS"].split(",")
    }

    assert targets == {
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    }
    deployment_rule = next(
        item
        for item in role["rules"]
        if item["apiGroups"] == ["apps"] and item["resources"] == ["deployments"]
    )
    assert set(deployment_rule["resourceNames"]) == targets
    assert set(deployment_rule["verbs"]) == {"get", "patch"}


def test_cluster_executor_dependency_install_has_bounded_retry() -> None:
    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    command = deployment["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "until python -m pip install" in command
    assert "--retries 3 --timeout 15" in command
    assert '[ "${attempt}" -ge 4 ]' in command
    assert command.index("touch /tmp/executor-ready") > command.index(
        "until python -m pip install"
    )
    result = subprocess.run(
        ["/bin/sh", "-n"], input=command, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cluster_executor_private_ca_does_not_replace_aws_trust() -> None:
    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    env = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert env["GPU_FAULT_CONTROL_PLANE_CA_FILE"] == ("/etc/gpu-fault/tls/ca.crt")
    assert "SSL_CERT_FILE" not in env
    assert "REQUESTS_CA_BUNDLE" not in env


def test_regional_executor_uses_independent_hyperpod_switches() -> None:
    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    env = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert "GPU_FAULT_ALLOW_HYPERPOD_MUTATION" not in env
    assert env["GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"] == "true"
    assert env["GPU_FAULT_ALLOW_HYPERPOD_REBOOT"] == "true"
    assert env["GPU_FAULT_ALLOW_HYPERPOD_REPLACE"] == "false"


def test_ambiguous_attempt_ownership_has_a_prometheus_alert() -> None:
    document = yaml.safe_load(
        (ROOT / "deploy/control-plane/regional/processor-alerts.yaml").read_text(
            encoding="utf-8"
        )
    )
    rules = {
        rule["alert"]: rule
        for group in document["spec"]["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }
    alert = rules["GpuFaultExclusiveNodeOwnershipInvariantViolation"]

    assert "gpu_fault_ambiguous_attempt_ownership_total" in alert["expr"]
    assert alert["labels"]["severity"] == "critical"


def test_amp_rules_cover_prometheus_rules_and_kept_metrics() -> None:
    amp = yaml.safe_load(
        (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8")
    )
    prometheus_rule = yaml.safe_load(
        (ROOT / "deploy/control-plane/regional/processor-alerts.yaml").read_text(
            encoding="utf-8"
        )
    )
    amp_alerts = {rule["alert"] for group in amp["groups"] for rule in group["rules"]}
    local_alerts = {
        rule["alert"]
        for group in prometheus_rule["spec"]["groups"]
        for rule in group["rules"]
    }
    assert local_alerts <= amp_alerts

    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    collector = next(item for item in documents if item["kind"] == "ConfigMap")
    config = yaml.safe_load(collector["data"]["collector.yaml"])
    scrape = config["receivers"]["prometheus"]["config"]["scrape_configs"][0]
    keep = next(
        item["regex"]
        for item in scrape["metric_relabel_configs"]
        if item["action"] == "keep"
    )
    pattern = re.compile(f"^(?:{keep})$")
    metric_names = {
        name
        for group in amp["groups"]
        for rule in group["rules"]
        for name in re.findall(r"\bgpu_fault_[a-z0-9_]+", rule["expr"])
    }
    assert {name for name in metric_names if not pattern.fullmatch(name)} == set()


def test_regional_executor_never_mounts_fleet_master() -> None:
    documents = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy/dataplane/cluster-action-executor.yaml").read_text(
                encoding="utf-8"
            )
        )
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    executor = pod["containers"][0]
    serialized = yaml.safe_dump(deployment)
    env_names = {item["name"] for item in executor.get("env", [])}

    assert "GPU_FAULT_NODE_ACTION_SECRET" not in env_names
    assert "GPU_FAULT_AGENT_REGISTRATION_SECRET" not in env_names
    assert "GPU_FAULT_NODE_ACTION_KEYS_DIR" in serialized
    node_keys = next(
        item for item in pod["volumes"] if item["name"] == "node-action-keys"
    )
    assert node_keys["secret"]["secretName"] == ("gpu-fault-node-action-keys")
    assert any(
        item["name"] == "node-action-keys"
        and item["mountPath"] == "/etc/gpu-fault/node-action-keys"
        for item in executor["volumeMounts"]
    ), (
        'expected any( item["name"] == "node-action-keys" and item["mountPath"] == "/etc/gpu-fault/node-action-keys" for item in executor["volumeMounts"] ) to be truthy'
    )


def test_boot011_documents_all_required_executor_inputs() -> None:
    document = (ROOT / "docs/区域模式端到端验收测试用例.md").read_text(encoding="utf-8")
    section = document.split("### GF-REGIONAL-BOOT-011", 1)[1].split(
        "### GF-REGIONAL-BOOT-012", 1
    )[0]
    assert "--from-file=hyperpod-confirm-cluster-name=" in section
    assert "EXECUTOR_IRSA_ROLE_ARN" in section
    assert "REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN" in section
    assert "provision-node-action-keys.sh" in section
    assert "GPU_FAULT_HYPERPOD_CLUSTER" in section
    assert "--from-file=node-action-secret=" not in section


def test_greenfield_manual_creates_all_control_plane_secrets() -> None:
    manual = (ROOT / "docs/部署和运维手册.md").read_text(encoding="utf-8")
    block = manual.split("create secret generic gpu-fault-control-plane-active", 1)[
        1
    ].split("fi", 1)[0]

    assert "--from-literal=execution-token=" in block
    assert "--from-literal=processor-replay-secret=" in block
    assert "--from-literal=node-action-secret=" in block


def test_regional_migration_removes_legacy_execution_token() -> None:
    manual = (ROOT / "docs/部署和运维手册.md").read_text(encoding="utf-8")
    section = manual.split("停止并删除原 GPU 集群中的控制面", 1)[1].split(
        "### 5.3 区域级 GPU 集群静态注册", 1
    )[0]

    assert "deployment gpu-fault-api-ha gpu-fault-api-canary" in section
    assert "poddisruptionbudget gpu-fault-api-ha" in section
    assert "service gpu-fault-api-canary" in section
    assert "secret gpu-fault-control-plane-active" in section
    assert 'has("execution-token")' in section


def test_boot_guard_probe_is_small_and_uses_isolated_schema() -> None:
    derive = (ROOT / "scripts/e2e/regional/boot_guard/derive.sh").read_text(
        encoding="utf-8"
    )
    manual = (ROOT / "docs/区域模式端到端验收测试用例.md").read_text(encoding="utf-8")
    section = manual.split("### 4.0 一次性探针的准备", 1)[1].split(
        "### GF-REGIONAL-BOOT-001", 1
    )[0]

    assert 'requests["cpu"] = "250m"' in derive
    assert 'requests["memory"] = "1Gi"' in derive
    assert '"GPU_FAULT_ALLOW_EMAIL": "false"' in derive
    assert '"GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true"' in derive
    assert "gpu-fault-aurora-guardprobe" in derive
    assert "PostgresStore(" in section
    assert 'path="/gpu_fault_guardprobe"' in section


def test_boot008_uses_a_real_cluster_token_route() -> None:
    document = (ROOT / "docs/区域模式端到端验收测试用例.md").read_text(encoding="utf-8")
    section = document.split("### GF-REGIONAL-BOOT-008", 1)[1].split(
        "### GF-REGIONAL-BOOT-009", 1
    )[0]

    assert "/v1/regional/executors/claim" in section
    assert (
        '"http://127.0.0.1:8080/v1/gpu-events/guardprobe-fake-cluster"' not in section
    )


def test_manual_command_order_gate_is_not_constant_green(tmp_path: Path) -> None:
    manual = (ROOT / "docs/部署和运维手册.md").read_text(encoding="utf-8")
    chapter = manual.index("## 5. 区域分离生产部署（唯一生产形态）")
    fence = manual.index("```bash\n", chapter) + len("```bash\n")
    script = ROOT / "scripts/check-manual-command-order.py"

    cases = {
        "object.md": (
            "kubectl set env deployment/gpu-fault-api-ha BOOT017_PROBE=true\n",
            "deployment/gpu-fault-api-ha",
        ),
        "variable.md": (
            'kubectl exec "${NOT_YET_SET_POD}" -- true\n',
            "NOT_YET_SET_POD",
        ),
        "temp-manifest.md": (
            "kubectl apply -f /tmp/not-created.yaml\n",
            "/tmp/not-created.yaml",
        ),
    }
    for filename, (injection, expected) in cases.items():
        path = tmp_path / filename
        path.write_text(manual[:fence] + injection + manual[fence:], encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script), "--manual", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1
        assert expected in result.stdout
