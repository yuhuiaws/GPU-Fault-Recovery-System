from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs/部署和运维手册.md"


def manual() -> str:
    return MANUAL.read_text(encoding="utf-8")


def test_deployment_manual_markdown_fences_are_balanced() -> None:
    fences = [line for line in manual().splitlines() if line.startswith("```")]
    assert len(fences) % 2 == 0


def test_regional_greenfield_orders_required_objects() -> None:
    text = manual()
    secret = text.index("--from-literal=processor-replay-secret=")
    metadata = text.index("create configmap gpu-fault-release-metadata")
    schema = text.index("job/gpu-fault-postgres-schema-ensure")
    role_split = text.index(
        "deploy/control-plane/tools/apply-control-plane-role-split.sh"
    )

    assert secret < metadata < schema < role_split
    assert "--from-literal=execution-token=" in text
    assert "--from-literal=node-action-secret=" in text
    node_keys = text.index(
        "deploy/node/provision-node-action-keys.sh", text.index("##### CPU-4a.")
    )
    assert (
        "GPU_FAULT_CONTROL_PLANE_KUBECONFIG"
        in text[text.index("##### CPU-4a.") : text.index("##### CPU-4b.")]
    )
    assert (
        "GPU_FAULT_CONTROL_PLANE_NAMESPACE"
        in text[text.index("##### CPU-4a.") : text.index("##### CPU-4b.")]
    )
    assert node_keys < role_split
    assert "required-regional-executor-protocol-version=" in text
    assert "CURRENT_AGENT_PROTOCOL_VERSION" in text
    assert "CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION" in text


def test_regional_region_is_operator_selected_and_fail_closed() -> None:
    text = manual()
    env_template = (
        ROOT / "deploy/control-plane/regional/regional-env.example.sh"
    ).read_text(encoding="utf-8")
    release_template = json.loads(
        (
            ROOT / "deploy/control-plane/regional/regional-release.example.json"
        ).read_text(encoding="utf-8")
    )

    assert "不能从当前 shell、默认 kubectl" in text
    assert "regional_validate_region" in text
    assert "regional_assert_eks_arn_region CPU_EKS_ARN" in text
    assert "regional_assert_eks_arn_region GPU_EKS_ARN" in text
    assert 'test "${GPU_NODE_RECOVERY}" = "None"' in text
    assert "AWS_REGION='REPLACE_WITH_AWS_REGION'" in env_template
    assert release_template["aws_region"] == "REPLACE_WITH_AWS_REGION"
    assert release_template["cpu_eks_arn"] == "REPLACE_WITH_CPU_EKS_ARN"


def test_manual_forbids_multiple_global_load_balancer_controllers() -> None:
    text = manual()
    section = text.split("#### CPU-3. 安装 AWS Load Balancer Controller", 1)[1].split(
        "#### CPU-4.", 1
    )[0]

    assert "managed_by=EKS" in section
    assert "必须复用它" in section
    assert "不得在同一集群运行两个" in section
    assert "平台管理的 CRD" in section


def test_manual_uses_current_regional_rollout_contracts() -> None:
    text = manual()

    assert (
        text.count("deploy/control-plane/tools/restart-regional-control-plane.sh") >= 4
    )
    assert "while [ $i -lt 300 ]" in text
    assert "--timeout=420s" in text
    assert "gpu-fault-telemetry-spool-worker" in text
    assert "gpu-fault-control-worker" in text
    assert "gpu-fault-api-ha" in text
    assert (
        "kubectl -n gpu-fault-system scale "
        "deployment/gpu-fault-control-plane --replicas=0" not in text
    )


def test_manual_uses_current_build_and_capacity_values() -> None:
    text = manual()

    assert "make test-postgres" in text
    assert "docker.io/library/postgres:16" in text
    assert "public.ecr.aws/docker/library/postgres:16" not in text
    assert "不能在验证与上传之间再次 build" in text
    assert "dist/current-release.json" in text
    assert "scripts/release-artifact-path.py" in text
    assert "tests/test_postgres_store.py" not in text
    assert "export GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS=5" in text
    assert "steady ceiling                    1080" in text
    assert "上界为 `1240`" in text
    assert "合计 1728" not in text
    assert "当前 1710" not in text


def test_manual_documents_both_training_submission_paths() -> None:
    text = manual()

    assert "gpu-training-submit" in text
    assert "gpu-fault-workload-annotate" in text
    assert "apply --dry-run=server" in text
    assert "store.get_profile" in text
    assert 'profile.cluster_id == os.environ["CLUSTER"]' in text
    assert "hyperpod-control-plane-recovery-v1" not in text


def test_manual_support_files_exist() -> None:
    for relative in (
        "deploy/control-plane/regional/regional-env.example.sh",
        "deploy/control-plane/regional/prepare-clean-redeploy.sh",
        "deploy/control-plane/tools/update-deployment-contracts.sh",
        "deploy/control-plane/tools/sync_installed_resource_registry.py",
        "deploy/control-plane/tools/collect_installed_resource_registry.py",
        "deploy/control-plane/tools/cleanup_state.py",
        "deploy/control-plane/tools/restart-regional-control-plane.sh",
        "deploy/control-plane/tools/apply-control-plane-role-split.sh",
        "deploy/control-plane/tools/verify_control_plane_role_split.py",
        "deploy/control-plane/tools/verify-control-plane-role-split.sh",
        "deploy/README.md",
        "scripts/verify-regional-alerting.py",
        "scripts/generate-cleanup-inventory.py",
    ):
        assert (ROOT / relative).is_file()


def test_manual_documents_safe_regional_reset_and_retirement() -> None:
    text = manual()
    reset = text.split("### 0.3 保留 EKS 集群并重新安装", 1)[1].split(
        "### 0.4 永久退役方案", 1
    )[0]
    retirement = text.split("#### REG-13. 区域方案永久退役", 1)[1].split(
        "#### REG-14.", 1
    )[0]

    dry_run = reset.index("deploy/control-plane/regional/prepare-clean-redeploy.sh")
    execute = reset.index("--execute", dry_run)
    assert dry_run < execute, "manual must show the dry run before execution"
    assert "--mode reset" in reset
    assert "--node-mode uninstall" in reset
    assert "--confirm-reset RESET_GPU_FAULT_INSTALLATION" in reset
    assert "CPU EKS 和每个 GPU EKS 必须仍存在" in reset
    assert "delete-db-cluster" in reset
    assert "delete-security-group" in reset
    assert "delete-certificate" in reset
    assert "delete-secret" in reset
    assert "uninstall aws-load-balancer-controller" in reset
    assert "--mode clean" in retirement
    assert "CPU_INFRA_DESTROY_COMMAND" in retirement
    assert "不得调用 GPU 集群的 `delete-cluster`" in retirement

    inventory = (
        ROOT / "deploy/control-plane/regional/cleanup-inventory.json"
    ).read_text(encoding="utf-8")
    for component in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
        "gpu-fault-cluster-executor",
        "gpu-fault-completion-watcher",
        "gpu-fault-kubernetes-node-resource-collector",
        "gpu-fault-node-installer-reconciler",
        "gpu-fault-metrics-collector",
        "gpu-fault-dcgm-exporter",
    ):
        assert component in inventory, f"regional reset omits {component}"
    assert (ROOT / "deploy/systemd/gpu-fault-gpu-persistence.service").is_file()


def test_manual_separates_manifest_developer_and_operator_workflows() -> None:
    text = manual()
    section = text.split("### 生产 Manifest 修改与自动生成", 1)[1].split(
        "## 客户端身份与迁移开关硬约束", 1
    )[0]

    assert "本节只面向修改本方案源代码的开发者" in section
    assert "管理员/客户边界：不编写方案 Manifest" in section
    assert "开发者完整变更步骤" in section
    assert "rollout-regional-release.sh" in section
    assert "bootstrap --config" in section
    assert "upgrade --config" in section
    assert "deployment-contracts-update" in section
    assert "deployment-contracts-check" in section
    assert "make PYTHON=.venv/bin/python check" in section
    assert "gpu-fault-installed-resources" in section
    assert "客户自己的训练任务 YAML 是明确例外" in section
    assert "直接编辑 deploy/control-plane/regional/generated/" in section
    assert "同一个 PR" in section


def test_manual_requires_external_state_before_aurora_deletion() -> None:
    text = manual()

    assert "READY_TO_DELETE_AURORA" in text
    assert "AURORA_DELETED" in text
    assert "cleanup_state.py verify" in text
    assert "content_sha256" in text
    assert "Aurora 只承载正常运行期清单" in text


def test_manual_separates_solution_and_training_images() -> None:
    text = manual()

    for name in (
        "GPU_FAULT_RUNTIME_IMAGE",
        "GPU_FAULT_NODE_INSTALLER_IMAGE",
        "GPU_FAULT_DCGM_EXPORTER_IMAGE",
        "GPU_FAULT_ADOT_IMAGE",
    ):
        assert name in text
    assert "客户自定义训练镜像" in text
    assert "gpu-fault-training-image-prepull" in text
    assert "imagePullSecrets" in text


def test_operations_manual_rename_has_no_stale_entrypoint() -> None:
    assert MANUAL.is_file()
    stale_name = "部署" + "手册.md"
    assert not (ROOT / "docs" / stale_name).exists()
    assert "docs/部署和运维手册.md" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_manual_exposes_only_regional_production_topology() -> None:
    text = manual()
    design = (ROOT / "docs/概要设计.md").read_text(encoding="utf-8")

    assert "本手册只交付一条生产部署架构" in text
    assert "唯一生产目标" in text
    assert "通用 Kubernetes 部署当前没有实现" in text
    assert "## 6. 通用 Kubernetes 部署" not in text
    assert "/tmp/collectors.yaml" not in text
    assert "四种部署方式" not in text
    assert "## 附录 A. 旧版 HyperPod 单集群（非生产）" in text
    assert "区域数据面节点 systemd 安装详解" in text
    assert "当前只交付一种生产部署形态" in design
    assert "通用 Kubernetes 三副本（未实现，不交付）" in design


def test_legacy_kubernetes_examples_cannot_be_mistaken_for_deployment() -> None:
    design = (ROOT / "docs/概要设计.md").read_text(encoding="utf-8")
    detail = (ROOT / "docs/详细设计.md").read_text(encoding="utf-8")
    operations = manual()

    assert not (ROOT / "examples/legacy").exists()
    assert "历史清单已从公开树移除" in design
    assert "历史设计参考，不可部署" in design
    assert "export GPU_FAULT_ALLOW_HYPERPOD_MUTATION=true" not in detail
    assert "unset GPU_FAULT_ALLOW_HYPERPOD_MUTATION" in operations
    assert "`GPU_FAULT_ALLOW_HYPERPOD_REPLACE`" in detail
    assert "恒为 `false`" in detail


def test_make_check_rebuilds_wheel_before_full_pytest() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    check = makefile.split("check:\n", 1)[1].split("\narchitecture-check:", 1)[0]

    assert check.index("$(MAKE) artifact-check") < check.index("$(PYTHON) -m pytest")


def test_release_preflight_is_required_and_load_bearing() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("release-preflight:\n", 1)[1].split("\n# 本地保留策略", 1)[
        0
    ]
    section = manual().split("## 4. 构建与测试", 1)[1].split("## 5.", 1)[0]

    assert "$(MAKE) artifacts-local-safety-check" in target
    assert "$(MAKE) artifact-check" in target
    assert "make release-preflight" in section
    assert "不得再次 build" in section


def test_manual_only_builds_through_artifact_check() -> None:
    text = manual()
    start = text.index("## 4. 构建与测试")
    end = text.index("## 5. 区域分离生产部署")
    build_chapter = text[start:end]

    assert "make check" in build_chapter
    assert "dist/current-release.json" in build_chapter
    assert not list(re.finditer(r"python(?:3(?:\.12)?)? -m build(?: --wheel)?", text))

    refresh = text.split("##### 2b. 部署 Aurora 凭据刷新 CronJob", 1)[1].split(
        "##### CPU-2R.", 1
    )[0]
    assert "python3.12 -m build --wheel" not in refresh
    assert "zipfile.ZipFile" in refresh
    assert "CONFIG_MAP_SHA256" in refresh


def test_regional_acceptance_checks_live_amp_alerting() -> None:
    text = manual()
    section = text.split("#### REG-8. 区域部署验收清单", 1)[1].split("#### REG-8.1", 1)[
        0
    ]

    assert "get prometheusrule" not in section.lower()
    assert "describe-rule-groups-namespace" in section
    assert "describe-alert-manager-definition" in section
    assert "scripts/verify-regional-alerting.py" in section
    assert "UNREACHABLE_ALERT_DEFECTS= 0" in section
    assert '"http://127.0.0.1:8080/v1/fleet/readiness"' in section
    assert 'method="POST"' in section
    assert "NODE_IDS_JSON" in section
    assert "/v1/fleet/readiness?cluster_id=" not in section


def test_regional_alerting_verifier_accepts_repository_contract(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    manager = tmp_path / "alertmanager.yaml"
    rules.write_text(
        (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manager.write_text(
        (ROOT / "deploy/observability/amp-alertmanager.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "REPLACE_WITH_SNS_TOPIC_ARN", "arn:aws:sns:us-west-2:123456789012:gpu-fault"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify-regional-alerting.py",
            "--live-rules",
            str(rules),
            "--live-alertmanager",
            str(manager),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNREACHABLE_ALERT_DEFECTS= 0" in result.stdout
    assert "gpu-fault-remote-command" in result.stdout
    assert "ALERTMANAGER_SNS_RECEIVERS= gpu-fault-sns" in (result.stdout)


def test_incident_purge_stops_and_restores_every_writer() -> None:
    text = manual()
    section = text.split("### 8.4 手工删除无 TTL 的 incident 审计记录", 1)[1].split(
        "## 9. 数据迁移", 1
    )[0]

    for deployment in (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
        "gpu-fault-control-plane",
    ):
        assert deployment in section
    assert "PURGE_WRITER_STATE" in section
    assert "trap restore_incident_purge_writers EXIT" in section
    assert "restore_order=(" in section
    assert "ORIGINAL_REPLICAS" not in section


def test_manual_keeps_provider_replace_fail_closed() -> None:
    text = manual()
    stage = text.split("### 阶段 D：HyperPod reboot（provider replace 恒禁用）", 1)[
        1
    ].split("## 11.", 1)[0]

    assert "### 2.2 区域生产方案硬约束" in text
    assert "GPU_FAULT_ALLOW_HYPERPOD_REBOOT=true" in stage
    assert "GPU_FAULT_ALLOW_HYPERPOD_REPLACE=false" in stage
    assert "unset GPU_FAULT_ALLOW_HYPERPOD_MUTATION" in stage
    assert "只能在独立 reboot 开关缺省时回退给 reboot" in text
    assert "设置为 true 时进程拒绝启动" in text
    assert "恒 false（§2.2 方案不变量）" in text


def test_manual_parameter_tables_have_headers() -> None:
    text = manual()

    assert ("| 环境变量 | 默认值 |\n|---|---|\n| `GPU_FAULT_LOG_STATE_PATH`") in text
    assert (
        "| 环境变量 | 默认值 | 说明 |\n|---|---|---|\n| `GPU_FAULT_HMA_QUEUE_URL`"
    ) in text


def test_manual_uses_public_numbering_and_keeps_legacy_anchors() -> None:
    text = manual()

    for heading in (
        "## 5. 区域分离生产部署（唯一生产形态）",
        "### 5.1 区域 CPU 控制面绿地部署",
        "### 5.2 从集群内控制面迁移到区域控制面",
        "### 5.3 区域级 GPU 集群静态注册",
        "### 5.4 区域数据面训练任务提交与验收",
        "### 5.5 区域数据面节点 systemd 安装详解",
        "## 6. 控制面参数",
        "## 7. Collector 和 Watcher 参数",
        "## 8. 邮件通知",
        "## 9. 数据迁移",
        "## 10. 分阶段启用破坏性能力",
        "## 11. 示例清单使用警告",
        "## 附录 A. 旧版 HyperPod 单集群（非生产）",
        "### A.6 过渡单集群验收清单（非生产）",
        "### A.7 过渡单集群故障排查（非生产）",
    ):
        assert heading in text

    for anchor in (
        '<a id="5a-区域分离生产部署唯一生产形态"></a>',
        '<a id="5a1-区域-cpu-控制面绿地部署"></a>',
        '<a id="5a2-从集群内控制面迁移到区域控制面"></a>',
        '<a id="5a3-区域级-gpu-集群静态注册"></a>',
        '<a id="5a4-区域数据面训练任务提交与验收"></a>',
        '<a id="5a5-区域数据面节点-systemd-安装详解"></a>',
        '<a id="5-旧版-hyperpod-单集群过渡部署非生产目标"></a>',
        '<a id="13-过渡单集群验收清单非生产"></a>',
        '<a id="14-过渡单集群故障排查非生产"></a>',
        '<a id="15-示例清单使用警告"></a>',
        '<a id="8-控制面参数"></a>',
        '<a id="81-启动与存储"></a>',
        '<a id="82-adapter-与-owner"></a>',
        '<a id="83-fleet-compatibility"></a>',
        '<a id="84-诊断训练与证据"></a>',
        '<a id="85-gpu-阈值"></a>',
        '<a id="86-hyperpod"></a>',
        '<a id="9-collector-和-watcher-参数"></a>',
        '<a id="91-collector-公共参数"></a>',
        '<a id="92-completion-watcher"></a>',
        '<a id="93-training-progress"></a>',
        '<a id="94-collector-上下文与-hma"></a>',
        '<a id="95-node-agent-完整参数"></a>',
        '<a id="96-fleet-cli-transport-变量"></a>',
        '<a id="10-邮件通知"></a>',
        '<a id="101-gpu-数量变化审批"></a>',
        '<a id="102-spare-不足时管理员手动降级恢复"></a>',
        '<a id="1021-适用范围和系统行为"></a>',
        '<a id="1022-第一步确认自动流程已经停止"></a>',
        '<a id="1023-第二步保持故障节点隔离"></a>',
        '<a id="1024-第三步复核并人工预留健康-spare"></a>',
        '<a id="1025-第四步修改训练-yaml-和训练参数"></a>',
        '<a id="1026-第五步生成并审核新的-attempt"></a>',
        '<a id="1027-第六步验证缩容-attempt"></a>',
        '<a id="1028-失败回滚和-spare-释放"></a>',
        '<a id="103-checkmechanicals-后由管理员提交明确处置"></a>',
        '<a id="104-手工删除无-ttl-的-incident-审计记录"></a>',
        '<a id="11-数据迁移"></a>',
        '<a id="12-分阶段启用破坏性能力"></a>',
        '<a id="13-示例清单使用警告"></a>',
    ):
        assert anchor in text

    assert "## 5A. 区域分离生产部署" not in text
    assert "## 5. 旧版 HyperPod 单集群过渡部署" not in text
    assert text.index("## 5. 区域分离生产部署") < text.index("## 6. 控制面参数")
    assert text.index("## 11. 示例清单使用警告") < text.index(
        "## 附录 A. 旧版 HyperPod 单集群"
    )
