from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs/部署和运维手册.md"
GUIDE = ROOT / "docs/部署和运维手册逐章解读.md"
QUICKSTART = ROOT / "docs/管理员快速部署.md"
PROFILE_APPROVAL = ROOT / "docs/管理员Profile变更审批.md"
DAILY_OPS = ROOT / "docs/管理员日常运维.md"
SAFETY = ROOT / "docs/安全与参数参考.md"
DEVELOPER = ROOT / "docs/开发者部署实现.md"


def test_guide_mentions_every_manual_chapter() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    headings = [
        line.lstrip("#").strip()
        for line in MANUAL.read_text(encoding="utf-8").splitlines()
        if line.startswith(("## ", "### ", "#### ", "##### "))
    ]

    assert headings
    assert not [heading for heading in headings if heading not in guide]


def test_manual_and_readme_link_the_guide() -> None:
    guide_name = "部署和运维手册逐章解读.md"

    assert guide_name in MANUAL.read_text(encoding="utf-8")
    assert guide_name in (ROOT / "README.md").read_text(encoding="utf-8")


def test_operator_document_layers_publish_a_single_normal_path() -> None:
    manual = MANUAL.read_text(encoding="utf-8")
    quickstart = QUICKSTART.read_text(encoding="utf-8")
    daily = DAILY_OPS.read_text(encoding="utf-8")
    safety = SAFETY.read_text(encoding="utf-8")
    developer = DEVELOPER.read_text(encoding="utf-8")

    for path in (QUICKSTART, PROFILE_APPROVAL, DAILY_OPS, SAFETY, DEVELOPER):
        assert path.is_file(), f"missing operator document: {path.name}"
        assert path.name in manual

    for value in (
        "gpu-fault-admin deploy",
        "--cpu-cluster-arn",
        "--gpu-cluster-arn",
        "--state-dir",
        "--admin-email",
        "不提供release-ref、artifact、site",
    ):
        assert value in quickstart
    assert "gpu-fault-admin deploy -f" not in quickstart
    assert "config/site.example.yaml" not in quickstart
    assert "## 10. 后续运维入口" in quickstart
    assert "[管理员日常运维](管理员日常运维.md)" in quickstart
    assert "[Runtime Profile变更审批](管理员Profile变更审批.md)" in quickstart
    assert len(quickstart.splitlines()) <= 300
    for value in (
        "gpu-fault-admin remove-cluster",
        "gpu-fault-admin uninstall",
        "READY_TO_DELETE_AURORA",
        "uninstall/state.json",
        "installation-resources-final.json",
        "## 11. 当前自动化缺口",
        "## 12. 自动化路线图",
    ):
        assert value not in quickstart
    assert "gpu-fault-admin uninstall \\\n  --cpu-cluster-arn" not in quickstart

    assert "Runbook 固定格式" in daily
    assert "gpu-fault-admin join-cluster" in daily
    assert "gpu-fault-admin remove-cluster" in daily
    assert "gpu-fault-admin uninstall" in daily
    uninstall_section = daily.split("## 3. ", 1)[1].split("\n## 4. ", 1)[0]
    assert "--cpu-cluster-arn" not in uninstall_section, (
        "the ARN discovery branch of uninstall was removed with legacy_site"
    )
    assert "--reset-database" in uninstall_section, (
        "the explicit database wipe must be documented next to keep mode"
    )
    assert "installation_resource" in daily
    assert "delete_policy_residuals" in daily
    assert "--cpu-cluster delete" in daily
    assert "READY_TO_DELETE_AURORA" in daily
    assert "当前待产品化命令" in daily
    assert "已完成P0" in daily
    assert "不可放宽的生产约束" in safety
    assert "| D.2 | warm-spare `nodeReplace`" in safety
    assert "`HEALTHY_WARM_SPARE_ONLY`" in safety
    assert "`GPU_FAULT_ALLOW_HYPERPOD_REPLACE=false`" in safety
    assert "管理员和客户不编写方案 Manifest" in developer
    assert "管理员不维护 `regional-release.json`" in developer
    assert "gpu-fault-admin join-cluster" in developer
    assert "gpu-fault-admin remove-cluster" in developer
    assert "Aurora中的`installation_resource`对象" in developer
