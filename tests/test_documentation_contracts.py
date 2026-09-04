from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DETAIL = DOCS / "详细设计.md"
OVERVIEW_V2 = DOCS / "概要设计-v2.md"
DETAIL_V2 = DOCS / "详细设计-v2.md"
INTERNAL_RESEARCH_DOCUMENTS = (
    "RESEARCH.md",
    "CLOUD_PROVIDER_SOURCE_REVIEW.md",
    "EKS_AGENT_SOURCE_REVIEW.md",
    "EKS_NVIDIA_CONSISTENCY.md",
    "HYPERPOD_SOURCE_REVIEW.md",
    "EKS_REFERENCE_IMPLEMENTATION.md",
    "HYPERPOD_ADAPTER.md",
    "HYPERPOD_GPU_METRICS_TEST.md",
    "IMPLEMENTATION.md",
    "K8S_COMPLETION_WATCHER_IMPLEMENTATION.md",
    "PORTABLE_DEPLOYMENT_ARCHITECTURE.md",
    "RUNTIME_WORKFLOW_INTEGRATION.md",
)
INTERNAL_DOC_NAMES = {
    "架构重构批次.md",
    "代码审阅报告整改矩阵.md",
    "性能瓶颈审阅报告整改矩阵.md",
}
PUBLIC_RELEASE_DOCUMENTS = (
    "README.md",
    "COLLECTORS.md",
    "docs/README.md",
    "docs/概要设计.md",
    "docs/概要设计-v2.md",
    "docs/详细设计.md",
    "docs/详细设计-v2.md",
    "docs/components/nvidia-policy.md",
    "docs/管理员快速部署.md",
    "docs/管理员Profile变更审批.md",
    "docs/管理员日常运维.md",
    "docs/管理员环境变量参考.md",
    "docs/安全与参数参考.md",
    "docs/部署和运维手册.md",
    "docs/部署和运维手册逐章解读.md",
    "docs/开发者部署实现.md",
    "docs/区域模式端到端验收测试用例.md",
    "docs/区域用例索引.md",
    "docs/故障模拟测试手册.md",
    "docs/性能压测验收方案.md",
    "deploy/hyperpod/README.md",
    "scripts/e2e/regional/boot_guard/README.md",
)


def test_document_code_and_test_references_are_valid() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-doc-references.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_document_reference_scan_covers_root_markdown_and_skips_urls() -> None:
    # 扫描面不能只含 docs/：公开根 README 同样会引用实现路径。
    # 反向也要守：上游仓库的链接里也有 src/... 路径，那不是本仓引用。
    spec = importlib.util.spec_from_file_location(
        "check_doc_references", ROOT / "scripts/check-doc-references.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses 要在 sys.modules 里找得到定义模块才能解析注解。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scanned = set(module.documents())
    assert ROOT / "README.md" in scanned
    assert DETAIL in scanned
    assert not any("history" in path.relative_to(ROOT).parts for path in scanned)
    for name in INTERNAL_RESEARCH_DOCUMENTS:
        assert ROOT / name not in scanned
        assert name in module.PRIVATE_FILES

    # 反向漂移守卫。豁免面现在从 .gitignore 派生，两个方向都要守：
    # 「名单里的都被 ignore」由下面 test_internal_research_... 的 `/name` 断言
    # 覆盖，这里守的是「被 ignore 的都不在扫描面里」。手抄名单守不住这一向，
    # 也守不住死条目——IMPLEMENTATION.md 等 6 个名字已经搬进 internal-docs/，
    # 留在根目录的豁免条目会静默豁免任何同名新建文档。
    anchored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("/")
    }
    directories = tuple(entry[1:] for entry in anchored if entry.endswith("/"))
    assert "internal-docs/" in module.PRIVATE_DIRECTORIES
    for path in scanned:
        relative = path.relative_to(ROOT).as_posix()
        assert f"/{relative}" not in anchored, relative
        assert not relative.startswith(directories), relative

    masked = module.mask_urls(
        "见 [上游](https://github.com/x/y/blob/abc/src/gpu_healthcheck/x.py) 与 "
        "`src/gpu_fault/policy/__init__.py`"
    )
    assert "gpu_healthcheck" not in masked
    assert "src/gpu_fault/policy/__init__.py" in masked


def load_gate(name: str, relative: str) -> object:
    """Import a hyphen-named gate script so its helpers can be tested directly."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    # dataclasses 要在 sys.modules 里找得到定义模块才能解析注解。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_document_anchor_links_resolve() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-doc-anchors.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_anchor_slugs_match_the_rendered_site() -> None:
    """The slug rule is GitHub's, and the docs already depend on every clause of it.

    区域用例索引 links to ``#gf-regional-destr-009``, 管理员日常运维 links to
    ``#81-gpu-数量变化审批`` and ``#83-check_mechanicals-...``. Get the punctuation or
    the underscore clause wrong and the gate either passes broken links or fails good
    ones, so the cases below are taken from links this repository actually ships.
    """
    module = load_gate("check_doc_anchors", "scripts/check-doc-anchors.py")

    assert module.heading_slug(" 8.1 GPU 数量变化审批") == "81-gpu-数量变化审批"
    # ``_`` 是 word 字符，渲染站点保留它；``（）`` 是标点，一个字符都不留。
    assert (
        module.heading_slug(" 8.3 CHECK_MECHANICALS 后由管理员提交明确处置")
        == "83-check_mechanicals-后由管理员提交明确处置"
    )
    assert (
        module.heading_slug(" 5A 区域分离生产部署（唯一生产形态）")
        == "5a-区域分离生产部署唯一生产形态"
    )
    assert module.heading_slug(" GF-REGIONAL-DESTR-009") == "gf-regional-destr-009"
    # 标题里的行内代码、强调和链接都是标记，渲染后的文字才参与 slug。
    assert (
        module.heading_slug(" **`policy/nvlink74.py`** 说明") == "policynvlink74py-说明"
    )
    assert module.heading_slug(" 见 [手册](部署和运维手册.md)") == "见-手册"


def test_repeated_headings_and_explicit_anchors_are_both_reachable(
    tmp_path: Path,
) -> None:
    document = tmp_path / "sample.md"
    document.write_text(
        '# 概览\n<a id="pinned-chapter"></a>\n## 步骤\n## 步骤\n## 步骤\n',
        encoding="utf-8",
    )
    module = load_gate("check_doc_anchors_repeat", "scripts/check-doc-anchors.py")

    assert module.anchors(document) == {
        "概览",
        "pinned-chapter",
        "步骤",
        "步骤-1",
        "步骤-2",
    }


def test_fenced_code_yields_neither_headings_nor_links(tmp_path: Path) -> None:
    """Shell comments are not headings and quoted markdown is not a link.

    The manuals are mostly fenced command blocks, and those blocks are full of
    ``# 说明`` comment lines. Treating them as headings would invent anchors that the
    rendered page does not have, which is how a gate starts passing broken links.
    """
    document = tmp_path / "sample.md"
    document.write_text(
        "# 真标题\n"
        "```bash\n"
        "# 这是注释不是标题\n"
        "echo '[见](#不存在的锚点)'\n"
        "```\n"
        "行内代码里的 `[见](#也不存在)` 同样不是链接。\n"
        "真链接：[真标题](#真标题)\n",
        encoding="utf-8",
    )
    module = load_gate("check_doc_anchors_fence", "scripts/check-doc-anchors.py")

    assert module.anchors(document) == {"真标题"}
    assert [link.target for link in module.local_links(document)] == ["#真标题"]


def test_the_anchor_gate_reports_a_fragment_that_no_heading_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "sample.md"
    document.write_text(
        "# 真标题\n[好链接](#真标题)\n[坏链接](#拼错的锚点)\n[坏路径](缺失的文件.md)\n",
        encoding="utf-8",
    )
    module = load_gate("check_doc_anchors_teeth", "scripts/check-doc-anchors.py")
    monkeypatch.setattr(module, "documents", lambda: [document])

    assert [(item.line, item.link) for item in module.check_anchors()] == [
        (3, "#拼错的锚点"),
        (4, "缺失的文件.md"),
    ]


def test_the_anchor_gate_covers_the_case_index_cross_document_links() -> None:
    """Non-vacuity: the 152 links that make 区域用例索引 navigable are in scope.

    A regex or fence bug in the gate would show up as silence rather than as a
    failure, and the case index is the largest inbound anchor surface in the
    repository -- if these are being scanned, the gate is doing its job.
    """
    module = load_gate("check_doc_anchors_scope", "scripts/check-doc-anchors.py")
    index = DOCS / "区域用例索引.md"
    cases = DOCS / "区域模式端到端验收测试用例.md"

    into_cases = [
        link
        for link in module.local_links(index)
        if link.path == cases.name and link.fragment
    ]

    assert len(into_cases) >= 150, len(into_cases)
    available = module.anchors(cases)
    assert "gf-regional-destr-009" in available
    for link in into_cases:
        assert link.fragment in available, link.target


def test_detailed_design_lists_every_console_script() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["scripts"]
    detail = DETAIL.read_text(encoding="utf-8")
    table = detail.split("| 命令 | 入口 |", 1)[1].split("\n\n", 1)[0]
    actual = dict(re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|$", table, re.MULTILINE))

    assert actual == expected


def test_detailed_design_names_current_packages() -> None:
    headings = "\n".join(
        line
        for line in DETAIL.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )

    assert "`src/gpu_fault/policy/`" in headings
    assert "`src/gpu_fault/collectors/`" in headings
    assert "`src/gpu_fault/node_agent/`" in headings
    assert "`src/gpu_fault/store/`" in headings
    assert "`policy.py`" not in headings
    assert "`collectors.py`" not in headings


def test_v2_designs_are_covered_by_every_documentation_impact_contract() -> None:
    contracts = (DOCS / "code-doc-contracts.yaml").read_text(encoding="utf-8")

    for document in ("docs/概要设计-v2.md", "docs/详细设计-v2.md"):
        assert contracts.count(document) == 3


def test_v2_capability_matrix_matches_registered_operations() -> None:
    from gpu_fault.models import WorkflowOperation

    overview = OVERVIEW_V2.read_text(encoding="utf-8")
    detail = DETAIL_V2.read_text(encoding="utf-8")
    overview_count = re.search(r"`WorkflowOperation` 共 (\d+) 个取值", overview)
    detail_count = re.search(r"`WorkflowOperation` \*\*(\d+)\*\* 种", detail)

    assert overview_count is not None
    assert detail_count is not None
    assert int(overview_count.group(1)) == len(WorkflowOperation)
    assert int(detail_count.group(1)) == len(WorkflowOperation)
    assert "| Fabric Manager 修复 | `RESTART_FABRIC_MANAGER` | 已实现" in overview
    assert "| 人工步骤 | `CHECK_MECHANICALS`、`ESCALATE_SUPPORT` | 已实现" in overview
    assert (
        "| fail-closed | `RESTART_FABRIC_MANAGER`、`CHECK_MECHANICALS`" not in overview
    )


def test_designs_agree_on_mechanical_inspection_waiting_semantics() -> None:
    overview = OVERVIEW_V2.read_text(encoding="utf-8")
    detail = DETAIL.read_text(encoding="utf-8")
    detail_v2 = DETAIL_V2.read_text(encoding="utf-8")

    assert "`CHECK_MECHANICALS` 等待 annotation" in overview
    assert "不自动 cordon 或停止健康 workload" in detail
    assert "`CHECK_MECHANICALS` 收到精确的 incident/fencing annotation" in detail_v2
    assert "未确认时保持 `WAITING`" in detail_v2


def test_v2_node_agent_contract_matches_runtime_models_and_routes() -> None:
    from gpu_fault.node_agent.app import create_node_agent_app
    from gpu_fault.node_agent.protocol import NodeActionCommand, NodeActionResult

    detail = DETAIL_V2.read_text(encoding="utf-8")
    module_section = detail.split("### 3.1 Node Agent", 1)[1].split(
        "### 3.2 NVIDIA", 1
    )[0]
    route_section = detail.split("### 8.6 Node Agent 接口", 1)[1].split("### 8.7", 1)[0]

    def documented_fields(marker: str) -> set[str]:
        line = next(line for line in module_section.splitlines() if marker in line)
        values = re.findall(r"`([^`]+)`", line)
        return set(values[1:])

    assert documented_fields("`NodeActionCommand` 一等字段") == set(
        NodeActionCommand.model_fields
    )
    assert documented_fields("`NodeActionResult` 一等字段") == set(
        NodeActionResult.model_fields
    )

    class Ledger:
        @staticmethod
        def get(_command_id):
            return None

    class Agent:
        secret = "s" * 32
        ledger = Ledger()

    app = create_node_agent_app(executor=Agent(), heartbeat_reporter=object())
    runtime_routes = {
        (method, route.path)
        for route in app.routes
        for method in route.methods or set()
        if route.path.startswith("/v1/node-actions") or route.path == "/healthz"
    }
    documented_routes = {
        (method, path.split("?", 1)[0])
        for method, path in re.findall(
            r"^\| (GET|POST) \| `([^`]+)`", route_section, re.MULTILINE
        )
    }
    assert documented_routes == runtime_routes


def test_v2_node_agent_transport_names_the_production_plaintext_exception() -> None:
    detail = DETAIL_V2.read_text(encoding="utf-8")
    installer = (ROOT / "deploy/node/run-hyperpod-installer-job.sh").read_text(
        encoding="utf-8"
    )

    assert "当前区域生产安装" in detail
    assert "自动生成" in detail
    assert "--allow-node-agent-plaintext" not in installer
    assert '"https://\\${TARGET_NODE_IP}:9099"' in installer


def test_v2_device_identity_is_cluster_scoped() -> None:
    detail = DETAIL_V2.read_text(encoding="utf-8")

    assert "`(cluster_id, node_id, gpu_uuid)`" in detail
    assert "设备身份是 `(node_id, gpu_uuid)`" not in detail


def test_docs_map_is_the_human_entrypoint() -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for document in DOCS.glob("*.md"):
        if document.name == "README.md" or document.name in INTERNAL_DOC_NAMES:
            continue
        assert document.name in index
    for document in (DOCS / "components").glob("*.md"):
        assert f"components/{document.name}" in index

    assert "docs/扩展指南.md" in contributing
    assert "docs/开发者部署实现.md" in contributing
    assert "docs/README.md" in root_readme
    assert "CONTRIBUTING.md" in root_readme


def test_docs_tree_has_no_vendored_wheels() -> None:
    assert not list(DOCS.rglob("*.whl"))
    assert "*.whl" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_developer_guides_split_code_extensions_from_production_delivery() -> None:
    guide = (DOCS / "扩展指南.md").read_text(encoding="utf-8")
    developer = (DOCS / "开发者部署实现.md").read_text(encoding="utf-8")

    for value in (
        "OPERATION_REGISTRY",
        "CHANNEL_REGISTRY",
        "POSTGRES_SCHEMA_MIGRATIONS",
        "authorization_bucket",
        "metric_contributors",
        "[开发者部署实现](开发者部署实现.md)",
    ):
        assert value in guide
    assert "目前只有" in guide
    assert "不会自动生效" in guide
    assert "不再重复" in guide
    assert guide.count("## 1. 新增 Workflow Operation") == 1

    for value in (
        "site.yaml",
        "gpu-fault-admin remove-cluster",
        "required/compatible",
        "gpu-fault.io/cleanup-phase",
        "deployment-contracts-update",
        "gpu-fault-installed-resources",
        "/opt/gpu-fault/installed-units.txt",
        "installation_resource",
        "READY_TO_DELETE_AURORA",
        "tests/admin/test_admin_uninstall.py",
    ):
        assert value in developer
    assert "管理员不维护 `regional-release.json`" in developer
    assert "三层事实源" in developer


def test_developer_manual_documents_every_runtime_profile_capability() -> None:
    developer = (DOCS / "开发者部署实现.md").read_text(encoding="utf-8")
    profile = yaml.safe_load(
        (ROOT / "config/runtime-profile.regional-hyperpod-safe.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    section = developer.split("### 4.1 Runtime Profile的作用", 1)[1].split("## 5.", 1)[
        0
    ]

    for claim in profile["claims"]:
        capability = claim["capability"]
        assert f"`{capability}`" in section, (
            f"developer manual omits Runtime Profile capability {capability}"
        )
    for value in ("OWN", "DELEGATE", "AUGMENT", "OBSERVE", "DISABLED"):
        assert f"`{value}`" in section, (
            f"developer manual omits Runtime Profile mode {value}"
        )
    for value in (
        "templateSource",
        "gpu-fault-admin approve-profile",
        "--plan-sha256",
        "profile-plan.json",
        "site_identity",
        "cpu_eks_arn",
        "plan_sha256",
        "regional-hyperpod-<digest12>",
        "gpu-training-submit --site",
        "gpu-fault-workload-annotate --site",
    ):
        assert value in section, f"developer manual omits Profile workflow {value}"


def test_internal_research_is_ignored_and_not_publicly_referenced() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    html_path = ROOT / "GPU_FAILURE_AUTOMATION_DESIGN.html"
    html = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
    public_documents = [
        path
        for path in (*ROOT.glob("*.md"), *DOCS.rglob("*.md"))
        if path.name not in INTERNAL_RESEARCH_DOCUMENTS
        and path.name not in INTERNAL_DOC_NAMES
        and "history" not in path.relative_to(ROOT).parts
        and "security" not in path.relative_to(ROOT).parts
        and not path.relative_to(ROOT).as_posix().startswith("docs/evidence/perf/")
    ]

    for name in INTERNAL_RESEARCH_DOCUMENTS:
        reference = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(name)}")
        assert f"/{name}" in ignore
        offenders = [
            path.relative_to(ROOT)
            for path in public_documents
            if reference.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == [], f"{name} is referenced by {offenders}"
        if html:
            assert reference.search(html) is None


def test_public_release_documents_contain_no_site_identifiers() -> None:
    patterns = {
        "customer AWS account": r"(?<!\d)51" r"4385905925(?!\d)",
        "concrete VPC": r"\bvpc-[0-9a-f]{8,}\b",
        "concrete subnet": r"\bsubnet-[0-9a-f]{8,}\b",
        "concrete AMP workspace": r"\bws-[0-9a-f]{8,}(?:-[0-9a-f]+)*\b",
        "concrete HyperPod node": r"\bhyperpod-i-[0-9a-f]{8,}\b",
        "site private address": r"\b10\.91(?:\.\d{1,3}){2}\b",
        "site HyperPod cluster": r"\bhp-cluster-hypd-[A-Za-z0-9-]+\b",
        "site EKS cluster": r"\beks-cluster-hypd-[A-Za-z0-9-]+\b",
        "site control plane": r"control-plane-" r"GPU-fault-solution",
        "local repository path": r"/home/ubuntu/GPU_failure_handling",
    }

    for relative in PUBLIC_RELEASE_DOCUMENTS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            assert re.search(pattern, text) is None, f"{relative}: {label}"

    html_path = ROOT / "GPU_FAILURE_AUTOMATION_DESIGN.html"
    if html_path.is_file():
        html = html_path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            assert re.search(pattern, html) is None, f"generated HTML: {label}"


def test_public_specs_do_not_embed_private_execution_history() -> None:
    regional = (DOCS / "区域模式端到端验收测试用例.md").read_text(encoding="utf-8")
    performance = (DOCS / "性能压测验收方案.md").read_text(encoding="utf-8")

    for value in (
        "区域分离部署验收执行史.md",
        "internal-docs/",
        "真实环境执行结论",
        "当前release真机复验",
        "当前release回归",
    ):
        assert value not in regional
    assert "2026-" not in performance
    assert "真实压测结果必须进入 CI artifact" in performance


def test_root_readme_is_a_bounded_public_entrypoint() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 260
    assert "本方案永不调用 `BatchReplaceClusterNodes`" in readme
    assert "docs/部署和运维手册.md" in readme
    manual_gates = readme.split("需要手工逐项执行时，建议按以下顺序：", 1)[1].split(
        "```", 2
    )[1]
    expected_order = (
        "make mypy-check",
        "make architecture-check",
        "make docs-check",
        "make artifact-check",
        "make test-parallel",
    )
    assert tuple(sorted(expected_order, key=manual_gates.index)) == expected_order
    for name in INTERNAL_RESEARCH_DOCUMENTS:
        assert name not in readme


def test_root_readme_separates_developer_and_admin_deployment() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    developer = readme.split("### 开发者：修改代码或Profile后发布", 1)[1].split(
        "### 管理员：首次部署和日常管理", 1
    )[0]
    administrator = readme.split("### 管理员：首次部署和日常管理", 1)[1].split(
        "## 训练任务提交", 1
    )[0]

    for value in (
        "gpu-fault-admin deploy",
        "--cpu-cluster-arn <cpu-arn>",
        "--gpu-cluster-arn <gpu-arn>",
        "--state-dir /secure/gpu-fault-staging",
        "--admin-email <operations-email>",
        "`staging_only`",
        "--plan-sha256",
        "release-ref",
        "不提供",
        "完整生产门禁",
    ):
        assert value in developer

    for value in (
        "--cpu-cluster-arn",
        "--gpu-cluster-arn",
        "--state-dir /secure/gpu-fault",
        "--admin-email",
        "完全相同的命令",
        "CPU/GPU身份集合未变化",
        "gpu-fault-admin join-cluster",
        "gpu-fault-admin remove-cluster",
        "--confirm REMOVE_GPU_CLUSTER",
        "--cpu-cluster keep",
        "--confirm UNINSTALL_GPU_FAULT",
        "--cpu-cluster delete",
        "--confirm DELETE_CPU_CONTROL_PLANE",
    ):
        assert value in administrator
    assert "gpu-fault-admin deploy -f" not in administrator


def test_admin_profile_approval_guide_closes_the_review_loop() -> None:
    guide = (DOCS / "管理员Profile变更审批.md").read_text(encoding="utf-8")
    builder = (ROOT / "build-html.sh").read_text(encoding="utf-8")
    index = (DOCS / "README.md").read_text(encoding="utf-8")

    for value in (
        "## 1. 最短操作路径",
        "## 7. 完整命令模板",
        "site_identity.cpu_eks_arn",
        "site_identity_sha256",
        "--plan-sha256",
        "profile-approvals/<plan_sha256>/",
        "UNKNOWN_BASELINE",
        "SUPERSEDED",
        "CONSUMED",
        "ALREADY_APPLIED",
        "不手工创建",
        "GPU_CLUSTER_ARNS=(",
        "APPROVED_PLAN_SHA256",
        "Stop: approve this plan externally",
        'test -f "${PLAN_FILE}" || exit "${DEPLOY_RC}"',
    ):
        assert value in guide, f"Profile approval guide omits {value}"
    assert "PROFILE_APPROVAL=" not in guide
    assert "--profile-approval" not in guide
    assert "docs/管理员Profile变更审批.md" in builder
    assert "[Runtime Profile变更审批](管理员Profile变更审批.md)" in index


def test_public_collector_docs_keep_node_log_disabled() -> None:
    collectors = (ROOT / "COLLECTORS.md").read_text(encoding="utf-8")
    overview = (DOCS / "概要设计.md").read_text(encoding="utf-8")
    default_block = collectors.split("默认HyperPod生产拓扑启用：", 1)[1].split(
        "`nvidia-smi`", 1
    )[0]

    assert "node-log节点采集器" not in default_block
    assert "`NodeLogCollector` 当前默认禁用" in collectors
    assert "`NodeLogCollector`" in overview
    assert "默认禁用，仅隔离验证" in overview


def test_internal_document_trees_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for value in (
        "/docs/history/",
        "/docs/security/",
        "/docs/evidence/perf/",
        "/docs/架构重构批次.md",
        "/docs/代码审阅报告整改矩阵.md",
        "/docs/性能瓶颈审阅报告整改矩阵.md",
    ):
        assert value in ignore


def test_large_acceptance_documents_publish_the_anchor_contract() -> None:
    for name in ("区域模式端到端验收测试用例.md",):
        preamble = "\n".join(
            (DOCS / name).read_text(encoding="utf-8").splitlines()[:12]
        )
        assert "维护契约" in preamble
        assert "锚点" in preamble


def test_html_build_uses_the_current_authoritative_document_set() -> None:
    builder = (ROOT / "build-html.sh").read_text(encoding="utf-8")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'PYTHONPATH="${root_dir}/src' in builder
    assert 'mermaid_version="11.16.0"' in builder
    assert "74d7c46dabca328c2294733910a8aa1ed0c37451776e8d5295da38a2b758fb9b" in builder
    assert "/GPU_FAILURE_AUTOMATION_DESIGN.html" in ignore
    assert "/html-assets/mermaid.min.js" in ignore
    assert (ROOT / "html-assets/README.md").is_file()
    assert (ROOT / "html-assets/MERMAID-LICENSE.txt").is_file()
    for path in (
        "docs/README.md",
        "docs/概要设计.md",
        "docs/详细设计.md",
        "docs/components/nvidia-policy.md",
        "docs/管理员快速部署.md",
        "docs/管理员日常运维.md",
        "docs/安全与参数参考.md",
        "docs/部署和运维手册.md",
        "docs/开发者部署实现.md",
        "docs/区域模式端到端验收测试用例.md",
    ):
        assert path in builder

    for historical_input in (
        "PORTABLE_DEPLOYMENT_ARCHITECTURE.md",
        "EKS_REFERENCE_IMPLEMENTATION.md",
        "K8S_COMPLETION_WATCHER_IMPLEMENTATION.md",
        "RESEARCH.md",
        "CLOUD_PROVIDER_SOURCE_REVIEW.md",
        "EKS_AGENT_SOURCE_REVIEW.md",
        "EKS_NVIDIA_CONSISTENCY.md",
        "HYPERPOD_SOURCE_REVIEW.md",
        "docs/架构重构批次.md",
        "docs/代码审阅报告整改矩阵.md",
        "docs/性能瓶颈审阅报告整改矩阵.md",
        "docs/security/credential-exposure-20260823.md",
    ):
        assert historical_input not in builder
