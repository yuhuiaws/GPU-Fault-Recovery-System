from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

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
    "docs/部署和运维手册.md",
    "docs/部署和运维手册逐章解读.md",
    "docs/区域模式端到端验收测试用例.md",
    "docs/区域用例索引.md",
    "docs/故障模拟测试手册.md",
    "docs/性能压测验收方案.md",
    "deploy/hyperpod/README.md",
    "scripts/boot-guard-probe/README.md",
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
    assert "`GPU_FAULT_NODE_AGENT_ALLOW_PLAINTEXT=true`" in detail
    assert "--allow-node-agent-plaintext" in installer
    assert '"http://\\${TARGET_NODE_IP}:9099"' in installer


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
    assert "docs/README.md" in root_readme
    assert "CONTRIBUTING.md" in root_readme


def test_docs_tree_has_no_vendored_wheels() -> None:
    assert not list(DOCS.rglob("*.whl"))
    assert "*.whl" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_extension_guide_covers_deployment_and_cleanup_extensions() -> None:
    guide = (DOCS / "扩展指南.md").read_text(encoding="utf-8")

    for value in (
        "gpu-fault.io/cleanup-phase",
        "deployment-contracts-update",
        "gpu-fault-installed-resources",
        "/opt/gpu-fault/installed-units.txt",
        "READY_TO_DELETE_AURORA",
        "tests/regional/test_cleanup_state.py",
    ):
        assert value in guide
    assert "管理员和客户不编写" in guide
    assert guide.count("## 1. 新增 Workflow Operation") == 1


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

    assert len(readme.splitlines()) <= 220
    assert "本方案永不调用 `BatchReplaceClusterNodes`" in readme
    assert "docs/部署和运维手册.md" in readme
    for name in INTERNAL_RESEARCH_DOCUMENTS:
        assert name not in readme


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
        "docs/部署和运维手册.md",
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
