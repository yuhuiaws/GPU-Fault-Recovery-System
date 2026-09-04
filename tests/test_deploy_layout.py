from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-deploy-layout.py"


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--deploy-root", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def minimal_tree(tmp_path: Path) -> Path:
    deploy = tmp_path / "deploy"
    generated = deploy / "control-plane" / "regional" / "generated"
    generated.mkdir(parents=True)
    (deploy / "README.md").write_text("# deploy\n")
    (generated / "README.md").write_text("# generated\n")
    (generated / ".generated").write_text("generated\n")
    (generated / "manifest-list.txt").write_text("")
    return deploy


def test_repository_deploy_tree_is_production_only() -> None:
    result = run_check(ROOT / "deploy")

    assert result.returncode == 0, result.stdout + result.stderr


def test_layout_gate_rejects_test_manifest(tmp_path: Path) -> None:
    deploy = minimal_tree(tmp_path)
    (deploy / "fault-inject-e2e.yaml").write_text(
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: inject\n"
    )

    result = run_check(deploy)

    assert result.returncode == 1
    assert "test-like filename" in result.stdout


def test_layout_gate_rejects_kmsg_payload_in_any_kind(tmp_path: Path) -> None:
    # 这条规则原来写成「顶层 kind 是 Pod 且正文含 /dev/kmsg」，而 deploy 下
    # 一份 kind: Pod 都没有（只有 Deployment/DaemonSet/Job/ConfigMap），所以
    # 那个 append 从来没执行过。注入载荷更可能是贴进某个控制器模板，
    # 文件名也不一定带 inject/e2e。
    deploy = minimal_tree(tmp_path)
    node = deploy / "node"
    node.mkdir()
    (node / "collector.yaml").write_text(
        "apiVersion: apps/v1\n"
        "kind: DaemonSet\n"
        "metadata:\n"
        "  name: gpu-fault-collector\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "        - name: inject\n"
        "          args:\n"
        '            - "echo NVRM: Xid 79 > /dev/kmsg"\n'
    )

    result = run_check(deploy)

    assert result.returncode == 1
    assert "fault-injection manifest under deploy: node/collector.yaml" in result.stdout


def test_layout_gate_rejects_unlisted_generated_file(tmp_path: Path) -> None:
    deploy = minimal_tree(tmp_path)
    generated = deploy / "control-plane" / "regional" / "generated"
    (generated / "gpu-fault-extra.yaml").write_text(
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: gpu-fault-extra\n"
    )

    result = run_check(deploy)

    assert result.returncode == 1
    assert "differs from manifest-list.txt" in result.stdout


def test_layout_gate_rejects_hyphenated_python_module(tmp_path: Path) -> None:
    deploy = minimal_tree(tmp_path)
    (deploy / "bad-tool.py").write_text("print('bad')\n")

    result = run_check(deploy)

    assert result.returncode == 1
    assert "non-importable Python filename" in result.stdout


def test_layout_gate_rejects_python_cache_artifacts(tmp_path: Path) -> None:
    deploy = minimal_tree(tmp_path)
    cache = deploy / "control-plane" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "tool.cpython-312.pyc").write_bytes(b"compiled")

    result = run_check(deploy)

    assert result.returncode == 1
    assert "Python cache artifact under deploy" in result.stdout


def test_lazy_path_module_does_not_write_python_cache(tmp_path: Path) -> None:
    script = tmp_path / "tool.py"
    script.write_text("VALUE = 7\n", encoding="utf-8")
    module = lazy_script_module(script)

    assert module.VALUE == 7
    assert not (tmp_path / "__pycache__").exists(), (
        "lazy path-module loading must not write bytecode beside deploy tools"
    )


def test_quality_gates_cover_deploy_tree() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    architecture = (ROOT / "scripts/check-python-architecture.py").read_text(
        encoding="utf-8"
    )

    assert "ruff check src tests deploy" in makefile
    assert "src tests deploy $(QUALITY_SCRIPTS)" in makefile
    assert "$(MAKE) deploy-check" in makefile
    assert "$(MAKE) shell-check" in makefile
    assert "$(MAKE) assert-message-check" in makefile
    assert "ruff check src tests deploy" in workflow
    assert "check-deploy-layout.py" in workflow
    assert "check-assert-messages.py" in workflow
    assert "shellcheck --severity=warning" in workflow
    assert "SOURCE_ROOTS = (SOURCE, DEPLOY, SCRIPTS, TOOLS, TESTS)" in architecture
