from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-public-release.py"


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def public_tree(tmp_path: Path, content: str) -> Path:
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.md").write_text(content, encoding="utf-8")
    return tmp_path


def test_repository_public_tree_contains_no_site_identity() -> None:
    result = run_check(ROOT)

    assert result.returncode == 0, result.stdout + result.stderr


def test_public_release_scan_uses_git_without_python_pathspec(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("/ignored/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("public documentation\n", encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "live.md").write_text("node i-0305bbcc538883eb6\n", encoding="utf-8")

    command = (
        "import runpy,sys;"
        "sys.modules['pathspec']=None;"
        f"sys.argv=['check-public-release.py','--root',{str(tmp_path)!r}];"
        f"runpy.run_path({str(SCRIPT)!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("arn:aws:eks:us-west-2:514385905925:cluster/site\n", "customer AWS account"),
        ("node i-0305bbcc538883eb6\n", "concrete EC2 instance ID"),
        ("network vpc-0123abc456def7890\n", "concrete AWS resource ID"),
        ("endpoint 10.91.48.46\n", "site private address"),
        ("cluster eks-cluster-hypd-site\n", "site-specific resource name"),
        ("email operator@amazon.com\n", "personal email"),
        (
            "boot 8ca90906-755a-4cd8-b27e-44382277cf70\n",
            "raw UUID in public documentation",
        ),
        ("node 026c36a28c56ea610\n", "raw node identity in public documentation"),
    ],
)
def test_public_release_gate_rejects_live_identity(
    tmp_path: Path, content: str, message: str
) -> None:
    result = run_check(public_tree(tmp_path, content))

    assert result.returncode == 1
    assert message in result.stderr


def test_public_release_gate_honors_private_archive_boundary(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("/internal-docs/\n", encoding="utf-8")
    internal = tmp_path / "internal-docs"
    internal.mkdir()
    (internal / "live.md").write_text("node i-0305bbcc538883eb6\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("public documentation\n", encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_public_release_gate_allows_documented_placeholders(tmp_path: Path) -> None:
    content = "\n".join(
        (
            "arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-a",
            "602401143452.dkr.ecr.us-west-2.amazonaws.com/hyperpod/adot:v1",
            "instance i-00000000000000001",
            "security group sg-0123456789abcdef0",
            "operator@example.invalid",
        )
    )

    result = run_check(public_tree(tmp_path, content))

    assert result.returncode == 0, result.stdout + result.stderr


def test_public_release_gate_is_wired_into_local_and_ci_checks() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "$(MAKE) public-release-check" in makefile
    assert "scripts/check-public-release.py" in workflow
