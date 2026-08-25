from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-artifacts.py"


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--artifacts-root", str(root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_repository_artifacts_are_safe() -> None:
    content = [
        path
        for path in (ROOT / "artifacts").rglob("*")
        if path.is_file() and path.relative_to(ROOT / "artifacts") != Path("README.md")
    ]
    if not content:
        pytest.skip(
            "local artifacts are absent in this checkout; "
            "run make artifacts-local-safety-check where artifacts exist"
        )
    result = run_check(ROOT / "artifacts")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"scanned {len(content)} file(s)" in result.stdout


def test_artifacts_gate_rejects_secret(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.yaml"
    path.write_text("apiVersion: v1\nkind: Secret\nmetadata:\n  name: leaked\n")

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "Kubernetes Secret" in result.stdout


def test_artifacts_gate_rejects_wheel_binary(tmp_path: Path) -> None:
    path = tmp_path / "configmaps.yaml"
    path.write_text(
        "binaryData:\n  gpu_fault_control_plane-0.10.0-py3-none-any.whl: UEsDBA==\n"
    )

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "wheel binaryData" in result.stdout


def test_artifacts_gate_rejects_large_file(tmp_path: Path) -> None:
    path = tmp_path / "oversized.log"
    path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "maximum" in result.stdout


def test_artifacts_gate_accepts_binary_digest_evidence(tmp_path: Path) -> None:
    path = tmp_path / "configmaps.yaml"
    path.write_text(
        "evidenceBinaryData:\n"
        "  gpu_fault_control_plane-0.10.0-py3-none-any.whl:\n"
        "    sha256: abc\n"
        "    decodedBytes: 123\n"
    )

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stdout


def test_artifacts_gate_rejects_a_cleartext_token_outside_a_secret_manifest(
    tmp_path: Path,
) -> None:
    # 真实泄漏就是这个形状：``registry-baseline.json`` 里一行
    # ``"token": "<64 hex>"``，既不是 ``kind: Secret`` 也不是 wheel，
    # 原来的两条规则都从它旁边走过去。
    token = "3f9c1a04be27d5610872ef4bc93d0a6f5e18720b4dcaf3961e05b8d2740cae63"
    path = tmp_path / "registry-baseline.json"
    path.write_text(f'[\n {{\n  "cluster_id": "hp-a",\n  "token": "{token}"\n }}\n]\n')

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "cleartext credential under key 'token'" in result.stdout


@pytest.mark.parametrize(
    "text",
    [
        # 已脱敏的摘要：键名不同，不该报。
        '{"token_sha256": "692e6286bf78aaaabbbbccccddddeeeeffff0000111122223333444455556666"}',
        # 模板占位符。
        "password = REPLACE_WITH_THE_REAL_DATABASE_PASSWORD\n",
        "GPU_FAULT_EXECUTION_TOKEN=${GPU_FAULT_EXECUTION_TOKEN_FROM_SECURE_DIR}\n",
        # Secret 的键名而不是键值。
        "secretKeyRef:\n  name: gpu-fault-node-action-token-binding\n",
        # 低熵重复串。
        'token: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n',
    ],
)
def test_artifacts_gate_keeps_redacted_and_placeholder_evidence(
    tmp_path: Path, text: str
) -> None:
    (tmp_path / "evidence.yaml").write_text(text)

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stdout


def test_artifacts_gate_scans_nested_readme_files(tmp_path: Path) -> None:
    # 豁免只对 artifacts/README.md 生效：按文件名豁免时，
    # artifacts/perf/<case>/README.md 可以装 Secret 而门禁读不到。
    (tmp_path / "README.md").write_text("# artifacts\n")
    nested = tmp_path / "perf" / "case"
    nested.mkdir(parents=True)
    (nested / "README.md").write_text("kind: Secret\n")

    result = run_check(tmp_path)

    assert result.returncode == 1
    assert "perf/case/README.md" in result.stdout


def test_empty_artifacts_are_reported_and_can_be_required(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# empty\n")

    optional = run_check(tmp_path)
    required = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifacts-root",
            str(tmp_path),
            "--require-content",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert optional.returncode == 0
    assert "no local artifact files" in optional.stdout
    assert required.returncode == 2


def test_artifacts_gate_is_wired_into_quality_checks() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "$(MAKE) artifacts-safety-check" in makefile
    assert "scripts/check-artifacts.py" in workflow
    assert "artifacts/*" in ignore
    assert "!artifacts/README.md" in ignore
