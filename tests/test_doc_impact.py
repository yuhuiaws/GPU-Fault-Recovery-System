from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-doc-impact.py"


CONTRACTS = dedent(
    """\
    version: 1
    coverage:
      - root: src
        extensions: [.py]
      - root: tests
        extensions: [.py]
      - root: scripts
        extensions: [.sh]
    contracts:
      - id: source-python
        paths: [src/**/*.py]
        documents: [docs/design.md]
      - id: test-python
        paths: [tests/**/*.py]
        documents: [docs/tests.md]
      - id: operational-shell
        paths: [scripts/**/*.sh]
        documents: [docs/operations.md]
    """
)


def make_repository(tmp_path: Path) -> Path:
    for relative, content in {
        "src/package/runtime.py": "VALUE = 1\n",
        "tests/test_runtime.py": "def test_runtime():\n    assert True\n",
        "scripts/operate.sh": "#!/bin/sh\nexit 0\n",
        "docs/design.md": "# Design\n",
        "docs/tests.md": "# Tests\n",
        "docs/operations.md": "# Operations\n",
        "docs/code-doc-contracts.yaml": CONTRACTS,
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def run_check(
    root: Path, *arguments: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    child_environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GPU_FAULT_DOC_IMPACT", "GPU_FAULT_DOC_IMPACT_REASON"}
    }
    child_environment.update(environment or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *arguments],
        cwd=ROOT,
        env=child_environment,
        text=True,
        capture_output=True,
        check=False,
    )


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_repository_documentation_contract_is_complete() -> None:
    result = run_check(ROOT, "--validate-only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "contract structure is valid" in result.stdout


def test_related_document_change_satisfies_contract(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    result = run_check(
        repository,
        "--changed-file",
        "src/package/runtime.py",
        "--changed-file",
        "docs/design.md",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "source-python" in result.stdout


def test_code_change_without_documentation_or_reason_fails(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    result = run_check(repository, "--changed-file", "tests/test_runtime.py")

    assert result.returncode == 1
    assert "test-python" in result.stdout
    assert "Documentation-Impact: none" in result.stdout
    assert "docs/tests.md" in result.stdout


def test_explicit_no_documentation_impact_requires_a_real_reason(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)

    rejected = run_check(
        repository,
        "--changed-file",
        "scripts/operate.sh",
        "--acknowledge-no-docs",
        "--reason",
        "TODO",
    )
    accepted = run_check(
        repository,
        "--changed-file",
        "scripts/operate.sh",
        "--acknowledge-no-docs",
        "--reason",
        "Only internal shell refactoring; its CLI is unchanged.",
    )

    assert rejected.returncode == 1
    assert "cannot be a placeholder" in rejected.stdout
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "explicit no-docs acknowledgement" in accepted.stdout


def test_pull_request_body_can_acknowledge_no_documentation_impact(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    event = repository / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": (
                        "Documentation-Impact: none\n"
                        "Documentation-Impact-Reason: "
                        "The public behavior and commands are unchanged.\n"
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_check(
        repository,
        "--changed-file",
        "src/package/runtime.py",
        "--event-path",
        str(event),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit no-docs acknowledgement" in result.stdout


def test_local_environment_can_acknowledge_no_documentation_impact(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)

    result = run_check(
        repository,
        "--changed-file",
        "tests/test_runtime.py",
        environment={
            "GPU_FAULT_DOC_IMPACT": "none",
            "GPU_FAULT_DOC_IMPACT_REASON": (
                "Only internal test organization changed; nodeids are stable."
            ),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit no-docs acknowledgement" in result.stdout


def test_git_base_diff_requires_a_related_document_change(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    git(repository, "init")
    git(repository, "config", "user.email", "tests@example.invalid")
    git(repository, "config", "user.name", "Documentation Contract Tests")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "initial")
    base = git(repository, "rev-parse", "HEAD")

    runtime = repository / "src" / "package" / "runtime.py"
    runtime.write_text("VALUE = 2\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "change runtime")

    rejected = run_check(repository, "--base", base, "--head", "HEAD")

    design = repository / "docs" / "design.md"
    design.write_text("# Design\n\nRuntime value changed.\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "update documentation")
    accepted = run_check(repository, "--base", base, "--head", "HEAD")

    assert rejected.returncode == 1
    assert "source-python" in rejected.stdout
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "source-python" in accepted.stdout


def test_contract_validation_rejects_uncovered_files(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    tool = repository / "tools" / "uncovered.py"
    tool.parent.mkdir()
    tool.write_text("VALUE = 1\n", encoding="utf-8")
    contract = repository / "docs" / "code-doc-contracts.yaml"
    contract.write_text(
        CONTRACTS.replace(
            "contracts:\n", "  - root: tools\n    extensions: [.py]\ncontracts:\n"
        ),
        encoding="utf-8",
    )

    result = run_check(repository, "--validate-only")

    assert result.returncode == 1
    assert "documentation contract does not cover: tools/uncovered.py" in result.stdout


def test_invalid_changed_file_inputs_fail_without_a_traceback(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    missing_list = run_check(
        repository, "--changed-files-from", str(repository / "missing.txt")
    )
    escaping_path = run_check(repository, "--changed-file", "../outside.py")

    assert missing_list.returncode == 1
    assert "cannot read changed-files list" in missing_list.stdout
    assert "Traceback" not in missing_list.stderr
    assert escaping_path.returncode == 1
    assert "must be repository-relative" in escaping_path.stdout
    assert "Traceback" not in escaping_path.stderr


def test_documentation_impact_gate_is_wired_into_local_and_ci_checks() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    template = (ROOT / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    assert "$(MAKE) docs-check" in makefile
    assert "doc-impact-check:" in makefile
    assert "scripts/check-doc-impact.py" in workflow
    assert "fetch-depth: 0" in workflow
    assert "--base" in workflow
    assert "Documentation-Impact: choose updated or none" in template
    assert "Documentation-Impact-Reason:" in template
