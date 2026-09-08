from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-public-release.py"


def run_check(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def public_tree(tmp_path: Path, content: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    (ignored / "live.md").write_text("node i-0123456789abcdef0\n", encoding="utf-8")

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


# These are positive-case fixtures: each value must trip a check to prove the
# gate fires, so they cannot be allowlisted placeholders. They are nonetheless
# written as obviously-synthetic values (all-zero accounts/UUIDs, sequential
# hex ids) rather than real-looking site identities, so that even though this
# file is exempt from the scan, no plausible live identifier is embedded here.
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "arn:aws:eks:us-west-2:999999999999:cluster/example\n",
            "customer AWS account",
        ),
        ("node i-0123456789abcdef0\n", "concrete EC2 instance ID"),
        ("network vpc-0123456789abcdef0\n", "concrete AWS resource ID"),
        ("endpoint 10.91.0.1\n", "site private address"),
        ("cluster eks-cluster-hypd-example\n", "site-specific resource name"),
        ("email operator@amazon.com\n", "personal email"),
        (
            "boot 00000000-0000-0000-0000-000000000000\n",
            "raw UUID in public documentation",
        ),
        ("node 0123456789abcdef0\n", "raw node identity in public documentation"),
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
    (internal / "live.md").write_text("node i-0123456789abcdef0\n", encoding="utf-8")
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


# Comfortably more than one shard's worth of bytes, so the scan has to cut the
# file up and stitch the line numbers back together.
SHARDED_FILE_LINES = 2000
SHARDED_FILE_LINE = "node i-0123456789abcdef0 " + "x" * 700


def sharded_tree(tmp_path: Path, *, filler_files: int) -> Path:
    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    (tmp_path / "big.txt").write_text(
        "".join(f"{SHARDED_FILE_LINE}\n" for _ in range(SHARDED_FILE_LINES)),
        encoding="utf-8",
    )
    for index in range(filler_files):
        (tmp_path / f"filler-{index}.txt").write_text(
            "public documentation\n", encoding="utf-8"
        )
    return tmp_path


def reported_lines(stderr: str) -> list[int]:
    return [
        int(line.split(":")[1])
        for line in stderr.splitlines()
        if line.startswith("- big.txt:")
    ]


@pytest.mark.parametrize("filler_files", [0, 64])
def test_public_release_gate_reports_true_lines_across_shards(
    tmp_path: Path, filler_files: int
) -> None:
    """Sharding a large file must not move the line a violation is reported on.

    Every line of the file is a violation, so whichever offsets the scan cuts
    on, a shard that lost its place would show up as a wrong number here. The
    filler files decide whether the scan fans out across processes, and both
    paths have to agree with the file on disk.
    """

    result = run_check(sharded_tree(tmp_path, filler_files=filler_files))

    assert result.returncode == 1
    assert reported_lines(result.stderr) == list(range(1, SHARDED_FILE_LINES + 1))


def test_public_release_verdict_cache_reuses_only_identical_content(
    tmp_path: Path,
) -> None:
    """A recorded pass is reused for the same bytes and dropped for any other.

    This is what lets a deploy scan the prepared snapshot without matching it a
    second time, so the invalidation is the part that matters: a tree that gained
    a live identity after the pass was recorded must still be rejected.
    """

    tree = public_tree(tmp_path / "tree", "public documentation\n")
    cache = tmp_path / "verdict.json"

    first = run_check(tree, "--verdict-cache", str(cache))
    second = run_check(tree, "--verdict-cache", str(cache))

    assert first.returncode == 0, first.stdout + first.stderr
    assert "scanned" in first.stdout
    assert second.returncode == 0, second.stdout + second.stderr
    assert "reused" in second.stdout

    (tree / "docs/public.md").write_text("node i-0123456789abcdef0\n", encoding="utf-8")
    third = run_check(tree, "--verdict-cache", str(cache))

    assert third.returncode == 1
    assert "concrete EC2 instance ID" in third.stderr


def test_public_release_verdict_cache_records_nothing_on_failure(
    tmp_path: Path,
) -> None:
    tree = public_tree(tmp_path / "tree", "node i-0123456789abcdef0\n")
    cache = tmp_path / "verdict.json"

    result = run_check(tree, "--verdict-cache", str(cache))

    assert result.returncode == 1
    assert not cache.exists(), "a failing scan must not record a reusable pass"


def test_public_release_gate_is_wired_into_local_and_ci_checks() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "$(MAKE) public-release-check" in makefile
    assert "scripts/check-public-release.py" in workflow


@pytest.mark.parametrize(
    "scanner_relative_path",
    [
        "scripts/check-public-release.py",
        "tests/test_fault_scenario_catalog.py",
        "tests/test_script_assets.py",
    ],
)
def test_public_release_gate_scans_its_own_scanner_files(
    tmp_path: Path, scanner_relative_path: str
) -> None:
    """The scanner and its clean sibling tests are no longer self-exempt.

    A live identity that lands in the scanner itself, or in an identity-facing
    test that has no legitimate reason to embed one, must be caught like any
    other file. Only the two files that cannot be scanned clean -- this test's
    own positive fixtures and the documentation-contract patterns -- stay
    exempt, so this asserts every other former exemption is gone.
    """

    (tmp_path / ".gitignore").write_text("", encoding="utf-8")
    target = tmp_path / scanner_relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("leaked node i-0123456789abcdef0\n", encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{scanner_relative_path}:1: concrete EC2 instance ID" in result.stderr
