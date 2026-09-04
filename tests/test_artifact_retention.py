from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/prune_artifacts.py"
MODULE = lazy_script_module(MODULE_PATH)


def test_retention_policy_has_an_entry_point() -> None:
    # 这个脚本此前只被本文件 import，Makefile、CI 和文档里一处入口都没有：
    # 保留策略写在 artifacts/README.md 里，但没人能跑它。
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (ROOT / "artifacts/README.md").read_text(encoding="utf-8")

    assert "artifacts-retention:" in makefile
    assert makefile.count("artifacts-retention") >= 2, "target 必须同时进 .PHONY"
    recipe = makefile.split("artifacts-retention:", 1)[1].splitlines()[1]
    assert "scripts/prune_artifacts.py" in recipe
    assert "--apply" not in recipe, "Makefile 入口必须是 dry run"
    assert "make artifacts-retention" in readme


def make_run(root: Path, name: str) -> Path:
    run = root / "perf" / "burst" / "release-a" / name
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"status":"ok"}\n')
    return run


def test_retention_keeps_latest_and_documented_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    docs = tmp_path / "docs"
    docs.mkdir()
    runs = [make_run(artifacts, f"2026080{day}T000000Z") for day in range(1, 5)]
    (docs / "evidence.md").write_text(
        "`artifacts/perf/burst/release-a/20260801T000000Z/summary.json`\n"
    )

    candidates = MODULE.retention_candidates(
        artifacts,
        docs,
        keep=2,
        aborted_days=7,
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )

    assert runs[0] not in candidates
    assert runs[1] in candidates
    assert runs[2] not in candidates
    assert runs[3] not in candidates


def test_retention_keeps_run_promoted_into_the_evidence_library(tmp_path: Path) -> None:
    # 真实仓库里没有任何文档写 ``artifacts/perf/...``；被引用的是提升后的扁平
    # 副本 ``docs/evidence/perf/<case>-<utc>/``。只认原始路径时白名单恒为空，
    # 这条保护在生产上等于不存在，而单元测试自己造的文档掩盖了这一点。
    artifacts = tmp_path / "artifacts"
    docs = tmp_path / "docs"
    docs.mkdir()
    runs = [make_run(artifacts, f"2026080{day}T000000Z") for day in range(1, 5)]
    (docs / "evidence.md").write_text(
        "见 `docs/evidence/perf/burst-release-a-20260801T000000Z/summary.json`\n"
    )

    candidates = MODULE.retention_candidates(
        artifacts,
        docs,
        keep=2,
        aborted_days=7,
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )

    assert runs[0] not in candidates
    assert runs[1] in candidates


def test_report_counts_protected_runs_not_whitelist_entries(tmp_path: Path) -> None:
    # 本仓真实情况：docs/security/credential-exposure-20260823.md 里写了
    # ``artifacts/perf/**/registry-baseline.json``，它进白名单但一个轮次都
    # 保护不了。按白名单条数报数会显示「有保护」，实际是 0。
    artifacts = tmp_path / "artifacts"
    docs = tmp_path / "docs"
    docs.mkdir()
    make_run(artifacts, "20260801T000000Z")
    make_run(artifacts, "20260802T000000Z")
    (docs / "incident.md").write_text(
        "泄露面是 `artifacts/perf/**/registry-baseline.json`。\n"
    )

    references = MODULE.referenced_patterns(docs)
    report = MODULE.retention_report(
        artifacts,
        docs,
        keep=1,
        aborted_days=7,
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )

    assert len(references) == 1
    assert report.protected == []
    assert len(report.candidates) == 1


def test_reference_scan_tolerates_documents_without_references(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "plain.md").write_text("没有引用任何一轮压测。\n")

    assert len(MODULE.referenced_patterns(docs)) == 0
    assert len(MODULE.referenced_patterns(tmp_path / "missing")) == 0


def test_missing_artifacts_tree_yields_no_candidates(tmp_path: Path) -> None:
    # artifacts/ 在 .gitignore 里，新克隆和 CI 上不存在；加了 Makefile 入口
    # 之后这条路径必须返回空而不是 FileNotFoundError。
    docs = tmp_path / "docs"
    docs.mkdir()

    assert (
        MODULE.retention_candidates(
            tmp_path / "artifacts",
            docs,
            keep=5,
            aborted_days=7,
            now=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )
        == []
    )


def test_old_aborted_run_is_candidate(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    docs = tmp_path / "docs"
    docs.mkdir()
    run = artifacts / "perf" / "_aborted" / "burst" / "release-a" / "20260801T000000Z"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"status":"aborted"}\n')

    candidates = MODULE.retention_candidates(
        artifacts,
        docs,
        keep=5,
        aborted_days=7,
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
    )

    assert candidates == [run]
