from __future__ import annotations

import re
from pathlib import Path

from tests.regional.test_regional_execution_order import CASE_HEADING, CHAPTER_HEADING

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs/区域模式端到端验收测试用例.md"
HISTORY = ROOT / "docs/evidence/regional-history/区域模式验收历史与覆盖审计.md"
EVIDENCE_README = ROOT / "docs/evidence/README.md"
SECTION_HEADING = re.compile(r"^(#{2,6})\s+(.+?)\s*#*$")
# "这一段是当时跑出来的结果"的写法词表。规格（步骤/判定/前置/执行方案/
# 跑前核实）留在手册里，这些留不住——它们只在回答"那一次跑成什么样"。
RECORD_TITLE = re.compile(
    "|".join(
        (
            "执行记录",
            "执行结论",
            "实测结果",
            "复验记录",
            "真机记录",
            "最终判定",
            "附带发现",
            "缺陷",
            "根因与修复",
            "根治记录",
            "由此暴露",
            "本轮发现",
            "修复后立即暴露",
            "实跑踩到",
            "新发现的",
            "全绿证据",
            "尝试",
            "已修复",
            "已补齐",
        )
    )
)


def _title_subject(title: str) -> str:
    """标题去掉括注后的主语。

    判据只看主语，因为括注里常常提到别的东西：
    `跑前必须离线核实（否则会卡在中途，且极易误判成缺陷）` 是规格，
    只是顺口提了「缺陷」二字。反过来没有损失——真正的记录段
    （`缺陷 A（本轮发现，已修）`、`第 1 轮执行记录（…）`）
    主语里就已经写明自己是记录。
    """
    for opening in ("（", "("):
        title = title.split(opening, 1)[0]
    return title


def _case_chapter_headings() -> list[tuple[int, str, str]]:
    """(行号, 层级, 标题) —— 只取用例章节里的小节标题。"""
    chapter: str | None = None
    result = []
    for index, line in enumerate(
        DOCUMENT.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith("## "):
            heading = CHAPTER_HEADING.match(line)
            chapter = heading.group(1) if heading else None
            continue
        if chapter is None:
            continue
        match = SECTION_HEADING.match(line)
        if match is None or CASE_HEADING.match(line):
            continue
        result.append((index, match.group(1), match.group(2)))
    return result


def test_case_chapters_carry_no_execution_records() -> None:
    # 规格与执行史分开：用例章节只写"要跑什么、怎么判"，
    # 脱敏历史叙述写进 docs/evidence/regional-history，原始证据留在私有库。
    # 挡的是实测过的可读性问题：`COLLECT-016` 的步骤与判定共 150 行，
    # 底下压着 800 行逐轮记录与缺陷根治，执行人翻不到自己要的那一段；
    # `DESTR-008` 更极端，660 行里只有约 60 行是规格。
    offending = [
        f"{index}: {'#' * len(level)} {title}"
        for index, level, title in _case_chapter_headings()
        if RECORD_TITLE.search(_title_subject(title))
    ]

    assert offending == []


def test_public_spec_has_no_internal_history_links() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert "区域分离部署验收执行史.md" not in text
    assert "internal-docs/" not in text


def test_public_spec_has_no_embedded_history_sections_or_results() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    forbidden = (
        "## 18.",
        "## 19.",
        "### 15.1",
        "§18",
        "§19",
        "执行史",
        "历史执行稿",
        "当前状态：**PASS",
        "实测结果",
        "现场结果",
        "修复状态（",
        "<gpu-node-",
        "<old-",
        "<new-",
    )
    assert [value for value in forbidden if value in text] == []
    assert re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text) == []


def test_redacted_history_is_separate_and_non_authoritative() -> None:
    public = DOCUMENT.read_text(encoding="utf-8")
    history = HISTORY.read_text(encoding="utf-8")
    evidence_boundary = EVIDENCE_README.read_text(encoding="utf-8")

    assert "evidence/regional-history/区域模式验收历史与覆盖审计.md" in public
    assert "不是当前操作步骤" in history
    assert "不构成当前 release 的" in history
    assert "PASS 证据" in history
    assert "testcases/fault-scenarios.yaml" in history
    assert "## 18. 改进 backlog" in history
    assert "## 19. 2026-08-18 测试覆盖审计" in history
    assert "`regional-history/`" in evidence_boundary
    assert "不是当前步骤或当前 PASS 证据" in evidence_boundary
