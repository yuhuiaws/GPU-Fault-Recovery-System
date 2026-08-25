from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs/部署和运维手册.md"
GUIDE = ROOT / "docs/部署和运维手册逐章解读.md"


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
