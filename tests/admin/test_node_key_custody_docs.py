from __future__ import annotations

from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]


def test_custody_component_document_has_valid_code_references_and_links(monkeypatch):
    document = ROOT / "docs/components/node-key-custody-evidence.md"
    references = lazy_script_module(ROOT / "scripts/check-doc-references.py")
    anchors = lazy_script_module(ROOT / "scripts/check-doc-anchors.py")
    for module in (references, anchors):
        monkeypatch.setattr(module, "documents", lambda: [document])
    assert references.check_references() == []
    assert anchors.check_anchors() == []
