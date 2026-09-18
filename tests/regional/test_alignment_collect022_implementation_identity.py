"""Deployed reader identity covers extracted helpers, not just the import facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.collectors.logs import (
    fabric_manager,
    fabric_manager_cursor,
    fabric_manager_receipts,
)
from scripts.e2e.regional.probes import collect022_fm_cursor_probe as probe


@pytest.mark.parametrize(
    "changed_module",
    [fabric_manager, fabric_manager_cursor, fabric_manager_receipts],
    ids=["collector", "cursor", "receipts"],
)
def test_deployed_reader_fingerprint_changes_for_each_implementation_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_module
) -> None:
    prefix = tmp_path / "private-runtime"
    prefix.mkdir()
    monkeypatch.setattr(probe, "NODE_PREFIX", prefix)
    monkeypatch.setattr(probe.sys, "prefix", str(prefix))
    paths = {}
    for index, module in enumerate(
        (fabric_manager, fabric_manager_cursor, fabric_manager_receipts)
    ):
        path = prefix / f"component-{index}.py"
        path.write_text(f"# private component {index}\n")
        paths[module.__name__] = path
        monkeypatch.setattr(module, "__file__", str(path))
    original = probe.deployed_identity()
    assert probe.deployed_identity() == original, (
        "unchanged implementation must be stable"
    )
    paths[changed_module.__name__].write_text("# modified private implementation\n")
    updated = probe.deployed_identity()
    assert updated["collector_module_sha256"] != original["collector_module_sha256"], (
        "a changed reader dependency must invalidate the deployed fingerprint"
    )
    assert updated["python_prefix"] == original["python_prefix"]
    assert updated["package_version"] == original["package_version"]
