"""Removal target identity under the mutation lock, and resuming past VERIFIED.

``remove_cluster`` re-reads the site once it holds the membership lock and
refuses when the target member changed in between. A removal interrupted
after its VERIFIED step is journaled but before the COMPLETED write resumes
without verifying again: every step is already recorded as done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin._cluster_removal_support import RemovalScenario


def test_target_changed_before_the_mutation_lock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = RemovalScenario(tmp_path, monkeypatch)
    document = yaml.safe_load(outer.path.read_text())
    document["spec"]["clusters"][0]["allowedNamespaces"] = ["training", "other"]
    changed = tmp_path / "changed-site.yaml"
    changed.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    changed.chmod(0o600)
    reloaded = load_site(changed)
    monkeypatch.setattr(removal, "reload_site_for_mutation", lambda _site: reloaded)
    with pytest.raises(BootstrapError, match="identity changed before mutation lock"):
        removal.remove_cluster(outer.request())
    assert outer.calls == []
    assert outer.path.read_bytes() == outer.original


def test_removal_interrupted_after_verified_resumes_without_reverifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = RemovalScenario(tmp_path, monkeypatch)
    original_write = removal.write_json_atomic
    interrupted: list[str] = []

    def write(path: Path, value: dict[str, Any]) -> None:
        if value.get("phase") == "COMPLETED" and not interrupted:
            interrupted.append(str(path))
            raise OSError("journal unavailable while writing COMPLETED")
        original_write(path, value)

    monkeypatch.setattr(removal, "write_json_atomic", write)
    with pytest.raises(OSError, match="journal unavailable"):
        removal.remove_cluster(outer.request())
    state = outer.state()
    assert state["phase"] != "COMPLETED"
    assert "VERIFIED" in state["completed_steps"]
    assert outer.calls.count("verify") == 1
    result = removal.remove_cluster(outer.request())
    assert result["phase"] == "COMPLETED"
    assert outer.state()["phase"] == "COMPLETED"
    assert outer.calls.count("verify") == 1
    assert outer.calls.count("cleanup") == 1
