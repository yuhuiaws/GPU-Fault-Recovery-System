from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.policy import load_xid_policy
from gpu_fault.policy.catalog_integrity import (
    validate_xid_catalog_document,
    xid_catalog_artifact_sha256,
)
from tools.generate_nvidia_xid_policy import HEADER, check_generated

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml"


def test_generated_catalog_integrity_is_self_consistent() -> None:
    document = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))

    digest = validate_xid_catalog_document(document)

    assert digest == document["metadata"]["generatedSha256"]
    assert digest == xid_catalog_artifact_sha256(document)
    check_generated(CATALOG)


def test_runtime_rejects_a_rule_edit_with_the_old_artifact_digest(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    document["spec"]["catalogRules"][0]["immediateAction"] = "IGNORE"
    mutated = tmp_path / "mutated-xid-catalog.yaml"
    mutated.write_text(
        HEADER
        + yaml.safe_dump(document, sort_keys=False, allow_unicode=False, width=1000),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        load_xid_policy(mutated)


def test_check_rejects_noncanonical_generated_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "changed-format.yaml"
    changed.write_text(
        CATALOG.read_text(encoding="utf-8") + "# stray edit\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="canonical generator format"):
        check_generated(changed)
