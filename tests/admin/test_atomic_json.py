"""What every caller of the shared atomic JSON writer is entitled to assume.

Six near-copies of this writer were merged into one, and each copy was the strict one
in some axis and the lax one in another. These tests pin the strict union, so a later
simplification cannot quietly reintroduce a weaker variant: an inherited directory
gets its mode enforced, the document is never left half-written, and a failed write
leaves neither a partial target nor a stray temporary file.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_document_is_sorted_indented_and_newline_terminated(tmp_path: Path) -> None:
    """The bytes are a stable diff target: operators review these files by eye."""

    path = tmp_path / "state" / "plan.json"

    write_json_atomic(path, {"b": 2, "a": {"d": 4, "c": 3}})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "c": 3,\n    "d": 4\n  },\n  "b": 2\n}\n'
    )


def test_new_parent_and_file_are_private(tmp_path: Path) -> None:
    path = tmp_path / "state" / "approval.json"

    write_json_atomic(path, {"reference": "CHG-1"})

    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


def test_inherited_parent_directory_mode_is_enforced(tmp_path: Path) -> None:
    """`mkdir(mode=...)` is a no-op on a directory that already exists.

    Three of the merged copies relied on that keyword alone, so a state directory
    created by an earlier tool -- or by an umask nobody checked -- stayed
    group-readable while holding approval records and cluster identities.
    """

    parent = tmp_path / "state"
    parent.mkdir(mode=0o755)
    path = parent / "desired.json"

    write_json_atomic(path, {"cluster_id": "gpu-a"})

    assert _mode(parent) == 0o700


def test_replacing_an_existing_document_keeps_the_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json_atomic(path, {"generation": 1})

    write_json_atomic(path, {"generation": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 2}
    assert _mode(path) == 0o600


def test_a_failed_write_leaves_the_previous_document_and_no_temporary(
    tmp_path: Path,
) -> None:
    """A document that cannot be serialised must not consume the target.

    This is the property that makes the writer safe to call on a file another command
    will read as authoritative: a plan that fails to render leaves the approved plan
    in place rather than truncating it.
    """

    path = tmp_path / "state" / "plan.json"
    write_json_atomic(path, {"generation": 1})

    with pytest.raises(TypeError):
        write_json_atomic(path, {"generation": object()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 1}
    assert sorted(item.name for item in path.parent.iterdir()) == ["plan.json"], (
        "the temporary file must be removed even when the write fails"
    )
