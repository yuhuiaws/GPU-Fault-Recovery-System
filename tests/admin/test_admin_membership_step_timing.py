"""Join and removal state records when each step completed."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_fault.admin.cluster_join_state import complete_step


def test_each_completed_step_keeps_its_own_timestamp(tmp_path: Path) -> None:
    """``updated_at`` only says when the record last moved; an operator asking
    where a join spent its time needs one timestamp per step."""

    path = tmp_path / "state.json"
    state: dict[str, object] = {"completed_steps": [], "evidence": {}}

    complete_step(path, state, "DISCOVERED", {"target": {}})
    first = dict(state["step_completed_at"])  # type: ignore[call-overload]
    complete_step(path, state, "PRECHECKED")

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(stored["step_completed_at"]) == ["DISCOVERED", "PRECHECKED"], (
        "every completed step must carry a timestamp"
    )
    assert stored["step_completed_at"]["DISCOVERED"] == first["DISCOVERED"], (
        "completing a later step must not rewrite an earlier step's timestamp"
    )
    assert stored["updated_at"] == stored["step_completed_at"]["PRECHECKED"], (
        "the record's updated_at is the latest step's timestamp"
    )
