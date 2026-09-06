"""``active_consumers`` is a decision, not a default.

FINAL-建议汇总 F-D9 (P2-14C). The deployed processor only ever runs
active-active; the ``False`` default silently selected the leader-mode claim
fork -- the one whose release path lacked the completed-row guard -- for any
caller that forgot the keyword.
"""

from __future__ import annotations

import pytest

from gpu_fault.processor import ProcessorCoordinator
from tests._builders import build_store


def test_active_consumers_is_required() -> None:
    with pytest.raises(TypeError, match="active_consumers"):
        ProcessorCoordinator(
            build_store(), owner_id="pod-a:1", internal_token="token-" + "x" * 32
        )


def test_active_consumers_is_honoured_when_given() -> None:
    processor = ProcessorCoordinator(
        build_store(),
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        active_consumers=True,
    )
    assert processor.active_consumers is True, "the explicit value must be kept"
