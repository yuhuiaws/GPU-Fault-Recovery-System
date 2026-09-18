"""Independent cleanup attempts for HA fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import TypeVar

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost

T = TypeVar("T")


def record_supervision_loss(result: dict[str, Any]) -> None:
    result.update(
        verdict="FAIL",
        supervision_lost=True,
        cleanup_preserved="command completion is unproven; no further remote commands issued",
    )


def run_cleanup(result: dict[str, Any], action: Callable[[], T]) -> T | None:
    if result.get("supervision_lost") is True:
        return None
    try:
        return action()
    except ProcessSupervisionLost:
        record_supervision_loss(result)
        return None


def attempt_cleanup(
    result: dict[str, Any], label: str, action: Callable[[], Any]
) -> bool:
    try:
        action()
    except Exception as exc:
        result["verdict"] = "FAIL"
        result.setdefault("cleanup_errors", []).append(
            f"{label}: {type(exc).__name__}: {exc}"
        )
        return False
    return True
