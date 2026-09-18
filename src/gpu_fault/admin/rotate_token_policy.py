"""Validate and bind the security policy of a resumable token rotation."""

from __future__ import annotations

from datetime import timedelta
from typing import Mapping, Protocol

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.regional import MAX_TOKEN_ROTATION_WINDOW

MIN_WINDOW_MINUTES = 10
MAX_WINDOW_MINUTES = int(MAX_TOKEN_ROTATION_WINDOW.total_seconds() // 60)


class RotationPolicyRequest(Protocol):
    @property
    def window(self) -> timedelta: ...

    @property
    def keep_window(self) -> bool: ...

    @property
    def quiet_seconds(self) -> int: ...

    @property
    def acceptance_timeout_seconds(self) -> int: ...


def rotation_policy(request: RotationPolicyRequest) -> dict[str, int | bool]:
    minutes = request.window.total_seconds() / 60
    if (
        minutes < MIN_WINDOW_MINUTES
        or request.window > MAX_TOKEN_ROTATION_WINDOW
        or not minutes.is_integer()
    ):
        raise BootstrapError(
            "rotate-token window must be between "
            f"{MIN_WINDOW_MINUTES} and {MAX_WINDOW_MINUTES} whole minutes"
        )
    if type(request.keep_window) is not bool:
        raise BootstrapError("rotate-token keep-window policy must be boolean")
    if (
        type(request.quiet_seconds) is not int
        or type(request.acceptance_timeout_seconds) is not int
        or not 0 < request.quiet_seconds < request.acceptance_timeout_seconds
    ):
        raise BootstrapError(
            "rotate-token quiet period must be positive and shorter than "
            "the acceptance timeout"
        )
    return {
        "window_minutes": int(minutes),
        "keep_window": request.keep_window,
        "quiet_seconds": request.quiet_seconds,
        "acceptance_timeout_seconds": request.acceptance_timeout_seconds,
    }


def require_rotation_policy(
    state: Mapping[str, object], expected: Mapping[str, int | bool]
) -> None:
    for key, value in expected.items():
        if key not in state:
            raise BootstrapError(f"rotate-token resume policy is unbound: {key}")
        if type(state[key]) is not type(value) or state[key] != value:
            raise BootstrapError(f"rotate-token resume policy conflicts on {key}")
