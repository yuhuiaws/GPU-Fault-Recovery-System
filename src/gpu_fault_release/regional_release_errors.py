"""The release engine's error hierarchy.

Imported back into ``regional_release_config`` (where it was defined) so every
existing ``from regional_release_config import ReleaseError`` keeps working;
a module that must not depend on the config module (one the config module
itself imports) takes it from here.
"""

from __future__ import annotations

from typing import Any


class ReleaseError(RuntimeError):
    pass


class ClusterLocalReleaseError(ReleaseError):
    pass


class WaveAgentSafetyError(ClusterLocalReleaseError):
    """A fleet wave's safety gate timed out on unhealthy Agents outside the wave.

    Carries the probe's blockers so the forward path can tell an Agent whose
    node left (``regional_release_departure_absorption``) from one that is down.
    """

    def __init__(
        self, message: str, *, blockers: list[Any], blocker_count: int
    ) -> None:
        super().__init__(message)
        self.blockers = list(blockers)
        self.blocker_count = blocker_count


class WaveWorkflowSafetyError(ReleaseError):
    """A fleet wave's safety gate refused on active destructive workflows.

    A plain release failure, as before; it carries the gate's evidence and the
    request ids it did not set aside, so the forward path can wait for a
    record of a node it proved departed to age past the product's dispatch
    guard (``regional_release_departure_absorption``).
    """

    def __init__(
        self,
        message: str,
        *,
        snapshot: dict[str, Any],
        remaining: list[str],
        remaining_count: int,
    ) -> None:
        super().__init__(message)
        self.snapshot = snapshot
        self.remaining = list(remaining)
        self.remaining_count = remaining_count


class PartialClusterRolloutError(ReleaseError):
    pass


class CheckFailure(ReleaseError):
    """A refused check that keeps the evidence it was refused on.

    A plain ``ReleaseError`` is recorded as a summary with no details, which is
    right for a refusal whose message is the whole finding. A judgement made
    over a snapshot -- which node, which collector, how stale -- carries the
    snapshot here, so the FAIL record shows what was seen and not only what was
    concluded.
    """

    def __init__(self, message: str, details: Any) -> None:
        super().__init__(message)
        self.details = details

    @staticmethod
    def details_of(error: BaseException) -> Any:
        """The evidence a failed check carries; ``None`` for every other error."""

        return error.details if isinstance(error, CheckFailure) else None
