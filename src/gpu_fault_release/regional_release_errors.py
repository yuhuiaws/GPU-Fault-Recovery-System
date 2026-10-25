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
