"""AUTH-016 token-rotation sampler: phase attribution and the phase gate."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import identity_auth_sampling as sampling
from scripts.e2e.regional.identity_acceptance_common import IdentityAcceptanceError


class ScriptedStop:
    """The sampler's stop event; every ``wait`` advances the rotation one step."""

    def __init__(self, steps: list[Any]) -> None:
        self.steps = steps
        self.stopped = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.steps:
            self.steps.pop(0)()
        else:
            self.stopped = True
        return self.stopped


def rotated_sampler(statuses: dict[str, int | str]) -> tuple[Any, list[str]]:
    claims: list[str] = []

    def sample(credential: str) -> int | str:
        claims.append(credential)
        return statuses[credential]

    sampler = sampling.TokenRotationSampler("token-old", "token-new", sample)
    sampler.stop = ScriptedStop(
        [
            lambda: sampler.enter("overlap", token="token-new"),
            lambda: sampler.enter("new-only"),
        ]
    )
    return sampler, claims


def test_sampler_attributes_each_claim_to_its_phase_and_credential_slot() -> None:
    sampler, claims = rotated_sampler({"token-old": 200, "token-new": 200})

    sampler.run()

    # baseline: the old token; overlap: both; afterwards: the token entered
    # with the overlap, which stays when a later phase passes no token.
    assert claims == ["token-old", "token-old", "token-new", "token-new"]
    assert [(item["phase"], item["credential_slot"]) for item in sampler.samples] == [
        ("baseline", "old"),
        ("overlap", "old"),
        ("overlap", "new"),
        ("new-only", "new"),
    ]
    assert all(item["observed_at"] for item in sampler.samples), (
        "every claim carries its observation time"
    )
    assert sampler.token == "token-new"
    assert sampler.stop.waits == [2, 2, 2]

    sampler.require_phase("baseline", {"old"})
    sampler.require_phase("overlap", {"old", "new"})
    sampler.require_phase("new-only", {"new"})


@pytest.mark.parametrize(
    ("phase", "slots", "statuses"),
    [
        # No claim was measured in the phase.
        ("revoked", {"old"}, {"token-old": 200, "token-new": 200}),
        # A slot the gate needs is absent from the phase.
        ("baseline", {"old", "new"}, {"token-old": 200, "token-new": 200}),
        # A claim failed (a transport error is a string, not a status).
        ("overlap", {"old", "new"}, {"token-old": 200, "token-new": "timeout"}),
        ("overlap", {"old", "new"}, {"token-old": 401, "token-new": 200}),
    ],
)
def test_require_phase_rejects_missing_failed_or_incomplete_phases(
    phase: str, slots: set[str], statuses: dict[str, int | str]
) -> None:
    sampler, _ = rotated_sampler(statuses)
    sampler.run()

    with pytest.raises(
        IdentityAcceptanceError,
        match=f"rotation {phase} contains failed or missing samples",
    ):
        sampler.require_phase(phase, slots)
