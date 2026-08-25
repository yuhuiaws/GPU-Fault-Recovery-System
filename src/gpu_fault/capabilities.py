from __future__ import annotations

from collections import Counter

from gpu_fault.models import (
    DESTRUCTIVE_CAPABILITIES,
    CapabilityMode,
    EffectiveCapability,
    EffectiveRuntimeProfile,
    RuntimeProfile,
)


class ProfileValidationError(ValueError):
    pass


def compile_runtime_profile(
    profile: RuntimeProfile,
) -> EffectiveRuntimeProfile:
    active_claims = [
        claim for claim in profile.claims if claim.mode is not CapabilityMode.DISABLED
    ]
    claim_counts = Counter(claim.capability for claim in active_claims)
    duplicate_destructive = sorted(
        capability.value
        for capability, count in claim_counts.items()
        if count > 1 and capability in DESTRUCTIVE_CAPABILITIES
    )
    if duplicate_destructive:
        raise ProfileValidationError(
            "multiple active writers for destructive capabilities: "
            + ", ".join(duplicate_destructive)
        )

    observed = {(item.capability, item.owner): item for item in profile.observed}
    effective: list[EffectiveCapability] = []
    warnings: list[str] = []

    for claim in active_claims:
        if (
            claim.capability in DESTRUCTIVE_CAPABILITIES
            and claim.mode is CapabilityMode.AUGMENT
        ):
            raise ProfileValidationError(
                f"{claim.capability.value} cannot use AUGMENT; "
                "destructive capabilities require a single writer"
            )
        if (
            claim.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}
            and not claim.adapter
        ):
            raise ProfileValidationError(
                f"{claim.capability.value} in {claim.mode.value} mode "
                "requires an adapter"
            )

        fact = observed.get((claim.capability, claim.owner))
        if claim.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}:
            if fact is None or not fact.available:
                warnings.append(
                    f"{claim.capability.value}/{claim.owner} unavailable; "
                    "effective mode reduced to OBSERVE"
                )
                mode = CapabilityMode.OBSERVE
            else:
                mode = claim.mode
        else:
            mode = claim.mode

        effective.append(
            EffectiveCapability(
                capability=claim.capability,
                mode=mode,
                owner=claim.owner,
                adapter=claim.adapter,
                observed_version=fact.version if fact else None,
            )
        )

    return EffectiveRuntimeProfile(
        cluster_id=profile.cluster_id,
        environment=profile.environment,
        profile_version=profile.profile_version,
        capabilities=effective,
        warnings=warnings,
    )
