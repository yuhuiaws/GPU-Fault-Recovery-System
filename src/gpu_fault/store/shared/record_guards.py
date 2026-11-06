"""Version guards for the whole-row writers without a version field.

``save_workflow`` (store review 2026-09-07, item B) refuses to overwrite a row
whose ``merge_revision`` / ``execution_epoch`` / ``fencing_token`` moved since
the caller read it. ``save_incident`` and ``save_plan`` were the blind
whole-row writes left over (architecture review 2026-09-07, items D1 and D2):
``FaultIncident`` and ``RecoveryPlan`` carry no version field, so the
coordinator's read-append-save of ``reasons`` and the dispatcher's plan status
mirror overwrote whatever landed in between. This module holds the guard the
three backends share so the refusal reads the same everywhere -- and the one
``register_profile`` applies to a Runtime Profile version, which is immutable
once written.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from gpu_fault.models import EffectiveRuntimeProfile, FaultIncident
from gpu_fault.store.shared.errors import RuntimeProfileConflict, StaleWriteError

# ``fencing_token`` is the incident's generation and only ever moves forward,
# so a caller whose copy carries a *lower* token than the row read the row
# before a generation change and would write the old generation back. Moving
# it forward is the generation change itself, made by whoever compiles the
# new generation (and by fixtures), so that is allowed without ``expected``.
# ``state``, ``reasons`` and ``workflow_request_id`` are what callers and the
# reconcile paths change through ``save_incident``; they cannot be guarded
# without ``expected``, which compares the whole payload the caller read.
INCIDENT_VERSION_FIELD = "fencing_token"


def stale_incident_versions(
    stored: FaultIncident, incident: FaultIncident
) -> StaleWriteError | None:
    """The refusal a version-guarded ``save_incident`` raises, or None."""

    if stored.fencing_token <= incident.fencing_token:
        return None
    return StaleWriteError(
        f"incident/{incident.incident_id} moved since it was read -- "
        f"{INCIDENT_VERSION_FIELD} (the incident moved to a new generation: "
        f"stored {stored.fencing_token}, caller {incident.fencing_token})"
        " -- re-read the row and reapply the change"
    )


def record_matches_expected(current: BaseModel | None, expected: BaseModel) -> bool:
    """Whole-payload equality for the ``expected=`` compare-and-set form on the
    backends without a SQL-level CAS (memory, SQLite)."""

    return current is not None and current.model_dump() == expected.model_dump()


def _profile_policy(profile: EffectiveRuntimeProfile) -> dict[str, Any]:
    # The registration anchor names the member a version was first registered
    # under, not policy; the release engine compares registered versions the
    # same way (``gpu_fault_release.regional_runtime_profile.profile_policy``).
    return profile.model_dump(mode="json", exclude={"cluster_id"})


def stored_runtime_profile(
    stored: EffectiveRuntimeProfile | None, profile: EffectiveRuntimeProfile
) -> EffectiveRuntimeProfile | None:
    """What a registration of ``profile`` resolves to against the stored row.

    ``None`` when the version is not stored yet (the caller writes ``profile``);
    the stored row when it carries the same policy, whatever member it was
    registered under; otherwise :class:`RuntimeProfileConflict` naming what
    differs, before anything is written.
    """

    if stored is None:
        return None
    before, after = _profile_policy(stored), _profile_policy(profile)
    if before == after:
        return stored
    drift = []
    for field in sorted(set(before) | set(after)):
        if before.get(field) == after.get(field):
            continue
        if field != "capabilities":
            drift.append(field)
            continue
        old, new = before.get(field) or [], after.get(field) or []
        names = sorted(
            {str(item.get("capability")) for item in old if item not in new}
            | {str(item.get("capability")) for item in new if item not in old}
        )
        drift.append(f"capabilities ({', '.join(names) or 'order'})")
    raise RuntimeProfileConflict(
        f"runtime profile {profile.profile_version} is already registered with "
        f"different content ({'; '.join(drift)}); a registered version is "
        "immutable: register the changed policy under a new profile_version"
    )
