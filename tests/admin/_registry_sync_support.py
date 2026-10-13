"""The deploy-host installation registry sync record a completed release leaves.

``uninstall`` and ``remove-cluster`` refuse to start an attempt unless
``<state-dir>/release-deploy/installation-registry-sync.json`` records a
successful sync (``gpu_fault.admin.resource_registry_outcome.deletion_gate``).
Harnesses that drive either verb over a fixture site record one first, the way
the site's last deploy would have.
"""

from __future__ import annotations

from pathlib import Path

from gpu_fault.admin.resource_registry_outcome import write_outcome

SYNCED_AT = "2026-01-01T00:00:00Z"


def record_registry_sync(
    state_dir: Path, *, release_id: str = "release-a", error: str | None = None
) -> Path:
    """Record a successful sync of ``release_id``, or the failure ``error``."""

    outcome: dict[str, object] = (
        {"status": "UNAVAILABLE", "synced_at": SYNCED_AT, "error": error}
        if error is not None
        else {
            "status": "SYNCED",
            "path": str(state_dir / "installation-resources.json"),
            "sha256": "a" * 64,
            "synced_at": SYNCED_AT,
        }
    )
    return write_outcome(state_dir, release_id=release_id, outcome=outcome)
