"""``schema_rollback_compatible`` was a promise the runtime could not keep (H2).

The release config let a manifest declare ``database.rollback_compatible: true``
and the engine then allowed automatic rollback across a schema change. But a
rolled-back wheel requires the exact ``POSTGRES_SCHEMA_VERSION`` and an exact
migration history, so it CrashLooped on the new schema. The flag is gone: a
schema change is never auto-rollback-compatible, a manifest that still claims
it is rejected, and the ``--accept-schema-change`` fail-forward path remains
the only way through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_config as CONFIG
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_schema_change as SCHEMA_CHANGE


def _manifest(database: dict[str, object] | None) -> dict:
    components = {
        name: {"sha256": "a" * 64}
        for name in (
            "collector",
            "cpu",
            "dcgm",
            "endpoint",
            "executor",
            "node",
            "observability",
            "schema",
            "watcher",
        )
    }
    delivery = {
        "schema_version": 1,
        "runtime_prebuilt": True,
        "components": components,
        "images": {
            name: {"reference": f"registry/{name}@sha256:" + "a" * 64}
            for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
        },
        "node_template_inputs": {"sha256": "b" * 64},
    }
    delivery["sha256"] = CONFIG.canonical_sha256(delivery)
    manifest = {
        "deployable": True,
        "delivery": delivery,
        "components": {"node_bundle": {"template_sha256": "b" * 64}},
    }
    if database is not None:
        manifest["database"] = database
    return manifest


def test_manifest_claiming_rollback_compatible_schema_is_rejected() -> None:
    manifest = _manifest({"rollback_compatible": True})

    with pytest.raises(CONFIG.ReleaseError, match="rollback_compatible") as error:
        CONFIG.parse_delivery_identity(manifest, manifest["components"])

    message = str(error.value)
    assert "exact" in message and "accept-schema-change" in message, (
        "the refusal must say why (exact schema match) and what to do instead"
    )


@pytest.mark.parametrize("database", [None, {}, {"rollback_compatible": False}])
def test_manifest_without_the_claim_still_parses(database) -> None:
    manifest = _manifest(database)

    parsed = CONFIG.parse_delivery_identity(manifest, manifest["components"])

    assert len(parsed) == 5, "the parser must not hand back a rollback flag"


def test_schema_change_always_needs_acceptance_under_auto_rollback() -> None:
    config = SimpleNamespace(auto_rollback=True)

    needs = SCHEMA_CHANGE.schema_change_needs_acceptance(
        config, frozenset({"database_schema"})
    )

    assert needs is True, "a schema change under autoRollback must ask for consent"


def test_rollback_across_a_schema_change_refuses_without_the_flag() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True),
        state={"release_diff": {"changed": ["database_schema"]}},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
    )

    with pytest.raises(CONFIG.ReleaseError, match="not.*declared backward-compatible"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})
