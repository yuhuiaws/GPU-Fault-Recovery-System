from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch, NodeHealthPolicy
from gpu_fault.models import HealthSignalState, RecoveryAction, Severity
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)

NOW = datetime(2026, 9, 14, tzinfo=UTC)
KEY = "semantic-cluster/semantic-node/efa_inventory_mismatch/node"
A = "a" * 64
B = "b" * 64


def at(seconds: float) -> datetime:
    return NOW + timedelta(seconds=seconds)


def claim(store, fingerprint, seconds, *, active=True, observed=None, minimum=0.0):
    return store.claim_health_signal_transitions(
        [(KEY, active, at(seconds) if observed is None else observed, minimum)],
        received_at=at(seconds),
        semantic_fingerprints=[fingerprint],
    )[0]


def acknowledge(store, fingerprint, seconds):
    store.mark_health_signal_notified(
        KEY, notified_at=at(seconds), semantic_fingerprint=fingerprint
    )


def host_batch(mode, seconds, *, value=1.0, labels=None, observed=None):
    return HostTelemetryBatch(
        batch_id=f"host-semantic-{seconds}-{mode}",
        cluster_id="semantic-cluster",
        node_id="semantic-node",
        observed_at=at(seconds) if observed is None else observed,
        received_at=at(seconds),
        samples=[
            HostMetricSample(
                name="efa_inventory_mismatch",
                value=value,
                labels={"failure_mode": mode, **(labels or {})},
            )
        ],
    )


def acknowledge_finding(store, finding, seconds):
    store.mark_health_signal_notified(
        finding_health_signal_key(finding),
        notified_at=at(seconds),
        semantic_fingerprint=finding_health_signal_fingerprint(finding),
    )


class LegacyMemoryStore(InMemoryStore):
    def seed_legacy_signal(self, payload):
        with self._lock:
            self._health_signal_states[KEY] = HealthSignalState.model_validate(payload)


class LegacySqliteStore(SqliteStore):
    def seed_legacy_signal(self, payload):
        with self._state_transaction(f"health_signal_state/{KEY}"):
            self._db.execute(
                "INSERT INTO objects(kind,key,payload) VALUES ('health_signal_state',?,?)",
                (KEY, json.dumps(payload)),
            )


class SemanticSignalContract:
    def test_changed_meaning_reemits_on_the_same_key(self, store):
        assert claim(store, A, 0) is True
        acknowledge(store, A, 0)
        assert claim(store, A, 1) is False
        assert claim(store, B, 2) is True
        current = store.get_health_signal_state(KEY)
        assert current.semantic_fingerprint == B
        assert current.active_since == NOW, (
            "meaning changes must preserve the active window"
        )
        assert current.semantic_since == at(2)
        assert current.notified is False
        acknowledge(store, B, 2)
        assert claim(store, B, 3) is False

    def test_a_to_b_to_a_is_three_episodes_without_a_clear(self, store):
        for fingerprint, seconds in ((A, 0), (B, 10), (A, 20)):
            assert claim(store, fingerprint, seconds) is True
            acknowledge(store, fingerprint, seconds)
            assert claim(store, fingerprint, seconds + 1) is False
        state = store.get_health_signal_state(KEY)
        assert state.active_since == NOW
        assert state.semantic_since == at(20)
        assert state.semantic_fingerprint == A

    def test_old_a_acknowledgement_cannot_latch_a_later_a_episode(self, store):
        assert claim(store, A, 0) is True
        assert claim(store, B, 1) is True
        assert claim(store, A, 2) is True
        acknowledge(store, A, 0)
        assert store.get_health_signal_state(KEY).notified is False
        assert claim(store, A, 3) is True, "the newer A meaning still owes delivery"
        acknowledge(store, A, 3)
        assert claim(store, A, 4) is False

    @pytest.mark.parametrize("ack", ["old-meaning", "unbound", "future-clock"])
    def test_acknowledgement_must_bind_the_meaning_and_claim_clock(self, store, ack):
        assert claim(store, A, 0) is True
        assert claim(store, B, 1) is True
        if ack == "old-meaning":
            acknowledge(store, A, 1)
        elif ack == "future-clock":
            acknowledge(store, B, 100)
        else:
            store.mark_health_signal_notified(KEY, notified_at=at(1))
        assert store.get_health_signal_state(KEY).notified is False
        acknowledge(store, B, 1)
        assert store.get_health_signal_state(KEY).notified is True

    def test_uncommitted_delivery_retries_without_changing_episode_identity(
        self, store
    ):
        assert claim(store, A, 0) is True
        assert claim(store, A, 1) is True
        state = store.get_health_signal_state(KEY)
        assert state.notified is False
        assert state.semantic_since == NOW
        acknowledge(store, A, 0)
        assert claim(store, A, 2) is False, (
            "a committed delivery for the same episode may latch a later unchanged sample"
        )

    def test_recovery_rearms_even_when_the_meaning_returns_unchanged(self, store):
        assert claim(store, A, 0) is True
        acknowledge(store, A, 0)
        assert claim(store, A, 1, active=False) is False
        state = store.get_health_signal_state(KEY)
        assert state.active is False
        assert state.active_since is state.semantic_since is None
        acknowledge(store, A, 0)
        assert claim(store, A, 2) is True
        acknowledge(store, A, 0)
        assert store.get_health_signal_state(KEY).notified is False
        acknowledge(store, A, 2)
        assert claim(store, A, 3) is False

    def test_stale_receive_clock_cannot_change_semantics_or_clear_state(self, store):
        assert claim(store, A, 10) is True
        acknowledge(store, A, 10)
        before = store.get_health_signal_state(KEY)
        assert claim(store, B, 9, observed=at(1000)) is False
        assert claim(store, B, 10, active=False, observed=at(2000)) is False
        assert store.get_health_signal_state(KEY) == before

    def test_receive_clock_preserves_duration_through_a_backwards_node_clock(
        self, store
    ):
        assert claim(store, A, 0, minimum=10) is False
        assert claim(store, B, 10, observed=at(-100), minimum=10) is True
        current = store.get_health_signal_state(KEY)
        assert current.active_since == NOW
        assert current.semantic_since == at(10)
        assert current.observed_at == at(-100)
        assert store.health_signal_clock_regressions_total == 1
        acknowledge(store, B, 10)
        assert claim(store, B, 11, observed=at(-99), minimum=10) is False

    def test_changed_semantics_do_not_skip_an_unfulfilled_sustain_window(self, store):
        assert claim(store, A, 0, minimum=10) is False
        assert claim(store, B, 5, minimum=10) is False
        assert claim(store, B, 10, minimum=10) is True
        state = store.get_health_signal_state(KEY)
        assert state.active_since == NOW
        assert state.semantic_since == at(5)

    @pytest.mark.parametrize("notified", [None, False, True])
    def test_legacy_active_state_binds_unknown_semantics_once(self, store, notified):
        payload = {
            "signal_key": KEY,
            "active": True,
            "observed_at": NOW.isoformat(),
            "active_since": NOW.isoformat(),
        }
        if notified is not None:
            payload["notified"] = notified
        store.seed_legacy_signal(payload)
        previous = store.get_health_signal_state(KEY)
        assert previous.semantic_fingerprint is previous.semantic_since is None
        assert claim(store, A, 1) is True, (
            "legacy state cannot prove the current meaning was delivered"
        )
        store.mark_health_signal_notified(KEY, notified_at=at(1))
        assert store.get_health_signal_state(KEY).notified is False
        acknowledge(store, A, 1)
        assert claim(store, A, 2) is False

    def test_legacy_callers_keep_the_original_latch_contract(self, store):
        assert claim(store, None, 0) is True
        store.mark_health_signal_notified(KEY, notified_at=NOW)
        assert claim(store, None, 1) is False
        assert claim(store, None, 2, active=False) is False
        assert claim(store, None, 3) is True
        state = store.get_health_signal_state(KEY)
        assert state.semantic_fingerprint is state.semantic_since is None

    def test_legacy_claim_cannot_erase_an_existing_semantic_binding(self, store):
        assert claim(store, A, 0) is True
        assert claim(store, None, 1) is True
        state = store.get_health_signal_state(KEY)
        assert state.semantic_fingerprint == A
        assert state.semantic_since == NOW
        store.mark_health_signal_notified(KEY, notified_at=at(1))
        assert store.get_health_signal_state(KEY).notified is False
        acknowledge(store, A, 1)
        assert claim(store, A, 2) is False

    @pytest.mark.parametrize("fingerprints", [[], [A, B], [""], [1]])
    def test_invalid_semantics_refuse_before_any_state_is_written(
        self, store, fingerprints
    ):
        with pytest.raises(ValueError, match="match the claim batch"):
            store.claim_health_signal_transitions(
                [(KEY, True, NOW, 0.0)], semantic_fingerprints=fingerprints
            )
        assert store.get_health_signal_state(KEY) is None

    def test_duplicate_keys_keep_fingerprints_aligned_with_each_item(self, store):
        assert store.claim_health_signal_transitions(
            [(KEY, True, NOW, 0.0), (KEY, True, at(1), 0.0)],
            semantic_fingerprints=[A, B],
        ) == [True, True]
        current = store.get_health_signal_state(KEY)
        assert current.semantic_fingerprint == B
        assert current.semantic_since == at(1)
        assert current.active_since == NOW
        assert store.claim_health_signal_transitions(
            [(KEY, True, at(2), 0.0), (KEY, True, at(2), 0.0)],
            semantic_fingerprints=[A, B],
        ) == [True, False], "a same-clock duplicate cannot replace the accepted meaning"
        assert store.get_health_signal_state(KEY).semantic_fingerprint == A

    def test_single_claim_supports_semantics_without_changing_legacy_arguments(
        self, store
    ):
        assert (
            store.claim_health_signal_transition(
                KEY, True, NOW, 0.0, received_at=at(1), semantic_fingerprint=A
            )
            is True
        )
        acknowledge(store, A, 1)
        assert (
            store.claim_health_signal_transition(
                KEY, True, at(2), semantic_fingerprint=B
            )
            is True
        )
        assert store.get_health_signal_state(KEY).semantic_since == at(2)

    def test_host_inventory_mode_change_and_a_b_a_reemit(self, store):
        policy = NodeHealthPolicy(store)
        actions = []
        for mode, seconds in (
            ("LINK_INACTIVE", 0),
            ("PCI_DEVICE_MISSING", 1),
            ("LINK_INACTIVE", 2),
        ):
            (finding,) = policy.evaluate_metrics(host_batch(mode, seconds))
            assert finding_health_signal_key(finding) == KEY
            actions.append(finding.recommended_action)
            acknowledge_finding(store, finding, seconds)
            assert policy.evaluate_metrics(host_batch(mode, seconds + 0.5)) == []
        assert actions == [
            RecoveryAction.RUN_DIAGNOSTICS,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.RUN_DIAGNOSTICS,
        ]
        assert policy.evaluate_metrics(host_batch("LINK_INACTIVE", 3, value=0)) == []
        assert len(policy.evaluate_metrics(host_batch("LINK_INACTIVE", 4))) == 1

    def test_failure_mode_change_reemits_even_with_the_same_action_and_severity(
        self, store
    ):
        policy = NodeHealthPolicy(store)
        (first,) = policy.evaluate_metrics(host_batch("LINK_INACTIVE", 0))
        acknowledge_finding(store, first, 0)
        (second,) = policy.evaluate_metrics(host_batch("EXCESS_DEVICE", 1))
        assert first.recommended_action is second.recommended_action
        assert first.severity is second.severity
        assert finding_health_signal_fingerprint(
            first
        ) != finding_health_signal_fingerprint(second)

    def test_omitted_failure_mode_matches_the_resolved_policy_default(self, store):
        policy = NodeHealthPolicy(store)
        implicit = host_batch("unused", 0).model_copy(
            update={
                "samples": [HostMetricSample(name="efa_inventory_mismatch", value=1)]
            }
        )
        (first,) = policy.evaluate_metrics(implicit)
        assert first.diagnostic_parameters["failure_mode"] == "PCI_DEVICE_MISSING"
        acknowledge_finding(store, first, 0)
        assert policy.evaluate_metrics(host_batch("PCI_DEVICE_MISSING", 1)) == []

    @pytest.mark.parametrize("changed_field", ["severity", "action"])
    def test_host_reemits_for_changed_meaning_not_values_or_incidental_labels(
        self, store, monkeypatch, changed_field
    ):
        policy = NodeHealthPolicy(store)
        batch = host_batch("LINK_INACTIVE", 0).model_copy(
            update={
                "samples": [HostMetricSample(name="gpu_inventory_mismatch", value=1.0)]
            }
        )
        (first,) = policy.evaluate_metrics(batch)
        acknowledge_finding(store, first, 0)
        repeated = batch.model_copy(
            update={
                "batch_id": "unchanged-inventory",
                "observed_at": at(1),
                "received_at": at(1),
                "samples": [
                    HostMetricSample(
                        name="gpu_inventory_mismatch",
                        value=2.0,
                        labels={"note": "different prose"},
                    )
                ],
            }
        )
        assert policy.evaluate_metrics(repeated) == []
        threshold, category, severity, action, reason = policy.METRIC_RULES[
            "gpu_inventory_mismatch"
        ]
        monkeypatch.setitem(
            policy.METRIC_RULES,
            "gpu_inventory_mismatch",
            (
                threshold,
                category,
                Severity.FATAL if changed_field == "severity" else severity,
                RecoveryAction.QUARANTINE if changed_field == "action" else action,
                reason,
            ),
        )
        changed = repeated.model_copy(
            update={
                "batch_id": "changed-severity",
                "observed_at": at(2),
                "received_at": at(2),
            }
        )
        (second,) = policy.evaluate_metrics(changed)
        if changed_field == "severity":
            assert second.severity is Severity.FATAL
        else:
            assert second.recommended_action is RecoveryAction.QUARANTINE
        assert finding_health_signal_fingerprint(
            first
        ) != finding_health_signal_fingerprint(second)
