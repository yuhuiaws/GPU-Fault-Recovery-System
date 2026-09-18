from __future__ import annotations

import copy
import json
from urllib.error import HTTPError

import pytest
from kubernetes.client.exceptions import ApiException

from gpu_fault.completion_attempt_store import attempt_state_record
from gpu_fault.completion_outbox import (
    KubernetesCompletionOutbox,
    completion_delivery_disposition,
    pointer_sized_completion_payload,
)
from tests.completion.test_completion_outbox import (
    ACTIVE_NAME,
    OUTBOX_NAME,
    ConfigMapCore,
    RecordingSink,
    payload,
)
from tests.completion.test_cov95_runtime_watcher import observation

ACTIVE_KEY = "active-attempts.json"
TERMINAL_PATH = "/v1/attempts/terminal"


def record():
    return attempt_state_record(observation("attempt").model_dump(mode="json"))


@pytest.mark.parametrize(
    "options",
    [
        {"max_records": 0},
        {"max_bytes": 1023},
        {"replay_batch_size": 0},
        {"max_buffered_tail_bytes": 0},
        {"replay_budget_seconds": 0},
        {"max_retry_age_seconds": 0},
    ],
)
def test_outbox_rejects_invalid_bounds_without_an_api_call(options):
    core = ConfigMapCore()
    with pytest.raises(ValueError, match="bounds? must be positive"):
        KubernetesCompletionOutbox(core, RecordingSink(), **options)
    assert core.reads == []


def test_pointer_sized_payload_preserves_non_tail_fields_and_utf8_boundaries():
    event = {
        **payload(),
        "workload_log_snapshots": [
            None,
            {"record_id": "no-tail"},
            {"record_id": "short", "tail": "A"},
            {"record_id": "trimmed", "tail": "\u20acA", "s3_uri": "s3://local/log"},
        ],
    }
    original = copy.deepcopy(event)
    bounded = pointer_sized_completion_payload(event, max_tail_bytes=2)
    snapshots = bounded["workload_log_snapshots"]
    assert snapshots[:3] == event["workload_log_snapshots"][:3]
    assert snapshots[-1] == {
        "record_id": "trimmed",
        "tail": "A",
        "s3_uri": "s3://local/log",
        "buffered_tail_bytes": 1,
        "buffered_tail_truncated": True,
    }
    assert event == original


@pytest.mark.parametrize("code,expected", [(422, "quarantine"), (503, "retry")])
def test_http_error_disposition_uses_the_shared_status_policy(code, expected):
    error = HTTPError("https://control.invalid", code, "unit", None, None)
    assert completion_delivery_disposition(error) == expected


@pytest.mark.parametrize("missing", ["cluster_id", "attempt_id"])
def test_missing_event_identity_is_not_buffered_or_delivered(missing):
    core, transport = ConfigMapCore(), RecordingSink()
    outbox = KubernetesCompletionOutbox(core, transport)
    invalid = payload()
    invalid.pop(missing)
    with pytest.raises(ValueError, match="requires cluster_id and attempt_id"):
        outbox.post(TERMINAL_PATH, invalid)
    with pytest.raises(ValueError, match="requires cluster_id and attempt_id"):
        outbox.save_attempt_observation(invalid)
    assert core.reads == []
    assert transport.posts == []


@pytest.mark.parametrize("document", ["{}", "[1]"])
def test_invalid_event_log_is_not_interpreted_as_empty(document):
    core = ConfigMapCore()
    core.objects[OUTBOX_NAME]["events.json"] = document
    transport = RecordingSink()
    outbox = KubernetesCompletionOutbox(core, transport)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        outbox.depth()
    assert core.objects[OUTBOX_NAME]["events.json"] == document
    assert transport.posts == []


@pytest.mark.parametrize("document", ["[]", '{"bad": 1}'])
def test_invalid_active_state_is_not_restored_as_empty(document):
    core = ConfigMapCore()
    core.objects[ACTIVE_NAME][ACTIVE_KEY] = document
    outbox = KubernetesCompletionOutbox(core, RecordingSink())
    with pytest.raises(RuntimeError, match="invalid JSON"):
        outbox.load_attempt_observations()
    assert core.objects[ACTIVE_NAME][ACTIVE_KEY] == document


@pytest.mark.parametrize("status", [403, 500])
def test_active_state_removal_refusal_keeps_the_record_and_honors_probe_window(status):
    class Core(ConfigMapCore):
        refuse = False

        def replace_namespaced_config_map(self, name, namespace, body):
            if name == ACTIVE_NAME and self.refuse:
                raise ApiException(status=status, reason="unit removal refusal")
            return super().replace_namespaced_config_map(name, namespace, body)

    core = Core()
    outbox = KubernetesCompletionOutbox(core, RecordingSink())
    active = record()
    assert outbox.save_attempt_observation(active) is True
    before = copy.deepcopy(core.objects)
    core.refuse = True
    if status == 500:
        with pytest.raises(ApiException):
            outbox.remove_attempt_observation(active)
        assert outbox.active_state_unavailable == 0
    else:
        outbox.remove_attempt_observation(active)
        assert outbox.active_state_unavailable == 1
        reads = len(core.reads)
        outbox.remove_attempt_observation(active)
        assert len(core.reads) == reads
    assert core.objects == before


def test_failed_owed_probe_is_deferred_until_next_window_then_recovers(caplog):
    clock = [0.0]

    class Core(ConfigMapCore):
        status = 403

        def read_namespaced_config_map(self, name, namespace):
            if name == ACTIVE_NAME and self.status:
                self.reads.append(name)
                raise ApiException(
                    status=self.status, reason="unit active state refusal"
                )
            return super().read_namespaced_config_map(name, namespace)

    core = Core()
    outbox = KubernetesCompletionOutbox(
        core, RecordingSink(), monotonic=lambda: clock[0]
    )
    assert outbox.load_attempt_observations() == []
    assert outbox.active_state_unavailable == 1
    core.status = 500
    clock[0] = 61
    outbox.probe_active_state()
    assert "owed probe" in caplog.text
    read_count = len(core.reads)
    outbox.probe_active_state()
    assert len(core.reads) == read_count
    assert outbox.active_state_unavailable == 1
    core.status = 0
    clock[0] = 122
    outbox.probe_active_state()
    assert outbox.active_state_unavailable == 0
    assert outbox.save_attempt_observation(record()) is True


def test_unreadable_legacy_and_active_objects_never_invent_restart_memory(caplog):
    class Core(ConfigMapCore):
        def read_namespaced_config_map(self, name, namespace):
            raise ApiException(
                status=403 if name == ACTIVE_NAME else 500, reason="unit"
            )

    outbox = KubernetesCompletionOutbox(Core(), RecordingSink())
    assert outbox.load_attempt_observations() == []
    assert outbox.active_state_unavailable == 1
    assert "cannot migrate persisted attempt observations" in caplog.text
    assert "cannot read the legacy attempt state" in caplog.text


@pytest.mark.parametrize("concurrent_drop", [True, False])
def test_legacy_drop_retry_does_not_restore_an_attempt_that_finished(
    caplog, concurrent_drop
):
    class Core(ConfigMapCore):
        refuse_drop = True

        def replace_namespaced_config_map(self, name, namespace, body):
            if name == OUTBOX_NAME and self.refuse_drop:
                raise ApiException(status=500, reason="unit legacy drop refusal")
            return super().replace_namespaced_config_map(name, namespace, body)

    core = Core()
    active = record()
    key = f"{active['cluster_id']}/{active['attempt_id']}"
    core.objects[OUTBOX_NAME][ACTIVE_KEY] = json.dumps({key: active})
    outbox = KubernetesCompletionOutbox(core, RecordingSink())
    assert outbox.load_attempt_observations() == [active]
    changed = {**active, "restart_budget": 2}
    assert outbox.save_attempt_observation(changed) is True
    outbox.remove_attempt_observation(changed)
    assert json.loads(core.objects[ACTIVE_NAME][ACTIVE_KEY]) == {}
    warnings = [
        message
        for message in caplog.messages
        if "cannot drop the migrated attempt state" in message
    ]
    assert len(warnings) == 1
    core.refuse_drop = False
    if concurrent_drop:
        core.objects[OUTBOX_NAME].pop(ACTIVE_KEY)
    outbox.remove_attempt_observation(changed)
    assert ACTIVE_KEY not in core.objects[OUTBOX_NAME]
    restarted = KubernetesCompletionOutbox(core, RecordingSink())
    assert restarted.load_attempt_observations() == []


def test_already_adopted_legacy_state_is_not_rewritten():
    core = ConfigMapCore()
    active = record()
    key = f"{active['cluster_id']}/{active['attempt_id']}"
    document = json.dumps({key: active}, sort_keys=True, separators=(",", ":"))
    core.objects[OUTBOX_NAME][ACTIVE_KEY] = document
    core.objects[ACTIVE_NAME][ACTIVE_KEY] = document
    version = core.versions[ACTIVE_NAME]
    assert KubernetesCompletionOutbox(
        core, RecordingSink()
    ).load_attempt_observations() == [active]
    assert core.versions[ACTIVE_NAME] == version
    assert ACTIVE_KEY not in core.objects[OUTBOX_NAME]


def test_accepted_event_with_unrecordable_cleanup_retains_replay_ambiguity(caplog):
    class Core(ConfigMapCore):
        replacements = 0
        refuse_cleanup = True

        def replace_namespaced_config_map(self, name, namespace, body):
            self.replacements += 1
            if self.replacements > 1 and self.refuse_cleanup:
                raise ApiException(status=500, reason="unit cleanup refusal")
            return super().replace_namespaced_config_map(name, namespace, body)

    core, transport = Core(), RecordingSink()
    outbox = KubernetesCompletionOutbox(core, transport)
    event = payload()
    assert outbox.post(TERMINAL_PATH, event) == {"accepted": True}
    assert outbox.append_failures_total == 1
    assert core.events(OUTBOX_NAME) == 1
    assert len(transport.posts) == 1
    assert "could not" in caplog.text
    core.refuse_cleanup = False
    assert outbox.replay() == 1
    assert transport.posts == [(TERMINAL_PATH, event), (TERMINAL_PATH, event)]
    assert core.events(OUTBOX_NAME) == 0
