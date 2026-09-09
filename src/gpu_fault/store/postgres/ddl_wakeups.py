from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import _ensure_trigger


def _create_wakeup_triggers(cursor: Any) -> None:
    # Two pollers read ``gpu_fault_objects``: the workflow dispatcher scans
    # ``kind='workflow'`` on a 5 s poll and the data-plane executor claims
    # ``kind='remote_command'`` on a 2 s poll, so every remediation step paid
    # 5-10 s of pure waiting and an eight-step chain lost about a minute.
    # The processor queue solved the same problem with
    # ``gpu_fault_processor_queue_notify_pending``; this is the same design for
    # the two remaining pollers: one row trigger, two channels, payloads that
    # say "scan now" and nothing the consumer would trust as state.
    #
    # One function for both kinds so every other kind pays a single plpgsql
    # call and two text compares; ``UPDATE OF payload`` is the only column
    # that changes on this table. The status literals and the compared fields
    # are pinned to ``gpu_fault.store.shared.wakeups`` (the in-process hub the
    # memory and SQLite stores publish from) by
    # ``tests/store/test_wakeup_listener.py``.
    #
    # Workflow: notify when the new row is in the dispatcher's executable set
    # (``EXECUTABLE_WORKFLOW_STATUSES``) and either it is new or one of the
    # fields the scan orders or filters on changed. The executor writes a
    # WAITING row back on every dispatch (D-7) and renews its lease every few
    # seconds -- each an UPDATE of a RUNNING row -- and waking the dispatcher
    # on those would re-dispatch the row it had just written, in a loop, until
    # the remote command completed (the spool trigger learned the same lesson
    # in E-8: do not wake a consumer on its own bookkeeping).
    #
    # Remote command: notify on every status transition, not only PENDING.
    # The executor claim filters on PENDING; the dispatcher wants SUCCEEDED /
    # FAILED so a WAITING step advances on the next cycle instead of after the
    # 5 s poll. Renewals (same status) and cancellation requests (still
    # LEASED) stay quiet.
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_objects_notify_wakeup()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.kind = 'workflow' THEN
                IF coalesce(NEW.payload->>'status', '')
                   IN ('PENDING', 'SAFETY_PENDING', 'RUNNING') THEN
                    IF TG_OP = 'UPDATE'
                       AND OLD.payload->>'status'
                           IS NOT DISTINCT FROM NEW.payload->>'status'
                       AND OLD.payload->>'not_before'
                           IS NOT DISTINCT FROM NEW.payload->>'not_before'
                       AND OLD.payload->>'merge_revision'
                           IS NOT DISTINCT FROM NEW.payload->>'merge_revision'
                       AND OLD.payload->>'execution_owner_id'
                           IS NOT DISTINCT FROM NEW.payload->>'execution_owner_id'
                       AND OLD.payload->>'fencing_token'
                           IS NOT DISTINCT FROM NEW.payload->>'fencing_token'
                    THEN
                        RETURN NULL;
                    END IF;
                    PERFORM pg_notify(
                        'gpu_fault_workflow_dispatch',
                        json_build_object(
                            'request_id', NEW.key,
                            'cluster_id', NEW.payload->>'cluster_id',
                            'status', NEW.payload->>'status',
                            'not_before', NEW.payload->>'not_before'
                        )::text
                    );
                END IF;
            ELSIF NEW.kind = 'remote_command' THEN
                IF TG_OP = 'UPDATE'
                   AND OLD.payload->>'status'
                       IS NOT DISTINCT FROM NEW.payload->>'status'
                THEN
                    RETURN NULL;
                END IF;
                PERFORM pg_notify(
                    'gpu_fault_remote_command',
                    json_build_object(
                        'command_id', NEW.key,
                        'cluster_id', NEW.payload->>'cluster_id',
                        'workflow_request_id',
                            NEW.payload->>'workflow_request_id',
                        'status', NEW.payload->>'status'
                    )::text
                );
            END IF;
            RETURN NULL;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_objects_notify_wakeup_trigger",
        "gpu_fault_objects",
        """
        CREATE TRIGGER
        gpu_fault_objects_notify_wakeup_trigger
        AFTER INSERT OR UPDATE OF payload
        ON gpu_fault_objects
        FOR EACH ROW
        EXECUTE FUNCTION
            gpu_fault_objects_notify_wakeup()
        """,
    )
