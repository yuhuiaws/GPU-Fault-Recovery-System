\set ON_ERROR_STOP on

\if :{?incident_id}
\else
  DO $audit_incident_argument_guard$
  BEGIN
      RAISE EXCEPTION 'incident_id is required';
  END
  $audit_incident_argument_guard$;
\endif

\if :{?confirm_incident_id}
\else
  DO $audit_confirmation_argument_guard$
  BEGIN
      RAISE EXCEPTION 'confirm_incident_id is required';
  END
  $audit_confirmation_argument_guard$;
\endif

SELECT :'incident_id' = :'confirm_incident_id' AS confirmed
\gset
\if :confirmed
\else
  DO $audit_confirmation_guard$
  BEGIN
      RAISE EXCEPTION 'confirmation does not match incident_id';
  END
  $audit_confirmation_guard$;
\endif

BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='2min';

SELECT pg_advisory_xact_lock(
    hashtextextended('gpu-fault-incident-purge/' || :'incident_id', 0)
);

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_control_records
    WHERE kind='incident' AND key=:'incident_id'
) AS incident_exists
\gset
\if :incident_exists
\else
  ROLLBACK;
  DO $audit_missing_incident_guard$
  BEGIN
      RAISE EXCEPTION 'incident does not exist';
  END
  $audit_missing_incident_guard$;
\endif

CREATE TEMP TABLE purge_workflow_ids ON COMMIT DROP AS
SELECT key, payload
FROM gpu_fault_workflow_records
WHERE incident_id=:'incident_id';

CREATE TEMP TABLE purge_remote_command_ids ON COMMIT DROP AS
SELECT key, payload
FROM gpu_fault_remote_command_records
WHERE incident_id=:'incident_id';

CREATE TEMP TABLE purge_notification_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_control_records
WHERE kind='notification'
  AND payload->>'incident_id'=:'incident_id';

CREATE TEMP TABLE purge_plan_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_control_records
WHERE kind='plan'
  AND payload->>'incident_id'=:'incident_id';

CREATE TEMP TABLE purge_decision_ids ON COMMIT DROP AS
SELECT key, payload
FROM gpu_fault_control_records
WHERE kind='decision'
  AND payload->>'recovery_plan_id' IN (
      SELECT key FROM purge_plan_ids
  );

CREATE TEMP TABLE purge_diagnostic_ids ON COMMIT DROP AS
SELECT DISTINCT payload->>'diagnostic_request_id' AS key
FROM purge_decision_ids
WHERE COALESCE(payload->>'diagnostic_request_id', '') <> '';

CREATE TEMP TABLE purge_event_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_links
WHERE kind='incident_by_event' AND value=:'incident_id';

DO $purge_terminal_guard$
BEGIN
    IF EXISTS (
        SELECT 1 FROM purge_workflow_ids
        WHERE (
            payload->>'status' IN ('SUCCEEDED', 'SUPERSEDED')
            OR (
                payload->>'status'='FAILED'
                AND NULLIF(payload->>'failure_handled_at', '') IS NOT NULL
            )
        ) IS NOT TRUE
    ) THEN
        RAISE EXCEPTION 'refusing purge: incident has a non-terminal or unhandled workflow';
    END IF;
    IF EXISTS (
        SELECT 1 FROM purge_remote_command_ids
        WHERE (payload->>'status' IN ('SUCCEEDED', 'FAILED')) IS NOT TRUE
    ) THEN
        RAISE EXCEPTION 'refusing purge: incident has an open or unknown remote command';
    END IF;
END
$purge_terminal_guard$;

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_workflow_records
    WHERE incident_id IS DISTINCT FROM :'incident_id'
      AND predecessor_workflow_id IN (
          SELECT key FROM purge_workflow_ids
      )
) AS has_external_successor
\gset
\if :has_external_successor
  ROLLBACK;
  DO $audit_successor_guard$
  BEGIN
      RAISE EXCEPTION 'refusing purge: another incident references this workflow; purge dependent incidents first';
  END
  $audit_successor_guard$;
\endif

DO $purge_control_state$
DECLARE victim RECORD;
BEGIN
    FOR victim IN
        SELECT 'remote_command' AS kind, key, payload FROM purge_remote_command_ids
        UNION ALL
        SELECT 'workflow', key, payload FROM purge_workflow_ids
        ORDER BY kind, key
    LOOP
        IF NOT gpu_fault_delete_control_state(victim.kind, victim.key, victim.payload) THEN
            RAISE EXCEPTION 'refusing purge: control-state record changed or could not be deleted';
        END IF;
    END LOOP;
END
$purge_control_state$;

DELETE FROM gpu_fault_links
WHERE kind='notification_dedup'
  AND value IN (SELECT key FROM purge_notification_ids);

DELETE FROM gpu_fault_objects
WHERE kind IN ('notification_delivery', 'notification_result')
  AND key IN (SELECT key FROM purge_notification_ids);

DELETE FROM gpu_fault_objects
WHERE kind='notification'
  AND key IN (SELECT key FROM purge_notification_ids);

DELETE FROM gpu_fault_objects
WHERE kind IN ('diagnostic', 'triage')
  AND key IN (SELECT key FROM purge_diagnostic_ids);

DELETE FROM gpu_fault_objects
WHERE kind IN ('decision', 'event')
  AND key IN (SELECT key FROM purge_decision_ids);

DELETE FROM gpu_fault_objects
WHERE kind='plan'
  AND key IN (SELECT key FROM purge_plan_ids);

DELETE FROM gpu_fault_objects
WHERE kind='marker'
  AND payload->>'incident_id'=:'incident_id';

DELETE FROM gpu_fault_objects
WHERE kind IN (
        'xid_correlation_event',
        'xid_policy_decision',
        'xid_correlation'
    )
  AND key IN (SELECT key FROM purge_event_ids);

DELETE FROM gpu_fault_links
WHERE
    (
        kind IN (
            'incident_by_event',
            'replacement_fault_group',
            'sxid_fault_group'
        )
        AND value=:'incident_id'
    );

DELETE FROM gpu_fault_objects
WHERE kind='incident' AND key=:'incident_id';

COMMIT;

\echo 'incident audit bundle purged'
