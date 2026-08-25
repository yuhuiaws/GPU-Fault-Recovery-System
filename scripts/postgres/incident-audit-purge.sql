\set ON_ERROR_STOP on

\if :{?incident_id}
\else
  \echo 'incident_id is required'
  \quit 2
\endif

\if :{?confirm_incident_id}
\else
  \echo 'confirm_incident_id is required'
  \quit 2
\endif

SELECT :'incident_id' = :'confirm_incident_id' AS confirmed
\gset
\if :confirmed
\else
  \echo 'confirmation does not match incident_id'
  \quit 2
\endif

BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='2min';

SELECT pg_advisory_xact_lock(
    hashtextextended('gpu-fault-incident-purge/' || :'incident_id', 0)
);

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_objects
    WHERE kind='incident' AND key=:'incident_id'
) AS incident_exists
\gset
\if :incident_exists
\else
  \echo 'incident does not exist'
  ROLLBACK;
  \quit 3
\endif

CREATE TEMP TABLE purge_workflow_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_objects
WHERE kind='workflow'
  AND payload->>'incident_id'=:'incident_id';

CREATE TEMP TABLE purge_notification_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_objects
WHERE kind='notification'
  AND payload->>'incident_id'=:'incident_id';

CREATE TEMP TABLE purge_plan_ids ON COMMIT DROP AS
SELECT key
FROM gpu_fault_objects
WHERE kind='plan'
  AND payload->>'incident_id'=:'incident_id';

CREATE TEMP TABLE purge_decision_ids ON COMMIT DROP AS
SELECT key, payload
FROM gpu_fault_objects
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

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_objects
    WHERE kind='workflow'
      AND key IN (SELECT key FROM purge_workflow_ids)
      AND payload->>'status' IN (
          'PENDING', 'RUNNING', 'SAFETY_PENDING'
      )
) AS has_active_workflow
\gset
\if :has_active_workflow
  \echo 'refusing purge: incident has an active workflow'
  ROLLBACK;
  \quit 4
\endif

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_objects
    WHERE kind='remote_command'
      AND payload->>'incident_id'=:'incident_id'
      AND payload->>'status' IN ('PENDING', 'WAITING', 'LEASED')
) AS has_open_command
\gset
\if :has_open_command
  \echo 'refusing purge: incident has an open remote command'
  ROLLBACK;
  \quit 4
\endif

SELECT EXISTS (
    SELECT 1
    FROM gpu_fault_objects
    WHERE kind='workflow'
      AND payload->>'incident_id'<>:'incident_id'
      AND payload->>'predecessor_workflow_id' IN (
          SELECT key FROM purge_workflow_ids
      )
) AS has_external_successor
\gset
\if :has_external_successor
  \echo 'refusing purge: another incident references this workflow'
  \echo 'purge dependent incidents first'
  ROLLBACK;
  \quit 4
\endif

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
WHERE kind='remote_command'
  AND payload->>'incident_id'=:'incident_id';

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
WHERE kind='workflow'
  AND key IN (SELECT key FROM purge_workflow_ids);

DELETE FROM gpu_fault_objects
WHERE kind='incident' AND key=:'incident_id';

COMMIT;

\echo 'incident audit bundle purged'
