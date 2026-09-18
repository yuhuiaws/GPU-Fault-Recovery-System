\set ON_ERROR_STOP on

\if :{?incident_id}
\else
  DO $audit_incident_argument_guard$
  BEGIN
      RAISE EXCEPTION 'incident_id is required: psql -v incident_id=inc-...';
  END
  $audit_incident_argument_guard$;
\endif

\echo 'Incident'
SELECT
    key AS incident_id,
    payload->>'cluster_id' AS cluster_id,
    payload->>'state' AS state,
    payload->>'created_at' AS created_at,
    payload->>'updated_at' AS updated_at
FROM gpu_fault_control_records
WHERE kind='incident' AND key=:'incident_id';

\echo 'Related object counts'
WITH
workflow_ids AS (
    SELECT key
    FROM gpu_fault_workflow_records
    WHERE incident_id=:'incident_id'
),
remote_command_ids AS (
    SELECT key
    FROM gpu_fault_remote_command_records
    WHERE incident_id=:'incident_id'
),
notification_ids AS (
    SELECT key
    FROM gpu_fault_control_records
    WHERE kind='notification'
      AND payload->>'incident_id'=:'incident_id'
),
plan_ids AS (
    SELECT key
    FROM gpu_fault_control_records
    WHERE kind='plan'
      AND payload->>'incident_id'=:'incident_id'
),
decision_ids AS (
    SELECT key, payload
    FROM gpu_fault_control_records
    WHERE kind='decision'
      AND payload->>'recovery_plan_id' IN (
          SELECT key FROM plan_ids
      )
),
diagnostic_ids AS (
    SELECT DISTINCT payload->>'diagnostic_request_id' AS key
    FROM decision_ids
    WHERE COALESCE(payload->>'diagnostic_request_id', '') <> ''
),
event_ids AS (
    SELECT key
    FROM gpu_fault_links
    WHERE kind='incident_by_event' AND value=:'incident_id'
),
related_objects AS (
    SELECT kind, key
    FROM gpu_fault_control_records
    WHERE kind NOT IN ('workflow', 'remote_command')
      AND (
        (kind='incident' AND key=:'incident_id')
        OR (
            kind IN (
                'notification',
                'notification_delivery',
                'notification_result'
            )
            AND key IN (SELECT key FROM notification_ids)
        )
        OR (kind='plan' AND key IN (SELECT key FROM plan_ids))
        OR (
            kind IN ('decision', 'event')
            AND key IN (SELECT key FROM decision_ids)
        )
        OR (
            kind IN ('diagnostic', 'triage')
            AND key IN (SELECT key FROM diagnostic_ids)
        )
        OR (
            kind='marker'
            AND payload->>'incident_id'=:'incident_id'
        )
        OR (
            kind IN (
                'xid_correlation_event',
                'xid_policy_decision',
                'xid_correlation'
            )
            AND key IN (SELECT key FROM event_ids)
        )
      )
    UNION ALL
    SELECT 'workflow', key FROM workflow_ids
    UNION ALL
    SELECT 'remote_command', key FROM remote_command_ids
),
related_links AS (
    SELECT kind, key
    FROM gpu_fault_links
    WHERE
        (
            kind IN (
                'incident_by_event',
                'replacement_fault_group',
                'sxid_fault_group'
            )
            AND value=:'incident_id'
        )
        OR (
            kind='notification_dedup'
            AND value IN (SELECT key FROM notification_ids)
        )
)
SELECT 'object:' || kind AS record_type, count(*) AS records
FROM related_objects
GROUP BY kind
UNION ALL
SELECT 'link:' || kind AS record_type, count(*) AS records
FROM related_links
GROUP BY kind
ORDER BY record_type;

\echo 'Workflow states'
SELECT key AS workflow_id, status, predecessor_workflow_id, updated_at
FROM gpu_fault_workflow_records
WHERE incident_id=:'incident_id'
ORDER BY created_at, key;

\echo 'Open or unknown remote commands; this result must be empty before purge'
SELECT key AS command_id, status, workflow_request_id AS workflow_id
FROM gpu_fault_remote_command_records
WHERE incident_id=:'incident_id'
  AND (status IN ('SUCCEEDED', 'FAILED')) IS NOT TRUE
ORDER BY key;

\echo 'External successor workflows; purge these incidents first'
WITH workflow_ids AS (
    SELECT key
    FROM gpu_fault_workflow_records
    WHERE incident_id=:'incident_id'
)
SELECT key AS workflow_id,
       incident_id AS dependent_incident_id,
       status, predecessor_workflow_id
FROM gpu_fault_workflow_records
WHERE incident_id IS DISTINCT FROM :'incident_id'
  AND predecessor_workflow_id IN (
      SELECT key FROM workflow_ids
  )
ORDER BY created_at, key;
