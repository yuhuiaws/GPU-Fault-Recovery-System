-- What blocks an incident audit purge: the one definition that both
-- incident-audit-preview.sql and incident-audit-purge.sql include, so the
-- preview lists exactly the records the purge's terminal guard refuses.
--
-- Session-temporary functions in pg_temp: nothing is written to the schema,
-- and they disappear with the psql session. Call them schema-qualified as
-- pg_temp.<name>(...); PostgreSQL never resolves an unqualified function
-- call to pg_temp. Plain SQL without psql meta-commands, so the tests run
-- this file verbatim.

-- A workflow blocks the purge unless it is SUCCEEDED or SUPERSEDED, or it is
-- FAILED and its failure handling is done (failure_handled_at set). Until
-- then an escalation may still be owed. A missing, empty or unknown value
-- proves nothing, so it blocks.
CREATE OR REPLACE FUNCTION pg_temp.gpu_fault_audit_workflow_blocks_purge(
    status TEXT,
    failure_handled_at TEXT
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
AS $audit_workflow_blocks_purge$
    SELECT (
        status IN ('SUCCEEDED', 'SUPERSEDED')
        OR (status = 'FAILED' AND NULLIF(failure_handled_at, '') IS NOT NULL)
    ) IS NOT TRUE
$audit_workflow_blocks_purge$;

-- A remote command blocks the purge unless it SUCCEEDED or FAILED. A missing
-- or unknown status blocks.
CREATE OR REPLACE FUNCTION pg_temp.gpu_fault_audit_remote_command_blocks_purge(
    status TEXT
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
AS $audit_remote_command_blocks_purge$
    SELECT (status IN ('SUCCEEDED', 'FAILED')) IS NOT TRUE
$audit_remote_command_blocks_purge$;
