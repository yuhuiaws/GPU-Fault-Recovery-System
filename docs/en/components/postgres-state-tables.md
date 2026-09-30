# Workflow And Remote Command State Tables

English edition of `docs/components/postgres-state-tables.md`; the Chinese file remains the source of record until both are maintained together.

This document preserves the 2026-09-07 Store F2 table-split design and records the implementation constraints that followed. The original draft came from
the ignored `docs/review/fragments/` in the `store-review-fixes` worktree; the original file is left untouched.
This file lives in the normal documentation tree and no longer depends on that worktree's lifecycle.

This implementation appends v15 (remote_command) and v16 (workflow) after schema v14, keeping the original wake-up semantics.
The later v17 fixes the conditional-delete compatibility between the raw JSON and the dedicated-table projection during the dual phase.
v18 fixes the lock order of the non-authoritative copy cleanup from legacy to dual, keeping the read/write protocol and the two-phase migration boundary.
The v12, index counts and source line numbers in the original draft below are historical design background, not current implementation facts.
New DDL must be appended as consecutive migrations; the checksums of historical migrations are never rewritten.

## Migration Implementation

`gpu-fault-store-migrate` keeps schema creation explicit. Status, backfill,
counter-state changes, legacy cleanup and logical copy operations open
`PostgresStore` with `initialize_schema=False`; an incompatible source or target
schema fails validation rather than being changed as a constructor side effect.
Prepare the schema through the schema release Jobs first. Only the explicit
schema/index/diagnostic operations may perform their declared schema work.
Copy helpers close an already-open source even when destination initialization
or cleanup fails; the SQLite compatibility reader uses an encoded read-only URI.
Logical copy is not a full PostgreSQL database backup. A nonempty dedicated
telemetry table on either side is now refused before any insert, because the
control-record view does not contain that state. Use a reviewed native PostgreSQL
backup/restore for such databases; do not purge data or change production modes
to bypass the refusal. Imported legacy hot-state records are backfilled into
their dedicated tables in the same destination transaction before success.

The first phase adds v15: `gpu_fault_remote_commands`, the database migration-mode record, the read view,
dual writes and the old-writer barrier. The default mode is `legacy`. An ordinary deploy only has the independent schema Job
create the structures; it does not enable dual writes, does not backfill, does not switch the data source and does not clean up historical data.
The only exception is a brand-new database: the schema Job runs with `--fresh-control-state-mode dedicated`, and after creating the tables and recording the migration history on a database with no
schema at all, it seeds both mode kinds directly as `dedicated` (revision 1,
`backfill_complete=true`, `legacy_purged=false`), because an empty database has no data to migrate; the implementation lives in
`store/postgres/fresh_control_state.py`, changes no DDL, and therefore involves neither migration checksums nor the schema version.
A database that already has `gpu_fault_schema_migrations` keeps its recorded modes regardless of that parameter and must still go through the
explicit legacy → dual → dedicated below. Before seeding, it re-checks that both rows are the freshly created defaults and that the three related tables are empty; otherwise
it reports an error and stops instead of switching silently.

The single source of truth for the mode is `gpu_fault_control_state_modes`, not each Pod's environment variables.
The new Store writes through database routing and reads through mode-aware views; so several processes can never hold different
mode caches. In `dual` the authoritative write still goes to the old table, and an AFTER trigger mirrors the writes and deletes that actually succeeded
in the same transaction; `dedicated` writes only the dedicated table and refuses old clients that keep writing the old table.

The dedicated tables lift status, identity and lease into columns. A remote command's large `snapshot` takes no part in the lease-renewal write;
`lease_expires_at`, lease owner/token and `updated_at` are not indexed, so that HOT updates are preserved.
Timestamps are stored as `TIMESTAMPTZ` with a no-timezone marker; the compatibility view restores the original fixed six-digit microsecond JSON,
so the semantics of timezone-less times, whole-row CAS and existing cursors do not change. Status changes keep sending the v14 wakeup;
lease renewals stay silent.

The maintenance commands below prefer the current `GPU_FAULT_STORE_URL_FILE` projected credential in the controlled session;
`GPU_FAULT_STORE_URL` is used only when no file is configured. An unreadable, empty or mis-encoded file is refused immediately,
with no fallback to the old start-time password. An explicit DSN that conflicts with the projection target is refused too; logical copy between two PostgreSQL databases
must run in an independent controlled process without a process-level projection override, so that source and target are not both redirected to the same database.
Never write a DSN containing a password into command arguments or logs. Finish the CPU rollout and acceptance of the corresponding release first, then run the migration:

```bash
gpu-fault-store-migrate --state-table-status --state-table-kind remote_command
gpu-fault-store-migrate --set-state-table-mode dual \
  --state-table-kind remote_command --expected-state-table-mode legacy
gpu-fault-store-migrate --backfill-state-table --state-table-kind remote_command
gpu-fault-store-migrate --state-table-status --state-table-kind remote_command
```

The backfill defaults to 100 rows per batch and at most 25 batches per run; the cursor and the completion state are committed together with that batch's writes;
while it is not finished, re-run the same command. A lock conflict or an invalid record rolls back the current batch and does not skip over unprocessed records.
New writes are covered by dual writes even when they land before the cursor. `--restart-state-table-backfill` explicitly re-verifies from the beginning.
Unfinished batches return only progress and row counts, with `verification_performed=false`, and never repeatedly scan all the large snapshots.
Backfill completion, an explicit `--state-table-status` and the switch to dedicated still run full verification.
Verification uses a server-side cursor and does not load all the large snapshots into the deploy host's memory at once; SQL waits and cursor reads share
a 30-second verification budget, and a timeout does not return a success report. The connection is limited to 10 seconds, the maintenance-lock wait to 5 seconds.
Maintenance commands first hold the shared schema barrier and check the current release's schema version, the complete migration history and
the state-table definitions; on a mismatch or a disabled trigger they refuse to change the mode, backfill or decommission. That session barrier does not serialise business DML,
but it does prevent a concurrent schema ensure from happening after verification.

Old records may lack optional fields or use early time strings. A conditional write locks the authoritative row in the same transaction, compares against the current model,
then performs the CAS with the raw JSON; a real field change is still refused. v17's dual conditional delete can match either the complete raw
JSON or the complete dedicated-table projection of the same row, and no longer wrongly returns not-deleted because of normalised times or filled-in optional columns.

After the stabilisation period, switch during a maintenance window with the remote command and workflow leases drained:

```bash
gpu-fault-store-migrate --set-state-table-mode dedicated \
  --state-table-kind remote_command --expected-state-table-mode dual \
  --confirm-state-table-change DEDICATED
```

The switch holds the database barrier lock and verifies that the backfill is complete, that there are no missing/extra/inconsistent records, and that the records are canonical JSON of the
current model. A failure leaves the mode unchanged; a verification timeout is refused as well. Old READ COMMITTED writes waiting for the lock read the
new mode and are refused; old REPEATABLE READ/serializable writes fail because their snapshot is stale.
`dedicated` cannot go back to `legacy/dual`. When `dual` is cancelled and then re-enabled, the non-authoritative dedicated-table copy is cleared and
backfilled again, so a stale copy is never reused.

v18's mode switch no longer queues for the exclusive mode barrier; when an active writer exists it refuses immediately and the caller retries within
the original maintenance procedure. The non-authoritative copy is cleaned up with row locks and `DELETE`, avoiding a deadlock between `TRUNCATE`'s table lock
and business transactions that first read the compatibility view and then request the write barrier. Copy cleanup is allowed only on the restricted path that is in legacy, names the kind explicitly
and only DELETEs; as long as a skipped locked row still exists, the whole mode change is rolled back.
This operation produces ordinary MVCC dead tuples and does not free space immediately the way TRUNCATE does; the 30-second statement budget is kept,
a very large old copy still needs a scheduled maintenance window, and a partial cleanup is never passed off as dual being complete.

After the dedicated stabilisation period, explicitly decommission the old rows and old indexes:

```bash
gpu-fault-store-migrate --purge-legacy-state-table \
  --state-table-kind remote_command --confirm-state-table-change PURGE_LEGACY
```

Old rows are deleted in batches; old indexes use `DROP INDEX CONCURRENTLY`. An interruption can be re-run; decommissioned indexes are no longer
rebuilt by schema ensure and no longer required by the business start-up validation. Control-record archiving and cross-PostgreSQL export read
the logical data source and must not read only the remaining `gpu_fault_objects`.
Online index building takes the exclusive session lock of the same schema maintenance barrier before reading the required-index inventory, mutually exclusive with ensure,
decommission and mode switches; on contention it returns a retryable error and does not rebuild decommissioned indexes from a stale inventory.
That lock does not serialise business DML and is released on success or on abnormal exit.

The second phase, v16, connects the workflow's status, version guard, lease and scheduling-time columns to the same mechanism;
`gpu_fault_workflow_records` keeps the dispatcher's time-string and pagination-cursor semantics.

Orphaned-command cleanup is done by `WorkflowStore.cancel_orphaned_remote_commands`, which completes the command status update and the workflow audit event write
in the same transaction. It takes the command and workflow locks in the established lock order,
validates the complete terminal snapshot inside the locks, and the audit contains only the commands that actually changed. That contract covers `legacy`, `dual`
and `dedicated`; an audit failure must not leave behind commands that were cancelled but that a retry no longer records.
A `LEASED` command has only received a cancellation request; it does not mean the remote action has stopped.
Workflow lease renewal updates only the lease columns and keeps the "do not write while more than half of the lease remains" optimisation; budget, merge,
predecessor, reconcile and archiving keep using the original business rules.
Mode filtering is applied uniformly outside the UNION so that PostgreSQL can keep each branch's ordered index path;
`dispatch_eligible_at` projects the original GREATEST expression inside the branch, so pagination does not lose
index order because of the compatibility view. The not-enabled legacy branch of dedicated is still pruned once by a condition.

The two mode kinds are independent. After remote_command is accepted and stable, workflow uses the same command family with
`--state-table-kind` changed to `workflow`, and runs dual, backfill, verification, dedicated and post-stabilisation
decommission separately. Both phases and the later fixes use consecutive migrations, but creating the structures enables no data migration of either kind.
When functions, triggers or views drift, business start-up only refuses and does not repair production structures on its own.
Trigger validation is bound to the current schema and target table and does not accept a same-named object from another schema as a substitute;
function validation covers the signature, return type, default parameters, STRICT, parallel attributes and the function body.
When the migration-mode row or an established registry is lost, ensure also refuses to re-seed legacy, so that dedicated-table data is not
hidden behind a wrong mode; a no-change ensure performs no mode INSERT and takes no RowExclusiveLock on that table.
A lost dedicated table that was already switched to dedicated must be restored first; ensure will not create an empty table posing as the original authoritative data.

A remote command with an expired fencing token can still save late execution evidence, but the result must match the lease that was issued;
a result for a different lease or for an unclaimed command cannot overwrite the status that way. Archiving must prove that the related commands and workflow
have terminated; an unknown, missing or empty status is not proof of termination.
An external successor's unknown status keeps its predecessor record, and an empty `failure_handled_at` does not prove the failure was handled.
The three incident-audit SQL scripts refuse requests with missing parameters,
mismatched confirmation or unmet cleanup conditions through `ON_ERROR_STOP` and explicit SQL exceptions, without relying on the exit-code `\quit` that older `psql` does not support.
preview also lists remote commands with unknown status, and a failure of any conditional delete in purge rolls back the whole transaction.
The three incident-audit scripts first filter by the typed incident and predecessor
columns of the workflow/remote-command views, then rebuild the payload for the selected rows. The OR branch for non-state objects explicitly excludes these two kinds, so that the outer OR does not again
project all the large snapshots early; the count query does not rebuild the state JSON. The legacy fallback is kept, so it cannot be
claimed that queries are equally accelerated in every mode. A legacy successor record with a missing or NULL incident ownership still
appears in preview and blocks purge as long as it references the target workflow, and is no longer skipped by SQL NULL comparison.

The expiry cleanup of hot state, legacy Observations and raw evidence runs
`FOR UPDATE SKIP LOCKED` when selecting candidates; rows being refreshed are left to later rounds, and a just-updated version is not deleted by its old key.
Raw evidence still first excludes records pinned by an open incident, and the LIMIT is spent only on deletable unlocked candidates;
after a refresh transaction rolls back, the old expired version can be selected again in the next round, without changing the existing node-evidence hard-cap rule.
PostgreSQL Collector state batch updates take transaction-level locks on the sorted state keys, then read, merge and write;
non-existent keys are protected too. A newer observation still keeps the existing success/error times and unresolved
`rejected-event:` records under the shared merge contract; a timestamp-only UPSERT guard alone cannot be relied on to overwrite concurrently committed history.

When the Processor claim window is full but blocked by observations/leases, subsequent calls continue the search from the index tuple instead of
repeatedly scanning the same prefix. Each call still keeps the head and the aging window so new high-priority and just-unblocked requests are seen promptly;
while continuing the search it also checks earlier ready STRICT predecessors, preventing retries of the same lane from being skipped.
The window and claim limits do not grow, and there is no unbounded rescan within one call; each Store keeps at most 64 discardable progress hints
for filter scopes, updated only after commit, while leases and fencing are still adjudicated by the database.
The scope intersection predicate is aligned with the existing GIN expression, avoiding per-scope path scans; whether GIN is chosen still depends on
maintenance state, the pending list and optimiser cost, and a local EXPLAIN cannot be taken as Aurora's fixed plan or as a latency promise.
Archive candidates exclude open remote commands and external successors before the LIMIT, so that stable old records
do not block and exhaust each round's quota; the post-upload SERIALIZABLE re-verification is kept. The archive withheld count is the number of
refusals after an actual attempt and does not include records already excluded in the candidate SQL.
Each `ControlRecordArchiver.archive_one` fixes a single retention cutoff instant and checks the actual incident's `updated_at` both at initial packaging and
in the final post-upload transaction; a direct call cannot bypass the retention period either.
When a refresh happens after candidate selection, or a timestamp is missing or malformed, deletion is refused and archiving does not continue on the old candidate ID alone.

Local PostgreSQL verification does not mean Aurora has created the tables, backfilled or switched; the two production phases still need their own release,
maintenance approval, acceptance and stabilisation cycles.

## Implementation Boundaries

- Implement remote_command first, then workflow; each migration surface keeps its own legacy, dual and
  dedicated states, and never backfills, switches or cleans up old data automatically at application start-up or in an ordinary deploy.
- Keep the common contract, lock order, CAS, fencing, cancellation and idempotency semantics of the three Store backends.
- The PostgreSQL dedicated tables lift the frequently changing status and lease fields into columns; lease renewal must not rewrite the large JSON snapshot.
  HOT updates also depend on whether the updated columns are indexed, on in-page space and on the actual execution plan; fillfactor alone
  cannot be used to claim that performance has improved.
- The backfill must be bounded and resumable and must not overwrite concurrent new writes; before switching to dedicated, prove completeness and stop old writers
  from writing back to legacy after the switch. Old indexes may be dropped only in the supervised decommission procedure.
- The current version's wakeup, archive, dispatcher cursor, reconcile, admin and SQL audit reads
  are all in the migration scope; changing only `_get`/`_put` while leaving direct SQL bypasses is not allowed.
- SQL, concurrency, EXPLAIN and HOT behaviour are verified only on an isolated PostgreSQL. The two production phases still need independent
  schema releases, maintenance approval, backfill acceptance and stabilisation periods; this code implementation does not mean Aurora has been migrated.

## Original F2 Design

### Why It Was Not Done in the Original Round

The complete form of review item F moves `workflow` and `remote_command` out of
`gpu_fault_objects(kind, key, payload)` into their own dedicated tables, lifts status/lease/timestamps into columns,
and keeps only the immutable part in the payload. This is a schema migration with data backfill; by repository precedent
(the processor queue's `GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE`
legacy/dual/dedicated tri-state, hot-state's `--backfill-hot-state`) it needs: new-table DDL,
dual writes, a backfill CLI, a status CLI, start-up validation, rewritten read/write paths for three backends, EXPLAIN regression tests,
and rewrites of every script that reads `kind='workflow'` / `kind='remote_command'` directly.
The original draft's inventory of read sites (counted 2026-09-07):

| File | workflow references | remote_command references |
|---|---|---|
| `src/gpu_fault/store/postgres/workflows.py` | 33 | none |
| `src/gpu_fault/store/sqlite/workflows.py` | 29 | none |
| `src/gpu_fault/store/shared/transactional_workflows.py` | 18 | none |
| `src/gpu_fault/store/postgres/ddl.py` | 6 | 4 |
| `src/gpu_fault/control_record_archive.py` | 6 | 2 |
| admin config modules | 10 | none |
| `src/gpu_fault/store/postgres/control_records.py` | 3 | none |
| `src/gpu_fault/store/sqlite/remote_commands.py` | 2 | 22 |
| `src/gpu_fault/store/postgres/remote_commands.py` | 2 | 18 |
| regional performance suite and cleanup | four files | none |
| stuck-workflow and node-health atomicity audits | 2 | none |
| incident-audit export/preview/purge SQL | 3 | none |

The original round judged that changing all of these sites at once, while only being able to verify against a local postgres:16 and not rehearse the backfill against Aurora,
had too large an error surface; it delivered instead the three direct mitigations F1 (fewer whole-row lease rewrites), B (guarded writes) and G/H (indexes).

### Sources of Write Amplification

- `WorkflowRequest`'s `step_executions`, `events`, `official_steps` and
  `completed_operations` only ever grow; the original lease renewal rewrote the whole row once before and once after every step, and after F1 it writes only
  when the remaining lease is at or below half.
- `RemoteActionCommand` embeds workflow and incident snapshots; every lease renewal rewrote the whole snapshot together with
  `lease_expires_at`.
- The legacy partial expression indexes take `payload` as their input column; a payload update produces a new
  heap tuple, TOAST and matching index writes, and HOT cannot be relied on to eliminate this cost.

### Original Target DDL

```sql
CREATE TABLE gpu_fault_workflows (
    request_id                 TEXT PRIMARY KEY,
    incident_id                TEXT NOT NULL,
    status                     TEXT NOT NULL,
    blocked_kind               TEXT,
    fencing_token              BIGINT NOT NULL,
    execution_epoch            BIGINT NOT NULL DEFAULT 0,
    execution_owner_id         TEXT,
    execution_lease_expires_at TIMESTAMPTZ,
    merge_revision             BIGINT NOT NULL DEFAULT 0,
    predecessor_workflow_id    TEXT,
    preempt_predecessor        BOOLEAN NOT NULL DEFAULT FALSE,
    not_before                 TIMESTAMPTZ,
    failure_handled_at         TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL,
    updated_at                 TIMESTAMPTZ NOT NULL,
    payload                    JSONB NOT NULL
) WITH (fillfactor = 70);

CREATE TABLE gpu_fault_remote_commands (
    command_id                TEXT PRIMARY KEY,
    cluster_id                TEXT NOT NULL,
    workflow_request_id       TEXT NOT NULL,
    incident_id               TEXT NOT NULL,
    fencing_token             BIGINT NOT NULL,
    execution_owner           TEXT NOT NULL,
    status                    TEXT NOT NULL,
    status_source             TEXT,
    lease_owner               TEXT,
    last_lease_owner          TEXT,
    lease_token               TEXT,
    lease_expires_at          TIMESTAMPTZ,
    cancellation_requested_at TIMESTAMPTZ,
    cancellation_reason       TEXT,
    error                     TEXT,
    created_at                TIMESTAMPTZ NOT NULL,
    updated_at                TIMESTAMPTZ NOT NULL,
    snapshot                  JSONB NOT NULL,
    result_details            JSONB NOT NULL DEFAULT '{}'::jsonb
) WITH (fillfactor = 70);
```

Index principle: no indexes on lease columns, in particular `execution_lease_expires_at`, `lease_expires_at`,
`lease_token` and `lease_owner`. The claim's expired-lease condition is filtered after the cluster/status/created_at
indexes narrow the range. The original draft suggested turning the related legacy expression indexes into column indexes.
The snapshot keeps the large objects such as step, workflow, incident and restart authorization;
the actual implementation must check every field and its mutability against the current model, and must not assume the whole workflow payload is immutable.

### Original Migration Order

1. Add the empty dedicated tables and column indexes; online indexes are created CONCURRENTLY by the index-build Job.
2. `GPU_FAULT_WORKFLOW_STATE_MODE` / `GPU_FAULT_REMOTE_COMMAND_STATE_MODE`:
   from legacy (read and write objects) to dual (dual write, read the dedicated table, fall back when missing) and then to dedicated.
3. Provide backfill/status CLIs that UPSERT in batches; switching to dedicated is allowed only once missing or inconsistent rows reach zero,
   and start-up validation uses EXISTS rather than every process repeatedly counting the whole table.
4. After dedicated has been stable for one release cycle, clean up the old rows and old partial indexes under supervision.
5. Schema, enabling dual writes, backfill verification, the final switch and old-state cleanup are handled as independent operations phases.

Do remote_command first: few read sites, large snapshots, frequent renewals, and claim/complete/renew/cancel
share the same per-command lock. Then do workflow: it must cover dispatcher pagination,
the three reconcile pieces, admin export and the SQL audits at the same time.

### Mitigations Delivered in the Original Round

- F1: the workflow lease is not rewritten while more than half of it remains; the three backends behave identically.
- B: `save_workflow` guards merge_revision, execution_epoch and fencing_token, and
  `expected=` keeps the whole-row CAS.
- G/H: covering indexes added for metrics aggregation and hot queries, reducing heap reads of the payload.
