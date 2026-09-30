English edition of `docs/扩展指南.md`; the Chinese file remains the source of record until both are maintained together.

# GPU Fault Extension Guide

This guide defines the registration points and the mandatory gates when adding new code capabilities. The goal is to avoid "the code runs,
but silently acquires default semantics because one collection was not updated".

As soon as an extension adds a production Manifest, systemd unit, AWS resource, configuration, credential or administrator command, you must
also follow [Developer Deployment Implementation](developer-deployment-implementation.md). The production delivery and uninstall contracts are maintained only in that file;
this guide does not repeat them.

## 1. Adding a Workflow Operation

1. Add an enum member to `models.WorkflowOperation`.
2. Add a complete semantic row to `operation_registry.OPERATION_REGISTRY`:
   capability, scope, destructive, rank, adapter, resource locks, DAG, preempt
   and barrier; plus the following decision fields:
   - `hardware_escalation_relevant`: may serve as attempt evidence for hardware escalation.
   - `host_proc_root_dependent`: requires the host `/proc` mount; the Node Agent
     verifies the runtime path at startup (`HOST_PROC_ROOT_OPERATIONS`).
   - `maintenance_generation_scoped`: the NodeAction must inherit the maintenance generation
     captured at quiesce time rather than read the live fleet endpoint
     (`MAINTENANCE_GENERATION_OPERATIONS`).
   - `generation_stable_command_id`: the `command_id` of long-running, mutating reinstall-type NodeActions (driver / firmware / EFA)
     carries no `agent-N` suffix, so a re-post after an Agent restart hits the same ledger row instead of executing a second time;
     mutually exclusive with `maintenance_generation_scoped` (`GENERATION_STABLE_COMMAND_OPERATIONS`).
   - `planning_only`: does not land on any adapter; participates only in orchestration decisions.
   Beyond `destructive`, `CONTAINMENT_ONLY_OPERATIONS` is also derived
   (`resource_claims` is only `SCHEDULER_MUTATION`, i.e. cordon/quarantine/
   restore scheduling) as well as `NODE_MUTATING_OPERATIONS` (the remaining destructive operations); within the latter,
   operations whose `resource_claims` contain only the device-plugin claim are further derived into
   `DEVICE_PLUGIN_RESTART_OPERATIONS`: restarting the plugin Pod only makes the kubelet re-register devices,
   containers already holding devices are unaffected, so the data-plane STOP ownership guard does not demand STOP receipts for them
   (the remaining merge / budget / busy-node rules still treat them as node-mutating).
   The two gates, boot generation fence and workload_state UNKNOWN, look only at the latter:
   refusing isolation because evidence is stale or the workload condition is unknown amounts to leaving a suspicious node schedulable.
3. Implement the execution logic in the target adapter. `OPERATIONS` is derived automatically from the registry;
   no independent collection may be maintained any more; derived collections with more than three members must not re-list literals at call sites,
   and `tests/test_registry_contracts.py` scans the AST to reject duplicates.
4. Add policy, workflow builder, adapter and negative tests.
5. Run `tests/test_operation_registry.py` and
   `tests/test_registry_contracts.py`. A missing registration fails directly at import time.
6. Before deployment run `gpu-fault-config validate` to check the generated Manifests, the control-plane allowlist and
   the Node Agent allowlist. This command does not read the Runtime Profile; Profile claims, observed
   capability and owner consistency are checked by `tests/regional/test_capabilities.py`, the regional release
   verifier and `gpu-fault-admin status --full`.
7. Loading a plugin does not mean automatic authorisation: the adapter must still match the operation registry, the control-plane
   allowlist and the Runtime Profile.

Dynamic plugins cannot add new `WorkflowOperation` enum values. External adapters can only provide implementations for existing operations;
adding a core operation must go through the core registry review above.

## 2. Adding a Collector / Processor Channel

1. Register path, priority mode, pool,
   lane, spool, edge filter, latest-wins and routine reason vocabulary in `channel_registry.CHANNEL_REGISTRY`.
2. Create the FastAPI route and the Collector sink call using the path constants exported by the registry.
3. Add route and processor tests. Application startup compares all
   `/v1/collector-events/*` routes with the registry; a missing registration fails directly.
4. New baseline spellings must be added to that channel's `routine_reasons` or prefix;
   collectors must not invent synonyms on their own.
5. Requests with `latest_wins=true` and an actual priority=100 automatically become
   `REORDERABLE`; all other requests are `STRICT`. A new channel must be tested for whether later requests in the same lane may overtake
   during delayed retries; fault/evidence channels must not be mislabelled as reorderable.

## 3. Supporting a new GPU product family

The product family mapping comes from a generator; do not edit
`src/gpu_fault/data/nvidia-xid-catalog-610.generated.yaml` directly. Add to the generator input of
`tools/generate_nvidia_xid_policy.py`:

```yaml
spec:
  catalog:
    productFamilies:
      - family: Z100
        modelPrefixes: [Z]
```

No Python regex needs changing. An unknown product produces
`gpu_fault_policy_unknown_product_total{product=...}` and triggers
`GpuFaultUnknownProductFamily`; automatic recovery stays fail-closed.

After regenerating with an approved NVIDIA Catalog XLSX with a pinned digest, update the pinned source/generated
SHA tests and run:

```bash
make PYTHON=.venv/bin/python xid-catalog-check
```

The Catalog is packaged data inside the wheel; a change alters `module_digest()` and requires a full rebuild of the
release rather than replacing the file online.

## 4. Adding a Store Domain or backend

1. Extend the Protocol in `store/contracts.py` with concrete model types; `Any` is forbidden.
2. All three implementations implement or explicitly inherit the method.
3. When the generic object codec is used, register the SQLite/PostgreSQL `_models` at the same time; when adding a dedicated table,
   spell out the dual/dedicated migration, backfill, cleanup and old-codec compatibility paths.
4. Add the behaviour to the shared contract fixture; when a PostgreSQL URL is configured, Memory, SQLite and
   PostgreSQL must run the same tests.
5. Run the mypy boundary check, the override shape check and the real PostgreSQL tests, and update the inventory of the
   current 48 logical kinds in the Detailed Design at the same time.

## 5. Adding a Schema migration

The latest version in the current registry is v18
`control-state-legacy-copy-reset-without-truncate`. The next migration must append v19:

1. Modify the latest idempotent schema in the corresponding `store/postgres/ddl*.py`; the checksum covers all
   `ddl*.py`, not just `ddl.py`.
2. Compute the new DDL source SHA-256.
3. Append a consecutive version to
   `POSTGRES_SCHEMA_MIGRATIONS` in `src/gpu_fault/schema_migrations.py`; historical rows must not be modified.
4. When backfill, type changes or similar steps are needed, provide an `apply(cursor)` callback.
5. New columns must remain writable by old code: use nullable columns, or provide a deterministic non-null default like
   `retry_count/lane_policy` in v7.
6. Modifying the DDL directly without adding a migration fails at import time.
7. Run `make test-postgres-stress` and the CAP-005 runner, covering lock contention, migration history,
   completion concurrency, counters, fencing, deadlock and zero skips.

## 6. Adding a notification Builder, Adapter or Sink

The project declares the following entry-point groups:

- `gpu_fault.collector_sinks`
- `gpu_fault.workflow_adapters`
- `gpu_fault.notification_builders`
- `gpu_fault.metric_contributors`

Plugins with the same name in the same group are failed closed by `gpu_fault.plugins.discover_plugins()`. Currently only
`gpu_fault.metric_contributors` is discovered and assembled automatically by the runtime.

Collector sinks, Workflow adapters and Notification builders are still assembled statically by their respective factory/registry.
An external package merely declaring an entry point does not take effect automatically; when adding these three kinds of extension you must also explicitly wire
the corresponding assembly point, configuration selection and tests. "Entry point installed" must not be taken as proof of production readiness.
The built-in `diagnostic-inconclusive` and the other shipped Notification builders keep their declarations consistent with the explicit allowlist;
checking the completeness of distribution registration does not amount to permitting arbitrary plugin installation, still less does it change the default refusal of metrics plugins.

The `gpu_fault.collectors` extension of the Collector CLI first discovers only names and metadata, rejects duplicate names and
conflicts with built-in commands, `outbox` and `validate-plugins`, and only then loads the explicitly selected plugin. Ordinary built-in
commands and help output do not import unselected plugins, so an unrelated broken plugin cannot block collection or outbox operations.
Installation or release candidates use `gpu-fault-collector validate-plugins` to fully validate all descriptor
and factory imports, without creating Collector instances, connecting to the control plane or performing collection.
That full validation must run in a verified candidate environment; on-demand loading does not replace distribution and dependency integrity verification.

## 7. Adding an HTTP route

Every FastAPI handler must explicitly add:

```python
@router.post("/v1/example")
@authorization_bucket("execution-token")
async def example(...):
    ...
```

The only allowed buckets are `public`, `metrics`, `cluster-token`,
`dual-credential` and `execution-token`. Directory prefixes do not inherit permissions;
a missing declaration fails at application startup.

`create_app()` currently mounts 89 application routes; after adding or removing routes you must re-assemble the real app to recount the total and the five-bucket
distribution, and update the two high-level designs and the two detailed designs. The non-regional local mode also mounts these routers,
but only the regional production mode installs the five-bucket default-deny middleware.

If the request schema contains a cluster identity, the field name must be registered in
`app.cluster_binding.CLUSTER_IDENTIFIER_FIELDS`. Currently
`cluster_id`, `clusterId` and `cluster` are supported, and all nested values are bound to the authenticated cluster.

## 8. Hand-off to production delivery

When a code extension involves any of the following changes, continue with
[Developer Deployment Implementation](developer-deployment-implementation.md):

- adding or modifying a Kubernetes Manifest, renderer, systemd unit or container image;
- adding a `site.yaml` field, environment variable, Secret, IAM or other AWS resource;
- changing the Runtime Profile, the Agent/Executor protocol or a required/compatible pin;
- adding an administrator command, deployment step, cleanup phase or irreversible action;
- needing to enter `preflight/deploy/verify/status/remove-cluster/uninstall`.

The historical in-cluster GPU metrics DaemonSet manifest has been deleted;
the GPU metrics collector is delivered by node systemd. Reintroducing a DaemonSet of this kind is a producer
migration and must first design dual-producer protection, node uninstall, rollback and resource registry changes.

Production delivery must cover the three layers of resource fact sources:

1. Kubernetes `gpu-fault-installed-resources`;
2. node `/opt/gpu-fault/installed-units.txt`;
3. Aurora `installation_resource` AWS registry.

The Extension Guide only decides "where the capability is registered"; the Developer Deployment Implementation decides "how it is generated, released, verified, rolled back and
uninstalled". When both kinds of change are present, the gates of both documents must be executed.

## 9. Adding Metrics

Implement a metric lines contributor for the subsystem and register it in the metrics registry; do not keep appending
string blocks to the core render function. Metric families are counted by actual role and by assembly conditions such as PostgreSQL/regional;
a historical baseline must not be taken as the current total for all roles; after adding, recount and update the prefix distribution in the Detailed Design.
The previous closed-loop round added four gauge families, and the same in-memory worker fixture went from 257 to 261,
with no increase for ingress/spool-worker. This round adds two counter drift scan evidence gauges and one terminal-state notification
event time gauge; the same three-role fixture gives 225/264/224 families for ingress/worker/spool-worker respectively,
an increase of 3 families each. The processor prefix is 111 families in all cases; the notification prefix is 8/12/8 families respectively.
These fixtures do not enable the conditional PostgreSQL, regional and refresh-file metrics and cannot be taken as the production inventory.
If a metric is used for AMP alerting, you must also, at the same time:

1. add it to the ADOT keep regex;
2. add a production rule in `deploy/observability/amp-rules.yaml`;
3. keep the rule equivalent on the optional
   Prometheus Operator path in `deploy/control-plane/regional/processor-alerts.yaml`;
4. run the alert reachability tests of `tests/regional/test_production_safety_config.py`.

The regional production control plane does not run Prometheus Operator by default, so modifying only
`processor-alerts.yaml` amounts to having no production alert; the AMP rules and the ADOT keep allowlist are the normal path.
