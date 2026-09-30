# GPU Fault Collectors

English edition of [`COLLECTORS.md`](COLLECTORS.md); the Chinese file remains the source of record until both are maintained together.

## Data Paths

`gpu-fault-collector` is a read-only collection process; it performs no node or job recovery:

| Collector | Data source | Control-plane entry | Idempotency ID |
|---|---|---|---|
| `kernel` | Node `/dev/kmsg` | `/v1/collector-events/nvidia-kernel` | boot ID + kmsg sequence |
| `fabric-manager` | Fabric Manager journal/file | `/v1/collector-events/fabric-manager` | journal cursor or file record ID |
| `kubernetes-node-resources` | Node GPU/EFA allocatable | `/v1/collector-events/host-telemetry` | node + sample timestamp |
| `dcgm` | DCGM Exporter Prometheus endpoint | `/v1/collector-events/gpu-metrics` | node + scrape timestamp |
| `nvidia-smi` | Local NVIDIA CLI | `/v1/collector-events/gpu-metrics` | node + sample timestamp |

The kernel collector only sends lines containing `NVRM ... Xid` or `SXid`; it does not send the full kernel log.
Kernel/FM share `gpu_fault.nvidia_logs.NvidiaLogNormalizer` and do not depend on HMA Node or
CloudWatch forwarding. AWS HMA itself is not a collector of this solution.

The idempotency ID in the table above is sent with the request as the HTTP `Idempotency-Key` header, and the control plane deduplicates by it. The sink only retries requests
**that carry** this header: 4 attempts by default with backoff, honouring `Retry-After` but waiting at most 30 seconds;
a payload that cannot produce an idempotency ID is sent once, and on failure is written to the outbox. `snapshot_id` (gpu-inventory
snapshot) also has an idempotency ID; the historical outbox `log_event_id` and `node/<name>/<resourceVersion>`
identity rules are retained, so record identity does not change because a collection entry was retired. The cost is that when the control plane is unreachable
`post()` blocks longer (4 attempts plus backoff); callers must not assume it returns quickly.

An HTTP 2xx response whose body is not a JSON object is treated as **result unknown** and goes through the same retry
ladder as a network failure; if it can never be parsed it enters the outbox as a replayable record. Retries and replays carry the same `Idempotency-Key`,
so a request that was in fact received is deduplicated on the control-plane side rather than stored twice.

The default HyperPod production topology enables:

- the kernel, fabric-manager, dcgm and host node collectors;
- the kubernetes-node-resources cluster collector;
- the completion watcher providing attempt/workload observations.

`NodeLogCollector` is currently disabled by default and is not part of the production-required set. The installer enables it only when
`--enable-node-log-collector` is passed explicitly in an isolated validation environment.

`nvidia-smi` is kept as the fallback when DCGM Exporter is unavailable; both modes share the same inventory cadence: after 3 consecutive inventory
validation failures (for example a misconfigured `GPU_FAULT_EXPECTED_GPU_COUNT`) the collector backs off at the inventory interval instead of failing the whole
round, and the node keeps emitting GPU_METRICS batches with samples; when temperature thresholds cannot be read on vGPU/MIG it likewise backs off instead of retrying every round.
The two optional Kubernetes HMA and CloudWatch HMA paths have been retired; the related commands, APIs and deployment assets have been deleted.

The FM file cursor also stores the observed truncation generation. A rename keeps the same file identity; when copytruncate
causes an already-sent offset to be reused, the new generation is persisted first, then a different record_id and evidence_ref are generated.
An ordinary process restart within the same boot keeps the retry identity; a missing or corrupt cursor still establishes a new baseline at EOF and does not replay old logs.
Polling cannot recover a truncation history that completed between two observations without leaving a size/identity change, so no atomic log protocol is claimed.

FM emits bounded `GPU_FAULT_FM_RECEIPT_V1` receipts through `fabric_manager_receipts.py`:
bound to the producer/systemd invocation, PID, sequence and cumulative counts; record and scope are emitted as digests only,
and no raw identity, payload, URL or free-form error text is recorded. The first two rounds, actual deliveries and failures are observable;
healthy empty polls are aggregated at the existing summary interval and are not logged at INFO per poll. The delivery-pair budget is 256 per minute,
and each message is at most 2048 bytes; omissions and saturation are recorded explicitly and do not block fault collection.
Acceptance must check the full journal window, attempt/completion and round progress; a gap cannot prove zero replay.
DELIVERED only means the sink acknowledged, BUFFERED only means the outbox holds it; neither means exactly-once downstream or physical recovery.
`GPU_FAULT_FM_RECEIPT_V1` is a log protocol label, not an environment variable, and does not enter the production configuration allowlist.

### Adding a Collector

Collectors, like operations, channels and node-actions, are table-driven:
`src/gpu_fault/collector_registry.py` is the only table to change; `validate_collector_registry()`
validates at import time, so a missing entry fails on the first `import gpu_fault.collectors_cli` rather than on
the first run on a node. Steps to add a collector:

1. **Channel**: if the data lands on a new control-plane entry, first add a row to `CHANNEL_REGISTRY` in
   `src/gpu_fault/channel_registry.py` (processor routing is reconciled by `validate_collector_routes()`);
   skip this when reusing an existing entry.
2. **Factory**: in the collector module write
   `build_from_environment(sink, context, arguments) -> collector`, and put that collector's
   own `os.getenv(...)` reads there (a collector that does not need the context uses the
   two-argument `(sink, arguments)` form and declares `needs_context=False` on its descriptor).
3. **Descriptor**: add a `CollectorDescriptor` to `COLLECTOR_REGISTRY`:
   `cli_command` (subcommand name), `kinds` (the `CollectorKind`s it produces; may be empty, and may be
   shared by several subcommands as `dcgm`/`nvidia-smi` do), `channel_paths` (every path it
   `post`s to), `export_name` (the class name in `gpu_fault.collectors._EXPORTS`),
   `factory` (`"module:callable"`), `runs_in` (`node`/`cluster`/`workload`),
   `needs_product_discovery`. If a new `CollectorKind` is introduced, also add a row to
   `COLLECTOR_KINDS`: producer name, systemd unit,
   `silent_threshold("GPU_FAULT_..._SILENT_AFTER_SECONDS", "<seconds>")`.
   `telemetry.COLLECTOR_PRODUCER_BY_CHANNEL`,
   `collector_requirements.COLLECTOR_SYSTEMD_UNITS` and
   `collector_silent_thresholds()` are all derived from these two tables and are no longer hand-written.
4. **Subcommand arguments**: when argparse options are needed, register an
   `add_arguments(parser)` in `collectors_cli.CLI_ARGUMENTS`; otherwise do not register one, and the subcommand appears automatically.
5. **Deployment**: the unit name under `deploy/systemd/` must equal the
   `systemd_unit` in `COLLECTOR_KINDS`; data-plane manifests live in `deploy/dataplane/`.

The validator rejects: a non-retired `CollectorKind` produced by no descriptor (or vice versa), `channel_paths` outside the
`CHANNEL_REGISTRY`/provider-events prefixes, an `export_name` not exported by `gpu_fault.collectors`,
a dictionary key that differs from `cli_command`, one node subcommand spanning two systemd units, and kinds sharing a unit
that declare different producers. Third-party collectors provide a `CollectorDescriptor` named after the entry point through the `gpu_fault.collectors` entry-point
group (`PluginGroup.COLLECTORS`); at CLI start-up they are merged via `collector_registry_with_plugins()` and go through the same validation; a name
that collides with a built-in subcommand is rejected. The old HMA kinds are kept only for decoding persisted data and cannot be registered as producers by new descriptors or plugins.
`tests/test_collector_registry.py` covers each of the rules above.

## Common Configuration

```text
GPU_FAULT_CONTROL_PLANE_URL=http://gpu-fault-api:8080
GPU_FAULT_CONTROL_PLANE_TOKEN=<optional bearer token>
GPU_FAULT_CLUSTER_ID=<cluster>
GPU_FAULT_RUNTIME_PROFILE_VERSION=<registered profile>
GPU_FAULT_GPU_PRODUCT=H100
GPU_FAULT_DRIVER_BRANCH=575
GPU_FAULT_CUDA_VERSION=12.9
```

Production must first register the corresponding runtime profile with the control plane. The token should be injected from a Secret and must not be written into a
ConfigMap. In regional mode the control plane itself verifies the cluster bearer token: the request looks up the registry by `X-GPU-Fault-Cluster-ID`,
the SHA-256 digest of `Authorization: Bearer` is compared with the registered token digest in constant time via
`secrets.compare_digest`; a missing bearer returns 401, an unregistered or mismatched cluster returns
403; the execution token is likewise compared in constant time, and routes that declare no authorization bucket are denied by default (403), see
`src/gpu_fault/app/middleware/auth.py::install_regional_authorization`. API Gateway,
service mesh, NetworkPolicy, security groups or a private load balancer are still recommended as layered defence in depth, but are no longer the
only token verification point. `scripts/check-doc-facts.py` guards this section against the code.

## Kubernetes

Install the collector dependencies and build the image:

```bash
python3 -m pip install '.[collectors]'
```

The historical generic Kubernetes template lives at
`examples/legacy/kubernetes/collectors.yaml`; it is only a protocol and migration reference, not a supported
production deployment entry. The regional HyperPod data plane uses the `deploy/dataplane/` manifests together with the node
Installer/Reconciler. Cluster-level resource collection is kept by `KubernetesNodeResourceCollector`, so
the GPU/EFA allocatable check is not lost by deleting the HMA watcher. Retiring the self-built forwarding path does not shut down
AWS HMA, nor does it remove the provider health labels on the Node.

`deploy/image/Dockerfile` can be used to build the collector image:

```bash
docker build -f deploy/image/Dockerfile -t gpu-fault-collector:0.6.1 .
```

The kernel DaemonSet reads the host's `/dev/kmsg` and usually needs privileged/CAP_SYSLOG. It should be restricted to GPU nodes via
nodeSelector, and an admission policy should allow only a pinned image digest. If the platform forbids reading
`/dev/kmsg`, a controlled journald/kern.log forwarder can be used instead, but the original kmsg sequence should still be kept
as the event ID.

Local runs:

```bash
gpu-fault-collector kernel --node-id worker-1
gpu-fault-collector kubernetes-node-resources
gpu-fault-collector dcgm --node-id worker-1 \
  --metrics-url http://127.0.0.1:9400/metrics
gpu-fault-collector nvidia-smi --node-id worker-1
```

EC2 or HyperPod Slurm nodes can use
`deploy/systemd/gpu-fault-kernel-collector.service`. After installing the wheel, place the CLI in
`/opt/gpu-fault/venv/bin` and write the common environment variables to
`/etc/gpu-fault/collector.env`.

### One-Command Install on GPU Instances

The node install script deploys both the XID/SXID kernel collector and the GPU metrics collector:

```bash
sudo deploy/node/install-gpu-fault-collector.sh \
  --control-plane-url https://gpu-fault.example.internal \
  --cluster-id <gpu-hyperpod-cluster> \
  --runtime-profile-version hyperpod-v1 \
  --node-id "$(hostname -f)" \
  --token-file /etc/gpu-fault/execution-token \
  --wheel-sha256 "${RELEASE_WHEEL_SHA256}" \
  --metrics-mode auto \
  --dcgm-exporter existing \
  --enable-node-agent \
  --node-action-secret "${NODE_ACTION_SECRET}" \
  --node-instance-id "${EC2_INSTANCE_ID}" \
  --node-agent-advertise-url "https://${NODE_PRIVATE_IP}:9099"
```

`--wheel-sha256` comes from the signed release manifest (not the wheel inside the install bundle), `--token-file` points to a token file with
`0600` permissions; dependencies are installed by default with `--require-hashes` against the hash lock in `requirements/node-runtime.lock`.
The full semantics of the three are given in the security-parameter notes at the end of this section.

Build a self-contained install bundle that can be uploaded to S3, SSM Distributor or a node image pipeline:

```bash
deploy/node/build-node-installer-bundle.sh
```

The output is `dist/<wheel-sha12>/gpu-fault-node-installer-<version>.tar.gz`, containing the project wheel, the install/uninstall/
self-check scripts, systemd units and the DCGM counter configuration. Third-party Python dependencies are not duplicated into the bundle;
air-gapped environments should also prepare a `--wheelhouse`.

`execution-token`, and `--node-action-secret` when the Node Agent is enabled, must be values of at least 32
characters with no trailing newline; an execution token with a newline cannot be used as an HTTP header. When using
Kubernetes Secret files, normalise them first and compare the SHA-256 fingerprints of the control-plane and node node-action secrets;
never write the token or secret itself to logs. When the node's default `python3` is below 3.12,
pass `--python-command /usr/bin/python3.12` explicitly.

Kubernetes/HyperPod installs should prefer
`--node-action-secret-file /mounted-secret/node-action-secret` so the secret does not appear in
command-line arguments. `deploy/node/run-hyperpod-installer-job.sh --node <node-name>` provides
a single-node-bound Job transport that can be used directly with the wave-by-wave coordination of fleet deployment.

Neither the standalone node installer nor the HyperPod production deployment enables `NodeLogCollector` by default.
The current production policy requires it to stay disabled; enable it explicitly via
`--enable-node-log-collector` only for dedicated validation. When enabled, the collector only sends lines that match the shared
`NODE_LOG_RULES`, and sends an empty health summary every 300 seconds; it does not upload whole journal segments.

The journal is read as a stream under a per-round budget and the whole window is never loaded into memory: a single poll stops at the first of three caps -- the "entries to
keep in the batch" count, 4 MiB and 200 ms -- and the cursor stays at the last entry actually read
(the historical implementation used `capture_output=True` to collect the whole window at once; nodes with high log volume were OOM-killed first,
the cursor was not saved, and the next start re-read the same window). The batch entry cap (`MAX_ENTRIES_PER_BATCH`,
default 1000) and the 4 MiB are **shared** between the journal and training logs: when training logs are configured,
`max(1, cap//4)` entries and bytes are reserved for them first and the journal uses only the remainder; when not configured the journal gets the whole budget. Everything
held back by the budget is counted in the discard counters, never dropped silently:

| discard key | Meaning |
| --- | --- |
| `journal-window-capped-seconds` | Time span (seconds) abandoned when the cursor was older than `MAX_JOURNAL_WINDOW_SECONDS` (default 900); that journal segment will not be read again |
| `unparseable-journal-entries` | Number of journal records read but unparseable; they are lost once the cursor advances, and are read again next round if the cursor is kept |
| `deferred-training-log-reads` | Number of training log files not read this round because the budget ran out; the offset is recorded and the next round continues after the file where it stopped |
| `unreadable-training-logs` | Number of training log files whose open or read failed (permissions, rotation); the file's offset is still recorded, so it is not re-baselined as a new file |

Only when the training logs are still left with nothing but deferrals for the 3rd consecutive round (never getting budget beyond the guaranteed share) is an extra error reported in the batch: the offset is not lost, but
a backlog exceeding one log rotation will genuinely lose content.

The `training-progress` reporter is off by default. Rank liveness generated by the host collector from procfs
is the default non-intrusive progress signal; the training-progress reporter is wired in only when step/loss/numerical-error semantics
are needed.

Unified self-check of the collection plane:

```bash
gpu-fault-config collector-readiness \
  --url https://CONTROL_PLANE \
  --cluster-id CLUSTER \
  --execution-token "${GPU_FAULT_EXECUTION_TOKEN}"
```

The endpoint merges Agent heartbeats, collector systemd status and each channel's last-success;
`unit_state` and `unit_enabled` are always strings.

With the Node Agent enabled, the Agent registers immediately at start-up via
`POST /v1/fleet/agents/heartbeat` and renews its lease every 30 seconds by default. The heartbeat is HMAC-signed with the
node-action secret and includes endpoint, package version, wheel
SHA-256, NVIDIA Catalog policy version, runtime profile, common configuration digest, boot ID and
allowed operations. Agent protocol v3 also carries the stable Node/instance UID, incarnation and
the state of the five collector systemd units, and obtains the generation from the
heartbeat response; an action command with a mismatched generation fails closed. The install script
prefers an explicit `--node-instance-id`, otherwise it reads the DMI product UUID or machine-id.
The HyperPod installer Job passes the Kubernetes Node UID automatically. `--node-agent-advertise-url`
must be an address the control plane can actually reach; do not rely on node-local hostnames that Pods cannot resolve.
When no cert/key is provided explicitly, the installer generates a local TLS cert/key on the node; the signed heartbeat
carries the server certificate and the control plane connects by certificate pin. Production must not pass
`--allow-node-agent-plaintext`.

Node reboot/replace does not depend on the Agent deregistering itself. The control plane first uses the execution-token-protected
`/drain` and `/revoke` endpoints to raise the generation, terminate the lease and retire the old incarnation. A new Agent
must register with a new boot ID or instance UID before it can return to `ACTIVE`; deleting registry records directly is forbidden.

The control plane coordinates batch upgrades through the following endpoints:

```text
POST /v1/fleet/deployments
POST /v1/fleet/deployments/{id}/next-wave
POST /v1/fleet/deployments/{id}/nodes/{node}/status
GET  /v1/fleet/deployments/{id}
```

`next-wave` atomically takes the `maxUnavailable` quota. EC2 uses SSM, EKS uses a DaemonSet/
Operator, and HyperPod uses a lifecycle script to execute the wave; the matching heartbeat after installation
automatically marks the node `READY`. While the current wave is not all READY, the control plane refuses to start the next wave.
`READY` cannot be written manually through the node status API; it can only be written by a signed heartbeat whose target agent version, artifact
SHA-256, policy, runtime profile and config digest all match.

`gpu-fault-fleet run-deployment` can execute the full wave-by-wave rollout. The transport command
is invoked once per node in parallel for the current wave; EC2/HyperPod can wrap SSM or a lifecycle script, EKS can wrap
Operator/DaemonSet node selection logic. Exit code 0 only means the deployment request was accepted; the runner still waits for
the target heartbeat and marks the current node `FAILED` on timeout or command failure:

```bash
gpu-fault-fleet \
  --control-plane-url https://gpu-fault.example.internal \
  run-deployment \
  --execution-token "${GPU_FAULT_EXECUTION_TOKEN}" \
  --deployment-id fleet-deployment-... \
  --transport-command \
    '/opt/gpu-fault-fleet/deploy-node --node {node_id} --version {agent_version} --sha256 {artifact_sha256}' \
  --wave-timeout-seconds 900
```

The transport command is not executed through a shell; it supports the `{node_id}`, `{cluster_id}`,
`{deployment_id}`, `{agent_version}` and `{artifact_sha256}` placeholders and also receives the same-named
`GPU_FAULT_FLEET_*` environment variables. After all waves complete, the runner runs the fleet readiness
gate once more; any expired node heartbeat, missing capability or version identity mismatch fails the rollout closed.

`auto` first probes `GPU_FAULT_DCGM_METRICS_URL` and uses DCGM when a DCGM Exporter is present.
`NvidiaSmiMetricsCollector` has not yet completed further production validation and is disabled by default; when DCGM is unavailable the installer
fails closed and no longer falls back automatically. If the node has no existing exporter, a dedicated container separate from HMA can be deployed;
the image tag must be specified explicitly by the cluster administrator according to the driver/GPU support matrix:

```bash
sudo deploy/node/install-gpu-fault-collector.sh \
  --control-plane-url https://gpu-fault.example.internal \
  --cluster-id <gpu-hyperpod-cluster> \
  --runtime-profile-version hyperpod-v1 \
  --dcgm-exporter docker \
  --dcgm-exporter-image nvcr.io/nvidia/k8s/dcgm-exporter:<approved-tag>
```

Docker mode uses the host network, but the exporter listens only on `127.0.0.1:9400` and mounts
the project's DCGM counter list. It does not enter the HMA Pod, does not modify HMA's `nv-hostengine`, and does not
request Kubernetes `nvidia.com/gpu` resources. When the installer detects an existing `nv-hostengine` or DCGM
endpoint it refuses to start a second exporter, avoiding conflicts with HMA, the GPU Operator or existing monitoring.

After installation run:

```bash
sudo /opt/gpu-fault/verify
sudo journalctl -u gpu-fault-kernel-collector -u gpu-fault-metrics-collector
```

Uninstall with `sudo /opt/gpu-fault/uninstall`. The installer requires Python 3.12; offline nodes specify a non-default interpreter via
`--python-command`, and the project wheel and full dependency directory via `--wheel` and `--wheelhouse`.
The access token is written to `/etc/gpu-fault/collector.env` with `0600` permissions and is not written into the systemd unit.
The self-check not only checks local services and the
exporter, but also waits for `/v1/gpu-metrics/.../latest` to show actual samples from this node.

The installer fails closed on supply-chain failures; the three security parameters must be provided together:

- `--token-file PATH` -- read the bearer token from a file with `0600` permissions so the token does not appear in
  `argv` (`--token` is still available but is exposed in the process list; use it only for temporary debugging).
- `--wheel-sha256 HEX` -- the expected project wheel SHA-256, which **must come from the signed release manifest, not from
  the install bundle itself**; the installer compares it with the digest in the release manifest and fails closed on mismatch.
- `--dependency-lock PATH` -- the hash lock for third-party dependencies, default `requirements/node-runtime.lock`
  (covering only the narrow dependency set of the node closure, not the control plane's `runtime.lock`). The installer always installs with
  `pip install --require-hashes --no-deps`, and every distribution in the lock carries `--hash=sha256:`,
  so no unpinned package is ever resolved online.

## GPU Metrics

Prefer `dcgm`, which reads the DCGM Exporter with the official `prometheus_client` parser. Supported:

- GPU/memory temperature, power, GPU/memory utilisation, memory and clocks.
- volatile/aggregate SBE/DBE ECC.
- retired pages, pending retirement and row remap.
- PCIe replay, NVLink CRC/data/replay/recovery counters.
- power/thermal violation duration.
- `DCGM_FI_DEV_XID_ERRORS`.

The DCGM Exporter must be configured with the fields in `deploy/dataplane/dcgm-counters.csv`; both start-up paths
(`deploy/dataplane/hyperpod-dcgm-exporter.yaml` and `deploy/systemd/gpu-fault-dcgm-exporter.service`)
sample with `-c 15000` and bind only `127.0.0.1:9400`; the `-c` value on the systemd path comes from
`GPU_FAULT_DCGM_EXPORTER_COLLECT_INTERVAL_MS` in `/etc/gpu-fault/dcgm-exporter.env` (an installer
parameter, default 15000, must be an integer number of milliseconds >= 1000); with `--dcgm-exporter docker` the installer also writes the value converted to seconds,
`GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS`, into collector.env, and the collector uses it at start-up to verify that
"exporter refresh period <= 8 x collection interval", warning once if out of range (in `existing` mode the period is unknown: nothing is written,
nothing is checked); the GPU metrics collector is run by the node systemd unit
`gpu-fault-metrics-collector.service` and accesses `http://127.0.0.1:9400/metrics` locally. The historical
in-cluster DaemonSet template `gpu-metrics-collector.yaml` has been deleted; a DaemonSet of that kind must not be reintroduced,
otherwise the same node would have two producers.

The direct consequence of binding only `127.0.0.1` is that scraping `:9400` from outside the cluster, across nodes, or from the Prometheus side
is no longer possible; these counters reach the control plane only through `/v1/collector-events/gpu-metrics`. The DaemonSet
does not use a nodeSelector but a `nodeAffinity` that lists the supported GPU instance types exactly by `node.kubernetes.io/instance-type`
(one entry each with and without the `ml.` prefix); the manifest is rendered by
`regional_release_rendering.render_dcgm_exporter_manifest` from the instance-type table in `node_installer_reconciler`;
if a placeholder is left unreplaced, rendering errors and fails closed, so an exporter that collects nothing never starts on an unknown instance type.

Different DCGM/GPU generations may expose old fields and aggregate fields at the same time. The current compatibility contract is:

- row-remap uses `DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS` and
  `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS`;
- NVLink aggregate counts in environments such as H200 use
  `DCGM_FI_DEV_NVLINK_ERROR_DL_CRC/RECOVERY/REPLAY`, while the old total fields are kept;
- when the exporter returns `N/A`, an empty value, or the target GPU does not support the field, the sample is ignored; it must not be converted to the number 0,
  nor may a finding be produced from it;
- `DCGM_FI_DEV_XID_ERRORS` is a "last XID" state, not an event counter.

The `NvidiaSmiMetricsCollector` implementation is kept as a capability for later validation. It may be used only when
`--enable-nvidia-smi-metrics-collector` is passed explicitly; it collects the four field groups core, ECC, retired-page
and row-remap. NVLink and PCIe counters should still use DCGM and cannot be inferred from
`nvidia-smi` as a fallback.

Control-plane endpoints:

```text
POST /v1/collector-events/gpu-metrics
GET /v1/gpu-metrics/{cluster_id}/{node_id}/latest
GET /v1/gpu-health-findings/{cluster_id}/{node_id}
GET /v1/collector-status/{cluster_id}?node_id={node_id}
GET /v1/evidence/{cluster_id}?node_id={node_id}&attempt_id={attempt_id}
```

Temperature, ECC DBE, row-remap, PCIe/NVLink deltas and power/thermal violations produce structured
findings whose disposition comes from the site metrics policy and cannot impersonate an NVIDIA XID Immediate Action.
A new non-zero `DCGM_FI_DEV_XID_ERRORS` state is converted to an `XidEvent` and enters the official policy; the same XID gauge value on the same GPU
does not fire again on every scrape. Because the field represents a persisted "last
XID", the first value after collector start-up only establishes a baseline; an event is generated only when it later changes from 0 or from another XID.
Raw `/dev/kmsg` remains the primary event source for capturing new events and recognising repeated occurrences of the same XID.

Temperature collection edges share `gpu_temperature_policy.temperature_decision` with the control plane, including device-derived
limits and the site fallback. Confirmation accumulates separately by severity/action semantics: a confirmed warning cannot swallow
a critical escalation or substitute for its confirmation sample; an unconfirmed change does not falsely report recovery of an old anomaly. When delivery is rejected the pending
edge is kept, and later identical samples are still retried. The rules for counter changes, normal-steady suppression, missing devices and periodic summaries are unchanged.

## Time Semantics and Cross-Source Correlation

Events keep the following time fields:

| Field | Meaning |
|---|---|
| `source_event_time` | RFC3339 wall clock from the Fabric Manager log; the raw text keeps the timezone offset |
| `source_monotonic_us` | Monotonic microseconds since boot from `/dev/kmsg` |
| `source_boot_id` | Linux boot ID the monotonic time belongs to |
| `collected_at` | UTC time at which the collector read the event |
| `ingested_at` | UTC time at which the control plane received the event |

Raw dmesg/kmsg monotonic time has no timezone and cannot be compared directly with wall clock. For the same kernel
source with the same boot ID, monotonic time is preferred and a 30-second window is kept; between Fabric Manager, kernel
and DCGM a 5-minute window is used to tolerate log delivery and collector start-up delay. Widening the window does not relax
the identity conditions: XID/SXID and node must match; when both sides have GPU UUIDs or PCI BDFs they must also intersect.

Finding queries return active state by default; use `?active_only=false` to query history. GPU latest,
counter baselines, active/history findings, batch idempotency results and collector status are all persisted through the
control plane's Store. Multiple replicas must use PostgreSQL/Aurora; each GPU/metric key is updated serially under a
transaction lock, so counter deltas are still computed correctly when requests land on different API replicas.

A GPU critical finding (for example DBE, row-remap failure or critical temperature) enters the
site safety policy and produces a quarantine workflow; a warning finding produces a diagnostics
workflow. The node step for a GPU/HBM temperature warning runs `dcgmi diag -r 1 -j` and stores structured
evidence; on pass it waits for a configurable cool-down window. If the diagnostic fails, the result is inconclusive, or the temperature finding is still
active after the window ends, the control plane escalates to `DRAIN`, keeps the node unschedulable and isolated, and does not restart training automatically.
From the DCGM JSON it extracts test, status, entity, error code and message, and generates
`recommended_actions` by fixed rules for thermal, PCIe, NVLink/NVSwitch, memory, driver/DCGM, GPU client and Field Diagnostic.
The result enters the step execution, the escalated incident and the fixed-template mail;
unrecognised tests use `DEEP_DIAGNOSTIC_REVIEW` and no repair action is guessed automatically.

The full batch also enters the persisted composite state machine. Within the same GPU event time window the state machine combines temperature,
thermal throttle/violation, power usage/limit, GPU utilization, DBE, row-remap,
PCIe replay, XID and NVLink signals, and identifies multi-GPU NVLink faults at node level. On a hit the original
component findings are kept, but only the composite state edge is submitted to the control plane, avoiding duplicate workflows.
Only the thermal bits of `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` can prove thermal throttling; SM/memory clocks
are context only, and a low value alone does not alert.

SBE, retired page and correctable row-remap counters are uniformly handled as positive deltas between adjacent samples:
SBE, retired SBE and correctable remap produce Warning; retired DBE produces Critical.
When at least two kinds of corrected-memory signals keep growing within the same GPU window,
`CORRECTABLE_MEMORY_DEGRADATION` is generated, escalating to `DRAIN` on the 3rd consecutive occurrence by default. The first collection of cumulative old values only establishes a
baseline and does not alert.
workflow. They use the `SITE_NODE_HEALTH` policy source and cannot override or impersonate an NVIDIA XID
Immediate Action.

GPU, host and log batches update each collector's `last_success_at`, `last_error_at`,
sample count and collection errors at the same time. The validation steps require a successful collection within the last two minutes; Agent
heartbeats cannot substitute for collector freshness.

`VALIDATE_HOST` checks recent CPU/load, memory and filesystem evidence and rejects nodes still above threshold.
`VALIDATE_FABRIC` additionally requires recent GPU/NVLink and host network evidence, and checks RDMA/EFA link/error
counters. The HyperPod manifests set `GPU_FAULT_VALIDATION_REQUIRE_RDMA=true` by default; non-RDMA
clusters keep false. This step is passive telemetry validation; active NCCL/EFA
canaries in a maintenance window should still be wired in later as deep diagnostics.

## Workload and Training Progress

The current production solution does not yet use `TrainingProgressCollector`; deployment defaults to
`GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR=false`. The control plane does not start the training health scan, nor does it
generate `training_hang` findings because a training Pod has no injected reporter. The collection, reporting and evaluation code below
is kept, and may be enabled explicitly only after reporter/sidecar integration and end-to-end validation are complete.

The CompletionWatcher publishes an allocation to
`POST /v1/workload-observations` on every reconciliation of a managed Pod. The control plane writes the
`cluster/attempt/workload/Pod/container/rank/node/GPU UUID` mapping into the shared Store.
When the GPU, host and explicitly enabled log collectors have no workload configured, the control plane automatically
fills in the workload state, workload IDs and runtime profile from the node and event time.

A training container or sidecar can run:

```bash
gpu-fault-collector training-progress \
  --attempt-id "${GPU_FAULT_ATTEMPT_ID}" \
  --rank "${RANK}" \
  --progress-file /var/run/gpu-fault/progress.json
```

The training application updates the progress file via atomic rename:

```json
{
  "step": 1200,
  "samples_per_second": 845.5,
  "loss": 1.73,
  "numerical_error": false,
  "checkpoint_ref": "s3://bucket/checkpoints/step-1200"
}
```

The reporter sends to `POST /v1/training-progress` every 15 seconds by default. The control plane records the last
heartbeat and the last step-advance time separately, and detects:

- heartbeat timeout or a step that does not advance for a long time;
- a rank's step lag relative to peers, or a throughput straggler;
- step regression;
- an explicit `numerical_error=true`, for NaN/Inf that JSON cannot express.

Related environment variables:

```text
GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR=false
GPU_FAULT_TRAINING_HEARTBEAT_TIMEOUT_SECONDS=120
GPU_FAULT_TRAINING_STARTUP_GRACE_SECONDS=300
GPU_FAULT_TRAINING_MAX_STEP_LAG=20
GPU_FAULT_TRAINING_MIN_THROUGHPUT_RATIO=0.5
GPU_FAULT_TRAINING_HEALTH_SCAN_SECONDS=15
```

## Evidence Retention

Raw GPU, host, node log and training progress batches are written to the shared Store. The default retention is 24 hours,
with at most 10000 records per node; expired records and the oldest records over the cap are deleted automatically on write:

```text
GPU_FAULT_EVIDENCE_RETENTION_HOURS=24
GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE=10000
```

This retention serves the short window around a fault and Support case evidence; it does not replace long-term archiving in CloudWatch Logs, S3 or a dedicated
time-series database. When enabled, the NodeLogCollector also saves the journal watermark and training log offsets atomically
to `/var/lib/gpu-fault/log-collector-state.json` and continues reading from there after a restart.

## FSx, Lustre and EFA

The host collector additionally collects:

- NFS/FSx/Lustre mount availability and capacity;
- Lustre read/write bytes, dirty-page hit/miss deltas;
- RDMA port/hardware error counters by category;
- EFA RNR, retry, CQ errors;
- NIC PFC pause and ECN counters (when the driver exposes them through `ethtool -S`).

Shared filesystem unavailability and EFA/RDMA error growth enter the node-health incident and
`VALIDATE_HOST`/`VALIDATE_FABRIC`.

The collector reads and caches the device slowdown, shutdown,
max-operating and memory max-operating temperature limits via `nvidia-smi -q -x`. When the read fails, fixed site fallbacks are used.
The PCIe and NVLink defaults come from NVIDIA DCGM Health; when overridden, the finding is marked as a site override:

```text
GPU_FAULT_GPU_TEMP_WARNING_C=85
GPU_FAULT_GPU_TEMP_CRITICAL_C=90
GPU_FAULT_MEMORY_TEMP_WARNING_C=90
GPU_FAULT_MEMORY_TEMP_CRITICAL_C=95
GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C=5
GPU_FAULT_GPU_TEMP_SHUTDOWN_MARGIN_C=3
GPU_FAULT_MEMORY_TEMP_WARNING_MARGIN_C=5
GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE=8
GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL=1
GPU_FAULT_POWER_VIOLATION_DELTA_WARNING_US=1
GPU_FAULT_THERMAL_VIOLATION_DELTA_WARNING_US=1
GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES=2
```

## HMA Forwarding Retirement

`kubernetes-hma`, `sqs-hma`, the Lambda handler and the four HMA provider APIs have been deleted; the old addresses return
404. Before upgrading, stop the flow under the old release and drain or archive the SQS/DLQ, collector outbox and processor
requests, then retire the two solution-owned Deployments. When preflight finds a historical Deployment or the state is unclear it
refuses to continue; the new API cannot be used to drain old requests of a retired entry.

Historical resource discovery and uninstall records are kept; AWS stacks, queues, logs and historical data are not cleaned up automatically by this source deletion.
For the operating sequence see the [Deployment and Operations Manual](docs/en/deployment-and-operations-manual.md) §7.4. The generic `SqsEventSink` is kept;
an HMA-specific consumer is no longer shipped.

The AWS-provided HMA, Node Agent quiesce container coordination and the hot-spare health label check remain unchanged; DCGM Prometheus path availability cannot be inferred from HMA Pod
Ready. Continuous GPU metrics are still provided by the independent DCGM collector.
