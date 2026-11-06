# Host Observation and Diagnostic Hardening

This implementation batch addresses observation integrity and targeted
verification from the general-fault review. It does not replace all GPU,
training, thermal, spare-health or RMA policies from that review.

## Implemented

- EFA inventory reads distinguish a successfully enumerated absent PCI device
  from unreadable vendor/device/driver/port information. An unreadable sample
  reports a collection error and breaks the mismatch streak. Without PCI
  evidence, missing inventory is `INVENTORY_UNAVAILABLE`, not proven PCI loss.
- Unknown EFA inventory failure modes select diagnostics, not reboot. Unknown
  and excess-device findings do not authorize a reboot on validation failure.
- Required netdev absence is reported. Enumeration/read errors remain unknown.
  Link state survives an unrelated statistics failure; counter baselines move
  only after the counter snapshot is valid.
- EFA rate aggregation requires a complete RX/TX pair, or a complete send/recv
  fallback pair when neither RX nor TX exists. Invalid, missing, reset or
  incomparable counters cannot contribute a valid zero-rate reading. Rates
  are added per port, not divided by the longest unrelated port interval.
- `efa_traffic_observation_valid` is an existing host-telemetry sample-envelope
  gauge: 0 means no trustworthy aggregate, 1 means the aggregate is usable.
  No new endpoint, operation, IAM permission or global Prometheus family is
  introduced. Unavailable samples break the control-plane zero streak while
  retaining the same attempt's prior active baseline. Old producers omitting
  the gauge retain legacy behavior.
- The collector emits initial/renewed activity and spike/drop edges for UNKNOWN
  as well as ACTIVE contexts. The control plane still requires one fresh,
  authoritative managed attempt; an explicit IDLE collector remains excluded.
- BMC parsing uses the fourth `ipmitool sensor` column (status), not the third
  column (units). A failed query, a response without sensor rows or a table
  whose every status is `na` is a collection error, not a healthy zero. A
  count ≥1 names the alerting sensors on the sample (`sensors`,
  `sensor_alerts`: reading, units, status, the threshold columns printed), and
  `bmc_critical_sensor_rule` writes them into the finding reason, so the
  incident reasons and the marker name them instead of a bare "2" (the
  quarantine chain sends no mail of its own). BMC the node cannot provide -- no `ipmitool`, or an
  `ipmitool` that finds no IPMI device node and none exists (EC2) -- is
  `bmc_status_unsupported` (TOOL_MISSING / NO_DEVICE) with no collection
  error, like SMART's, so it never holds host validation. Like every
  capability marker it is a state, and
  its appearance, a `reason` change and its clearing are edges while every
  accepted batch keeps it in the node's latest records (see the EFA bullet).
- SMART query failures are `smart_query_failed` plus a collection error and
  are retried, rather than masquerading as `smart_health_failed` and causing
  hardware quarantine. Genuine failed self-assessments remain unchanged. A
  `--scan-open` that fails (non-zero exit, timeout, or the
  `# scan_smart_devices:` line smartctl prints before exiting 0) is
  `SCAN_FAILED` rather than "no drives"; a drive the scan cannot open is
  `OPEN_FAILED` rather than silently dropped; a hung or non-boolean `-H` is
  `QUERY_FAILED` for that drive only. SMART the node cannot provide (no
  smartctl, no drive, no SMART status) is `smart_status_unsupported`:
  NOT_SUPPORTED, with no collection error.
- EFA `ethtool -S` deltas come only from real readings: the first reading is a
  baseline, a reset or an unparseable counter is unknown (collection error), a
  failed or hung `ethtool` is a collection error for that netdev only, and a
  counter the netdev lacks is `efa_ethtool_counters_unsupported` instead of 0.
  Every capability marker -- this one, `bmc_status_unsupported` and
  `smart_status_unsupported`, registered in
  `host_capability_markers.CAPABILITY_MARKERS` -- is a state, not an edge of
  its own; a marker that appears, changes or clears after the first tick is
  one (`not-supported:<marker>:<device|node>:<reason>[:<metrics>]`), delivered
  once, and every accepted batch, the health summary included, keeps the
  marker with its `reason` and device in `telemetry_metric_latest`
  (`NodeHealthPolicy.LATEST_METRICS_REQUIRED`), so the records say when a
  netdev stopped exposing a counter and what a node's capability state is at
  any moment. The health summary itself is not kept as raw evidence, so the
  edge and the latest record are what make a marker observable.
- The sysfs RDMA port counters (`rdma_errors_delta`, `rdma_<counter>_delta`)
  and the netdev `statistics` counters (`network_errors_delta`,
  `network_drops_delta`) follow the same reset rule: a counter below its
  baseline is unknown for that tick (a collection error naming the reader, the
  device/port and the counter, in the `ethtool` wording), the baseline is
  re-armed from the new value and no delta is shipped for it; counters that did
  not reset still report. A clamped 0 would clear the active signal and let
  the continuing increase open a second incident.
- An NVML failure while `/proc/driver/nvidia/gpus` is missing or unreadable is
  `INVENTORY_UNAVAILABLE`: no GPU count, no mismatch, no streak advance -- two
  failed reads are not zero GPUs.
- Inode exhaustion is `filesystem_inodes_used_percent`, graded like capacity
  (98 %, CRITICAL, QUARANTINE) and checked by VALIDATE_HOST.
- Host metrics needed by verification are persisted. New diagnostic plans carry
  `host_diagnostic_probe`: metric, exact device, comparison, threshold and
  trigger time. Missing/stale target evidence waits; unrelated healthy devices
  cannot satisfy the check.
- A failed host/performance trigger check alone cannot climb the hardware
  reset/reboot/DRAIN ladder. Real independent GPU failures remain subject to
  their existing recovery rules. Historical workflows are not rewritten.

## Validation

Focused regressions:

```bash
.venv/bin/python -m pytest \
  tests/collectors/test_host_observation_validity.py \
  tests/host_health/test_efa_observation_validity.py \
  tests/execution/test_host_diagnostic_probe.py \
  tests/host_health/test_host_diagnostic_e2e.py \
  tests/regional/test_efa_traffic_harness_main.py -q -o addopts=''
```

The swap-pressure E2E uses the real memory collector, event API, policy,
workflow builder and validation adapter with temporary proc data and local
state. The EFA harness runs its actual main path with an isolated loopback API,
real counter reader and synthetic EFA identity. Its dispatcher and hardware
adapters remain disabled: it proves detection and planning, not physical
recovery. No live fault injection or deployment is implied.

## Remaining Review Work

See the [32-category audit and acceptance specifications](general-fault-e2e-audit.md)
for per-category change status, existing proof boundaries and unexecuted E2E
cases. The additional host software E2E tests exercise actual validation and
Kubernetes isolation adapters, but do not establish physical recovery.

Thermal cooldown/readmission versus permanent RMA isolation, per-architecture
ECC/NVLink recovery, application-aware NaN/checkpoint/hang recovery, shared
storage probes, MIG/shared-resource accounting, and spare-health classification
are separate follow-up batches. This batch does not silently change their
thresholds or claim that the full 32-category review is implemented.

Matched control-plane/executor/node-runtime rollout is required before relying
on the new validity and diagnostic metadata. Existing workflows without the
probe retain their prior verification contract; historical PASS records are
not evidence of the new behavior.
