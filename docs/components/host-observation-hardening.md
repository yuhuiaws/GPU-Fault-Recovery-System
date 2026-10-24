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
  column (units). A failed query or a response without sensor rows is a
  collection error, not a healthy zero.
- SMART query failures are `smart_query_failed` plus a collection error and
  are retried, rather than masquerading as `smart_health_failed` and causing
  hardware quarantine. Genuine failed self-assessments remain unchanged.
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
