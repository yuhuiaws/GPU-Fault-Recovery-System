# Validation limitations

The following cases are not claimed as fully validated by the current
staging environment as of 2026-08-27:

- The Kubernetes shutdown budget is checked arithmetically, but the revised
  `GF-REGIONAL-HA-007` still needs a staging run that holds legal 30/70/110
  second requests while a control-worker terminates. The current worker uses a
  130-second lifespan budget and a 240-second Kubernetes termination grace;
  exactly-once completion or explicit lease release must be proven live.
- The fatal processor deadline callback has not been run in a subprocess that
  verifies exit code 70, lease release, and takeover by a replacement worker
  (`GF-REGIONAL-HA-008`).
- Notification outbox behavior is well covered as a service, but the
  service-role/lifespan wiring that starts the production worker dispatcher
  has not been exercised end to end. Restart takeover while the provider is
  throttling is also unverified (`GF-REGIONAL-NOTIFY-006`).
- `VALIDATE_GPU` grants a bounded grace window to WARNING-only findings that
  belong to a single self-clearing class before it treats them as a validation
  failure. Thermal stress, correctable-memory degradation, and power-limit
  throttling (`composite:POWER_LIMIT_THROTTLING` with its `power_violation_total_us`
  component) all recover on their own once temperature, ECC scrubbing, or the
  workload's power draw settles. While one of these classes is the only active
  finding and the workflow is still inside the grace window, the step returns
  WAITING rather than failing, so a benign DCGM WARNING does not escalate a
  `RUN_DIAGNOSTICS` workflow into a node-quarantining DRAIN. A finding that
  persists past the window, a mix that is not all one transient class, or any
  CRITICAL finding still fails validation and escalates as before.
- PostgreSQL-specific test count is intentionally not pinned in this document.
  Any test skipped because `GPU_FAULT_TEST_POSTGRES_URL` is absent means the
  default local suite is not a production Aurora Store gate
  (`GF-REGIONAL-CAP-005`).
- Remote commands have no command-level `max_waiting_rounds` or deadline.
  `GF-REGIONAL-CMD-013` successfully reclaimed the same command after each
  of 20 consecutive WAITING results, carrying the prior result details every
  time. A permanently WAITING adapter can therefore keep a workflow open until
  the separate workflow execution deadline intervenes; command-level
  dead-letter behavior is not implemented.
- `GF-REGIONAL-HA-006` measured the default executor-crash takeover delay:
  after the owning process was SIGKILLed, a completed Node Agent diagnostic
  remained behind the old 120-second command lease and reached terminal state
  125.981 seconds after the kill. This is expected by the current lease model
  but remains a recovery-latency limitation for time-critical actions.
- `HttpEventSink` now has a per-collector bounded disk outbox. Network errors,
  429 and exhausted 5xx retries are replayable; permanent 4xx records remain
  non-replayable. The residual limitation is bounded capacity, unwritable or
  damaged local storage, and needing one successful live post to wake replay.
  After that wake-up a single background worker continues bounded batches
  without waiting for another collection cycle; it stops on a zero-progress
  round and a later successful live post wakes it again. Completion Watcher
  critical failure/terminal events now use a ConfigMap-backed write-ahead
  outbox and replay independently after restart; its residual bound is the
  configured ConfigMap record/byte limit.

- `COLLECT-004` debounce was validated through the documented safe substitute:
  expected GPU count was temporarily set to 9, the first mismatch established
  a baseline, and the second sample 15.35 seconds later emitted the threshold.
  A physical GPU removal was intentionally not performed.
- XID154 `IGNORE` and XID74 reset-first were driven through real `/dev/kmsg`.
  The remaining destructive XID154 labels reuse already validated stop/reset/
  reboot execution chains and were not each repeated as separate physical
  actions. XID151 `RESTART_VM` was driven through real `/dev/kmsg` and mapped
  to a successful HyperPod node reboot. XID159 `CHECK_UVM` is applicable only
  to B100/GB200 and is `NOT_APPLICABLE` on this H200 fleet; a test must not
  falsify the product identity.
- Multi-cluster isolation is supported by the regional control plane, but the
  current environment has only one enabled, formally registered physical GPU
  cluster. A disabled temporary `b300-isolation-test` registration is not a
  second production cluster. The user explicitly deferred ISO-006/E2E-002.
- Provider node replacement is intentionally prohibited and is validated by
  configuration, IAM denial, and CloudTrail absence rather than by executing
  `BatchReplaceClusterNodes`.

No longer limitations after the 2026-08-11 rerun:

- BOOT-018 compares source, wheel, node bundle, runtime image component venvs,
  every running control/data plane process, release pins, and every live Agent
  by content digest.
- AUTH-015 now provisions per-node keys on the trusted deployment host. The
  GPU cluster contains no fleet master; installer Jobs and Cluster Executor
  mount only node-scoped keys, cross-node forgeries fail, and one node key was
  rotated without restarting its peer.
- HA-003 recovered from a real RESET=`LEASED` Aurora failover without duplicate
  action; restore, validation, and scheduling restoration succeeded.
- 8e/8f/8g/8h/8i were exercised through real events and cluster state.
- Fabric Manager journal used the production collector/control-plane sink,
  persisted its cursor, and did not replay after collector restart.
- Drill IDs now propagate from markers, XID/SXID raw messages, and node-health
  findings to incidents and notification subjects/bodies.
- DESTR-006 fail-closed preflight was exercised against an existing
  `NodeRecovery=Automatic` HyperPod without provider mutation.

These limitations must remain visible in acceptance reports. They are not
equivalent to PASS results.
