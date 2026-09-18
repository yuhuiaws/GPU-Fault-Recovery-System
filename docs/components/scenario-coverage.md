# Code and Scenario Coverage

## Different Questions

Code coverage measures executed statements and branch outcomes. It does not
measure the fraction of possible GPU faults, deployment failures or physical
outcomes that the scheme handles.

Normal CI and local coverage require at least 95% statements **and** 95% branches within
each fixed scope:

| Scope | Source roots |
|---|---|
| Production | `src/gpu_fault`, `src/gpu_fault_release`, `deploy/control-plane/tools` |
| Runner | `scripts/e2e/regional`, `tools` |

`scripts/perf` is reported separately and cannot inflate the runner comparison.
The unit-test percentage describes coverage of production code, not coverage of
the test files themselves.

```bash
python -m tools.coverage_objectives \
  --coverage-json artifacts/coverage-local.json --scope production
python -m tools.coverage_objectives \
  --coverage-json artifacts/runner-coverage.json --scope runner --require-target
```

The report checks that every current Python source file in the selected roots
was measured, requires branch-enabled data, and lists unexecuted points and
existing exclusions. The standalone report enforces the target when requested;
normal combined gates always enforce both scopes, while retaining the production
78% combined floor and all per-module floors. Runner results cannot inflate the
production percentage. Both module floors and this report use actual covered
statement/branch counts, not `num_partial_branches`.
Module-floor checks also compare each configured group against the current
repository inventory. Missing source members and nonexistent reported members
are failures, not opportunities to improve the group's percentage.

Measurements that exercise both threads and multiprocessing children must enable
both coverage concurrency modes. Selecting `multiprocessing` alone omits work
performed in threads. Each attempt starts with an empty private data directory;
only that attempt's source-matching shards may be combined. SIGTERM collection
does not recover traces lost to SIGKILL, and a crashed pytest worker cannot supply
complete verification evidence.

## Independent Requirements

[`testcases/scenario-requirements.yaml`](../../testcases/scenario-requirements.yaml)
is the reviewed mechanism inventory. Each requirement names implementation
sources, triggering conditions, expected assertions, canonical cases and exact
local checks. Registered operations and channels must all appear, including
blocked or planning-only capabilities. Their requirement is the negative
invariant when mutation is unsupported, not a request to enable it.

The inventory is broader than the current regional case list: known missing
runner or safety implementations remain explicit `gap` entries in the denominator.
There are separate fault-policy, operation, ingestion, protocol, security,
lifecycle, availability and recovery groups. Requirements can group related
steps whose safety property depends on their order, such as quiesce, reset and
restore. This is a reviewed scenario inventory, not a mathematical enumeration
of all combinations of failures, hardware models or concurrent interleavings.
Adding, removing or broadening requirements requires design review.

The pinned XID catalog independently supplies the expected generic and B200
replay IDs. Missing replay cases are counted as missing, not removed. These rule
variants have their own denominator; hundreds of XID replays cannot conceal a
missing lifecycle or safety scenario. NVLink5 and product-family policy matrices
also require their explicit parameter counts.

## What Counts

| Stage | Necessary proof |
|---|---|
| Designed | The reviewed requirement maps to a non-retired canonical case |
| Implemented | Sources, checks and required runner exist, assertions are reviewed, and no implementation gap remains |
| Local verified | Every declared check passed all three pytest phases on the unchanged source with the exact expected parameter count |
| LIVE verified | Not measured by this tool; local test results are never promoted to deployed/physical evidence |

The implementation stage is a traceability and review assertion, not a proof
that an arbitrary function name implements the English requirement. Tests and
the per-case review must inspect that relationship. Missing selectors, duplicate
requirements, duplicate YAML keys, invalid registry members, foreign paths and
retired/`DO_NOT_RUN` cases are rejected.
The fault catalog's schema version must be an integer; YAML boolean `true`
cannot stand in for version `1`.

`tests/test_scenario_requirement_inventory.py` checks every declared parameter
count against isolated pytest discovery, including whether all matching variants
were selected. Changing a test's parameterization therefore requires updating
its reviewed binding. This collection-only check is not execution evidence.

The 95% goal must hold for the mechanism total, **every family** and the separate
fault-rule inventory. Critical requirements require 100% at the selected stage.
An aggregate majority cannot excuse a missing critical safeguard.

## Local Evidence

The existing `tools.pytest_case_reporter` emits additive session metadata:
source identity at start and finish, timestamps, exit status and collected
and discovered nodeids, plus collection errors and skips. It preserves the
schema-1 record format for existing consumers.

Cached CI shards additionally bind their selected tests and content identity.
Every required test must have successful setup, call and teardown phases;
incomplete discovery and mandatory stress skips are rejected. Reuse preserves
the original source/session and first execution producer. Schema-2 aggregate
reports retain that provenance and carry `validated_source_identity`, not an
invented fresh whole-source session. The scenario gate does not promote such
an aggregate into local or LIVE execution evidence.

```bash
PYTEST_GPU_FAULT_CASE_REPORT=artifacts/local-pytest.json \
python -m pytest -p tools.pytest_case_reporter -n 16 <ordinary-test-selection>

python -m tools.scenario_coverage \
  --pytest-results artifacts/local-pytest.json \
  --pytest-results artifacts/local-postgres-pytest.json \
  --require-stage local_verified
```

Use the repository's normal PostgreSQL selection **serially** against a newly
owned local test database. A SQLite or Memory simulation is not an Aurora
acceptance result. No command above authorizes LIVE tests or real GPU actions.

Reports must match the current complete source identity, have a successful
session, include a result for every collected test, and be at most seven days
old. Timestamps must be timezone-aware and ordered. A passing result needs
successful setup, call and teardown. Skips, xfails, missing variants and a FAIL
in any supplied report do not count as passing checks. Parameterized selectors
state an independent expected count; this cannot substitute for the complete
discovered inventory, even when a declared count is too small.

The scenario consumer uses the same decoded-receipt parser and selector matching
as the runners. It requires explicit clean collection metadata and merges
discovery across source-matching reports. Complementary shards can jointly
complete a check, but one filtered shard cannot erase its missing variants.
Canonical path matching also makes a failure under an absolute test path override
a pass under the equivalent relative path. Report files are decoded only once.
The reporter resolves the initial working directory to its Git top-level source
root and captures pytest's collection root separately. Discovery, collected tests
and results use paths relative to that captured source root; changing `--rootdir`
or starting inside an ignored subdirectory cannot make a same-name copy
stand in for a source test. Worker discovery is already canonical when merged.
The final source hash also uses the captured root, not a test-mutated working
directory. Failed or non-absolute Git root discovery is not replaced with CWD.
File-path mappings established during collection are reused by result callbacks
and cleared for each new pytest configuration. This avoids filesystem lookups
while tests temporarily replace OS APIs, and avoids repeating the same path
resolution for every parameter variant.

Run reports after source freeze. Concurrent edits invalidate them. Old pytest
records without session metadata remain usable only for their original signed
CI/fault-report protocol, not as new local scenario verification. New-session
source drift is also rejected by the ordinary fault-result consumer.

Fresh individual pytest cases, pytest command wrappers and batched pytest
execution require the reporter's structured receipt. Process exit zero or
human-readable pytest output alone cannot establish PASS. Every collected test
must have a result, and setup, call and teardown must all pass. Inherited pytest
selection and shard settings are removed when the fault runner launches a case.
Normal batch failures retain the independently passing cases; an unexplained
nonzero exit or incomplete collection invalidates the receipt.
The reporter also records pytest's discovered leaf inventory before parameter
selection and deselection, including xdist workers. Reusing one passed parameter
cannot complete a generic catalog selector; every discovered variant is required.
Module-level collection skips remain explicit and prevent a command wrapper from
claiming success. A separate passing native case in the same batch keeps its own
verdict.
Local pytest prerequisites inside live runners use the same receipt schema through
the shared supervised command boundary. Each prerequisite's entire discovered
focused selection must pass; a filtered subset, skip or xfail cannot satisfy it.
The warm-spare read-only audit batches independent prerequisites and attributes
complete results per case. A sibling's failed test does not erase an independent
passing prerequisite, but incomplete collection or an unexplained nonzero exit
invalidates the batch. Human-readable failure summaries are not evidence.
These processes receive an isolated unit-test environment, not live credentials
or an inherited pytest shard/report path.
CAP-005 explicitly opts its direct contract subprocess into its generated loopback
PostgreSQL database. That opt-in rejects the administrative database, non-loopback
targets and connection-target overrides; it is not an inherited-environment
exception. Its JUnit path and serial-worker arguments are explicit pytest options.

`python -m tools.scenario_coverage` without result files is a read-only design
and implementation audit. It cannot claim local execution. Reports keep LIVE
status `NOT_MEASURED` and list every remaining gap.

## Verification Boundaries

The inventory retains every unimplemented safety or evidence requirement.
BOOT-032 now supplies isolated native uninstall/resume coverage, and HA-011
supplies isolated deployed-runtime worker takeover coverage. NOTIFY-008 supplies
isolated notification crash-window coverage with a separate SIMULATED provider.
Their local regressions do not establish LIVE execution. Independent reboot
safeguards, expiring spare-action cancellation and prospective installation
custody now have connected runners and local regression bindings. Historical
snapshots alone still cannot establish installation custody.
The physical ownership runner covers controlled changes after Agent queueing
and before its final read and permit. Arbitrary changes after an individual
read or permit still lack a coordinating fence through the physical call;
that stronger requirement remains an explicit gap.

The model is fail-closed: existing blocked runner variants cannot become runnable
to increase a percentage. Physical evidence, release identity, maintenance
approval, cleanup and rollback requirements still apply independently. See the
[per-case review](regional-acceptance-review.md) and
[validation limits](../validation-limitations.md).

The notification crash contract separately tests loss before provider acceptance,
after acceptance but before Store commit, and after commit but before caller
acknowledgement. The middle window can cause two provider deliveries even though
the Store retains one notification identity. Deduplication of stored events is
not atomic exactly-once delivery across a provider and PostgreSQL.
`tests/notifications/test_cov95_commit_window_postgres.py` is the serial
PostgreSQL counterpart of the backend-neutral local contract; it does not execute
SNS/SES or replace an isolated LIVE runner. NOTIFY-008 adds actual worker loss,
fresh-process reconstruction, independent durable provider receipts and a
separately owned PostgreSQL database. Its regional wrapper validates the admitted
CPU Pod before arming either container and requires observed process termination
and UID-bound namespace cleanup. These are implementation and local regression
proofs, not an executed regional or external-provider acceptance result.

`tests/processor/test_cov95_busy_takeover_postgres.py` applies the busy-worker
lease contract to PostgreSQL. It keeps a request in flight, expires the original
ownership, requires one replacement and rejects the original late completion.
Its handlers and local HTTP boundary are controlled test doubles. HA-011 adds a
separate isolated production-image runner and real-process PostgreSQL regressions,
including rejection of an old callback while a replacement is still working.
Neither proof establishes business-Pod failover or CPU-saturation SLOs.

Coverage.py measures Python statements and branch arcs in the declared files,
not MC/DC coverage of every Boolean operand. SQL, shell, and Python embedded in
strings for remote execution are not independently counted as Python branch
denominators; they need their own semantic tests and scenario evidence. The
reported percentage must not be described as 95% of all possible distributed
failure interleavings.
