# Contributing

English edition of [`CONTRIBUTING.md`](CONTRIBUTING.md); the Chinese file remains the source of record until both are maintained together.

Before making a change, pick the documentation by the type of change:

| Change | Required reading |
|---|---|
| Adding an operation, channel, Store, route, plugin or metric | [GPU Fault Extension Guide](docs/en/extension-guide.md) |
| Modifying Manifests, renderers, systemd, AWS resources, the configuration model or the administrator CLI | [Developer Deployment Implementation](docs/en/developer-deployment-implementation.md) |
| Modifying Release CI, signing, `dist/` artifacts or the deploy-host offline bundle | [CI Release Process](docs/en/ci-release-process.md) and [Developer Deployment Implementation](docs/en/developer-deployment-implementation.md) |
| Finishing a code change and deploying to staging or production for verification | [EC2 Source Unified Deployment Process](docs/en/ec2-source-staging-reproduction.md) |
| Adding a code capability and a production resource at the same time | Read both, run both sets of gates |
| Only deploying an existing release | [Administrator Quick Deploy](docs/en/administrator-quick-deploy.md) |

The operation/channel registry, explicit authorization, the production resource lifecycle and administrator entry points
must never be extended only on the calling side or by adding literals in production. The extension guide decides the
capability registration points; the developer deployment implementation decides how things are generated, released,
verified, rolled back and uninstalled.

Documentation responsibilities and the order of authority are in [docs/README.md](docs/en/README.md). Historical material
must not be used as the basis for the current implementation.

A development checkout uniformly runs `make deploy-host-setup-online`: it first validates the paths and version configuration, then uses the system Python 3.12
to prepare, in parallel, the pinned Python/admin CLI and the isolated supply-chain/PromQL tool environments; tool package installation and the promtool download may also overlap,
while pip writes inside the same venv remain serial. If any step fails, it waits for the tasks already started to finish and then returns failure. Tools whose version, dependencies and pinned archive
digest verify successfully can be reused; online Python initialization may still reach the network. CI keeps independent tool targets, and
the signed offline `deploy-host-setup` adds no network or local test tool requirements.

`src/**/*.py`, `tests/**/*.py`, `scripts/**/*.sh` and the production deployment scripts are bound by the
[code and documentation impact contract](docs/code-doc-contracts.yaml). When modifying these files:

- also modify the related documents listed in the contract; or
- fill in `Documentation-Impact: none` in the Pull Request body and explain in
  `Documentation-Impact-Reason` why public behavior, commands and the acceptance contract are all unchanged.

An empty value, `N/A` or `TODO` cannot replace the reason. CI runs `scripts/check-doc-impact.py` against the Git diff
between the target branch and the current commit.

When you have confirmed locally that a change is purely internal, you can run explicitly:

```bash
GPU_FAULT_DOC_IMPACT=none \
GPU_FAULT_DOC_IMPACT_REASON='internal refactor only; public behavior and commands unchanged' \
make PYTHON=.venv/bin/python check
```

For day-to-day development, first run reverse impact selection against the actual diff relative to the target branch:

```bash
make test-impact BASE=origin/main
make regional-impact-plan BASE=origin/main
```

`test-impact` runs the related static checks and pytest; `regional-impact-plan` only prints the affected regional cases and
does not automatically execute live or destructive cases. Files that cannot be matched, shared contracts, Python/dependency/runtime
image, schema/transaction changes, or changes spanning more than three impact domains fail closed and escalate to the full gate.
The unified staging release computes the digested impact plan only once internally, and test execution and the regional plan both consume it;
the isolated PostgreSQL 16 is started only when the plan requires PostgreSQL.
A trusted local production candidate runs the equivalent local full gate when the personal commit or main candidate is unavailable; that gate first
runs static, then runs the ordinary pytest, the PostgreSQL stress and the source-only
artifact build in parallel within the deploy host's CPU budget. Inside static, Ruff, mypy, compile, architecture, contract, documentation, deployment configuration, YAML,
Shell and security checks are split into independent groups; only after all of them pass is the runtime image built and signed for release.
A clean `HEAD == origin/main` may first verify the signature and consume the same-commit main CI candidate; a GitHub Release likewise only promotes
candidates that the signed main CI gate has already proven equivalent for the static, coverage, artifact and PostgreSQL stress gates,
without repeating the tests. Dirty source and personal clean commits do not query GitHub. Full regional acceptance only follows the first
go-live, major architecture changes, or an explicit requirement in the impact plan. The rules and explanation are in
[Change Impact and Test Selection](docs/en/change-impact-and-test-selection.md).

The administrator's first deployment completes the base resources, the release build and the application deployment with one ARN command:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

When the first site build needs to override the capacity defaults, you may additionally pass a strict, permission-`0600`
`--config <AdminConfig.yaml>`; without it the original four-parameter default path still applies. Capacity changes on an existing site use
`gpu-fault-admin config --state-dir ... --reference ...`; see
[Administrator Capacity Configuration](docs/en/administrator-capacity-configuration.md) for details. Shell environment variables or `kubectl set env`
must not be used as a substitute.

After modifying code, developers also use the same four-parameter command; the dirty/clean level, impact tests, build, signature verification and deployment
are selected by the internal process:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

When the Runtime Profile policy changes, the first deploy generates
`<state-dir>/release-deploy/profile-plan.json` and stops. After review, run:

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault-staging \
  --approve-profile-plan <the plan_sha256 printed in the stop message> \
  --reference CHG-12345
```

Approval and release complete in the same command; the parameter is accepted only when `profile-plan.json` already exists and its digest matches. The approval reference must not be injected through hidden parameters or environment variables.
The full administrator process is in [Runtime Profile Change Approval](docs/en/administrator-profile-change-approval.md).

The complete GitHub Actions sequence from trigger, OIDC/ECR, quality gates and artifact signing to `gpu-fault-release` artifact
delivery is in [CI Release Process](docs/en/ci-release-process.md).
The shortest test, staging deployment and same-artifact production promotion sequence after a developer code change is in
[EC2 Source Unified Deployment Process](docs/en/ec2-source-staging-reproduction.md).

`release-build`, `release-deploy`, artifact paths, site generation paths and release-ref belong to the internal
release implementation and are not parameters for ordinary developers or administrators.

`gpu-fault-admin` is delivered by a separate deploy-host distribution and is not part of the Control Plane Runtime
wheel. When only administrator deployment code is modified, the deploy-host bundle and the deployment impact domain must be verified; this must not
cause unchanged application Runtime components to be rebuilt or rolled. The content-addressed deploy-host payload is decoupled from Git commit authorization;
when only the deploy-host changes on an existing site, only the deploy-host environment is updated and a read-only preflight is executed.

The final full test run of `make check` and `make test-parallel` choose 4 to 16 workers based on the CPU count,
which can be set explicitly with `PYTEST_XDIST_WORKERS=16`; they use `--dist=worksteal` and do not connect to an external
PostgreSQL. `make test-postgres` always runs serially with `-n 0` and does not inherit the worker budget of ordinary tests;
the concurrent contention inside the stress is still controlled by its own dedicated parameters and is not the same as running pytest cases that share a database concurrently.
When the local release gate holds a valid isolated authorization, `test-postgres-stress` delegates to independent-instance shards via
`POSTGRES_TEST_PARALLEL=1`: by default a quarter of the core count, clamped to 4 to 16 groups (the same derivation as `PYTEST_XDIST_WORKERS`),
and `POSTGRES_TEST_WORKERS` allows 1 to 16 groups, each with its own PostgreSQL 16, private authorization and `-n 0` process. All shards must have an identical
full discovery manifest, the actual selection must have no duplicates and no omissions, and the stress must have no skips, for the whole gate to pass.
External test connections and custom impact commands keep their original paths; to run the local parallel entry point directly, use:

```bash
make test-postgres-stress-parallel POSTGRES_TEST_WORKERS=8
```

This entry point does not treat an existing external database as a shard, nor does it allow an unverified inherited connection to replace the local isolated authorization.
CI's existing PostgreSQL shard and the coverage append path below remain serial and do not mix in receipts from the parallel entry point.
The native test fixture must clean up the previous case's data before schema initialization and before the configured-mode Store starts,
including the first use of the shared schema cache; leftovers from legacy cases must not depend on other cases completing backfill first.
Environment changes such as database extensions should be confined to the temporary database owned by the case and must not change the premise of later cases in the same shard.
The production Store's schema or dedicated startup validation must not be relaxed to fix test-order dependencies.
`make coverage` first collects non-PostgreSQL coverage in parallel, then serially appends the coverage of the isolated PostgreSQL 16
test database, and finally uniformly enforces
the 78% combined floor for the production scope, and applies the per-module floor to `coverage.module_floors` in `config/ci-unit-gate.json`;
the two complete scopes, production and runner, additionally each require statements and branches to reach 95%.
The runner's high coverage cannot raise the production scope's ratio. Documentation and CI contract tests run independently via `make docs-check` and `make ci-tooling-check`
and are not double-counted in coverage.

The per-module ratio strictly uses `(covered_lines + covered_branches) /
(num_statements + num_branches)`; `num_partial_branches` is only a diagnostic for partially covered branch lines and
cannot stand in for all uncovered branches. Missing, negative or total-inconsistent counts are rejected outright.
Each group is also compared against the full source manifest of the current working tree; omitting a file within the group or adding a nonexistent well-covered file cannot
pass the gate, and it is not enough to check only the files that happen to appear in the report.
The 95% gate measures statements and branches separately; use
`python -m tools.coverage_objectives --coverage-json <report> --scope production`
or `--scope runner` to view the results for the fixed scope; the standalone report command returns failure only with `--require-target`.
Normal CI merges and `make coverage` already enforce the 95% gate for both scopes, without depending on running the report command by hand,
and without lowering or replacing the original CI floor.

Scenario coverage is counted according to the [independent requirement matrix](docs/components/scenario-coverage.md)
and is not the same as code coverage. The matrix distinguishes design, implementation, local verification and unmeasured LIVE verification;
local simulation must not replace conclusions from real machines, and incomplete items must not be removed from the denominator.

A single repository-wide floor can be lifted by the well-covered majority of modules, letting an entire family of modules sit near zero coverage and still pass;
the deployment-only administrator modules have exactly this shape, because they are excluded by every runtime shard
and only the combined report knows their real coverage. Therefore `coverage.module_floors` declares, for each
deployment-only family, both a group floor (the whole family must not be lifted by the rest of the repository) and a file
floor (one well-covered module in the family must not vouch for its siblings). A newly added deployment-only source
file must fall within some group's glob, otherwise `tests/test_ci_unit_gate.py` fails; a group that
matches no file under test is likewise a failure, so that a module rename cannot silently void the floor.

main CI splits the fresh gate into four logical domains, `runtime`, `deployment`, `fault_runner` and `postgres`;
runtime is further split by a stable hash of pytest nodeids into `runtime_0..2`, so there are six parallel physical
shards in total. Each shard computes its content identity from its own source, tests, dependencies and Runner environment, and independently restores, verifies, re-signs and uploads;
the aggregating `unit` job finally runs `coverage combine`, simultaneously enforcing the production 78% combined floor,
the per-module floor and the 95% statement/branch gate for both scopes, and then generates the
fault report and the signed unit gate. Deploy-host-only administrator source enters only the deployment shard;
documentation or `.github/` changes are verified by the current static and may reuse the six historical shards. All pytest shards retain
structured `--durations` evidence. When the repository has a higher-spec Runner configured, `CI_TEST_RUNNER` and a matching
`CI_PYTEST_WORKERS` may be set; when unset, `ubuntu-latest` and 4 workers are kept.

A shard receipt must have complete discovery, the actual selection manifest, successful setup/call/teardown and a matching
content identity; the mandatory PostgreSQL stress must not be skipped. Historical reuse keeps the producer, source identity and session of the first execution,
and the schema-2 aggregate report declares only the current verification identity without rewriting old receipts as new executions.
An ordinary shard's integer worker budget must match the actual and recorded request count; `auto`/`logical` must record
both the original request mode and the resolved actual process count. Illegal, empty or reduced integer budgets cannot pass receipt validation, and
PostgreSQL still explicitly runs serially with `-n 0` without inheriting the ordinary tests' process count.
The `cryptography` used by the signing and TLS fixtures belongs explicitly to the `dev` and deploy-host test dependency locks and does not enter the CPU/GPU
runtime dependencies. New tests must not depend on packages that happen to be installed on a development machine but are missing from a clean locked environment.
Dynamic parameters such as timestamps must provide stable pytest IDs, so that the collection manifest of the same source does not change with the start time;
this is not a reason to fix or trim the data values under test. The optional PostgreSQL driver boundary also verifies collection of the full import graph,
rather than scanning only direct imports. When the driver is missing, the offline I/O guard still runs, and real SQL operations still require the real dependency;
successful collection does not mean PostgreSQL tests were run.
Tests clear by default the `KUBECONFIG` inherited from the deployment process; cases that need a Kubernetes configuration must explicitly set
their own local fixture. This isolation does not replace the cluster command guard, nor does it authorize tests to access the real cluster on the deploy host.
Every shard may execute the runner through cross-domain tests, so a runner source change conservatively invalidates all six shards;
a pure deploy-host change still invalidates only deployment and does not widen what the application wheel includes.

Coverage may go up; it must not hide untested new branches by lowering `COVERAGE_FLOOR`, lowering `coverage.module_floors`,
skipping shards or discarding the PostgreSQL stress. A `file_floor` of 0, a group with empty
globs, or a `file_floor` higher than `group_floor` are all rejected outright by configuration validation, because such a floor
reads like a guarantee but can never fail.
Test quality itself also has a ratchet. `make private-test-coupling-check` restricts tests from reaching across public boundaries
to private members; `make test-source-assertion-check` restricts tests from asserting on the code under test as text,
that is, `inspect.getsource(...)` or reading a `.py` file and grepping for strings. Such assertions pass as long as the implementation keeps
the same spelling, stay green when the behavior breaks, and go red on a pure rename; they should be changed to call-sequence spies or observable
results. A few files genuinely audit static text (the source roots declared by the CI gate, the gate list, the worker cap);
they are recorded together with the reason and the site count in `test-source-assertion-baseline.json`, which may only shrink, never grow;
a newly added file fails outright, and an entry dropping to 0 also fails, so that the baseline cannot be silently voided.

Local `make check` first runs the parallel static DAG, then runs the ordinary pytest and the artifact build in parallel;
`tests/test_artifact_consistency.py` runs only once on the artifact branch, avoiding a race with a `dist/` that has not been generated yet.
When Make detects `.venv/bin/python` in the checkout it uses that interpreter automatically; a source package without `.venv`
falls back to `python3`, and an explicit `PYTHON=...` always takes precedence.

When only documentation is modified, you must still run:

```bash
make docs-check
```
