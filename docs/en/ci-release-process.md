# CI Release Process

English edition of `docs/CI发布流程.md`; the Chinese file remains the source of record until both are maintained together.

This document describes the complete process of the current GitHub Actions `CI` and `Release` workflows, from source commit, quality gates and
trusted candidate to the signed release artifacts. It is written for Release engineers, developers who maintain the release automation,
and administrators responsible for downloading artifacts and verifying their signatures.

This document only describes the CI build and the release artifacts. Actual AWS/Kubernetes deployment continues to use the
inspected entry points defined by [Administrator Quick Deploy](administrator-quick-deploy.md), [Administrator Operations](administrator-operations.md) and
[Developer Deployment Implementation](developer-deployment-implementation.md).
When a developer needs a directly executable main path for testing and going live after changing code, use
[EC2 Source Unified Deployment Process](ec2-source-staging-reproduction.md).
Dirty-source validation on a single EC2 instance uses the four-parameter
`gpu-fault-admin deploy` from [EC2 Source Staging Unified Deployment Process](ec2-source-staging-reproduction.md)
and is not part of the Release CI.

The implementation sources of truth for the current process are:

- `.github/workflows/ci.yml`: parallel quality gates, main candidate, CI gate and candidate signing;
- `.github/workflows/release.yml`: candidate resolution, signature verification, ECR promotion and final artifact upload;
- `release-build-promoted` and `deploy-host-sign` in the `Makefile`, plus `release-build` and `deploy-host-bundle`
  used by trusted local environments;
- `scripts/ci_coverage_gate.py`, `scripts/ci_gate_artifacts.py`,
  `scripts/ci_unit_gate.py`, `scripts/ci_gate.py`, `scripts/resolve_ci_run.py`:
  content-addressed coverage shards, aggregated unit gate, main CI candidate identity and Release run selection;
- `scripts/build-release-runtime-image.py`, `scripts/build-release-artifacts.py`,
  `scripts/build-release-attestation.py`: standalone runtime image, Manifest v4 and attestation;
- `scripts/build-deploy-host-bundle.py`, `scripts/deploy_host_bundle.py`: deploy host offline bundle.

Whenever any of the entry points above is changed, this document must be updated in step; it is not acceptable to change only the workflow or the command while keeping the old process description.

## 1. Process Boundary

The current main chain is split into two segments, main CI and Release promotion:

```text
main push
  -> static, artifact, 5 non-PG coverage shards and 1 PG shard run in parallel
  -> each coverage shard:
       -> compute the domain content identity
       -> hit: a signed gate with the same identity from a historical successful main CI: verify the signature and reuse the evidence
       -> miss: run this domain's pytest, branch coverage and duration collection
       -> generate the current run gate and sign it independently
  -> unit: verify the signatures of all twelve physical shards
       -> coverage combine + production 78% floor + per-module floor + two-scope 95% statement/branch
       -> merge pytest results, generate the fault report and the duration summary
       -> generate and sign the aggregated unit gate
  -> artifact: source-only canonical component artifacts + unsigned deploy-host bundle
  -> test: validate the unit domain gate, summarise this run's static/artifact, generate and sign the commit CI gate
  -> upload gpu-fault-ci-candidate

workflow_dispatch or v* tag
  -> checkout the target commit
  -> resolve the successful main CI run for that commit
  -> download the candidate and verify the CI gate signature, source identity and full file inventory
  -> obtain temporary AWS identity through GitHub OIDC and log in to ECR
  -> make release-build-promoted
       -> verify the CI gate again
       -> build or reuse the immutable Runtime Image from the candidate component artifacts
       -> generate the deployable Manifest v4 and the attestation bound to the CI gate
       -> run the final artifact consistency check and sign the attestation
  -> make deploy-host-sign
       -> only sign the deploy-host archive already built by CI; do not rebuild
  -> upload dist/ as the gpu-fault-release artifact
```

PRs also run the same twelve fresh shards, the unified coverage floor, PostgreSQL stress, `static` and
`artifact`, but the shards are not signed with main trust, no deploy-host bundle is built, and no signed candidate that Release
could download across runs is generated. The final job is still named `test` so that branch protection keeps a single aggregated status.

The Release workflow only accepts the clean commit obtained from checkout, and requires its candidate to come from a successful push CI on
`refs/heads/main`. It produces:

- `staging_only=false` in the Manifest;
- `release_tier=production` in the attestation;
- a promotion gate record bound to the signed CI gate.

`release-build-staging` is only for the internal source preparer of the unified CLI to handle dirty isolated snapshots. That tier performs impact
selection and escalates to the full gate when uncertain, but its Manifest is fixed at `staging_only=true`, which ordinary production signature verification
refuses by default; the GitHub Release workflow does not build or upload that tier.

That workflow:

- does query, build and push the ECR Runtime Image, and may read and write a separate BuildKit cache repository;
- does not run `make release-deploy`;
- does not create or modify EKS, HyperPod, Aurora, NLB, Route53 or Kubernetes resources;
- does not hold the CPU/GPU kubeconfig and does not touch the site token, the Node Action key or the production database password;
- does not automatically copy GitHub Actions artifacts to `/secure/release/`.

A source-ARN deploy can also automatically download the same-commit candidate when the checkout is clean and `HEAD == refs/remotes/origin/main`:
it first verifies the main run, the CI/unit/twelve shard signatures, the commit/tree/repository and the artifact inventory, and then enters
`release-build-promoted`. Dirty source and personal clean commits do not query GitHub. When the candidate is unavailable,
the local `release-build` first runs static/contracts and then, within the CPU budget, runs ordinary pytest, PostgreSQL
stress and the source-only artifact build in parallel; only after all of them pass does it build the runtime image, generate the deployable Manifest and sign it.
Any gate failure stops the other tasks in the same group, and the source-only artifacts already produced cannot be deployed. This is the trusted local reproduction path;
it is not equivalent to forging a main CI conclusion.
Inside the static branch, Ruff, mypy, compile, architecture, documentation, deployment configuration, YAML, Shell, security and
code contracts run in parallel again; the default `make check` runs artifact and ordinary pytest in parallel after static.
The quality gate subprocesses do not inherit the outer deployment's deadline, API budget or shim PATH, so the tests use their own
clock and fake tools. The build is still supervised by the outer process and bound by its time limit; this isolation does not remove the budget of the subsequent real release.
The local PostgreSQL fallback uses a supervised, exact-CID-bound private authorization protocol; only the PG test subprocess replaces
HOME/PGPASSFILE, while the other build and signing steps are unchanged. Credentials and authorizations live outside the repository in
`<state-dir>/release-postgres` and do not enter ordinary artifacts. When the result of creation or deletion cannot be confirmed, the build fails
and keeps the evidence, and existing signed artifacts cannot bypass the unfinished cleanup. This local fallback is not the lifecycle owner of the Actions service,
which is still managed by the CI job.
Under a valid local allocation authorization, the original `test-postgres-stress` gate delegates to independent-instance shards;
`POSTGRES_TEST_WORKERS` defaults to a quarter of the core count clamped to 4 through 16 (the same derivation as `PYTEST_XDIST_WORKERS`) and allows 1 through 16; each shard owns an independent PG16, a private authorization and a serial `-n 0`
pytest process. The gate is accepted only when the full discovery inventory, a non-overlapping and complete execution union, successful phases and zero skips all pass;
the 8 contending workers and 40 rounds of stress are unchanged. When any shard fails, only the other shards are asked to stop
pytest, each waiting for its own managed cleanup to finish; the cleanup supervisor processes are not terminated at the same time. When the dirty impact plan requires the full gate,
the same ordinary-test, PostgreSQL and artifact parallel group after static is reused; the CI service, coverage append
and custom external test connections still take the original serial path, and local reports cannot serve as CI shard receipts.
The report paths and partition selection of the local shards are bound inside the current pytest session; they do not rewrite the reporter's public environment variable constants,
nor do they pass the outer report or partition settings to nested tests. The nested runner still independently prepares and validates its own complete receipt,
and cannot overwrite the outer report or inherit the outer partition and run fewer tests.

After the optional HMA CloudWatch forwarding was decommissioned, the Lambda/CloudFormation templates are no longer delivered, and the dedicated
cfn-lint tool installation and the CI/static target were removed along with them. YAML, lazy exports, pip-audit, SBOM,
promtool and the artifact consistency check still run; removing the collector must not leave unresolved Python exports or
redeployable old manifests behind.

Besides the syntax check, `promtool-check` also runs the alert behaviour tests with a pinned real promtool version and
validates the complete successful pytest receipt; a missing tool fails both locally and in CI. The two related test groups are owned by the always
re-executed static gate and are not treated as historical evidence of a reusable coverage shard.

The file enumeration of a dirty candidate includes untracked files and excludes working-tree deletions explicitly reported by Git; a deletion also changes
the content identity and cannot reuse the original shard. Symbolic links, directories, an unreadable Git deletion list, and inputs that disappear after enumeration
without being in the deletion list are still refused; staging or committing files is not required in order to run the local gates.

## 2. Triggers, Permissions and Inputs

### 2.1 Trigger Conditions

`.github/workflows/ci.yml` only responds to Pull Requests and `main` pushes. Ordinary feature branch pushes
do not run CI again; new commits on the same PR or branch cancel the old run through concurrency.

`.github/workflows/release.yml` supports two trigger methods:

| Method | Purpose |
|---|---|
| `workflow_dispatch` | Promote the selected commit after approval; a successful CI run ID may optionally be specified |
| Pushing a `v*` tag | Promote the successful main CI candidate for the commit the tag points to |

Both workflows use `fetch-depth: 0`. By default Release queries the most recent successful main push CI by `git rev-parse HEAD`;
an explicit `ci_run_id` only skips the query, and the subsequent source commit, Git tree, repository and
artifact inventory must still match the current checkout exactly.

### 2.2 GitHub Permissions

Both workflows declare only:

| Permission | Purpose |
|---|---|
| `actions: read` | Release downloads the trusted CI candidate across runs |
| `contents: read` | checkout of the source |
| `id-token: write` | keyless signing of the CI gate, release attestation and deploy-host archive, plus AWS OIDC for Release |

The GitHub workflows do not configure `COSIGN_SIGNING_KEY`, so keyless `cosign sign-blob` is used.
Before requesting an AWS identity, Release pins the CI gate certificate identity to
`ci.yml@refs/heads/main` and verifies the OIDC issuer. The deploy side must likewise verify the approved Release identity
and issuer, and must not only check that the file exists or its SHA-256.

### 2.3 Repository variables

The Release workflow consumes the following GitHub repository or environment variables:

| Name | Required | Role |
|---|---|---|
| `RELEASE_ROLE_ARN` | Yes | Release role that GitHub OIDC assumes |
| `AWS_REGION` | Yes | Region for ECR login and image operations |
| `RUNTIME_IMAGE_REPOSITORY` | Yes | ECR repository URI of the immutable Runtime Image |
| `RUNTIME_IMAGE_CACHE_REPOSITORY` | No | Separate BuildKit registry cache repository URI |

Both repository values must match an ECR URI. The cache repository is only used to speed up layer builds;
it cannot serve as a trusted release artifact, nor replace the Runtime Image digest check.

CI tests also support the following optional repository variables:

| Name | Default | Role |
|---|---|---|
| `CI_TEST_RUNNER` | `ubuntu-latest` | Configured larger/self-hosted Runner label used by the coverage and PostgreSQL shards |
| `CI_PYTEST_WORKERS` | `4` | xdist worker count for non-PostgreSQL shards; should match the Runner CPU/IO capacity |

When no larger runner is configured, a non-existent label must not be entered. A change in the worker count enters the shard protocol identity; CI always
uses `--dist=worksteal` to reduce the long tail, but does not substitute more workers for slow-test analysis.
An integer setting must equal the actual worker count in the receipt exactly; empty override values, negative numbers, booleans or invalid text are
refused outright. `auto`/`logical` keep the original request mode and record the positive integer process count actually resolved by xdist; the aggregating machine
does not recompute the CPU count. A request shrunk by the worker cap cannot pose as the original integer budget having been executed; PostgreSQL is always `-n 0`.

Only the CI `postgres` shard creates a temporary PostgreSQL 16 service and sets a
`GPU_FAULT_TEST_POSTGRES_URL` that points only at that service. The Release job no longer creates PostgreSQL and does not re-run tests already proven by the
signed CI gate. That test connection is not a production database connection and must not be replaced with production Aurora.
`ci_postgres_grant.py` binds the process test authorization to the full service container ID, the current
repository/run/attempt/job ownership and a loopback-only port mapping. The test URL contains no password;
`HOME/.pgpass` and the authorization record live in a private `RUNNER_TEMP` directory outside the checkout and the artifacts,
with directory permission 0700 and file permission 0600. The `always()` cleanup after evidence upload only revokes that authorization file and the environment reference;
the service lifecycle is still managed by Actions, and containers are not taken over by name, relabelled or deleted.

When the deploy host automatically downloads the main CI candidate it may use a
`GPU_FAULT_GITHUB_TOKEN_FILE` with permission `0600`, or inherit `GITHUB_TOKEN` or `GH_TOKEN` from the controlled process.
The token is used only for the read-only GitHub Actions API and must not enter command output, state, release artifacts or the site.
Candidate lookup does not `git fetch` automatically; when the local `origin/main` is not the current HEAD it is skipped outright.

## 3. CI Execution Steps

### 3.1 Parallel CI Gates

All jobs use Python 3.12 with a pip cache, install the hash-pinned `requirements/build.lock` and then
install the project test extras. The CI execution units are as follows:

| job | Main responsibility |
|---|---|
| `static` | Ruff, mypy strict, compileall, architecture, deployment contracts, documentation and CI tool tests, configuration, YAML, Shell and artifact security checks |
| `coverage-runtime_0..2` | Ordinary Runtime pytest split into three parts by a stable nodeid hash, each collecting branch coverage and duration evidence |
| `coverage-deployment_0..3` | Release, regional orchestration and deploy-host administrator pytest split into four parts by a stable nodeid hash, each collecting the corresponding coverage |
| `coverage-fault_runner` | fault scheduler/runner tests and duration evidence |
| `coverage-postgres_0..3` | PostgreSQL contract pytest split into four parts by a stable nodeid hash; each part collects coverage serially (`-n 0`) on its own job-bound PostgreSQL service and then runs the 8 workers × 40 rounds stress over the same part |
| `shuffled-order-0..3` | The same test set as `test-parallel-release`, shuffled as a whole with the run-id seed and cut into four disjoint subsequences (`tests/conftest.py`), each run with xdist to expose order coupling |
| `unit` | Verify the signatures of the twelve physical shards, merge coverage, unified floor, fault report, duration summary and aggregated signature |
| `artifact` | Build and validate the three source-only component wheels and the Node bundle; main additionally builds the unsigned deploy-host bundle |
| `test` | Aggregate static/unit/shuffle/artifact and generate the commit-level CI gate for main |

`config/ci-unit-gate.json` defines the test partitions, the content identity groups, the coverage protocol and the
documentation/CI tests that do not enter coverage twice. The four logical domains run within the following boundaries; `protocol.partitions`
(matching `PARTITIONED_DOMAINS` in `scripts/ci_coverage_config.py`) splits runtime, deployment and postgres into three, four and four
physical shards while fault_runner stays a single shard, twelve in total:

| shard | Test boundary | Content identity highlights |
|---|---|---|
| `runtime_0..2` | Ordinary tests other than the static, deployment, fault runner and PostgreSQL entry points; each concrete nodeid enters exactly one part by SHA-256 modulo | dependencies, Runtime source, Runtime tests, shared fixtures, partition total and index |
| `deployment_0..3` | `tests/admin/`, executable regional tests and the release/deploy/artifact root tests; each concrete nodeid enters exactly one part by SHA-256 modulo | dependencies, shared Runtime source, deployment/deploy-host-only source, deployment tests, partition total and index |
| `fault_runner` | `tests/test_case_scheduler.py`; the full catalog contract is still run by static | dependencies, shared Runtime source, testcases, runner/scheduler and runner tests |
| `postgres_0..3` | The complete explicit inventory in `tests.postgres_files`; each concrete nodeid enters exactly one part by SHA-256 modulo, and the contract and stress passes use the same part | dependencies, shared Runtime source, PostgreSQL tests, actual PostgreSQL image, partition total and index |

The entry point of the PostgreSQL shard is governed by `tests.postgres_files` in `config/ci-unit-gate.json`
and is consistent with the serial PostgreSQL inventory check in the Makefile; a newly added native regression must not be placed only in an ordinary shard.

It mainly covers `src/gpu_fault/store/postgres/**`, `store/contracts.py`,
`store/shared/**`, schema/DDL, the connection pool and reconnection, as well as processor claim, lease, counter and
fencing. Because these tests consume the shared model, the Store contract and Runtime helper code, its identity conservatively
binds the Runtime source rather than only digesting `store/postgres/`; the deploy-host-only administrator source is not part of that
identity.

Each shard identity also includes the Python version/ABI, the Runner image, the actual worker protocol and the installed distributions;
the PostgreSQL shard additionally includes the container image ID. A main push first looks up historical artifacts by
`gpu-fault-coverage-<shard>-<identity>` and only accepts artifacts from completed main pushes in which the
corresponding shard job succeeded; failures of other jobs do not invalidate a shard that was already independently signed as successful.
On a hit:

1. verify the historical shard gate's Cosign workflow identity, producer run and all evidence SHAs;
2. require that the producer commit is still an ancestor of the current commit;
3. reuse the coverage, pytest and duration evidence;
4. generate the current run shard gate with `reused_from` and sign it again.

A shard that misses runs only its own pytest. The five non-PostgreSQL shards use
`--dist=worksteal` on independent Runners; every pytest command also prints `--durations=50` and writes a structured
`durations.json`. The wall clock of a fresh run is therefore determined by the slowest shard, instead of chaining roughly 3800 ordinary tests,
PostgreSQL contract and stress in the same job.

A Each shard's pytest command line comes from `shard_arguments` in `scripts/ci_coverage_config.py`: a directory
whose test files belong to this shard (other shards' files below it at most one fifth) is passed as a directory
argument, the few exceptions are excluded with `--ignore=`, and the remaining files are listed individually. pytest
re-collects a file argument's whole parent directory for every file argument (`Session.collect` bypasses its
collection cache for bare file paths), so naming the 843 files under `tests/regional` one by one builds about 800k
Module nodes and makes collection alone three times slower than the directory form (190 s against 63 s, measured
2026-10-03). The receipt's `selection.targets` records the actual arguments while `collected_files` must still equal
the shard's file inventory exactly, so a wrong argument set fails the shard gate instead of running the wrong tests;
`tests/test_ci_unit_gate.py` proves for every shard that the argument set collects exactly the target files.

pytest nodeid is "test file path + test class/function + parametrized case ID", for example:

```text
tests/collectors/test_xid_kmsg_catalog_replay.py::test_xid_kmsg_catalog_replay_b200[GF-XID-KMSG-B200-143]
```

It is not a GPU or Kubernetes node identity. Splitting by nodeid rather than by file spreads the hundreds of
parametrized cases in the same file across the three runtime Runners; the partition algorithm is deterministic, mutually exclusive and complete in its union. A change to a test function or parameter ID
changes the runtime identity, invalidating and re-running all three runtime shards at once.

The runtime, fault runner and PostgreSQL coverage explicitly exclude the deploy-host-only modules;
the deployment shard collects coverage of the full application source. This way, when only the standalone administrator CLI code changes, the old Runtime
coverage does not carry stale line numbers of the changed modules, and only the deployment shard is invalidated. The final `unit` job again
checks that the twelve shards belong to the current run, verifies their signatures and runs `coverage combine`, and only continues when the production scope reaches the 78% combined
floor, every module gate passes, and production and runner each reach 95% for both statements and branches.
All shards measure the cross-domain calls of the runner, so a runner source change conservatively invalidates every shard; a pure
deploy-host change still does not invalidate the Runtime shards.

The per-module floor is enforced only on the merged report: deployment-only modules are excluded by every runtime shard, and
a single shard's data cannot tell how much of them is really covered. `coverage.module_floors` in `config/ci-unit-gate.json`
declares a `group_floor` and a `file_floor` for each family; the former prevents a whole family from being lifted by
the rest of the repository, the latter prevents one well-covered module in a family from vouching for its siblings. An empty module with no measurable points
does not compute a per-file ratio, while a whole group with no measurable points still fails. Locally,
`make coverage` performs the same check with `ci_coverage_gate.py module-floors`.

Counts are computed as `covered_lines + covered_branches`, not as the total minus
`num_partial_branches`. The latter only describes partially covered branch lines and misses whole unexecuted branches;
missing, non-integer, negative and mutually contradictory statistics are all refused. The 95% gate checks statements and branches of both scopes independently;
see [Coverage and Scenario Matrix](../components/scenario-coverage.md); it does not narrow the existing source scope.

The twelve pytest results are merged after validating discovery, the selection inventory, the complete execution phases and the shard content identity;
a numeric worker protocol must also match the actual process count and the recorded request count; an auto-mode receipt must explicitly bind the request
mode. A missing auto-mode proof or a wrongly typed count is never reused, and an old numeric receipt cannot bypass the actual-count check either.
The PostgreSQL stress must also provide a complete receipt with no skips. The source identity, session and
first-execution producer of the original receipt are all preserved. The schema-2 aggregated report records the current
validation subject in `validated_source_identity` and does not rewrite a historical execution as a new session of the current source. Shard gates use schema 2, the aggregated
unit gate uses schema 3, and old formats must not bypass the newly added gates.
`make fault-test-cases-ci` maps these results directly onto the 64 unit/component fault cases without
starting pytest again. The documentation/CI contract tests are run by the `static` job of the current commit, so docs or `.github/`
changes can reuse every shard but cannot skip the current static and artifact gates.

The local pytest reporter additionally records the start/end source identity, time, exit status and actual collection.
A PASS for a fault case must have three successful phases, setup/call/teardown; a report carrying a session may not be reused if the execution
failed or the source changed midway. Scenario validation requires a complete session, seven-day freshness and
an exact parametrization count from local reports; an old signed shard merged report does not automatically become local scenario proof, and its original purpose is still
guaranteed by the CI signing identity protocol. The requirement files, the scenario tools and their tests enter the fault-runner content identity; the
integrity of requirement references across code/documentation is checked by the catalog contract tests of the current static.

#### 3.1.1 Fresh Measured Baseline

The following figures come from the default `ubuntu-latest` with 4 xdist workers and no larger runner configured; they are for regression comparison,
not a fixed SLA:

| main commit / run | Process | End to end | Longest test path | Aggregation |
|---|---|---:|---:|---:|
| `863a557` / `33615185401` | Old monolithic unit | 18 min 44 s | unit 18 min 19 s; of which ordinary coverage 14 min 44 s | Included in the same unit |
| `a6dd742` / `33620340339` | Six signed shards | 6 min 18 s | `runtime_2` 4 min 53 s | unit 57 s |
| `f90fdad` / `33622247519` | Fresh after the stability fixes | 6 min 10 s | `runtime_1` 4 min 41 s | unit 59 s |
| `adcaadff` / `37124211673` | Six shards + a single shuffled job (after the suite grew) | 39 min 14 s | `shuffled-order` 39 min; `coverage-deployment` 36 min; `coverage-postgres` 30 min | unit not run |
| `453b0dcb` / `37139294375` | Twelve shards + four shuffled shards (PR run, all fresh) | 16 min 42 s | `coverage-postgres_1` 14 min 30 s; `coverage-deployment_2` 11 min 48 s; `shuffled-order-2` 11 min 36 s; `coverage-runtime_2` 10 min 54 s; static 8 min 18 s | unit 2 min (combine 93 s) |

Before the 2026-10-03 re-sharding the suite had grown to about 43,000 nodeids: the single shuffled job, the single
deployment shard and the serial PostgreSQL shard each needed 30-40 minutes, and even in parallel they held the whole
pipeline at 40-45 minutes. Re-sharding only changes which slice each job receives, not the test set, the coverage floors,
the signing protocol or the shuffled order itself (see the partition proofs in `tests/test_ci_unit_gate.py` and
`tests/test_test_suite_contracts.py`). The fixed overhead per job (checkout, cached pip install, cosign) is about one
minute and is not the bottleneck; the first sharded run still showed deployment shards at 10-19 minutes, 5-10 of which
were the per-file collection cost described above, which the directory arguments cut to under two minutes. The same
shard can differ by 2x in wall clock between Runners (the single deployment shard once took 20 and once 36 minutes),
so compare several runs.

The long tail of a single test cannot be spread by sharding: the whole-suite collection probe in
`tests/test_optional_postgres_collection.py` takes about 4.5 minutes and several cases in
`tests/regional/test_clean_redeploy_script.py` take 40-55 seconds, so the shard that holds them cannot be shorter than
they are. Further reduction needs a larger Runner (`CI_TEST_RUNNER` together with `CI_PYTEST_WORKERS`) or shorter tests.

Later comparisons should also check the artifact's `test-durations.json`, so that Runner queueing, dependency
installation or the long tail of a single test is not missed by looking only at the total duration.

`artifact-check` generates only one set of canonical Control Plane, Executor and Node Runtime wheels and the
Node bundle. The source-only Manifest has `deployable=false` but includes the physical SHAs,
`module_digest`, the wheels embedded in the bundle and the component build identity. The component identity also covers
Python/platform, the build lock, the component source and all Node bundle inputs; an old bundle cannot be paired with new source.

The deploy-host bundle is built once in the main CI artifact job. `actions/cache` stores the wheelhouse keyed by the two locks,
the Runner OS/architecture and Python 3.12; the archive is unsigned at this point and is left for the Release
workflow to sign after the candidate's signature verification succeeds.

### 3.2 Generating the Trusted CI Candidate

The aggregation job `test` depends on three jobs and requires `success` from each of them. On a main push it:

1. downloads the complete `dist/` generated by the artifact job;
2. downloads the aggregated signed gate generated by the unit job in this run, verifies the signature again, and checks the twelve shard gates,
   signature bundles, content identities and evidence inside it;
3. requires a clean working tree and reads the source-only schema v3 Manifest;
4. records the Git commit, Git tree, repository, main CI workflow ref and run ID;
5. records the relative path, permissions, size and SHA-256 of every file in `dist/` except the CI gate itself;
6. writes the unit gate's identity, producer run/commit, SHA and reused shard inventory into
   `domains.unit`;
7. keyless-signs `dist/ci-gate.json`;
8. uploads `gpu-fault-ci-candidate`, retained for 30 days.

The CI gate can only be generated by `ci.yml@refs/heads/main`. The PR aggregation job only provides the branch protection result; it does not sign or upload
that candidate.

### 3.3 Release Verifies Signatures Before Touching the Cloud

After Release checks out the target commit, it first resolves the matching successful main CI run and downloads
`gpu-fault-ci-candidate`. It then, in order:

1. verifies the CI gate signature with the exact CI workflow certificate identity and the GitHub Actions issuer;
2. verifies that the gate's commit, tree and repository match the current checkout;
3. recomputes the candidate Manifest SHA and the complete file inventory;
4. verifies again the independent Cosign signatures of the aggregated unit gate and the twelve coverage shards;
5. only after everything matches does it obtain the AWS OIDC identity, log in to ECR and install the release dependencies.

Therefore a manually supplied other run ID, a tag pointing at a commit that did not pass main CI, a candidate with missing files, or artifacts mixed across runs all
fail before any AWS/ECR action.

### 3.4 Run `make release-build-promoted`

`release-build-promoted` requires `RUNTIME_IMAGE_REPOSITORY`, `CI_GATE`, a clean working tree and
Cosign. It does not repeat `make check` or the PostgreSQL stress run; instead it runs `ci_gate.py verify` again
and then continues with the following steps.

#### 3.4.1 Build or reuse the Runtime Image

`scripts/build-release-runtime-image.py` first verifies the source-only
Manifest left behind by `artifact-check`, the delivery identity, the wheel SHAs, the `module_digest` values and the wheel embedded in the Node bundle,
then computes the canonical image input digest from the following inputs:

- the Dockerfile and the pinned base image digest;
- the runtime dependency lock;
- the target platform and build args;
- each runtime image's own component wheel SHA-256 and `module_digest`;
- the other Runtime Image inputs defined by the builder.

When the target immutable tag does not exist, the Release builds and pushes the image; when the tag already exists, reuse is allowed only if the target platform and all
`gpu-fault.*` labels match this run's inputs exactly. Any label mismatch fails; an existing tag must never be
overwritten as if it were an ordinary mutable tag.
A missing registry entry must come from an explicit target reference or a registry error code; credential helper, authentication, TLS,
network, process and output errors fail immediately, do not trigger a cold download or a rebuild, and do not echo the raw credential helper output.
Signature schema v3 shared images and v4 independent images use the same ECR inventory verification, bound to the Region, registry ID,
repository and the complete digest set; a successful exit that returns empty data does not mean the image exists.

Image content and Collector plugin verification run in an isolated container with no network and a read-only filesystem. Before starting the command the verifier
binds the full CID, the private CID file, the ownership label and the image identity; normal completion, timeout or main-thread interruption all
clean up that same container and its anonymous volumes and confirm the container no longer exists. When the creation or cleanup result cannot be confirmed the gate stops, keeping the
private ownership record for reconciliation; cleanup must never be performed merely on the basis of a container with the same name or the same label.

On success it generates:

```text
dist/release-runtime-image.json
```

This image-set descriptor uses schema v3 and records, separately for the Control Plane, the Executor and the node offline dependencies, the
OCI digest, platform, build inputs and component/file identity. At most 3 builds run, sharing the original immutable repository;
changing only the Control Plane wheel does not rebuild the Executor image. Each CPU/Executor image has exactly one application package environment:
`/opt/gpu-fault/runtime` installs all hash-locked runtime dependencies and this component's wheel inside an isolated venv,
no longer installs dependencies into the base Python, and does not use `--system-site-packages` or `.pth` bridging.
The image's default `PATH` and `VIRTUAL_ENV` point at that environment; the original `control-plane` or `executor` directory is only a compatibility symlink to the same
environment, not a second set of packages. Component parameters and the wheel sit after the shared dependency installation layer, so the two images
can still share BuildKit dependency layers. The base image's interpreter, standard library and packaging tools do not count as a second application environment.
The content gate actually exercises the default `python`, the compatibility paths, `-I` imports, console entry points and `pip check`, and rejects external
package paths, the wrong component or a CLI interpreter. Only known old Dockerfile digests keep the old-layout verification, which still verifies component digests,
plugins and the isolation boundary; the descriptor format, signature and rollback identity of old releases are unchanged.
The node wheelhouse downloads hash-pinned Linux amd64/Python 3.12 wheels according to `node-runtime.lock` and
`node-tools.lock`, including py-spy, and generates an
inventory. The wheelhouse is not placed in the Node bundle or a ConfigMap; the inventory SHA is bound by the final signed Manifest.
The BuildKit cache still uses a separate mutable repository, but each component exports to a tag with the
`-control-plane`, `-executor` or `-node-dependencies` suffix; the local cache uses subdirectories of the same names.
A plain import keeps the old shared cache as a compatibility read source; imports with a pinned digest or a selector are not rewritten.
The export of a split build accepts only registry/local targets that can be partitioned unambiguously; ambiguous CSV options, overlapping targets and unsupported
export types are refused before any build starts. Cache locations do not enter the trusted release identity and cannot replace content signature verification.
The same trusted deploy host may reuse a verified dependency image before downloading wheels: a local private HMAC receipt binds the full inputs,
and the registry labels and platform are then re-checked against the pinned OCI digest. The cache stores only receipts; it does not cache or authorize arbitrary
wheel directories; a miss triggers a cold build, and an authentication failure or identity drift is refused. The receipt and its local authentication key are not delivered with the source,
CI artifacts or the release, and cannot serve as a signed release proof across deploy hosts.
Old schema v2 shared-image descriptors can still be read, and source-only candidates remain Manifest v3.
Before emitting the descriptor, `verify_release_images` checks the real module digests,
component isolation, entry points and every wheelhouse file in a network-disabled, read-only container; on failure nothing is signed and no deployable descriptor is emitted.

#### 3.4.2 Build the final release artifacts

`scripts/build-release-artifacts.py` uses the Runtime Image descriptor and the verified
canonical artifacts to generate the final release:

1. reuse the Control Plane wheel;
2. reuse the Cluster Executor wheel;
3. reuse the Node Runtime wheel;
4. reuse the Node installer bundle that contains only that exact Node Runtime wheel.

The builder recomputes the source identity, the four physical SHAs, the three `module_digest` values and the wheel embedded in the bundle,
and compares each against the Runtime Image descriptor. Any mismatch prevents the final release from being generated. Component builds use
the build frontend/backend pinned in `requirements/build.lock` together with `--no-isolation`, and do not
create a separate isolated build environment for each component.

When the canonical component artifacts are generated for the first time, the three components' independent build subprocesses run in parallel; the Node bundle waits only for
the Node Runtime wheel, not for the other two components. Build outputs are matched by distribution, and `umask 022`
applies only to each build subprocess without changing the parent process's global umask. The Manifest is published atomically only after all build tasks finish and pass their checks;
a failure does not overwrite the previously published artifacts.

The final `release_id` is computed from the four artifact SHAs and the delivery identity, the release directory is
`dist/<release-id>/`, and a Manifest v4 with `deployable=true` is generated:

```text
dist/<release-id>/release.json
dist/current-release.json
```

`current-release.json` is a stable entry point to the same Manifest, not a separate independently editable source of truth.

#### 3.4.3 Independent artifact consistency test

The Release sets `GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1` and runs
`tests/test_artifact_consistency.py`, which verifies that the actual component contents in the source, the three wheels, the Node bundle and the Runtime
Image agree. When this check fails, the already pushed OCI must not be kept or published on its own.

#### 3.4.4 Generate and sign the release attestation

`scripts/build-release-attestation.py` generates:

```text
dist/<release-id>/attestation.json
dist/current-attestation.json
```

The attestation binds:

- `release_id`;
- the SHA-256 of `dist/current-release.json`;
- the delivery identity SHA;
- the Git commit and the clean working tree state;
- `release_tier=production`;
- the SHA-256 of `dist/ci-gate.json`;
- the PASS verdicts of `ci_gate.py verify` and the final artifact consistency check.

The static, coverage, artifact and PostgreSQL stress verdicts from main CI are kept in the bound CI gate;
the Release job does not pretend to have run `make check` again.

`cosign sign-blob` then signs `dist/current-attestation.json` keyless and generates:

```text
dist/current-attestation.bundle.json
```

The deploy phase must verify the signature identity, the Manifest SHA, the release ID and the delivery identity together.

### 3.5 Sign the deploy-host archive and upload

The deploy host offline package and the Node installer bundle are two independent artifacts. This step does not reuse the Node bundle.

`scripts/build-deploy-host-bundle.py` in the CI artifact job:

1. requires a clean source tree again;
2. copies `requirements/build.lock` and `requirements/deploy-host.lock`;
3. completes the full offline wheelhouse from a directory cached by the two locks, OS, architecture and Python ABI;
4. builds the standalone
   `gpu_fault_deploy_host-*.whl` in a temporary venv that installs only `build.lock`;
5. adds the deploy host tool inventory and optional reviewed binaries;
6. records the payload identity, platform, Python ABI, libc and the SHA-256, size and mode of every file;
7. generates a deterministic tar.gz and a `.sha256` sidecar.

The CI workflow persists `DEPLOY_HOST_WHEELHOUSE` with `actions/cache`. On a cache miss it still downloads by
hash lock; on a cache hit it only verifies and fills in missing wheels. Deploy host installation always uses `--no-index`
and cannot resolve dependencies online again.

The deploy-host wheel is rooted at `gpu_fault.admin.cli` and installs the administrator deploy CLI, its Python closure and
`deploy-host-tools.json`; the closure tracks both `gpu_fault` and `gpu_fault_release`, including process supervision, the
private stdio HTTP worker, the SNS policy helper, the prerequisite refresh transaction, the Store proof and the Job cleanup modules.
The workload RBAC check and cleanup modules imported by the standalone deploy scripts are listed explicitly as deploy-host build roots
rather than relying only on the administrator CLI's Python import closure; otherwise a source checkout would run while the offline installation lacked modules.
These helpers do not enter the Control Plane, Executor or Node Runtime business wheels.
Non-Python deploy inputs are still resolved from the trusted checkout: the shared
`deploy/observability/amp-sns-publish-policy.json` enters the Manifest inputs and the observability component
identity, and `deploy/migrations/postgres-schema-preflight-job.yaml`, reused by the database proof, also enters the
Manifest inputs. `scripts/deploy_source_identity.py` additionally binds the release package source and
`wait-for-kubernetes-job.sh` under the deploy orchestration identity; a helper cannot be updated alone while reusing an unbound source snapshot.
The Control Plane Runtime wheel no longer contains `gpu_fault.admin.cli`, the
`gpu-fault-admin` entry point or the deploy-host tool inventory. Administrator code changes therefore rebuild only the deploy-host
bundle; only when a helper alone changes while the Manifest, renderer, node and business component inputs stay the same does the
Runtime Image or the application release diff stay unchanged. Shared asset changes still enter the release component they belong to.

Archive schema v2 no longer writes the Git commit into the payload. Its content identity covers the deploy-host Python dependency
closure, the two locks, the builder, the administrator config template, the tool inventory, the Python ABI, OS/architecture/libc and the optional tool
directory; the same payload produces the same archive across application commits. Commit/tree/repository authorization is carried by the independently signed
CI gate or by the deploy host's locally signed `source-deploy-success.json` record. setup still recomputes the payload identity of the current
checkout and cannot accept an archive merely on the basis of a cached path.

The default Ubuntu x86-64, CPython 3.12 Runner generates:

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz.sha256
```

After verifying the complete candidate, the Release runs only `make deploy-host-sign`, which signs the raw archive above with Cosign
keyless and does not rebuild the project wheel or the wheelhouse:

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json
```

The file name is computed from the build Runner's OS, CPU architecture and Python cache tag. Other platforms must be
rebuilt and signed in the corresponding trusted environment; renaming by hand is not allowed.

After all steps succeed, `actions/upload-artifact` uploads the whole `dist/` as:

```text
artifact name: gpu-fault-release
retention: 30 days
```

`if-no-files-found: error` guarantees that an empty directory cannot form a successful release. This object is a GitHub Actions
artifact, not an automatically created GitHub Release asset, and it is not copied to the deploy host automatically.

## 4. Release artifact inventory

| Location | Producer | Purpose |
|---|---|---|
| `dist/ci-domains/unit/unit-gate.json` | main CI `unit` job | Aggregates the twelve current-run shards and the unified coverage/fault/duration evidence |
| `dist/ci-domains/unit/unit-gate.bundle.json` | main CI `cosign sign-blob` | Sigstore signature of the unit domain gate |
| `dist/ci-domains/unit/shards/<shard>/coverage-shard-gate.json` | the corresponding coverage job | Binds a single shard's content identity, producer and coverage/pytest/duration evidence |
| `coverage-shard-gate.bundle.json` in the same directory | `cosign sign-blob` of the corresponding coverage job | Independent Sigstore signature of a single shard |
| `dist/ci-gate.json` | main CI `test` job | Binds the current source, the source-only candidate, the unit domain gate and the combined quality gates |
| `dist/ci-gate.bundle.json` | main CI `cosign sign-blob` | Sigstore signature material of the CI gate |
| Immutable Runtime Image digests in ECR | `build-release-runtime-image.py` | Runtime images of the CPU Control Plane and the GPU Executor |
| `dist/release-runtime-image.json` | same as above | Binds the OCI digests, platform, inputs and the components inside the images |
| `dist/current-release.json` | `build-release-artifacts.py` | Stable entry point of the current deployable Manifest v4 |
| `dist/<release-id>/release.json` | same as above | Content-addressed release Manifest |
| `dist/<release-id>/gpu_fault_control_plane-*.whl` | same as above | CPU Control Plane |
| `dist/<release-id>/gpu_fault_cluster_executor-*.whl` | same as above | GPU EKS Executor, Watcher, Collector, Reconciler |
| `dist/<release-id>/gpu_fault_node_runtime-*.whl` | same as above | GPU node Agent/Collector |
| `dist/<release-id>/gpu-fault-node-installer-*.tar.gz` | same as above | GPU node install script, unit and the exact Node Runtime wheel |
| `dist/<release-id>/attestation.json` | `build-release-attestation.py` | Content-addressed release attestation |
| `dist/current-attestation.json` | same as above | The attestation consumed by the deploy entry point |
| `dist/current-attestation.bundle.json` | `cosign sign-blob` | Sigstore signature material of the release attestation |
| `dist/gpu-fault-deploy-host-<platform>.tar.gz` | `build-deploy-host-bundle.py` | Deploy host offline Python environment and tool payload |
| `.tar.gz.sha256` of the same name | same as above | SHA-256 sidecar of the deploy host offline package |
| `dist/gpu-fault-deploy-host-<platform>.sigstore.json` | `cosign sign-blob` | Sigstore signature material of the deploy host offline package |

Do not confuse the following three names:

- **Node installer bundle**: installs the Node Runtime on GPU nodes; belongs to Manifest v4, and the old v3 is still supported;
- **deploy-host bundle**: initializes the deploy host venv; does not install GPU nodes;
- **Sigstore bundle**: signature and transparency log verification material; contains no runtime software.

## 5. Trust chain

The release chain is bound layer by layer through the following relationships:

1. the twelve coverage shards each bind their own domain content, test environment and evidence, and are signed by the fixed main CI identity;
2. the unit domain gate verifies the signatures of and aggregates the twelve current-run shards and the unified coverage floor, fault and duration evidence;
3. the CI gate of the current commit verifies the signature of and binds the unit domain gate, and also binds the current tree, static and artifact results;
4. the Runtime Image descriptor binds the OCI digests, build inputs and the image components in the candidate;
5. Manifest v4 binds the three wheels, the Node bundle, the independent runtime images, the node wheelhouse inventory and the delivery identity;
6. the attestation binds the Manifest SHA, release ID, delivery identity, source state and CI gate SHA;
7. the Release Sigstore bundle proves that the attestation came from the approved GitHub OIDC identity;
8. the deploy-host archive uses an independent `gpu-fault-deploy-host` distribution, Manifest and signature,
   binding the platform, dependency identity and the whole payload; the commit authorization of schema v2 is carried separately by the CI gate or the deploy host's locally signed
   success record, and is not written into an archive that can be reused across commits.

A version number, tag, file name, ConfigMap name or Kubernetes annotation alone is never a substitute for the bindings above.
Whenever any layer is missing, a digest disagrees, an identity does not match or an artifact comes from a different workflow run, the process must stop.

## 6. Download, signature verification and deploy consumption

### 6.1 Download and controlled landing

The Release administrator should:

1. select a workflow run that corresponds to the approved commit or `v*` tag and succeeded as a whole;
2. download the complete `gpu-fault-release` artifact; do not assemble artifacts from other runs by file name;
3. restore the release contents to the `dist/...` layout under the matching checkout;
4. as needed, copy the deploy-host archive, `.sha256` and `.sigstore.json` to a
   permission-controlled directory such as `/secure/release/`;
5. keep the file contents and Manifest-relative paths unchanged; do not repackage, rename the platform or rewrite the JSON.

`/secure/release/` is a controlled landing directory chosen by the administrator, not a path generated by CI or discovered automatically.
The 30-day retention of GitHub Actions artifacts is also not a long-term retention policy for production artifacts; approved releases that
need long-term retention should enter a controlled artifact repository before they expire.

### 6.2 Initialize the deploy host

The current CI uses keyless signing, so the normal way to consume it is:

```bash
make deploy-host-setup \
  DEPLOY_HOST_ARCHIVE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz \
  DEPLOY_HOST_SIGNATURE_BUNDLE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json \
  CERTIFICATE_IDENTITY=<approved-release-identity> \
  CERTIFICATE_OIDC_ISSUER=<approved-release-issuer> \
  DEPLOY_HOST_VENV=/secure/gpu-fault/deployer-venv
```

Before installing, setup verifies the signature, the archive contents, platform compatibility and the payload identity of the current clean checkout;
old schema v1 bundles are still matched exactly by the embedded Git commit.
The bundle Manifest binds the OS, CPU architecture, Python implementation/3.12 ABI cache tag, sysconfig platform and
libc implementation; any mismatch fails closed. Payload reuse under schema v2 cannot replace the commit authorization of section 5
or the signature verification of the current release. Dependency installation always uses `--no-index`. The
`dependency_identity_sha256` in the bundle covers the two locks and the platform compatibility information; the same dependency identity creates the shared
dependency venv only once, and subsequent different project wheels create only a lightweight overlay venv that references the shared
site-packages through `.pth`, without installing the two locks again. A missing, corrupted or identity-mismatched dependency layer fails closed; old
bundles without a dependency identity keep the original full installation path. The target venv is replaced atomically only after the full dependency, project CLI and system tool checks pass;
re-running the same bundle only verifies and reuses. The initializer does not call the system package manager, and
the report and the venv must not contain AWS credentials, tokens, private keys, database passwords or kubeconfig.
The deploy-host wheel delivers `gpu-fault-admin`, `gpu-training-submit` and
`gpu-fault-workload-annotate`; the latter two remain the existing Control Plane compatibility entry points.
When the unified deploy internally selects the new venv, it binds the Python PATH of both the CLI and the shell helpers, without depending on the old venv
the caller may still have activated. site commands run directly from the inspected CLI also adopt its interpreter directory; the API budget shim still takes precedence.
The bundle also carries `config/admin-config.example.yaml`, verified against the Manifest digest, which setup installs
to `<deploy-host-venv>/share/gpu-fault/admin-config.example.yaml`. This file is a read-only template for the administrator to prepare the
`0600` configuration input before the first deploy; it contains no credentials and is never written into any state-dir automatically.

A development checkout uniformly runs `make deploy-host-setup-online`: it first validates the paths and version configuration read-only, then runs two Make tasks
in parallel that install the two hash-locked Python/admin CLI environments and prepare the existing `ci-supply-chain-tools` isolated tools.
Both paths are initialized by the system Python 3.12, and an interpreter path can be specified via `DEPLOY_HOST_BOOTSTRAP_PYTHON`;
there is no need to add `make -j`, and if either path fails it waits for the other already started path to finish before returning failure, without rolling back the environment that completed.
CI can still call `ci-supply-chain-tools` on its own.
`DEPLOY_HOST_VENV` and `SUPPLY_CHAIN_TOOLS_VENV` must be separate; explicit paths may contain spaces.
`SUPPLY_CHAIN_PYTHON` must live in the `bin` directory of the tools venv; `PROMTOOL` may point to another explicit installation path
or to a tool already on PATH, but must not fall inside the main venv or the installer's own runtime venv.
The versions and the Prometheus archive SHA-256 are still declared only by the Makefile.
The Python tools first check the actual environment, package versions, entry points and dependency consistency; promtool first verifies the cached archive digest,
then compares the bytes of the binary inside it with the selected tool, and finally reuses the existing tool-only preflight to verify the exact version.
Archive preparation may run in parallel with the Python tool installation, but the binary installation must wait for both to finish; pip writes within the same venv remain serial.
The tool-only preflight depends only on the Python standard library; the full rule check and behaviour tests still need the locked environment.
Only missing or inconsistent tools are reinstalled; a missing or corrupted cached archive is re-downloaded and its digest verified.
The online Python initialization itself still rebuilds the venv and installs by lock; re-running it does not promise to be fully offline.
The tool steps above do not enter the signed `deploy-host-setup`, and do not add tool or test prerequisites to the offline installation.

### 6.3 Consume the signed release

After the controlled artifact sync automation restores the same CI artifact to the trusted checkout, the administrator still runs the unified four-parameter command:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

The CLI internally resolves and verifies the signatures of the attestation, the Manifest and the OCI digests; artifact, certificate identity,
site and low-level `release-deploy` parameters do not enter the ordinary administrator command. Actual Kubernetes/AWS mutations remain
constrained by the Profile, maintenance windows, rollback and fail-closed gates.
The public `preflight` does not modify resources; the `preflight --for-deploy` inside the deploy driver may first persist
the refresher prerequisite journal, update the selected refreshers and create/clean up the CPU proof Job, in order to restore the Store read prerequisites.
This belongs to the authorized deploy transaction and is not a CI gate or a public read-only check; it cannot replace the Store safety proof, the candidate host
preflight or the schema gate before the business startup. See [Developer Deployment Implementation](developer-deployment-implementation.md) for the complete order.

## 7. Failures and reruns

| Failed phase | Handling principle |
|---|---|
| Any parallel main CI job fails | Fix the corresponding static, test, PostgreSQL or artifact problem and push a new commit |
| A single coverage shard misses | Run only that shard; the other shards with the same identity continue to be reused |
| GitHub artifact query or download temporarily unavailable | Safely fall back to a fresh run of that shard; never treat unverifiable historical evidence as passing |
| Shard signature, identity, evidence or historical run is invalid | Fail closed; that shard must not be reused; investigate the artifact and rerun |
| coverage combine or the 78% floor fails | Check for missing shards, stale data or a coverage regression; a single shard must never be accepted on its own |
| per-module floor fails | Add tests for that family; never lower `group_floor`/`file_floor` or narrow the globs to get around it |
| No matching main CI run found | First let the target commit pass main push CI; never substitute a candidate from another commit |
| CI gate signature, source or file inventory fails | Stop the promotion; never acquire an AWS identity or splice in files from another run |
| OIDC, AWS credentials or ECR login fails | Fix the GitHub environment, role trust or least-privilege permissions and rerun |
| repository URI validation fails | Correct the GitHub variable; never bypass the ECR URI check |
| PostgreSQL stress fails | Fix the concurrency/schema problem or the CI service; never substitute ordinary pytest |
| Runtime Image tag/label mismatch | Investigate input or registry drift; never overwrite a mismatched immutable tag |
| wheel, Node bundle or image consistency fails | Stop the release and investigate toolchain, lock or source closure drift |
| attestation/Cosign fails | Check the OIDC identity, the `id-token` permission and the signing service, then rerun |
| deploy-host bundle fails | Check the lock wheels, Python 3.12, the platform and the clean working tree |
| artifact upload fails | The whole run counts as failed; never deploy from files taken only from the Runner's temporary directory |

If the OCI has been pushed but a later step fails, that OCI cannot be treated as a releasable release on its own. Only after the complete,
signed and successfully uploaded Manifest, attestation and related artifacts are obtained may they be handed to the deploy phase.

When the same clean commit is rerun, an exactly matching immutable OCI may be reused; the new run should still be treated as an independent set of release
results, and the signatures or files of two runs must not be mixed.

## 8. Local trusted environment reproduction

A source environment first runs `make deploy-host-setup-online`, which prepares the administrator CLI and the pinned-version isolated gate tools in one go;
CI keeps the separate `ci-supply-chain-tools` target. The local full test run, `make check` and the release gates run
`promtool-preflight` before the expensive steps, verifying executability and the exact version declared by the Makefile,
and pass the path to pytest, xdist and subprocesses. A missing tool or a version mismatch fails immediately, is not recorded as a skip, and is not
downloaded or installed automatically during tests. Success of the separate CI `promtool-check` cannot replace the dependency preflight of another local test run.
Targeted impact tests require the tool only when a PromQL test from the Make inventory is selected, and pass the absolute path returned by the preflight
to the actual pytest subprocess; other selections and the reuse of signed CI artifacts do not add local test or tool requirements because of this.

Local reproduction is for Release engineering and troubleshooting; it does not replace formal CI approval. Prerequisites include Python 3.12,
an isolated PostgreSQL 16, ECR login, Docker Buildx, Cosign, and a
`GPU_FAULT_TEST_POSTGRES_URL` that has been set securely but is not printed.

```bash
test -n "${GPU_FAULT_TEST_POSTGRES_URL}"

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<ecr-repository>

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python deploy-host-bundle
```

The local example uses a controlled private key; the GitHub workflow uses keyless OIDC. The deploy signature verification parameters differ between the two signing modes,
and a keyless bundle must not be handled as a fixed public key signature.

## 9. Changes and verification

When changing the release process, at least synchronize:

| Change | Must review |
|---|---|
| workflow trigger, permissions, Runner or GitHub variables | Sections 2 and 3 of this document |
| CI parallel jobs, the CI gate or `release-build-promoted` | Sections 3.1 to 3.4 of this document, Developer Deployment Implementation and the operations manual |
| `release-build-staging` or the two-tier attestation boundary | The EC2 Source Staging process, the security reference and section 1 of this document |
| Manifest, wheel, Node bundle, attestation | Sections 4 and 5 of this document and the corresponding artifact tests |
| deploy-host bundle, shared dependency layer, platform or lock | Sections 3.5 and 6.2 of this document and Developer Deployment Implementation |
| artifact name, path or retention | Sections 3.2, 3.5, 4 and 6.1 of this document |
| The responsibility boundary between CI and the actual deploy | Sections 1 and 6.3 of this document and the administrator documents |

Documentation-only changes must still run:

```bash
make docs-check
```
