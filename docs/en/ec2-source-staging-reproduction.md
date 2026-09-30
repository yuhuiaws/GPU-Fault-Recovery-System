# EC2 Source Unified Deployment Process

English edition of `docs/EC2源码Staging复现流程.md`; the Chinese file remains the source of record until both are maintained together.

This document describes how to complete the first deploy, dirty-code testing, failure resume and subsequent upgrades from source on one controlled CPU EC2 instance.
The four scenarios use exactly the same public command:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

For multiple GPU clusters, pass `--gpu-cluster-arn` repeatedly. After the first cluster forms the baseline, the remaining clusters join through independent batch joins
(rolling per the site's `upgradeMaxParallelClusters`, one cluster at a time by default). The operator does not provide release-ref, artifact, site,
release-build, release-deploy, bundle or venv paths.

## 1. Prerequisites

- An existing CPU EKS or HyperPod EKS cluster with at least 3 Ready nodes.
- At least one existing GPU HyperPod EKS cluster with `NodeRecovery=None`.
- The GPU VPC already has a NAT egress.
- The EC2 execution identity holds the AWS, EKS and Kubernetes permissions required by the staging bootstrap.
- The EC2 instance has Python 3.12, Git, Make, Docker/Buildx, AWS CLI, Cosign, kubectl, Helm,
  curl, jq, OpenSSL and sha256sum installed.
- No pre-installed signed deploy-host bundle is required: build the source development venv with `make deploy-host-setup-online` per §2;
  the unified deploy command builds, verifies the signature of and binds this site's trusted deploy-host by itself inside `STATE_DIR/deployer-venv` (see §8);
  the signed offline bundle is only for production deploy hosts, see [CI Release Process](ci-release-process.md).
- The CPU/GPU cluster ARNs belong to the same account and Region.

The state directory must be located outside the Git repository. This process performs no GPU reset, node reboot, warm-spare switch or
fault injection.

## 2. Clone and initialize

```bash
git clone https://github.com/yuhuiaws/GPU-Fault-Recovery-System.git
cd GPU-Fault-Recovery-System

git switch main
git pull --ff-only origin main
git status --short

make deploy-host-setup-online
. .venv/bin/activate
```

The first deploy should be executed from the repository root. After success, the internal state records the trusted source root directory, and later runs do not require the operator
to provide the repo path.

## 3. Check system tools and identity

```bash
python3.12 --version
docker info
docker buildx version
aws --version
cosign version
kubectl version --client
helm version
aws sts get-caller-identity
```

If any command fails, first fix the EC2 image, the tool installation or the AWS temporary identity.

## 4. Set the public inputs

```bash
export CPU_CLUSTER_ARN='<CPU EKS or HyperPod ARN>'
export GPU_CLUSTER_ARN='<GPU EKS or HyperPod ARN>'
export STATE_DIR=/secure/gpu-fault-staging
export ADMIN_EMAIL='<operations email>'
```

## 5. First and subsequent deploys

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn "${CPU_CLUSTER_ARN}" \
  --gpu-cluster-arn "${GPU_CLUSTER_ARN}" \
  --state-dir "${STATE_DIR}" \
  --admin-email "${ADMIN_EMAIL}"
```

The same command applies to:

1. no site exists in state: first deploy;
2. a site already exists in state: subsequent upgrade or NOOP verification;
3. the previous run failed midway: idempotent resume;
4. a developer modified the current checkout: build and deploy a new source candidate.

## 6. Internal state machine

```text
no site exists in state
  -> scan and pin the current source identity
  -> discover the CPU/GPU clusters
  -> create the runtime/cache ECR
  -> bind the base resource task inputs
  -> release build, signing/signature verification and independent base resource tasks in parallel
  -> converge Aurora, NLB, DNS/PKI, monitoring and IAM
  -> generate the site internally
  -> write the registry of the first baseline GPU cluster -> CPU/endpoint -> first GPU bootstrap
  -> verify -> stability -> remaining GPU clusters join in independent batches

a site already exists in state
  -> while the first multi-GPU join is incomplete, verify the original targets are unchanged and the current members are a monotonic subset of them
  -> once all original targets are managed or a complete finished removal proof exists, require the CPU unchanged and the requested GPU set to be the managed set or a superset of it; extra clusters auto-join after the release
  -> scan and pin the current source identity
  -> application identity unchanged:
       -> deploy-host changed: update the deploy-host and run the impact gates and the read-only preflight
       -> tests/docs only changed: run only the impact gates
       -> completely unchanged: reuse the authorization only when site, live release, commit status and NOOP classification have all not drifted
  -> application identity changed: verify the signature of or build a new signed release
  -> compute NOOP/CONTROL_PLANE_ONLY/DATA_PLANE_COMPATIBLE/FULL
  -> upgrade -> verify -> stability
```

`deploy` accepts a superset of the managed set and joins the added clusters after the release through an independent checked join transaction (`join-cluster` is its alias).
An incomplete multi-target first deploy stays bound to the original targets and order; the commitment cannot be released with a partial site.
Passing fewer managed clusters or replacing the CPU is still refused; removal must use the checked `remove-cluster` process.

## 7. The two release tiers, Dirty and Clean

| Current source | Internal gates | Release tier |
|---|---|---|
| dirty working tree | public scan, impact tests, regional impact plan; escalates to full when uncertain | `staging_only=true` |
| clean commit and `HEAD != origin/main` | local full production gates | production |
| clean commit and `HEAD == origin/main` | prefer verifying the signature of the same-commit main CI candidate; fall back to the local full gates when unavailable | production |

Both clean and dirty source are first copied to an isolated
checkout under `state-dir/source-snapshots/<fingerprint>/`; subsequent build, signing, site and config-only releases reference only that checkout and no longer reference the mutable
development working tree. Clean source keeps the original commit and produces a production release; a dirty working tree is additionally converted into
an isolated local temporary commit, the current branch, index and working tree are not modified, and a staging-only release is produced.
The staging attestation is bound to the impact-test baseline. Ordinary production signature verification refuses staging-only releases by default.

The release-ref is computed internally from the Git commit or the dirty snapshot; it is not a public parameter.

## 8. Automatic reuse

The same command redoes work only when an identity has actually changed:

| Object | Condition under which reuse is allowed |
|---|---|
| Source snapshot | HEAD, tracked diff, untracked source, file permissions and the post-preparation tree digest are all identical |
| deploy-host bundle and venv | deploy-host dependency closure, both locks, Python ABI, OS/architecture/libc, tool inventory and signature are identical |
| deploy-host wheelhouse | `build.lock`, `deploy-host.lock`, platform and Python ABI are identical |
| deploy-host dependency layer | the dependency identity computed from both locks and the platform is identical and healthy |
| source-only component artifacts | component build identity is identical; the delivery Manifest is regenerated |
| Runtime Image | the full image input digest is identical and the ECR digest still exists |
| Signed release | commit, release tier, impact baseline, signature and runtime repository are identical |
| AWS resources | site ownership, ARN, tags, configuration and lifecycle policy are identical |
| Kubernetes release | the release diff is explicitly classified as NOOP or a limited upgrade |

The deploy-host archive is stored under
`STATE_DIR/deploy-host/by-content/<payload-sha256>/` and is no longer rebuilt because of unrelated application commit changes.
The Git commit, source fingerprint, application/deploy-host identities and success mode are written to a separate locally Cosign-signed
`source-deploy-success.json` authorization record; when the record is missing, the signature does not match or the application identity changed, the
deploy-host-only fast path must not be taken. `UNCHANGED` additionally requires that the site digest, live release ID,
`phase=complete`, `transaction_committed=true`, release-state digest and the live Runtime Profile
policy digest in the signed record match the current read-only status exactly, and that `next_deploy.kind=NOOP`; any drift falls back to the checked
application release. The Runtime Profile template (`spec.runtimeProfile.templateSource`) lives outside the repository
and the site file, so the live evidence also compares the template with the live Profile using the release engine's `plan_runtime_profile`:
whenever the result is not `UNCHANGED`, every mode (including deploy-host-only and quality-only) switches to the
application release and stops as expected at `release-deploy/profile-plan.json` waiting for approval.

Clean source accesses GitHub Actions only when all of the following conditions hold:

1. the working tree is clean;
2. `HEAD` is exactly identical to the local `refs/remotes/origin/main`;
3. the current isolated checkout has no candidate that has already been signature-verified and is reusable;
4. a `GPU_FAULT_GITHUB_TOKEN_FILE` with permission `0600` is configured, or
   `GITHUB_TOKEN`/`GH_TOKEN` exists in the controlled process environment.

A dirty working tree, a personal clean commit, or a local `origin/main` that does not point at the current commit does not query GitHub and
directly uses the local impact gates or full gates; the implementation never runs `git fetch` automatically. After downloading a candidate, before any AWS/ECR
action it verifies the main run, commit/tree/repository, CI gate, unit gate, the six shard signatures and the full
artifact inventory. With no credentials, no candidate, or a temporarily unavailable API it falls back to the local gates; an already downloaded candidate whose identity
or signature does not match fails closed.

The local production fallback places static/contract, ordinary pytest and PostgreSQL stress in three isolated cache directories:
the static gate, about 1 minute, runs alone first, and on failure stops without starting the other two; after static passes, ordinary pytest and
PostgreSQL stress run in parallel, adaptively at 1 to 3 ways according to CPU capacity, and can be
narrowed further with `GPU_FAULT_RELEASE_GATE_PARALLELISM=1..3`. Any failing parallel gate terminates the remaining gates,
and after the progress output reprints the last roughly 20 lines of that gate as `release-gates: first failing gate: <name>`.
`release-build: release_source=…` at the start of the build states whether this run takes the local full gates or a main CI candidate.
Artifacts are built and checked only after everything passes.
The gate set is the same as the original `make check` plus stress; it does not lower the coverage floor or omit the PostgreSQL stress.
Inside the static branch, Ruff, mypy, compile, architecture, code contracts, security, deployment configuration, docs,
YAML and Shell are still split into up to 10 ways; `GPU_FAULT_STATIC_GATE_PARALLELISM=1..10` can narrow it deliberately.

The mypy/ruff/pytest cache directories of the gates live under `state-dir` and are reused by both the dirty and clean paths. After each successful deploy,
`state-dir/source-snapshots/<fingerprint>/` and `state-dir/.deployer-venv.versions/*` are cleaned per the retention rules:
snapshots referenced by `source-deploy.json` and `source-deploy-success.json`, the snapshot of the current deploy and the latest
`GPU_FAULT_SOURCE_SNAPSHOT_RETAINED` (default 5, do not go below 2) are kept; venv versions keep the currently activated one, the state-bound one,
the one pointed to by `.deployer-venv.previous` and the latest `GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED` (default 3).
One deploy collects the pre-deploy live evidence once inside the site lock, and collects it once more after a successful deploy as the success record.

Whenever any signature, commit, platform, digest, cluster identity or resource state is uncertain, it fails closed.
The content-addressed release Manifest and signing material in the snapshot are retained along with the site;
later test runs or newly generated `dist/` in the development checkout do not change the deployed site's
`status/verify/config` inputs.

Dirty staging first generates a
`dist/staging-impact-plan.json` bound to `BASE`, the changed files and the plan SHA. The static checks, pytest, the regional impact list and the attestation
all consume that file; on read the current changed files are still re-checked, and plan drift fails closed. Only when the plan
is marked `postgres=true` is a temporary PostgreSQL 16 started; ordinary non-database changes no longer
start a container just to satisfy the empty-parameter check.

The component cache lives at `STATE_DIR/component-artifacts/` and is not part of the development checkout's `dist/`. It only reuses
the physical files of the three wheels and the Node installer bundle; a new isolated snapshot still recomputes the delivery identity
and generates its own Manifest. The cache identity covers inputs such as the component source and the Node bundle scripts/units, and any
mismatch or file corruption automatically falls back to a rebuild.

The trusted deploy-host venv installed by the unified source deploy binds itself to its owning
`state-dir` with a `0600` file. From then on, if `gpu-fault-admin` in that installation receives another `--state-dir`, it refuses outright before the source scan,
AWS queries or Kubernetes access. To manage another site you must use the deploy-host installed by that site's own deploy
process; the CLI of another state directory cannot be reused.

The third-party dependencies in the deploy-host bundle are installed per lock and platform to
`STATE_DIR/.deployer-venv.dependencies/<dependency identity>` (the shared dependency layer adjacent to the deployer-venv).
When the project wheels are updated but the dependency identity is unchanged, only the lightweight
overlay venv is rebuilt, and all third-party wheels are no longer reinstalled; when the dependency layer check fails, the new venv is not activated.

## 9. The loop after changing code

When the related tests are known, you can first run one minimal test for fast feedback:

```bash
.venv/bin/python -m pytest -q tests/<known-related-test>.py
```

Then directly repeat the four-parameter command of section 5. When tests or the deploy fail, keep changing the source and repeat the same command again.

Make the formal commit only after all staging tests pass:

```bash
git diff --check
git status --short
git add <files confirmed for this change>
git commit -m "<change message>"
git status --short
```

Once the working tree becomes clean, repeat the same four-parameter command again. Now it switches internally to the production tier and does not reuse the earlier
staging-only attestation; a personal commit runs the local production gates, and only when
`HEAD == origin/main` does it try to consume the same-commit signed main CI candidate.

A real Runtime Profile change must still go through independent approval. The first four-parameter command writes
`STATE_DIR/release-deploy/profile-plan.json` and stops; after reviewing the plan and the change ticket, rerun on the same deploy
with the approval parameter:

```bash
gpu-fault-admin deploy \
  --state-dir "${STATE_DIR}" \
  --approve-profile-plan "$(jq -er '.plan_sha256' \
    "${STATE_DIR}/release-deploy/profile-plan.json")" \
  --reference CHG-12345
```

This command approves first and then continues the same deploy; there is no separate approval verb. The approval record is bound to the specific plan digest and the live
Profile baseline and enters only through this one explicit parameter. When the release fails and the plan is unchanged, the approval is kept for the resume;
when the template or the live baseline drifts, the old approval is archived and invalidated; the approval is consumed and the active record deleted only after verify, stability and commit all
succeed.

## 10. Success and failure

When the application release changes, the deploy, verify, stability window and release
summary are already complete before the command returns successfully. When only the deploy-host changes, only the deploy host environment is updated and the impact gates and read-only preflight run; no
application release is generated, the GPU rollout is not accessed and no stability window is waited for; when only tests/docs change or the source is completely unchanged, a
NOOP is recorded and no AWS/Kubernetes mutation is executed. The operator no longer needs to provide the internal site path to run a second deploy command.

On failure:

1. keep `STATE_DIR`;
2. fix the source, permissions, AWS resources or external dependencies;
3. rerun with exactly the same four-parameter command;
4. do not delete the bootstrap/release state, do not hand-edit artifacts or edit Deployments online.

## 11. Separation of production duties

Dirty candidates may only be used for isolated staging. For formal production, it is recommended that GitHub Release CI produces
signed artifacts for the final clean commit; after the deployment automation places the approved artifacts in the trusted environment, the administrator still runs the same four-parameter
`gpu-fault-admin deploy`. The artifact and site paths are resolved internally and do not enter the public command.
