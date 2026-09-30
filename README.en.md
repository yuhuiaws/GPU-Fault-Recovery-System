# Automated Fault Handling for Multi-Node Distributed GPU Training

English edition of [`README.md`](README.md); the Chinese file remains the source of record until both are maintained together.

This repository implements fault collection, policy decisions, isolation, recovery orchestration, training-job recovery and acceptance tooling for large-scale GPU training clusters.

## Supported Scope

The only production delivery form today is:

```text
Regional CPU EKS control plane + Aurora PostgreSQL + NLB
                    |
                    +-- HyperPod EKS GPU data cluster A
                    +-- HyperPod EKS GPU data cluster B
                    +-- ...
```

Do not infer additional supported scope from historical code or examples in the repository:

- The single-cluster all-in-one HyperPod form is used only for migration, canary and maintenance of historical environments.
- Generic Kubernetes deployment is not implemented as a production delivery capability.
- HyperPod Slurm orchestration is not supported.
- The systemd Agent on GPU nodes belongs to the regional data plane; it is not a standalone deployment architecture.

## Deployment Recommendations, the EC2 Deploy Host and the Verified Environment

- **Cluster creation**: create the GPU data-plane and CPU control-plane clusters separately from the SageMaker HyperPod UI, then deploy this solution with `gpu-fault-admin`.
- **Separation of duties**: the recommended chain is "developer / Codex workspace -> GitHub Release CI build and signing -> controlled EC2 deploy host that verifies signatures, deploys and operates". The production deploy host does not rebuild or re-sign releases and is not a day-to-day source development machine.
- **EC2 deploy host**: use a dedicated CPU EC2 instance in a private subnet with no public IP; access it through SSM Session Manager, and enable IMDSv2, encrypted EBS, system auditing and controlled egress. Python, Docker/Buildx, AWS CLI, Cosign, kubectl, Helm and the optional Codex CLI should come from a pinned AMI or an audited installation process.
- **IAM**: never bind `AdministratorAccess` to the EC2 instance long-term. The instance profile only carries SSM and controlled artifact reads; CI uses a GitHub OIDC build role, and deployment and operations assume separate least-privilege roles through STS temporary credentials. The first bootstrap needs broader permissions, but they must still be restricted to the account, Region, clusters and solution resources, with break-glass auditing retained.
- **Codex boundary**: the Codex CLI is an optional development, review and runbook assistant, not a production trust root. Never write tokens, private keys, database passwords or kubeconfig contents into its prompts, logs or workspace; every AWS/Kubernetes mutation still needs human confirmation, a maintenance window, stop conditions and a rollback plan. Combining development, build and deploy on one EC2 instance is allowed only for disposable, isolated staging verification.
- **Codex installation**: install according to the [official Codex CLI documentation](https://developers.openai.com/codex/cli). Authentication, endpoint and model mapping are decided by the organisation-approved OpenAI, AWS Bedrock or internal gateway configuration; this README does not fix a provider or model command.

  ```bash
  npm install -g @openai/codex
  ```

  Do not make a specific provider or model name a release gate; release facts still come from the Git commit, tests, Manifest and signatures.
- **Test baseline**: verified mainly on AWS `ml.p5en.48xlarge` H200; other instance types require re-verification of GPU/EFA, drivers, DCGM and interconnect topology. The components consume no GPU compute; by default each node samples lightly every 15 seconds, reports inventory every 60 seconds and checks the Fabric Manager log every 5 seconds.

## Hard Constraints of the Solution

The following constraints are not tunable defaults:

1. Managed HyperPod GPU clusters must be set to `NodeRecovery=None`.
2. This solution never calls `BatchReplaceClusterNodes`; node replacement only uses healthy warm spare nodes that are already managed.
3. HyperPod Job Auto Restart and EKS auto-resume must be disabled.
4. Training-job recovery is managed exclusively by this solution's Kubernetes Adapter and restart budget.
5. When the policy is unknown, evidence is missing, versions are inconsistent or the workload state is unknown, destructive actions must
   fail closed.

## Core Path

```text
Collectors / Watcher -> Regional ingress and queue -> Policy / Incident / Workflow
                     -> Per-cluster Executor / signed Node Agent
                     -> Validation / scheduling and workload recovery
```

Main capabilities:

- NVIDIA XID Catalog and Fabric Manager SXID policies.
- Correlation of GPU UUID, node, attempt, workload and fabric partition.
- Multi-replica processor, lane lease, fencing token and idempotent workflows.
- GPU service quiesce, GPU reset, node reboot and warm-spare failover.
- Completion Watcher, training restart budget and GPU-count consistency gate.
- Administrator notifications through the site SNS topic (optional SES), AMP/Alertmanager/SNS observability and long-term evidence archiving.

## Documentation Entry Points

Links below point at the English editions; the corresponding Chinese originals are listed in the [Documentation Index](docs/en/README.md).

- [Documentation Index](docs/en/README.md) / [Contributing Guide](CONTRIBUTING.en.md)
- [Administrator Quick Deploy](docs/en/administrator-quick-deploy.md) / [Runtime Profile Change Approval](docs/en/administrator-profile-change-approval.md) / [Administrator Operations](docs/en/administrator-operations.md) / [Security and Parameters Reference](docs/en/security-and-parameters-reference.md)
- [High-Level Design](docs/en/high-level-design.md) / [High-Level Design v2](docs/en/high-level-design-v2.md) / [Fault Categories and Actions](docs/en/fault-categories-and-actions.md)
- [Detailed Design](docs/en/detailed-design.md) / [Detailed Design v2](docs/en/detailed-design-v2.md)
- [NVIDIA Policy Supply Chain and Implementation](docs/en/components/nvidia-policy.md) / [Deployment and Operations Manual](docs/en/deployment-and-operations-manual.md) / [Manual Walkthrough](docs/en/deployment-and-operations-manual-walkthrough.md)
- [Unified EC2 Source Deployment Process](docs/en/ec2-source-staging-reproduction.md) / [CI Release Process](docs/en/ci-release-process.md) / [Developer Deployment Implementation](docs/en/developer-deployment-implementation.md)
- [Environment Variables Reference](docs/en/administrator-environment-variables.md) / [Performance Acceptance Plan](docs/en/performance-acceptance-plan.md) / [Extension Guide](docs/en/extension-guide.md)
- [Collectors](COLLECTORS.en.md) / [Training Job Examples](examples/README.en.md) / [Deployment Layout](deploy/README.md) / [Scripts and Tools](scripts/README.md)

## Build and Verification

Python 3.12 is required.

```bash
make deploy-host-setup-online
. .venv/bin/activate
make PYTHON=.venv/bin/python check
```

After validating paths and versions, `deploy-host-setup-online` uses the system Python 3.12 to prepare the locked Python/admin CLI venv and the
separate tooling venv in parallel; no extra command or `make -j` is needed, and the promtool download overlaps with the tooling install. If either
path fails, the target waits for the started tasks to finish and then returns failure. Tools are reused once their version and cached archive digest
verify, and are repaired only when missing or mismatched; the online Python initialisation still rebuilds the environment and may reach the network.
A production deploy host must use the CI-built, signature-verified offline bundle, see [CI Release Process](docs/en/ci-release-process.md).
After initialisation, Make automatically uses `.venv/bin/python`; a source package without a local venv falls back to `python3`, and CI or
advanced invocations can still override it explicitly with `PYTHON=...`.

When running the gates one by one, the recommended order is:

```bash
make mypy-check
make architecture-check
make docs-check
make artifact-check
make test-parallel
```

The local full entry point first checks the pinned `PROMTOOL` version and passes it to subprocesses; `make check` uses 4 to 16 workers by default.
External PostgreSQL does not run under xdist; set `GPU_FAULT_TEST_POSTGRES_URL` and run `make test-postgres` separately.

`make artifact-check` builds the three independent wheels, the Node bundle and the content-addressed
`dist/current-release.json`, and verifies module boundaries, digests and reproducible rebuilds.

## Deployment: Identify Your Role First

Every production deployment must satisfy these common prerequisites before it starts:

1. An existing CPU EKS/HyperPod cluster with at least 3 Ready nodes.
2. At least one existing GPU EKS/HyperPod cluster; a GPU HyperPod cluster must have `NodeRecovery=None`.
3. The GPU VPC already has NAT egress.
4. The executing identity holds the required AWS, EKS and Kubernetes administrative permissions.
5. The deploy host has the Python 3.12 project environment plus `aws`, `cosign`, Docker Buildx, `kubectl`, `helm`, `curl`, `jq`, `openssl`, `sha256sum` and `make`; the command only verifies these prerequisites and never installs system tools.
Do not apply `deploy/` directly. A first deployment uses only the ARN-based single-command entry point:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-eks-or-hyperpod-arn> \
  --gpu-cluster-arn <gpu-eks-or-hyperpod-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

Repeat `--gpu-cluster-arn` for multiple GPU clusters. The command decides by itself whether this is a first or a subsequent deployment, manages
the release, signing, bundle, venv and site internally, and completes preflight, deploy/upgrade, verify and stability.
A first deployment first sends an SNS confirmation e-mail to `--admin-email`; while unconfirmed, the command exits within the first minute (nothing was deployed). Click the link and rerun with the same arguments, or add `--wait-for-email-confirmation <minutes>` to wait in place.
Day-to-day changes after the first deployment use `<state-dir>/deployer-venv/bin/gpu-fault-admin`.
A development checkout that is not bound to a site cannot replace that CLI for changes to an existing site; ordinary source deploys and truly
read-only queries are validated by their own entry points, and rollback does not enjoy the exemption granted to ordinary source deploys. See
the administrator entry-point notes in [Administrator Operations](docs/en/administrator-operations.md).

### Developers: Releasing After Code or Profile Changes

A first source staging deployment, dirty iteration, resuming after failure and clean-commit verification all use the same command:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

A dirty working tree is turned by the internal preparer into an isolated `staging_only` snapshot with impact tests; a clean commit runs the
full production gates. Users never supply a release-ref, artifact, site, bundle or venv path. The full process is in
[Unified EC2 Source Deployment Process](docs/en/ec2-source-staging-reproduction.md).

If the Runtime Profile policy changes, the first deploy generates a private plan and stops. After review, rerun the same deploy with the
approval arguments to approve and continue the deployment:

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault-staging \
  --approve-profile-plan "$(jq -er '.plan_sha256' \
    /secure/gpu-fault-staging/release-deploy/profile-plan.json)" \
  --reference CHG-12345
```

There is no separate approval verb. The approval is bound to the plan and the live baseline and is consumed once on success; plan drift
requires a new approval.

### Administrators: First Deployment and Day-to-Day Management

Administrators repeat the same four-argument command for the first and every subsequent deployment. The Region is derived from the cluster ARNs
and must be identical across them; an existing site first verifies that the CPU/GPU identity set is unchanged, then verifies signatures and runs the upgrade.

#### 1. First Deployment

Signing material is generated or reused by the internal preparer in private state and never appears in public arguments. The site's base
resources for a first deployment are created and managed uniformly by the ARN-based administrator entry point.

#### 2. Upgrading an Existing Site

Keep using exactly the same command as the first deployment, without supplying the internally generated site path:

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

Other lifecycle operations still use the validated administrator subcommands:

```bash
# Register an existing GPU cluster
gpu-fault-admin join-cluster --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <gpu-arn>

# Deregister a GPU cluster; keeps that GPU EKS/HyperPod cluster and the other clusters
gpu-fault-admin remove-cluster --state-dir /secure/gpu-fault \
  --gpu-cluster-arn <gpu-arn> --confirm REMOVE_GPU_CLUSTER

# Uninstall the whole solution control plane and data plane, keeping the underlying CPU/GPU clusters
gpu-fault-admin uninstall --state-dir /secure/gpu-fault \
  --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT
```

For permanent decommissioning that also deletes the underlying CPU EKS/HyperPod cluster, use the stronger confirmation:

```bash
gpu-fault-admin uninstall --state-dir /secure/gpu-fault \
  --cpu-cluster delete \
  --aurora-final-snapshot retain \
  --confirm DELETE_CPU_CONTROL_PLANE
```

GPU EKS/HyperPod clusters are always kept. After a `deploy` failure or automatic rollback, rerun the same four-argument command;
`join-cluster` and `remove-cluster` also resume idempotently through persistent state.
Detailed upgrade, troubleshooting and decommissioning rules are in
[Administrator Operations](docs/en/administrator-operations.md); object-by-object audit and break-glass are in
[Deployment and Operations Manual](docs/en/deployment-and-operations-manual.md).

## Training Job Submission

Two managed submission methods are supported:

```bash
gpu-training-submit --site /path/to/site.yaml customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1
```

Or inject the managed metadata first and let the customer run the apply:

```bash
gpu-fault-workload-annotate --site /path/to/site.yaml customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1 \
  --training-container trainer \
  -o /tmp/customer-job.managed.yaml

kubectl apply --dry-run=server -f /tmp/customer-job.managed.yaml
kubectl apply -f /tmp/customer-job.managed.yaml
```

The customer training image is not this solution's control-plane image. The training image only has to satisfy the requirements of the
framework, NCCL/EFA, checkpointing and the training command.

## Security

- Never cause real hardware damage on production GPUs.
- `/dev/kmsg` injection may only run on approved isolated test nodes during a maintenance window.
- Execution tokens, cluster tokens, Node Action keys, database passwords and the private CA
  must never be written into source code, documentation, command history or ordinary artifacts.
- Test, canary, probe and fault-injection manifests are not production deployment assets.
- Every node action must pass the target-node, boot/incarnation, fencing-token, workload
  and maintenance-window gates.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
