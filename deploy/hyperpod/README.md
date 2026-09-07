# AWS HyperPod deployment

> This entry point is retained for historical single-cluster migration and
> canary validation. New production deployments use the regional workflow in
> `docs/部署和运维手册.md`.

> **Reminder for anyone editing `deploy.sh` (verified against the code on the
> regional architecture):** nothing in `src/gpu_fault/`, the `Makefile`,
> `scripts/` or `src/gpu_fault_release/` executes this script. The only
> production path is `gpu-fault-admin deploy` → `scripts/staging_deploy.py` →
> `deploy/control-plane/regional/rollout-regional-release.sh`. A change made
> only here never reaches a regional site. The file still changes because
> contract tests read its text as a specification
> (`tests/regional/test_production_safety_config.py`,
> `tests/regional/test_performance_defaults.py`,
> `tests/node_agent/test_node_deployment.py`), so when you add behaviour here,
> land the regional equivalent first — schema Jobs in
> `src/gpu_fault_release/regional_release_rendering.py`, rollout
> phases in `src/gpu_fault_release/regional_release_orchestration.py`,
> Runtime Profile claims in
> `config/runtime-profile.regional-hyperpod-safe.example.yaml` — and treat the
> edit here as parity, not as a deployment. Do not debug a regional release by
> reading this script.

The deployment entry point installs the GPU fault control plane on a
HyperPod EKS cluster and verifies the complete passive recovery path.

## Prerequisites

- AWS CLI credentials with EKS, EC2, RDS, Secrets Manager and IAM access.
- `aws`, `kubectl`, `python3.12`, `openssl` and `sha256sum`.
- At least two private EKS subnets in different availability zones.
- HyperPod GPU nodes with Python 3.12, systemd, NVIDIA driver and
  `nvidia-smi`.
- HMA/DCGM hostengine already provided by HyperPod.

## Deploy

The script refuses to run unless `GPU_FAULT_ALLOW_LEGACY_DEPLOY=1` is set
(exit code 64 with a pointer to `gpu-fault-admin deploy`), so it cannot be
started by habit next to a regional site.

```bash
export GPU_FAULT_ALLOW_LEGACY_DEPLOY=1
export AWS_REGION=us-west-2
export EKS_CLUSTER_NAME='<gpu-eks-cluster>'
export HYPERPOD_CLUSTER_NAME='<gpu-hyperpod-cluster>'
export GPU_FAULT_DCGM_EXPORTER_MODE=auto

deploy/hyperpod/deploy.sh deploy
```

`GPU_FAULT_DCGM_EXPORTER_MODE` accepts `auto`, `existing`, `managed`, or
`disabled`. `auto` probes every HyperPod node and reuses an existing
compatible Prometheus endpoint; otherwise it deploys this solution's
exporter. To require a platform-provided exporter:

```bash
export GPU_FAULT_DCGM_EXPORTER_MODE=existing
export GPU_FAULT_DCGM_METRICS_URL='http://{node_ip}:9400/metrics'
```

The endpoint must expose the configured fault-decision DCGM fields on every
node. An observability add-on that only remote-writes metrics to AMP is not a
compatible local endpoint.

`NvidiaSmiMetricsCollector` is disabled by default because it has not
completed additional production validation. Setting
`GPU_FAULT_DCGM_EXPORTER_MODE=disabled` fails deployment unless
`GPU_FAULT_ENABLE_NVIDIA_SMI_METRICS_COLLECTOR=true` is also explicitly set
for an isolated validation environment.

The control plane enables the read-only HyperPod managed-recovery observer
by default. It refreshes `NodeLogicalId` identity mappings every 20 seconds
and waits up to 45 minutes for AWS-managed recovery. Override these values
with `GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER`,
`GPU_FAULT_HYPERPOD_IDENTITY_REFRESH_SECONDS`, and
`GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS`.

The Kubernetes HMA Node watcher and CloudWatch HMA forwarding pipeline are
disabled by default. Their current control-plane path only acts on XID/SXID,
which are covered by the node kernel and Fabric Manager collectors. Keep
`GPU_FAULT_ENABLE_KUBERNETES_HMA_COLLECTOR=false` and
`GPU_FAULT_ENABLE_CLOUDWATCH_HMA_COLLECTOR=false` unless validating those
forwarding paths in an isolated environment.

The numbered steps:

1. Connect to EKS and validate API access.
2. Create the namespace and cryptographic execution secrets.
3. Create or reuse encrypted Aurora PostgreSQL Serverless v2 with a writer
   and cross-AZ failover reader.
4. Build and publish the wheel and node installer ConfigMaps.
5. Deploy three control-plane replicas, shared Service, RBAC and PDB.
6. Deploy CompletionWatcher and configure the selected DCGM metrics source.
7. Install the kernel collector, metrics collector and signed Agent on every
   discovered HyperPod node.
8. Validate Aurora, endpoints, DCGM and Agent readiness.
9. Run a no-GPU Indexed Job E2E through failure, suspend, triage, fenced
   workflow, retry and success.

Wheel ConfigMaps are content-addressed with the wheel SHA-256 prefix. The
deployment renders that immutable name into the control-plane and watcher
manifests so a rollout cannot install a stale cached wheel.

The E2E does not execute GPU reset, node reboot or node replacement.

## Repeat validation

```bash
RUN_E2E=false deploy/hyperpod/deploy.sh validate
deploy/hyperpod/deploy.sh e2e
```

Aurora master credentials are managed by AWS Secrets Manager. The script
synchronizes the current connection URL into the `gpu-fault-aurora`
Kubernetes Secret. Configure External Secrets or Secrets Store CSI before
enabling automatic password rotation.
