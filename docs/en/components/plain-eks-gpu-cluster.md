# Plain EKS GPU Clusters (`eks-managed`) and Mixed-Provider Sites

English edition of `docs/components/plain-eks-gpu-cluster.md`; the Chinese file remains the source of record until both are maintained together.

Status: implemented and joined once on a real cluster (2026-10-04, `gpu-eks-cluster`, 2× p5en.48xlarge,
AL2023_x86_64_NVIDIA AMI, joining an existing `hyperpod-v1` HyperPod site). **No acceptance campaign has run on a mixed
site yet**; section 5 lists what one join cannot prove. This is the summary of branch `feature/plain-eks-gpu-cluster`
(base `0d151512`, 36 commits, 184 files) for the later PR description.

## 1. Goal and boundary

- Let a **plain EKS cluster without a HyperPod orchestrator** (managed node groups) be a site's GPU cluster:
  `join-cluster`/`deploy`/`remove-cluster`, the release engine, the data-plane manifests, the node installer and the
  control plane's fleet pin all branch on the provider.
- One site may manage HyperPod and plain EKS clusters together (mixed providers), each cluster on its own Runtime
  Profile version.
- Everything a HyperPod cluster renders (selectors, Secret keys, registry digests, manifests, installer environment,
  CPU pins) stays byte-identical, pinned by tests (`tests/*eks_managed*`, the digest constants in
  `tests/regional/test_regional_token_rotation.py`).
- Out of scope: provider-side node replacement (`nodeReplace` stays OBSERVE), warm spares, HyperPod
  health-monitoring-agent capabilities, installing DCGM host tools on nodes.

## 2. Design

### 2.1 Shared contract `gpu_fault.cluster_scope`
`ClusterProvider` (`hyperpod`|`eks-managed`), `GPU_FAULT_CLUSTER_PROVIDER`/`GPU_FAULT_NODE_SELECTOR`/
`GPU_FAULT_EKS_CLUSTER_NAME`, connection Secret keys `cluster-provider`/`node-selector`/`eks-cluster-name`,
`node_selector(target)` (HyperPod `sagemaker.amazonaws.com/cluster-name=<name>`; EKS `nvidia.com/gpu.present=true` or
`eks.amazonaws.com/nodegroup=<name>` from `--gpu-nodegroup`), `selector_matches`, `PROVIDER_NODE_REBOOT_OWNERS`
(`gpu-fault-hyperpod-adapter` / `gpu-fault-eks-adapter`), `PROVIDER_PROFILE_SOURCES`/`_VERSIONS`
(`config/runtime-profile.regional-eks-safe.example.yaml` / `eks-v1`).

### 2.2 Discovery and site
Discovery decides the provider from the ARN (an EKS ARN with no HyperPod orchestrator and an `ACTIVE`
`describe-cluster` is `eks-managed`); the site entry records `provider` and `nodeSelector` and no
`hyperpodClusterName`; Agent endpoint CIDRs come from the GPU node group subnets; there is no `NodeRecovery` gate;
node names are FQDNs and installer Job names are DNS-1123 truncated.

### 2.3 Per-cluster Runtime Profile (mixed sites)
The site profile is the default pin. When its `nodeReboot` owner cannot serve the joining provider, the join records
`spec.clusters[].runtimeProfile` (`source`/`templateSource`/`version`/`agentConfigDigest`); the new version is
reviewed through `release-deploy/profile-plan.json` (`change_kind: CLUSTER_PROFILE_NEW`) and approved with
`join-cluster --approve-profile-plan <sha> --reference <CHG>`; the consumed approval writes the immutable snapshot
`profiles/<version>.yaml`, reused by later clusters of that provider. The release registers every version in use and
renders each cluster's Executor/Watcher/Reconciler/installer pins and Agent config digest from its own version; the
registry entry carries `runtime_profile_version`/`agent_config_digest` (digest-excluded while equal to the site
default); `FleetRegistry.cluster_policy` checks Agents per cluster while `GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION`
keeps meaning the site default; a first `deploy` still requires one provider.

### 2.4 Executor and node lifecycle
An `eks-managed` Executor builds `gpu-fault-eks-adapter`: `nodeReboot` calls `RebootInstances` on the bound EC2
instance (IAM: `ec2:DescribeInstances`, `ec2:RebootInstances` conditioned on `aws:ResourceTag/eks:cluster-name`,
ARN-scoped `eks:DescribeCluster/DescribeNodegroup`, no `sagemaker:*`), confirmed by instance running + a new Agent
incarnation/boot_id; a departed node is proven through `ec2 describe-instances`; failure-domain labels add
`eks.amazonaws.com/nodegroup` after the instance-group label.

### 2.5 Data plane and nodes
- DCGM exporter: the HyperPod AMI runs DCGM as a host service and the DaemonSet dials `127.0.0.1:5555`; the AL2023
  NVIDIA AMI has no nv-hostengine/dcgmi, so the `eks-managed` render drops `DCGM_REMOTE_HOSTENGINE_INFO` and runs the
  embedded hostengine (`NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=utility`, no SYS_ADMIN, no
  `nvidia.com/gpu` request); hostNetwork, the loopback 9400 bind, instance-type affinity and the wildcard toleration
  are unchanged; a mixed site's plan adds `dcgm_eks_managed`.
- The node collector scrapes DCGM metrics from the exporter over HTTP, never dcgmi; the node agent's quick DCGM
  diagnostic reports `WARN`/`DCGM_TOOL_UNAVAILABLE` on nodes without dcgmi.
- Data-plane Deployments (executor, watcher, reconciler, node-resource collector, ADOT) carry no node pin and tolerate
  no GPU taint; they land on CPU nodes.
- The node preflight Job checks the host prerequisites inside `chroot /host` and reports every missing one at once
  (`ERROR: host prerequisites unavailable: ...` plus a JSON verdict); AL2023 needs `python3.12 strace` from the launch
  template (administrator-operations §5.0.1).
- Node Action keys: every node inventory read, including the CPU mirror guard, uses the cluster selector.

### 2.6 Join recoverability and diagnostics (product gaps fixed on this branch)
- `allowedNamespaces` defaults to the list the site's clusters share and no longer forces `training`; the join checks
  every workload namespace exists on its first read of the GPU cluster.
- A failed sensitive command keeps the script's own `ERROR:` lines (else the last stderr line, redacted) in both the
  admin and the release runner.
- The Runtime Profile registration probe names its failure (HTTP status / 30 s no answer) and is retried once.
- Rollback: a cluster that never received a node runtime is an empty, proven cleanup target set (SKIPPED); a cleanup
  journal that never left PREFLIGHT is archived and re-planned after a site upgrade; `ROLLBACK_FAILED` recovers by
  rerunning the same `join-cluster`, which then starts a fresh attempt.
- Deploy-host gates: the test-impact selection and the role-split render test run inside the deploy's
  deadline/API-budget scope without inheriting it.

## 3. Operator prerequisites (plain EKS cluster)

1. Node group: `AL2023_x86_64_NVIDIA` (or equivalent) AMI; launch template user data installing `python3.12 strace`
   (administrator-operations §5.0.1); NVIDIA and EFA device plugins running; a GPU taint is fine.
2. Namespaces: every workload namespace in `allowedNamespaces` (the site's default list, e.g. `training`) must exist,
   or pass `--allowed-namespace` with namespaces that do.
3. Network: the GPU VPC's NAT EIP is whitelisted on the NLB, the private Route53 zone associated, Agent CIDRs taken
   from the node group subnets -- done by the join, which needs the corresponding AWS permissions.
4. Approval: the first plain EKS cluster joining a HyperPod site needs `--approve-profile-plan <sha> --reference <CHG>`
   (eks-v1).
5. Command: `gpu-fault-admin join-cluster --state-dir <dir> --gpu-cluster-arn <eks arn> --gpu-nodegroup <ng>`.

## 4. Limitations

- `nodeReplace` is OBSERVE only; no HyperPod health agent, so `gpuDetection`/`providerHealthIsolation` are not claimed.
- No DCGM host tools on the node: `dcgmi diag` is unavailable (WARN); the diagnostic bundle records the missing tool.
- One profile version per cluster; site profile changes stay site-level approvals; the override version is the example
  profile's version name.
- A first `deploy` refuses mixed providers (deploy one provider, join the other).
- The embedded hostengine relies on the AMI's default NVIDIA container runtime; a cluster without `nvidia` as the
  default runtime would need `runtimeClassName` on the DaemonSet (not needed on this cluster).

## 5. Validation status (honest)

- Verified live, once: discovery, profile approval, IAM/network/node-key prerequisites, per-cluster profile
  registration and rendering, DCGM embedded-hostengine rollout, DNS/TLS gate, data-plane Deployments on CPU nodes,
  node preflight and installer Jobs (2 nodes), collector readiness, VERIFIED, registry update, site commit,
  activation; several failed attempts rolled back and reran (including after site upgrades).
- Unit/contract: `tests/*eks_managed*`, `tests/fleet/test_cluster_profile_pins.py`,
  `tests/eks_provider/test_mixed_site_owner_resolution.py`, `tests/deploy/test_node_preflight_host_prerequisites.py`
  and others; HyperPod neutrality is pinned by digest constants and byte-identical render assertions.
- **Not validated**: the acceptance campaign on a mixed site (fault injection, real `nodeReboot` execution and
  confirmation, departed nodes, token rotation, the live `remove-cluster`/`uninstall` paths for eks-managed, Agent
  upgrade/rollback, long-run stability); more than one plain EKS cluster; non-AL2023 AMIs; workload namespaces other
  than `training`; clusters without the NVIDIA runtime as default.

## 6. Commits (base 0d151512 -> 3dedcd07, excerpt)

Contract and runtime: `a36f600f` `b84a033b` `d1244a34` `5fcf6fe6` `3b28ab2e` `2cd0135d`; admin/release/deploy:
`d3acba29` `0b6e3823` `daab59e1` `fc3ad683`; per-cluster profile: `b9dbf3de` `df71aefe` `114ab02e` `0ced160e`
`2beb24e0` `9c1ec98e`; live-trial fixes: `3017aba4` (node-key inventory), `8ca598db` (registration probe/rollback
targets), `f7dfb5fa` (rollback journal re-plan), `5a06f7b9` (DCGM embedded hostengine), `457829f3` (allowed
namespaces), `3dedcd07` (node preflight verdict); deploy-host gates: `3ce79c81` `89153747` `29cdddd1` `526c0a73`.
