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
- Out of scope: provider-side node replacement (the adapter refuses `REPLACE` and no EC2 instance is replaced;
  `nodeReplace` is OWNed by `gpu-fault-eks-adapter` only as the warm-spare handover of §2.4), HyperPod
  health-monitoring-agent capabilities, installing DCGM host tools on nodes.

## 2. Design

### 2.1 Shared contract `gpu_fault.cluster_scope`
`ClusterProvider` (`hyperpod`|`eks-managed`), `GPU_FAULT_CLUSTER_PROVIDER`/`GPU_FAULT_NODE_SELECTOR`/
`GPU_FAULT_EKS_CLUSTER_NAME`, connection Secret keys `cluster-provider`/`node-selector`/`eks-cluster-name`,
`node_selector(target)` (the entry's recorded selector; a HyperPod entry from before the convention defaults to
`sagemaker.amazonaws.com/cluster-name=<name>`), the ownership label convention `OWNERSHIP_LABEL=gpu-fault.io/control-plane`,
`ownership_node_selector` (EKS `gpu-fault.io/control-plane=<site id>`; HyperPod adds the cluster-name label as a
cross-check), `NodeSelectorSource` (`label-convention`/`provider-default-legacy`/`flag-legacy`) with the Secret key
`node-selector-source`, `selector_matches`, `PROVIDER_NODE_REBOOT_OWNERS`
(`gpu-fault-hyperpod-adapter` / `gpu-fault-eks-adapter`), `PROVIDER_PROFILE_SOURCES`/`_VERSIONS`
(`config/runtime-profile.regional-eks-safe.example.yaml` / `eks-v1`).

### 2.2 Discovery and site
Discovery decides the provider from the ARN (an EKS ARN with no HyperPod orchestrator and an `ACTIVE`
`describe-cluster` is `eks-managed`); the node scope is resolved from the ownership label on the node groups
(`admin/bootstrap_ownership.py`: the managed node groups / HyperPod instance groups whose `KubernetesConfig.CurrentLabels`
carry `gpu-fault.io/control-plane=<site id>`; no label, another site's label or an instance type outside the GPU
inventory refuses with the evidence; an already-managed cluster keeps its recorded selector and
`gpu-fault-admin node-scope --converge` is the journaled convergence step); the site entry records `provider`,
`nodeSelector` and `nodeSelectorSource` and no `hyperpodClusterName`; Agent endpoint CIDRs come from the GPU node group
subnets; the counterpart of the `NodeRecovery`
gate is that the GPU managed node groups' EKS node auto repair must be off (discovery reads `describe-nodegroup` for every
node group hosting GPU nodes, refuses `nodeRepairConfig.enabled=true` and records `node_repair` on the identity);
node names are FQDNs and installer Job names are DNS-1123 truncated.

### 2.3 Per-cluster Runtime Profile (mixed sites)
The site profile is the default pin. When its `nodeReboot` owner cannot serve the joining provider, the join records
`spec.clusters[].runtimeProfile` (`source`/`templateSource`/`version`/`agentConfigDigest`); the new version is
reviewed through `release-deploy/profile-plan.json` (`change_kind: CLUSTER_PROFILE_NEW`) and approved with
`join-cluster --approve-profile-plan <sha> --reference <CHG>`; the consumed approval writes the immutable snapshot
`profiles/<version>.yaml`, reused by later clusters of that provider. A batch join (several `--gpu-cluster-arn`,
`deploy --gpu-cluster-arn`) is reviewed like a single join; a version is one plan naming every cluster that needs it,
and the release compares a registered version without its registration anchor (`cluster_id`), see
[Runtime Profile Change Approval](../administrator-profile-change-approval.md) §1.2. The release registers every version in use and
renders each cluster's Executor/Watcher/Reconciler/installer pins and Agent config digest from its own version; the
registry entry carries `runtime_profile_version`/`agent_config_digest` (digest-excluded while equal to the site
default); `FleetRegistry.cluster_policy` checks Agents per cluster while `GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION`
keeps meaning the site default; a first `deploy` still requires one provider.

### 2.4 Executor and node lifecycle
An `eks-managed` Executor builds `gpu-fault-eks-adapter`: `nodeReboot` calls `RebootInstances` on the bound EC2
instance (IAM: `ec2:DescribeInstances`, `ec2:RebootInstances` conditioned on `aws:ResourceTag/eks:cluster-name`,
ARN-scoped `eks:DescribeCluster/DescribeNodegroup`, no `sagemaker:*`), confirmed by instance running + a new Agent
incarnation/boot_id; a departed node is proven through `ec2 describe-instances`; failure-domain labels add
`eks.amazonaws.com/nodegroup` after the instance-group label. `nodeReplace` is OWNed by the same `gpu-fault-eks-adapter`
(`config/runtime-profile.regional-eks-safe.example.yaml`, observed `warm-spare-v1`) as a Kubernetes-level warm-spare
handover: the Executor builds the spare coordinator on both providers, and `REPLACE_NODE` (`HEALTHY_WARM_SPARE_ONLY`)
moves the workload to a spare declared with `gpu-fault-admin config spare --declare` (topology: node group + instance
type); `REPLACE` is never in the adapter allowlist and `submit` refuses it, so without a declared spare the step fails
closed. A cluster still pinned to an older `eks-v1` snapshot whose `nodeReplace` is OBSERVE keeps that claim until
`deploy` plans the successor `regional-eks-<digest12>` (`CLUSTER_PROFILE_NEW`) and the plan is approved.

### 2.5 Data plane and nodes
- DCGM exporter: the HyperPod AMI runs DCGM as a host service and the DaemonSet dials `127.0.0.1:5555`; the AL2023
  NVIDIA AMI has no nv-hostengine/dcgmi, so the `eks-managed` render drops `DCGM_REMOTE_HOSTENGINE_INFO` and runs the
  embedded hostengine (`NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=utility`, no SYS_ADMIN, no
  `nvidia.com/gpu` request); hostNetwork, the loopback 9400 bind and instance-type affinity are unchanged; the
  tolerations become explicit `nvidia.com/gpu` and `gpu-fault.io/quarantined` (not the quiesce taint of §2.7; the
  HyperPod manifest's wildcard toleration is byte for byte unchanged); this is the (default) embedded mode, and the plan
  adds `dcgm_eks_managed`.
- DCGM host mode: a cluster whose GPU node groups carry `gpu-fault.io/dcgm-hostengine=host` (its only value, §3 item 7)
  runs in host mode. Every deploy/join evaluates the node groups the selector selects -- all `host` is host (source
  `node-group-label`), none is embedded (source `default`), a mix or any other value is refused before any change,
  naming every group as `<node group> gpu-fault.io/dcgm-hostengine=<value|absent>`; Karpenter NodePools are not read, and
  HyperPod is always host (source `provider`, never recorded). The eks entry of `site.yaml` records
  `dcgmHostEngine`/`dcgmHostEngineSource`, rewritten from the node groups by every deploy/join (a node-image capability,
  not kept the way `nodeSelector` is). The host variant of the exporter keeps `DCGM_REMOTE_HOSTENGINE_INFO=127.0.0.1:5555`
  with the annotation `gpu-fault.io/dcgm-hostengine: host`, the explicit tolerations and the rest of the eks render are
  unchanged, and the plan shows it as `dcgm_eks_managed_host`; the embedded variant is byte for byte unchanged. A mode
  change is the release-diff input `dcgm_hostengines`: it plans DCGM + reconciler + Agent (`DATA_PLANE_COMPATIBLE`), the
  exporter is re-applied and the cluster's Agents are re-installed in waves (reconciler node annotation
  `gpu-fault.io/installer-dcgm-hostengine=host`), and the node preflight checks the host hostengine before any change.
  `node_scope.dcgm_hostengine`/`dcgm_hostengine_source`/`dcgm_hostengine_pending_nodes` of `status --full` and the
  `dcgm_hostengine` of `node-scope` show the mode, its source and the nodes still to re-install. A deploy of an unchanged
  source re-reads the label with the same discovery reads before its NOOP shortcut and takes the application release when
  the declaration differs from the record (a mixed one is refused before any change); `status` (quick and `--full`) and
  `node-scope` show a declaration not yet released as
  `next deploy: DATA_PLANE_COMPATIBLE (dcgm_hostengine: <cluster> declared host, recorded embedded)`. An exporter or
  reconciler whose live mode (the DaemonSet annotation/environment, the reconciler's `GPU_FAULT_DCGM_HOSTENGINE`) differs
  from the record is the release-diff input `dcgm_hostengine_drift`, planned like a flip (DCGM + reconciler + Agent), and
  the `gpu_cluster:<id>` check of `status --full`/verify fails on it (`node_scope.dcgm_hostengine_served`); after a
  refused and rolled-back release the rollback alignment restores the record to the mode that runs and `sync-state`
  records the digest of the mode the exporters run, and a node preflight refusal names every node with its whole
  reasons and records every node's DCGM state.
- The node collector scrapes DCGM metrics from the exporter over HTTP, never dcgmi; in embedded mode the node agent's
  quick DCGM diagnostic (`dcgmi diag -r 1 -j`) reports a missing dcgmi or an unreachable hostengine as a configuration
  `WARN` (`DCGM_TOOL_UNAVAILABLE`/`DCGM_HOSTENGINE_UNAVAILABLE`, never drained); in host mode (Agent environment
  `GPU_FAULT_DCGM_HOSTENGINE=host`, written on eks-managed nodes only and not part of the config digest) it returns
  real verdicts, and a missing dcgmi or an unreachable host hostengine fails as on HyperPod with `DCGM_EXECUTION_REVIEW`
  (`DRAIN_AND_QUARANTINE`).
- Data-plane Deployments (executor, watcher, reconciler, node-resource collector, ADOT) carry no node pin and tolerate
  no GPU taint; they land on CPU nodes.
- The node preflight Job checks the host prerequisites inside `chroot /host` and reports every missing one at once
  (`ERROR: host prerequisites unavailable: ...` plus a JSON verdict); AL2023 needs `python3.12 strace` from the launch
  template (administrator-operations §5.0.1); host mode additionally needs `dcgmi`, `nv-hostengine`, an active
  `nvidia-dcgm` and a successful `timeout 30 dcgmi discovery -l`, listed in the same verdict. The preflight records every
  node's DCGM state (`not-installed`/`installed-inactive`/`host-active`/`cuda-mismatch`/`host-failed`/`unmanaged-hostengine`)
  in the verdict's `dcgm` object: host mode accepts `host-active` only, and embedded mode refuses a host hostengine next
  to the embedded one (running, active/activating or enabled; `ERROR: DCGM host engine conflict: ...`); a HyperPod node is
  always checked as host mode (`dcgmi`, `nv-hostengine`, an active `nvidia-dcgm` and `dcgmi discovery -l` refuse, a CUDA
  flavor mismatch or a foreign nv-hostengine only warns); the per-node
  states of the last preflight are recorded in the release state `node_dcgm_hostengine` and shown by `status --full` as
  `node_scope.dcgm_hostengine_nodes`/`dcgm_hostengine_conflict_nodes`. The Agent always grades by the cluster's declared
  mode.
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

### 2.7 The quiesce taint (eks-managed only)
- A plain EKS quiesce leaves kubelet running, so the GPU client containers the Agent stops (the product's own exporter,
  the device plugin, GFD) are back within seconds, recreated by their DaemonSets, and the `VERIFY_NO_GPU_CLIENTS` before
  the reset never sees the GPU free (live, 2026-10-05). The Executor taints the node
  `gpu-fault.io/quiesce=<incident value>:NoExecute` before it dispatches QUIESCE_GPU_SERVICES
  (`adapters/node_action/quiesce_taint.py`; if the taint cannot be applied the step FAILS and no Agent command is sent)
  and removes it before it dispatches RESTORE_GPU_SERVICES (if the removal fails the host services are still restored
  and the step FAILS naming the error); `cluster_executor/bootstrap.py` wires the Kubernetes core client into the Node
  Action adapter only under `GPU_FAULT_CLUSTER_PROVIDER=eks-managed`, so HyperPod is unchanged.
- RESTORE_SCHEDULING (the validated restore) strips a leftover quiesce taint together with the cordon and the
  quarantine taint; the node evidence of `workflow-reconcile`, the orphaned-annotation cleanup and `config spare`
  treat it as isolation still in place.
- Every stage of the taint's life has an exit: each QUIESCE outcome records `quiesce_taint_nodes` and a FAILED quiesce
  untaints at once; a QUIESCE that tainted but never completed owes the RESTORE_GPU_SERVICES compensation too; the
  successor's MARK_UNSCHEDULABLE strips an orphaned taint; RESTART_GPU_DEVICE_PLUGIN WAITs under a taint somebody still
  holds and names it, stripping an orphaned one first; a starved VALIDATE_GPU writes `quiesce_taint_present`/
  `quiesce_taint_holders` into its pending details; `--disposition restore` runs RESTORE_SCHEDULING alone for a node whose
  only isolation of the incident is its quiesce taint (`quiesce_taint_only`). See administrator operations §5.0 and §9.2.
- An in-place plan replacement of a queued successor no longer bumps the fencing token (it shares its RUNNING
  predecessor's generation), a RUNNING reset chain absorbs the EFA/GPU allocatable dips its own taint causes, a step the
  executor refused before dispatch climbs neither ladder, and the takeover of a predecessor's isolation follows the whole
  escalation lineage (`escalation_descends_from`; `escalation_lineage` of
  `/executors/incident-ownership?successor_workflow_id=`) instead of one hop.
- A busy reset with no process holding a device node unloads the idle `nvidia_drm`/`nvidia_modeset`, retries at once
  and reloads them after the reset (`released_kernel_modules`/`reloaded_kernel_modules`; the GPUs are cold after the
  unload, so the read-only `nvidia-smi` queries and the reload's `modprobe` are bounded by the node's GPU count, 55 s and
  52 s on eight GPUs; `/proc/modules` and `/sys/class/drm` check the reload -- every GPU's DRM device must come back --,
  a missing module earns one retry after a 5 s pause, a partial load one `rmmod nvidia_drm && modprobe nvidia_drm`, and a
  stack still not back is `display_stack_restored=false`, the GPUs without a DRM device in
  `display_drm_devices_missing`: the reset stays successful, it is not handled as a GPU fault, and the GPU reset
  completion mail carries `[display stack not restored]` and the remedy `modprobe nvidia_drm` or a node reboot); the
  `nvidia-persistenced` the
  window kept (on the other GPUs only) and the Fabric Manager the window kept (named in its `kept_services`, on
  `/dev/nvidia-caps/*`/`/dev/nvidia-nvswitch*`/`/dev/nvidia-nvlink` only) do not stop the unload and are recorded in
  `display_release_bystanders` (Fabric Manager itself only: on NVL5 an `nvlsm` of the same unit holding an NVIDIA device
  node is still a client, and the quiesce never signals it either); QUIESCE details carry
  `stopped_services`/`inactive_services`/`protected_services`.
- DaemonSet Pods that do not tolerate the taint are evicted for the window and return after it; third-party GPU
  clients with a key-less wildcard toleration (`operator: Exists`) stay inside the window and the reset still answers
  `In use by another client` -- see §3 item 6.

## 3. Operator prerequisites (plain EKS cluster)

1. Node group: `AL2023_x86_64_NVIDIA` (or equivalent) AMI; launch template user data installing `python3.12 strace`
   (administrator-operations §5.0.1); the node group's EKS node auto repair off (`nodeRepairConfig.enabled=false` or
   unset, administrator-operations §5.0.1); the node group's `updateConfig.updateStrategy` `MINIMAL` (a `DEFAULT` roll
   launches one more GPU instance before it terminates one; administrator-operations §5.0.1); NVIDIA and EFA device
   plugins running; a GPU taint is fine.
   For `AL2023_x86_64_NVIDIA` node groups `gpu-fault-admin prepare-eks-nodegroup` automates this item's user data, node
   auto repair and update strategy and item 5's ownership label (a read-only plan by default, `--apply` runs it;
   administrator-operations §5.0.1a).
2. Namespaces: every workload namespace in `allowedNamespaces` (the site's default list, e.g. `training`) must exist,
   or pass `--allowed-namespace` with namespaces that do.
3. Network: the GPU VPC's NAT EIP is whitelisted on the NLB, the private Route53 zone associated, Agent CIDRs taken
   from the node group subnets -- done by the join, which needs the corresponding AWS permissions.
4. Approval: the first plain EKS cluster joining a HyperPod site needs `--approve-profile-plan <sha> --reference <CHG>`
   (eks-v1).
5. Ownership label: `gpu-fault-admin site-id --cpu-cluster-arn <CPU ARN>` prints the site id;
   `aws eks update-nodegroup-config --cluster-name <eks> --nodegroup-name <ng> --labels addOrUpdateLabels={gpu-fault.io/control-plane=<site id>}`
   labels the GPU node group (HyperPod: `update-cluster` with `KubernetesConfig.Labels`; Karpenter: NodePool template
   labels). Command: `gpu-fault-admin join-cluster --state-dir <dir> --gpu-cluster-arn <eks arn>` (ARNs only).
   De-registration releases it: the last step of `remove-cluster`/`uninstall`, `OWNERSHIP_RELEASED`, removes the key
   from the node groups the site selected with
   `aws eks update-nodegroup-config --cluster-name <eks> --nodegroup-name <ng> --labels removeLabels=gpu-fault.io/control-plane`
   (HyperPod: `update-cluster` with the instance group's full definition minus the key and the running image named,
   refused while an update or operation is in flight or a change is pending; a Karpenter NodePool is never edited and
   recorded `manual`) and waits, bounded, until `describe-nodegroup` shows it gone; the
   operator identity additionally needs `eks:UpdateNodegroupConfig`/`sagemaker:UpdateCluster` (the runtime roles stay
   read-only); `--keep-ownership-label` keeps it and records `retained`. A de-registered cluster whose node groups still
   carry the label is a stale claim `status`/`node-scope` report as a finding (the same cluster joined again is a
   member and is not reported); re-label with the commands `site-id`
   prints before joining it again (administrator-operations §5.0.0, the "release" row).
6. Third-party GPU client DaemonSets (the customer's own dcgm-exporter, GPU feature discovery, monitoring agents)
   **must not tolerate** `gpu-fault.io/quiesce` (no key-less `operator: Exists` wildcard toleration; list the keys such
   as `nvidia.com/gpu` instead) and must tolerate being stopped/evicted inside a quiesce window; the default NVIDIA
   device plugin and GFD manifests tolerate only `nvidia.com/gpu` and need no change.
7. Host DCGM (optional, real `dcgmi diag` verdicts): the launch template installs a pinned
   `datacenter-gpu-manager-4-core-<ver>` and `datacenter-gpu-manager-4-cuda<CUDA major>-<ver>` (the major of the
   "CUDA Version" `nvidia-smi` prints), writes the `Restart=on-failure` drop-in
   `nvidia-dcgm.service.d/90-gpu-fault-restart.conf` (the package unit's `Restart=on-abort` leaves a non-zero exit or a
   crash loop down for good) and runs `systemctl enable --now nvidia-dcgm`; confirm on a node with
   `dcgmi discovery -l` and `dcgmi diag -r 1`, label **every** GPU node group of the cluster
   `gpu-fault.io/dcgm-hostengine=host` and deploy/join right away (also over an unchanged source: the deploy re-reads the
   label, and `status` shows the pending `next deploy` before it; a deploy with the service enabled but no label is
   refused by the embedded conflict check); to leave, remove the label, take `nvidia-dcgm` out of the launch template and
   run `systemctl disable --now nvidia-dcgm` on the nodes, then deploy at once (order and commands in
   administrator-operations §5.0.1, "Host DCGM").
8. Automatic node updates: managed node groups have no scheduled update, but a Karpenter NodePool's `expireAfter`
   (default 720h), drift and consolidation, EKS Auto Mode node pools, an ASG `MaxInstanceLifetime` and automation
   calling `UpdateNodegroupVersion` on a schedule all replace GPU nodes: give the NodePools `expireAfter: Never` and a
   `nodes: "0"` budget, keep `MaxInstanceLifetime` unset and move the automation into the maintenance window; the
   Capacity Rebalance EKS enables on Spot node groups is handled by the replacement-node join window. `status
   --full`/`node-scope` record and announce them (never refuse); the operator identity needs
   `autoscaling:DescribeAutoScalingGroups` (administrator-operations §5.0.5).

## 4. Limitations

- No provider-side node replacement: `nodeReplace` is only the warm-spare handover and needs a declared spare (§2.4);
  no HyperPod health agent, so `gpuDetection`/`providerHealthIsolation` are not claimed.
- Embedded mode (the default, no node group declares host DCGM): the node has no host hostengine, the quick diagnostic
  `dcgmi diag` only records a configuration WARN and the diagnostic bundle records the missing tool; real verdicts need
  host mode (§3 item 7).
- One profile version per cluster; site profile changes stay site-level approvals; the override version is the example
  profile's version name.
- A first `deploy` refuses mixed providers (deploy one provider, join the other).
- The embedded hostengine relies on the AMI's default NVIDIA container runtime; a cluster without `nvidia` as the
  default runtime would need `runtimeClassName` on the DaemonSet (not needed on this cluster).
- The quiesce taint evicts only Pods that do not tolerate it: DaemonSets with a wildcard toleration, static Pods and
  non-Kubernetes processes are untouched, and a reset may still answer `In use by another client` because of them (the
  step's `details.gpu_client_holders` names the holders).
- Multi-cluster sites (Administrator Operations §5.0.4): node names must be unique across the site, and plain EKS node
  names derive from the private IPv4 address, so two plain EKS clusters in overlapping VPC CIDRs can collide (HyperPod
  node names cannot); a HyperPod cluster and a plain EKS cluster must not share a name (cluster id and IAM role names
  derive from it, and `--cluster-id` is not kept by the next `deploy`); with many clusters a release has one canary, the
  rollback is serial per cluster, and both share the fixed 6-hour deadline and 2-hour recovery reserve.
- When `remove-cluster` removes the registration cluster (`spec.runtimeProfile.registrationClusterId`) while other
  members remain, a remaining member of the same provider becomes the registration cluster; without one the removal is
  refused, so a removal never changes the base family.

## 5. Validation status (honest)

- Verified live, once: discovery, profile approval, IAM/network/node-key prerequisites, per-cluster profile
  registration and rendering, DCGM embedded-hostengine rollout, DNS/TLS gate, data-plane Deployments on CPU nodes,
  node preflight and installer Jobs (2 nodes), collector readiness, VERIFIED, registry update, site commit,
  activation; several failed attempts rolled back and reran (including after site upgrades).
- Unit/contract: `tests/*eks_managed*`, `tests/fleet/test_cluster_profile_pins.py`,
  `tests/eks_provider/test_mixed_site_owner_resolution.py`, `tests/deploy/test_node_preflight_host_prerequisites.py`
  and others; HyperPod neutrality is pinned by digest constants and byte-identical render assertions.
- **Not validated**: DCGM host mode (§2.5) has unit/contract tests only and is still to be proven live; the quiesce taint (§2.7) and the restore exit for a RECOVERED incident have unit tests only; the third live DESTR-001 round is still to run; the acceptance campaign on a mixed site (fault injection, real `nodeReboot` execution and
  confirmation, departed nodes, token rotation, the live `remove-cluster`/`uninstall` paths for eks-managed, Agent
  upgrade/rollback, long-run stability); more than one plain EKS cluster (a batch join of several plain EKS clusters has
  unit tests only); non-AL2023 AMIs; workload namespaces other
  than `training`; clusters without the NVIDIA runtime as default.
- p6-b200/p6-b300 (HGX B200/B300) are graded by the same rules as Hopper SXM (the fabric state window, VALIDATE_GPU's
  fabric and unit checks, the observe-only default of a solo XID 45), but no such node is on the site: not observed live.

## 6. Commits (base 0d151512 -> 3dedcd07, excerpt)

Contract and runtime: `a36f600f` `b84a033b` `d1244a34` `5fcf6fe6` `3b28ab2e` `2cd0135d`; admin/release/deploy:
`d3acba29` `0b6e3823` `daab59e1` `fc3ad683`; per-cluster profile: `b9dbf3de` `df71aefe` `114ab02e` `0ced160e`
`2beb24e0` `9c1ec98e`; live-trial fixes: `3017aba4` (node-key inventory), `8ca598db` (registration probe/rollback
targets), `f7dfb5fa` (rollback journal re-plan), `5a06f7b9` (DCGM embedded hostengine), `457829f3` (allowed
namespaces), `3dedcd07` (node preflight verdict); deploy-host gates: `3ce79c81` `89153747` `29cdddd1` `526c0a73`.
