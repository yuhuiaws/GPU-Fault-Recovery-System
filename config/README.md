# Operator-editable configuration examples

Files under `config/` are examples that an operator copies or renders for a
specific deployment. Every file matching
`runtime-profile.*.example.yaml` is one valid `RuntimeProfile` document and
can be POSTed to `/v1/runtime-profiles` after replacing its site-specific
fields.

`site.example.yaml` is the single administrator input for
`gpu-fault-admin preflight/deploy/verify/status/join-cluster/remove-cluster/uninstall`.
`runtimeProfile.templateSource` is the one editable policy template.
`make release-deploy` renders an immutable content-addressed profile snapshot,
updates `runtimeProfile.source/version`, and generates the regional release JSON
in a private temporary directory; operators do not maintain these derived files.
`spec.notifications.adminEmail`, `emailSender`, `emailRecipients` and
`emailSubjectPrefix` are filled by ARN-only deploy from the one `--admin-email`
address: it is both the SES sender and the recipient, and the prefix is empty.
For an existing site, `make release-deploy SITE=... ADMIN_EMAIL=...` updates the
declarative value; omitting `ADMIN_EMAIL` preserves the site value or discovers
the AWS account email when a legacy site has none.

Each `spec.clusters[]` entry names its `provider`: `hyperpod` (the default and
the fully validated form; `hyperpodClusterName` is required and the GPU nodes are
the ones labelled `sagemaker.amazonaws.com/cluster-name=<name>`) or `eks-managed`
(a plain EKS cluster with managed node groups and no HyperPod orchestrator; a
newer, less validated form in which provider-side node replacement is
observe-only). An `eks-managed` entry omits `hyperpodClusterName` and must carry
`nodeSelector`, the label selector of its GPU nodes (`gpu-fault-admin
join-cluster --node-selector`, default `nvidia.com/gpu.present=true`;
`--gpu-nodegroup <name>` derives `eks.amazonaws.com/nodegroup=<name>`). One
helper, `gpu_fault.cluster_scope.node_selector`, turns the entry into the
selector every component uses, so a HyperPod entry still renders the historical
label text byte for byte. `spec.failureDomainLabels` left unset follows the
providers: HyperPod-only sites keep the instance-group/zone priority, a site with
an `eks-managed` cluster also reads `eks.amazonaws.com/nodegroup`. The site runtime
profile (`spec.runtimeProfile`) is the default pin for every GPU cluster; a cluster whose
provider that profile cannot reboot carries its own `spec.clusters[].runtimeProfile`
(`source`, `templateSource`, `version`, `agentConfigDigest`), which deploy/join-cluster
fill in from the provider's example profile (`eks-v1` for a plain EKS cluster in a
HyperPod site) after the profile-plan approval. Every distinct version is registered on
the control plane; the cluster's registration carries its version and Agent config
digest so the fleet pin is checked per cluster, while clusters without an override keep
the site pins and render exactly as before.

Successful deployment also registers AWS resource IDs, ownership and delete
policy in Aurora. Operators do not add those fields to `site.yaml`. Uninstall
uses the generated registry and always preserves GPU EKS/HyperPod clusters;
`--cpu-cluster keep|delete` controls only the CPU control-plane cluster.
Automatic deployment does not share solution-specific resources. A matching
resource is adopted only when its `gpu-fault:site-id` ownership proves it is a
leftover from the same site; foreign name collisions fail closed.

| Deployment path | Profile terminology | Runtime profile | Required changes | Registration |
|---|---|---|---|---|
| Regional CPU control plane with GPU-cluster Executor | `regional-hyperpod-safe` means the gated regional production profile | `runtime-profile.regional-hyperpod-safe.example.yaml` | registration `cluster_id`; diagnostic bundle, GPU reset and fabric recovery are OWN after signed Node Agent/fleet convergence | Register once per profile version on the regional control plane |
| Regional CPU control plane with a plain EKS (`eks-managed`) GPU cluster Executor | `regional-eks-safe` is the same gated profile for a cluster without HyperPod: `nodeReboot` OWN by `gpu-fault-eks-adapter` (EC2 RebootInstances), `nodeReplace` OBSERVE, no `hyperpod-hma` claims | `runtime-profile.regional-eks-safe.example.yaml` (profile version `eks-v1`, environment `eks`) | registration `cluster_id`; the executor runs with `GPU_FAULT_CLUSTER_PROVIDER=eks-managed` | Register once per profile version on the regional control plane |

The API does not accept a `profiles:` wrapper or multiple profiles in one
request. Historical single-cluster and multi-environment capability examples
are retained only in the private archive because those deployment modes are
not public release targets.

Runtime profiles are stored by `profile_version` and shared by every registered
GPU cluster using that version. The payload `cluster_id` is an explicit
registration anchor and must name one cluster present when a new profile is
first registered; it does not make a second copy for that cluster. After
registration the immutable profile can outlive that cluster and is not
recomputed when GPU clusters join or leave. To change capability content, use
a new profile version instead of overwriting a live version.

The regional release orchestrator treats `runtime_profile.version` as the
single source for the CPU fleet pin, data-plane Resource Collector, and
the Node Installer template. Training submission must use the same version in
its managed Pod annotations; Completion Watcher reads that annotation rather
than a static environment default.

Example rendering:

```bash
PROFILE=config/runtime-profile.regional-hyperpod-safe.example.yaml
REGISTRATION_CLUSTER_ID='<stable-regional-cluster-id>'
PROFILE_VERSION='hyperpod-v1'

python3 - "${PROFILE}" "${REGISTRATION_CLUSTER_ID}" "${PROFILE_VERSION}" \
  > /tmp/runtime-profile.json <<'PY'
import json
import pathlib
import sys
import yaml

payload = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
payload["cluster_id"] = sys.argv[2]
payload["profile_version"] = sys.argv[3]
print(json.dumps(payload, separators=(",", ":")))
PY
```

Use the authentication and endpoint command from
`docs/部署和运维手册.md` §5.5 or REG registration. A response containing
warnings means an `OWN`/`DELEGATE` capability was reduced to `OBSERVE`; do
not enable destructive actions until that warning is resolved.

## Boundary with packaged data

`src/gpu_fault/data/` contains versioned catalogs shipped inside the wheel:

- `nvidia-xid-catalog-610.generated.yaml`
- `nvidia-fabric-manager-sxid-2025-11-14.yaml`
- `env-inventory.json`

The HTML design document generates its XID appendix directly from the pinned
catalog. There is intentionally no editable XID policy example in this
directory: `GPU_FAULT_XID_POLICY_PATH` accepts only a complete, reviewed file
with the same schema as the packaged catalog. The file must be generated by
`tools/generate_nvidia_xid_policy.py` and contain a valid
`metadata.generatedSha256`; the runtime recomputes that digest before accepting
the policy.

Changing packaged data changes the `module_digest` of every component whose
dependency closure includes that data and requires a release rebuild. It does
not force unrelated components to roll: the regional deployer compares the
Control Plane, Executor and Node Runtime digests independently. Changing a
runtime profile under `config/` does not require rebuilding a wheel; it
requires validation and one immutable regional registration.

Regional deployment orchestration examples are intentionally colocated with
their consumer rather than this directory:

- `deploy/control-plane/regional/regional-env.example.sh`
- `deploy/control-plane/regional/regional-release.example.json`

The JSON example remains the low-level rollout contract and break-glass input.
The normal administrator path starts from `config/site.example.yaml`.
