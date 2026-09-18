# Node Key Custody Evidence

AUTH-015 separates prospective provisioning custody, deployed key activation,
and the full single-node rotation verdict. A current Secret map, host scan,
heartbeat, or unsigned digest cannot establish historical installation custody.

## Trust Boundary

The trusted deploy host runs the actual
`deploy/node/provision_node_action_keys.py` path. An independent approval
authority authorizes its exact source identity, release, site, cluster and
node inventory before provisioning. A separate provisioner authority signs
observations from that path. A third, independent witness authority signs
fresh challenges against the deployed Node Agent endpoints.

Trust is supplied separately from the evidence using an explicitly approved
SHA-256 pin of a trust document. Each authority has a P-256 public key, its
canonical DER SubjectPublicKeyInfo SHA-256, and an immutable KMS key ARN.
Distinct PEM spellings of the same key are not independent authorities.
The administrator and witness also reject reuse of the release verification key.
KMS `ECDSA_SHA_256` signatures cover the SHA-256 of canonical statement bytes;
verification uses the pinned public key and local OpenSSL. A self-computed
digest without the required signature has no authorization value.

The approval key must not be available to the provisioner. Provisioner and
witness KMS permissions belong to separately controlled trusted roles, never
GPU Pods, Executors, or nodes. Signing permissions must be restricted to the
reviewed workflows; signatures cannot establish truth if their authorities
or the trusted deploy host are compromised. The implementation refuses to
issue an approval through its signing API.

This protocol attests the controlled provisioning path. It does **not**
claim that a master never existed in an unobserved location, that arbitrary
host activity was monitored, or that a present snapshot establishes the past.

## Receipt Chain

The strict schemas are in
`gpu_fault.admin.node_key_custody_models`. Unknown fields, duplicate JSON keys,
numeric substitutes for boolean protocol fields, and incomplete chains fail
closed.

1. `Authorization` binds the signed release identity and manifest digest,
   site/Region/CPU and GPU EKS ARNs, both cluster and namespace incarnations,
   complete NodeName-to-UID inventory, CPU master Secret UID/key/digest,
   producer and witness source identities, operation, predecessor and an
   explicit window of at most 24 hours.
2. Before any key write, the real provisioner verifies live cluster/namespace
   anchors and Node UIDs, privately compares the controlled master file to
   the bound CPU Secret, and signs an exclusive, fsynced `Started` receipt.
   Genesis requires the GPU key Secret to be absent and no target keys in
   the CPU map. Matching derived keys already present are not genesis proof.
3. The existing UID/resourceVersion CAS path writes only per-node key maps,
   retains unrelated CPU keys, and confirms both maps. Every custody-mode
   write rechecks the approval window and signed key plan. GPU metadata
   cannot contain the master or raw node key material.
4. `Completed` is signed only after fresh identity checks and exact byte
   digest readbacks. Its `runtime_activation_proved` is explicitly `false`.
   A rotation requires an independently activated predecessor and advances
   exactly one node's key generation; sibling states must be unchanged.
5. A separate `Activated` witness binds that completion, runtime Node UID,
   boot/incarnation/generation, endpoint and TLS certificate, and signed
   digests of the actual protocol and runtime identity observations.
   Installation activation is a useful prerequisite, not a complete
   AUTH-015 rotation PASS.

The currently supported v1 chain retains one release/site/Node UID binding.
Identity or release transitions need new explicitly authorized evidence;
there is no automatic provenance migration or grandfathering.
Receipt capture also requires a quiet CPU key-map window: CAS preserves
unrelated keys, but concurrent changes to their digest invalidate completion
rather than being attributed to this transaction.

## Provisioning Integration

The existing trusted-host wrapper accepts these additional arguments:

```text
--custody-request <controlled-request.json>
--custody-trust-sha256 <externally-approved-trust-document-sha256>
--custody-input-sha256 <administrator-bound-input-identity>
--custody-activation-state <private-activation-journal.json>
--custody-activation-sha256 <administrator-bound-activation-state-sha256>
```

The input-identity argument is optional for standalone callers and mandatory in the
configured administrator path. It binds the request, authorization, trust and
public keys, manifest and predecessor bytes across the subprocess handoff;
the loader checks it before and after loading. It is an integrity guard, not
an authority signature.

The activation pair is supplied together by the administrator's same-release
rotation coordinator, after its key-writer intent is durable. It binds the
actual helper to that journal's fixed deadline and exact owned installer wave.
The helper checks the state digest, UTC and monotonic deadlines, and current
ConfigMap identity/data immediately before each Secret submission. These are
internal provisioning arguments, not new `gpu-fault-admin` options or authority
to extend an approval. Missing or conflicting supplied bindings refuse.

The request has exactly these fields:

| Field | Contract |
| --- | --- |
| `trust` | Trust document path; public-key paths are relative to that document |
| `authorization` | Independently signed `Signed[Authorization]` document |
| `release_manifest` | Exact manifest bytes pinned by the authorization |
| `previous_chain` | Prior complete signed chain, or `null` for genesis |
| `state_directory` | Existing, owned mode-0700 directory for public receipts |
| `retired_key_file` | `null` for genesis; otherwise an exclusive mode-0600 output in a separate controlled directory |

Request paths are explicit deploy-host paths. The master file is an owned,
mode-0600 regular file whose exact bytes match the CPU Secret; custody mode
does not strip whitespace. Only the per-node keys use the existing GPU Secret
and target-node controlled-file delivery. No master is sent to GPU.
Subcommands receive an allowlisted environment containing role/config-file
references, not raw key credentials. Sensitive data is private memory/stdio;
only public identities, digests and signatures enter receipts.

The retired key is necessary for the negative runtime challenge. It is not
a receipt or release asset, cannot be placed under the receipt directory,
`artifacts`, or `dist`, and must remain on the trusted deploy host. Its
approved retention and removal belong to the enclosing rotation lifecycle.
The witness never copies it to a GPU node or changes its contents.

`gpu_fault.admin.node_key_proof.load_node_key_custody_request` keeps the
implementation in the deploy-host distribution's import closure. The admin
integration fingerprints custody inputs and implementations and waits for
the release; ordinary Secret membership is not a substitute for the receipt.
Custody modules are in the deploy-host source closure and release delivery's
node-template identity, not the Control Plane, Executor or Node Runtime wheels.
`gpu_fault.admin.node_key_custody_admin_entry` owns command registration, site
preflight and join-pause recording. The provisioning module does not import
bootstrap discovery; only this higher-level entry module does. Runtime Profile
selection belongs to the configured node-key task graph.

Successful provisioning writes `<transaction_id>.started.json` and
`<transaction_id>.chain.json`. An existing start is not replay authority.
A failure after the start or an uncertain write leaves incomplete evidence
for reconciliation, not a manufactured completion from current state.
In custody mode, an unacknowledged Secret create is never recovered from
matching current bytes: those bytes cannot identify which creator won.
Legacy provisioning without custody inputs can still synchronize keys but
does not produce custody evidence or qualify a pre-existing install.

## Administrator Enrollment

Custody is off by default. It is enabled only by the explicit local command:

```text
gpu-fault-admin node-key-custody configure --state-dir <state-dir> \
  --file <mode-0600-selection.json> --trust-sha256 <external-trust-pin>
```

The selection has `schema_version: 1`, `trust` (a public trust-document path),
`allow_staging` (optional, default `false`), and `clusters`, a nonempty map
from full GPU EKS ARN to either `null` or a provisioning request path. Paths
are resolved relative to the selection file. No environment variable enables
custody. The command takes the normal site lock and persists a mode-0600
registration under `<state-dir>/node-key-custody/`; it does not call KMS Sign,
create namespaces or invent future identities.
Registration pins request, authorization, trust/public-key, manifest and
predecessor contents, not merely their paths. Configuration and consumption
recheck those identities; changed files require explicit reconfiguration.

1. Enroll a target with a `null` request and run the normal deploy or join.
   Access/resource preparation and release build retain their existing
   prerequisites. Custody key tasks wait for the verified release and real
   namespaces/master source, then emit a content-addressed unsigned preparation
   with `authorization: false` and stop before key writes.
2. Review that preparation through the independent authority. It contains the
   actual cluster/namespace/Node UIDs, master-source digest, release and source
   identities. The authority supplies the fresh signed authorization; neither
   the administrator command nor a digest turns the preparation into authority.
3. Create the provisioning request, replace that target's `null` with its path,
   and repeat `configure` with the external trust pin. Resume the same deploy
   or join. Inputs and live bindings are rechecked before the actual trusted
   provisioning helper is invoked.
4. A completed signed receipt proves provisioning even if the enclosing
   bootstrap checkpoint was lost. Rotation additionally requires its bound
   runtime activation journal; a Secret receipt alone cannot satisfy it.
   An old key-shape-only checkpoint,
   matching current keys, an incomplete start or a changed input file cannot.
   Join pauses retain the prepared namespace and local inputs, rather than
   automatically rolling them back and changing the preauthorized UIDs.

When enabled, key tasks are serialized behind release completion, including
mixed configured/legacy targets sharing the CPU key map; unrelated IAM and
resource preparation still run independently. Custody-enabled batch joins are
serialized. Without enrollment the original first-deploy and batch parallelism
is unchanged. The source-deploy classifier must reach the application/bootstrap
path for enrolled custody rather than skipping it as unchanged or deploy-host-only.

Public preflight only verifies current signed receipts and live bindings. It
never signs or creates a preparation, even when evidence is missing. Supplying
custody does not waive the normal Profile, release, identity or runtime gates.
An existing Profile/template policy change is unresolved custody input and
stops before it can be preauthorized as the old Profile version. The initial
default uses the same known `hyperpod-v1` version and config-digest computation
as the normal first-bootstrap release path.

Enrollment cannot silently remove a cluster or replace its trust pin. A
missing/corrupt registration is an error, not an opt-out. Unstarted requests
may be explicitly reconfigured. A started request can be replaced only by a
new independently authorized rotation whose complete, activated predecessor
matches that enrolled transaction's signed completion. Use new immutable
request/authorization files and a separate private retired-key destination;
the next deploy rechecks the bindings and executes the existing CAS rotation.
Configuration itself never rotates keys. Incomplete starts still require
custody reconciliation. This version does not provide an automatic provenance
migration, cancellation of an uncertain writer, or a trust downgrade.

## Same-Release Activation

An independently authorized rotation still needs delivery when its software
release is unchanged. `node_key_custody_activation.CustodyActivation` keeps
operational progress under the same site lock and authorization. It captures the
committed release, namespace/Node/consumer identities, sibling keys and installer
wave; guards and drains the selected Agent, fences an owned one-node wave, then
allows the signed key provisioner to write. It refreshes Executor signing,
installs only the selected node using a bound activation identity, refreshes CPU
verification, observes fresh matching Agents and restores the owned wave.

Each step records intent before mutation and verifies its original bindings.
Once fenced, a resumed transaction and each later mutation must still observe
the same wave UID, generation, allowed node and budget. A premature return to
the unrestricted original wave is not ownership. Only the already-started final
unfence phase may reconcile its own lost acknowledgement without repeating
earlier mutation phases.
The deadline cannot exceed the authorization expiry or the fixed 1800-second
activation budget. It is fixed from the persisted start, not renewed on resume.
A step returning after that deadline cannot be recorded as complete; even a
read-only completion probe validates ordered, in-window completion times.
The actual key-writer subprocess shares that remaining budget and rechecks it
at each submission, not just when the coordinator consumes the result. A
request submitted before expiry can still have an unknown outcome afterward;
its incomplete `Started` record is retained for reconciliation.
An interrupted or uncertain step retains the journal and
fence for fail-forward/reconciliation; it does not write old keys back, infer
success from identical Secret bytes, or allow an ordinary release NOOP to bypass
pending activation. Read-only probes do not execute these steps.

Consumer reload evidence must prove the expected projected key was available
before the process loaded its cache; Pod Ready or a current file digest alone
is insufficient. The probe binds a stable projected-file descriptor and path,
mtime/ctime, and the unchanged container identity. Linux PID 1 start ticks and
bounded realtime/boottime samples provide a conservative process-start lower
bound, cross-checked with the Kubernetes timestamp interval. Second-precision
timestamps do not justify rounding the file time backward. Ambiguous sub-tick
ordering or inconsistent clocks refuses the proof; consumer reloads have at
most three owned attempts. The node clock/kernel and trusted Kubelet remain
platform assumptions, not an attestation against privileged clock tampering.
CPU capture and projection select the actual container of its recognized
rendered role; worker and spool-worker containers are not named `api`.
The operational result is `DEPLOYED_NOT_WITNESSED`, not an
independent signed `Activated` receipt. The separate witness below remains
mandatory and does not deploy or restart anything.

## AUTH-015 Witness

The guarded identity runner consumes:

```text
--auth015-release-proof <existing-release-proof-descriptor>
--auth015-custody-proof <custody-descriptor>
--auth015-custody-trust-sha256 <externally-approved-trust-document-sha256>
--node <rotated-node>
--node <unchanged-sibling>
```

The custody descriptor contains exactly `trust`, `chain`, and
`retired_key_file`; paths are relative to the descriptor, and the last field
is `null` for initial activation. The approved plan binds every input's
contents, including all public keys and the controlled retired-key file.
Replacing bytes at the same path invalidates the plan.
Custody mode forbids `--fleet-master-file` and `--host-probe-image`.

The runner verifies the complete signed chain, current release, full Node
UID inventory, namespace/cluster anchors and actual CPU/GPU key bytes. It
requires fresh authenticated heartbeats after provisioning for both nodes,
then uses the release-bound Agent records and pinned HTTPS
certificates to prove:

- the new key authenticates command and result-query requests;
- sibling and retired keys receive exact invalid-signature rejections for both;
- the challenged command remains absent before and after the test;
- the sibling key and runtime incarnation remain bound to the predecessor.

The existing A-to-B cross-node subproof remains intact. Rotation adds the
B-to-A and retired-A-to-A checks, and the independent witness signs a digest
of both protocol observations.

Challenges use the existing non-admissible time window and disallowed
`FREEZE_EVIDENCE` operation. They cannot dispatch a physical action, including
if the operation later gains a handler. The witness creates no host probes,
rotates no Secrets, restarts no process and performs no restore. It does
request an independent KMS signature and writes a content-addressed public
chain receipt. Existing live-runner approval, target and maintenance-window
guards remain mandatory.

Missing provenance returns FAIL/requires-new-authorized-evidence before
master reads, probes or key mutation. A key's presence in a Secret does not
prove runtime activation: the endpoint must actually accept the new key and
deny its predecessor. An initial activation produces a signed prerequisite
but remains full-case FAIL until the authorized rotation is deployed and
verified.
The standalone signature subproof reports supplied-key denial only and keeps
its custody/rotation proof flags false. Only the complete independently
verified and signed custody chain establishes those claims.

## Verification Limits

Hermetic tests exercise real ECDSA verification, a real local OpenSSL
verifier, stateful fake provisioning I/O, the guarded entry with fake I/O,
and the actual Node Agent application without dispatching an action.
They cover forged or incomplete chains, wrong authorities, replay, identity
and generation drift, byte mismatches, missing activation, old/sibling key
acceptance, input replacement and incomplete writes.
Fake administrator deploy/join flows cover explicit configuration, preparation,
authorized resume, old-checkpoint rejection, read-only preflight, successor
rotation, conditional release dependencies and unchanged default parallelism.

These tests are not LIVE installation, KMS IAM, Kubernetes admission,
node installer, or hardware evidence. Prospective capture must be exercised
through a separately authorized installation/rotation on the actual trusted
path. No historical installation claim can be reconstructed by running the
current witness against an old, unproven installation.
