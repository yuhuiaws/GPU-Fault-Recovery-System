"""Offline contract of the GF-REGIONAL-BOOT-029 driver.

The driver runs the five admin lifecycle stages in order against fake
``gpu-fault-admin``, ``docker`` and ``aws`` binaries that log every invocation
to one shared file, so the order across tools (cache prune before each cold
deploy, no prune between the middle stages) is observable.

Stage 5 follows README.md's procedure from a pristine copy of ``--repo`` in a
clean environment, so ``--repo`` is a self-contained git checkout built from
this working tree whose ``Makefile`` is replaced by a fixture: its
``deploy-host-setup-online`` target creates the ``.venv`` the README activates,
and the ``gpu-fault-admin`` inside that venv is the same fake, reached only the
way the README reaches it.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from scripts.e2e.regional import boot029_readme

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "acceptance" / "admin-lifecycle-sequence.sh"
CPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/cpu-fixture"
GPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-fixture"
ADMIN_EMAIL = "admin@example.com"

FAKE_ADMIN = """#!/usr/bin/env bash
set -euo pipefail
printf 'gpu-fault-admin %s\\n' "$*" >> "$FAKE_TOOL_LOG"
verb="$1"; shift
state_dir=""
snapshot_policy=retain
approve_plan=""
reference=""
request_arns=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) state_dir="$2"; shift 2 ;;
    --aurora-final-snapshot) snapshot_policy="$2"; shift 2 ;;
    --gpu-cluster-arn) request_arns+=("$2"); shift 2 ;;
    --approve-profile-plan) approve_plan="$2"; shift 2 ;;
    --reference) reference="$2"; shift 2 ;;
    *) shift ;;
  esac
done
# The GPU clusters this invocation names, in order; the first-deploy default
# is the fixture's single cluster.
export FAKE_REQUEST_GPU_ARNS="${request_arns[*]:-$FAKE_GPU_ARN}"
if [[ "$verb" == "${FAKE_ADMIN_FAIL_VERB:-}" ]]; then
  echo "injected failure in $verb" >&2
  exit 7
fi
if [[ "$verb" == join-cluster && "$state_dir" == "${FAKE_ADMIN_FAIL_JOIN_STATE:-}" ]]; then
  echo "injected failure in $verb for $state_dir" >&2
  exit 7
fi
case "$verb" in
  deploy)
    mkdir -p "$state_dir"
    echo "#1 [internal] load build definition from Dockerfile"
    echo "#5 [1/6] FROM registry.access.redhat.com/ubi9/python-312-minimal@sha256:0"
    echo "#5 resolve registry.access.redhat.com/ubi9/python-312-minimal done"
    echo "#5 extracting sha256:1111111111111111111111111111111111111111111111111111111111111111"
    echo "#6 [2/6] WORKDIR /app"
    if [[ -n "${FAKE_ADMIN_CACHED:-}" ]]; then echo "#6 CACHED"; fi
    echo "#7 [3/6] COPY requirements/runtime.lock /tmp/gpu-fault-runtime.lock"
    echo "#8 [4/6] RUN python -m pip install"
    echo "#9 [5/6] COPY wheels /tmp/gpu-fault-wheels"
    echo "#10 [6/6] RUN python -m venv"
    echo "#11 exporting to image"
    printf '{"schema_version": 1, "site_id": "fixture-site", "phase": "site-ready"}' \\
      > "$state_dir/bootstrap-state.json"
    python3 - "$state_dir" <<'PY'
import json, os, pathlib, sys
state = pathlib.Path(sys.argv[1])
dist = state / "snapshot/dist"
dist.mkdir(parents=True, exist_ok=True)
mapping = {"control_plane": "runtime", "executor": "executor", "node_dependencies": "node_dependencies"}
images = {name: {"reference": "repo@sha256:" + str(index) * 64, "deployable": True,
    "registry_reused": False, "image_input_sha256": "a" * 64,
    "source_identity_sha256": "b" * 64} for index,name in enumerate(mapping)}
(dist / "current-release.json").write_text(json.dumps({"delivery": {"images": {
    field: {"reference": images[name]["reference"]} for name,field in mapping.items()}}}))
(dist / "release-runtime-image.json").write_text(json.dumps({
    "schema_version": 3, "source_identity_sha256": "b" * 64, "images": images}))
def identity(role):
    return {"input_arn": os.environ[f"FAKE_{role}_ARN"],
        "eks_arn": os.environ.get(f"FAKE_{role}_EKS_ARN", os.environ[f"FAKE_{role}_ARN"]),
        "hyperpod_arn": os.environ.get(f"FAKE_{role}_HYPERPOD_ARN", "")}
def gpu_identity(arn):
    # The fixture's own cluster keeps its HyperPod alias mapping; any other
    # cluster of a mixed site is a plain-EKS member whose identity is its ARN.
    if arn == os.environ["FAKE_GPU_ARN"]:
        return identity("GPU")
    return {"input_arn": arn, "eks_arn": arn, "hyperpod_arn": ""}
(state / "bootstrap-state.json").write_text(json.dumps({
    "schema_version": 3, "site_id": "fixture-site",
    "phase": "aws-infrastructure-ready" if os.getenv("FAKE_ADMIN_FAIL_DEPLOY_STATE") == str(state) else "site-ready",
    "resources": {"release": {"manifest": str(dist / "current-release.json")},
    "initial_deploy_target": {"cpu": identity("CPU"),
       "gpu_clusters": [gpu_identity(arn) for arn in os.environ["FAKE_REQUEST_GPU_ARNS"].split()]}}}))
PY
    if [[ "$state_dir" == "${FAKE_ADMIN_FAIL_DEPLOY_STATE:-}" ]]; then exit 19; fi
    # The release step records its installation registry sync through the
    # product's own writer; remove-cluster and uninstall judge that record.
    python3 - "$state_dir" <<'PY'
import os, pathlib, sys
sys.path.insert(0, os.environ["FAKE_REPO_SRC"])
from gpu_fault.admin.resource_registry_outcome import record_path, write_outcome
state, mode = pathlib.Path(sys.argv[1]), os.getenv("FAKE_REGISTRY_SYNC", "synced")
if mode == "unreadable":
    record_path(state).parent.mkdir(parents=True, exist_ok=True)
    record_path(state).write_text("{")
elif mode in {"synced", "failed"}:
    outcome = {"status": "SYNCED", "path": str(state / "installation-resources.json"),
        "sha256": "a" * 64, "synced_at": "2026-01-01T00:00:00Z"}
    if mode == "failed":
        outcome = {"status": "UNAVAILABLE", "synced_at": "2026-01-01T00:00:00Z",
            "error": "fixture registry sync failed"}
    write_outcome(state, release_id="fixture-release", outcome=outcome)
PY
    ;;
  remove-cluster)
    python3 - "$state_dir" <<'PY'
import hashlib, json, os, pathlib, sys
arn = os.environ["FAKE_REQUEST_GPU_ARNS"].split()[0]
if arn == os.environ["FAKE_GPU_ARN"]:
    name, eks = "cluster-a", os.environ.get("FAKE_GPU_EKS_ARN", arn)
    hyperpod = os.environ.get("FAKE_GPU_HYPERPOD_ARN", "")
    provider = "hyperpod"
    cluster = hyperpod.rsplit("/", 1)[-1] if hyperpod else "fixture-hyperpod"
    groups = [("spot-p5en", "instance group")]
else:
    name, eks, hyperpod = "cluster-" + hashlib.sha256(arn.encode()).hexdigest()[:8], arn, ""
    provider, cluster = "eks-managed", arn.rsplit("/", 1)[-1]
    groups = [("gpu-p5en-efa-spot-node-group", "managed node group")]
# The product's OWNERSHIP RELEASE step: the label leaves the node groups the
# site selected (the fake aws reads the marker), unless the operator kept it.
keep = bool(os.getenv("FAKE_ADMIN_KEEP_LABEL"))
outcome = "retained" if keep else "released"
marker = pathlib.Path(os.environ["FAKE_TOOL_LOG"]).parent / "ownership-released"
if not keep:
    marker.touch()
path = pathlib.Path(sys.argv[1]) / "remove-cluster" / name / "state.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "phase": "COMPLETED", "attempt_id": "a" * 32,
    "completed_steps": ["AURORA_UPDATED", "AWS_DETACHED", "CONTROL_REGISTRY_DRAINING",
        "CONTROL_REGISTRY_REMOVED", "DISCOVERED", "KUBERNETES_QUIESCED", "KUBERNETES_REMOVED",
        "OWNERSHIP_RELEASED", "RELEASE_STATE_UPDATED", "SITE_COMMIT_STARTED", "SITE_UPDATED",
        "VERIFIED"],
    "target": {"eks_cluster_arn": eks, "provider": provider,
        "hyperpod_cluster_name": cluster if provider == "hyperpod" else ""},
    "evidence": {"DISCOVERED": {"provider_identity": {
        "eks_arn": eks, "hyperpod_arn": hyperpod}},
        "OWNERSHIP_RELEASED": {"label": "gpu-fault.io/control-plane", "outcome": outcome,
            "provider": provider, "cluster": cluster, "nodepools": [],
            "groups": [{"name": group, "kind": kind, "before": "site", "outcome": outcome,
                "after": "site" if keep else None} for group, kind in groups]}}}))
PY
    ;;
  join-cluster)
    # A cluster of the other provider than the site's first deploy needs its
    # own Runtime Profile version: without that version's snapshot the join
    # writes the CLUSTER_PROFILE_NEW plan and stops; the approved rerun writes
    # the snapshot (later joins of the provider reuse it) and completes.
    python3 - "$state_dir" "$approve_plan" "$reference" <<'PY'
import hashlib, json, os, pathlib, sys
sys.path.insert(0, os.environ["FAKE_REPO_SRC"])
from gpu_fault.admin.profile_approval import profile_plan_sha256
state, approve, reference = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
aliases = {os.environ["FAKE_GPU_ARN"], os.environ.get("FAKE_GPU_EKS_ARN", ""),
    os.environ.get("FAKE_GPU_HYPERPOD_ARN", "")}
def provider(arn):
    return "hyperpod" if arn in aliases else "eks-managed"
versions = {"hyperpod": "hyperpod-v1", "eks-managed": "eks-v1"}
target = json.loads((state / "bootstrap-state.json").read_text())["resources"]["initial_deploy_target"]
site_provider = provider(target["gpu_clusters"][0]["input_arn"])
plan_path = state / "release-deploy/profile-plan.json"
for arn in os.environ["FAKE_REQUEST_GPU_ARNS"].split():
    kind = provider(arn)
    snapshot = state / "profiles" / (versions[kind] + ".yaml")
    if kind != site_provider and not snapshot.exists():
        plan = {"schema_version": 1, "site_identity": {"site_name": "fixture-site",
            "aws_region": "us-west-2", "cpu_eks_arn": os.environ["FAKE_CPU_ARN"]},
            "site_identity_sha256": "0" * 64,
            "registration_cluster_id": "cluster-" + hashlib.sha256(arn.encode()).hexdigest()[:8],
            "current_version": versions[site_provider], "desired_version": versions[kind],
            "current_policy_digest": "1" * 64, "policy_digest": "2" * 64,
            "live_profile_sha256": None, "active_source_sha256": None,
            "source_sha256": "3" * 64, "snapshot_sha256": "4" * 64,
            "change_kind": "CLUSTER_PROFILE_NEW",
            "changes": ["cluster runs its own Runtime Profile " + versions[kind]],
            "approval_required": True}
        plan["plan_sha256"] = profile_plan_sha256(plan)
        if approve != plan["plan_sha256"]:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan))
            print("Runtime Profile policy changed; the join stopped for review.\\n"
                "  plan_sha256: " + plan["plan_sha256"], file=sys.stderr)
            sys.exit(2)
        if not reference:
            print("--approve-profile-plan requires --reference", file=sys.stderr)
            sys.exit(2)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("version: " + versions[kind] + "\\n")
        (state / "release-deploy/profile-approval.json").write_text(
            json.dumps({"plan_sha256": approve, "reference": reference}))
        plan_path.unlink(missing_ok=True)
    # A re-join of a removed cluster starts a new attempt (the product archives
    # the completed one), so the stage-3 proof is newer than its baseline.
    path = state / "join-cluster" / hashlib.sha256(arn.encode()).hexdigest()[:12] / "state.json"
    previous = json.loads(path.read_text())["attempt"] if path.exists() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": "COMPLETED", "attempt": previous + 1, "gpu_cluster_arn": arn}))
PY
    ;;
  uninstall)
    mkdir -p "$state_dir/uninstall"
    python3 - "$state_dir" "$snapshot_policy" <<'PY'
import json, os, pathlib, sys
state = pathlib.Path(sys.argv[1])
value = {
    "phase": "COMPLETED", "cpu_disposition": "keep", "reset_database": True,
    "final_snapshot_policy": os.getenv("FAKE_UNINSTALL_SNAPSHOT_POLICY", sys.argv[2]),
    "site_identity": {"cpu_eks_arn": os.getenv("FAKE_CPU_EKS_ARN", os.environ["FAKE_CPU_ARN"])},
}
if os.getenv("FAKE_OMIT_UNINSTALL_SNAPSHOT_POLICY"):
    value.pop("final_snapshot_policy")
# The product's OWNERSHIP RELEASE phase, per GPU cluster the site lists (the
# clusters stage 3 re-joined); the fake aws reads the marker.
keep = bool(os.getenv("FAKE_ADMIN_KEEP_LABEL"))
outcome = "retained" if keep else "released"
aliases = {os.environ["FAKE_GPU_ARN"], os.environ.get("FAKE_GPU_EKS_ARN", ""),
    os.environ.get("FAKE_GPU_HYPERPOD_ARN", "")}
clusters = []
for path in sorted((state / "join-cluster").glob("*/state.json")):
    arn = json.loads(path.read_text())["gpu_cluster_arn"]
    if arn in aliases:
        hyperpod = os.environ.get("FAKE_GPU_HYPERPOD_ARN", "")
        eks = os.environ.get("FAKE_GPU_EKS_ARN") or os.environ["FAKE_GPU_ARN"]
        cluster = hyperpod.rsplit("/", 1)[-1] if hyperpod else "fixture-hyperpod"
        provider, groups = "hyperpod", ["spot-p5en"]
    else:
        eks, cluster, provider, groups = arn, arn.rsplit("/", 1)[-1], "eks-managed", ["gpu-p5en-efa-spot-node-group"]
    clusters.append({"cluster_id": cluster, "eks_cluster_arn": eks,
        "hyperpod_cluster_name": cluster if provider == "hyperpod" else "",
        "provider": provider, "cluster": cluster, "outcome": outcome, "nodepools": [],
        "groups": [{"name": group, "outcome": outcome} for group in groups]})
value["ownership_release"] = {"label": "gpu-fault.io/control-plane", "outcome": outcome,
    "clusters": clusters}
if not keep:
    (pathlib.Path(os.environ["FAKE_TOOL_LOG"]).parent / "ownership-released").touch()
(state / "uninstall/state.json").write_text(json.dumps(value))
PY
    ;;
esac
if [[ "$verb" == "${FAKE_ADMIN_FAIL_AFTER_WRITE:-}" ]]; then exit 17; fi
"""

FAKE_DOCKER = """#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$FAKE_TOOL_LOG"
case "$*" in
  "buildx inspect"*) printf 'Name:   fixture-builder\\nDriver: docker-container\\n' ;;
  images*) printf 'gpu-fault-runtime-local:latest\\nubuntu:24.04\\n<none>:<none>\\n' ;;
esac
"""

FAKE_AWS = """#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_TOOL_LOG"])
with log.open("a") as stream:
    stream.write("aws " + " ".join(args) + "\\n")
operation = args[1]
def argument(name):
    return args[args.index(name) + 1]
if operation == "describe-repositories":
    if not os.getenv("FAKE_ECR_EXISTS"):
        print("An error occurred (RepositoryNotFoundException) when calling the DescribeRepositories operation: absent", file=sys.stderr)
        sys.exit(254)
    name = argument("--repository-names")
    print(json.dumps({"repositories": [{"repositoryName": name, "registryId": "123456789012",
        "repositoryArn": "arn:aws:ecr:us-west-2:123456789012:repository/" + name}]}))
    sys.exit(0)
elif operation == "list-tags-for-resource":
    print(json.dumps({"tags": [{"Key": "gpu-fault:site-id", "Value": "fixture-site"}]}))
    sys.exit(0)
# The CPU control plane's HyperPod identity, as the product derives the site id
# from it (Region, HyperPod name, first 8 hex digits of the ARN digest): a
# SageMaker CPU ARN names it directly, an EKS CPU ARN is orchestrated by the
# fixture CPU HyperPod cluster.
cpu_env = os.environ["FAKE_CPU_ARN"]
if cpu_env.startswith("arn:aws:sagemaker:"):
    cpu_name, cpu_hyperpod_arn, cpu_eks_arn = cpu_env.rsplit("/", 1)[-1], cpu_env, "arn:aws:eks:us-west-2:123456789012:cluster/cpu-fixture-eks"
else:
    cpu_name, cpu_hyperpod_arn, cpu_eks_arn = "fixture-cpu-hyperpod", "arn:aws:sagemaker:us-west-2:123456789012:cluster/fixture-cpu-hyperpod", cpu_env
site_id = "us-west-2-" + cpu_name + "-" + hashlib.sha256(cpu_hyperpod_arn.encode()).hexdigest()[:8]
# The label is on the GPU node groups until a de-registration releases it (the
# fake admin leaves the marker); the operator's update calls put it back.
marker = log.parent / "ownership-released"
ownership = {} if os.getenv("FAKE_UNLABELLED_GPU") or marker.exists() else {"gpu-fault.io/control-plane": site_id}
if operation == "list-clusters":
    # The Region's HyperPod clusters: the CPU control plane's own HyperPod
    # cluster (the site id derives from it) and the fixture cluster that
    # orchestrates the fixture GPU EKS cluster, so that ARN is a HyperPod
    # cluster and every other EKS ARN is plain EKS. A twin naming the same EKS
    # cluster is the configuration the product refuses to guess about.
    names = [cpu_name, "fixture-hyperpod"]
    if os.getenv("FAKE_HYPERPOD_DUPLICATE"):
        names.append("fixture-hyperpod-twin")
    print(json.dumps({"ClusterSummaries": [{"ClusterName": name} for name in names]}))
elif operation == "describe-cluster":
    name = argument("--cluster-name")
    name = name.rsplit("/", 1)[-1] if name.startswith("arn:") else name
    labels = ownership
    if name == cpu_name:
        print(json.dumps({"ClusterName": name, "ClusterArn": cpu_hyperpod_arn,
            "Orchestrator": {"Eks": {"ClusterArn": cpu_eks_arn}},
            "InstanceGroups": [{"InstanceGroupName": "cpu-group", "InstanceType": "ml.m5.2xlarge"}]}))
    else:
        print(json.dumps({"ClusterName": name, "ClusterStatus": "InService",
            "ClusterArn": os.environ.get("FAKE_GPU_HYPERPOD_ARN") or "arn:aws:sagemaker:us-west-2:123456789012:cluster/" + name,
            "Orchestrator": {"Eks": {"ClusterArn": os.environ.get("FAKE_GPU_EKS_ARN") or os.environ["FAKE_GPU_ARN"]}},
            "InstanceGroups": [{"InstanceGroupName": "spot-p5en", "InstanceType": "ml.p5en.48xlarge",
                "CurrentCount": 2, "TargetCount": 2, "Status": "InService",
                "LifeCycleConfig": {"SourceS3Uri": "s3://fixture/", "OnCreate": "on_create.sh"},
                "ExecutionRole": "arn:aws:iam::123456789012:role/fixture-hyperpod",
                "KubernetesConfig": {"CurrentLabels": labels, "DesiredLabels": labels}},
                {"InstanceGroupName": "accelerated-g6", "InstanceType": "ml.g6.12xlarge"}]}))
elif operation == "update-nodegroup-config":
    # The operator's own re-label, exactly as `gpu-fault-admin site-id` prints it.
    labels = argument("--labels")
    if labels != "addOrUpdateLabels={gpu-fault.io/control-plane=" + site_id + "}":
        raise RuntimeError("unexpected node group label update: " + labels)
    marker.unlink(missing_ok=True)
    print(json.dumps({"update": {"id": "fixture-update", "status": "InProgress"}}))
elif operation == "update-cluster":
    specs = json.loads(argument("--instance-groups"))
    if not all(spec["KubernetesConfig"]["Labels"].get("gpu-fault.io/control-plane") == site_id
            and spec["InstanceCount"] == 2 and spec["ExecutionRole"] for spec in specs):
        raise RuntimeError("unexpected instance group update: " + json.dumps(specs))
    marker.unlink(missing_ok=True)
    print(json.dumps({"ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/fixture-hyperpod"}))
elif operation == "list-nodegroups":
    print(json.dumps({"nodegroups": ["cpu-ng", "gpu-p5en-efa-spot-node-group"]}))
elif operation == "describe-nodegroup":
    name = argument("--nodegroup-name")
    gpu = name != "cpu-ng"
    labels = ownership if gpu else {}
    print(json.dumps({"nodegroup": {"nodegroupName": name, "status": "ACTIVE",
        "instanceTypes": ["p5en.48xlarge"] if gpu else ["m7i-flex.xlarge"], "labels": labels}}))
elif operation in {"list-images", "batch-delete-image"}:
    name = argument("--repository-name")
    marker = log.parent / ("removed-" + hashlib.sha256(name.encode()).hexdigest())
    if operation == "list-images":
        print(json.dumps({"imageIds": [] if marker.exists() else [{"imageDigest": "sha256:" + "1" * 64, "imageTag": "build-component"}]}))
    else:
        marker.write_text("removed")
        print(json.dumps({"imageIds": json.loads(argument("--image-ids")), "failures": []}))
else:
    raise RuntimeError("unexpected test AWS operation")
"""


# The README's ``make deploy-host-setup-online`` in the fixture checkout: it
# runs in the pristine copy under the clean environment, so every value it
# needs is baked in, and it refuses the two things the stage asserts against.
FIXTURE_MAKEFILE = """.PHONY: deploy-host-setup-online
deploy-host-setup-online:
\tbash fixture-setup.sh
"""

FIXTURE_SETUP = """#!/usr/bin/env bash
set -euo pipefail
printf 'make deploy-host-setup-online\\n' >> {tool_log}
[[ -z "$(git status --porcelain)" ]] || {{ echo 'fixture: source copy is dirty' >&2; exit 43; }}
[[ ! -e .venv ]] || {{ echo 'fixture: .venv pre-exists' >&2; exit 44; }}
[[ "$PWD" != {checkout} ]] || {{ echo 'fixture: ran inside the checkout' >&2; exit 48; }}
mkdir -p .venv/bin
printf 'export VIRTUAL_ENV=%q\\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\\n' "$PWD/.venv" > .venv/bin/activate
cat > .venv/bin/gpu-fault-admin <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
printf 'readme-admin-env %s\\n' "$(compgen -e | sort | tr '\\n' ' ')" >> {tool_log}
printf 'readme-admin-path %s\\n' "$PATH" >> {tool_log}
printf 'readme-admin-cwd %s\\n' "$PWD" >> {tool_log}
source {env_file}
exec {fake_admin} "$@"
WRAPPER
chmod 755 .venv/bin/gpu-fault-admin
"""

_TEMPLATE: Path | None = None


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checkout_template() -> Path:
    """This working tree (tracked and untracked files) as one committed repo."""

    global _TEMPLATE
    if _TEMPLATE is not None:
        return _TEMPLATE
    base = Path(tempfile.mkdtemp(prefix="boot029-checkout-template-"))
    atexit.register(shutil.rmtree, base, True)
    template = base / "repo"
    template.mkdir()
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    for raw in listing.split(b"\0"):
        source = ROOT / os.fsdecode(raw)
        if not raw or source.is_symlink() or not source.is_file():
            continue
        target = template / os.fsdecode(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _git(template, "init", "-q")
    _git(template, "config", "user.name", "Fixture")
    _git(template, "config", "user.email", "fixture@example.invalid")
    _git(template, "add", "-A", "-f", ".")
    _git(template, "commit", "-q", "-m", "checkout template")
    _TEMPLATE = template
    return template


def _install_checkout(tmp_path: Path, fake_bin: Path) -> Path:
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", str(_checkout_template()), str(checkout))
    _git(checkout, "config", "user.name", "Fixture")
    _git(checkout, "config", "user.email", "fixture@example.invalid")
    (checkout / "Makefile").write_text(FIXTURE_MAKEFILE, encoding="utf-8")
    (checkout / "fixture-setup.sh").write_text(
        FIXTURE_SETUP.format(
            tool_log=shlex.quote(str(tmp_path / "tools.log")),
            checkout=shlex.quote(str(checkout)),
            env_file=shlex.quote(str(tmp_path / "fake-env.sh")),
            fake_admin=shlex.quote(str(fake_bin / "gpu-fault-admin")),
        ),
        encoding="utf-8",
    )
    _git(checkout, "add", "-A", "-f", ".")
    _git(checkout, "commit", "-q", "-m", "fixture makefile")
    return checkout


def _install_fakes(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in (
        ("gpu-fault-admin", FAKE_ADMIN),
        ("docker", FAKE_DOCKER),
        ("aws", FAKE_AWS),
    ):
        path = fake_bin / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    _install_checkout(tmp_path, fake_bin)
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_TOOL_LOG": str(tmp_path / "tools.log"),
        "FAKE_CPU_ARN": CPU_ARN,
        "FAKE_GPU_ARN": GPU_ARN,
        # The fake join computes the product's own plan digest from this tree.
        "FAKE_REPO_SRC": str(tmp_path / "checkout" / "src"),
        # The driver's environment must not reach the README deploy.
        "GPU_FAULT_SENTINEL": "driver",
        "AWS_REGION": "us-west-2",
    }


def _run(
    tmp_path: Path, env: dict[str, str], *extra: str
) -> subprocess.CompletedProcess[str]:
    # The README deploy runs in a clean environment; the fake it reaches
    # through the venv reads this run's FAKE_* knobs from a file instead.
    (tmp_path / "fake-env.sh").write_text(
        "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in sorted(env.items())
            if name.startswith("FAKE_")
        ),
        encoding="utf-8",
    )
    # ``--gpu-cluster-arn`` repeats per GPU cluster, so a caller naming its own
    # clusters replaces the fixture's default instead of adding to it.
    gpu = [] if "--gpu-cluster-arn" in extra else ["--gpu-cluster-arn", GPU_ARN]
    return subprocess.run(
        [
            str(SCRIPT),
            "--state-dir",
            str(tmp_path / "state"),
            "--cpu-cluster-arn",
            CPU_ARN,
            *gpu,
            "--admin-email",
            ADMIN_EMAIL,
            "--repo",
            str(tmp_path / "checkout"),
            "--confirm-isolated-build-host",
            *extra,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _tool_log(env: dict[str, str]) -> list[str]:
    path = Path(env["FAKE_TOOL_LOG"])
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def _admin_verbs(lines: list[str]) -> list[str]:
    return [line.split()[1] for line in lines if line.startswith("gpu-fault-admin ")]


def _record(tmp_path: Path) -> dict:
    path = tmp_path / "state" / "acceptance" / "admin-lifecycle-sequence.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_runs_the_five_stages_in_order_and_prunes_before_each_cold_deploy(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
    ], "the five stages must run exactly once each, in lifecycle order"
    deploys = [index for index, line in enumerate(lines) if "admin deploy " in line]
    prunes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("docker buildx prune")
    ]
    builder_prunes = [
        index for index, line in enumerate(lines) if line == "docker builder prune -af"
    ]
    assert len(prunes) == len(builder_prunes) == 2, "one prune pair per cold deploy"
    assert prunes[0] < builder_prunes[0] < deploys[0], (
        "stage 1 prunes the caches before its deploy"
    )
    assert deploys[0] < prunes[1] < builder_prunes[1] < deploys[1], (
        "stage 5 prunes again, after the middle stages and before its deploy"
    )
    assert "docker buildx prune --builder fixture-builder -af" in lines, (
        "the prune targets the builder buildx reports"
    )
    assert "docker image rm -f gpu-fault-runtime-local:latest" in lines, (
        "every local gpu-fault-runtime image is removed, nothing else"
    )
    assert not any("ubuntu" in line for line in lines if "image rm" in line), (
        "unrelated local images are left alone"
    )
    first_deploy = lines[deploys[0]]
    assert f"--state-dir {tmp_path / 'state'} " in first_deploy, first_deploy
    assert f"--cpu-cluster-arn {CPU_ARN}" in first_deploy, first_deploy
    assert f"--admin-email {ADMIN_EMAIL}" in first_deploy, first_deploy
    assert f"--state-dir {tmp_path / 'state-second'} " in lines[deploys[1]], (
        "stage 5 deploys into the second, empty state directory"
    )
    assert any(
        line.startswith("gpu-fault-admin remove-cluster ")
        and f"--gpu-cluster-arn {GPU_ARN} --confirm REMOVE_GPU_CLUSTER" in line
        for line in lines
    ), "remove-cluster names the GPU cluster and carries its confirmation token"
    assert any(
        "--cpu-cluster keep --reset-database --aurora-final-snapshot retain "
        "--confirm UNINSTALL_GPU_FAULT" in line
        for line in lines
    ), "uninstall keeps the CPU cluster and resets the database"

    record = _record(tmp_path)
    assert record["case_id"] == "GF-REGIONAL-BOOT-029", record
    assert sorted(record["stages"]) == ["1", "2", "3", "4", "5"], record["stages"]
    for number, stage in record["stages"].items():
        assert stage["status"] == "PASS", (number, stage)
        assert isinstance(stage["wall_seconds"], int), (number, stage)
        log = tmp_path / "state" / "acceptance" / stage["log_name"]
        assert stage["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest(), (
            number,
            "the record binds each stage to the digest of its log",
        )
    for number in ("1", "5"):
        cold = record["stages"][number]["cold_build"]
        assert cold["cached_layers"] == 0 and cold["base_image_pulled"] is True
        assert set(cold["images"]) == {"control_plane", "executor", "node_dependencies"}
        assert all(
            image["registry_reused"] is False for image in cold["images"].values()
        ), "each cold deployment must build every current image without registry reuse"
    for number in ("2", "3", "4"):
        assert record["stages"][number]["phase"] == "COMPLETED", (number, record)
    for number in ("2", "3"):
        journals = record["stages"][number]["journals"]
        assert [item["gpu_arn_sha256"] for item in journals] == [_sha256(GPU_ARN)], (
            number,
            "one completion journal per GPU cluster, in the run's order",
        )
    assert record["stages"]["4"]["aurora_final_snapshot"] == "retain", (
        "the default reset keeps its final snapshot"
    )
    expected_inputs = {
        "cpu_cluster_arn_sha256": _sha256(CPU_ARN),
        "gpu_cluster_arns_sha256": _sha256(GPU_ARN),
        "admin_email_sha256": _sha256(ADMIN_EMAIL),
        "state_dir_sha256": _sha256(str(tmp_path / "state")),
        "second_state_dir_sha256": _sha256(str(tmp_path / "state-second")),
        "aurora_final_snapshot_sha256": _sha256("retain"),
    }
    assert {key: record["inputs"][key] for key in expected_inputs} == expected_inputs
    assert set(record["inputs"]) == set(expected_inputs) | {
        "repository_sha256",
        "source_identity_sha256",
        "previous_site_id_sha256",
        "email_wait_sha256",
    }, "resume must also bind the repository, source and wait/cache policy"
    serialized = json.dumps(record)
    for secret in (CPU_ARN, GPU_ARN, ADMIN_EMAIL, "cpu-fixture", str(tmp_path)):
        assert secret not in serialized, f"{secret} leaked into the record"


def test_stops_at_the_first_failing_stage_and_resumes_from_it(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})

    assert failed.returncode != 0, failed.stdout
    assert "FAILED at stage 3 (join-cluster)" in failed.stderr, failed.stderr
    assert "--stage 3" in failed.stderr, "the resume hint names the failed stage"
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
    ], "nothing after the failing stage runs"
    record = _record(tmp_path)
    assert {number: stage["status"] for number, stage in record["stages"].items()} == {
        "1": "PASS",
        "2": "PASS",
        "3": "FAIL",
    }, record["stages"]
    assert record["stages"]["3"]["failure"], "the failing stage records a reason"
    stage_one_digest = record["stages"]["1"]["log_sha256"]

    resumed = _run(tmp_path, env, "--stage", "3")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _admin_verbs(_tool_log(env))[3:] == [
        "join-cluster",
        "uninstall",
        "deploy",
    ], "--stage 3 resumes at the failed stage without repeating 1 and 2"
    record = _record(tmp_path)
    assert record["resumed_from_stage"] == 3, record
    assert record["stages"]["1"]["log_sha256"] == stage_one_digest, (
        "earlier stage records survive a resume"
    )
    assert record["stages"]["3"]["status"] == "PASS", record["stages"]["3"]
    assert record["stages"]["3"]["failure"] is None, record["stages"]["3"]
    assert record["stages"]["5"]["status"] == "PASS", record["stages"]["5"]


def test_a_cached_layer_fails_the_cold_build_criterion_at_stage_one(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, {**env, "FAKE_ADMIN_CACHED": "1"})

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 1 (cold-first-deploy)" in result.stderr, result.stderr
    assert "CACHED" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy"], "stage 2 never starts"
    stage = _record(tmp_path)["stages"]["1"]
    assert stage["status"] == "FAIL", stage
    assert stage["failure"], stage
    assert "CACHED" in (tmp_path / "state/acceptance" / stage["log_name"]).read_text()


def test_ecr_tags_are_deleted_only_when_the_site_repositories_exist(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    digest = _sha256("fixture-site")[:12]

    absent = _run(tmp_path, env)

    assert absent.returncode == 0, absent.stdout + absent.stderr
    lines = _tool_log(env)
    assert any(
        f"ecr describe-repositories --repository-names gpu-fault/runtime-{digest}"
        in line
        for line in lines
    ), "the runtime repository is derived from the previous site id"
    assert any(
        f"ecr describe-repositories --repository-names gpu-fault/runtime-cache-{digest}"
        in line
        for line in lines
    ), "the build-cache repository is derived from the previous site id"
    assert not any("batch-delete-image" in line for line in lines), (
        "no delete is attempted when the repositories are gone"
    )
    separate = tmp_path / "second-case"
    separate.mkdir()
    env = _install_fakes(separate)
    present = _run(separate, {**env, "FAKE_ECR_EXISTS": "1"})

    assert present.returncode == 0, present.stdout + present.stderr
    deletes = [line for line in _tool_log(env) if "batch-delete-image" in line]
    assert len(deletes) == 2, deletes
    assert any(
        f"--repository-name gpu-fault/runtime-{digest} --image-ids " in line
        for line in deletes
    ), deletes
    assert any(
        f"--repository-name gpu-fault/runtime-cache-{digest} --image-ids " in line
        for line in deletes
    ), deletes


def test_a_first_deploy_refuses_a_state_directory_that_already_bootstrapped(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "bootstrap-state.json").write_text("{}", encoding="utf-8")

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "already holds bootstrap-state.json" in result.stderr, result.stderr
    assert _admin_verbs(_tool_log(env)) == [], "no admin command runs"


def test_usage_errors_exit_two_without_touching_any_tool(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    missing = subprocess.run(
        [str(SCRIPT), "--state-dir", str(tmp_path / "state")],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    bad_stage = _run(tmp_path, env, "--stage", "6")

    assert missing.returncode == 2, missing.stderr
    assert "usage:" in missing.stderr, missing.stderr
    assert bad_stage.returncode == 2, bad_stage.stderr
    assert _tool_log(env) == [], "argument errors never reach the tools"


def test_completed_journal_cannot_override_failed_admin_exit(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)
    result = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_AFTER_WRITE": "remove-cluster"})
    assert result.returncode != 0
    assert _admin_verbs(_tool_log(env)) == ["deploy", "remove-cluster"]
    journal = tmp_path / "state/remove-cluster/cluster-a/state.json"
    assert json.loads(journal.read_text())["phase"] == "COMPLETED"
    record = _record(tmp_path)
    assert record["stages"]["2"]["status"] == "FAIL"
    assert "3" not in record["stages"]


def test_resume_rejects_changed_bound_inputs_without_rewriting_the_receipt(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})
    assert failed.returncode != 0
    path = tmp_path / "state/acceptance/admin-lifecycle-sequence.json"
    before, tools = path.read_bytes(), _tool_log(env)
    resumed = _run(
        tmp_path, env, "--stage", "3", "--admin-email", "changed@example.invalid"
    )
    assert resumed.returncode != 0
    assert "input identity" in resumed.stderr
    assert path.read_bytes() == before
    assert _tool_log(env) == tools


@pytest.mark.parametrize("stage,verb", [(2, "remove-cluster"), (4, "uninstall")])
def test_hyperpod_alias_sequence_resumes_after_failed_exit_with_completed_journal(
    tmp_path, stage, verb
):
    cpu = "arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu-fixture"
    gpu = "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu-fixture"
    env = {
        **_install_fakes(tmp_path),
        "FAKE_CPU_ARN": cpu,
        "FAKE_GPU_ARN": gpu,
        "FAKE_CPU_EKS_ARN": CPU_ARN,
        "FAKE_GPU_EKS_ARN": GPU_ARN,
        "FAKE_CPU_HYPERPOD_ARN": cpu,
        "FAKE_GPU_HYPERPOD_ARN": gpu,
    }
    arguments = ("--cpu-cluster-arn", cpu, "--gpu-cluster-arn", gpu)
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_AFTER_WRITE": verb}, *arguments)
    assert failed.returncode != 0
    before = _record(tmp_path)
    assert before["stages"][str(stage)]["status"] == "FAIL"
    assert str(stage + 1) not in before["stages"]
    assert _admin_verbs(_tool_log(env))[-1] == verb
    resumed = _run(tmp_path, env, *arguments, "--stage", str(stage))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    assert all(value["status"] == "PASS" for value in after["stages"].values()), (
        "resuming the original HyperPod aliases must complete every lifecycle stage"
    )
    assert after["inputs"] == before["inputs"]
    assert after["inputs"]["cpu_cluster_arn_sha256"] == _sha256(cpu)
    assert after["inputs"]["gpu_cluster_arns_sha256"] == _sha256(gpu)
    assert after["stages"][str(stage)]["attempts"] == 2
    assert (
        after["stages"][str(stage)]["journal_baseline"]
        == before["stages"][str(stage)]["journal_baseline"]
    )
    for number in range(1, stage):
        assert after["stages"][str(number)] == before["stages"][str(number)]


def test_explicit_snapshot_skip_is_not_a_retain_snapshot_companion_pass(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot", "skip")

    assert result.returncode == 0, result.stdout + result.stderr
    assert any(
        "--reset-database --aurora-final-snapshot skip --confirm UNINSTALL_GPU_FAULT"
        in line
        for line in _tool_log(env)
    ), "only the explicitly approved policy reaches uninstall"
    record = _record(tmp_path)
    assert record["inputs"]["aurora_final_snapshot_sha256"] == _sha256("skip"), (
        "snapshot approval must be bound before the first stage"
    )
    stage = record["stages"]["4"]
    assert stage["status"] == "PASS" and stage["aurora_final_snapshot"] == "skip", (
        "the driver records completion of the selected skip branch"
    )
    assert stage["companion_cases"] == {
        "GF-REGIONAL-BOOT-027": {
            "status": "NOT_RUN",
            "reason": "Aurora final snapshot explicitly skipped",
        }
    }, "skip cannot satisfy BOOT-027's available final snapshot requirement"
    assert "GF-REGIONAL-BOOT-027: NOT_RUN" in result.stderr, (
        "the terminal summary must preserve the same evidence distinction"
    )


@pytest.mark.parametrize("policy", ["bogus", "", "RETAIN", "retain|skip"])
def test_invalid_snapshot_policy_never_reaches_a_tool(
    tmp_path: Path, policy: str
) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot", policy)

    assert result.returncode == 2 and "usage:" in result.stderr, (
        "unknown snapshot policy must fail argument validation"
    )
    assert _tool_log(env) == [], "invalid approval cannot start a lifecycle stage"
    assert not (tmp_path / "state").exists(), (
        "invalid approval cannot create acceptance state"
    )


def test_missing_snapshot_policy_value_is_a_usage_error(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)

    result = _run(tmp_path, env, "--aurora-final-snapshot")

    assert result.returncode == 2 and "usage:" in result.stderr, (
        "the snapshot option requires an explicit value"
    )
    assert _tool_log(env) == [], "a missing approval value cannot reach any tool"


@pytest.mark.parametrize(
    "policy,stage,verb",
    [
        ("retain", 3, "join-cluster"),
        ("retain", 4, "uninstall"),
        ("skip", 4, "uninstall"),
    ],
)
def test_snapshot_approval_is_immutable_across_driver_retries(
    tmp_path: Path, policy: str, stage: int, verb: str
) -> None:
    env = _install_fakes(tmp_path)
    failed = _run(
        tmp_path,
        {**env, "FAKE_ADMIN_FAIL_VERB": verb},
        "--aurora-final-snapshot",
        policy,
    )
    assert failed.returncode != 0, "the fixture must stop at its requested stage"
    path = tmp_path / "state/acceptance/admin-lifecycle-sequence.json"
    before, tools = path.read_bytes(), _tool_log(env)

    refused = _run(
        tmp_path,
        env,
        "--stage",
        str(stage),
        "--aurora-final-snapshot",
        "skip" if policy == "retain" else "retain",
    )

    assert refused.returncode != 0 and "input identity" in refused.stderr, (
        "resume must not change the approval even before uninstall starts"
    )
    assert path.read_bytes() == before, "refusal must preserve the original receipt"
    assert _tool_log(env) == tools, "policy drift must fail before any tool invocation"

    resumed = _run(
        tmp_path, env, "--stage", str(stage), "--aurora-final-snapshot", policy
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    assert after["inputs"] == json.loads(before)["inputs"], (
        "same-policy retry must retain every original input binding"
    )
    assert after["stages"]["4"]["aurora_final_snapshot"] == policy, (
        "the completed uninstall must prove the original approval"
    )


@pytest.mark.parametrize(
    "approved,reported", [("retain", "skip"), ("skip", "retain"), ("retain", None)]
)
def test_uninstall_completion_requires_the_approved_snapshot_policy(
    tmp_path: Path, approved: str, reported: str | None
) -> None:
    env = _install_fakes(tmp_path)
    if reported is None:
        env["FAKE_OMIT_UNINSTALL_SNAPSHOT_POLICY"] = "1"
    else:
        env["FAKE_UNINSTALL_SNAPSHOT_POLICY"] = reported

    result = _run(tmp_path, env, "--aurora-final-snapshot", approved)

    assert result.returncode != 0 and "approved snapshot policy" in result.stderr, (
        "a successful command exit cannot substitute for the selected journal policy"
    )
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], "policy mismatch must prevent the second deployment"
    record = _record(tmp_path)
    assert record["stages"]["4"]["status"] == "FAIL" and "5" not in record["stages"], (
        "mismatched or missing approval evidence cannot produce a stage PASS"
    )


@pytest.mark.parametrize("stage", [1, 5])
def test_failed_initial_deploy_resumes_only_its_bound_checkpoint(
    tmp_path: Path, stage: int
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / ("state" if stage == 1 else "state-second")
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_DEPLOY_STATE": str(state)})
    assert failed.returncode != 0, "the fixture must interrupt the selected deploy"
    checkpoint = state / "bootstrap-state.json"
    before = checkpoint.read_bytes()
    assert json.loads(before)["phase"] == "aws-infrastructure-ready", (
        "the interrupted first deployment must retain its actual bootstrap checkpoint"
    )
    assert not (state / "site.yaml").exists(), "the fixture is not an installed site"
    record = _record(tmp_path)
    first = record["stages"][str(stage)]
    log = tmp_path / "state/acceptance" / first["log_name"]
    original_log = log.read_bytes()
    tools = _tool_log(env)

    resumed = _run(tmp_path, env, "--stage", str(stage))

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = _record(tmp_path)
    current = after["stages"][str(stage)]
    assert current["status"] == "PASS" and current["attempts"] == 2, (
        "the original failed stage must be retried exactly once"
    )
    assert current["resumed_first_deploy"] is True, (
        "the receipt must distinguish an owned resume from a new cold deployment"
    )
    assert current["journal_baseline"] == first["journal_baseline"], (
        "retry cannot redefine its pre-deployment journal baseline"
    )
    assert after["inputs"] == record["inputs"], "all original inputs remain bound"
    assert log.read_bytes() == original_log, "the failed attempt log remains immutable"
    assert current["attempt_logs"][0] == {
        "name": log.name,
        "sha256": _sha256(original_log.decode()),
    }, "cold-build verification must retain the failed attempt's original proof"
    assert current["cold_build"]["cached_layers"] == 0, (
        "resuming must not bypass the uncached-build requirement"
    )
    new_tools = _tool_log(env)[len(tools) :]
    redeploy = next(
        index
        for index, line in enumerate(new_tools)
        if line.startswith("gpu-fault-admin deploy ")
    )
    assert not any(
        line.startswith(("docker ", "aws ecr ")) for line in new_tools[:redeploy]
    ), "a resumed deployment must not prune its already-started build or ECR tags"
    assert any(line.startswith("aws sagemaker ") for line in new_tools[:redeploy]), (
        "the resume still asks the product's discovery for each GPU ARN's provider"
    )
    assert f"--state-dir {state} " in new_tools[redeploy], (
        "the resume must use the same state directory"
    )
    if stage == 5:
        readme_tools = [
            line for line in new_tools if not line.startswith("aws sagemaker ")
        ]
        assert readme_tools[0] == "make deploy-host-setup-online", (
            "a resumed README deploy repeats the README procedure from its start "
            "(after the read-only provider discovery)"
        )
        readme = current["readme"]
        assert readme["attempt"] == 2 and readme["venv_preexisted"] is False, (
            "each attempt builds the venv in a fresh pristine copy"
        )
    for number in range(1, stage):
        assert after["stages"][str(number)] == record["stages"][str(number)], (
            "already passed stages cannot be rewritten by a later resume"
        )


@pytest.mark.parametrize("stage", [1, 5])
def test_installed_site_without_bootstrap_proof_is_not_a_fresh_deploy(
    tmp_path: Path, stage: int
) -> None:
    env = _install_fakes(tmp_path)
    state = tmp_path / ("state" if stage == 1 else "state-second")
    state.mkdir()
    site = state / "site.yaml"
    site.write_text("spec: {}\n", encoding="utf-8")

    result = _run(tmp_path, env)

    assert result.returncode != 0 and "already holds site.yaml" in result.stderr, (
        "missing bootstrap proof cannot authorize treating an installed site as fresh"
    )
    tools = _tool_log(env)
    assert _admin_verbs(tools) == (
        [] if stage == 1 else ["deploy", "remove-cluster", "join-cluster", "uninstall"]
    ), "the installed target must not receive another deploy"
    assert sum(line.startswith("docker buildx prune") for line in tools) == (
        0 if stage == 1 else 1
    ), "the refused deploy must not prune its target's build caches"
    assert site.read_text(encoding="utf-8") == "spec: {}\n", (
        "the installed site must be preserved for reconciliation"
    )


def test_unfinished_second_deploy_without_predecessor_receipts_cannot_resume(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    second = tmp_path / "state-second"
    second.mkdir()
    checkpoint = second / "bootstrap-state.json"
    checkpoint.write_text(
        json.dumps({"schema_version": 3, "phase": "aws-infrastructure-ready"}),
        encoding="utf-8",
    )
    before = checkpoint.read_bytes()

    result = _run(tmp_path, env, "--stage", "5")

    assert result.returncode != 0 and "preceding PASS receipts" in result.stderr, (
        "an unfinished bootstrap alone is not a BOOT-029 resume authorization"
    )
    assert _tool_log(env) == [], "unbound resume must not reach any tool"
    assert checkpoint.read_bytes() == before, "unbound checkpoints must not be changed"


def _install_bound_admin(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    venv = state / "deployer-venv"
    (venv / "bin").mkdir(parents=True)
    binding = venv / "gpu-fault-managed-state-dir.json"
    binding.write_text(
        json.dumps({"schema_version": 1, "state_dir": str(state.resolve())}),
        encoding="utf-8",
    )
    binding.chmod(0o600)
    admin = venv / "bin" / "gpu-fault-admin"
    body = FAKE_ADMIN.replace(
        "printf 'gpu-fault-admin %s\\n' \"$*\"", "printf 'bound-admin %s\\n' \"$*\""
    ).replace(
        'verb="$1"; shift',
        "[[ ! -v PYTHONPATH && ! -v PYTHONHOME && ! -v GPU_FAULT_REPOSITORY_ROOT "
        "&& ! -v GPU_FAULT_REPO_ROOT ]] || exit 31\n"
        '[[ "${PATH%%:*}" == "${0%/*}" ]] || exit 32\n'
        'verb="$1"; shift',
    )
    admin.write_text(body, encoding="utf-8")
    admin.chmod(0o700)
    return admin


def test_middle_stages_use_the_bound_cli_without_checkout_overrides(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    _install_bound_admin(tmp_path)
    env["GPU_FAULT_REPOSITORY_ROOT"] = str(tmp_path / "foreign-source")
    env["GPU_FAULT_REPO_ROOT"] = str(tmp_path / "foreign-source")

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert [line.split()[1] for line in lines if line.startswith("bound-admin ")] == [
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], lines
    assert _admin_verbs(lines) == ["deploy", "deploy"], (
        "cold deploys must still use the source-preparing PATH CLI"
    )
    assert {
        number: stage["status"] for number, stage in _record(tmp_path)["stages"].items()
    } == {str(number): "PASS" for number in range(1, 6)}


@pytest.mark.parametrize("damage", ["binding", "executable", "dangling-binding"])
def test_broken_bound_installation_never_falls_back_to_the_checkout(
    tmp_path: Path, damage: str
) -> None:
    env = _install_fakes(tmp_path)
    admin = _install_bound_admin(tmp_path)
    binding = admin.parent.parent / "gpu-fault-managed-state-dir.json"
    if damage == "binding":
        binding.write_text("{", encoding="utf-8")
    elif damage == "executable":
        admin.chmod(0o600)
    else:
        binding.unlink()
        binding.symlink_to(tmp_path / "missing.json")

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout + result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == ["deploy"], "no mutating stage may use the checkout"
    assert not any(line.startswith("bound-admin ") for line in lines), lines
    stages = _record(tmp_path)["stages"]
    assert {number: stage["status"] for number, stage in stages.items()} == {
        "1": "PASS",
        "2": "FAIL",
    }, stages


def test_failed_bound_command_stops_the_sequence_and_preserves_resume_guards(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    _install_bound_admin(tmp_path)
    env["FAKE_ADMIN_FAIL_VERB"] = "join-cluster"

    failed = _run(tmp_path, env)

    assert failed.returncode != 0, failed.stdout + failed.stderr
    assert {
        number: stage["status"] for number, stage in _record(tmp_path)["stages"].items()
    } == {"1": "PASS", "2": "PASS", "3": "FAIL"}
    original_inputs = _record(tmp_path)["inputs"]
    env.pop("FAKE_ADMIN_FAIL_VERB")
    resumed = _run(tmp_path, env, "--stage", "3")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _record(tmp_path)["inputs"] == original_inputs
    lines = _tool_log(env)
    assert [line.split()[1] for line in lines if line.startswith("bound-admin ")] == [
        "remove-cluster",
        "join-cluster",
        "join-cluster",
        "uninstall",
    ], lines
    assert _admin_verbs(lines) == ["deploy", "deploy"]


def test_stage_five_refuses_an_installed_site_even_with_a_bootstrap_checkpoint(
    tmp_path: Path,
) -> None:
    """A second state directory holding both bootstrap-state.json and site.yaml
    is an installed site, not the unfinished first deploy the driver may resume
    (that resume is bound to the receipt of this run's own failed attempt); the
    stage-5 deploy must not run against it and the site stays untouched."""

    env = _install_fakes(tmp_path)
    installed = tmp_path / "state-second"
    installed.mkdir()
    checkpoint = installed / "bootstrap-state.json"
    checkpoint.write_text(
        json.dumps({"schema_version": 3, "phase": "site-ready"}), encoding="utf-8"
    )
    site = installed / "site.yaml"
    site.write_text("spec: {}\n", encoding="utf-8")
    before = (checkpoint.read_bytes(), site.read_bytes())

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "already holds bootstrap-state.json" in result.stderr, result.stderr
    assert not any(
        line.startswith("gpu-fault-admin deploy ")
        and f"--state-dir {installed} " in line
        for line in _tool_log(env)
    ), "the installed target must not receive another deploy"
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ], "stages 1-4 ran against the first state directory only"
    stages = _record(tmp_path)["stages"]
    assert {number: stage["status"] for number, stage in stages.items()} == {
        "1": "PASS",
        "2": "PASS",
        "3": "PASS",
        "4": "PASS",
    }, "the refusal happens before a stage-5 attempt is opened"
    assert (checkpoint.read_bytes(), site.read_bytes()) == before, (
        "the installed site must be preserved for reconciliation"
    )


def _readme_lines(lines: list[str], prefix: str) -> list[str]:
    return [line[len(prefix) :] for line in lines if line.startswith(prefix)]


def test_stage_five_deploys_by_the_readme_procedure_from_a_pristine_copy(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    checkout = tmp_path / "checkout"
    readme_text = (checkout / "README.md").read_text(encoding="utf-8")

    result = _run(tmp_path, env, "--email-wait-minutes", "3")

    assert result.returncode == 0, result.stdout + result.stderr
    lines = _tool_log(env)
    deploys = [index for index, line in enumerate(lines) if "admin deploy " in line]
    prunes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("docker buildx prune")
    ]
    makes = [
        index
        for index, line in enumerate(lines)
        if line == "make deploy-host-setup-online"
    ]
    assert len(makes) == 1 and prunes[1] < makes[0] < deploys[1], (
        "stage 5 runs the README's make target after its cache prune and before"
        " the README deploy; stage 1 does not touch make"
    )
    assert "--wait-for-email-confirmation 3" in lines[deploys[0]], (
        "--email-wait-minutes still reaches the stage-1 deploy"
    )
    assert "--wait-for-email-confirmation" not in lines[deploys[1]], (
        "the README command has no wait and the stage adds none"
    )
    assert lines[deploys[1]].split()[2:] == [
        "--cpu-cluster-arn",
        CPU_ARN,
        "--gpu-cluster-arn",
        GPU_ARN,
        "--state-dir",
        str(tmp_path / "state-second"),
        "--admin-email",
        ADMIN_EMAIL,
    ], "the README deploy carries exactly the four rendered arguments"
    (names,) = _readme_lines(lines, "readme-admin-env ")
    (path,) = _readme_lines(lines, "readme-admin-path ")
    (cwd,) = _readme_lines(lines, "readme-admin-cwd ")
    incidental = {"PATH", "VIRTUAL_ENV", "PWD", "OLDPWD", "SHLVL", "_"}
    for name in set(names.split()) - incidental:
        assert name in boot029_readme.KEPT_NAMES or name.startswith(
            boot029_readme.KEPT_PREFIXES
        ), f"{name} leaked from the driver into the README deploy"
    assert "GPU_FAULT_SENTINEL" not in names and "AWS_REGION" not in names
    assert "PYTHONPATH" not in names and "BUILDKIT_PROGRESS" not in names
    attempt_dir = tmp_path / "state/acceptance/stage-5-readme/attempt-1"
    operator = boot029_readme.operator_path(env["PATH"], attempt_dir / "operator-path")
    assert path == f"{cwd}/.venv/bin:{operator}", (
        "only the README's activation extends the operator's filtered PATH"
    )
    assert str(tmp_path / "bin") not in operator.split(os.pathsep), (
        "the directory offering the driver's gpu-fault-admin is replaced by its"
        " stand-in, so the driver's CLI is not on the README's PATH"
    )
    for entry in operator.split(os.pathsep):
        assert not (Path(entry) / "gpu-fault-admin").exists(), (
            f"{entry} still offers a gpu-fault-admin to the README procedure"
        )
    copy = Path(cwd)
    assert copy.is_relative_to(
        tmp_path / "state/acceptance/stage-5-readme/attempt-1"
    ), "the pristine copy lives in this attempt's stage area"
    assert copy != checkout, "the README procedure never runs inside --repo"
    assert (copy / "README.md").read_text(encoding="utf-8") == readme_text
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=copy, capture_output=True, text=True
        ).stdout
        == ""
    ), "the venv the README creates is ignored; the copy stays clean"
    assert not (checkout / ".venv").exists(), "--repo receives no venv"

    stage = _record(tmp_path)["stages"]["5"]
    assert stage["status"] == "PASS" and stage["cold_build"]["cached_layers"] == 0
    receipt = stage["readme"]
    assert receipt["readme_sha256"] == _sha256(readme_text)
    assert receipt["executed_lines"][:2] == [
        "make deploy-host-setup-online",
        ". .venv/bin/activate",
    ]
    assert receipt["executed_lines"][2].startswith("gpu-fault-admin deploy"), receipt[
        "executed_lines"
    ]
    assert f"sha256:{_sha256(CPU_ARN)}" in receipt["executed_lines"][2]
    assert (
        f"sha256:{_sha256(str(tmp_path / 'state-second'))}"
        in (receipt["executed_lines"][2])
    )
    assert receipt["skipped_verification_lines"] == [
        "make PYTHON=.venv/bin/python check"
    ]
    assert receipt["venv_preexisted"] is False and receipt["attempt"] == 1
    assert receipt["source_snapshot"] is False
    assert (
        receipt["source_commit"]
        == receipt["source_base_commit"]
        == _git(checkout, "rev-parse", "HEAD")
    )
    assert receipt["path_sha256"] == _sha256(operator)
    assert receipt["path_entries_shadowed"] >= 1, (
        "the fake admin directory was replaced by a stand-in"
    )
    assert receipt["path_entries_dropped"] >= 0, "the dropped count is recorded"
    assert "path" not in receipt, "PATH entries never appear in clear"


def test_stage_five_carries_uncommitted_checkout_changes_into_the_copy(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    checkout = tmp_path / "checkout"
    head = _git(checkout, "rev-parse", "HEAD")
    readme_path = checkout / "README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8").replace(
            ". .venv/bin/activate\n",
            ". .venv/bin/activate\necho readme-fixture-marker\n",
        ),
        encoding="utf-8",
    )
    (checkout / "untracked-marker.txt").write_text("carried\n", encoding="utf-8")

    result = _run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    stage = _record(tmp_path)["stages"]["5"]
    log = (tmp_path / "state/acceptance" / stage["log_name"]).read_text()
    assert "readme-fixture-marker" in log, (
        "the uncommitted README line is executed verbatim from the pristine copy"
    )
    receipt = stage["readme"]
    assert receipt["executed_lines"][:3] == [
        "make deploy-host-setup-online",
        ". .venv/bin/activate",
        "echo readme-fixture-marker",
    ]
    assert receipt["source_snapshot"] is True
    assert receipt["source_base_commit"] == head != receipt["source_commit"], (
        "a dirty checkout is deployed as its own snapshot commit"
    )
    (cwd,) = _readme_lines(_tool_log(env), "readme-admin-cwd ")
    assert (Path(cwd) / "untracked-marker.txt").read_text() == "carried\n"
    assert "untracked-marker.txt" in _git(checkout, "status", "--porcelain"), (
        "the checkout itself stays dirty and untouched"
    )


def test_stage_five_fails_when_the_readme_no_longer_states_the_procedure(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    readme_path = tmp_path / "checkout/README.md"
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8").replace(
            boot029_readme.DEPLOY_HEADING + "\n", "## 部署\n"
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path, env)

    assert result.returncode != 0, result.stdout
    assert "FAILED at stage 5 (cold-redeploy-readme)" in result.stderr, result.stderr
    assert "README lacks the heading" in result.stderr, result.stderr
    lines = _tool_log(env)
    assert _admin_verbs(lines) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
    ]
    assert "make deploy-host-setup-online" not in lines, (
        "a README that lost its procedure runs nothing"
    )
    stage = _record(tmp_path)["stages"]["5"]
    assert stage["status"] == "FAIL" and "readme" not in stage
    assert not (tmp_path / "state-second/bootstrap-state.json").exists(), (
        "no deploy reaches the second state directory"
    )
