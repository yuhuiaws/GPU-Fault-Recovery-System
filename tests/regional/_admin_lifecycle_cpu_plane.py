"""The CPU control plane the BOOT-029 driver suite's fake ``kubectl`` answers.

``admin-lifecycle-sequence.sh`` reads the CPU cluster around every join-cluster
and remove-cluster (``scripts/e2e/regional/boot029_registry_drift.py``):
``get deployment``/``get pod`` right before the verb and each Running Pod's
``/healthz`` (``exec``) right after it. The fake answers from the shared tool
log's directory: the fake ``gpu-fault-admin`` bumps ``registry-epoch`` after
every completed join-cluster or remove-cluster, and each epoch is one
membership change.

The default is the product as designed since batch 245: api-ha and
telemetry-spool-worker keep their Pods; the control-worker rolls to a new
ReplicaSet per verb because its Pod template gets the new failure-domain map
digest (the previous epoch's first worker Pod is still terminating); every Pod
reports ``secret_source=mounted-file`` without drift. Knobs:

* ``FAKE_REGISTRY_DRIFT_READS=N`` -- each Pod drifts for its first N
  ``/healthz`` reads of an epoch (kubelet re-projecting the Secret), ``always``
  -- it never converges;
* ``FAKE_REGISTRY_GENERATION=pre-secret-source`` -- a product before batch 245:
  no ``secret_source`` key, and every Pod that did not roll compares the
  Secret it started with, so it drifts after the first verb for good;
* ``FAKE_CPU_POD_REPLACED=<app>`` -- every verb replaces that role's first Pod;
* ``FAKE_WORKER_ROLL_WITHOUT_MAP=1`` -- the control-worker rolls while its
  template keeps the same failure-domain map digest;
* ``FAKE_KUBECTL_FAIL=<verb>`` -- that kubectl verb exits non-zero.

Every call is logged as ``kubectl <verb> <resource>`` (the in-Pod client
script is never logged); any verb but ``get`` and ``exec`` is refused.
"""

from __future__ import annotations

from gpu_fault.failure_domains import FAILURE_DOMAIN_MAP_ANNOTATION

FAKE_KUBECTL = (
    r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys

args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_TOOL_LOG"])
kubeconfig = args[args.index("--kubeconfig") + 1]
namespace = args[args.index("-n") + 1]
rest = args[args.index("-n") + 2:]
verb = rest[0]
with log.open("a") as stream:
    stream.write("kubectl " + " ".join(rest[:3] if verb == "exec" else rest) + "\n")
if not kubeconfig.endswith("/cpu.kubeconfig") or namespace != "gpu-fault-system":
    print("fixture: not the CPU control plane site.yaml names", file=sys.stderr)
    sys.exit(3)
if verb not in {"get", "exec"}:
    print("fixture: the registry check only reads the CPU cluster", file=sys.stderr)
    sys.exit(3)
if verb == os.environ.get("FAKE_KUBECTL_FAIL"):
    print("error: injected kubectl failure", file=sys.stderr)
    sys.exit(1)
epoch_file = log.parent / "registry-epoch"
epoch = int(epoch_file.read_text()) if epoch_file.exists() else 0
WORKER = "gpu-fault-control-worker"
REPLICAS = {"gpu-fault-api-ha": 2, WORKER: 2, "gpu-fault-telemetry-spool-worker": 1}
replaced = os.environ.get("FAKE_CPU_POD_REPLACED", "")
old_generation = os.environ.get("FAKE_REGISTRY_GENERATION") == "pre-secret-source"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def uid(text):
    value = digest("uid/" + text)
    return "-".join((value[:8], value[8:12], value[12:16], value[16:20], value[20:32]))


def pod(app, index, revision, terminating=False):
    name = app + "-" + digest(app + "/" + str(revision))[:10] + "-" + str(index)
    if app == replaced and index == 0 and epoch:
        name += "-r" + str(epoch)
    metadata = {"name": name, "uid": uid(name), "labels": {"app": app},
        "creationTimestamp": "2030-01-01T00:00:00Z"}
    if terminating:
        metadata["deletionTimestamp"] = "2030-01-01T00:05:00Z"
    return {"metadata": metadata,
        "status": {"phase": "Running",
            "containerStatuses": [{"name": app, "restartCount": 0}]},
        "spec": {"containers": [{"ports": [
            {"name": "livez", "containerPort": 8090},
            {"name": "http", "containerPort": 8080}]}]}}


def pods():
    items = []
    for app, count in REPLICAS.items():
        revision = epoch if app == WORKER else 0
        items.extend(pod(app, index, revision) for index in range(count))
    if epoch:
        items.append(pod(WORKER, 0, epoch - 1, terminating=True))
    return items


def deployment(app):
    annotations = {"gpu-fault.io/artifact-sha256": digest("artifact")}
    if app == WORKER:
        revision = 0 if os.environ.get("FAKE_WORKER_ROLL_WITHOUT_MAP") else epoch
        annotations["'''
    + FAILURE_DOMAIN_MAP_ANNOTATION
    + r""""] = digest("map/" + str(revision))
    return {"metadata": {"name": app, "uid": uid("deployment/" + app)},
        "spec": {"template": {"metadata": {"labels": {"app": app},
            "annotations": annotations},
            "spec": {"containers": [{"name": app}]}}}}


def healthz(name):
    counter = log.parent / ("healthz-" + str(epoch) + "-" + name)
    reads = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(reads))
    setting = os.environ.get("FAKE_REGISTRY_DRIFT_READS", "0")
    head = digest("head/" + str(epoch))
    if old_generation:
        drifting = epoch > 0 and not name.startswith(WORKER)
        secret = digest("head/0") if drifting else head
        return {"ready": True, "secret_drift": drifting, "secret_config_sha256": secret}
    drifting = epoch > 0 and (setting == "always" or reads <= int(setting))
    return {"ready": True, "secret_drift": drifting,
        "secret_config_sha256": digest("head/" + str(epoch - 1)) if drifting else head,
        "secret_source": "mounted-file", "secret_startup_config_sha256": digest("head/0"),
        "secret_error": None, "durable_config_sha256": head}


if verb == "exec":
    name = rest[2]
    print(json.dumps({"http_status": 200,
        "payload": {"status": "ok", "regional_registry": healthz(name)}}))
elif rest[:2] == ["get", "deployment"]:
    names = [item for item in rest[2:] if not item.startswith("-") and item != "json"]
    print(json.dumps({"kind": "List", "items": [deployment(app) for app in names]}))
elif rest[:2] == ["get", "pod"]:
    print(json.dumps({"kind": "List", "items": pods()}))
else:
    print("fixture: unexpected kubectl get " + " ".join(rest[1:2]), file=sys.stderr)
    sys.exit(3)
"""
)
