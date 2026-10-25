"""GF-GENERAL-LIVE managed jobs: a workload name per job, and a cleanup that removes it.

Live 2026-10-25 (plain-EKS lane, live pass 4): every managed-job variant of 003
(``gpu_idle``, ``rank_cpu_idle``, ``ambiguous_ownership``) failed before any fault
with "managed workload name or job identity already exists", as in the pass
before. In the way was a Suspended PyTorchJob ``gpu-fault-single-node-warm-spare``
without Pods (job-id ``gl28nz-nz-a1-c5ab1930``, attempt ``...-a001-r-8856497f``)
that pass 3's 028.nonzero_exit had left: the product restarted that job in place
-- its UID, a derived attempt id -- and the case's cleanup refused it ("managed
workload ownership or UID changed") before deleting anything, because a
journaled fixture kept the strict adoption rule and can bind no restart. Every
managed job rendered the manifest's fixed name, so that one object refused all
of them whatever their job id, and the refusal named nothing.

The runner, the job wrappers, the image prewarm, the managed workload fixture
and its journals are the runner's own here; only kubectl is a fake: one
in-memory namespace whose training operator gives a PyTorchJob its replica Pods
and Services (and orphans them on an Orphan delete, as the garbage collector
does), and whose product restarts a job in place the way
``_apply_workload_mutation`` patches it.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_attempts as attempts
from scripts.e2e.regional import general_live_nccl_job as nccl_job
from scripts.e2e.regional import general_live_progress as progress
from scripts.e2e.regional import general_live_utilization as utilization
from scripts.e2e.regional import run_general_live as runner
from scripts.e2e.regional.general_live_fixture import (
    Evidence,
    GeneralLiveSettings,
    ProbeHandle,
    VariantContext,
    run_identity,
)
from scripts.e2e.regional.managed_workload_fixture import (
    MANAGED_LABEL,
    NOTHING_CREATED,
    OWNER_LABEL,
    TRAINING_IMAGE,
    WORKLOAD_NAME_LIMIT,
    ManagedJobCleanupError,
    ManagedWorkloadFixture,
    WorkloadOwnershipError,
    managed_workload_name,
    occupant,
)
from scripts.e2e.regional.regional_commands import (
    RegionalCommandFailed,
    RegionalFixtureError,
)
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from tests.regional.test_general_live_managed_job_start import (
    EXECUTOR,
    NAMESPACE,
    NODE,
    PEER,
    REGISTRY,
    configured,
    probe_cleanup,
    probe_execute,
)

SINGLE_NODE = "gpu-fault-single-node-warm-spare"
TWO_NODE = "gpu-fault-two-node-replace"
#: The product's restart suffix of the live leftover's attempt id.
RESTART = "-r-8856497f"
RESTART_OPERATION = "workflow-restart/0/RESTART_WORKLOAD"
JOB_LABEL = "gpu-fault.io/job-id"
ATTEMPT_LABEL = "gpu-fault.io/attempt-id"
PLURALS = {
    "pods": "pod",
    "jobs": "job",
    "pytorchjobs": "pytorchjob",
    "services": "service",
}
#: Each case's managed-job wrapper, built as its variants build it.
WRAPPERS = {
    "GF-GENERAL-LIVE-003": lambda ctx, name: utilization.ManagedIdleJob(ctx, name=name),
    "GF-GENERAL-LIVE-011": lambda ctx, name: nccl_job.ManagedNcclJob(ctx, name=name),
    "GF-GENERAL-LIVE-027": lambda ctx, name: progress.ManagedProgressJob(
        ctx, name=name
    ),
    "GF-GENERAL-LIVE-028": lambda ctx, name: attempts.ManagedFailJob(
        ctx, name=name, mode="exit:1"
    ),
}
BASES = {
    "GF-GENERAL-LIVE-003": SINGLE_NODE,
    "GF-GENERAL-LIVE-011": TWO_NODE,
    "GF-GENERAL-LIVE-027": SINGLE_NODE,
    "GF-GENERAL-LIVE-028": SINGLE_NODE,
}
CASE_VARIANTS = {
    "GF-GENERAL-LIVE-003": "GF-GENERAL-LIVE-003.gpu_idle",
    "GF-GENERAL-LIVE-011": "GF-GENERAL-LIVE-011.active_to_zero",
    "GF-GENERAL-LIVE-027": "GF-GENERAL-LIVE-027.nan",
    "GF-GENERAL-LIVE-028": "GF-GENERAL-LIVE-028.nonzero_exit",
}


@pytest.fixture(autouse=True)
def private_umask() -> Iterator[None]:
    """``run_general_live.main`` runs under 0o077; the journals require it."""

    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def gpu_total(job: dict[str, Any]) -> int:
    total = 0
    for replica in job["spec"]["pytorchReplicaSpecs"].values():
        for container in replica["template"]["spec"]["containers"]:
            limits = (container.get("resources") or {}).get("limits") or {}
            total += int(replica.get("replicas", 1)) * int(
                limits.get("nvidia.com/gpu", 0)
            )
    return total


class Cluster:
    """kubectl over one in-memory namespace with a training operator."""

    def __init__(self, *, cached: bool = True) -> None:
        self.cached = cached
        self.objects: dict[tuple[str, str], dict[str, Any]] = {
            ("deployment", EXECUTOR["metadata"]["name"]): copy.deepcopy(EXECUTOR)
        }
        self.requests: list[tuple[str, ...]] = []
        self.deletes: list[tuple[str, str, str]] = []
        self.serial = 0

    # -- kubectl -------------------------------------------------------------
    def run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: float = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "kubectl", command
        assert command[command.index("-n") + 1] == NAMESPACE, command
        arguments = tuple(command[command.index("-n") + 2 :])
        if "--context" not in command:
            assert arguments[:3] == (
                "get",
                "configmap",
                "gpu-fault-regional-release-state",
            ), f"the only CPU read is the release state: {arguments}"
            state = {
                "phase": "complete",
                "transaction_committed": True,
                "release_id": "rel-test",
            }
            stdout = json.dumps({"data": {"state.json": json.dumps(state)}})
            return subprocess.CompletedProcess(command, 0, stdout, "")
        self.requests.append(arguments)
        verb = arguments[0]
        if arguments == ("api-resources", "-o", "name"):
            stdout = "pods\njobs.batch\npytorchjobs.kubeflow.org\nservices\n"
        elif verb == "get":
            stdout = self.get(list(arguments[1:]))
        elif verb == "create":
            assert input_text is not None, arguments
            document = json.loads(input_text)
            stdout = self.create(document, acknowledge="-o" in arguments)
        elif verb == "delete":
            assert arguments[1] == "--raw" and input_text is not None, arguments
            stdout = self.delete(arguments[2], json.loads(input_text))
        elif verb == "logs":
            stdout = self.logs(arguments[1])
        elif verb == "wait":
            stdout = ""  # the prewarm Pod pulled the image and exited
        else:
            raise AssertionError(f"unexpected kubectl call: {arguments}")
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def get(self, arguments: list[str]) -> str:
        assert arguments[-2:] == ["-o", "json"], arguments
        ignore = "--ignore-not-found" in arguments
        arguments = [item for item in arguments[:-2] if item != "--ignore-not-found"]
        if arguments == ["node"]:
            images = [{"names": [TRAINING_IMAGE]}] if self.cached else []
            nodes = [
                {"metadata": {"name": node}, "status": {"images": images}}
                for node in (NODE, PEER)
            ]
            return json.dumps({"items": nodes})
        if "-l" in arguments:
            kinds, _flag, selector = arguments
            key, value = selector.split("=", 1)
            items = [
                item
                for (kind, _name), item in sorted(self.objects.items())
                if kind in kinds.split(",")
                and (item["metadata"].get("labels") or {}).get(key) == value
            ]
            return json.dumps({"kind": "List", "items": items})
        kind, name = arguments
        item = self.objects.get((kind, name))
        if item is None:
            if ignore:
                return ""
            raise RegionalCommandFailed(1, f'{kind} "{name}" not found')
        return json.dumps(item)

    def create(self, document: dict[str, Any], *, acknowledge: bool) -> str:
        kind, name = document["kind"].lower(), document["metadata"]["name"]
        if (kind, name) in self.objects:
            raise RegionalCommandFailed(1, f'{kind} "{name}" already exists')
        stored = self.store(document)
        if kind == "pytorchjob":
            self.reconcile(stored)
        return json.dumps(stored) if acknowledge else f"{kind}/{name} created"

    def delete(self, path: str, options: dict[str, Any]) -> str:
        plural, name = path.split("/")[-2:]
        kind = PLURALS[plural]
        current = self.objects.get((kind, name))
        if current is None:
            raise RegionalCommandFailed(1, f'{kind} "{name}" not found')
        metadata = current["metadata"]
        if options["preconditions"] != {
            "uid": metadata["uid"],
            "resourceVersion": metadata["resourceVersion"],
        }:
            raise RegionalCommandFailed(1, "Precondition failed: UID in precondition")
        del self.objects[(kind, name)]
        policy = options["propagationPolicy"]
        self.deletes.append((kind, name, policy))
        for key, item in list(self.objects.items()):
            owners = item["metadata"].get("ownerReferences") or []
            if not any(owner.get("uid") == metadata["uid"] for owner in owners):
                continue
            if policy == "Foreground":
                del self.objects[key]
                continue
            # Orphan: the garbage collector strips the deleted owner.
            kept = [owner for owner in owners if owner.get("uid") != metadata["uid"]]
            item["metadata"].pop("ownerReferences")
            if kept:
                item["metadata"]["ownerReferences"] = kept
            self.bump(item)
        return ""

    def logs(self, name: str) -> str:
        pod = self.objects[("pod", name)]
        owner = pod["metadata"]["ownerReferences"][0]["name"]
        world = gpu_total(self.objects[("pytorchjob", owner)])
        return (
            f"HEARTBEAT rank=0 step=1 node={pod['spec'].get('nodeName')} "
            f"world_size={world} all_reduce={world * (world + 1) / 2} elapsed=2.0\n"
        )

    # -- API server, training operator and product ----------------------------
    def store(self, document: dict[str, Any]) -> dict[str, Any]:
        self.serial += 1
        value = copy.deepcopy(document)
        value["metadata"].update(
            namespace=NAMESPACE,
            uid=f"uid-{self.serial}-{value['metadata']['name']}",
            resourceVersion=str(self.serial),
        )
        self.objects[(value["kind"].lower(), value["metadata"]["name"])] = value
        return value

    def bump(self, document: dict[str, Any]) -> None:
        self.serial += 1
        document["metadata"]["resourceVersion"] = str(self.serial)

    def reconcile(self, job: dict[str, Any]) -> None:
        """The operator: replica Pods and Services, none while suspended."""

        uid, name = job["metadata"]["uid"], job["metadata"]["name"]
        for key, item in list(self.objects.items()):
            owners = item["metadata"].get("ownerReferences") or []
            if key[0] in {"pod", "service"} and any(
                owner.get("uid") == uid for owner in owners
            ):
                del self.objects[key]
        if (job["spec"].get("runPolicy") or {}).get("suspend") is True:
            return
        owner = {
            "apiVersion": job["apiVersion"],
            "kind": job["kind"],
            "name": name,
            "uid": uid,
            "controller": True,
            "blockOwnerDeletion": True,
        }
        for role, replica in job["spec"]["pytorchReplicaSpecs"].items():
            template = replica["template"]
            for index in range(int(replica.get("replicas", 1))):
                replica_name = f"{name}-{role.lower()}-{index}"
                operator_labels = {
                    "training.kubeflow.org/job-name": name,
                    "training.kubeflow.org/replica-type": role.lower(),
                }
                containers = template["spec"]["containers"]
                self.store(
                    {
                        "apiVersion": "v1",
                        "kind": "Pod",
                        "metadata": {
                            "name": replica_name,
                            "labels": {
                                **template["metadata"]["labels"],
                                **operator_labels,
                            },
                            "annotations": dict(
                                template["metadata"].get("annotations") or {}
                            ),
                            "ownerReferences": [dict(owner)],
                        },
                        "spec": copy.deepcopy(template["spec"]),
                        "status": {
                            "phase": "Running",
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "containerStatuses": [
                                {"name": item["name"], "ready": True}
                                for item in containers
                            ],
                        },
                    }
                )
                self.store(
                    {
                        "apiVersion": "v1",
                        "kind": "Service",
                        "metadata": {
                            "name": replica_name,
                            "labels": operator_labels,
                            "ownerReferences": [dict(owner)],
                        },
                        "spec": {"clusterIP": "None"},
                    }
                )

    def restart_in_place(self, name: str, *, then_suspend: bool) -> dict[str, Any]:
        """RESTART_WORKLOAD of a job that is not terminal, as the product patches it.

        The same object (UID) gets a derived attempt id on itself and its Pod
        template and the restart annotations, and runs again;
        ``then_suspend`` is the STOP_WORKLOADS after the restarted attempt
        failed too -- the Suspended job without Pods the live pass left.
        """

        job = self.objects[("pytorchjob", name)]
        source = job["metadata"]["labels"][ATTEMPT_LABEL]
        target = f"{source[:52]}{RESTART}"
        job["metadata"]["labels"][ATTEMPT_LABEL] = target
        job["metadata"].setdefault("annotations", {}).update(
            {
                "gpu-fault.io/operation-id": RESTART_OPERATION,
                "gpu-fault.io/restart-count": "1",
            }
        )
        for replica in job["spec"]["pytorchReplicaSpecs"].values():
            template = replica["template"]["metadata"]
            template["labels"][ATTEMPT_LABEL] = target
            template.setdefault("annotations", {})["gpu-fault.io/restart-count"] = "1"
        job["spec"].setdefault("runPolicy", {})["suspend"] = then_suspend
        self.bump(job)
        self.reconcile(job)
        return job

    def leftover(self, name: str, *, job_id: str, owner: str) -> dict[str, Any]:
        """A Suspended managed PyTorchJob another acceptance run left, no Pods."""

        return self.store(
            {
                "apiVersion": "kubeflow.org/v1",
                "kind": "PyTorchJob",
                "metadata": {
                    "name": name,
                    "labels": {
                        MANAGED_LABEL: "true",
                        OWNER_LABEL: owner,
                        JOB_LABEL: job_id,
                        ATTEMPT_LABEL: f"{job_id}-a001{RESTART}",
                    },
                },
                "spec": {"runPolicy": {"suspend": True}, "pytorchReplicaSpecs": {}},
                "status": {"conditions": [{"type": "Suspended", "status": "True"}]},
            }
        )

    def names_requested(self) -> set[str]:
        """Every bare word and selector value a GPU-side request carried."""

        return {
            part.split("=", 1)[-1].rsplit("/", 1)[-1]
            for request in self.requests
            for part in request
        }

    def kinds(self) -> set[str]:
        return {kind for kind, _name in self.objects}

    def children(self) -> dict[str, str]:
        """``kind/name`` -> UID of every Pod and Service now in the namespace."""

        return {
            f"{kind}/{name}": item["metadata"]["uid"]
            for (kind, name), item in self.objects.items()
            if kind in {"pod", "service"}
        }


def install(monkeypatch: pytest.MonkeyPatch, cluster: Cluster) -> None:
    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(cluster.run))
    monkeypatch.setattr(ProbeHandle, "execute", probe_execute)
    monkeypatch.setattr(ProbeHandle, "cleanup", probe_cleanup)


def variant_context(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: Cluster
) -> VariantContext:
    """The context ``execute_variant`` builds, over the runner's own settings."""

    spec = REGISTRY[key]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    install(monkeypatch, cluster)
    case_dir = run_dir / "cases" / spec.key
    return VariantContext(
        spec=spec,
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        run_id=run_identity(spec.case_id, spec.variant, 1, run_dir),
        attempt=1,
        deadline=datetime.now(timezone.utc) + timedelta(hours=3),
        evidence=Evidence(case_dir),
    )


def execute(key: str, settings: GeneralLiveSettings, run_dir: Path) -> dict[str, Any]:
    deadline = datetime.now(timezone.utc) + timedelta(hours=3)
    return runner.execute_variant(
        REGISTRY[key], settings, run_dir, 1, deadline, {"errors": []}
    )


def journal(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


# --------------------------------------------------------------------------- #
# one name per job
# --------------------------------------------------------------------------- #
def test_the_workload_name_is_the_manifest_name_and_a_digest_of_the_job_id() -> None:
    job_ids = [
        "gl3-idle-1",
        "gl3-idle-2",
        "gl3-attA-1",
        "gl3-atta-1",  # a label value may differ from another only in case
        "gl28nz-nz-a1-c5ab1930",
        run_identity("GF-GENERAL-LIVE-027", "x" * 40, 9, Path("/tmp/a")),
    ]
    names = [managed_workload_name(SINGLE_NODE, job_id) for job_id in job_ids]
    assert len(set(names)) == len(job_ids), names
    assert names == [managed_workload_name(SINGLE_NODE, job) for job in job_ids], (
        "the name is a function of the job id: plan, execute and cleanup agree"
    )
    for name in [*names, managed_workload_name(TWO_NODE, "gl11-x-a1-00000000")]:
        assert name.startswith(("gpu-fault-single-node-warm-spare-", TWO_NODE)), name
        assert len(name) <= WORKLOAD_NAME_LIMIT, name
        assert name.islower() and name[0].isalpha() and name[-1].isalnum(), name
        # The product's terminal restart copy and its training-operator
        # replica Service stay inside the 63-character DNS-1035 label.
        assert len(f"{name[:52]}{RESTART}-master-0") <= 63, name
    for base, job_id in (("", "gl3-idle-1"), (SINGLE_NODE, ""), ("9-x", "j")):
        with pytest.raises(ValueError, match="cannot name a managed workload"):
            managed_workload_name(base, job_id)


@pytest.mark.parametrize("case", sorted(WRAPPERS))
def test_every_wrapper_names_its_workload_after_its_job_id(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = Cluster()
    ctx = variant_context(CASE_VARIANTS[case], tmp_path, monkeypatch, cluster)
    first, second = (WRAPPERS[case](ctx, name) for name in ("gl-job-a", "gl-job-b"))
    assert (first.fixture.name, second.fixture.name) == (
        managed_workload_name(BASES[case], "gl-job-a"),
        managed_workload_name(BASES[case], "gl-job-b"),
    )
    assert first.fixture.name != second.fixture.name != BASES[case]
    assert first.fixture.settings.job_id == "gl-job-a"
    assert cluster.objects.keys() == {("deployment", EXECUTOR["metadata"]["name"])}


def test_two_jobs_on_one_node_run_side_by_side_under_their_own_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """003.ambiguous_ownership starts two managed jobs at once on one node; with
    one fixed name its second create was always refused."""

    cluster = Cluster()
    ctx = variant_context(
        "GF-GENERAL-LIVE-003.ambiguous_ownership", tmp_path, monkeypatch, cluster
    )
    jobs = []
    for name in ("gl3-ambA-1", "gl3-ambB-1"):
        job = utilization.ManagedIdleJob(ctx, name=name, gpu_count=4)
        ctx.cleanup.push(f"job-{name}", job.cleanup)
        job.start()
        jobs.append(job)

    names = [job.fixture.name for job in jobs]
    assert names == [managed_workload_name(SINGLE_NODE, job.name) for job in jobs]
    for job in jobs:
        workload = job.fixture.workload()
        assert workload["metadata"]["labels"][JOB_LABEL] == job.name
        assert [pod["name"] for pod in job.identity()["pods"]] == [
            f"{job.fixture.name}-master-0"
        ]
    assert SINGLE_NODE not in cluster.names_requested(), (
        "no request names the manifest's fixed name"
    )

    results = ctx.cleanup.run()
    assert [item["ok"] for item in results] == [True, True], results
    assert cluster.kinds() == {"deployment"}, cluster.objects
    for job, item in zip(reversed(jobs), results, strict=True):
        deleted = item["result"]["workload"]["deleted"]
        assert set(deleted) == {
            f"pytorchjob/{job.fixture.name}",
            f"pod/{job.fixture.name}-master-0",
            f"service/{job.fixture.name}-master-0",
        }, deleted
        assert item["result"]["prewarm"] == NOTHING_CREATED


# --------------------------------------------------------------------------- #
# the live sequence: a product restart in place, then cleanup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("then_suspend", [True, False], ids=["suspended", "running"])
def test_cleanup_removes_the_job_the_product_restarted_in_place(
    then_suspend: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """028.nonzero_exit through ``execute_variant``: the product restarts the
    failed job in place while the case waits for its decision; the cleanup
    deletes that job -- its own creation, relabelled -- and records it."""

    key = "GF-GENERAL-LIVE-028.nonzero_exit"
    cluster = Cluster()
    settings, run_dir = configured(tmp_path, monkeypatch, REGISTRY[key])
    install(monkeypatch, cluster)
    restarted: dict[str, Any] = {}
    children: dict[str, str] = {}

    def product_records(
        regional: Any, cluster_id: str, job_id: str, attempt_ids: list[str]
    ) -> dict[str, Any]:
        if not restarted:
            name = managed_workload_name(SINGLE_NODE, job_id)
            job = cluster.restart_in_place(name, then_suspend=then_suspend)
            restarted.update(copy.deepcopy(job))
            children.update(cluster.children())
        (attempt,) = attempt_ids
        entry = {
            "decision": {"status": "PLAN_CREATED"},
            "event": {"terminal_status": "FAILED"},
            "containment_incident": {"incident_id": "incident-1"},
            "containment_workflows": [
                {"official_steps": [{"operation": "STOP_WORKLOADS"}]}
            ],
            "recovery": [
                {"workflow": {"official_steps": [{"operation": "RESTART_WORKLOAD"}]}}
            ],
        }
        budget = {"restart_count": 1, "reservation_ids": [RESTART_OPERATION]}
        return {"attempts": {attempt: entry}, "restart_budget": budget}

    monkeypatch.setattr(attempts, "attempt_records", product_records)

    result = execute(key, settings, run_dir)

    job_id = result["attempt"].removesuffix("-a001")
    name = managed_workload_name(SINGLE_NODE, job_id)
    assert restarted["metadata"]["name"] == name
    assert restarted["metadata"]["labels"][ATTEMPT_LABEL] == (
        f"{job_id}-a001{RESTART}"
    ), "the product relabelled the job in place"
    assert (result["verdict"], result.get("errors")) == ("PASS", []), result
    (cleanup,) = result["cleanup"]
    assert cleanup["ok"] is True, cleanup
    workload = cleanup["result"]["workload"]
    # Suspended: no Pod and no Service, as live; running: the restarted
    # attempt's replica Pod and Service go too.
    assert set(children) == (
        set() if then_suspend else {f"pod/{name}-master-0", f"service/{name}-master-0"}
    ), children
    assert workload["deleted"] == {
        f"pytorchjob/{name}": restarted["metadata"]["uid"],
        **children,
    }, workload
    assert workload["product_restarts"] == sorted(
        [f"pytorchjob/{name}", *(item for item in children if item.startswith("pod/"))]
    ), workload
    assert cluster.kinds() == {"deployment"}, cluster.objects
    case_dir = run_dir / "cases" / key
    record = journal(case_dir / f"{job_id}.ownership.json")
    assert record["completed"] is True, record
    assert (
        record["creations"][f"pytorchjob/{name}"]["uid"]
        == (restarted["metadata"]["uid"])
    ), "the job the product relabelled is the one this fixture created"
    assert SINGLE_NODE not in cluster.names_requested()


def test_the_live_leftover_no_longer_blocks_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass 3's leftover under the fixed name, another job id: 003 now starts."""

    cluster = Cluster()
    ctx = variant_context(
        "GF-GENERAL-LIVE-003.gpu_idle", tmp_path, monkeypatch, cluster
    )
    stale = copy.deepcopy(
        cluster.leftover(
            SINGLE_NODE, job_id="gl28nz-nz-a1-c5ab1930", owner="e60b36bf" * 4
        )
    )
    job = utilization.ManagedIdleJob(ctx, name="gl3-idle-1")
    snapshot = job.start()
    assert snapshot["workload"]["name"] == managed_workload_name(
        SINGLE_NODE, "gl3-idle-1"
    )
    job.cleanup()
    assert cluster.objects[("pytorchjob", SINGLE_NODE)] == stale, (
        "another run's object is never touched"
    )
    assert ("pytorchjob", SINGLE_NODE) not in {item[:2] for item in cluster.deletes}


# --------------------------------------------------------------------------- #
# the pre-create guard
# --------------------------------------------------------------------------- #
def test_a_leftover_of_the_same_job_id_is_refused_by_name_and_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """003's job ids repeat every pass (``gl3-idle-<attempt>``): an earlier
    pass's leftover of the same job holds the derived name and the job id."""

    key = "GF-GENERAL-LIVE-003.gpu_idle"
    cluster = Cluster()
    settings, run_dir = configured(tmp_path, monkeypatch, REGISTRY[key])
    install(monkeypatch, cluster)
    name = managed_workload_name(SINGLE_NODE, "gl3-idle-1")
    stale = copy.deepcopy(cluster.leftover(name, job_id="gl3-idle-1", owner="0" * 32))

    result = execute(key, settings, run_dir)

    assert result["error"] == (
        "RegionalFixtureError: managed workload name or job identity already "
        f"exists: pytorchjob/{name} (uid {stale['metadata']['uid']}, job-id "
        f"gl3-idle-1, attempt-id gl3-idle-1-a001{RESTART}, {OWNER_LABEL} "
        f"{'0' * 32}, Suspended) is another acceptance run's managed workload, "
        "left over unless that run still runs; it is never deleted "
        "automatically -- check it and remove it by hand: kubectl -n "
        f"{NAMESPACE} delete pytorchjob {name}"
    ), result
    assert cluster.objects[("pytorchjob", name)] == stale
    assert cluster.deletes == [], "nothing is deleted on a refusal"
    assert not any(request[0] == "create" for request in cluster.requests), (
        "the refused job sends no create"
    )
    (job,) = result["cleanup"]
    assert job["result"] == {"workload": NOTHING_CREATED, "prewarm": NOTHING_CREATED}


def test_the_guard_says_whose_each_object_is() -> None:
    labels = {JOB_LABEL: "j", ATTEMPT_LABEL: "j-a001", OWNER_LABEL: "a" * 32}
    document = {
        "kind": "Pod",
        "metadata": {"name": "p", "uid": "u", "labels": labels},
        "status": {"phase": "Running"},
    }
    mine = occupant(document, owner="a" * 32, namespace=NAMESPACE)
    assert mine.endswith("Running) is this fixture's own submission"), mine
    labels[MANAGED_LABEL] = "true"
    theirs = occupant(document, owner="b" * 32, namespace=NAMESPACE)
    assert "another acceptance run's managed workload" in theirs, theirs
    assert theirs.endswith(f"kubectl -n {NAMESPACE} delete pod p"), theirs
    foreign = occupant(
        {"kind": "PyTorchJob", "metadata": {"name": "x", "uid": "v"}},
        owner="b" * 32,
        namespace=NAMESPACE,
    )
    assert foreign == (
        "pytorchjob/x (uid v, job-id None, attempt-id None, "
        f"{OWNER_LABEL} None, no condition) is not an acceptance-managed "
        "workload and is left alone"
    )


# --------------------------------------------------------------------------- #
# both parts of a job's cleanup run, and their record survives a failure
# --------------------------------------------------------------------------- #
def started_fail_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cluster: Cluster
) -> attempts.ManagedFailJob:
    ctx = variant_context(
        "GF-GENERAL-LIVE-028.nonzero_exit", tmp_path, monkeypatch, cluster
    )
    job = attempts.ManagedFailJob(ctx, name="gl28nz-nz-a1-c5ab1930", mode="exit:1")
    job.start()
    return job


@pytest.mark.parametrize("resumed", [False, True], ids=["creator", "reopened"])
def test_the_products_restart_copy_goes_with_the_job_that_made_it(
    resumed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal job is restarted as a copy, ``<name[:52]>-r-<8 hex>``, under a
    new UID the journal never recorded. Its creator deletes it and its Pod
    under the lineage rule and journals them; a journal another process
    reopened deletes nothing it did not record."""

    cluster = Cluster()
    job = started_fail_job(tmp_path, monkeypatch, cluster)
    name = job.fixture.name
    restart = copy.deepcopy(cluster.objects[("pytorchjob", name)])
    for key in ("uid", "resourceVersion"):
        restart["metadata"].pop(key)
    restart["metadata"]["name"] = f"{name[:52]}{RESTART}"
    restart["metadata"]["annotations"] = {
        "gpu-fault.io/operation-id": RESTART_OPERATION
    }
    target = f"{job.attempt_id[:52]}{RESTART}"
    restart["metadata"]["labels"][ATTEMPT_LABEL] = target
    for replica in restart["spec"]["pytorchReplicaSpecs"].values():
        replica["template"]["metadata"]["labels"][ATTEMPT_LABEL] = target
    cluster.reconcile(cluster.store(restart))
    assert job.fixture.ownership is not None
    fixture = job.fixture
    if resumed:
        path = job.fixture.ownership.path
        fixture = ManagedWorkloadFixture(
            job.fixture.regional, job.fixture.settings, state_path=path
        )
        with pytest.raises(WorkloadOwnershipError, match="ownership or UID changed"):
            fixture.delete()
        assert cluster.deletes == [], "nothing is deleted before the set is proven"
        return

    deleted = fixture.delete()

    copy_name = restart["metadata"]["name"]
    assert set(deleted) == {
        resource
        for owner in (name, copy_name)
        for resource in (
            f"pytorchjob/{owner}",
            f"pod/{owner}-master-0",
            f"service/{owner}-master-0",
        )
    }, deleted
    assert fixture.unbound_restart_resources == {
        f"pytorchjob/{copy_name}",
        f"pod/{copy_name}-master-0",
    }
    record = journal(job.fixture.ownership.path)
    assert (
        record["observed"][f"pytorchjob/{copy_name}"]
        == deleted[f"pytorchjob/{copy_name}"]
    ), "the copy the cleanup deleted is journaled"
    assert record["completed"] is True, record
    assert cluster.kinds() == {"deployment"}, cluster.objects


def test_a_failing_prewarm_proof_does_not_undo_or_hide_the_workload_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = Cluster(cached=False)
    job = started_fail_job(tmp_path, monkeypatch, cluster)
    (prewarm,) = [key for key in cluster.objects if key[1] in job.prewarm.pods]
    cluster.objects[prewarm]["metadata"]["labels"][OWNER_LABEL] = "someone-else"

    with pytest.raises(ManagedJobCleanupError) as failure:
        job.delete()

    workload = failure.value.record["workload"]
    assert set(workload["deleted"]) == {
        f"pytorchjob/{job.fixture.name}",
        f"pod/{job.fixture.name}-master-0",
        f"service/{job.fixture.name}-master-0",
    }, workload
    assert failure.value.record["prewarm"] == {
        "error": "RegionalFixtureError: image prewarm Pod ownership changed"
    }
    assert f"pytorchjob/{job.fixture.name}" in str(failure.value)
    assert cluster.kinds() == {"deployment", "pod"}, "only the foreign Pod is left"
    assert prewarm in cluster.objects


def test_a_refused_workload_cleanup_still_removes_the_prewarm_pods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = Cluster(cached=False)
    job = started_fail_job(tmp_path, monkeypatch, cluster)
    (prewarm,) = [key for key in cluster.objects if key[1] in job.prewarm.pods]
    stranger = copy.deepcopy(
        cluster.store(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": "stranger", "labels": {JOB_LABEL: job.name}},
                "spec": {},
                "status": {"phase": "Running"},
            }
        )
    )

    with pytest.raises(ManagedJobCleanupError) as failure:
        job.delete()

    record = failure.value.record
    assert record["workload"]["deleted"] == {}, "the whole set is proven first"
    assert record["workload"]["error"].startswith("WorkloadOwnershipError"), record
    assert record["prewarm"] == {prewarm[1]: False}, record
    assert prewarm not in cluster.objects, "the prewarm cleanup ran anyway"
    assert cluster.objects[("pod", "stranger")] == stranger


def test_a_controller_another_owner_relabelled_is_never_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart lineage widens deletion only to what carries this
    fixture's nonce: the journaled cleanup still refuses everything else."""

    cluster = Cluster()
    job = started_fail_job(tmp_path, monkeypatch, cluster)
    pytorchjob = cluster.objects[("pytorchjob", job.fixture.name)]
    pytorchjob["metadata"]["labels"][OWNER_LABEL] = "f" * 32  # not this fixture's

    with pytest.raises(RegionalFixtureError, match="managed job cleanup is unproven"):
        job.delete()
    assert ("pytorchjob", job.fixture.name) in cluster.objects
    assert cluster.deletes == []
