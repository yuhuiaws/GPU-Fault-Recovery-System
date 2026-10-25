"""GF-GENERAL-LIVE managed jobs started with the objects the runner builds.

Live 2026-10-24, GF-GENERAL-LIVE-003.gpu_idle on the plain-EKS lane:
``ManagedIdleJob.start`` gave the image prewarm the case's
``RegionalLiveSettings`` instead of the node name. The prewarm journaled its
Pod intent and then failed serializing the Pod -- ``TypeError: Object of type
RegionalLiveSettings is not JSON serializable`` -- so cleanup refused "fixture
creation outcome is unknown" for a Pod that was never sent and the variant
failed "cleanup did not prove the node clean". 028/029 handed it the cluster
id: a Pod pinned to a node that does not exist. The variant tests replace both
fixtures with fakes that take any node list, so no start path ever ran.

Here every object is the runner's own: ``install_site_profile``,
``add_arguments`` and ``configure`` over a private site profile naming a
loadable site file, then ``execute_variant`` with the real
``RegionalLiveFixture``, ``VariantContext``, job wrapper, image prewarm,
managed workload fixture and ownership journals. Only the kubectl subprocess
(``RegionalLiveFixture.run``), ``cpu_python`` and the host-probe calls are
fakes. The fake API answers the reads and stops the run at the first mutation
it is asked for, so each variant runs up to the point just before it would
touch a cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.training_submit_cli import TRAINING_CONTAINER_ANNOTATION
from scripts.e2e.regional import run_general_live as runner
from scripts.e2e.regional.general_live_fixture import GeneralLiveSettings, ProbeHandle
from scripts.e2e.regional.general_live_registry import VariantSpec
from scripts.e2e.regional.managed_workload_fixture import (
    NOTHING_CREATED,
    OWNER_LABEL,
    TRAINING_IMAGE,
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    managed_workload_name,
    prewarm_node_names,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
)
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.admin.test_admin_site import site_file as write_site_file

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-44-6.us-west-2.compute.internal"
ELSEWHERE = "ip-10-0-99-9.us-west-2.compute.internal"
NAMESPACE = "gpu-fault-system"
CLUSTER = "gpu-a"  # the site file's own cluster
CONTEXT = "gpu-a"
MANAGED_JOB_CASES = (
    "GF-GENERAL-LIVE-003",  # general_live_utilization: ManagedIdleJob
    "GF-GENERAL-LIVE-011",  # general_live_nccl / _nccl_job: ManagedNcclJob
    "GF-GENERAL-LIVE-027",  # general_live_progress: ManagedProgressJob
    "GF-GENERAL-LIVE-028",  # general_live_attempts: ManagedFailJob
    "GF-GENERAL-LIVE-029",  # general_live_attempts: ManagedFailJob
    "GF-GENERAL-LIVE-017",  # general_live_applogs / _applog_job: ManagedApplogJob
)
#: The 017 variants that replay the journal source, not a managed job's CRI log.
JOURNAL_VARIANTS = frozenset(
    f"GF-GENERAL-LIVE-017.{variant}"
    for variant in ("cgroup_oom", "nccl_error", "distributed_error")
)
#: Variants outside 011 that run the two-node NCCL job (``--node-b``).
TWO_NODE_VARIANTS = frozenset({"GF-GENERAL-LIVE-017.missing_flight_recorder"})
#: The 003 variants that drive the host and storage probes, not a managed job.
HOST_VARIANTS = frozenset(
    f"GF-GENERAL-LIVE-003.{variant}"
    for variant in (
        "high_page_cache",
        "idle_node",
        "low_available_memory",
        "scratch_full",
    )
)
#: The 029 variants that drive the store-level restart-budget probe in an API
#: Pod (general_live_restart_budget), not a managed job.
STORE_VARIANTS = frozenset(
    {"GF-GENERAL-LIVE-029.concurrent_claim", "GF-GENERAL-LIVE-029.missing_context"}
)
REGISTRY = runner.load_registry()
MANAGED_JOB_VARIANTS = sorted(
    key
    for key, spec in REGISTRY.items()
    if spec.case_id in MANAGED_JOB_CASES
    and spec.implemented
    and key not in HOST_VARIANTS | STORE_VARIANTS
    and key not in JOURNAL_VARIANTS
)
#: One variant per case, the live failure first.
ONE_PER_CASE = (
    "GF-GENERAL-LIVE-003.gpu_idle",
    "GF-GENERAL-LIVE-011.active_to_zero",
    "GF-GENERAL-LIVE-027.nan",
    "GF-GENERAL-LIVE-028.nonzero_exit",
    "GF-GENERAL-LIVE-029.budget_consumed",
    "GF-GENERAL-LIVE-017.live_pid",
)
MUTATING_VERBS = frozenset(
    {
        "annotate",
        "apply",
        "cordon",
        "cp",
        "create",
        "delete",
        "drain",
        "edit",
        "exec",
        "label",
        "patch",
        "replace",
        "rollout",
        "scale",
        "set",
        "taint",
        "uncordon",
    }
)
SURVEY = {"residue": {}, "tools": {}, "collector_env": {}}
#: The GPU cluster's executor runs the release's runtime image; 027's progress
#: reporter sidecars run the same image (``gpu-fault-collector`` is in it).
RUNTIME_IMAGE = "111122223333.dkr.ecr.us-west-2.amazonaws.com/gpu-fault@sha256:" + (
    "e" * 64
)
EXECUTOR = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "gpu-fault-cluster-executor", "namespace": NAMESPACE},
    "spec": {
        "template": {
            "spec": {"containers": [{"name": "executor", "image": RUNTIME_IMAGE}]}
        }
    },
}


class FirstMutation(BaseException):
    """The run asked for its first cluster mutation; nothing was sent.

    A BaseException, like an operator's abort: no ``except Exception`` on the
    way swallows it, and ``execute_variant`` still tears down and records it.
    """


@dataclass(frozen=True)
class Mutation:
    plane: str
    command: tuple[str, ...]
    body: str | None

    def document(self) -> dict[str, Any]:
        assert self.body is not None, f"{self.command} carried no request body"
        value = json.loads(self.body)
        assert isinstance(value, dict), f"{self.command} body is not an object"
        return value


class ClusterApi:
    """Answers the runner's reads; refuses, and records, every mutation."""

    def __init__(self, *, nodes: list[str], cached: bool) -> None:
        self.nodes = nodes
        self.cached = cached
        self.reads: list[tuple[str, ...]] = []
        self.mutations: list[Mutation] = []

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
        assert command[0] == "kubectl" and "-n" in command, (
            f"only namespaced kubectl reaches the cluster fake: {command}"
        )
        plane = "gpu" if "--context" in command else "cpu"
        if plane == "gpu":
            assert command[command.index("--context") + 1] == CONTEXT, command
        assert command[command.index("-n") + 1] == NAMESPACE, command
        arguments = tuple(command[command.index("-n") + 2 :])
        if arguments[0] in MUTATING_VERBS:
            self.mutations.append(Mutation(plane, arguments, input_text))
            raise FirstMutation(f"kubectl {arguments[0]}")
        self.reads.append(arguments)
        return subprocess.CompletedProcess(command, 0, self.read(plane, arguments), "")

    def read(self, plane: str, arguments: tuple[str, ...]) -> str:
        if arguments == ("api-resources", "-o", "name"):
            return "pods\njobs.batch\npytorchjobs.kubeflow.org\nservices\n"
        assert arguments[0] == "get", f"unexpected kubectl read: {arguments}"
        if arguments[1:3] == ("configmap", "gpu-fault-regional-release-state"):
            assert plane == "cpu", arguments
            state = {
                "phase": "complete",
                "transaction_committed": True,
                "release_id": "rel-test",
            }
            return json.dumps({"data": {"state.json": json.dumps(state)}})
        assert plane == "gpu", arguments
        if arguments[1:] == ("node", "-o", "json"):
            images = [{"names": [TRAINING_IMAGE]}] if self.cached else []
            return json.dumps(
                {
                    "items": [
                        {"metadata": {"name": node}, "status": {"images": images}}
                        for node in self.nodes
                    ]
                }
            )
        if "-l" in arguments:
            return json.dumps({"items": []})
        if arguments[1:] == ("deployment", "gpu-fault-cluster-executor", "-o", "json"):
            return json.dumps(EXECUTOR)
        assert "--ignore-not-found" in arguments, f"unexpected read: {arguments}"
        return ""  # no such object yet


def unexpected_store_read(
    regional: RegionalLiveFixture, script: str, *arguments: str, **options: Any
) -> dict[str, Any]:
    raise AssertionError("a managed job start reads no store record")


def probe_execute(
    handle: ProbeHandle, *arguments: str, timeout: int = 180
) -> dict[str, Any]:
    if handle.script.name == "general_live_node_probe.py" and arguments[0] == "survey":
        return dict(SURVEY)
    raise FirstMutation(f"host probe {handle.script.name} {arguments[0]}")


def probe_cleanup(handle: ProbeHandle) -> dict[str, bool]:
    return {"pod": False, "configmap": False}


@pytest.fixture
def private_umask() -> Iterator[None]:
    """``run_general_live.main`` runs under 0o077; the journals require it."""

    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


pytestmark = pytest.mark.usefixtures("private_umask")


def configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: VariantSpec
) -> tuple[GeneralLiveSettings, Path]:
    """``configure`` over a private site profile, parsed as ``main`` parses."""

    kube = tmp_path / "kube"
    kube.mkdir()
    for plane in ("cpu", "gpu"):
        (kube / plane).write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    site_root = tmp_path / "site"
    site_root.mkdir()
    site = write_site_file(site_root)
    arguments: dict[str, Any] = {
        "cpu_kubeconfig": str(kube / "cpu"),
        "gpu_kubeconfig": str(kube / "gpu"),
        "gpu_context": CONTEXT,
        "namespace": NAMESPACE,
        "cluster_id": CLUSTER,
        "region": "us-west-2",
        "fault_node": NODE,
        "host_probe_image": "public.ecr.aws/gfrs/probe@sha256:" + "1" * 64,
        "site_file": str(site),
    }
    if spec.case_id == "GF-GENERAL-LIVE-011" or spec.key in TWO_NODE_VARIANTS:
        arguments["node_b"] = PEER  # the lane's second node
    profile = tmp_path / "site-profile.json"
    profile.write_text(
        json.dumps({"arguments": arguments, "environment": {}}), encoding="utf-8"
    )
    profile.chmod(0o600)
    monkeypatch.setenv(SITE_PROFILE_ENV, str(profile))
    run_dir = tmp_path / "run"
    argv = [
        "--run-dir",
        str(run_dir),
        "--case",
        spec.case_id,
        "--variant",
        spec.variant,
        "--execute",
        "--confirm",
        spec.confirmation,
    ]
    for token in spec.approvals:
        # A variant's required tokens (017's CRI window runtime-enables the
        # node log collector); without them the run is NOT_RUN before any job.
        argv += ["--approve", token]
    runner.install_site_profile(argv)
    parser = argparse.ArgumentParser()
    runner.add_arguments(parser)
    settings = runner.configure(parser.parse_args(argv))
    assert (settings.node, settings.site_file) == (NODE, site), settings
    return settings, run_dir


def install_fakes(monkeypatch: pytest.MonkeyPatch, api: ClusterApi) -> None:
    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(api.run))
    monkeypatch.setattr(RegionalLiveFixture, "cpu_python", unexpected_store_read)
    monkeypatch.setattr(ProbeHandle, "execute", probe_execute)
    monkeypatch.setattr(ProbeHandle, "cleanup", probe_cleanup)


def execute(
    spec: VariantSpec, settings: GeneralLiveSettings, run_dir: Path
) -> dict[str, Any]:
    deadline = datetime.now(timezone.utc) + timedelta(hours=3)
    return runner.execute_variant(spec, settings, run_dir, 1, deadline, {"errors": []})


def journal(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def run_to_first_mutation(
    spec: VariantSpec, settings: GeneralLiveSettings, run_dir: Path
) -> dict[str, Any]:
    """``execute_variant`` until its first mutation; the result it recorded.

    The runner records the stop like any abort and re-raises it; a variant
    that fails before that point records its own error instead.
    """

    try:
        execute(spec, settings, run_dir)
    except FirstMutation:
        pass
    return journal(run_dir / "cases" / spec.key / f"{spec.key}.json")


def job_journals(case_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """The first managed job's id and its prewarm and workload journals."""

    prewarms = sorted(case_dir.glob("*.prewarm.json"))
    assert len(prewarms) == 1, f"one job reaches its first mutation: {prewarms}"
    job_id = prewarms[0].name.removesuffix(".prewarm.json")
    return (
        job_id,
        journal(prewarms[0]),
        journal(case_dir / f"{job_id}.ownership.json"),
    )


def templates(document: dict[str, Any]) -> list[dict[str, Any]]:
    replicas = document["spec"]["pytorchReplicaSpecs"].values()
    return [replica["template"] for replica in replicas]


@pytest.mark.parametrize("cached", [False, True], ids=["prewarm-pod", "image-cached"])
@pytest.mark.parametrize("key", MANAGED_JOB_VARIANTS)
def test_every_managed_job_reaches_its_first_mutation_pinned_to_the_target_node(
    key: str, cached: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = REGISTRY[key]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    nodes = [NODE, PEER] if settings.peer_node else [NODE]
    api = ClusterApi(nodes=nodes, cached=cached)
    install_fakes(monkeypatch, api)

    result = run_to_first_mutation(spec, settings, run_dir)

    assert result.get("error") == "FirstMutation: kubectl create", (
        "the run must reach its first mutation, not fail preparing it: "
        f"{result.get('error')}"
    )
    assert len(api.mutations) == 1, f"one mutation, none in teardown: {api.mutations}"
    (mutation,) = api.mutations
    assert (mutation.plane, mutation.command[:3]) == ("gpu", ("create", "-f", "-"))
    document = mutation.document()
    job_id, prewarm, workload = job_journals(run_dir / "cases" / key)
    if not cached:
        # The prewarm Pod for the target node is the first thing sent.
        assert (document["kind"], document["spec"]["nodeName"]) == ("Pod", NODE)
        assert prewarm["started"] is True, prewarm
        assert {
            name: (item["uid"], item["expected"]["spec"]["nodeName"])
            for name, item in prewarm["creations"].items()
        } == {f"pod/{document['metadata']['name']}": (None, NODE)}
        assert (workload["started"], workload["creations"]) == (False, {})
        return
    # The image is on every node: no prewarm Pod, the workload is first.
    metadata = document["metadata"]
    assert (document["kind"], metadata.get("namespace")) == ("PyTorchJob", NAMESPACE), (
        document
    )
    assert (prewarm["started"], prewarm["creations"]) == (True, {})
    # Every template is pinned to a lane node and every lane node is used:
    # 028.cleanup_timeout runs two ranks on the one node (a set, not a list).
    assert {item["spec"]["nodeName"] for item in templates(document)} == set(nodes), (
        document["spec"]
    )
    assert {
        item["metadata"]["annotations"][TRAINING_CONTAINER_ANNOTATION]
        for item in templates(document)
    } == {"pytorch"}, "the trainer, never a sidecar, is the training container"
    sidecars = [
        container
        for item in templates(document)
        for container in item["spec"]["containers"]
        if container["name"] != "pytorch"
    ]
    assert {container["image"] for container in sidecars} <= {RUNTIME_IMAGE}, (
        "a sidecar runs the cluster's own runtime image, never the training image"
    )
    labels = metadata["labels"]
    assert (labels["gpu-fault.io/job-id"], labels["gpu-fault.io/attempt-id"]) == (
        job_id,
        f"{job_id}-a001",
    )
    assert labels[OWNER_LABEL] == workload["owner"], (labels, workload["owner"])
    assert workload["started"] is True, workload
    assert list(workload["creations"]) == [f"pytorchjob/{metadata['name']}"]
    # The job's own name, never the manifest's: one leftover of another job no
    # longer holds every later job's name (live 2026-10-25).
    base = (
        "gpu-fault-two-node-replace"
        if spec.case_id == "GF-GENERAL-LIVE-011" or spec.key in TWO_NODE_VARIANTS
        else "gpu-fault-single-node-warm-spare"
    )
    assert metadata["name"] == managed_workload_name(base, job_id), metadata["name"]


@pytest.mark.parametrize("key", ONE_PER_CASE)
def test_a_target_node_the_cluster_lacks_is_refused_and_cleans_up_as_nothing_created(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = REGISTRY[key]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    api = ClusterApi(nodes=[ELSEWHERE], cached=False)
    install_fakes(monkeypatch, api)

    result = execute(spec, settings, run_dir)

    targets = [NODE, PEER] if settings.peer_node else [NODE]
    assert result["error"] == (
        "RegionalFixtureError: image prewarm nodes are not nodes of the GPU "
        "cluster: " + ", ".join(targets)
    ), result
    assert api.mutations == [], "a refused preparation sends nothing"
    (job,) = result["cleanup"]
    assert (job["ok"], job["result"]) == (
        True,
        {"workload": NOTHING_CREATED, "prewarm": NOTHING_CREATED},
    ), job
    assert result["cleanup_pending"] == [], result
    assert "cleanup did not prove the node clean" not in result.get("errors", [])
    job_id, prewarm, workload = job_journals(run_dir / "cases" / key)
    assert (prewarm["creations"], prewarm["completed"]) == ({}, True), prewarm
    assert (workload["started"], workload["creations"]) == (False, {}), workload


def standalone_regional(tmp_path: Path) -> RegionalLiveFixture:
    kube = tmp_path / "kube"
    kube.mkdir()
    for plane in ("cpu", "gpu"):
        (kube / plane).write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=kube / "cpu",
            gpu_kubeconfig=kube / "gpu",
            gpu_context=CONTEXT,
            namespace=NAMESPACE,
            cluster_id=CLUSTER,
            region="us-west-2",
        )
    )


def test_the_live_settings_object_is_refused_before_the_prewarm_journal_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = standalone_regional(tmp_path)
    api = ClusterApi(nodes=[NODE], cached=False)
    install_fakes(monkeypatch, api)
    path = tmp_path / "case" / "gl3-idle-1.prewarm.json"
    prewarm = ImagePrewarmFixture(
        regional, case_id="GF-GENERAL-LIVE-003.gpu_idle", run_id="r", state_path=path
    )

    with pytest.raises(RegionalFixtureError, match="not RegionalLiveSettings"):
        prewarm.create([regional.settings])

    assert (api.reads, api.mutations) == ([], []), "refused before any request"
    assert (journal(path)["started"], journal(path)["creations"]) == (False, {})
    assert prewarm.nothing_created() is True
    assert prewarm.cleanup() == {}
    assert api.mutations == [], "nothing to delete"


def test_a_creation_that_cannot_be_sent_is_never_journaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = standalone_regional(tmp_path)
    install_fakes(monkeypatch, ClusterApi(nodes=[NODE], cached=False))
    path = tmp_path / "case" / "x.prewarm.json"
    prewarm = ImagePrewarmFixture(regional, case_id="c", run_id="r", state_path=path)
    ownership = prewarm.ownership
    assert ownership is not None, "a state path journals the prewarm"
    ownership.begin()
    for value in (regional.settings, float("nan")):
        manifest = prewarm.manifest(NODE, 0)
        manifest["spec"]["nodeName"] = value
        with pytest.raises(RegionalFixtureError, match="not a JSON document"):
            ownership.intend(manifest)
    assert ownership.nothing_intended() is True
    assert journal(path)["creations"] == {}


def test_a_rendered_workload_that_cannot_be_sent_is_refused_before_its_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = standalone_regional(tmp_path)
    api = ClusterApi(nodes=[NODE], cached=False)
    install_fakes(monkeypatch, api)
    source = (
        Path(runner.__file__).with_name("manifests")
        / "training"
        / "single-node-warm-spare-pytorchjob.yaml"
    )
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    # An unquoted YAML date survives rendering as a date, which JSON cannot carry.
    document["metadata"]["annotations"] = {"gfrs.test/started": date(2026, 10, 24)}
    manifest = tmp_path / "job.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    site_root = tmp_path / "site"
    site_root.mkdir()
    path = tmp_path / "case" / "job.ownership.json"
    fixture = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=write_site_file(site_root),
            job_id="gl3-idle-1",
            attempt_id="gl3-idle-1-a001",
            restart_budget=0,
            expected_pods=1,
            expected_gpu_count=8,
        ),
        state_path=path,
    )

    with pytest.raises(RegionalFixtureError, match="not a JSON document.*date"):
        fixture.submit()

    assert api.mutations == [], "nothing was sent"
    assert (journal(path)["started"], journal(path)["creations"]) == (False, {})
    assert fixture.nothing_created() is True
    fixture.delete()
    assert api.mutations == [], "nothing to delete"


@pytest.mark.parametrize(
    "nodes,message",
    [
        (NODE, "a list of node names"),
        ([], "nonempty unique node set"),
        ([NODE, NODE], "nonempty unique node set"),
        ([NODE, ""], "nonempty unique node set"),
        ([NODE, 7, {"name": PEER}], "must be strings, not dict, int"),
    ],
)
def test_prewarm_node_names_are_refused_by_shape(nodes: Any, message: str) -> None:
    with pytest.raises(RegionalFixtureError, match=message):
        prewarm_node_names(nodes)
    assert prewarm_node_names((NODE, PEER)) == [NODE, PEER]
