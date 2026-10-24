"""GF-GENERAL-LIVE-030 warm-spare variants started with the runner's objects.

Live 2026-10-24 (HyperPod lane), GF-GENERAL-LIVE-030.not_ready and
.stale_metrics both failed ``error: TypeError: 'str' object is not callable``.
``SpareSession`` stores the spare node *name* in ``self.node`` and also defines
a ``node()`` method that returns the node snapshot; the instance attribute
shadows the method, so every ``session.node()`` / ``self.node()`` call tries to
call a string. The runner had already declared the spare before the abort, and
the cleanup receipt did not prove the release.

The verdict tests in ``test_general_live_spares.py`` replace ``SpareSession``
with a fake that keeps the name in ``_node`` and a separate ``node()`` method,
so they never hit the collision. Here every object is the runner's own:
``install_site_profile`` / ``add_arguments`` / ``configure`` over a private
HyperPod site profile and a loadable site file, then ``execute_variant`` with
the real ``RegionalLiveFixture``, ``VariantContext``, ``SpareSession``, the
admin warm-spare declare/release path and the warm-spare live fixture. Only the
kubectl subprocess (the regional ``run`` and the admin ``run_command``),
``cpu_python`` and the host-probe create/cleanup are fakes. The fake applies the
warm-spare node patches in memory -- so a spare is really declared and really
released -- and stops the run at the first *workload* mutation (a host-probe
Pod), the first thing that would reach a cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import operator_identity, warm_spare
from scripts.e2e.regional import general_live_spares as sp
from scripts.e2e.regional import run_general_live as runner
from scripts.e2e.regional.general_live_fixture import GeneralLiveSettings, ProbeHandle
from scripts.e2e.regional.general_live_registry import VariantSpec
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.regional_case_contract import case_evidence_path
from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from scripts.e2e.regional.warm_spare_fixture import (
    HYPERPOD_CLUSTER_LABEL,
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    SPARE_LABEL,
)
from scripts.e2e.regional.warm_spare_fixture import STORE_PROBE as WARM_STORE_PROBE
from tests.admin.test_admin_site import site_file as write_site_file

NODE = "ip-10-0-43-9.us-west-2.compute.internal"
NODE_UID = "11111111-2222-3333-4444-555555555555"
NAMESPACE = "gpu-fault-system"
CLUSTER = "gpu-a"  # the site file's own registration cluster id
CONTEXT = "gpu-a"
HYPERPOD_CLUSTER = "hp-control"
EXECUTABLE = ("not_ready", "stale_metrics", "configuration_error", "reserved_spare")
WORKLOAD_VERBS = frozenset(
    {"apply", "create", "delete", "exec", "cp", "drain", "scale", "rollout", "set"}
)


class FirstMutation(BaseException):
    """The run asked for its first workload mutation; nothing was sent.

    A ``BaseException`` (like an operator abort) so no ``except Exception`` on
    the way swallows it and ``execute_variant`` still tears down and records it.
    """


def _fresh_node() -> dict[str, Any]:
    return {
        "metadata": {
            "name": NODE,
            "uid": NODE_UID,
            "resourceVersion": "1000",
            "labels": {
                HYPERPOD_HEALTH_LABEL: "Schedulable",
                HYPERPOD_CLUSTER_LABEL: HYPERPOD_CLUSTER,
                INSTANCE_GROUP_LABEL: "ig-a",
                "node.kubernetes.io/instance-type": "ml.p5.48xlarge",
            },
            "annotations": {},
        },
        "spec": {
            # HyperPod providerID: the spare's Agent identity is the Node UID,
            # not an EC2 instance id, so this carries none.
            "providerID": "aws:///us-west-2a/sagemaker/cluster/hp-control",
            "unschedulable": False,
            "taints": [],
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "allocatable": {"nvidia.com/gpu": "8"},
            "images": [],
        },
    }


def _apply_merge(node: dict[str, Any], body: dict[str, Any]) -> None:
    metadata = body.get("metadata") or {}
    for section in ("labels", "annotations"):
        patch = metadata.get(section)
        if not patch:
            continue
        target = node["metadata"].setdefault(section, {})
        for key, value in patch.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
    spec = body.get("spec") or {}
    if "unschedulable" in spec:
        node["spec"]["unschedulable"] = bool(spec["unschedulable"])
    current = int(node["metadata"]["resourceVersion"])
    node["metadata"]["resourceVersion"] = str(current + 1)


class SpareCluster:
    """One spare node whose warm-spare patches apply in memory; no cluster.

    ``run`` answers the regional fixture's kubectl and ``run_command`` the admin
    warm-spare node API; both share one node and record every mutation. Node
    label/cordon/annotation patches (declare, release, the injected condition)
    are applied, so the spare is genuinely declared and released. A Pod/service
    create -- the host probe -- is the first thing that reaches a cluster and
    raises ``FirstMutation``.
    """

    def __init__(self) -> None:
        self.node = _fresh_node()
        self.reads: list[tuple[str, ...]] = []
        self.node_patches: list[dict[str, Any]] = []

    # --- the two kubectl transports -------------------------------------- #
    def run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "kubectl", command
        stdout = self._dispatch(command, input_text)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def run_command(
        self, command: list[str], *, timeout_seconds: float | None = None, **_: Any
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "kubectl", command
        stdout = self._dispatch(command, None)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    # --- shared request handling ----------------------------------------- #
    def _dispatch(self, command: list[str], input_text: str | None) -> str:
        verb, rest = self._split(command)
        if verb in WORKLOAD_VERBS:
            raise FirstMutation(f"kubectl {verb} {rest[0] if rest else ''}".strip())
        if verb == "patch":
            return self._patch(command)
        assert verb == "get", f"unexpected kubectl verb: {command}"
        self.reads.append(tuple(rest))
        return self._get(rest)

    @staticmethod
    def _split(command: list[str]) -> tuple[str, list[str]]:
        verbs = WORKLOAD_VERBS | {"get", "patch", "annotate", "label", "api-resources"}
        for index, token in enumerate(command):
            if token in verbs:
                return token, command[index + 1 :]
        raise AssertionError(f"no kubectl verb found: {command}")

    def _get(self, rest: list[str]) -> str:
        resource = rest[0]
        if resource == "configmap":
            state = {
                "phase": "complete",
                "transaction_committed": True,
                "release_id": "rel-test",
            }
            return json.dumps({"data": {"state.json": json.dumps(state)}})
        if resource == "pod":
            return json.dumps({"items": []})
        assert resource == "node", f"unexpected read: {rest}"
        if "-l" in rest:
            selector = rest[rest.index("-l") + 1]
            assert selector == f"{SPARE_LABEL}=true", selector
            labelled = self.node["metadata"]["labels"].get(SPARE_LABEL) == "true"
            return json.dumps({"items": [self.node] if labelled else []})
        # get node <name> -o json
        name = rest[1]
        assert name == NODE, f"unexpected node: {name}"
        return json.dumps(self.node)

    def _patch(self, command: list[str]) -> str:
        assert command[command.index("patch") + 1] == "node"
        body = json.loads(command[command.index("-p") + 1])
        _apply_merge(self.node, body)
        self.node_patches.append(body)
        return json.dumps(self.node) if "-o" in command else ""

    # --- the store reads -------------------------------------------------- #
    def cpu_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        if script == WARM_STORE_PROBE:  # the warm-spare Agent lookup
            return {"agents": [{"node_id": NODE, "lifecycle_state": "ACTIVE"}]}
        if script == sp.SPARE_STORE_PROBE:  # the read-only spare-health grade
            return {
                "agent_reasons": [],
                "fault_reasons": [],
                "observation_reasons": [],
                "spare_health_incidents": [],
                "spare_health_workflows": {},
            }
        raise AssertionError("unexpected cpu_python script")


def _probe_create(self: HostProbeFixture) -> None:
    raise FirstMutation(f"host probe {self.settings.probe_script.name}")


def _probe_handle_cleanup(self: ProbeHandle) -> dict[str, bool]:
    return {"pod": False, "configmap": False}


@pytest.fixture
def private_umask() -> Iterator[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


pytestmark = pytest.mark.usefixtures("private_umask")


def configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: VariantSpec
) -> tuple[GeneralLiveSettings, Path]:
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
        "hyperpod_cluster": HYPERPOD_CLUSTER,
        "host_probe_image": "public.ecr.aws/gfrs/probe@sha256:" + "1" * 64,
        "site_file": str(site),
        "approve": list(spec.approvals),
    }
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
    runner.install_site_profile(argv)
    parser = argparse.ArgumentParser()
    runner.add_arguments(parser)
    settings = runner.configure(parser.parse_args(argv))
    assert (settings.node, settings.hyperpod_cluster) == (NODE, HYPERPOD_CLUSTER)
    assert set(spec.approvals) <= set(settings.approvals), settings.approvals
    return settings, run_dir


def install_fakes(monkeypatch: pytest.MonkeyPatch, cluster: SpareCluster) -> None:
    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(cluster.run))
    monkeypatch.setattr(RegionalLiveFixture, "cpu_python", cluster.cpu_python)
    monkeypatch.setattr(warm_spare, "run_command", cluster.run_command)
    monkeypatch.setattr(HostProbeFixture, "create", _probe_create)
    monkeypatch.setattr(ProbeHandle, "cleanup", _probe_handle_cleanup)
    # Never shell out to ``aws sts``; the admin path attributes the mutation to
    # the local shell identity when the ARN cannot be read.
    monkeypatch.setattr(operator_identity, "caller_identity_arn", lambda **_: None)


def run_variant(
    spec: VariantSpec, settings: GeneralLiveSettings, run_dir: Path
) -> dict[str, Any]:
    deadline = datetime.now(timezone.utc) + timedelta(hours=3)
    try:
        runner.execute_variant(spec, settings, run_dir, 1, deadline, {"errors": []})
    except FirstMutation:
        pass
    path = case_evidence_path(run_dir, spec.key)
    return json.loads(path.read_text(encoding="utf-8"))


def warm_spare_receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipts = [
        item for item in result.get("cleanup") or [] if item["name"] == "warm-spare"
    ]
    assert len(receipts) == 1, (
        f"exactly one warm-spare cleanup ran: {result.get('cleanup')}"
    )
    return receipts[0]


SPECS = {spec.variant: spec for spec in sp.VARIANTS}


@pytest.mark.parametrize("variant", EXECUTABLE)
def test_variant_reaches_its_first_workload_mutation_without_the_str_collision(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SPECS[variant]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    cluster = SpareCluster()
    install_fakes(monkeypatch, cluster)

    result = run_variant(spec, settings, run_dir)

    error = result.get("error") or ""
    assert "object is not callable" not in error, (
        f"{variant} still calls a str (self.node collision): {error}"
    )
    assert "AttributeError" not in error, f"{variant} hit a latent type bug: {error}"
    # A declared spare was really declared (the node was labelled + cordoned)
    # and then released, so the fake saw both patches.
    assert len(cluster.node_patches) >= 2, cluster.node_patches


@pytest.mark.parametrize("variant", EXECUTABLE)
def test_a_declared_spare_is_released_and_the_receipt_proves_it(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SPECS[variant]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    cluster = SpareCluster()
    install_fakes(monkeypatch, cluster)

    result = run_variant(spec, settings, run_dir)

    receipt = warm_spare_receipt(result)
    assert receipt["ok"] is True, receipt
    release = receipt["result"]
    assert release["released"] is True, f"the receipt must prove the release: {release}"
    assert release["restored"], f"the receipt must record the restored node: {release}"
    assert release["declared_spares"] == [], (
        f"the released node must be out of the declared spare set: {release}"
    )
    # The node is really uncordoned and unlabelled.
    assert cluster.node["metadata"]["labels"].get(SPARE_LABEL) is None
    assert cluster.node["spec"]["unschedulable"] is False
    assert result["cleanup_pending"] == [], result
    assert "cleanup did not prove the node clean" not in result.get("errors", [])


def test_a_declare_refused_before_any_cluster_change_cleans_up_as_nothing_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SPECS["configuration_error"]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    cluster = SpareCluster()
    # The node is already a declared spare: the declare refuses before it
    # patches anything, so nothing changed on the cluster.
    cluster.node["metadata"]["labels"][SPARE_LABEL] = "true"
    cluster.node["spec"]["unschedulable"] = True
    install_fakes(monkeypatch, cluster)

    result = run_variant(spec, settings, run_dir)

    # The node is left exactly as it was found (still the pre-existing spare).
    assert cluster.node["metadata"]["labels"].get(SPARE_LABEL) == "true"
    assert cluster.node["spec"]["unschedulable"] is True
    receipt = warm_spare_receipt(result)
    assert receipt["ok"] is True, receipt
    assert receipt["result"]["released"] is False, receipt["result"]
    assert result["cleanup_pending"] == [], result


def test_spare_session_node_snapshot_is_callable_not_a_shadowed_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression at its root: the node *name* and the node *snapshot*
    accessor must not share the name ``node`` on ``SpareSession``."""

    spec = SPECS["not_ready"]
    settings, run_dir = configured(tmp_path, monkeypatch, spec)
    cluster = SpareCluster()
    install_fakes(monkeypatch, cluster)
    from scripts.e2e.regional.general_live_fixture import Evidence, VariantContext

    case_dir = run_dir / "cases" / spec.key
    ctx = VariantContext(
        spec=spec,
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        run_id="gl030-not-ready-a1-deadbeef",
        attempt=1,
        deadline=datetime.now(timezone.utc) + timedelta(hours=3),
        evidence=Evidence(case_dir),
    )
    session = sp.SpareSession(ctx)
    assert session.node == NODE, "the spare node name stays a plain string"
    snapshot = session.node_snapshot()
    assert snapshot["name"] == NODE and snapshot["uid"] == NODE_UID, snapshot
