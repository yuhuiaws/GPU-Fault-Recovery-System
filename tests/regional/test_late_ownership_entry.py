from __future__ import annotations

import argparse
import importlib
import io
import json
import runpy
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_entry as entry
from scripts.e2e.regional import late_ownership_live as live
from scripts.e2e.regional import late_ownership_probe_bundle as bundle
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from tests.regional._late_ownership_support import evidence


def test_entry_requires_explicit_case_and_scenario_without_invoking_cli(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(entry.normal, "parser", lambda **kw: argparse.ArgumentParser())
    parser = entry.parser()
    for values in ([], ["--case", entry.CASES[0]], ["--scenario", entry.SCENARIOS[0]]):
        with pytest.raises(SystemExit):
            parser.parse_args(values)
    arguments = parser.parse_args(
        [
            "--case",
            entry.CASES[0],
            "--scenario",
            entry.SCENARIOS[1],
            "--ordinary-destr015-evidence",
            str(
                tmp_path
                / "ordinary"
                / "cases"
                / "GF-REGIONAL-DESTR-015"
                / "GF-REGIONAL-DESTR-015.json"
            ),
        ]
    )
    arguments.run_dir = tmp_path / "companion"
    base = SimpleNamespace(
        environment=lambda: {"owned": "local"},
        predecessor_path=tmp_path / "ordinary.json",
    )
    monkeypatch.setattr(entry.normal, "configure", lambda args: base)
    configured = entry.configure(arguments)
    assert configured.base is base and configured.case_id == entry.CASES[0]
    assert configured.environment() == {
        "owned": "local",
        "GPU_FAULT_ORDINARY_DESTR015_EVIDENCE": str(
            arguments.ordinary_destr015_evidence
        ),
    }
    assert configured.confirmation == entry.CONFIRMATION
    for case, scenario in (
        ("unknown", entry.SCENARIOS[0]),
        (entry.CASES[0], "unknown"),
    ):
        with pytest.raises(ValueError, match="unsupported"):
            entry.Settings(base, case, scenario, arguments.ordinary_destr015_evidence)


@pytest.mark.parametrize("defect", ["none", "base-preflight", "legacy"])
def test_case_preflight_never_runs_protocol_probe_before_basic_scope_checks(
    monkeypatch, tmp_path, defect
):
    path = (
        tmp_path
        / "ordinary"
        / "cases"
        / "GF-REGIONAL-DESTR-015"
        / "GF-REGIONAL-DESTR-015.json"
    )
    settings = entry.Settings(
        SimpleNamespace(regional=SimpleNamespace(cluster_id="local"), nodes=("a", "b")),
        entry.CASES[0],
        "late-sibling",
        path,
    )
    calls = []
    monkeypatch.setattr(
        entry.normal,
        "read_only_preflight",
        lambda *a, **kw: {
            "errors": ["unapproved context"] if defect == "base-preflight" else []
        },
    )
    monkeypatch.setattr(live, "RegionalLiveFixture", lambda value: value)
    monkeypatch.setattr(
        live, "read_ordinary_completion", lambda *a, **kw: {"cleanup_verified": True}
    )

    def inspect(*args):
        calls.append("probe")
        if defect == "legacy":
            raise ValueError("private diagnostic")
        return {"protocol_ready": True}

    monkeypatch.setattr(live, "inspect_protocol", inspect)
    result = entry.read_only_preflight(
        settings, tmp_path / "companion" / "cases" / entry.CASES[0]
    )
    if defect == "base-preflight":
        assert calls == [] and result["errors"] == ["unapproved context"]
    elif defect == "legacy":
        assert result["errors"] == ["final ownership protocol preflight: ValueError"]
        assert "private diagnostic" not in json.dumps(result)
    else:
        assert result["late_ownership"] == {
            "protocol_ready": True,
            "case_id": entry.CASES[0],
            "scenario": "late-sibling",
            "ordinary_destr015": {"cleanup_verified": True},
        }


def test_companion_plan_does_not_weaken_normal_case_or_promote_its_verdict(
    monkeypatch, tmp_path
):
    settings = entry.Settings(object(), entry.CASES[1], "ownership-drift", tmp_path)
    monkeypatch.setattr(entry.normal, "plan_details", lambda *a: {"normal": "retained"})
    value = entry.plan_details(settings, {})
    assert value["normal"] == "retained"
    assert value["boundary"] == "AGENT_PRE_SPAWN"
    assert value["promotes_ordinary_case"] is False
    assert value["related_cases"] == list(entry.CASES)
    assert "provider replacement is authorized" in value["mutation"]
    assert value["rollback"]["unproven_quiescence_requires_operator_reconciliation"]


@pytest.mark.parametrize("overlap", ["inside", "ordinary", "nested", "separate"])
def test_companion_storage_cannot_overwrite_or_nest_in_ordinary_results(
    tmp_path, overlap
):
    ordinary = tmp_path / "ordinary"
    marker = ordinary / "cases" / entry.CASES[0] / f"{entry.CASES[0]}.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"verdict":"PASS"}')
    root = (
        ordinary
        if overlap == "ordinary"
        else ordinary / "nested"
        if overlap == "nested"
        else tmp_path / "companion"
    )
    predecessor = (
        root / "predecessor.json"
        if overlap == "inside"
        else ordinary / "predecessor.json"
    )
    if overlap == "separate":
        entry.require_companion_directory(root, predecessor)
    else:
        with pytest.raises(ValueError, match="separate|overlaps"):
            entry.require_companion_directory(root, predecessor)
    assert marker.read_text() == '{"verdict":"PASS"}'


def test_entry_assembly_uses_existing_guard_and_never_calls_a_runner_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(
        entry, "run_selected_case", lambda case: calls.append(case) or 17
    )
    monkeypatch.setattr(entry, "run_case_main", lambda function: function())
    assert entry.main() == 17 and calls == [entry.CASE]
    from scripts.e2e.regional import run_late_ownership_acceptance as runner

    monkeypatch.setattr(entry, "main", lambda: 23)
    assert runner.main() == 23


def test_module_footer_delegates_to_mocked_guard_without_any_cli_or_live_io(
    monkeypatch,
):
    from scripts.e2e.regional import run_late_ownership_acceptance as runner

    calls = []
    monkeypatch.setattr(entry, "main", lambda: calls.append("mocked-guard") or 23)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(runner.__file__, run_name="__main__")
    assert caught.value.code == 23
    assert calls == ["mocked-guard"]


@pytest.mark.parametrize("role", ["executor", "node", "reset-interval"])
def test_bundle_contains_only_pinned_owned_acceptance_sources(monkeypatch, role):
    program, digest = bundle.probe_program(role)
    assert len(digest) == 64
    assert "node_action_secret" not in program
    modules, roots = [], []

    def load(name):
        modules.append(name)
        root = sys.path[0]
        roots.append(root)
        from pathlib import Path

        files = sorted(
            path.relative_to(root).as_posix() for path in Path(root).rglob("*.py")
        )
        assert all(name.startswith("scripts/") for name in files), (
            "the probe bundle may contain only acceptance-side source modules"
        )
        assert not any("src/gpu_fault" in name for name in files), (
            "bundling product modules could replace the installed guard under test"
        )
        assert bundle.ENTRIES[role] in files
        return SimpleNamespace(main=lambda: 19)

    monkeypatch.setattr(importlib, "import_module", load)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as caught:
        exec(compile(program, "<owned-bundle-test>", "exec"), {})
    assert caught.value.code == 19
    assert modules == [bundle.ENTRIES[role].removesuffix(".py").replace("/", ".")]
    from pathlib import Path

    assert not Path(roots[0]).exists(), (
        "temporary probe sources must be removed on exit"
    )


def test_tampered_bundle_cannot_import_or_run(monkeypatch):
    program, digest = bundle.probe_program("node")
    monkeypatch.setattr(
        importlib, "import_module", lambda *a: pytest.fail("tampered code imported")
    )
    with pytest.raises(SystemExit) as caught:
        exec(
            compile(
                program.replace(repr(digest), repr("0" * 64)),
                "<tampered-bundle>",
                "exec",
            ),
            {},
        )
    assert caught.value.code == 125


@pytest.mark.parametrize("defect", ["none", "changed", "truncated"])
def test_small_loader_hashes_exact_source_bytes_before_using_stdin_protocol(
    monkeypatch, capsys, defect
):
    program = "import sys\nassert sys.stdin.readline() == 'owned-payload\\n'\nprint('owned-result')\n"
    source = (
        program
        if defect == "none"
        else program.replace("owned-result", "other-result")
        if defect == "changed"
        else program[:12]
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(source + ("owned-payload\n" if defect != "truncated" else "")),
    )
    loader = bundle.stdin_loader(program)
    assert len(loader) < 512
    if defect == "none":
        exec(compile(loader, "<owned-loader>", "exec"), {})
        assert capsys.readouterr().out == "owned-result\n"
    else:
        with pytest.raises(SystemExit) as caught:
            exec(compile(loader, "<owned-loader>", "exec"), {})
        assert caught.value.code == 125 and capsys.readouterr().out == ""


@pytest.fixture
def executor_pod():
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "executor",
            "namespace": "owned",
            "uid": "original",
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "ReplicaSet",
                    "name": "executor-rs",
                    "uid": "rs-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {
            "containers": [
                {"name": "executor", "image": "registry/exec@sha256:" + "a" * 64}
            ]
        },
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "executor",
                    "ready": True,
                    "containerID": "container",
                    "imageID": "docker-pullable://registry/exec@sha256:" + "a" * 64,
                }
            ],
        },
    }


@pytest.fixture
def executor_resources(executor_pod):
    template = {"spec": deepcopy(executor_pod["spec"])}
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "gpu-fault-cluster-executor",
            "namespace": "owned",
            "uid": "deployment-uid",
        },
        "spec": {"template": template},
    }
    replica = {
        "apiVersion": "apps/v1",
        "kind": "ReplicaSet",
        "metadata": {
            "name": "executor-rs",
            "namespace": "owned",
            "uid": "rs-uid",
            "ownerReferences": [
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "gpu-fault-cluster-executor",
                    "uid": "deployment-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {"template": deepcopy(template)},
    }
    return {"pod": executor_pod, "replicaset": replica, "deployment": deployment}


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "absent",
        "uid",
        "deleting",
        "ready",
        "duplicate",
        "container",
        "image",
        "name",
        "namespace",
        "list-uid",
        "running-image",
        "pod-image",
        "rs-uid",
        "deployment-uid",
        "missing-spec",
        "phase",
        "condition",
        "gvk",
    ],
)
def test_executor_identity_pins_exact_ready_container(executor_resources, defect):
    document = executor_resources["pod"]
    replica = executor_resources["replicaset"]
    deployment = executor_resources["deployment"]
    pods = [{"name": "executor", "uid": "original"}]
    if defect == "absent":
        pods = []
    elif defect == "uid":
        document["metadata"].pop("uid")
    elif defect == "deleting":
        document["metadata"]["deletionTimestamp"] = "now"
    elif defect == "ready":
        document["status"]["containerStatuses"][0]["ready"] = False
    elif defect == "duplicate":
        document["status"]["containerStatuses"] *= 2
    elif defect == "container":
        document["status"]["containerStatuses"][0].pop("containerID")
    elif defect == "image":
        document["status"]["containerStatuses"][0].pop("imageID")
    elif defect in {"name", "namespace"}:
        document["metadata"][defect] = "foreign"
    elif defect == "list-uid":
        pods[0]["uid"] = "foreign"
    elif defect == "running-image":
        document["status"]["containerStatuses"][0]["imageID"] = "sha256:" + "b" * 64
    elif defect == "pod-image":
        document["spec"]["containers"][0]["image"] = "foreign-image"
    elif defect == "rs-uid":
        replica["metadata"]["uid"] = "foreign"
    elif defect == "deployment-uid":
        deployment["metadata"]["uid"] = "foreign"
    elif defect == "missing-spec":
        document.pop("spec")
    elif defect == "phase":
        document["status"]["phase"] = "Pending"
    elif defect == "condition":
        document["status"]["conditions"][0]["status"] = "False"
    elif defect == "gvk":
        document["kind"] = "Service"

    def kubectl(*args):
        return json.dumps(executor_resources[args[2]])

    regional = SimpleNamespace(
        settings=SimpleNamespace(namespace="owned"),
        ready_pods=lambda *a: pods,
        kubectl=kubectl,
    )
    if defect == "none":
        assert live.executor_identity(regional) == {
            "name": "executor",
            "uid": "original",
            "container_id": "container",
            "image_id": "docker-pullable://registry/exec@sha256:" + "a" * 64,
            "image": "registry/exec@sha256:" + "a" * 64,
            "replicaset_uid": "rs-uid",
            "deployment_uid": "deployment-uid",
        }
    else:
        with pytest.raises(BoundaryDenied):
            live.executor_identity(regional)


@pytest.mark.parametrize(
    "resource,path,value",
    [
        ("pod", ("apiVersion",), "foreign/v1"),
        ("pod", ("status", "containerStatuses", 0, "imageID"), " "),
        ("pod", ("status", "containerStatuses", 0, "imageID"), 17),
        ("pod", ("metadata", "ownerReferences"), []),
        ("pod", ("metadata", "ownerReferences", 0, "apiVersion"), "foreign/v1"),
        ("pod", ("metadata", "ownerReferences", 0, "kind"), "Job"),
        ("pod", ("metadata", "ownerReferences", 0, "controller"), False),
        ("pod", ("metadata", "ownerReferences", 0, "name"), ""),
        ("pod", ("metadata", "ownerReferences", 0, "uid"), ""),
        ("replicaset", ("apiVersion",), "foreign/v1"),
        ("replicaset", ("kind",), "Deployment"),
        ("replicaset", ("metadata", "name"), "foreign"),
        ("replicaset", ("metadata", "namespace"), "foreign"),
        ("replicaset", ("metadata", "uid"), ""),
        ("replicaset", ("metadata", "deletionTimestamp"), "now"),
        ("replicaset", ("metadata", "ownerReferences"), []),
        ("replicaset", ("metadata", "ownerReferences", 0, "apiVersion"), "foreign/v1"),
        ("replicaset", ("metadata", "ownerReferences", 0, "kind"), "Job"),
        ("replicaset", ("metadata", "ownerReferences", 0, "controller"), False),
        ("replicaset", ("metadata", "ownerReferences", 0, "name"), "foreign"),
        ("replicaset", ("spec", "template", "spec", "containers"), []),
        ("replicaset", ("spec", "template", "spec", "containers", 0, "image"), ""),
        ("deployment", ("metadata", "namespace"), "foreign"),
        (
            "deployment",
            ("spec", "template", "spec", "containers", 0, "image"),
            "foreign",
        ),
    ],
)
def test_executor_owner_gvk_namespace_and_image_are_checked_before_exec(
    executor_resources, resource, path, value
):
    current = executor_resources[resource]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    calls = []

    def kubectl(*args):
        calls.append(args)
        assert args[:2] == ("gpu", "get"), "identity refusal must never exec"
        return json.dumps(executor_resources[args[2]])

    regional = SimpleNamespace(
        settings=SimpleNamespace(namespace="owned"),
        ready_pods=lambda *a: [{"name": "executor", "uid": "original"}],
        kubectl=kubectl,
    )
    with pytest.raises(BoundaryDenied):
        live.executor_identity(regional)
    assert calls and len(calls) <= 3


def test_executor_identity_retains_running_image_id_for_non_digest_declarations(
    executor_resources,
):
    executor_resources["pod"]["spec"]["containers"][0]["image"] = (
        "registry/exec:approved"
    )
    for kind in ("replicaset", "deployment"):
        executor_resources[kind]["spec"]["template"]["spec"]["containers"][0][
            "image"
        ] = "registry/exec:approved"
    regional = SimpleNamespace(
        settings=SimpleNamespace(namespace="owned"),
        ready_pods=lambda *a: [{"name": "executor", "uid": "original"}],
        kubectl=lambda *args: json.dumps(executor_resources[args[2]]),
    )
    result = live.executor_identity(regional)
    assert result["image"] == "registry/exec:approved"
    assert (
        result["image_id"]
        == executor_resources["pod"]["status"]["containerStatuses"][0]["imageID"]
    )


@pytest.mark.parametrize(
    "defect", ["none", "replaced", "legacy", "cluster", "nodes", "shape"]
)
def test_readonly_protocol_inspection_uses_small_loader_and_current_uid(
    monkeypatch, defect
):
    calls = []
    identity = {"name": "executor", "uid": "original"}
    reads = iter(
        [identity, identity | {"uid": "new"} if defect == "replaced" else identity]
    )
    monkeypatch.setattr(live, "executor_identity", lambda regional: next(reads))
    monkeypatch.setattr(live, "probe_program", lambda role: ("pass\n", "a" * 64))
    value = {"protocol_ready": True, "cluster_id": "local", "nodes": ["a", "b"]}
    if defect == "legacy":
        value["protocol_ready"] = False
    elif defect == "cluster":
        value["cluster_id"] = "other"
    elif defect == "nodes":
        value["nodes"] = ["other"]
    elif defect == "shape":
        value = []

    def kubectl(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(value)

    settings = SimpleNamespace(
        regional=SimpleNamespace(cluster_id="local"), nodes=("a", "b")
    )
    if defect == "none":
        assert live.inspect_protocol(settings, SimpleNamespace(kubectl=kubectl)) == {
            "executor": identity,
            "bundle_sha256": "a" * 64,
            "protocol_ready": True,
        }
    else:
        with pytest.raises(BoundaryDenied):
            live.inspect_protocol(settings, SimpleNamespace(kubectl=kubectl))
    args, kwargs = calls[0]
    assert args[-1] == bundle.stdin_loader("pass\n")
    assert kwargs["input_text"].startswith("pass\n"), (
        "the measured source must precede its framed inspection request on stdin"
    )
    request = json.loads(kwargs["input_text"][len("pass\n") :])
    assert request == {"inspect_only": True, "cluster_id": "local", "nodes": ["a", "b"]}


@pytest.mark.parametrize("defect", ["none", "source", "pods", "owners", "one-node"])
def test_live_scope_binds_source_participants_node_boots_and_release(
    monkeypatch, defect
):
    proof = evidence()
    binding = proof.scope
    source = {
        "metadata": {
            "name": binding.workload.name,
            "uid": binding.workload.uid,
            "ownerReferences": [],
        }
    }
    if defect == "source":
        source = None
    elif defect == "owners":
        source["metadata"]["ownerReferences"] = [{"uid": "other"}]
    pods = [
        {"uid": pod.pod_uid, "node": binding.nodes[index].name}
        for index, pod in enumerate(binding.participants)
    ]
    if defect == "pods":
        pods[0]["uid"] = "replaced"
    monkeypatch.setattr(live, "read_resource", lambda *a: source)
    monkeypatch.setattr(
        live, "executor_identity", lambda *a: {"uid": binding.executor_uid}
    )
    from scripts.e2e.regional import live_driver_guard

    monkeypatch.setattr(
        live_driver_guard, "source_digest", lambda: binding.source_sha256
    )
    run = SimpleNamespace(
        regional=SimpleNamespace(
            kubectl=lambda *a: json.dumps({"metadata": {"uid": binding.namespace_uid}})
        ),
        settings=SimpleNamespace(
            nodes=tuple(node.name for node in binding.nodes)[
                : 1 if defect == "one-node" else 2
            ],
            regional=SimpleNamespace(
                namespace=binding.workload.namespace,
                region=binding.region,
                gpu_context=binding.context,
                cluster_id=binding.cluster_id,
            ),
            attempt_id=binding.workload.attempt_id,
        ),
        workload=SimpleNamespace(name=binding.workload.name, pods=lambda: pods),
        source_uids={pod.pod_uid for pod in binding.participants},
        baselines={node.name: {"boot_id": node.boot_id} for node in binding.nodes},
        preflight={
            "nodes": {node.name: {"uid": node.uid} for node in binding.nodes},
            "release_id": binding.release_id,
            "store": {"profile": {"profile_version": binding.runtime_profile}},
        },
        run_id=binding.run_id,
        maintenance_window_end=binding.maintenance_end,
    )
    if defect == "none":
        actual, observed, executor = live.build_scope(
            run, case_id=binding.case_id, scenario=binding.scenario
        )
        assert actual.workload.uid == binding.workload.uid
        assert actual.workload.owner_uid == binding.workload.uid
        assert (
            actual.nodes == binding.nodes
            and actual.source_sha256 == binding.source_sha256
        )
        assert len(actual.challenge) == 64 and actual.execution_epoch == 1
        assert observed is source and executor["uid"] == binding.executor_uid
    else:
        with pytest.raises(BoundaryDenied):
            live.build_scope(run, case_id=binding.case_id, scenario=binding.scenario)
