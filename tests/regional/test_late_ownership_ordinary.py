from __future__ import annotations

import errno
import hashlib
import json
import stat
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_entry as entry
from scripts.e2e.regional import late_ownership_live as live
from scripts.e2e.regional import late_ownership_ordinary as binding
from scripts.e2e.regional import run_destr015_parallel_branch_join as ordinary
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from tests.regional._late_ownership_support import evidence


@pytest.fixture
def completed(tmp_path):
    nodes = ("node-a", "node-b")
    preflight = {
        "errors": [],
        "release_id": "release-local",
        "nodes": {
            node: {"uid": "uid-" + node, "boot_id": "boot-" + node} for node in nodes
        },
        "runtime_identity": {"release_state": {"release_id": "release-local"}},
        "store": {"profile": {"profile_version": "profile-local"}},
    }
    components = ordinary.evidence_components(
        {
            "preflight_identity": ordinary.plan_identity(preflight, nodes=nodes),
            "workflow": {"status": "SUCCEEDED"},
            "incident": {"state": "RECOVERED"},
            "hosts": {node: {"verified": True} for node in nodes},
        }
    )
    value = {
        "case_id": ordinary.CASE_ID,
        "verdict": "PASS",
        "errors": [],
        "release_id": preflight["release_id"],
        "cluster_id": "cluster-local",
        "nodes": list(nodes),
        "components": components,
        "case_digest": ordinary.case_digest(components),
        "cleanup": {
            "errors": [],
            "quiescence": {"safe_to_delete": True},
            "workload_residual": {"residual": False},
            "prewarm_cleanup": {},
            **{
                f"probe_cleanup_{node}": {
                    f"pod/probe-{node}": False,
                    f"configmap/probe-{node}": False,
                    "host_script": False,
                    "creation_unresolved": False,
                }
                for node in nodes
            },
            "restore_isolated_nodes": {
                nodes[0]: {"isolated": False},
                nodes[1]: {"isolated": True, "restore": "SUCCEEDED"},
            },
            "runtime_identity": deepcopy(preflight["runtime_identity"]),
        },
    }
    path = (
        tmp_path / "ordinary" / "cases" / ordinary.CASE_ID / f"{ordinary.CASE_ID}.json"
    )
    path.parent.mkdir(parents=True)

    def save():
        path.write_text(json.dumps(value), encoding="utf-8")

    def read():
        return binding.read_ordinary_completion(
            path, preflight=preflight, cluster_id="cluster-local", nodes=nodes
        )

    save()
    return SimpleNamespace(
        path=path, preflight=preflight, value=value, read=read, save=save, nodes=nodes
    )


def test_explicit_ordinary_pass_and_cleanup_bind_current_release_and_nodes(completed):
    result = completed.read()
    assert result == {
        "path": str(completed.path),
        "sha256": hashlib.sha256(completed.path.read_bytes()).hexdigest(),
        "case_id": ordinary.CASE_ID,
        "case_digest": completed.value["case_digest"],
        "release_id": "release-local",
        "cluster_id": "cluster-local",
        "nodes": list(completed.nodes),
        "cleanup_verified": True,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "GF-REGIONAL-DESTR-012"),
        ("verdict", "FAIL"),
        ("verdict", "NOT_RUN"),
        ("status", "RUNNING"),
        ("errors", None),
        ("errors", ["cleanup failed"]),
        ("error", "interrupted"),
        ("evidence_mode", "LOCAL_TEST"),
        ("subproof", "physical-late-ownership"),
        ("execution_scope", "selective"),
        ("formal_sequence_satisfied", False),
        ("release_id", "foreign"),
        ("cluster_id", "foreign"),
        ("nodes", ["node-b", "node-a"]),
        ("nodes", ["node-a", "foreign"]),
    ],
)
def test_failed_unclosed_or_foreign_ordinary_result_is_not_authority(
    completed, field, value
):
    completed.value[field] = value
    completed.save()
    with pytest.raises(BoundaryDenied, match="completed PASS"):
        completed.read()


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "extra",
        "wrong-identity",
        "digest",
        "shape",
        "component-type",
        "component-short",
        "component-hex",
    ],
)
def test_ordinary_component_digests_cannot_hide_foreign_runtime_or_incomplete_proof(
    completed, defect
):
    components = completed.value["components"]
    if defect == "missing":
        components.pop("workflow")
    elif defect == "extra":
        components["unrecognized"] = "e" * 64
    elif defect == "wrong-identity":
        components["preflight_identity"] = "e" * 64
    elif defect == "digest":
        completed.value["case_digest"] = "f" * 64
    elif defect == "shape":
        completed.value["components"] = []
    else:
        components["workflow"] = {
            "component-type": None,
            "component-short": "abc",
            "component-hex": "g" * 64,
        }[defect]
    if defect != "digest":
        completed.value["case_digest"] = ordinary.case_digest(
            completed.value["components"]
        )
    completed.save()
    with pytest.raises(BoundaryDenied, match="identity differs"):
        completed.read()


@pytest.mark.parametrize(
    "field", ["release_id", "node-uid", "node-boot", "profile", "runtime"]
)
def test_ordinary_evidence_is_bound_to_fresh_target_identity(completed, field):
    current = completed.preflight
    if field == "release_id":
        current[field] = "other"
    elif field == "node-uid":
        current["nodes"]["node-a"]["uid"] = "replacement"
    elif field == "node-boot":
        current["nodes"]["node-b"]["boot_id"] = "rebooted"
    elif field == "profile":
        current["store"]["profile"]["profile_version"] = "other"
    else:
        current["runtime_identity"] = {"foreign": True}
    with pytest.raises(BoundaryDenied):
        completed.read()


@pytest.mark.parametrize(
    "field,value",
    [
        ("errors", ["still running"]),
        ("errors", None),
        ("quiescence", None),
        ("quiescence", {"safe_to_delete": False}),
        ("workload_residual", None),
        ("workload_residual", {"residual": True}),
        ("workload_cleanup_deferred", True),
        ("isolation_restore_deferred", True),
        ("runtime_identity", {"different": True}),
        ("prewarm_cleanup", None),
        ("prewarm_cleanup", {"pod": True}),
        ("probe_cleanup_node-a", None),
        ("probe_cleanup_node-b", {"pod": "false"}),
        ("restore_isolated_nodes", None),
        ("restore_isolated_nodes", {"node-a": {"isolated": False}}),
        ("restore_isolated_nodes", {"node-a": {}, "node-b": {"isolated": False}}),
        ("restore_isolated_nodes", {"node-a": [], "node-b": {"isolated": False}}),
        (
            "restore_isolated_nodes",
            {
                "node-a": {"isolated": True, "restore": "RUNNING"},
                "node-b": {"isolated": False},
            },
        ),
    ],
)
def test_ordinary_cleanup_requires_confirmed_quiescence_and_every_resource(
    completed, field, value
):
    completed.value["cleanup"][field] = value
    completed.save()
    with pytest.raises(BoundaryDenied):
        completed.read()


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "pod",
        "configmap",
        "host_script",
        "creation_unresolved",
        "unknown-kind",
        "empty-name",
        "nested-name",
        "extra",
        "unresolved",
        "script",
    ],
)
def test_probe_cleanup_cannot_omit_resource_kinds_or_host_custody(completed, defect):
    residuals = completed.value["cleanup"]["probe_cleanup_node-b"]
    if defect == "empty":
        residuals.clear()
    elif defect in {"pod", "configmap"}:
        residuals.pop(f"{defect}/probe-node-b")
    elif defect in {"host_script", "creation_unresolved"}:
        residuals.pop(defect)
    elif defect in {"unknown-kind", "empty-name", "nested-name"}:
        residuals.pop("pod/probe-node-b")
        residuals[
            {
                "unknown-kind": "job/foreign",
                "empty-name": "pod/",
                "nested-name": "pod/ns/name",
            }[defect]
        ] = False
    elif defect == "extra":
        residuals["pod/foreign"] = False
    else:
        residuals[
            "creation_unresolved" if defect == "unresolved" else "host_script"
        ] = True
    completed.save()
    with pytest.raises(BoundaryDenied, match="cleanup"):
        completed.read()


@pytest.mark.parametrize("prewarm", [{}, {"cached-image-pod": False}])
def test_prewarm_can_be_empty_when_both_nodes_are_cached(completed, prewarm):
    completed.value["cleanup"]["prewarm_cleanup"] = prewarm
    completed.save()
    assert completed.read()["cleanup_verified"] is True, (
        "cached prewarm images need no created Pods but still require complete host-probe cleanup"
    )


@pytest.mark.parametrize("part", ["result", "cleanup"])
def test_missing_or_nonobject_result_and_cleanup_are_not_pass(completed, part):
    value = [] if part == "result" else {**completed.value, "cleanup": None}
    completed.path.write_text(json.dumps(value))
    with pytest.raises(BoundaryDenied):
        completed.read()


@pytest.mark.parametrize(
    "kind", ["missing", "nonregular", "oversized", "invalid-json", "nofollow"]
)
def test_ordinary_proof_is_a_bounded_regular_file_without_link_following(
    completed, monkeypatch, kind
):
    real_open = binding.os.open
    calls = []

    def opened(path, flags):
        calls.append(flags)
        if kind == "nofollow":
            raise OSError(errno.ELOOP, "test-only refused link")
        return real_open(path, flags)

    monkeypatch.setattr(binding.os, "open", opened)
    if kind == "missing":
        completed.path.unlink()
    elif kind == "invalid-json":
        completed.path.write_text("{")
    elif kind == "oversized":
        completed.path.write_bytes(b" " * (binding.MAX_EVIDENCE_BYTES + 1))
    elif kind == "nonregular":
        real_stat = binding.os.fstat
        monkeypatch.setattr(
            binding.os,
            "fstat",
            lambda fd: SimpleNamespace(
                st_mode=stat.S_IFIFO, st_size=real_stat(fd).st_size
            ),
        )
    with pytest.raises((OSError, ValueError, BoundaryDenied)):
        completed.read()
    assert len(calls) == 1, (
        "ordinary evidence must not be reopened through a permissive fallback"
    )
    assert calls[0] & binding.os.O_NOFOLLOW and calls[0] & binding.os.O_NONBLOCK, (
        "evidence reads must neither follow final links nor block on nonregular inputs"
    )


@pytest.mark.parametrize("defect", ["size", "same-size", "replaced", "overread"])
def test_changed_ordinary_file_cannot_be_bound_from_partial_or_stale_read(
    completed, monkeypatch, defect
):
    real_fstat = binding.os.fstat
    reads = 0

    def changed(fd):
        nonlocal reads
        reads += 1
        info = real_fstat(fd)
        if reads == 1:
            if defect in {"size", "same-size"}:
                completed.path.write_bytes(
                    b" " * (info.st_size + (1 if defect == "size" else 0))
                )
            elif defect == "replaced":
                replacement = completed.path.with_name("replacement.json")
                replacement.write_bytes(completed.path.read_bytes())
                replacement.replace(completed.path)
            elif defect == "overread":
                monkeypatch.setattr(binding, "MAX_EVIDENCE_BYTES", info.st_size - 1)
        return info

    monkeypatch.setattr(binding.os, "fstat", changed)
    with pytest.raises(BoundaryDenied, match="changed while reading|bounded regular"):
        completed.read()


@pytest.mark.parametrize(
    "relative",
    [
        "result.json",
        "cases/GF-REGIONAL-DESTR-015/result.json",
        "wrong/GF-REGIONAL-DESTR-015/GF-REGIONAL-DESTR-015.json",
        "cases/foreign/GF-REGIONAL-DESTR-015.json",
    ],
)
def test_ordinary_result_must_be_explicit_canonical_artifact(tmp_path, relative):
    with pytest.raises(BoundaryDenied, match="canonical"):
        binding.require_external_ordinary(tmp_path / "companion", tmp_path / relative)


@pytest.mark.parametrize("relation", ["same", "child", "parent", "separate"])
def test_ordinary_and_companion_roots_are_non_nested(completed, tmp_path, relation):
    root = completed.path.parent.parent.parent
    run_dir = {
        "same": root,
        "child": root / "child",
        "parent": tmp_path,
        "separate": tmp_path / "companion",
    }[relation]
    before = completed.path.read_bytes()
    if relation == "separate":
        binding.require_external_ordinary(run_dir, completed.path)
    else:
        with pytest.raises(BoundaryDenied, match="non-nested"):
            binding.require_external_ordinary(run_dir, completed.path)
    assert completed.path.read_bytes() == before


@pytest.mark.parametrize("defect", ["missing", "failed", "unclosed", "foreign", "none"])
def test_entry_requires_ordinary_destr015_before_protocol_probe_or_physical_setup(
    completed, tmp_path, monkeypatch, defect
):
    if defect == "missing":
        completed.path.unlink()
    elif defect != "none":
        if defect == "failed":
            completed.value["verdict"] = "FAIL"
        elif defect == "unclosed":
            completed.value["cleanup"]["quiescence"]["safe_to_delete"] = False
        else:
            completed.value["cluster_id"] = "foreign"
        completed.save()
    calls = []
    settings = entry.Settings(
        SimpleNamespace(
            regional=SimpleNamespace(cluster_id="cluster-local"),
            nodes=completed.nodes,
            predecessor_path=tmp_path / "destr012.json",
        ),
        ordinary.CASE_ID,
        "late-sibling",
        completed.path,
    )
    monkeypatch.setattr(
        ordinary, "read_only_preflight", lambda *a, **kw: deepcopy(completed.preflight)
    )
    monkeypatch.setattr(live, "RegionalLiveFixture", lambda value: value)
    monkeypatch.setattr(
        live,
        "inspect_protocol",
        lambda *a: calls.append("protocol") or {"protocol_ready": True},
    )
    monkeypatch.setattr(
        ordinary, "_start_job_and_probes", lambda run: pytest.fail("physical setup ran")
    )
    directory = tmp_path / "companion"
    result = live.read_only_preflight(settings, directory / "cases" / ordinary.CASE_ID)
    if defect == "none":
        assert calls == ["protocol"] and not result["errors"]
        assert result["late_ownership"]["ordinary_destr015"]["cleanup_verified"] is True
    else:
        assert calls == [], "invalid ordinary completion must prevent protocol probing"
        assert result["errors"] == [
            "ordinary DESTR-015 evidence preflight: "
            + ("FileNotFoundError" if defect == "missing" else "BoundaryDenied")
        ]
        with pytest.raises(live.RegionalFixtureError, match="preflight failed"):
            live.execute_case(settings, directory, 1, evidence().scope.maintenance_end)
        assert calls == [], (
            "execution must also refuse invalid ordinary evidence before any target I/O"
        )


def test_changed_ordinary_completion_changes_companion_plan_identity(completed):
    first = completed.read()
    completed.value["operator_annotation"] = "new local evidence revision"
    completed.save()
    second = completed.read()
    assert first["case_digest"] == second["case_digest"]
    assert first["sha256"] != second["sha256"], (
        "the plan must bind exact external evidence bytes"
    )
    planned = ordinary.identity_digest(
        ordinary.plan_identity(
            {**completed.preflight, "late_ownership": {"ordinary_destr015": first}},
            nodes=completed.nodes,
        )
    )
    current = ordinary.identity_digest(
        ordinary.plan_identity(
            {**completed.preflight, "late_ownership": {"ordinary_destr015": second}},
            nodes=completed.nodes,
        )
    )
    assert planned != current, (
        "execute must not reuse a plan for changed external evidence"
    )
