from __future__ import annotations

import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import (
    live_driver_guard,
    notify008_bundle,
    run_notify008_commit_ambiguity,
)
from scripts.e2e.regional import notify008_runner as runner
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, Target
from tests.regional._cov95_notify008_lifecycle import setup_run, target


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_id", ""),
        ("node", "contains space"),
        ("runtime_version", "v" * 257),
        ("source_pod_uid", None),
        ("runtime_image", "example.invalid/runtime:latest"),
        ("postgres_image", "https://example.invalid/pg@sha256:" + "1" * 64),
        ("runtime_module_digest", "missing"),
        ("deployment_generation", True),
        ("deployment_generation", 0),
        ("seconds", True),
        ("seconds", 299),
        ("seconds", 601),
    ],
)
def test_approved_target_requires_finite_complete_identity(field, value):
    fields = {**asdict(target()), field: value}
    with pytest.raises(ProbeError):
        Target(**fields)


@pytest.mark.parametrize("failure", ["missing-file", "empty-context", "invalid-region"])
def test_configuration_refuses_incomplete_connection_before_transport(
    tmp_path, failure
):
    config = tmp_path / "cpu"
    config.write_text("fake connection")
    values = dict(
        cpu_kubeconfig=str(config),
        cpu_context="cpu",
        namespace="gpu-fault-system",
        cluster_id="cluster-local",
        region="us-west-2",
        postgres_image=target().postgres_image,
    )
    if failure == "missing-file":
        config.unlink()
    elif failure == "empty-context":
        values["cpu_context"] = ""
    else:
        values["region"] = None
    with pytest.raises(ProbeError, match="incomplete"):
        runner.configure(SimpleNamespace(**values))


@pytest.mark.parametrize("failure", ["version", "namespace"])
def test_read_only_preflight_refuses_unsupported_gate_or_existing_namespace(
    tmp_path, monkeypatch, failure
):
    settings, api, case_dir, _ = setup_run(tmp_path, monkeypatch)
    call = api.call
    if failure == "version":

        def old_version(*args, **kwargs):
            return '{"minor":"29+"}' if "--raw" in args else call(*args, **kwargs)

        monkeypatch.setattr(api, "call", old_version)
    else:
        api.objects["namespace"] = {"metadata": {"name": "preexisting"}}
    assert runner.read_only_preflight(settings, case_dir)["errors"], (
        "missing stable scheduling gates or a namespace collision must block preflight"
    )
    assert not any(item[0] in {"create", "patch", "delete"} for item in api.calls), (
        "preflight refusal must remain read-only"
    )


@pytest.mark.parametrize("failure", ["not-inert", "preparation", "report", "pod-uid"])
def test_complete_host_lifecycle_refuses_contradictory_probe_evidence(
    tmp_path, monkeypatch, failure
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    call = api.call

    def changed_reply(*args, **kwargs):
        result = call(*args, **kwargs)
        module = "scripts.e2e.regional.probes.notify008_probe"
        if module not in args:
            return result
        command = args[args.index(module) + 1]
        value = json.loads(result)
        if failure == "not-inert" and command == "inspect":
            value["armed"] = True
        elif failure == "preparation" and command == "prepare":
            value["postgres_major"] = 15
        elif command == "run":
            if failure == "report":
                value["children_reaped"] = False
            elif failure == "pod-uid":
                value["pod_uid"] = "foreign-pod"
        return json.dumps(value)

    monkeypatch.setattr(api, "call", changed_reply)
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "a complete transport lifecycle cannot excuse contradictory probe observations"
    )
    report = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL" and api.objects == {}, (
        "the evidence must remain failed while still cleaning owned resources"
    )
    if failure == "not-inert":
        assert not api.armed, "a non-inert first readback must block ARM entirely"


def test_cleanup_interrupt_is_persisted_and_propagated_without_remote_retry(
    tmp_path, monkeypatch
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    before = []

    def interrupted(self):
        before.extend(api.calls)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.Sandbox, "cleanup", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_case(settings, tmp_path, 1, deadline)
    report = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL" and not report["cleanup"]["namespace_absent"], (
        "an interrupted cleanup must publish non-PASS before propagating the abort"
    )
    assert api.calls == before, "failure serialization must issue no new remote command"


@pytest.mark.parametrize("drift", ["missing", "extra", "symlink"])
def test_bundle_refuses_a_changed_shipped_module_closure(tmp_path, monkeypatch, drift):
    for name in notify008_bundle.REQUIRED:
        (tmp_path / name).write_text("pass\n")
    name = sorted(notify008_bundle.REQUIRED)[0]
    if drift == "missing":
        (tmp_path / name).unlink()
    elif drift == "extra":
        (tmp_path / "notify008_unapproved.py").write_text("pass\n")
    else:
        (tmp_path / name).unlink()
        outside = tmp_path / "outside"
        outside.write_text("pass\n")
        (tmp_path / name).symlink_to(outside)
    monkeypatch.setattr(notify008_bundle, "PROBES", tmp_path)
    with pytest.raises(ProbeError, match="closure differs"):
        notify008_bundle.source_bundle()


def test_script_entry_bootstraps_import_path_and_uses_only_the_guarded_driver(
    monkeypatch,
):
    path = Path(run_notify008_commit_ambiguity.__file__).resolve()
    root = str(path.parents[3])
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != root])
    calls = []
    monkeypatch.setattr(
        live_driver_guard, "run_standard_case", lambda case: calls.append(case) or 0
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(path), run_name="__main__")
    assert raised.value.code == 0 and calls == [runner.CASE], (
        "script-mode startup must delegate exactly once to the approved CaseRunner"
    )
    assert sys.path[0] == root, (
        "standalone startup must insert only its repository root"
    )
