"""Refusal and fallback paths of three runner helpers.

``restore_validated_quarantine`` must refuse a node outside the incident, an
incident that is not parked, an incident whose workflow is still live and a
DSN file that is missing or empty; ``boot020_release_prerequisites`` must fail
closed on every malformed administrator-state input and must only trust
candidate metadata of the recorded shape; ``synthetic_replacement_route`` must
tie ``--open``/``--close`` to their confirmations and must refuse a close
whose restored env does not match the recorded baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import site as admin_site
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    EffectiveRuntimeProfile,
    Environment,
    FaultIncident,
    IncidentState,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import boot020_release_prerequisites as prerequisites
from scripts.e2e.regional import restore_validated_quarantine as restore
from scripts.e2e.regional import synthetic_replacement_route as route
from scripts.e2e.regional.regional_commands import RegionalFixtureError

# --- restore_validated_quarantine ---------------------------------------------


def _profile(cluster_id: str = "cluster-a") -> EffectiveRuntimeProfile:
    return EffectiveRuntimeProfile(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        profile_version="profile-a",
        capabilities=[
            EffectiveCapability(
                capability=CapabilityName.DEEP_DIAGNOSTICS,
                mode=CapabilityMode.DELEGATE,
                owner="site-validation-adapter",
                adapter="regional-cluster-executor",
            ),
            EffectiveCapability(
                capability=CapabilityName.SCHEDULER_DRAIN,
                mode=CapabilityMode.OWN,
                owner="gpu-fault-kubernetes-adapter",
                adapter="regional-cluster-executor",
            ),
        ],
    )


def _incident(
    state: IncidentState = IncidentState.QUARANTINED,
    *,
    workflow_request_id: str | None = None,
) -> FaultIncident:
    return FaultIncident(
        incident_id="inc-a",
        event_id="xid-a",
        event_type="XID",
        cluster_id="cluster-a",
        node_ids=["node-a"],
        policy_version="610",
        policy_source="NVIDIA",
        state=state,
        fencing_token=4,
        workflow_request_id=workflow_request_id,
    )


def _workflow(status: WorkflowStatus) -> WorkflowRequest:
    now = datetime.now(timezone.utc)
    return WorkflowRequest(
        request_id="workflow-previous",
        incident_id="inc-a",
        status=status,
        fencing_token=4,
        created_at=now,
        updated_at=now,
    )


def _build(store: InMemoryStore) -> tuple[FaultIncident, WorkflowRequest]:
    return restore.build_restore_workflow(
        store,
        incident_id="inc-a",
        node_id="node-a",
        reason="validated restore",
        runtime_profile_version="profile-a",
    )


def test_restore_refuses_a_node_outside_the_incident() -> None:
    store = InMemoryStore()
    store.save_profile(_profile())
    store.save_incident(_incident())

    with pytest.raises(ValueError, match="node-z is outside incident inc-a"):
        restore.build_restore_workflow(
            store,
            incident_id="inc-a",
            node_id="node-z",
            reason="validated restore",
            runtime_profile_version="profile-a",
        )


def test_restore_refuses_an_incident_that_is_not_parked() -> None:
    store = InMemoryStore()
    store.save_profile(_profile())
    store.save_incident(_incident(IncidentState.RECOVERED))

    with pytest.raises(ValueError, match="not quarantined/escalated.*RECOVERED"):
        _build(store)


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING],
)
def test_restore_refuses_an_incident_whose_workflow_is_still_live(
    status: WorkflowStatus,
) -> None:
    store = InMemoryStore()
    store.save_profile(_profile())
    store.save_incident_and_workflow(
        _incident(workflow_request_id="workflow-previous"), _workflow(status)
    )

    with pytest.raises(ValueError, match="still has active workflow workflow-previous"):
        _build(store)


def test_restore_proceeds_past_a_finished_predecessor_workflow() -> None:
    store = InMemoryStore()
    store.save_profile(_profile())
    store.save_incident_and_workflow(
        _incident(IncidentState.ESCALATED, workflow_request_id="workflow-previous"),
        _workflow(WorkflowStatus.SUCCEEDED),
    )

    incident, workflow = _build(store)

    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.workflow_request_id == workflow.request_id
    assert workflow.request_id != "workflow-previous"
    assert [step.node_ids for step in workflow.official_steps] == [["node-a"]] * 4


def test_store_dsn_reads_the_configured_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text("postgresql://example/db\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))

    assert restore.store_dsn() == "postgresql://example/db"


def test_store_dsn_refuses_an_empty_path_or_an_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")
    with pytest.raises(RuntimeError, match="DSN file path is empty"):
        restore.store_dsn()

    empty = tmp_path / "empty"
    empty.write_text("   \n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(empty))
    with pytest.raises(RuntimeError, match="DSN file is empty"):
        restore.store_dsn()


def test_store_dsn_missing_configured_file_is_an_error_not_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "absent"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://fallback/db")

    with pytest.raises(FileNotFoundError):
        restore.store_dsn()


def test_store_dsn_falls_back_to_the_environment_without_the_default_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a configured file the default /etc path is consulted; a test
    # host does not carry it, so the environment DSN is the answer.
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://fallback/db")

    assert restore.store_dsn() == "postgresql://fallback/db"


# --- boot020_release_prerequisites --------------------------------------------

RUNTIME = "1.dkr.ecr.us-west-2.amazonaws.com/gpu-fault/runtime-abc@sha256:" + "0" * 64


def _state_dir(
    tmp_path: Path, *, runtime: str = RUNTIME, region: str = "us-west-2"
) -> Path:
    state = tmp_path / "state"
    snapshot = state / "source-snapshots" / "deadbeef"
    dist = snapshot / "repository-1-live" / "dist"
    dist.mkdir(parents=True)
    (dist / "current-release.json").write_text(json.dumps({"release_id": "rel-live"}))
    (state / "source-deploy-success.json").write_text(
        json.dumps(
            {
                "live": {"release_id": "rel-live"},
                "prepared_repository_root": str(snapshot / "repository-1-live"),
            }
        )
    )
    region_line = f"  awsRegion: {region}\n" if region else ""
    (state / "site.yaml").write_text(
        "apiVersion: gpu-fault.aws/v1alpha1\nkind: Site\nspec:\n"
        f"{region_line}  repositoryRoot: {snapshot}\n"
        f"  images:\n    runtime: {runtime}\n"
    )
    return state


def _write_configs(out_dir: Path, manifest: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in prerequisites.CONFIG_FILES.values():
        (out_dir / name).write_text(
            json.dumps(
                {
                    "release": {"manifest": manifest},
                    "runtime_profile": {"version": "hp-v1"},
                }
            )
        )
    return prerequisites.candidate_paths(out_dir)


def test_site_spec_requires_a_site_file_with_a_spec(tmp_path: Path) -> None:
    with pytest.raises(RegionalFixtureError, match="has no site.yaml"):
        prerequisites.site_spec(tmp_path)

    (tmp_path / "site.yaml").write_text("- not a mapping\n")
    with pytest.raises(RegionalFixtureError, match="has no spec"):
        prerequisites.site_spec(tmp_path)


def test_live_release_id_requires_the_deploy_record_and_a_release(
    tmp_path: Path,
) -> None:
    with pytest.raises(RegionalFixtureError, match="no source-deploy-success.json"):
        prerequisites.live_release_id(tmp_path)

    (tmp_path / "source-deploy-success.json").write_text(json.dumps({"live": {}}))
    with pytest.raises(RegionalFixtureError, match="names no live release"):
        prerequisites.live_release_id(tmp_path)


def test_a_non_string_prepared_root_is_ignored_in_favour_of_the_site_root(
    tmp_path: Path,
) -> None:
    state = _state_dir(tmp_path)
    (state / "source-deploy-success.json").write_text(
        json.dumps({"live": {"release_id": "rel-live"}, "prepared_repository_root": 7})
    )

    repo = prerequisites.live_snapshot_repository(state)

    assert repo.name == "repository-1-live"


def test_candidate_inputs_require_a_digest_pinned_runtime_image(tmp_path: Path) -> None:
    state = _state_dir(tmp_path, runtime="example.com/gpu-fault/runtime-abc:latest")

    with pytest.raises(RegionalFixtureError, match="not digest-pinned"):
        prerequisites.candidate_inputs(state)


def test_candidate_inputs_require_a_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state_dir(tmp_path, region="")
    monkeypatch.setattr(
        admin_site,
        "load_site",
        lambda *_a, **_k: SimpleNamespace(
            release_config={"runtime_profile": {"version": "hp-v1"}}
        ),
    )

    with pytest.raises(RegionalFixtureError, match="awsRegion is missing"):
        prerequisites.candidate_inputs(state)


def test_candidate_inputs_derive_cache_repository_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state_dir(tmp_path)
    seen: list[tuple[Path, Path]] = []

    def fake_load_site(path: Path, *, repository_root: Path) -> SimpleNamespace:
        seen.append((path, repository_root))
        return SimpleNamespace(release_config={"runtime_profile": {"version": "hp-v1"}})

    monkeypatch.setattr(admin_site, "load_site", fake_load_site)

    derived = prerequisites.candidate_inputs(state)

    assert derived.region == "us-west-2"
    assert derived.runtime_repository == RUNTIME.split("@")[0]
    assert derived.cache_repository == RUNTIME.split("@")[0].replace(
        "runtime-", "runtime-cache-"
    )
    assert derived.runtime_profile == "hp-v1"
    assert derived.snapshot_repo.name == "repository-1-live"
    assert seen == [(state / "site.yaml", prerequisites.candidates.ROOT)]


def test_candidate_inputs_without_the_cache_infix_have_no_cache_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = "1.dkr.ecr.us-west-2.amazonaws.com/other/image@sha256:" + "1" * 64
    state = _state_dir(tmp_path, runtime=runtime)
    monkeypatch.setattr(
        admin_site,
        "load_site",
        lambda *_a, **_k: SimpleNamespace(
            release_config={"runtime_profile": {"version": "hp-v1"}}
        ),
    )

    assert prerequisites.candidate_inputs(state).cache_repository is None


def test_site_inputs_require_a_site_file(tmp_path: Path) -> None:
    with pytest.raises(RegionalFixtureError, match="has no site.yaml"):
        prerequisites.site_inputs(tmp_path)


def test_recorded_site_inputs_are_only_read_from_a_mapping(tmp_path: Path) -> None:
    (tmp_path / prerequisites.CANDIDATES_METADATA_NAME).write_text(
        json.dumps({"site_inputs": ["not", "a", "mapping"]})
    )

    assert prerequisites.recorded_site_inputs(tmp_path) is None


def test_candidate_metadata_ignores_a_build_summary_that_is_not_a_mapping(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / prerequisites.CANDIDATES_METADATA_NAME).write_text(json.dumps([1, 2]))
    out = tmp_path / "out"
    out.mkdir()

    target = prerequisites.write_candidate_metadata(
        work, out, site={"site_yaml_sha256": "a", "admin_config_sha256": "b"}
    )

    assert json.loads(target.read_text()) == {
        "site_inputs": {"site_yaml_sha256": "a", "admin_config_sha256": "b"}
    }


def test_ensure_refuses_a_build_that_does_not_name_the_live_snapshot(
    tmp_path: Path,
) -> None:
    state = _state_dir(tmp_path)
    out = tmp_path / "candidates"
    snapshot = prerequisites.live_snapshot_repository(state)
    inputs = prerequisites.CandidateInputs(
        snapshot_repo=snapshot,
        region="us-west-2",
        runtime_repository=RUNTIME.split("@")[0],
        cache_repository=None,
        runtime_profile="hp-v1",
        site_file=state / "site.yaml",
    )
    calls: list[str] = []

    def write(namespace: argparse.Namespace) -> None:
        calls.append("write")
        # Five configs exist but name another snapshot's manifest.
        _write_configs(namespace.out_dir, str(tmp_path / "elsewhere.json"))

    with pytest.raises(RegionalFixtureError, match="do not name the live snapshot"):
        prerequisites.ensure_release_candidates(
            state,
            out,
            gpu_kubeconfig=tmp_path / "kubeconfig",
            build=lambda _namespace: calls.append("build"),
            write=write,
            check=lambda _namespace: 0,
            inputs=lambda _state: inputs,
        )

    assert calls == ["build", "write"]


def test_identity_of_a_record_without_locations_still_binds_the_manifests(
    tmp_path: Path,
) -> None:
    state = _state_dir(tmp_path)
    manifest = prerequisites.live_snapshot_repository(state) / "dist"
    configs = _write_configs(
        tmp_path / "configs", str(manifest / "current-release.json")
    )

    identity = prerequisites.candidate_identity(state, configs, {"action": "built"})

    assert identity["source"] == "derived"
    assert "candidates_dir" not in identity
    assert "snapshot_repo" not in identity
    assert "candidates" not in identity
    assert identity["release_ids"] == dict.fromkeys(configs, "rel-live")
    assert identity["runtime_profile"] == "hp-v1"


def test_identity_fails_closed_on_a_config_without_a_manifest(tmp_path: Path) -> None:
    state = _state_dir(tmp_path)
    configs = _write_configs(tmp_path / "configs", "")

    with pytest.raises(RegionalFixtureError, match="names no manifest"):
        prerequisites.candidate_identity(state, configs, {})


def test_identity_fails_closed_on_a_manifest_without_a_release_id(
    tmp_path: Path,
) -> None:
    state = _state_dir(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"release_id": ""}))
    configs = _write_configs(tmp_path / "configs", str(manifest))

    with pytest.raises(RegionalFixtureError, match="names no release_id"):
        prerequisites.candidate_identity(state, configs, {})


def test_identity_ignores_candidate_metadata_that_is_not_a_mapping(
    tmp_path: Path,
) -> None:
    state = _state_dir(tmp_path)
    snapshot = prerequisites.live_snapshot_repository(state)
    out = tmp_path / "candidates"
    configs = _write_configs(out, str(snapshot / "dist" / "current-release.json"))
    (out / prerequisites.CANDIDATES_METADATA_NAME).write_text(
        json.dumps({"candidates": ["B", "C"]})
    )

    identity = prerequisites.candidate_identity(
        state, configs, {"candidates_dir": str(out), "snapshot_repo": str(snapshot)}
    )

    assert identity["candidates"] == {}
    assert identity["snapshot_release_id"] == "rel-live"
    assert identity["candidates_dir"] == str(out)


def test_ensure_for_execute_builds_a_pending_record_with_the_env_kubeconfig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}

    def fake_ensure(state: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
        received.update({"state": state, "out": out, **kwargs})
        return {"action": "built"}

    monkeypatch.setattr(prerequisites, "ensure_release_candidates", fake_ensure)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "env-kubeconfig"))
    arguments = argparse.Namespace(
        admin_state_dir=tmp_path / "state", gpu_kubeconfig=None, replicas_delta=-2
    )

    result = prerequisites.ensure_for_execute(
        arguments,
        {"action": "build-at-execute", "candidates_dir": str(tmp_path / "out")},
    )

    assert result == {"action": "built"}
    assert received["gpu_kubeconfig"] == tmp_path / "env-kubeconfig"
    assert received["out"] == tmp_path / "out"
    assert received["replicas_delta"] == -2


def test_ensure_for_execute_prefers_an_explicit_gpu_kubeconfig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, Any] = {}
    monkeypatch.setattr(
        prerequisites,
        "ensure_release_candidates",
        lambda state, out, **kwargs: received.update(kwargs) or {"action": "built"},
    )
    monkeypatch.delenv("KUBECONFIG", raising=False)
    arguments = argparse.Namespace(
        admin_state_dir=tmp_path,
        gpu_kubeconfig=tmp_path / "explicit",
        replicas_delta=-1,
    )

    prerequisites.ensure_for_execute(
        arguments, {"action": "build-at-execute", "candidates_dir": str(tmp_path)}
    )

    assert received["gpu_kubeconfig"] == tmp_path / "explicit"


# --- synthetic_replacement_route ----------------------------------------------


class _Regional:
    """A control plane answering ``kubectl`` from a scripted env state.

    ``apply_set`` False models an API tier whose ``set env`` is acknowledged
    but never lands, which is what the restored-state comparison exists for.
    """

    def __init__(
        self, *, present: bool, value: str | None, apply_set: bool = True
    ) -> None:
        self.present = present
        self.value = value
        self.apply_set = apply_set
        self.commands: list[tuple[str, ...]] = []
        self.settings = SimpleNamespace(cluster_id="cluster-a")

    def _env(self) -> list[dict[str, Any]]:
        if not self.present:
            return []
        return [{"name": route.ROUTE_ENV, "value": self.value}]

    def kubectl(self, plane: str, *arguments: str, **_kwargs: Any) -> str:
        assert plane == "cpu", plane
        self.commands.append(arguments)
        if arguments[0] == "get":
            return json.dumps(
                {
                    "metadata": {
                        "uid": "uid-a",
                        "generation": 7,
                        "resourceVersion": "1",
                    },
                    "spec": {
                        "replicas": 2,
                        "template": {
                            "spec": {
                                "containers": [
                                    {"name": route.CONTAINER, "env": self._env()}
                                ]
                            }
                        },
                    },
                    "status": {
                        "observedGeneration": 7,
                        "replicas": 2,
                        "readyReplicas": 2,
                        "updatedReplicas": 2,
                        "availableReplicas": 2,
                    },
                }
            )
        if arguments[0] == "set":
            if self.apply_set:
                assignment = arguments[-1]
                if assignment.endswith("-"):
                    self.present, self.value = False, None
                else:
                    self.present, self.value = True, assignment.split("=", 1)[1]
            return ""
        if arguments[0] == "rollout":
            return "deployment rolled out\n"
        if arguments[0] == "exec":
            return json.dumps({"enabled": self.value if self.present else None})
        raise AssertionError(arguments)

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        assert (plane, app) == ("cpu", route.DEPLOYMENT), (plane, app)
        return [{"name": "api-a"}, {"name": "api-b"}]


def _settings(tmp_path: Path) -> route.Settings:
    return route.Settings(
        baseline=tmp_path / "synthetic-route.json", rollout_timeout_seconds=0
    )


def test_a_closed_record_does_not_block_a_fresh_open(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.baseline.write_text(
        json.dumps(
            {
                "opened_at": "2026-09-01T00:00:00Z",
                "closed_at": "2026-09-01T01:00:00Z",
                "baseline": {"route_env_present": False, "route_env_value": None},
            }
        )
    )
    regional = _Regional(present=False, value=None)

    record = route.open_window(
        settings, regional, route.survey(regional), sleep=lambda _s: None
    )

    assert "closed_at" not in record
    assert "resumed_at" not in record
    assert record["baseline"]["route_env_present"] is False
    assert record["opened_state"]["route_env_value"] == "true"
    assert any(command[:2] == ("set", "env") for command in regional.commands), (
        regional.commands
    )


def test_a_close_whose_env_did_not_restore_is_refused_after_recording(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.baseline.write_text(
        json.dumps(
            {
                "opened_at": "2026-09-01T00:00:00Z",
                "baseline": {"route_env_present": True, "route_env_value": "false"},
            }
        )
    )
    regional = _Regional(present=True, value="true", apply_set=False)

    with pytest.raises(RegionalFixtureError, match="does not match the recorded"):
        route.close_window(
            settings, regional, route.survey(regional), sleep=lambda _s: None
        )

    record = json.loads(settings.baseline.read_text())
    assert record["closed_at"], record
    assert record["restored_state"]["route_env_value"] == "true"


def _run_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str
) -> tuple[dict[str, Any], list[str]]:
    calls: list[str] = []
    report = {"deployment": {"route_env_value": None}, "pod_gates": []}
    monkeypatch.setattr(route, "install_site_profile", lambda: calls.append("profile"))
    monkeypatch.setattr(route, "install_abort_signals", lambda: calls.append("signals"))
    monkeypatch.setattr(route, "settings_from_arguments", lambda _arguments: None)
    monkeypatch.setattr(route, "RegionalLiveFixture", lambda _settings: object())
    monkeypatch.setattr(route, "survey", lambda _regional: dict(report))
    monkeypatch.setattr(
        route,
        "open_window",
        lambda *_a, **_k: calls.append("open") or {"opened_at": "t", "x_survey": {}},
    )
    monkeypatch.setattr(
        route,
        "close_window",
        lambda *_a, **_k: calls.append("close") or {"closed_at": "t", "y_survey": {}},
    )
    monkeypatch.setattr(
        sys, "argv", ["route", "--baseline", str(tmp_path / "baseline.json"), *argv]
    )
    return report, calls


def test_main_is_read_only_without_a_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _report, calls = _run_main(monkeypatch, tmp_path)

    assert route.main() == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "read-only"
    assert calls == ["profile", "signals"]
    assert not (tmp_path / "report.json").exists(), "read-only mode writes no report"


def test_main_open_requires_its_exact_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _report, calls = _run_main(monkeypatch, tmp_path, "--open", "--confirm", "nope")

    with pytest.raises(RegionalFixtureError, match="--open requires --confirm"):
        route.main()

    assert "open" not in calls


def test_main_close_requires_its_exact_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _report, calls = _run_main(monkeypatch, tmp_path, "--close")

    with pytest.raises(RegionalFixtureError, match="--close requires --confirm"):
        route.main()

    assert "close" not in calls


def test_main_open_and_close_write_the_survey_free_record_and_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    _report, calls = _run_main(
        monkeypatch,
        tmp_path,
        "--open",
        "--confirm",
        route.OPEN_CONFIRMATION,
        "--report",
        str(report_path),
    )
    assert route.main() == 0
    written = json.loads(report_path.read_text())
    assert written["open"] == {"opened_at": "t"}
    assert calls[-1] == "open"
    capsys.readouterr()

    _report, calls = _run_main(
        monkeypatch, tmp_path, "--close", "--confirm", route.CLOSE_CONFIRMATION
    )
    assert route.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["close"] == {"closed_at": "t"}
    assert calls[-1] == "close"
