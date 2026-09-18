from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault_release.regional_release_diff import ReleaseChangeKind
from scripts.e2e.regional import run_boot023_release_history as runner
from tests.regional._cov95_boot_history import HistoryModel


@pytest.mark.parametrize("missing", [False, True])
def test_public_history_read_distinguishes_absence_from_transport_failure(
    missing, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.absent = missing
    assert runner.history_entries(model.regional) == (
        [] if missing else model.history
    ), "only an acknowledged empty API result means no history exists"
    model.read_failure = True
    with pytest.raises(runner.RegionalFixtureError, match="history read failed"):
        runner.history_entries(model.regional)


@pytest.mark.parametrize("missing_port", [False, True])
def test_registry_probes_use_each_pods_declared_http_port(
    missing_port, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.missing_port = missing_port
    if missing_port:
        with pytest.raises(
            runner.RegionalFixtureError, match="no containerPort named http"
        ):
            runner.registry_probes(model.regional, ("api",))
        assert model.probe_calls == [], "missing port must stop before exec"
    else:
        values = runner.registry_probes(model.regional, ("api", "worker"))
        assert [(item["app"], item["pod"]) for item in values] == [
            ("api", "api-pod"),
            ("worker", "worker-pod"),
        ], "registry evidence must retain application and Pod identity"
        assert all(
            args[-1] == "8080" and kwargs["timeout"] == 120
            for args, kwargs in model.probe_calls
        ), "the probe must use the declared HTTP port with a bounded command"


@pytest.mark.parametrize("test_code", [0, 1])
def test_preflight_collects_current_history_registry_and_local_test_evidence(
    test_code, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.test_code = test_code
    preflight = model.plan(tmp_path)
    assert preflight["classification"]["kind"] == "NOOP", (
        "bind the actual release classification"
    )
    assert preflight["focused_tests"]["passed"] is (test_code == 0)
    assert bool(preflight["errors"]) is bool(test_code), (
        "test failures must remain preflight errors"
    )
    assert len(preflight["registry_probes"]) == 3, (
        "every CPU runtime role must be observed"
    )
    assert model.actions == [], "preflight may not execute the NOOP release"
    assert len(model.focused_calls) == 1, (
        "focused regression tests are a recorded fake subprocess"
    )
    path = tmp_path / "cases" / runner.CASE_ID / "focused-tests.log"
    assert path.stat().st_mode & 0o777 == 0o600, (
        "captured test output must remain private"
    )


@pytest.mark.parametrize("mirror", [False, True])
def test_noop_history_lifecycle_requires_matching_append_and_mirror(
    mirror, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.plan(tmp_path)
    model.write_mirror = mirror
    monkeypatch.setenv(runner.HISTORY_DIR_ENV, str(tmp_path / "caller-history"))
    code = runner.execute_case(
        model.settings, tmp_path, 2, datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    assert code == int(not mirror), (
        "a missing mirror must fail otherwise correct history append"
    )
    path = tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json"
    result = json.loads(path.read_text())
    assert result["verdict"] == ("PASS" if mirror else "FAIL"), (
        "case verdict must include the mirror proof"
    )
    assert model.actions == ["noop"], "execute exactly one approved NOOP"
    assert result["history_entries_after"] == result["history_entries_before"] + 1
    assert runner.os.environ[runner.HISTORY_DIR_ENV] == str(
        tmp_path / "caller-history"
    ), "the caller's history destination must survive the attempt"
    assert result["cleanup"]["errors"] == [], (
        "final runtime identity must be revalidated"
    )


def test_non_noop_classification_stops_before_release_execution(
    tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.kind = ReleaseChangeKind.FULL
    with pytest.raises(runner.RegionalFixtureError, match="only runs a NOOP"):
        runner.run_noop_release(model.settings, model.modules, tmp_path)
    assert model.actions == [], (
        "a changed release must not enter the NOOP acceptance path"
    )


@pytest.mark.parametrize("failure", ["preflight", "deadline", "interrupt"])
def test_case_cannot_keep_pass_after_admission_or_final_identity_failure(
    failure, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    model.plan(tmp_path)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    if failure == "preflight":
        model.test_code = 1
    elif failure == "deadline":
        deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    else:

        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt("fixture final identity interrupted")

        monkeypatch.setattr(runner, "verify_runtime_identity_allowing", interrupted)
    if failure in {"preflight", "interrupt"}:
        with pytest.raises(
            (runner.RegionalFixtureError, KeyboardInterrupt),
            match="preflight|identity interrupted",
        ):
            runner.execute_case(model.settings, tmp_path, 1, deadline)
    else:
        assert runner.execute_case(model.settings, tmp_path, 1, deadline) == 1
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert result["verdict"] == "FAIL", (
        "admission failure and interruption must persist FAIL"
    )
    if failure != "interrupt":
        assert model.actions == [], "failed admission must not execute a release"


def test_runtime_deployment_inventory_falls_back_only_when_not_supplied() -> None:
    assert (
        runner.cpu_runtime_deployments(
            {
                "regional_deployment_inventory": SimpleNamespace(
                    CPU_RUNTIME_DEPLOYMENTS=()
                )
            }
        )
        == runner.CPU_RUNTIME_DEPLOYMENTS
    ), "retain the known role inventory for older modules"
    assert runner.cpu_runtime_deployments(
        {
            "regional_deployment_inventory": SimpleNamespace(
                CPU_RUNTIME_DEPLOYMENTS=["custom"]
            )
        }
    ) == ("custom",), "prefer the release's published runtime role inventory"


@pytest.mark.parametrize("reference", [None, "absolute"])
def test_manifest_binding_requires_a_reference_and_supports_absolute_paths(
    reference, tmp_path, monkeypatch
) -> None:
    model = HistoryModel(tmp_path, monkeypatch)
    path = model.settings.noop_config
    path.write_text(
        json.dumps(
            {} if reference is None else {"manifest": str(tmp_path / "manifest.json")}
        )
    )
    if reference is None:
        with pytest.raises(runner.RegionalFixtureError, match="names no manifest"):
            runner.manifest_with_rollback_flag(path)
    else:
        manifest = runner.manifest_with_rollback_flag(path)
        assert manifest["database"]["rollback_compatible"] is True, (
            "the refusal probe must set the claim"
        )
        assert "database" not in json.loads((tmp_path / "manifest.json").read_text()), (
            "refusal testing must not rewrite the candidate's manifest"
        )
