"""The cluster-executor env window survives a replica rolling away mid-survey."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import deployment_window_guard as guard
from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional.acceptance_runner_common import replica_vanished
from tests.regional.test_deployment_window_safety import WindowAPI

ATTEMPTS = "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS"
assert ATTEMPTS in env_window.ALLOWED_VARIABLES


class _RollingRegional:
    def __init__(self, vanished: str, values: dict[str, str], *, stderr: str) -> None:
        self._vanished = vanished
        self._values = values
        self._stderr = stderr
        self.exec_targets: list[str] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": self._vanished}, {"name": "survivor"}]

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        assert arguments[0] == "exec", arguments
        target = arguments[1]
        self.exec_targets.append(target)
        if target == self._vanished:
            raise env_window.RegionalFixtureError(
                f"command failed (1): kubectl ... exec {target} ...; stderr={self._stderr}"
            )
        return json.dumps(self._values)


@pytest.mark.parametrize(
    "stderr",
    [
        'Error from server (NotFound): pods "gone" not found',
        "error: cannot exec into a container in a completed pod; current phase is Succeeded",
    ],
)
def test_replica_values_skip_a_replica_that_rolled_away(stderr: str) -> None:
    """DESTR-014 attempt 3 (2026-09-08) died inside the open window's converge
    on exactly this NotFound and left the executor running the compressed
    timing set until the window was closed by hand."""
    values = {ATTEMPTS: "6"}
    regional = _RollingRegional("gone-68rp8", values, stderr=stderr)

    replicas = env_window.replica_values(regional)

    assert regional.exec_targets == ["gone-68rp8", "survivor"], regional.exec_targets
    assert replicas == [{"pod": "survivor", "values": values}], replicas


def test_a_real_exec_failure_still_raises() -> None:
    regional = _RollingRegional("a", {ATTEMPTS: "6"}, stderr="OCI runtime exec failed")
    with pytest.raises(env_window.RegionalFixtureError, match="OCI runtime"):
        env_window.replica_values(regional)


def test_the_shared_marker_list_names_every_kubelet_wording() -> None:
    for text in (
        'pods "x" not found',
        "cannot exec into a container in a completed pod; current phase is Succeeded",
        'unable to upgrade connection: container not found ("executor")',
        "container is not running",
    ):
        assert replica_vanished(RuntimeError(text)), text
    assert not replica_vanished(RuntimeError("permission denied")), "a real error"


def test_lease_and_poll_variables_are_allowed_and_validated() -> None:
    assignments = env_window.parse_assignments(
        [
            "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=10",
            "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS=2",
        ]
    )
    assert assignments == {
        "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS": "10",
        "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS": "2",
    }
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(["GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=0"])
    with pytest.raises(env_window.RegionalFixtureError):
        env_window.parse_assignments(["GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=ten"])


def test_allow_list_still_refuses_anything_else() -> None:
    with pytest.raises(env_window.RegionalFixtureError, match="allow-list"):
        env_window.parse_assignments(["GPU_FAULT_ALLOW_HYPERPOD_REBOOT=1"])


def test_restore_arguments_cover_the_lease_and_poll_baseline() -> None:
    baseline = {
        "variables": {
            "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS": {
                "present": True,
                "value": "120",
            },
            "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS": {
                "present": False,
                "value": None,
            },
        }
    }
    assert env_window.restore_arguments(baseline) == [
        "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS=120",
        "GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS-",
    ]


def _closed_record(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": scope,
        "state": "CLOSED",
        "closed_at": "2026-09-18T08:19:45+00:00",
        "assignments": {"GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS": "10"},
        "baseline": {"uid": "u-1", "variables": {}},
    }


def test_a_closed_record_from_another_release_is_archived_not_refused(tmp_path) -> None:
    """HA-004's re-run after a deploy met attempt 1's CLOSED window record under
    the previous release identity and was refused ("baseline is unbound or
    invalid"); a closed record owns nothing, so it is moved aside."""
    from scripts.e2e.regional import deployment_window_guard as guard

    path = tmp_path / "executor-env-window.json"
    old_scope = {"plane": "gpu", "identity": {"release_id": "e22b92bb2ec4"}}
    new_scope = {"plane": "gpu", "identity": {"release_id": "ce392e83fc98"}}
    record = _closed_record(old_scope)
    path.write_text(json.dumps(record))

    assert guard.retire_foreign_closed_record(path, record, new_scope) is None
    assert not path.exists(), "the foreign closed record must be moved aside"
    archived = list(tmp_path.glob("executor-env-window.closed-*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text()) == record


def test_only_closed_foreign_records_are_retired(tmp_path) -> None:
    from scripts.e2e.regional import deployment_window_guard as guard

    path = tmp_path / "executor-env-window.json"
    scope = {"plane": "gpu", "identity": {"release_id": "ce392e83fc98"}}
    same = _closed_record(scope)
    path.write_text(json.dumps(same))
    assert guard.retire_foreign_closed_record(path, same, scope) is same
    assert path.exists(), "a closed record of this very identity stays in place"
    open_foreign = {
        **_closed_record({"plane": "gpu", "identity": {"release_id": "old"}}),
        "state": "OPEN",
    }
    open_foreign.pop("closed_at")
    path.write_text(json.dumps(open_foreign))
    assert guard.retire_foreign_closed_record(path, open_foreign, scope) is open_foreign
    assert path.exists(), "an open window from another identity is never moved aside"
    with pytest.raises(env_window.RegionalFixtureError, match="unbound or invalid"):
        guard.require_window_record(
            open_foreign,
            scope,
            {"uid": "u-1"},
            ["GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS"],
        )
    assert guard.retire_foreign_closed_record(path, None, scope) is None


LEASE = "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS"


def _window_scope(release_id: str, **overrides: Any) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "plane": "gpu",
        "deployment": env_window.DEPLOYMENT,
        "container": env_window.CONTAINER,
        "environment": {"namespace": "gpu-fault-system", "context": "gpu"},
        "identity": {"release_id": release_id, "cluster_id": "cluster-a"},
    }
    scope.update(overrides)
    return scope


def _open_record(
    scope: dict[str, Any],
    *,
    state: str = "OPEN",
    uid: str = "u-1",
    generation: int = 13,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": scope,
        "state": state,
        "opened_at": "t0",
        "assignments": {LEASE: "10"},
        "baseline": {
            "uid": uid,
            "generation": generation,
            "variables": {LEASE: {"present": True, "value": "120"}},
        },
    }


def _live(
    *, uid: str = "u-1", generation: int = 13, lease: str | None = "120"
) -> dict[str, Any]:
    return {
        "uid": uid,
        "generation": generation,
        "variables": {LEASE: {"present": lease is not None, "value": lease}},
    }


@pytest.mark.parametrize("state", ["OPEN", "OPENING"])
@pytest.mark.parametrize(
    "live",
    [_live(uid="u-2", generation=1), _live(generation=15)],
    ids=["new-uid", "new-generation"],
)
def test_an_open_record_a_release_change_closed_is_retired(
    tmp_path: Path, state: str, live: dict[str, Any]
) -> None:
    """DESTR-014 attempt 6 met an earlier attempt's OPEN records under the
    release that was live then; two deploys had re-rendered both Deployments,
    so the window was physically gone, yet no later attempt could ever match
    the old release id and the case stayed wedged until the files were deleted
    by hand."""
    path = tmp_path / "executor-env-window.json"
    record = _open_record(_window_scope("old"), state=state)
    path.write_text(json.dumps(record))

    retired = guard.retire_open_record_closed_by_release(
        path, record, _window_scope("new"), live
    )

    assert retired is None, "a window the release change closed owns nothing"
    assert not path.exists(), "the retired record must be moved aside"
    archived = list(tmp_path.glob("executor-env-window.retired-*.json"))
    assert len(archived) == 1, archived
    saved = json.loads(archived[0].read_text())
    assert saved["retired_by"] == "release change"
    assert saved["retired_at"], "the archive must say when it was retired"
    assert saved["retired_live_uid"] == live["uid"]
    assert saved["retired_live_release_id"] == "new"
    assert saved["retired_archive"] == str(archived[0])
    assert (saved["state"], saved["scope"], saved["assignments"]) == (
        state,
        _window_scope("old"),
        {LEASE: "10"},
    ), "the record itself is archived verbatim"


def test_a_window_still_in_effect_under_another_release_is_not_retired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "executor-env-window.json"
    record = _open_record(_window_scope("old"))
    path.write_text(json.dumps(record))
    still_open = _live(uid="u-2", generation=1, lease="10")

    kept = guard.retire_open_record_closed_by_release(
        path, record, _window_scope("new"), still_open
    )

    assert kept is record, "a window another identity still holds is not ours"
    assert path.exists(), "an open window from another identity is never moved aside"
    assert not list(tmp_path.glob("*.retired-*")), "nothing may be archived"
    assert "retired_by" not in record, "the record must not be annotated"
    with pytest.raises(env_window.RegionalFixtureError, match="unbound or invalid"):
        guard.require_window_record(record, _window_scope("new"), still_open, [LEASE])


@pytest.mark.parametrize(
    ("record", "live"),
    [
        pytest.param(
            _open_record(_window_scope("new")), _live(uid="u-2"), id="same-release"
        ),
        pytest.param(
            _open_record(_window_scope("old", plane="cpu")),
            _live(uid="u-2"),
            id="other-plane",
        ),
        pytest.param(
            _open_record(_window_scope("old", deployment="gpu-fault-control-worker")),
            _live(uid="u-2"),
            id="other-deployment",
        ),
        pytest.param(
            _open_record(
                _window_scope(
                    "old", identity={"release_id": "old", "cluster_id": "cluster-b"}
                )
            ),
            _live(uid="u-2"),
            id="other-cluster",
        ),
        pytest.param(_open_record(_window_scope("old")), _live(), id="not-re-rendered"),
        pytest.param(
            _open_record(_window_scope("old"), state="CLOSING"),
            _live(uid="u-2"),
            id="closing",
        ),
        pytest.param(
            {**_open_record(_window_scope("old"), state="CLOSED"), "closed_at": "t1"},
            _live(uid="u-2"),
            id="closed-belongs-to-the-closed-retirement",
        ),
        pytest.param(
            _open_record(_window_scope("old")),
            {"uid": "u-2", "generation": 1, "variables": {}},
            id="assignment-not-observed-live",
        ),
    ],
)
def test_records_the_release_change_retirement_does_not_own_stay_in_place(
    tmp_path: Path, record: dict[str, Any], live: dict[str, Any]
) -> None:
    path = tmp_path / "executor-env-window.json"
    path.write_text(json.dumps(record))
    before = copy.deepcopy(record)

    kept = guard.retire_open_record_closed_by_release(
        path, record, _window_scope("new"), live
    )

    assert kept is record, "only a foreign OPEN record the deploy closed is retired"
    assert record == before, "an untouched record must not be annotated"
    assert json.loads(path.read_text()) == before, "the record on disk is untouched"
    assert not list(tmp_path.glob("*.retired-*")), "nothing may be archived"


class _RecordingAPI(WindowAPI):
    """The safety-test fake, also listing every kubectl verb it is asked for."""

    def __init__(self, module: Any) -> None:
        super().__init__(module)
        self.verbs: list[str] = []

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        self.verbs.append(arguments[0])
        return super().kubectl(plane, *arguments, **kwargs)


def _stale_window_after_a_deploy(tmp_path: Path) -> tuple[_RecordingAPI, Any]:
    """An OPEN record under release-one, then what a deploy does to the fake:
    the rendered env is back, the Deployment is re-rendered (new generation)
    and the release identity moves on."""
    api = _RecordingAPI(env_window)
    settings = env_window.Settings(
        baseline=tmp_path / "executor-env-window.json", rollout_timeout_seconds=1
    )
    container = api.deployment["spec"]["template"]["spec"]["containers"][0]
    rendered = copy.deepcopy(container["env"])
    opened = env_window.open_window(
        settings, api, env_window.survey(api), {LEASE: "10"}
    )
    assert opened["scope"]["identity"]["release_id"] == "release-one"
    container["env"] = rendered
    api.deployment["metadata"]["generation"] += 1
    api.deployment["metadata"]["resourceVersion"] = str(
        int(api.deployment["metadata"]["resourceVersion"]) + 1
    )
    api.deployment["status"]["observedGeneration"] = api.deployment["metadata"][
        "generation"
    ]
    api.release_id = "release-two"
    return api, settings


def test_close_window_retires_a_record_a_release_change_closed_without_a_mutation(
    tmp_path: Path,
) -> None:
    api, settings = _stale_window_after_a_deploy(tmp_path)
    survey = env_window.survey(api)
    patches_before = len(api.patches)
    api.verbs.clear()

    report = env_window.close_window(settings, api, survey)

    assert report["state"] == "RETIRED"
    assert report["retired_by"] == "release change"
    assert report["retired_live_release_id"] == "release-two"
    assert report["live_state"]["uid"] == "deployment-one"
    assert report["assignments"] == {LEASE: "10"}
    assert api.verbs == ["get"], "one Deployment read; no patch, rollout or exec"
    assert len(api.patches) == patches_before, "a retired record restores nothing"
    assert not settings.baseline.exists(), "the retired record is moved aside"
    archive = Path(report["archive"])
    assert archive.parent == tmp_path and archive.is_file(), archive
    saved = json.loads(archive.read_text())
    assert saved["retired_live_release_id"] == "release-two"
    assert saved["scope"]["identity"]["release_id"] == "release-one"


def test_open_window_retires_the_stale_record_and_opens_under_the_new_release(
    tmp_path: Path,
) -> None:
    """The live failure: attempt 6's open met attempt 4's OPEN record and was
    refused; it must retire that record and open its own window."""
    api, settings = _stale_window_after_a_deploy(tmp_path)

    record = env_window.open_window(
        settings, api, env_window.survey(api), {LEASE: "10"}
    )

    assert record["state"] == "OPEN"
    assert record["scope"]["identity"]["release_id"] == "release-two"
    assert "retired_by" not in record, "the new record is not the retired one"
    assert len(api.patches) == 2, (
        "the first open and this one; retirement patches nothing"
    )
    archived = list(tmp_path.glob("executor-env-window.retired-*.json"))
    assert len(archived) == 1, archived
    saved = json.loads(archived[0].read_text())
    assert saved["retired_by"] == "release change"
    assert saved["scope"]["identity"]["release_id"] == "release-one"
    assert json.loads(settings.baseline.read_text())["state"] == "OPEN"


def _window_still_in_effect_under_a_new_release(
    tmp_path: Path,
) -> tuple[_RecordingAPI, Any]:
    """An OPEN record under release-one whose env assignment survived the next
    deploy: ``kubectl apply`` keeps env entries the manifest never listed, so
    only the release identity moved on (live 2026-09-19, the control-worker two
    releases later)."""
    api = _RecordingAPI(env_window)
    settings = env_window.Settings(
        baseline=tmp_path / "executor-env-window.json", rollout_timeout_seconds=1
    )
    opened = env_window.open_window(
        settings, api, env_window.survey(api), {LEASE: "10"}
    )
    assert opened["scope"]["identity"]["release_id"] == "release-one"
    api.release_id = "release-two"
    return api, settings


def test_close_window_closes_a_foreign_window_that_is_still_in_effect(
    tmp_path: Path,
) -> None:
    api, settings = _window_still_in_effect_under_a_new_release(tmp_path)
    survey = env_window.survey(api)
    patches_before = len(api.patches)
    assert survey["deployment"]["variables"][LEASE] == {
        "present": True,
        "value": "10",
    }, "the fixture must present the window as still in effect"

    with pytest.raises(env_window.RegionalFixtureError, match="unbound or invalid"):
        env_window.close_window(settings, api, survey)
    assert len(api.patches) == patches_before, (
        "a case's own cleanup never closes another release's window"
    )

    record = env_window.close_window(settings, api, survey, foreign_release_ok=True)

    assert record["state"] == "CLOSED", (
        "a window still in effect is closed, not retired"
    )
    assert record["closed_under_release"] == "release-two", (
        "the record must say which release closed another release's window"
    )
    assert record["scope"]["identity"]["release_id"] == "release-one", (
        "the record keeps the scope of the release that opened it"
    )
    assert len(api.patches) == patches_before + 1, "closing restores the baseline once"
    restored = env_window.deployment_env(api)
    assert restored["variables"][LEASE] == {"present": False, "value": None}, (
        "the assignment the window set must be removed, as the baseline recorded"
    )
    assert not list(tmp_path.glob("*.retired-*")), (
        "an in-effect window is never retired"
    )
    saved = json.loads(settings.baseline.read_text())
    assert saved["state"] == "CLOSED" and saved["closed_under_release"] == "release-two"


def test_open_window_refuses_a_foreign_window_still_in_effect_and_names_the_close(
    tmp_path: Path,
) -> None:
    api, settings = _window_still_in_effect_under_a_new_release(tmp_path)
    patches_before = len(api.patches)

    with pytest.raises(
        env_window.RegionalFixtureError,
        match=r"opened under release release-one is still in effect on .*--close-foreign-release --close --baseline",
    ):
        env_window.open_window(settings, api, env_window.survey(api), {LEASE: "20"})

    assert len(api.patches) == patches_before, "a refused open patches nothing"
    assert settings.baseline.exists(), "the foreign record stays until it is closed"
    assert not list(tmp_path.glob("*.retired-*")), (
        "an in-effect window is never retired"
    )


@pytest.mark.parametrize(
    ("record", "live", "expected"),
    [
        pytest.param(
            _open_record(_window_scope("old")), _live(lease="10"), True, id="in-effect"
        ),
        pytest.param(
            _open_record(_window_scope("old")),
            _live(uid="u-2", lease="10"),
            False,
            id="other-uid",
        ),
        pytest.param(
            _open_record(_window_scope("old")), _live(lease="120"), False, id="restored"
        ),
        pytest.param(
            _open_record(_window_scope("old")), _live(lease=None), False, id="absent"
        ),
        pytest.param(
            _open_record(_window_scope("new")),
            _live(lease="10"),
            False,
            id="same-release",
        ),
        pytest.param(
            _open_record(_window_scope("old", plane="cpu")),
            _live(lease="10"),
            False,
            id="other-plane",
        ),
        pytest.param(
            {**_open_record(_window_scope("old")), "state": "CLOSED"},
            _live(lease="10"),
            False,
            id="closed",
        ),
    ],
)
def test_foreign_open_window_in_effect_answers_only_for_a_live_window(
    record: dict[str, Any], live: dict[str, Any], expected: bool
) -> None:
    assert guard.foreign_open_window_in_effect(record, _window_scope("new"), live) is (
        expected
    ), "the predicate must name exactly a foreign window the Deployment still carries"
