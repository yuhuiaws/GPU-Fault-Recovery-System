from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import synthetic_replacement_route as helper


class _Regional:
    """A control plane that answers `kubectl` from a scripted env state."""

    def __init__(
        self,
        *,
        present: bool = False,
        value: str | None = None,
        value_from: bool = False,
        container: str = helper.CONTAINER,
        pods: tuple[str, ...] = ("api-a", "api-b"),
        gate_lags: bool = False,
    ) -> None:
        self.present = present
        self.value = value
        self.value_from = value_from
        self.container = container
        self.pods = pods
        # A rollout that has not reached every replica: the Deployment says one
        # thing and a still-running old Pod reports another.
        self.gate_lags = gate_lags
        self.lagging_value: str | None = None
        self.commands: list[tuple[str, ...]] = []
        self.settings = type("_S", (), {"cluster_id": "cluster-a"})()

    def _env(self) -> list[dict[str, Any]]:
        if not self.present:
            return [{"name": "OTHER", "value": "x"}]
        entry: dict[str, Any] = {"name": helper.ROUTE_ENV}
        if self.value_from:
            entry["valueFrom"] = {"configMapKeyRef": {"name": "c", "key": "k"}}
        else:
            entry["value"] = self.value
        return [{"name": "OTHER", "value": "x"}, entry]

    def kubectl(self, plane: str, *arguments: str, **kwargs: Any) -> str:
        assert plane == "cpu", plane
        self.commands.append(arguments)
        if arguments[0] == "get":
            return json.dumps(
                {
                    "metadata": {"generation": 7, "resourceVersion": "1"},
                    "spec": {
                        "replicas": len(self.pods),
                        "template": {
                            "spec": {
                                "containers": [
                                    {"name": self.container, "env": self._env()}
                                ]
                            }
                        },
                    },
                }
            )
        if arguments[0] == "set":
            assignment = arguments[-1]
            if assignment.endswith("-"):
                self.present, self.value = False, None
            else:
                self.present = True
                self.value = assignment.split("=", 1)[1]
            return ""
        if arguments[0] == "rollout":
            return "deployment rolled out\n"
        if arguments[0] == "exec":
            if self.gate_lags:
                return json.dumps({"enabled": self.lagging_value})
            return json.dumps({"enabled": self.value if self.present else None})
        raise AssertionError(arguments)

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        assert (plane, app) == ("cpu", helper.DEPLOYMENT), (plane, app)
        return [{"name": name} for name in self.pods]


def _no_sleep(seconds: float) -> None:
    """Settle immediately: these tests exercise the loop, not the clock."""


def _settings(tmp_path: Path) -> helper.Settings:
    # Zero, so a fleet that never settles fails on the first poll instead of
    # holding the suite for a real wall-clock timeout.
    return helper.Settings(
        baseline=tmp_path / "synthetic-route.json", rollout_timeout_seconds=0
    )


def test_a_window_closes_back_to_absent_not_to_false(tmp_path: Path) -> None:
    # The shipped manifest carries no such variable at all. Restoring it as
    # `false` leaves config the release does not know it has, and the next
    # reviewer cannot tell a deliberate `false` from a forgotten cleanup.
    regional = _Regional(present=False)
    settings = _settings(tmp_path)

    opened = helper.open_window(settings, regional, helper.survey(regional))
    assert opened["baseline"]["route_env_present"] is False
    assert all(item["enabled"] == "true" for item in opened["pod_gates"]), opened

    closed = helper.close_window(settings, regional, helper.survey(regional))

    assert closed["restored_state"]["route_env_present"] is False
    assert regional.present is False
    assignments = [item[-1] for item in regional.commands if item[0] == "set"]
    assert assignments == [f"{helper.ROUTE_ENV}=true", f"{helper.ROUTE_ENV}-"]


def test_a_window_closes_back_to_the_recorded_literal(tmp_path: Path) -> None:
    # A site that deliberately shipped `false` must get `false` back, not have
    # the variable deleted out from under its own configuration.
    regional = _Regional(present=True, value="false")
    settings = _settings(tmp_path)

    helper.open_window(settings, regional, helper.survey(regional))
    closed = helper.close_window(settings, regional, helper.survey(regional))

    assert closed["restored_state"]["route_env_value"] == "false"
    assert regional.value == "false"


def test_opening_refuses_a_route_someone_else_already_enabled(tmp_path: Path) -> None:
    # Two owners of one switch means the first one to finish closes it under the
    # second, whose case is still driving replacements through it.
    regional = _Regional(present=True, value="true")

    with pytest.raises(helper.RegionalFixtureError, match="already true"):
        helper.open_window(_settings(tmp_path), regional, helper.survey(regional))


def test_opening_refuses_a_record_whose_window_is_not_actually_open(
    tmp_path: Path,
) -> None:
    # A record saying "open" over a Deployment that is not is a record about some
    # other cluster or some other release. Opening on top of it would overwrite
    # the only description of the state to restore.
    settings = _settings(tmp_path)
    settings.baseline.write_text(
        json.dumps({"opened_at": "2026-09-05T00:00:00Z"}), encoding="utf-8"
    )
    regional = _Regional(present=False)

    with pytest.raises(helper.RegionalFixtureError, match="records an open window"):
        helper.open_window(settings, regional, helper.survey(regional))


def test_an_interrupted_open_is_resumed_rather_than_refused(tmp_path: Path) -> None:
    # An open that set the variable and then died waiting for the rollout leaves
    # the operator between an open that refuses (a record exists) and a close
    # that has not verified anything. The record already owns this window, so the
    # second --open finishes it instead of starting a second one.
    regional = _Regional(present=False, gate_lags=True)
    settings = _settings(tmp_path)
    with pytest.raises(helper.RegionalFixtureError):
        helper.open_window(settings, regional, helper.survey(regional), sleep=_no_sleep)
    first = json.loads(settings.baseline.read_text(encoding="utf-8"))
    regional.gate_lags = False

    resumed = helper.open_window(
        settings, regional, helper.survey(regional), sleep=_no_sleep
    )

    assert resumed["opened_at"] == first["opened_at"]
    assert resumed["baseline"]["route_env_present"] is False
    assert resumed["resumed_at"] > ""
    assignments = [item[-1] for item in regional.commands if item[0] == "set"]
    assert assignments == [f"{helper.ROUTE_ENV}=true"], assignments


def test_closing_refuses_without_a_record(tmp_path: Path) -> None:
    regional = _Regional(present=True, value="true")

    with pytest.raises(helper.RegionalFixtureError, match="no synthetic-route base"):
        helper.close_window(_settings(tmp_path), regional, helper.survey(regional))


def test_closing_twice_is_refused(tmp_path: Path) -> None:
    regional = _Regional(present=False)
    settings = _settings(tmp_path)
    helper.open_window(settings, regional, helper.survey(regional))
    helper.close_window(settings, regional, helper.survey(regional))

    with pytest.raises(helper.RegionalFixtureError, match="already closed"):
        helper.close_window(settings, regional, helper.survey(regional))


def test_a_replica_that_missed_the_rollout_fails_the_open(tmp_path: Path) -> None:
    # `preflight_errors` grades every ready replica, so an open that reports
    # success while one Pod still answers 404 turns into a mid-case failure with
    # a workload already submitted to a real node. `kubectl rollout status`
    # returns before the old Pods are gone, and they are still Ready, so the
    # rollout returning is not the condition to check.
    regional = _Regional(present=False, gate_lags=True)

    with pytest.raises(helper.RegionalFixtureError, match="report the route enabled"):
        helper.open_window(
            _settings(tmp_path), regional, helper.survey(regional), sleep=_no_sleep
        )


def test_a_replica_still_serving_the_route_fails_the_close(tmp_path: Path) -> None:
    # The mirror image, and the one that matters for residual: the Deployment is
    # restored but an old replica is still Ready and still serving the route.
    # Reporting that as closed is reporting a live route as retired.
    regional = _Regional(present=False)
    settings = _settings(tmp_path)
    helper.open_window(settings, regional, helper.survey(regional), sleep=_no_sleep)
    survey = helper.survey(regional)
    regional.gate_lags = True
    regional.lagging_value = "true"

    with pytest.raises(helper.RegionalFixtureError, match="report the route disabled"):
        helper.close_window(settings, regional, survey, sleep=_no_sleep)

    assert json.loads(settings.baseline.read_text(encoding="utf-8"))["closed_at"] > ""


def test_a_referenced_value_is_refused(tmp_path: Path) -> None:
    # A restore writes a literal, so a `valueFrom` entry would be silently
    # converted into a different kind of variable.
    regional = _Regional(present=True, value_from=True)

    with pytest.raises(helper.RegionalFixtureError, match="not a literal"):
        helper.survey(regional)


def test_a_missing_api_container_is_named(tmp_path: Path) -> None:
    regional = _Regional(container="sidecar")

    with pytest.raises(helper.RegionalFixtureError, match="no container named"):
        helper.survey(regional)


def test_read_only_by_default() -> None:
    arguments = helper.parser().parse_args(["--baseline", "/tmp/b.json"])

    assert arguments.open is False
    assert arguments.close is False
    assert arguments.confirm == ""
    assert helper.OPEN_CONFIRMATION == "OPEN_SYNTHETIC_REPLACEMENT_WINDOW"
    assert helper.CLOSE_CONFIRMATION == "CLOSE_SYNTHETIC_REPLACEMENT_WINDOW"


def test_the_managed_variable_cannot_be_chosen_on_the_command_line() -> None:
    # This helper edits a live control-plane Deployment. If the variable name
    # were an argument it would be a general-purpose "set any env on the API
    # tier" tool, which is precisely what the acceptance standard forbids.
    assert "--env" not in helper.parser().format_help()
    assert helper.ROUTE_ENV == "GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS"


def test_helper_carries_no_site_topology() -> None:
    source = Path(helper.__file__).read_text(encoding="utf-8")

    assert "/secure/gpu-fault-bootstrap" not in source
    assert "514385905925" not in source
    assert "hyperpod-i-" not in source
