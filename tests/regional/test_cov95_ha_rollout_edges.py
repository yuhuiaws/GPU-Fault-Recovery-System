from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from tests.regional._cov95_ha001_harness import Clock
from tests.regional.test_ha_capacity_callers import CallerEnvironment, run_entry


class Environment(CallerEnvironment):
    def __init__(self, mode: str) -> None:
        super().__init__(ha005, "")
        self.mode = mode
        self.clock = Clock()
        self.reads: dict[str, int] = {}
        self.post_phase = False
        self.post_reads = 0
        self.stopped = False

    def deployment(self, name: str = ha005.INGRESS_DEPLOYMENT) -> dict:
        self.reads[name] = self.reads.get(name, 0) + 1
        result = super().deployment(name)
        rolled = ("rollout", name) in self.events
        if rolled and self.mode in {"delayed", "timeout"}:
            if self.mode == "timeout" or self.reads[name] == 2:
                result["updated"] = 0
        if rolled and name == ha005.ALL_DEPLOYMENTS[-1] and self.reads[name] >= 3:
            self.post_phase = True
        if self.stopped and self.mode == "ready":
            result["ready"] = 0
        if self.stopped and self.mode == "restarts":
            result["pods"][0][1]["restarts"] = 1
        return result

    def dataplane(self, *args: str, **kwargs: Any) -> str:
        if args[0] == "exec" and args[-1] == "/state/stop":
            self.stopped = True
        return super().dataplane(*args, **kwargs)

    def probe(self, **kwargs: Any) -> dict:
        result = super().probe(**kwargs)
        if self.mode == "buffered":
            result["counters"].update(
                event_accepted=self.samples - 1, event_failures=1, event_buffered=1
            )
            result["admissions"][0]["replay"] = True
            response = result["wire_responses"][0]
            result["wire_responses"] = [
                {**response, "status": 503, "retry_after": "2"},
                *(dict(response) for _ in result["admissions"]),
            ]
            result["error_types"] = {"http-503": 1}
        if self.mode == "forbidden" and any(
            event[0] == "rollout" for event in self.events
        ):
            result["error_types"] = {"http-500": 1}
        if self.mode == "timeout" and any(
            event[0] == "rollout" for event in self.events
        ):
            self.clock.now += 601
        if self.post_phase:
            self.post_reads += 1
            if (self.mode == "outbox-delay" and self.post_reads == 2) or (
                self.mode == "outbox-timeout" and self.post_reads >= 2
            ):
                result["outbox"] = {"records": 1, "replayable": 1}
                if self.mode == "outbox-timeout":
                    self.clock.now += 301
        return result


@pytest.mark.parametrize(
    "mode",
    [
        "delayed",
        "timeout",
        "forbidden",
        "outbox-delay",
        "outbox-timeout",
        "restarts",
        "ready",
        "buffered",
        "registry-residual",
        "kubernetes-residual",
    ],
)
def test_all_role_rollout_keeps_deadlines_evidence_and_cleanup_failures_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    environment = Environment(mode)
    environment.install(monkeypatch)
    monkeypatch.setattr(ha005, "time", environment.clock)
    if mode in {"registry-residual", "kubernetes-residual"}:
        probe = (
            "registry_residuals"
            if mode == "registry-residual"
            else "kubernetes_residuals"
        )
        monkeypatch.setattr(
            ha005,
            probe,
            lambda: {
                "count": int(
                    any(event[0] == "deregister" for event in environment.events)
                )
            },
        )
    status, report = run_entry(ha005, tmp_path)
    successful = mode in {"delayed", "outbox-delay", "buffered"}
    assert status == (0 if successful else 1), report.get("error", report.get("errors"))
    assert environment.items == {}
    assert environment.rows == 0
    if successful:
        assert [
            event[1] for event in environment.events if event[0] == "rollout"
        ] == list(ha005.ALL_DEPLOYMENTS)
        assert report["coverage_scope"] == "all-enabled-roles"
        if mode == "buffered":
            assert report["outbox_exercised"] is True, (
                "the failed foreground send must exercise outbox replay"
            )
            replay = report["probe_final"]["admissions"][0]
            assert replay["replay"] is True, (
                "the buffered batch must have an observed replay admission"
            )
            assert replay["request_id"] in {
                item["request_id"] for item in report["processor_receipts"]["requests"]
            }, (
                "replay queue admissions require the same completion proof as direct sends"
            )
    elif mode in {"registry-residual", "kubernetes-residual"}:
        assert "residuals" in report["postflight_error"]
    elif mode in {"ready", "restarts"}:
        assert report["errors"], (
            "replacement availability/restart errors must fail acceptance"
        )
    else:
        assert report["error"], (
            "deadline and probe errors must stop the rollout sequence"
        )


@pytest.mark.parametrize("probe", ["registry_residuals", "kubernetes_residuals"])
def test_dirty_rollout_preflight_never_registers_or_creates_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probe: str
) -> None:
    environment = Environment("")
    environment.install(monkeypatch)
    monkeypatch.setattr(ha005, probe, lambda: {"count": 1})
    status, report = run_entry(ha005, tmp_path)
    assert status == 1
    assert "preflight residuals" in report["error"]
    assert environment.events == []
