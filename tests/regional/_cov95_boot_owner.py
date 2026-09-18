from __future__ import annotations

import subprocess
from typing import Any

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from tests.regional._cov95_cap_cases import Clock
from tests.regional.test_boot_acceptance_behavior import RuntimeFixture


class OwnerFixture(RuntimeFixture):
    def __init__(self, clock: Clock, *, failure: str = "") -> None:
        super().__init__()
        self.clock = clock
        self.failure = failure
        self.config["health"] = {"amp_workspace_id": "unit-workspace"}
        self.command_id: str | None = None
        self.rows = {"existing-command"}
        self.probes: list[str] = []
        self.metric_targets: list[tuple[str, str, str]] = []

    def pod_json(
        self, _plane: str, pod: str, _script: str, *_args: str
    ) -> dict[str, Any]:
        return {
            "owners": ["different-owner"]
            if self.failure == "owners" and pod == "replica-b"
            else ["gpu-fault-kubernetes-adapter"]
        }

    def cpu_python(self, script: str, *args: str, **_kwargs: Any) -> dict[str, Any]:
        if script == runtime.PROFILE_OWNER_PROBE:
            self.probes.append("profile")
            return {
                "profiles": [
                    {
                        "profile_version": "profile-a",
                        "orphan_owners": ["missing-owner"]
                        if self.failure == "profile"
                        else [],
                    }
                ]
            }
        if script == runtime.REMOTE_BASELINE_PROBE:
            self.probes.append("baseline")
            if self.failure == "baseline" and self.command_id is None:
                return {"ids": [], "count": None}
            return {"ids": sorted(self.rows), "count": len(self.rows)}
        if script == runtime.REMOTE_INJECT_PROBE:
            self.probes.append("insert")
            self.command_id = args[1]
            self.rows.add(self.command_id)
            return {"command_id": self.command_id}
        if script == runtime.REMOTE_STATE_PROBE:
            self.probes.append("state")
            return {
                "exists": True,
                "status": "LEASED" if self.failure == "leased" else "PENDING",
                "lease_owner": "other" if self.failure == "leased" else None,
            }
        if script == runtime.REMOTE_DELETE_PROBE:
            self.probes.append("delete")
            if self.failure == "delete":
                raise RuntimeError("synthetic delete failure")
            self.rows.discard(args[0])
            return {"remaining": len(self.rows)}
        raise AssertionError("unexpected owner probe")

    def pod_python(
        self, plane: str, app: str, script: str, *args: str
    ) -> dict[str, Any]:
        assert (plane, app, args) == (
            "cpu",
            "gpu-fault-control-worker",
            (self.cluster_id,),
        ), "remote-command metrics must be sampled on the target CPU control-worker"
        assert script == runtime.METRIC_PROBE, (
            "the worker boundary must run the remote-command metric probe"
        )
        self.metric_targets.append((plane, app, args[0]))
        self.probes.append("metrics")
        return {
            "duration_seconds": 6 if self.failure == "metrics" else 1,
            "pending": 1,
            "oldest_unclaimed_seconds": 700 + self.clock.elapsed,
        }

    def exec(
        self, _plane: str, pod: str, *args: str, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((pod, *args))
        blocked = self.command_id in self.rows and self.failure != "readiness"
        return subprocess.CompletedProcess(
            args,
            int(blocked),
            "",
            "open backlog needs execution owners" if blocked else "",
        )

    def amp_request(self, **kwargs: Any) -> dict[str, Any]:
        pending = self.command_id in self.rows
        if kwargs["path"].endswith("query"):
            return {
                "data": {
                    "result": [{"metric": {"cluster_id": self.cluster_id}}]
                    if pending
                    else []
                }
            }
        return {
            "data": {
                "alerts": [
                    {
                        "labels": {
                            "alertname": "GpuFaultRemoteCommandUnclaimed",
                            "cluster_id": self.cluster_id,
                        },
                        "state": "firing",
                    }
                ]
                if pending and self.failure != "alert"
                else []
            }
        }
