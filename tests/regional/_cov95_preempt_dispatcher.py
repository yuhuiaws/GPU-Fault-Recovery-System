from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from scripts.e2e.regional import run_preempt037_dispatcher_liveness as runner

V = runner.verdicts


class DispatcherModel:
    def __init__(self, tmp_path, monkeypatch) -> None:
        self.now = 10_000.0
        self.failure = ""
        self.events = []
        self.patches = []
        self.census_reads = 0
        self.recovery_reads = 0
        self.settings = SimpleNamespace(
            namespace="fixture-cpu", environment=lambda: {"namespace": "fixture-cpu"}
        )
        self.container = {"name": V.CONTAINER, "image": "fixture@sha256:" + "a" * 64}
        self.document = {
            "metadata": {
                "uid": "owned-deployment",
                "resourceVersion": "1",
                "generation": 1,
            },
            "spec": {"template": {"spec": {"containers": [self.container]}}},
        }
        rules = tmp_path / "rules.yaml"
        rules.write_text(
            json.dumps(
                {
                    "groups": [
                        {
                            "rules": [
                                {
                                    "alert": V.STALLED_ALERT,
                                    "expr": f"time() - max by (control_plane_cluster, region) ({V.DISPATCH_METRIC}) > 300",
                                    "for": "5m",
                                    "labels": {"severity": "critical"},
                                    "annotations": {"runbook_url": "#dispatcher-alert"},
                                }
                            ]
                        }
                    ]
                }
            )
        )
        runbook = tmp_path / "runbook.md"
        runbook.write_text("# dispatcher-alert\n")
        monkeypatch.setattr(runner, "RULES", rules)
        monkeypatch.setattr(runner, "RUNBOOK", runbook)
        monkeypatch.setattr(runner, "DispatcherWatchdog", self.watchdog)
        monkeypatch.setattr(
            runner,
            "time",
            SimpleNamespace(
                time=lambda: self.now, monotonic=lambda: self.now, sleep=self.sleep
            ),
        )
        monkeypatch.setattr(runner, "utc_now", self.utc_now)

    def utc_now(self):
        return datetime.fromtimestamp(self.now, timezone.utc).isoformat()

    def sleep(self, seconds):
        self.now += seconds

    def value(self):
        for item in self.container.get("env", []):
            if item["name"] == V.VARIABLE:
                return item["value"]
        return "true"

    def ready_pods(self, *_args):
        return [{"name": "worker-one"}, {"name": "worker-two"}]

    def evidence_identity(self):
        return {"release_id": "fixture-release", "cluster_id": "fixture-cluster"}

    def cpu_python(self, _source):
        self.census_reads += 1
        failing_read = {"busy-before": 1, "busy-after": 2, "busy-during": 3}.get(
            self.failure
        )
        return {
            "status_counts": {"RUNNING": 1}
            if self.census_reads == failing_read
            else {"SUCCEEDED": 2}
        }

    def metrics(self):
        stamp = self.now
        if self.failure == "already-stalled":
            stamp = 0
        elif self.value() == "false" and self.failure != "never-stall":
            stamp = 10_000.0
        elif len(self.patches) >= 2:
            self.recovery_reads += 1
            if self.failure == "no-recovery" or (
                self.failure == "slow-recovery" and self.recovery_reads <= 2
            ):
                stamp = 0
        return f"{V.DISPATCH_METRIC} {stamp}\n{V.PERIODIC_METRIC} {self.now}\n"

    def apply(self, patch):
        assert patch[:2] == [
            {
                "op": "test",
                "path": "/metadata/uid",
                "value": self.document["metadata"]["uid"],
            },
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": self.document["metadata"]["resourceVersion"],
            },
        ], "every dispatcher patch must bind UID and the last read resource version"
        for item in patch[2:]:
            path = item["path"]
            assert path.startswith("/spec/template/spec/containers/0/env"), (
                "the window patch must stay inside the chosen container environment"
            )
            if item["op"] == "remove":
                self.container["env"].pop(int(path.rsplit("/", 1)[1]))
            elif item["op"] == "replace":
                self.container["env"][int(path.split("/")[-2])]["value"] = item["value"]
            elif item["op"] == "add" and path.endswith("/-"):
                self.container["env"].append(item["value"])
            elif item["op"] == "add" and path.endswith("/env"):
                self.container["env"] = item["value"]
            else:
                raise AssertionError(
                    f"unexpected dispatcher patch operation: {item['op']}"
                )
        self.patches.append(patch)
        self.document["metadata"]["resourceVersion"] = str(len(self.patches) + 1)
        if len(self.patches) == 2 and self.failure == "restore-drift":
            self.container["image"] = "replacement-image"
        if len(self.patches) == 1 and self.failure == "open-ack-loss":
            raise RuntimeError("window patch acknowledgement lost")

    def kubectl(self, plane, *args, **kwargs):
        assert plane == "cpu", "the dispatcher window must stay on the CPU plane"
        if args[:2] == ("get", "deployment"):
            return json.dumps(self.document)
        if args[:2] == ("patch", "deployment"):
            assert "--patch-file=/dev/stdin" in args, (
                "environment data belongs in stdin"
            )
            self.apply(json.loads(kwargs["input_text"]))
            return "patched"
        if args[:2] == ("rollout", "status"):
            return "ready"
        if args[0] == "exec":
            if kwargs.get("input_text"):
                return json.dumps({"metrics": self.metrics()})
            value = "false" if self.failure == "effective-disabled" else self.value()
            return json.dumps({V.VARIABLE: value})
        raise AssertionError(f"unexpected dispatcher transport operation: {args[:2]}")

    def watchdog(self, _regional, **kwargs):
        def arm():
            self.events.append("arm")
            if self.failure == "arm":
                raise RuntimeError("watchdog admission failed")

        return SimpleNamespace(
            name="fixture-watchdog",
            run_id="fixture-run",
            restore_at=kwargs["restore_at"],
            arm=arm,
            cleanup=lambda: self.events.append("disarm"),
        )

    def execute(self, case_dir):
        return runner.execute(
            self, case_dir, datetime.fromtimestamp(self.now + 5_000, timezone.utc)
        )
