from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "deploy/control-plane/tools/apply_control_plane_deployment.py"
)
MODULE = lazy_script_module(SCRIPT)


def deployment(*, old: bool, role: str = "gpu-fault-api-ha") -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": role,
            "namespace": "gpu-fault-system",
            **({"uid": "same-deployment", "resourceVersion": "17"} if old else {}),
        },
        "spec": {
            "template": {
                "metadata": {"annotations": {"pin": "old" if old else "candidate"}},
                "spec": {
                    "containers": [
                        {
                            "name": "api",
                            "image": "old" if old else "candidate",
                            "env": (
                                [
                                    {"name": "GPU_FAULT_ALLOW_EMAIL", "value": "true"},
                                    {
                                        "name": "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL",
                                        "value": "false",
                                    },
                                    {
                                        "name": "GPU_FAULT_PROCESSOR_WORKERS",
                                        "value": "8",
                                    },
                                    {"name": "UNRELATED", "value": "keep"},
                                ]
                                if old
                                else []
                            ),
                        }
                    ]
                },
            }
        },
    }


class Api:
    def __init__(self, *, conflicts: int = 0, wrong_uid: bool = False) -> None:
        self.calls: list[str] = []
        self.writes: list[dict] = []
        self.conflicts = conflicts
        self.wrong_uid = wrong_uid

    def __call__(self, args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        value = json.loads(kwargs["input"])
        if "--dry-run=server" in args:
            self.calls.append("preview")
            merged = copy.deepcopy(value)
            merged["metadata"].update(uid="same-deployment", resourceVersion="17")
            if self.wrong_uid:
                merged["metadata"]["uid"] = "replacement"
            merged["spec"]["template"]["spec"]["containers"][0]["env"] = deployment(
                old=True
            )["spec"]["template"]["spec"]["containers"][0]["env"]
            merged["spec"]["template"]["spec"]["nodeSelector"] = {
                "user-managed": "keep"
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(merged), "")
        verb = "replace" if "replace" in args else "apply"
        self.calls.append(verb)
        if verb == "replace" and self.conflicts:
            self.conflicts -= 1
            return subprocess.CompletedProcess(
                args, 1, "", "Error from server (Conflict)"
            )
        self.writes.append(value)
        return subprocess.CompletedProcess(args, 0, "configured\n", "")


@pytest.mark.parametrize(
    "role",
    [
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ],
)
def test_legacy_cleanup_is_part_of_the_only_final_template_write(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    api = Api()
    monkeypatch.setattr(MODULE.subprocess, "run", api)
    MODULE.apply_deployment(
        deployment(old=False, role=role),
        {"items": [deployment(old=True, role=role)]},
        ["fake-api"],
    )
    assert api.calls == ["preview", "replace"]
    assert len(api.writes) == 1
    written = api.writes[0]
    assert written["metadata"]["resourceVersion"] == "17"
    template = written["spec"]["template"]
    assert template["metadata"]["annotations"]["pin"] == "candidate"
    assert template["spec"]["containers"][0]["image"] == "candidate"
    expected_env = [{"name": "UNRELATED", "value": "keep"}]
    if role != "gpu-fault-api-ha":
        expected_env.insert(0, {"name": "GPU_FAULT_PROCESSOR_WORKERS", "value": "8"})
    assert template["spec"]["containers"][0]["env"] == expected_env
    assert template["spec"]["nodeSelector"] == {"user-managed": "keep"}


def test_conflict_retries_the_preview_and_never_overwrites_a_new_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = Api(conflicts=1)
    monkeypatch.setattr(MODULE.subprocess, "run", api)
    MODULE.apply_deployment(
        deployment(old=False), {"items": [deployment(old=True)]}, ["fake-api"]
    )
    assert api.calls == ["preview", "replace", "preview", "replace"]
    assert len(api.writes) == 1
    api = Api(wrong_uid=True)
    monkeypatch.setattr(MODULE.subprocess, "run", api)
    with pytest.raises(ValueError, match="identity changed"):
        MODULE.apply_deployment(
            deployment(old=False), {"items": [deployment(old=True)]}, ["fake-api"]
        )
    assert api.calls == ["preview"]
    assert api.writes == []


def test_repeated_conflicts_stop_without_a_partial_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = Api(conflicts=3)
    monkeypatch.setattr(MODULE.subprocess, "run", api)
    with pytest.raises(ValueError, match="changed repeatedly"):
        MODULE.apply_deployment(
            deployment(old=False), {"items": [deployment(old=True)]}, ["fake-api"]
        )
    assert api.calls == ["preview", "replace"] * 3
    assert api.writes == []


def test_verbatim_rollback_keeps_the_supplied_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = Api()
    monkeypatch.setattr(MODULE.subprocess, "run", api)
    MODULE.apply_deployment(
        deployment(old=True), {"items": [deployment(old=True)]}, ["fake-api"]
    )
    assert api.calls == ["apply"]
    assert api.writes == [deployment(old=True)]
