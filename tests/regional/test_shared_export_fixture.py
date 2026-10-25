"""The GF-GENERAL-LIVE-005 test export server fixture over a recording kubectl.

No cluster: a fake regional answers ``kubectl`` from a script and records every
call, so the tests pin the manifests (a bounded, unprivileged, tmpfs-backed
NFSv4 server Pod scheduled away from the client nodes), the create/wait/pause/
resume/delete protocol and its refusals (foreign occupant, replaced object,
never-stopped server).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import shared_export_fixture as export
from scripts.e2e.regional.regional_commands import RegionalFixtureError

IMAGE = "public.ecr.aws/gfrs/nfs-ganesha@sha256:" + "b" * 64
CLIENTS = (
    "ip-10-0-43-5.us-west-2.compute.internal",
    "ip-10-0-32-162.us-west-2.compute.internal",
)


class FakeRegional:
    """``kubectl(plane, *args, input_text=..., ...)`` from a per-verb script."""

    def __init__(self) -> None:
        self.settings = type("Settings", (), {"namespace": "gpu-fault-system"})()
        self.calls: list[tuple[str, tuple[str, ...], str | None]] = []
        self.objects: dict[str, dict[str, Any] | None] = {
            "configmap": None,
            "pod": None,
        }
        self.phases: list[dict[str, Any]] = []
        self.exec_output = "State:\tT (stopped)"
        self.uid = 100

    def kubectl(
        self, plane: str, *arguments: str, input_text: str | None = None, **_: Any
    ) -> str:
        self.calls.append((plane, arguments, input_text))
        verb = arguments[0]
        if verb == "get":
            kind = arguments[1]
            current = self.objects.get(kind)
            if kind == "pod" and current is not None and self.phases:
                current = {**current, "status": self.phases.pop(0)}
                self.objects["pod"] = current
            return json.dumps(current) if current is not None else ""
        if verb == "create":
            manifest = json.loads(input_text or "{}")
            kind = manifest["kind"].lower()
            self.uid += 1
            created = {
                **manifest,
                "metadata": {**manifest["metadata"], "uid": f"uid-{self.uid}"},
            }
            self.objects[kind] = created
            return json.dumps(created)
        if verb == "exec":
            return self.exec_output
        if verb == "logs":
            return "ganesha started\n"
        if verb == "delete":
            plural = arguments[2].rsplit("/", 2)[-2]
            kind = {"pods": "pod", "configmaps": "configmap"}[plural]
            options = json.loads(input_text or "{}")
            current = self.objects[kind]
            assert current is not None, "delete of an absent object"
            assert options["preconditions"]["uid"] == current["metadata"]["uid"], (
                options
            )
            self.objects[kind] = None
            return ""
        raise AssertionError(f"unexpected kubectl {arguments}")


def settings(**overrides: Any) -> export.ExportServerSettings:
    base: dict[str, Any] = {
        "image": IMAGE,
        "case_id": "GF-GENERAL-LIVE-005",
        "run_id": "gl005-stat-timeout-a1-deadbeef",
        "avoid_nodes": CLIENTS,
    }
    base.update(overrides)
    return export.ExportServerSettings(**base)


def server(
    regional: FakeRegional | None = None, **overrides: Any
) -> export.TestExportServer:
    clock = {"now": 0.0}

    def tick() -> float:
        clock["now"] += 1.0
        return clock["now"]

    return export.TestExportServer(
        regional or FakeRegional(),
        settings(**overrides),
        clock=tick,
        sleep=lambda _s: None,
    )


def running(ip: str = "10.0.9.7", node: str = "ip-10-0-99-1") -> dict[str, Any]:
    return {
        "phase": "Running",
        "podIP": ip,
        "startTime": "2026-10-25T00:00:00Z",
        "conditions": [{"type": "Ready", "status": "True"}],
    }


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def test_manifests_are_a_bounded_unprivileged_tmpfs_backed_nfs_server() -> None:
    configmap, pod = server().manifests()
    assert configmap["metadata"]["name"] == pod["metadata"]["name"]
    assert pod["metadata"]["name"].startswith("gfrs-export-") and pod["metadata"][
        "name"
    ].endswith("-a")
    labels = pod["metadata"]["labels"]
    assert (
        labels["app"] == export.APP_LABEL
        and labels[export.CASE_LABEL] == "GF-GENERAL-LIVE-005"
    )
    config = configmap["data"]["ganesha.conf"]
    for expected in (
        "Protocols = 4;",
        "Squash = Root_Squash;",
        f"Path = {export.EXPORT_PATH};",
        f"Pseudo = {export.PSEUDO_PATH};",
        "Transports = TCP;",
        "Graceless = true;",
        "Attr_Expiration_Time = 1;",
        "Name = VFS;",
    ):
        assert expected in config, expected
    spec = pod["spec"]
    assert spec["restartPolicy"] == "Never" and spec["activeDeadlineSeconds"] == 3600
    assert spec["tolerations"] == [{"operator": "Exists"}]
    assert "nodeName" not in spec
    terms = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    expression = terms["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert expression == {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": list(CLIENTS),
    }
    container = spec["containers"][0]
    assert (
        container["image"] == IMAGE
        and container["securityContext"]["privileged"] is False
    )
    assert container["securityContext"]["capabilities"] == {
        "add": ["DAC_READ_SEARCH", "SYS_RESOURCE"]
    }
    assert container["readinessProbe"]["tcpSocket"]["port"] == export.NFS_PORT
    assert (
        container["command"][:2] == ["/bin/sh", "-c"]
        and container["command"][2] == export.ENTRY_SCRIPT
    )
    volumes = {item["name"]: item for item in spec["volumes"]}
    assert volumes["export"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "256Mi"}
    assert (
        volumes["ganesha-config"]["configMap"]["name"] == configmap["metadata"]["name"]
    )
    assert container["resources"]["limits"]["memory"] == "768Mi", (
        "tmpfs counts against the container"
    )


def test_entry_script_runs_ganesha_as_a_pausable_child() -> None:
    script = export.ENTRY_SCRIPT
    assert "ganesha.nfsd" in script and script.rstrip().endswith("wait")
    assert f"echo $! > {export.PID_FILE}" in script, (
        "the child pid is what pause/resume signal"
    )
    assert f"chmod 1777 {export.EXPORT_PATH}" in script, (
        "root-squashed clients must write"
    )
    assert "-STOP" in export.PAUSE_SCRIPT and "-CONT" in export.RESUME_SCRIPT


def test_a_pinned_node_replaces_the_anti_affinity() -> None:
    _configmap, pod = server(node="ip-10-0-99-1", avoid_nodes=CLIENTS).manifests()
    assert pod["spec"]["nodeName"] == "ip-10-0-99-1" and "affinity" not in pod["spec"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"image": "repo/nfs:latest"}, "immutable digest"),
        ({"size_mib": 16}, "MiB"),
        ({"node": CLIENTS[0]}, "client node"),
        ({"suffix": "A1"}, "lower-case"),
        ({"plane": "edge"}, "gpu or cpu"),
        ({"active_deadline_seconds": 60}, "deadline"),
    ],
)
def test_settings_refuse_unsafe_values(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(RegionalFixtureError, match=match):
        settings(**overrides)


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
def test_create_wait_pause_resume_delete(tmp_path: Path) -> None:
    regional = FakeRegional()
    fixture = export.TestExportServer(
        regional,
        settings(),
        journal=tmp_path / "export-a.json",
        clock=lambda: 0.0,
        sleep=lambda _s: None,
    )
    created = fixture.create()
    kinds = [
        args[1].lower() if args[0] == "get" else json.loads(text)["kind"].lower()
        for _plane, args, text in regional.calls
        if args[0] in {"create"}
    ]
    assert kinds == ["configmap", "pod"], (
        "the ConfigMap exists before the Pod references it"
    )
    assert created["configmap"] == "uid-101" and created["pod"] == "uid-102"
    assert all(plane == "gpu" for plane, _args, _text in regional.calls), (
        "every call went to the GPU plane"
    )
    journal = json.loads((tmp_path / "export-a.json").read_text())
    assert journal["created"] == {"configmap": "uid-101", "pod": "uid-102"}
    regional.phases = [{"phase": "Pending"}, running()]
    ready = fixture.wait_ready(timeout_seconds=600, poll_seconds=1)
    assert (
        ready["ip"] == "10.0.9.7"
        and ready["export"] == "/a"
        and ready["uid"] == "uid-102"
    )
    paused = fixture.pause()
    assert "T" in paused["state"]
    _plane, args, _text = regional.calls[-1]
    assert args[:2] == ("exec", fixture.name) and args[-1] == export.PAUSE_SCRIPT
    regional.exec_output = "State:\tS (sleeping)"
    fixture.resume()
    assert regional.calls[-1][1][-1] == export.RESUME_SCRIPT
    regional.exec_output = "0"
    assert fixture.set_export_mode("000")["mode"] == "000"
    with pytest.raises(RegionalFixtureError, match="000 or 1777"):
        fixture.set_export_mode("755")
    deleted = fixture.delete()
    assert deleted == {"pod": "deleted", "configmap": "deleted"}
    deletes = [args for _plane, args, _text in regional.calls if args[0] == "delete"]
    assert [args[2].rsplit("/", 2)[-2] for args in deletes] == ["pods", "configmaps"]
    assert regional.objects == {"configmap": None, "pod": None}
    assert fixture.describe()["created"] == {}


def test_pause_and_resume_verify_the_process_state() -> None:
    regional = FakeRegional()
    fixture = server(regional)
    regional.exec_output = "State:\tS (sleeping)"
    with pytest.raises(RegionalFixtureError, match="did not stop"):
        fixture.pause()
    regional.exec_output = "State:\tT (stopped)"
    with pytest.raises(RegionalFixtureError, match="still stopped"):
        fixture.resume()


def test_wait_ready_fails_on_a_dead_pod_or_a_deadline() -> None:
    regional = FakeRegional()
    fixture = server(regional)
    fixture.create()
    regional.phases = [{"phase": "Failed"}]
    with pytest.raises(RegionalFixtureError, match="ended Failed"):
        fixture.wait_ready(timeout_seconds=600)
    regional.phases = [{"phase": "Pending"}] * 50
    with pytest.raises(RegionalFixtureError, match="not Ready after"):
        fixture.wait_ready(timeout_seconds=5, poll_seconds=1)


def test_a_foreign_occupant_is_never_adopted_or_deleted() -> None:
    regional = FakeRegional()
    fixture = server(regional)
    regional.objects["configmap"] = {
        "kind": "ConfigMap",
        "metadata": {
            "name": fixture.name,
            "namespace": "gpu-fault-system",
            "uid": "x",
            "labels": {},
        },
    }
    with pytest.raises(RegionalFixtureError, match="foreign"):
        fixture.create()
    assert regional.objects["pod"] is None, "nothing else was created"
    with pytest.raises(RegionalFixtureError, match="foreign"):
        fixture.delete()
    assert regional.objects["configmap"] is not None


def test_delete_refuses_a_replaced_object_and_tolerates_absence() -> None:
    regional = FakeRegional()
    fixture = server(regional)
    fixture.create()
    pod = regional.objects["pod"]
    assert pod is not None
    regional.objects["pod"] = {**pod, "metadata": {**pod["metadata"], "uid": "uid-999"}}
    with pytest.raises(RegionalFixtureError, match="replaced"):
        fixture.delete()
    regional.objects["pod"] = None
    regional.objects["configmap"] = None
    assert fixture.delete() == {"pod": "absent", "configmap": "absent"}


def test_the_cpu_plane_is_selectable() -> None:
    regional = FakeRegional()
    server(regional, plane="cpu").create()
    assert {plane for plane, _args, _text in regional.calls} == {"cpu"}
