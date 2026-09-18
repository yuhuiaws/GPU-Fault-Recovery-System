from __future__ import annotations

import base64
import copy
import json
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import identity_acceptance_auth as auth
from scripts.e2e.regional.auth015_context import Auth015Context
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from tests.regional._cov95_auth015_live import LiveSite
from tests.regional._cov95_auth015_support import KEY_A, KEY_B
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def case(tmp_path, monkeypatch):
    site = LiveSite(tmp_path, monkeypatch)
    events = []
    current = {
        "cpu": site.current_key_document,
        "gpu": copy.deepcopy(site.key_document),
    }
    current["gpu"]["metadata"]["uid"] = "gpu-key-uid"
    original = copy.deepcopy(current)
    stamp = datetime.now(timezone.utc)

    def agents(after=False):
        return {
            "agents": {
                node: {
                    "generation": 7,
                    "lifecycle_state": "ACTIVE",
                    "last_heartbeat_at": (
                        stamp + timedelta(seconds=int(after))
                    ).isoformat(),
                    "node_action_key_version": 2,
                }
                for node in ("node-a", "node-b")
            }
        }

    class Probe:
        def __init__(self, node):
            self.settings = SimpleNamespace(node=node)

        def create(self):
            events.append("create:" + self.settings.node)

        def execute(self, *args):
            assert args == ("--master-sha256", "7" * 64)
            return {
                "master_matches": [],
                "values_scanned": 5,
                "host_tmp_exists": True,
                "systemd_environment_exists": True,
            }

        def cleanup(self):
            events.append("cleanup:" + self.settings.node)
            return {"pod": False, "configmap": False}

    def kubectl(plane, *arguments, **kwargs):
        assert arguments[:2] == ("patch", "secret")
        patch = json.loads(kwargs["input_text"])
        document = current[plane]
        assert patch[0]["value"] == document["metadata"]["uid"]
        assert patch[1]["value"] == document["metadata"]["resourceVersion"]
        assert patch[2]["value"] == document["data"]
        events.append("restore:" + plane)
        document["data"] = patch[3]["value"]
        document["metadata"]["resourceVersion"] = "3"
        return ""

    regional = SimpleNamespace(cpu_python=lambda *args: agents(True), kubectl=kubectl)
    baseline = Auth015Context(
        "7" * 64,
        regional,
        original["gpu"],
        original["cpu"],
        auth.node_key_digests(original["gpu"]),
        auth.node_key_digests(original["cpu"]),
        ["node-a", "node-b"],
        agents(),
        [Probe("node-a"), Probe("node-b")],
    )
    site.cpu_kubeconfig = tmp_path / "cpu"
    site.gpu_kubeconfig = tmp_path / "gpu"
    site.regional = lambda _: regional
    monkeypatch.setattr(auth, "prepare_auth015", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(
        auth, "secret_document", lambda _, plane, *args: copy.deepcopy(current[plane])
    )
    monkeypatch.setattr(
        auth,
        "gpu_master_reference_scan",
        lambda *args: {"installer_resources": ["Job/owned-installer"], "hits": []},
    )
    monkeypatch.setattr(auth, "time", SimpleNamespace(sleep=lambda _: None))

    def rotate(command, **kwargs):
        assert command == ["bash", "deploy/node/provision-node-action-keys.sh"]
        assert kwargs["env"]["GPU_FAULT_ROTATE_NODE_ACTION_KEY"] == "node-a"
        assert not any(value in repr(command) for value in (KEY_A, KEY_B)), (
            "node keys must never enter a provisioning command's argv"
        )
        events.append("rotate")
        for document in current.values():
            document["data"]["node-a"] = base64.b64encode(
                b"synthetic-rotated-" + b"r" * 48
            ).decode()
            document["metadata"]["resourceVersion"] = "2"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(auth, "run", rotate)
    options = {
        "nodes": ("node-a", "node-b"),
        "fleet_master_file": tmp_path / "synthetic-master",
        "host_probe_image": "test@sha256:" + "a" * 64,
        "case_dir": tmp_path,
        "focused_tests": {"passed": True},
        "release_inputs": site.files.inputs(),
    }
    try:
        yield SimpleNamespace(
            site=site,
            options=options,
            events=events,
            original=original,
            current=current,
        )
    finally:
        site.close()


def test_protocol_inputs_alone_cannot_authorize_custody_or_rotated_activation(case):
    result = auth.run_auth015(case.site, case.site.target, **case.options)
    assert result["verdict"] == "FAIL", "the full AUTH015 claim remains unproved"
    assert set(result["not_evaluated"]) == {
        "installation_time_master_custody",
        "deployed_node_a_key_activation",
        "deployed_cross_node_command_and_result_signatures",
    }
    assert result["requires_new_authorized_evidence"] is True
    assert result["checks"]["no_node_or_secret_mutation"] is True
    assert case.events == [] and case.site.pair.requests == []
    assert case.site.pair.executors["node-a"].secret == KEY_A
    assert case.site.pair.executors["node-b"].secret == KEY_B
    assert all(
        case.current[plane]["data"] == case.original[plane]["data"]
        for plane in case.current
    ), "both owned Secret snapshots must be restored after the partial proof"
    assert KEY_A not in repr(result) and KEY_B not in repr(result)


@pytest.mark.parametrize("failure", ["refused", "invalid-return", "changed-node"])
def test_missing_custody_stops_before_even_attempting_a_partial_protocol_proof(
    case, monkeypatch, failure
):
    if failure == "refused":

        def refuse(*args, **kwargs):
            raise Auth015ProofError("synthetic proof refusal")

        monkeypatch.setattr(auth, "prove_deployed_protocol", refuse)
    elif failure == "invalid-return":
        monkeypatch.setattr(
            auth, "prove_deployed_protocol", lambda *a, **k: {"verdict": "FAIL"}
        )
    else:
        case.site.after_capture = lambda: case.site.raw["nodes"]["items"][1][
            "metadata"
        ].update(uid="changed")
    result = auth.run_auth015(case.site, case.site.target, **case.options)
    assert "rotate" not in case.events and not any(
        item.startswith("create:") for item in case.events
    )
    assert set(result["not_evaluated"]) == {
        "installation_time_master_custody",
        "deployed_node_a_key_activation",
        "deployed_cross_node_command_and_result_signatures",
    }
    assert case.events == [] and case.site.pair.requests == []
    assert result["verdict"] == "FAIL"


def test_omitting_signed_release_inputs_preserves_all_three_unproved_gaps(case):
    case.options["release_inputs"] = None
    result = auth.run_auth015(case.site, case.site.target, **case.options)
    assert result["verdict"] == "FAIL"
    assert len(result["not_evaluated"]) == 3
    assert result.get("deployed_protocol") is None
    assert case.site.pair.requests == [], (
        "absence of a release proof must not probe a node"
    )
