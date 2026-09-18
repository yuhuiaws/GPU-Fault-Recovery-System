from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.admin.execution import run_command
from gpu_fault_release.regional_release_config import ReleaseError
from tests._script_loader import lazy_script_module
from tests.deploy._node_action_key_api import (
    HYPERPOD,
    NAMESPACE,
    Api,
    encoded,
    kubectl_node_api,
    node_list,
    secret,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/node/provision-node-action-keys.sh"
MODULE = lazy_script_module(SCRIPT.with_name("provision_node_action_keys.py"))


def run_script(
    tmp_path: Path,
    state: dict[str, Any],
    *,
    extra_environment: dict[str, str] | None = None,
    script: Path = SCRIPT,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    kubectl = binary / "kubectl"
    shutil.copyfile(Path(__file__).with_name("_node_action_key_api.py"), kubectl)
    kubectl.chmod(0o700)
    api_file = tmp_path / "api.json"
    api_file.write_text(json.dumps(state))
    api_file.chmod(0o600)
    master = tmp_path / "master"
    master.write_text("fixture-only-master-" * 4)
    master.chmod(0o600)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={
            "HOME": str(tmp_path),
            # The deploy host runs this script with the admin venv's bin first on
            # PATH (site.effective_environment), so its plain ``python3`` sees the
            # installed environment; the fixture mirrors that instead of relying
            # on an activated venv in the pytest process.
            "PATH": os.pathsep.join(
                (str(binary), str(Path(sys.executable).parent), os.environ["PATH"])
            ),
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": "/dev/null",
            "TEST_NODE_KEY_API": str(api_file),
            "GPU_FAULT_KUBECTL_CONTEXT": "fixture-gpu",
            "GPU_FAULT_CLUSTER_ID": "fixture-cluster",
            "GPU_FAULT_HYPERPOD_CLUSTER": HYPERPOD,
            "GPU_FAULT_FLEET_MASTER_FILE": str(master),
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": str(tmp_path / "cpu.kubeconfig"),
            **(extra_environment or {}),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, json.loads(api_file.read_text())


def test_script_preserves_other_cluster_cpu_keys(tmp_path: Path) -> None:
    result, state = run_script(
        tmp_path,
        {
            "nodes": node_list("node-current"),
            "secrets": {
                "gpu": secret("gpu", {"node-current": encoded("rotated")}),
                "cpu": secret(
                    "cpu", {"node-other": encoded("other")}, last_applied=True
                ),
            },
        },
    )
    assert result.returncode == 0, "provisioning did not complete"
    assert set(state["secrets"]["cpu"]["data"]) == {"node-current", "node-other"}
    assert state["secrets"]["gpu"]["data"]["node-current"] == encoded("rotated"), (
        "provisioning replaced an existing randomly rotated GPU key"
    )


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
def test_script_read_failure_is_not_absence(tmp_path: Path, scope: str) -> None:
    result, state = run_script(
        tmp_path,
        {
            "nodes": node_list("node-current"),
            "secrets": {
                "gpu": secret("gpu", {"node-current": encoded("rotated")}),
                "cpu": secret("cpu", {"node-other": encoded("other")}),
            },
            "events": [
                {
                    "on": f"{scope}:get",
                    "occurrence": 2 if scope == "gpu" else 1,
                    "returncode": 1,
                    "stderr": "Error from server (Forbidden): fixture read refused",
                }
            ],
        },
    )
    assert result.returncode != 0, "a failed Secret read was accepted as absence"
    assert not state["writes"], "a failed Secret read allowed a mutation"
    assert state["secrets"]["gpu"]["data"]["node-current"] == encoded("rotated"), (
        "a failed read replaced the current random key"
    )


def test_script_refuses_a_same_node_name_cpu_key_conflict(tmp_path: Path) -> None:
    result, state = run_script(
        tmp_path,
        {
            "nodes": node_list("node-current"),
            "secrets": {
                "gpu": secret("gpu", {"node-current": encoded("current")}),
                "cpu": secret("cpu", {"node-current": encoded("other-owner")}),
            },
        },
    )
    assert result.returncode != 0, "a conflicting CPU NodeName was overwritten"
    assert not state["writes"], "a NodeName conflict was checked after mutation"


def fixture_api(
    *, cpu_data: dict[str, str] | None = None, gpu_data: dict[str, str] | None = None
) -> Api:
    return Api(
        {
            "nodes": node_list("node-current"),
            "secrets": {
                "gpu": secret(
                    "gpu",
                    {"node-current": encoded("rotated")}
                    if gpu_data is None
                    else gpu_data,
                ),
                "cpu": secret(
                    "cpu",
                    {"node-other": encoded("other")} if cpu_data is None else cpu_data,
                ),
            },
        }
    )


def provision(
    tmp_path: Path, api: Api, *, rotate_node: str = "", cpu: bool = True
) -> int:
    master = tmp_path / "master"
    master.write_text("fixture-only-master-" * 4)
    master.chmod(0o600)
    result = MODULE.provision(
        MODULE.Scope(("kubectl", "--context", "fixture-gpu"), NAMESPACE),
        MODULE.Scope(("kubectl", "--kubeconfig", "fixture-cpu"), NAMESPACE)
        if cpu
        else None,
        cluster_id="fixture-cluster",
        hyperpod_cluster=HYPERPOD,
        master_file=master,
        rotate_node=rotate_node,
        runner=api.run,
    )
    assert isinstance(result, int), "provisioning must return an integer node count"
    return result


def test_confirmed_absence_creates_each_secret_without_apply(tmp_path: Path) -> None:
    api = fixture_api()
    api.state["secrets"] = {"gpu": None, "cpu": None}
    assert provision(tmp_path, api) == 1
    assert [(item["scope"], item["verb"]) for item in api.state["writes"]] == [
        ("gpu", "create"),
        ("cpu", "create"),
    ]
    assert api.state["secrets"]["gpu"]["data"] == api.state["secrets"]["cpu"]["data"], (
        "new key maps differ"
    )
    assert all("apply" not in args for args in api.state["calls"]), (
        "new Secrets must be created without apply"
    )


def test_same_keys_are_a_read_only_noop(tmp_path: Path) -> None:
    api = fixture_api(
        cpu_data={"node-current": encoded("rotated"), "node-other": encoded("other")}
    )
    assert provision(tmp_path, api) == 1
    assert api.state["writes"] == []


def test_gpu_prunes_stale_keys_but_cpu_does_not_claim_their_ownership(
    tmp_path: Path,
) -> None:
    data = {"node-current": encoded("rotated"), "node-stale": encoded("stale")}
    api = fixture_api(gpu_data=data, cpu_data={**data, "node-other": encoded("other")})
    assert provision(tmp_path, api) == 1
    assert set(api.state["secrets"]["gpu"]["data"]) == {"node-current"}
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-current",
        "node-stale",
        "node-other",
    }


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize(
    "error",
    [
        "Error from server (Forbidden)",
        "Error from server (NotFound)",
        "authentication helper not found",
        "TLS trust rejected",
        "context deadline exceeded",
    ],
)
def test_nonzero_get_never_means_absence(
    tmp_path: Path, scope: str, error: str
) -> None:
    api = fixture_api(gpu_data={})
    api.state["events"] = [
        {"on": f"{scope}:get-secret", "returncode": 1, "stderr": error}
    ]
    with pytest.raises((MODULE.ProvisionError, ReleaseError)):
        provision(tmp_path, api)
    assert api.state["writes"] == []
    assert api.state["write_attempts"] == []


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize(
    "mutation",
    [
        {"apiVersion": "wrong/v1"},
        {"kind": "ConfigMap"},
        {"type": "kubernetes.io/tls"},
        {"metadata": {"name": "other"}},
        {"metadata": {"namespace": "other"}},
        {"metadata": {"uid": ""}},
        {"metadata": {"resourceVersion": ""}},
        {"metadata": {"resourceVersion": 1}},
        {"metadata": {"deletionTimestamp": "2026-01-01T00:00:00Z"}},
        {"metadata": {"annotations": []}},
        {"data": []},
        {"data": {"../bad": encoded("invalid")}},
        {"data": {"node-current": "not-base64"}},
        {"data": {"node-current": "c2hvcnQ="}},
        {"data": {"node-current": 17}},
        {"data": {"node-current": "/w==" * 32}},
        {"stringData": {}},
        {"immutable": "false"},
    ],
)
def test_secret_shape_or_identity_error_stops_before_writes(
    tmp_path: Path, scope: str, mutation: dict[str, Any]
) -> None:
    api = fixture_api(gpu_data={})
    document = api.state["secrets"][scope]
    for name, value in mutation.items():
        if name == "metadata":
            document[name].update(value)
        else:
            document[name] = value
    with pytest.raises((MODULE.ProvisionError, ReleaseError)):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


@pytest.mark.parametrize(
    "output",
    [
        "null",
        "[]",
        "not-json",
        '{"kind":"Secret","kind":"ConfigMap"}',
        '{"kind":"Secret","metadata":null}',
    ],
)
def test_invalid_secret_json_is_not_absence(tmp_path: Path, output: str) -> None:
    api = fixture_api()
    api.state["events"] = [{"on": "gpu:get-secret", "output": output}]
    with pytest.raises((MODULE.ProvisionError, ReleaseError)):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"apiVersion": "other"},
        {"kind": "PodList"},
        {"items": []},
        {"items": None},
        {"items": [None]},
        {"metadata": {"continue": "more"}},
        {"item": {"apiVersion": "apps/v1"}},
        {"item": {"kind": "Pod"}},
        {"node": {"name": "../bad"}},
        {"node": {"uid": ""}},
        {"node": {"labels": {}}},
        {"node": {"labels": {"sagemaker.amazonaws.com/cluster-name": "other"}}},
        {"node": {"deletionTimestamp": "2026-01-01T00:00:00Z"}},
        {"duplicate": True},
    ],
)
@pytest.mark.parametrize("kind", ["NodeList", "List"])
def test_invalid_or_ambiguous_node_inventory_stops_before_writes(
    tmp_path: Path, mutation: dict[str, Any], kind: str
) -> None:
    api = fixture_api()
    api.state["nodes"]["kind"] = kind
    if "node" in mutation:
        api.state["nodes"]["items"][0]["metadata"].update(mutation["node"])
    elif "item" in mutation:
        api.state["nodes"]["items"][0].update(mutation["item"])
    elif "duplicate" in mutation:
        api.state["nodes"]["items"] *= 2
    else:
        api.state["nodes"].update(mutation)
    with pytest.raises(MODULE.ProvisionError):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


@pytest.mark.parametrize("kind", ["NodeList", "List"])
def test_typed_and_generic_node_lists_accept_verified_members(
    tmp_path: Path, kind: str
) -> None:
    api = fixture_api()
    api.state["nodes"] = {**node_list("node-current", "node-new"), "kind": kind}
    assert provision(tmp_path, api) == 2
    assert set(api.state["secrets"]["gpu"]["data"]) == {"node-current", "node-new"}
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-current",
        "node-new",
        "node-other",
    }


@pytest.mark.allows_cluster_binaries("kubectl")
def test_actual_kubectl_node_list_output_is_accepted(tmp_path: Path) -> None:
    if shutil.which("kubectl") is None or shutil.which("openssl") is None:
        pytest.skip("isolated TLS client integration requires kubectl and openssl")
    nodes = node_list("node-current")
    # The API's typed list can omit member TypeMeta; kubectl supplies it.
    nodes["items"][0].pop("apiVersion")
    nodes["items"][0].pop("kind")
    response_kinds: list[str] = []

    def runner(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result: subprocess.CompletedProcess[str] = run_command(arguments, **kwargs)
        if result.returncode == 0:
            response_kinds.append(json.loads(result.stdout)["kind"])
        return result

    with kubectl_node_api(tmp_path, nodes) as (kubeconfig, requests):
        client = MODULE.SecretClient(
            MODULE.Scope(
                ("kubectl", "--kubeconfig", str(kubeconfig), "--context", "fixture"),
                NAMESPACE,
            ),
            runner,
        )
        try:
            observed = client.nodes(HYPERPOD)
        except MODULE.ProvisionError:
            pytest.fail(
                f"helper rejected actual kubectl node list kinds {response_kinds}",
                pytrace=False,
            )
    assert observed == {"node-current": "fixture-uid-node-current"}
    assert len(response_kinds) == 1 and response_kinds[0] in {"NodeList", "List"}
    node_requests = [
        urlsplit(request)
        for request in requests
        if urlsplit(request).path == "/api/v1/nodes"
    ]
    assert node_requests, "node discovery did not query the API"
    assert all(
        parse_qs(request.query).get("labelSelector")
        == [f"sagemaker.amazonaws.com/cluster-name={HYPERPOD}"]
        for request in node_requests
    ), "node discovery must stay scoped to the HyperPod cluster"


def test_node_uid_change_before_mutation_is_rejected(tmp_path: Path) -> None:
    api = fixture_api(gpu_data={})
    changed = node_list("node-current")
    changed["items"][0]["metadata"]["uid"] = "recreated-node"
    api.state["events"] = [{"on": "gpu:get-nodes", "occurrence": 2, "nodes": changed}]
    with pytest.raises(MODULE.ProvisionError, match="identities changed"):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


def test_cpu_cas_retry_merges_a_concurrent_cluster_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    api = fixture_api()
    api.state["secrets"]["cpu"]["metadata"]["labels"] = {"external-owner": "keep"}
    api.state["events"] = [
        {
            "on": "cpu:replace",
            "merge_data": {"node-concurrent": encoded("concurrent")},
            "returncode": 1,
            "stderr": "an unclassified failed write",
        }
    ]
    assert provision(tmp_path, api) == 1
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-current",
        "node-other",
        "node-concurrent",
    }
    assert api.state["secrets"]["cpu"]["metadata"]["labels"] == {
        "external-owner": "keep"
    }
    attempts = [item for item in api.state["write_attempts"] if item["scope"] == "cpu"]
    assert [item["version"] for item in attempts] == ["1", "2"]
    assert all(item["uid"] == "fixture-cpu-secret" for item in attempts), (
        "CPU retries must retain the original Secret UID"
    )


def test_repeated_cpu_conflicts_exhaust_the_bound_without_lost_keys(
    tmp_path: Path,
) -> None:
    api = fixture_api()
    api.state["events"] = [
        {
            "on": "cpu:replace",
            "occurrence": index,
            "merge_data": {f"node-concurrent-{index}": encoded(str(index))},
            "returncode": 1,
        }
        for index in range(1, 4)
    ]
    with pytest.raises(MODULE.ProvisionError, match="CPU.*changed repeatedly"):
        provision(tmp_path, api)
    assert len(api.state["write_attempts"]) == 3
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-other",
        "node-concurrent-1",
        "node-concurrent-2",
        "node-concurrent-3",
    }


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize("disappear", [True, False])
def test_secret_replacement_or_disappearance_is_not_retry_authority(
    tmp_path: Path, scope: str, disappear: bool
) -> None:
    api = fixture_api(gpu_data={})
    replacement = secret(scope, {})
    replacement["metadata"]["uid"] = "recreated-secret"
    api.state["events"] = [
        {
            "on": f"{scope}:replace",
            "document": None if disappear else replacement,
            "returncode": 1,
        }
    ]
    with pytest.raises(MODULE.ProvisionError, match="disappeared|UID changed"):
        provision(tmp_path, api)
    attempts = [item for item in api.state["write_attempts"] if item["scope"] == scope]
    assert len(attempts) == 1
    assert all(item["verb"] == "replace" for item in attempts), (
        "Secret identity drift must not trigger create or apply"
    )


def test_gpu_cas_retry_preserves_a_concurrently_rotated_key(tmp_path: Path) -> None:
    api = fixture_api()
    api.state["nodes"] = node_list("node-current", "node-new")
    api.state["events"] = [
        {
            "on": "gpu:replace",
            "merge_data": {"node-current": encoded("concurrent-rotation")},
            "returncode": 1,
        }
    ]
    assert provision(tmp_path, api) == 2
    assert api.state["secrets"]["gpu"]["data"]["node-current"] == encoded(
        "concurrent-rotation"
    ), "GPU CAS retry reverted a concurrent rotation"
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-current",
        "node-new",
        "node-other",
    }
    attempts = [item for item in api.state["write_attempts"] if item["scope"] == "gpu"]
    assert [item["version"] for item in attempts] == ["1", "2"]


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
def test_create_race_revalidates_and_preserves_current_data(
    tmp_path: Path, scope: str
) -> None:
    api = fixture_api()
    api.state["secrets"][scope] = None
    data = (
        {"node-current": encoded("concurrent-rotation")}
        if scope == "gpu"
        else {"node-concurrent": encoded("concurrent")}
    )
    api.state["events"] = [
        {"on": f"{scope}:create", "document": secret(scope, data), "returncode": 1}
    ]
    assert provision(tmp_path, api) == 1
    assert all(
        api.state["secrets"][scope]["data"].get(name) == value
        for name, value in data.items()
    ), "create race discarded an existing key"
    assert any(item["verb"] == "create" for item in api.state["write_attempts"]), (
        "confirmed Secret absence must trigger a create attempt"
    )


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize("create", [True, False])
def test_lost_write_acknowledgement_is_confirmed_by_readback(
    tmp_path: Path, scope: str, create: bool
) -> None:
    api = fixture_api(gpu_data={})
    if create:
        api.state["secrets"][scope] = None
    verb = "create" if create else "replace"
    api.state["events"] = [{"on": f"{scope}:{verb}", "returncode": 1, "lost_ack": True}]
    assert provision(tmp_path, api) == 1
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == scope])
        == 1
    )


def test_failed_cpu_write_can_resume_from_current_gpu_values(tmp_path: Path) -> None:
    api = fixture_api()
    api.state["nodes"] = node_list("node-current", "node-new")
    api.state["events"] = [{"on": "cpu:replace", "returncode": 1}]
    with pytest.raises(MODULE.ProvisionError, match="without confirmed progress"):
        provision(tmp_path, api)
    saved = dict(api.state["secrets"]["gpu"]["data"])
    api.state["events"] = []
    assert provision(tmp_path, api) == 2
    assert api.state["secrets"]["gpu"]["data"] == saved, "retry changed the GPU key map"
    assert all(
        api.state["secrets"]["cpu"]["data"].get(name) == value
        for name, value in saved.items()
    ), "retry did not synchronize the current GPU keys"


@pytest.mark.parametrize("repeat_rotation_flag", [True, False])
def test_partial_random_rotation_resumes_without_regenerating_the_key(
    tmp_path: Path, repeat_rotation_flag: bool
) -> None:
    api = fixture_api(
        cpu_data={"node-current": encoded("rotated"), "node-other": encoded("other")}
    )
    api.state["events"] = [{"on": "cpu:replace", "returncode": 1}]
    with pytest.raises(MODULE.ProvisionError, match="without confirmed progress"):
        provision(tmp_path, api, rotate_node="node-current")
    pending_gpu = api.state["secrets"]["gpu"]
    rotated = pending_gpu["data"]["node-current"]
    marker = pending_gpu["metadata"]["annotations"][MODULE.ROTATION_ANNOTATION]
    assert encoded("rotated") not in marker and rotated not in marker, (
        "pending rotation marker contains key material"
    )
    assert api.state["secrets"]["cpu"]["data"]["node-current"] == encoded("rotated"), (
        "failed write modified the CPU key"
    )
    api.state["events"] = []
    assert (
        provision(
            tmp_path, api, rotate_node="node-current" if repeat_rotation_flag else ""
        )
        == 1
    )
    assert (
        api.state["secrets"]["cpu"]["data"]["node-current"]
        == api.state["secrets"]["gpu"]["data"]["node-current"]
        == rotated
    ), "retry generated or installed a different rotation key"
    assert (
        MODULE.ROTATION_ANNOTATION
        not in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    assert "node-other" in api.state["secrets"]["cpu"]["data"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 2),
        ("cluster_id", "another-cluster"),
        ("node_uid", "recreated-node"),
        ("cpu_namespace", "another-namespace"),
        ("cpu_secret", "another-secret"),
        ("cpu_uid", "another-uid"),
        ("after_sha256", "0" * 64),
    ],
)
def test_pending_rotation_binding_drift_cannot_authorize_a_cpu_replacement(
    tmp_path: Path, field: str, value: object
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [{"on": "cpu:replace", "returncode": 1}]
    with pytest.raises(MODULE.ProvisionError):
        provision(tmp_path, api, rotate_node="node-current")
    annotations = api.state["secrets"]["gpu"]["metadata"]["annotations"]
    marker = json.loads(annotations[MODULE.ROTATION_ANNOTATION])
    marker[field] = value
    annotations[MODULE.ROTATION_ANNOTATION] = json.dumps(marker)
    before = len(api.state["write_attempts"])
    api.state["events"] = []
    with pytest.raises(MODULE.ProvisionError, match="binding differs"):
        provision(tmp_path, api)
    assert len(api.state["write_attempts"]) == before


def test_explicit_rotation_does_not_override_an_unknown_cpu_owner(
    tmp_path: Path,
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("other-owner")})
    with pytest.raises(MODULE.ProvisionError, match="NodeName key conflict"):
        provision(tmp_path, api, rotate_node="node-current")
    assert api.state["write_attempts"] == []


def test_unknown_rotation_node_fails_before_secret_mutation(tmp_path: Path) -> None:
    api = fixture_api()
    with pytest.raises(MODULE.ProvisionError, match="rotation target"):
        provision(tmp_path, api, rotate_node="node-unknown")
    assert api.state["write_attempts"] == []


def test_gpu_only_mode_does_not_access_the_cpu(tmp_path: Path) -> None:
    api = fixture_api(gpu_data={})
    assert provision(tmp_path, api, cpu=False) == 1
    assert all("--kubeconfig" not in args for args in api.state["calls"]), (
        "GPU-only provisioning must not use a CPU kubeconfig"
    )
    assert all(item["scope"] == "gpu" for item in api.state["writes"]), (
        "GPU-only provisioning must not write CPU Secrets"
    )


def test_identical_cpu_gpu_secret_uids_are_refused(tmp_path: Path) -> None:
    api = fixture_api(gpu_data={})
    api.state["secrets"]["cpu"]["metadata"]["uid"] = "fixture-gpu-secret"
    with pytest.raises(MODULE.ProvisionError, match="targets must be distinct"):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


def test_script_does_not_echo_sensitive_failure_output(tmp_path: Path) -> None:
    sentinel = "synthetic-sensitive-response-" * 3
    result, state = run_script(
        tmp_path,
        {
            "nodes": node_list("node-current"),
            "secrets": {
                "gpu": secret("gpu", {"node-current": encoded("rotated")}),
                "cpu": secret("cpu", {"node-other": encoded("other")}),
            },
            "events": [
                {
                    "on": "gpu:get-secret",
                    "returncode": 1,
                    "stdout": sentinel,
                    "stderr": sentinel,
                }
            ],
        },
    )
    assert result.returncode != 0
    assert sentinel not in result.stdout + result.stderr, "raw API output escaped"
    assert not state["write_attempts"]


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
def test_successful_write_with_changed_data_is_not_blessed(
    tmp_path: Path, scope: str
) -> None:
    api = fixture_api(gpu_data={})
    api.state["events"] = [
        {
            "on": f"{scope}:replace",
            "admission_data": {"node-current": encoded("unexpected")},
        }
    ]
    with pytest.raises(MODULE.ProvisionError, match="acknowledgement differs"):
        provision(tmp_path, api)
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == scope])
        == 1
    )


def test_an_immutable_cpu_secret_is_not_mutated(tmp_path: Path) -> None:
    api = fixture_api(gpu_data={})
    api.state["secrets"]["cpu"]["immutable"] = True
    with pytest.raises(MODULE.ProvisionError, match="immutable"):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []


def test_rotation_completion_ack_loss_does_not_repeat_rotation(tmp_path: Path) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [
        {"on": "gpu:replace", "occurrence": 2, "returncode": 1, "lost_ack": True}
    ]
    assert provision(tmp_path, api, rotate_node="node-current") == 1
    assert (
        MODULE.ROTATION_ANNOTATION
        not in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    assert (
        api.state["secrets"]["cpu"]["data"]["node-current"]
        == api.state["secrets"]["gpu"]["data"]["node-current"]
    ), "lost completion ACK changed the rotation target"
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == "gpu"])
        == 2
    )


def test_completed_cpu_rotation_with_pending_marker_is_resumable(
    tmp_path: Path,
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [{"on": "gpu:replace", "occurrence": 2, "returncode": 1}]
    with pytest.raises(MODULE.ProvisionError, match="without confirmed progress"):
        provision(tmp_path, api, rotate_node="node-current")
    pending = api.state["secrets"]["gpu"]["data"]["node-current"]
    before = len(api.state["writes"])
    api.state["events"] = []
    assert provision(tmp_path, api, rotate_node="node-current") == 1
    assert api.state["secrets"]["gpu"]["data"]["node-current"] == pending, (
        "completion retry performed another rotation"
    )
    assert [(item["scope"], item["verb"]) for item in api.state["writes"][before:]] == [
        ("gpu", "replace")
    ]


def test_no_secret_value_is_transmitted_as_a_command_argument(tmp_path: Path) -> None:
    api = fixture_api()
    assert provision(tmp_path, api) == 1
    all_arguments = "\n".join(" ".join(call) for call in api.state["calls"])
    assert "fixture-only-master-" not in all_arguments
    assert all(
        value not in all_arguments
        for document in api.state["secrets"].values()
        for value in document["data"].values()
    ), "a key appeared in argv"
    assert "--from-literal" not in all_arguments
    assert all("exec" not in call for call in api.state["calls"]), (
        "provisioning transmitted a credential to a Pod"
    )


def test_cpu_scope_can_use_an_explicit_context_without_kubeconfig(
    tmp_path: Path,
) -> None:
    result, state = run_script(
        tmp_path,
        fixture_api().state,
        extra_environment={
            "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": "",
            "GPU_FAULT_CONTROL_PLANE_CONTEXT": "fixture-cpu-context",
        },
    )
    assert result.returncode == 0, "CPU context-only synchronization failed"
    assert any(
        "--context" in args and "fixture-cpu-context" in args for args in state["calls"]
    ), "CPU Secret provisioning must use the configured context"
    assert all("--kubeconfig" not in args for args in state["calls"]), (
        "context-only provisioning must not inject a kubeconfig"
    )


def test_a_concurrent_cpu_addition_after_our_write_does_not_fail_or_lose_keys(
    tmp_path: Path,
) -> None:
    api = fixture_api()
    api.state["events"] = [
        {
            "on": "cpu:get-secret",
            "occurrence": 3,
            "merge_data": {"node-concurrent": encoded("concurrent")},
        }
    ]
    assert provision(tmp_path, api) == 1
    assert set(api.state["secrets"]["cpu"]["data"]) == {
        "node-current",
        "node-other",
        "node-concurrent",
    }
    assert len(api.state["write_attempts"]) == 1


@pytest.mark.parametrize(
    "expected", ['["node-current"]', '{"node-current":"fixture-uid-node-current"}']
)
def test_script_consumes_the_join_runners_verified_node_binding(
    tmp_path: Path, expected: str
) -> None:
    result, state = run_script(
        tmp_path,
        fixture_api().state,
        extra_environment={"GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON": expected},
    )
    assert result.returncode == 0, "matching verified node inventory was refused"
    assert set(state["secrets"]["cpu"]["data"]) == {"node-current", "node-other"}


@pytest.mark.parametrize(
    "expected",
    [
        '["node-other"]',
        '["node-current","node-extra"]',
        '{"node-current":"stale-node-uid"}',
        "",
        "not-json",
        "null",
        "[]",
        "{}",
        '["node-current","node-current"]',
        '[{"name":"node-current"}]',
        '{"node-current":""}',
        '{"node-current":"uid","node-other":"uid"}',
        '{"node-current":"old","node-current":"new"}',
    ],
)
def test_invalid_or_changed_verified_inventory_stops_the_script_before_writes(
    tmp_path: Path, expected: str
) -> None:
    result, state = run_script(
        tmp_path,
        fixture_api().state,
        extra_environment={"GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON": expected},
    )
    assert result.returncode != 0, "unbound or changed node inventory was accepted"
    assert state["write_attempts"] == []


def test_bundle_layout_uses_the_sibling_helper_with_deploy_host_imports(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "node-bundle" / "deploy" / "node"
    directory.mkdir(parents=True)
    for name in ("provision-node-action-keys.sh", "provision_node_action_keys.py"):
        shutil.copyfile(SCRIPT.with_name(name), directory / name)
    result, state = run_script(
        tmp_path, fixture_api().state, script=directory / SCRIPT.name
    )
    assert result.returncode == 0, (
        "bundled entry could not use the deploy-host environment"
    )
    assert state["writes"], "bundle-layout smoke test did not exercise synchronization"


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize("create", [True, False])
@pytest.mark.parametrize("disappear", [True, False])
def test_successful_write_readback_rejects_secret_reincarnation_or_absence(
    tmp_path: Path, scope: str, create: bool, disappear: bool
) -> None:
    api = fixture_api(gpu_data={})
    if create:
        api.state["secrets"][scope] = None
    replacement = secret(scope, {})
    replacement["metadata"]["uid"] = "recreated-secret"
    api.state["events"] = [
        {
            "on": f"{scope}:get-secret",
            "occurrence": 2 if scope == "gpu" else 3,
            "document": None if disappear else replacement,
        }
    ]
    with pytest.raises(MODULE.ProvisionError, match="disappeared|UID changed"):
        provision(tmp_path, api)
    attempts = [item for item in api.state["write_attempts"] if item["scope"] == scope]
    assert len(attempts) == 1
    assert attempts[0]["verb"] == ("create" if create else "replace")
    if scope == "gpu":
        assert all(item["scope"] == "gpu" for item in api.state["write_attempts"]), (
            "failed GPU identity readback must block CPU writes"
        )


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
@pytest.mark.parametrize(
    "output", ["", "null", "not-json", '{"kind":"Secret","kind":"ConfigMap"}']
)
def test_successful_write_requires_a_parseable_unambiguous_acknowledgement(
    tmp_path: Path, scope: str, output: str
) -> None:
    api = fixture_api(gpu_data={})
    api.state["events"] = [{"on": f"{scope}:replace", "output": output}]
    with pytest.raises(MODULE.ProvisionError):
        provision(tmp_path, api)
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == scope])
        == 1
    )
    assert api.state["counts"][f"{scope}:get-secret"] == (1 if scope == "gpu" else 2)


@pytest.mark.parametrize("create", [True, False])
def test_gpu_successful_ack_readback_preserves_a_concurrent_random_key(
    tmp_path: Path, create: bool
) -> None:
    api = fixture_api(gpu_data={})
    if create:
        api.state["secrets"]["gpu"] = None
    api.state["events"] = [
        {
            "on": "gpu:get-secret",
            "occurrence": 2,
            "merge_data": {"node-current": encoded("concurrent-rotation")},
        }
    ]
    assert provision(tmp_path, api) == 1
    assert (
        api.state["secrets"]["gpu"]["data"]["node-current"]
        == api.state["secrets"]["cpu"]["data"]["node-current"]
        == encoded("concurrent-rotation")
    ), "readback retry discarded a concurrent random GPU key"
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == "gpu"])
        == 1
    )
    assert "node-other" in api.state["secrets"]["cpu"]["data"]


def test_cpu_successful_ack_readback_conflict_is_not_overwritten(
    tmp_path: Path,
) -> None:
    api = fixture_api()
    api.state["events"] = [
        {
            "on": "cpu:get-secret",
            "occurrence": 3,
            "merge_data": {"node-current": encoded("other-owner")},
        }
    ]
    with pytest.raises(MODULE.ProvisionError, match="NodeName key conflict"):
        provision(tmp_path, api)
    assert len(api.state["write_attempts"]) == 1
    assert api.state["secrets"]["cpu"]["data"]["node-current"] == encoded(
        "other-owner"
    ), "readback conflict overwrote another CPU key owner"
    assert "node-other" in api.state["secrets"]["cpu"]["data"]


@pytest.mark.parametrize("scope", ["gpu", "cpu"])
def test_successful_ack_with_unreadable_readback_resumes_from_current_values(
    tmp_path: Path, scope: str
) -> None:
    api = fixture_api(gpu_data={})
    api.state["events"] = [
        {
            "on": f"{scope}:get-secret",
            "occurrence": 2 if scope == "gpu" else 3,
            "returncode": 1,
        }
    ]
    with pytest.raises(ReleaseError):
        provision(tmp_path, api)
    saved = dict(api.state["secrets"]["gpu"]["data"])
    api.state["events"] = []
    assert provision(tmp_path, api) == 1
    assert api.state["secrets"]["gpu"]["data"] == saved, (
        "retry replaced an already committed GPU key"
    )
    assert all(
        api.state["secrets"]["cpu"]["data"].get(name) == value
        for name, value in saved.items()
    ), "retry did not complete the CPU mirror"
    assert [item["scope"] for item in api.state["write_attempts"]] == ["gpu", "cpu"]


def test_rotation_completion_cas_retry_preserves_unrelated_metadata(
    tmp_path: Path,
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [
        {
            "on": "gpu:replace",
            "occurrence": 2,
            "merge_metadata": {"labels": {"external-owner": "keep"}},
        }
    ]
    assert provision(tmp_path, api, rotate_node="node-current") == 1
    gpu = api.state["secrets"]["gpu"]
    assert gpu["metadata"]["labels"] == {"external-owner": "keep"}
    assert MODULE.ROTATION_ANNOTATION not in gpu["metadata"]["annotations"]
    assert gpu["data"] == api.state["secrets"]["cpu"]["data"], (
        "rotation completion retry changed the mirrored keys"
    )
    attempts = [item for item in api.state["write_attempts"] if item["scope"] == "gpu"]
    assert [item["version"] for item in attempts] == ["1", "2", "3"]


def test_repeated_rotation_completion_conflicts_leave_a_resumable_marker(
    tmp_path: Path,
) -> None:
    api = fixture_api(
        cpu_data={"node-current": encoded("rotated"), "node-other": encoded("other")}
    )
    api.state["events"] = [
        {
            "on": "gpu:replace",
            "occurrence": index,
            "merge_metadata": {"labels": {"external-owner": str(index)}},
        }
        for index in range(2, 5)
    ]
    with pytest.raises(MODULE.ProvisionError, match="completion changed repeatedly"):
        provision(tmp_path, api, rotate_node="node-current")
    saved = dict(api.state["secrets"]["gpu"]["data"])
    assert (
        MODULE.ROTATION_ANNOTATION
        in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    assert (
        len([item for item in api.state["write_attempts"] if item["scope"] == "gpu"])
        == 4
    )
    before = len(api.state["writes"])
    api.state["events"] = []
    assert provision(tmp_path, api, rotate_node="node-current") == 1
    assert api.state["secrets"]["gpu"]["data"] == saved, (
        "completion retry generated another rotation"
    )
    assert [(item["scope"], item["verb"]) for item in api.state["writes"][before:]] == [
        ("gpu", "replace")
    ]
    assert (
        MODULE.ROTATION_ANNOTATION
        not in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    assert set(api.state["secrets"]["cpu"]["data"]) == {"node-current", "node-other"}


def test_another_confirmed_writer_can_finish_the_same_rotation(tmp_path: Path) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [
        {
            "on": "gpu:get-secret",
            "occurrence": 4,
            "merge_metadata": {"annotations": {"external-owner": "keep"}},
        }
    ]
    assert provision(tmp_path, api, rotate_node="node-current") == 1
    assert api.state["secrets"]["gpu"]["metadata"]["annotations"] == {
        "external-owner": "keep"
    }
    assert api.state["secrets"]["gpu"]["data"] == api.state["secrets"]["cpu"]["data"], (
        "concurrent completion changed the mirrored rotation"
    )
    assert [item["scope"] for item in api.state["write_attempts"]] == ["gpu", "cpu"]


@pytest.mark.parametrize("marker", ["", "{}"])
def test_final_read_cannot_bless_a_new_pending_rotation(
    tmp_path: Path, marker: str
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [
        {
            "on": "gpu:get-secret",
            "occurrence": 3,
            "merge_metadata": {"annotations": {MODULE.ROTATION_ANNOTATION: marker}},
        }
    ]
    with pytest.raises(MODULE.ProvisionError, match="did not converge"):
        provision(tmp_path, api)
    assert api.state["write_attempts"] == []
    assert (
        MODULE.ROTATION_ANNOTATION
        in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )


def test_rotation_completion_keeps_the_marker_after_node_uid_drift(
    tmp_path: Path,
) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    changed = node_list("node-current")
    changed["items"][0]["metadata"]["uid"] = "recreated-node"
    api.state["events"] = [{"on": "gpu:get-nodes", "occurrence": 4, "nodes": changed}]
    with pytest.raises(MODULE.ProvisionError, match="identities changed"):
        provision(tmp_path, api, rotate_node="node-current")
    assert (
        MODULE.ROTATION_ANNOTATION
        in api.state["secrets"]["gpu"]["metadata"]["annotations"]
    ), "Node UID drift allowed the recovery marker to be removed"
    assert [item["scope"] for item in api.state["write_attempts"]] == ["gpu", "cpu"]


def test_pending_rotation_requires_its_cpu_scope(tmp_path: Path) -> None:
    api = fixture_api(cpu_data={"node-current": encoded("rotated")})
    api.state["events"] = [{"on": "cpu:replace", "returncode": 1}]
    with pytest.raises(MODULE.ProvisionError, match="without confirmed progress"):
        provision(tmp_path, api, rotate_node="node-current")
    api.state["events"] = []
    attempts = len(api.state["write_attempts"])
    with pytest.raises(MODULE.ProvisionError, match="binding differs"):
        provision(tmp_path, api, cpu=False)
    assert len(api.state["write_attempts"]) == attempts


@pytest.mark.parametrize("failed_phase", ["cpu", "completion"])
@pytest.mark.parametrize("repeat_rotation_flag", [True, False])
def test_script_rotation_retry_roundtrip_keeps_the_committed_random_key(
    tmp_path: Path, failed_phase: str, repeat_rotation_flag: bool
) -> None:
    sentinel = "synthetic-sensitive-write-response-" * 3
    api = fixture_api(
        cpu_data={"node-current": encoded("rotated"), "node-other": encoded("other")}
    )
    api.state["events"] = [
        {
            "on": "cpu:replace" if failed_phase == "cpu" else "gpu:replace",
            "occurrence": 1 if failed_phase == "cpu" else 2,
            "returncode": 1,
            "stderr": sentinel,
        }
    ]
    first, state = run_script(
        tmp_path,
        api.state,
        extra_environment={"GPU_FAULT_ROTATE_NODE_ACTION_KEY": "node-current"},
    )
    assert first.returncode != 0, "the interrupted rotation was reported complete"
    saved = dict(state["secrets"]["gpu"]["data"])
    assert (
        MODULE.ROTATION_ANNOTATION in state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    before = len(state["writes"])
    state["events"] = []
    second, state = run_script(
        tmp_path,
        state,
        extra_environment={
            "GPU_FAULT_ROTATE_NODE_ACTION_KEY": (
                "node-current" if repeat_rotation_flag else ""
            )
        },
    )
    assert second.returncode == 0, "the pending rotation did not resume"
    assert state["secrets"]["gpu"]["data"] == saved, (
        "a new process generated another GPU rotation key"
    )
    assert all(
        state["secrets"]["cpu"]["data"].get(name) == value
        for name, value in saved.items()
    ), "the CPU mirror did not receive the committed random key"
    assert [item["scope"] for item in state["writes"][before:]] == (
        ["cpu", "gpu"] if failed_phase == "cpu" else ["gpu"]
    )
    assert (
        MODULE.ROTATION_ANNOTATION
        not in state["secrets"]["gpu"]["metadata"]["annotations"]
    )
    assert "node-other" in state["secrets"]["cpu"]["data"]
    before = len(state["write_attempts"])
    third, state = run_script(tmp_path, state)
    assert third.returncode == 0, "ensure did not converge after rotation completion"
    assert len(state["write_attempts"]) == before, "completed ensure was not read-only"
    diagnostics = "".join(
        result.stdout + result.stderr for result in (first, second, third)
    )
    assert sentinel not in diagnostics, "write diagnostics exposed raw API output"
    assert "fixture-only-master-" not in diagnostics
    assert all(
        value not in diagnostics
        and base64.b64decode(value, validate=True).decode() not in diagnostics
        for document in state["secrets"].values()
        for value in document["data"].values()
    ), "rotation diagnostics exposed key material"
