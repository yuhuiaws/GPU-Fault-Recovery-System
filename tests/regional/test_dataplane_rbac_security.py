"""Guards from the 2026-09-07 data-plane security review (H-1, M-14, H-4, M-4/M-9, H-7).

Each check names the failure it exists to stop, because a green scan that
reads no files reports the same thing as a fixed manifest:

* A ClusterRole granting ``pods/exec`` is a shell into every Pod on the
  cluster -- ``resourceNames`` does not apply to subresources, so only a
  namespaced Role can scope it. The same ClusterRole shape with ``pods
  delete`` or ``jobs create`` lets a compromised data-plane identity stop or
  schedule work in namespaces the executor's own allow-list never mentions.
* A migration Pod that puts a Secret on an argv publishes it through
  ``/proc/*/cmdline`` for the lifetime of the process; under ``hostPID`` that
  is every process on the node.
* A reconciler that trusts the template ConfigMap turns "can update one
  ConfigMap" into "runs a privileged hostPath Pod on every GPU node".
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import yaml

from gpu_fault.node_installer_reconciler import (
    TEMPLATE_CONTENT_SHA256_ENV,
    NodeInstallerReconciler,
    load_job_template,
)
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_gpu_rollout as ROLLOUT

ROOT = Path(__file__).resolve().parents[2]
DATAPLANE = ROOT / "deploy/dataplane"
MIGRATIONS = ROOT / "deploy/migrations"

#: Verbs no data-plane ClusterRole may hold on a namespaced workload kind. The
#: read side (get/list/watch) is deliberately absent: the watcher's Pod watch
#: and spare activation's idle check legitimately span every namespace.
FORBIDDEN_CLUSTER_WIDE = {
    "pods/exec": {"create", "get", "*"},
    "pods": {"delete", "deletecollection", "*"},
    "jobs": {"create", "*"},
    "pytorchjobs": {"create", "*"},
    "jobsets": {"create", "*"},
}


def documents(path: Path) -> list[dict[str, Any]]:
    return [
        item
        for item in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(item, dict)
    ]


def cluster_roles() -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(DATAPLANE.rglob("*.yaml")):
        for document in documents(path):
            if document.get("kind") == "ClusterRole":
                yield path, document


def containers(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    spec = (document.get("spec") or {}).get("template", {}).get("spec") or {}
    if document.get("kind") == "Pod":
        spec = document.get("spec") or {}
    yield from spec.get("initContainers") or []
    yield from spec.get("containers") or []


# --------------------------------------------------------------------------- H-1 / M-14


def test_no_dataplane_clusterrole_grants_exec_delete_or_create_cluster_wide() -> None:
    offending = []
    scanned = 0
    for path, role in cluster_roles():
        scanned += 1
        for rule in role.get("rules") or []:
            verbs = {str(item).lower() for item in rule.get("verbs") or []}
            for resource in rule.get("resources") or []:
                forbidden = FORBIDDEN_CLUSTER_WIDE.get(str(resource).lower())
                if forbidden and verbs & forbidden:
                    offending.append(
                        (path.name, role["metadata"]["name"], resource, sorted(verbs))
                    )

    assert offending == []
    # Positive control: the scan saw the real ClusterRoles, and the matcher does
    # fire on the shape this test forbids.
    assert scanned >= 3, scanned
    assert FORBIDDEN_CLUSTER_WIDE["pods/exec"] & {"create"}


def test_watcher_write_verbs_moved_into_the_namespaced_role() -> None:
    """The exec the watcher needs still exists -- in the Role, not the ClusterRole."""

    rendered = ROLLOUT.render_workload_namespace_rbac(
        ("training",), system_namespace="gpu-fault-system"
    )
    watcher = rendered[inventory.GPU_WATCHER_DEPLOYMENT]
    role = next(item for item in watcher if item["kind"] == "Role")
    binding = next(item for item in watcher if item["kind"] == "RoleBinding")
    verbs = {
        resource: set(rule["verbs"])
        for rule in role["rules"]
        for resource in rule["resources"]
    }

    assert role["metadata"]["namespace"] == "training"
    assert verbs["pods/exec"] == {"get", "create"}
    assert verbs["pods/log"] == {"get"}
    assert verbs["pods"] == {"patch"}
    assert verbs["jobs"] == verbs["pytorchjobs"] == verbs["jobsets"] == {"patch"}
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": "gpu-fault-completion-watcher",
    }
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "gpu-fault-completion-watcher",
            "namespace": "gpu-fault-system",
        }
    ]


def test_executor_namespaced_role_holds_exactly_the_write_verbs() -> None:
    rendered = ROLLOUT.render_workload_namespace_rbac(
        ("training", "gpu-fault-system"), system_namespace="gpu-fault-system"
    )
    executor = rendered[inventory.GPU_EXECUTOR_DEPLOYMENT]
    by_namespace = {
        item["metadata"]["namespace"]: item
        for item in executor
        if item["kind"] == "Role"
        and item["metadata"]["name"] == "gpu-fault-cluster-executor"
    }
    device_plugin = [
        item
        for item in executor
        if item["kind"] == "Role"
        and item["metadata"]["name"] == "gpu-fault-cluster-executor-device-plugin"
    ]

    assert set(by_namespace) == {"training", "gpu-fault-system"}
    verbs = {
        resource: set(rule["verbs"])
        for rule in by_namespace["training"]["rules"]
        for resource in rule["resources"]
    }
    assert verbs == {
        "pods": {"patch", "delete"},
        "jobs": {"create", "patch"},
        "pytorchjobs": {"create", "patch"},
        "jobsets": {"create", "patch"},
    }
    # No exec for the executor: it never runs a command inside a training Pod.
    assert "pods/exec" not in verbs
    # RESTART_*_DEVICE_PLUGIN deletes one plugin Pod in kube-system and nothing
    # else there -- so that namespace gets a Role with a single verb.
    assert [item["metadata"]["namespace"] for item in device_plugin] == ["kube-system"]
    assert device_plugin[0]["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["delete"]}
    ]
    # Every object carries the prune label and no other file-level identity.
    assert all(
        item["metadata"]["labels"] == {ROLLOUT.WORKLOAD_NAMESPACE_RBAC_LABEL: "true"}
        for item in executor
    ), [item["metadata"].get("labels") for item in executor]


def test_rollout_attaches_namespaced_rbac_to_the_identity_it_binds() -> None:
    """Applied with the Deployment, so the same dry-run preflight covers it."""

    manifests = {
        inventory.GPU_EXECUTOR_DEPLOYMENT: "kind: Deployment\nmetadata: {name: executor}\n",
        inventory.GPU_WATCHER_DEPLOYMENT: "kind: Deployment\nmetadata: {name: watcher}\n",
        inventory.GPU_COLLECTOR_DEPLOYMENT: "kind: Deployment\nmetadata: {name: c}\n",
    }
    target = SimpleNamespace(allowed_namespaces=("training",))

    ROLLOUT.append_workload_namespace_rbac(
        manifests, target, system_namespace="gpu-fault-system"
    )

    executor_kinds = [
        item["kind"]
        for item in yaml.safe_load_all(manifests[inventory.GPU_EXECUTOR_DEPLOYMENT])
    ]
    watcher_kinds = [
        item["kind"]
        for item in yaml.safe_load_all(manifests[inventory.GPU_WATCHER_DEPLOYMENT])
    ]
    assert executor_kinds == [
        "Deployment",
        "Role",
        "RoleBinding",
        "Role",
        "RoleBinding",
    ]
    assert watcher_kinds == ["Deployment", "Role", "RoleBinding"]
    # The collector never writes to a workload namespace and gets nothing.
    assert manifests[inventory.GPU_COLLECTOR_DEPLOYMENT].count("kind:") == 1


def test_rollout_leaves_a_target_without_an_allow_list_attribute_alone() -> None:
    manifests = {inventory.GPU_EXECUTOR_DEPLOYMENT: "executor-yaml"}

    ROLLOUT.append_workload_namespace_rbac(
        manifests, SimpleNamespace(cluster_id="gpu-a"), system_namespace="ns"
    )

    assert manifests == {inventory.GPU_EXECUTOR_DEPLOYMENT: "executor-yaml"}


def test_prune_removes_roles_only_from_namespaces_no_longer_allowed() -> None:
    calls: list[list[str]] = []
    listing = {
        "items": [
            {"metadata": {"namespace": "training"}},
            {"metadata": {"namespace": "retired-team"}},
            {"metadata": {"namespace": "kube-system"}},
        ]
    }

    class Runner:
        dry_run = False

        def run(self, args, *, capture=False, **_kwargs):
            calls.append(list(args))
            return json.dumps(listing) if capture else ""

    release = SimpleNamespace(runner=Runner(), _gpu=lambda _target, *args: list(args))
    target = SimpleNamespace(allowed_namespaces=("training",))

    pruned = ROLLOUT.prune_workload_namespace_rbac(release, target)

    assert pruned == ["retired-team"]
    deletes = [args for args in calls if "delete" in args]
    assert deletes == [
        [
            "-n",
            "retired-team",
            "delete",
            "rolebinding,role",
            "-l",
            f"{ROLLOUT.WORKLOAD_NAMESPACE_RBAC_LABEL}=true",
            "--ignore-not-found",
        ]
    ]
    # kube-system is kept without being in the allow-list: the device-plugin
    # Role lives there by design.
    assert not any("kube-system" in args for args in deletes), deletes


def test_prune_is_skipped_on_dry_run() -> None:
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=True, run=lambda *a, **k: pytest.fail("ran")),
        _gpu=lambda _target, *args: list(args),
    )

    assert (
        ROLLOUT.prune_workload_namespace_rbac(
            release, SimpleNamespace(allowed_namespaces=("training",))
        )
        == []
    )


# --------------------------------------------------------------------------- M-4 / M-9 / H-7

#: ``env VAR=value cmd`` -- bare or as ``/usr/bin/env`` -- which places VAR on
#: cmd's argv for the life of the process.
ENV_ON_ARGV = re.compile(r"(?:^|[\s;&|(/])env\s+(?:-\S+\s+)*[A-Za-z_][A-Za-z0-9_]*=")


def secret_env_names(container: dict[str, Any]) -> set[str]:
    return {
        item["name"]
        for item in container.get("env") or []
        if ((item.get("valueFrom") or {}).get("secretKeyRef")) is not None
    }


def shell_scripts(container: dict[str, Any]) -> list[str]:
    command = [str(item) for item in container.get("command") or []]
    args = [str(item) for item in container.get("args") or []]
    if (
        command[:1] in (["/bin/sh"], ["sh"], ["/bin/bash"], ["bash"])
        and "-c" in command
    ):
        return args or command[command.index("-c") + 1 :]
    return []


def test_no_migration_manifest_passes_a_secret_on_an_argv() -> None:
    findings = []
    scanned = 0
    for path in sorted(MIGRATIONS.glob("*.yaml")):
        for document in documents(path):
            for container in containers(document):
                scanned += 1
                secrets = secret_env_names(container)
                scripts = shell_scripts(container)
                for script in scripts:
                    # `env VAR=... cmd` puts VAR on cmd's argv for its lifetime.
                    if ENV_ON_ARGV.search(script):
                        findings.append(
                            (path.name, container["name"], "env VAR= on argv")
                        )
                    # A Secret-sourced variable expanded inside `sh -c` lands on
                    # the argv of whatever the shell execs.
                    for name in secrets:
                        if re.search(rf"\$\{{?{re.escape(name)}\b", script):
                            findings.append((path.name, container["name"], name))
                # Kubernetes' own $(VAR) expansion into args is the same leak
                # without a shell.
                for argument in container.get("args") or []:
                    for name in secrets:
                        if f"$({name})" in str(argument):
                            findings.append(
                                (path.name, container["name"], f"$({name})")
                            )

    assert findings == []
    assert scanned >= 5, scanned
    # Positive control for the argv matcher.
    assert ENV_ON_ARGV.search("chroot /host /usr/bin/env SECRET=x python"), (
        "env assignment after a chroot prefix must be caught"
    )
    assert ENV_ON_ARGV.search("  env A=1 cmd"), (
        "leading whitespace must not hide an env assignment"
    )
    assert not ENV_ON_ARGV.search('printf "environment=%s"'), (
        "a printf format string is not an env assignment"
    )


def test_endpoint_migration_reads_the_fleet_master_from_stdin_only() -> None:
    text = (MIGRATIONS / "gpu-node-collector-endpoint-migration.yaml").read_text(
        encoding="utf-8"
    )

    # Never captured into a shell variable (which the old script then put on
    # an argv), and no `unset` theatre pretending to retract it.
    assert not re.search(
        r'\$\(\s*cat\s+"?\$\{connection_dir\}/node-action-secret', text
    ), "the node-action secret is captured into a shell variable"
    assert "GPU_FAULT_FLEET_SECRET=" not in text
    assert "unset node_action_master" not in text
    assert '<"${connection_dir}/node-action-secret"' in text
    assert "sys.stdin.read()" in text


def test_endpoint_migration_never_writes_plaintext_unconditionally() -> None:
    text = (MIGRATIONS / "gpu-node-collector-endpoint-migration.yaml").read_text(
        encoding="utf-8"
    )
    daemonset = next(
        item
        for item in documents(MIGRATIONS / "gpu-node-collector-endpoint-migration.yaml")
        if item["kind"] == "DaemonSet"
    )
    container = next(containers(daemonset))
    env = {item["name"]: item.get("value") for item in container["env"]}

    assert not re.search(r"ALLOW_PLAINTEXT\s+(true|\"true\"|'true')", text), (
        "the migration must not write ALLOW_PLAINTEXT true as a fixed value"
    )
    assert env["MIGRATION_ALLOW_PLAINTEXT_NODE_AGENT"] == "false"
    assert 'if [ "${allow_plaintext}" = "true" ]' in text


def test_aurora_migration_runs_the_tool_directly_with_urls_in_env() -> None:
    job = next(
        item
        for item in documents(MIGRATIONS / "postgres-to-aurora-migration.yaml")
        if item["kind"] == "Job"
    )
    container = next(containers(job))
    names = secret_env_names(container)

    assert container["command"] == ["gpu-fault-store-migrate"]
    assert container["args"] == ["--source-postgres-url-env", "SOURCE_POSTGRES_URL"]
    assert {"SOURCE_POSTGRES_URL", "GPU_FAULT_STORE_URL"} <= names
    assert not any("$" in str(item) for item in container["args"]), container["args"]


# --------------------------------------------------------------------------- H-4

TEMPLATE_TEXT = "apiVersion: batch/v1\nkind: Job\nspec: {template: {spec: {}}}\n"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE_TEXT.encode()).hexdigest()


def test_reconciler_refuses_a_template_whose_digest_does_not_match() -> None:
    with pytest.raises(RuntimeError, match="does not match"):
        load_job_template(
            TEMPLATE_TEXT + "# tampered\n",
            expected_sha256=TEMPLATE_SHA,
            origin="test/job.yaml",
        )


def test_reconciler_refuses_to_start_without_a_template_digest() -> None:
    with pytest.raises(RuntimeError, match=TEMPLATE_CONTENT_SHA256_ENV):
        load_job_template(TEMPLATE_TEXT, expected_sha256=None, origin="test/job.yaml")
    with pytest.raises(RuntimeError, match=TEMPLATE_CONTENT_SHA256_ENV):
        load_job_template(TEMPLATE_TEXT, expected_sha256="", origin="test/job.yaml")


def test_reconciler_accepts_the_pinned_template() -> None:
    template = load_job_template(
        TEMPLATE_TEXT.encode(), expected_sha256=TEMPLATE_SHA, origin="test/job.yaml"
    )

    assert template["kind"] == "Job"


def test_reconciler_has_no_default_identity_pin() -> None:
    """A template digest derived from the config digest matched anything."""

    with pytest.raises(TypeError):
        NodeInstallerReconciler(  # type: ignore[call-arg]
            object(),
            object(),
            namespace="ns",
            cluster_name="hp",
            version="0.10.0",
            config_digest="cfg",
            artifact_sha256="a" * 64,
            job_template={},
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        )


def test_deploy_pins_the_content_digest_of_the_job_yaml_it_ships() -> None:
    script = (ROOT / "deploy/node/deploy-node-installer-reconciler.sh").read_text(
        encoding="utf-8"
    )
    manifest = (DATAPLANE / "node-installer-reconciler.yaml").read_text(
        encoding="utf-8"
    )

    # Render path: the digest of the file that becomes the ConfigMap.
    assert 'TEMPLATE_CONTENT_SHA256="${TEMPLATE_SHA256}"' in script
    # Override path: the digest of what the named ConfigMap actually holds.
    assert 'text = (json.load(sys.stdin).get("data") or {}).get("job.yaml")' in script
    assert (
        "REPLACE_WITH_INSTALLER_TEMPLATE_CONTENT_SHA256#${TEMPLATE_CONTENT_SHA256}"
        in script
    )
    assert "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256" in manifest
    assert "REPLACE_WITH_INSTALLER_TEMPLATE_CONTENT_SHA256" in manifest
