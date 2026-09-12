"""The reconciler deploy script's products are made once per release and reused.

Live 2026-09-12 (join, 4 nodes): ``deploy-node-installer-reconciler.sh`` ran
three times in one release, and the two mutating runs each provisioned the node
action keys again, rendered the template again and synced the Secret again.
The first mutating run now reports what it produced; ``deploy_reconciler``
hands that back to the later runs as hints, and the script re-checks every hint
against the live cluster before trusting it.

Two layers are exercised: the Python side (scope of the hints, state record,
what an explicit override or a foreign release does to them) and the script
itself, run end to end against a fake ``kubectl`` and fake sibling scripts so
that the reuse decision, its fallbacks and the read-only preflight are observed
rather than grepped for.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_fleet_rollout as FLEET_ROLLOUT
from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release import regional_release_state as STATE

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/node/deploy-node-installer-reconciler.sh"
MANIFEST = ROOT / "deploy/dataplane/node-installer-reconciler.yaml"
BUNDLE = "b" * 64
TEMPLATE = "e" * 64
NODE_SET = hashlib.sha256(b"node-a\nnode-b\n").hexdigest()
TEMPLATE_CM = "gpu-fault-node-installer-template-0123456789ab"
PRODUCTS = {
    "node_set_sha256": NODE_SET,
    "node_action_keys_provisioned": True,
    "template_config_map": TEMPLATE_CM,
    "template_content_sha256": "0123456789ab" + "f" * 52,
    "template_rendered": True,
}
BASE_ENVIRONMENT = {
    "GPU_FAULT_KUBECTL_CONTEXT": "gpu-a-context",
    "GPU_FAULT_NAMESPACE": "gpu-fault-system",
    "GPU_FAULT_CLUSTER_ID": "gpu-a",
    "GPU_FAULT_HYPERPOD_CLUSTER": "hp-gpu-a",
    "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
    "GPU_FAULT_INSTALLER_BUNDLE_SHA256": BUNDLE,
    "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": TEMPLATE,
    "GPU_FAULT_FLEET_MASTER_FILE": "/secure/fleet-master",
}


# --- the Python side -------------------------------------------------------


def _release(
    *,
    release_id: str = "rel-1",
    products: dict[str, Any] | str | None = PRODUCTS,
    dry_run: bool = False,
) -> tuple[SimpleNamespace, list[dict[str, str]]]:
    environments: list[dict[str, str]] = []

    class Runner:
        def __init__(self) -> None:
            self.dry_run = dry_run

        def run(self, arguments: list[str], *, env=None, **_keywords: Any) -> str:
            environments.append(dict(env or {}))
            path = (env or {}).get(RENDERING.INSTALLER_PRODUCTS_FILE_ENV)
            if path and products is not None:
                text = products if isinstance(products, str) else json.dumps(products)
                Path(path).write_text(text, encoding="utf-8")
            return ""

    release = SimpleNamespace(
        runner=Runner(),
        state={},
        release_id=release_id,
        bundle_sha=BUNDLE,
        node_template_sha=TEMPLATE,
        config=SimpleNamespace(upgrade_max_unavailable=0),
        _settle_installer_jobs=lambda _target: None,
    )
    return release, environments


def _stub_reconciler_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    def environment(_release: Any, _target: Any, **keywords: Any) -> dict[str, str]:
        value = dict(BASE_ENVIRONMENT)
        value["GPU_FAULT_INSTALLER_ARTIFACT_SHA256"] = keywords["artifact_sha"]
        if keywords.get("template_config_map"):
            value["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] = keywords[
                "template_config_map"
            ]
        return value

    monkeypatch.setattr(FLEET_ROLLOUT, "build_reconciler_environment", environment)
    monkeypatch.setattr(
        FLEET_ROLLOUT, "wait_deployment_rollout", lambda *_a, **_k: {"object": {}}
    )
    monkeypatch.setattr(
        FLEET_ROLLOUT,
        "reconciler_container_env",
        lambda *_a, **_k: {
            FLEET_ROLLOUT.INSTALLER_BUNDLE_ENV: BUNDLE,
            FLEET_ROLLOUT.INSTALLER_TEMPLATE_ENV: TEMPLATE,
        },
    )


def _deploy(release: Any, *, artifact: str = "a" * 64, **keywords: Any) -> None:
    FLEET_ROLLOUT.deploy_reconciler(
        release,
        SimpleNamespace(cluster_id="gpu-a", fleet_master_file="/secure/fleet-master"),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha=artifact,
        config_digest="c" * 64,
        **keywords,
    )


def test_first_deploy_records_products_and_the_next_one_hands_them_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_reconciler_seams(monkeypatch)
    release, environments = _release()

    _deploy(release)
    _deploy(release)

    first, second = environments
    assert RENDERING.INSTALLER_REUSE_NODE_SET_ENV not in first, (
        "the first run of a release has nothing to reuse"
    )
    assert RENDERING.INSTALLER_REUSE_TEMPLATE_ENV not in first
    record = release.state[RENDERING.INSTALLER_PRODUCTS_STATE_KEY]["gpu-a"]
    assert record["release_id"] == "rel-1"
    assert record["node_set_sha256"] == NODE_SET
    assert record["template_config_map"] == TEMPLATE_CM
    assert record["template_content_sha256"] == PRODUCTS["template_content_sha256"]
    assert record["node_action_keys_provisioned"] is True
    assert record["inputs_sha256"] == RENDERING.installer_product_inputs_digest(first)
    assert second[RENDERING.INSTALLER_REUSE_NODE_SET_ENV] == NODE_SET
    assert second[RENDERING.INSTALLER_REUSE_TEMPLATE_ENV] == TEMPLATE_CM
    products_files = {
        env[RENDERING.INSTALLER_PRODUCTS_FILE_ENV] for env in environments
    }
    assert len(products_files) == 2, "each run reports into its own file"
    assert not any(Path(path).parent.exists() for path in products_files), (
        "the product report directory is removed once it has been read"
    )


def test_an_explicit_template_override_outranks_the_recorded_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback names the steady template it wants; that is not a hint."""

    _stub_reconciler_seams(monkeypatch)
    release, environments = _release()

    _deploy(release)
    _deploy(release, template_config_map="gpu-fault-node-installer-template-steady")

    second = environments[1]
    assert RENDERING.INSTALLER_REUSE_TEMPLATE_ENV not in second
    assert (
        second["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"]
        == "gpu-fault-node-installer-template-steady"
    )
    assert RENDERING.INSTALLER_REUSE_NODE_SET_ENV not in second, (
        "an override changes the render inputs, so the earlier products are "
        "not those of these inputs"
    )


def test_products_of_another_release_or_other_inputs_are_not_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_reconciler_seams(monkeypatch)
    release, environments = _release()
    _deploy(release)

    _deploy(release, artifact="9" * 64)
    assert RENDERING.INSTALLER_REUSE_NODE_SET_ENV not in environments[1], (
        "a different artifact renders a different template"
    )

    release.release_id = "rel-2"
    _deploy(release)
    assert RENDERING.INSTALLER_REUSE_NODE_SET_ENV not in environments[2], (
        "another release must provision and render for itself"
    )
    assert set(release.state[RENDERING.INSTALLER_PRODUCTS_STATE_KEY]) == {"gpu-a"}
    assert (
        release.state[RENDERING.INSTALLER_PRODUCTS_STATE_KEY]["gpu-a"]["release_id"]
        == "rel-2"
    ), "the record of the release that just ran replaces the older one"


@pytest.mark.parametrize(
    "products",
    [
        None,
        "not json",
        {"node_set_sha256": "short"},
        {**PRODUCTS, "template_config_map": "Bad_Name"},
    ],
    ids=["missing", "malformed", "bad-digest", "bad-name"],
)
def test_a_missing_or_malformed_product_report_records_nothing(
    monkeypatch: pytest.MonkeyPatch, products: Any
) -> None:
    """The report is advisory: without it the next run pays the full path."""

    _stub_reconciler_seams(monkeypatch)
    release, environments = _release(products=products)

    _deploy(release)
    _deploy(release)

    assert RENDERING.INSTALLER_PRODUCTS_STATE_KEY not in release.state
    assert RENDERING.INSTALLER_REUSE_NODE_SET_ENV not in environments[1]


def test_a_dry_run_neither_hints_nor_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_reconciler_seams(monkeypatch)
    release, environments = _release(dry_run=True)

    _deploy(release)

    assert RENDERING.INSTALLER_PRODUCTS_FILE_ENV not in environments[0]
    assert RENDERING.INSTALLER_PRODUCTS_STATE_KEY not in release.state


def test_the_product_inputs_digest_covers_no_credential() -> None:
    """The digest sits in the release state ConfigMap; nothing secret feeds it."""

    sensitive_looking = {
        name
        for name in RENDERING.INSTALLER_PRODUCT_INPUT_ENV
        if STATE.SENSITIVE_CONFIG_KEY.search(name)
    }
    assert sensitive_looking == {"GPU_FAULT_NODE_ACTION_KEYS_SECRET"}, (
        "the only input named like a credential is the *name* of the key Secret"
    )
    base = RENDERING.installer_product_inputs_digest(BASE_ENVIRONMENT)
    with_token = RENDERING.installer_product_inputs_digest(
        {**BASE_ENVIRONMENT, "GPU_FAULT_CLUSTER_TOKEN": "t0ken", "AWS_SECRET": "x"}
    )
    other_artifact = RENDERING.installer_product_inputs_digest(
        {**BASE_ENVIRONMENT, "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "9" * 64}
    )
    assert with_token == base, "a credential in the environment is not an input"
    assert other_artifact != base, "a render input is"


# --- the script, end to end ------------------------------------------------

FAKE_KUBECTL = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"
args=("$@")
positional=()
i=0
while ((i < ${#args[@]})); do
    case "${args[i]}" in
        --context|-n) i=$((i + 2)) ;;
        *) positional+=("${args[i]}"); i=$((i + 1)) ;;
    esac
done
verb="${positional[0]:-}"
wants_json() { [[ " ${positional[*]} " == *" -o "* ]]; }
case "${verb}" in
    get)
        case "${positional[1]}" in
            nodes)
                for node in ${FAKE_NODES}; do printf '%s\n' "${node}"; done ;;
            configmap)
                name="${positional[2]}"
                wants_json || exit 0
                case "${name}" in
                    gpu-fault-node-installer-0100)
                        printf '{"data":{"gpu-fault-node-installer-0.10.0.tar.gz":"x"}}\n' ;;
                    "${FAKE_WHEEL_CM}")
                        printf '{"binaryData":{"gpu_fault_cluster_executor-0.10.0-py3-none-any.whl.xz":"x"}}\n' ;;
                    gpu-fault-node-installer-template-*)
                        if [[ -f "${FAKE_TEMPLATE_DIR}/${name}" ]]; then
                            python3 -c 'import json, sys; print(json.dumps({"data": {"job.yaml": open(sys.argv[1]).read()}}))' "${FAKE_TEMPLATE_DIR}/${name}"
                        else
                            printf 'Error from server (NotFound): configmaps "%s" not found\n' "${name}" >&2
                            exit 1
                        fi ;;
                    *) printf '{}\n' ;;
                esac ;;
            secret)
                name="${positional[2]}"
                wants_json || exit 0
                case "${name}" in
                    gpu-fault-node-action-keys)
                        python3 -c '
import base64, json, os
nodes = os.environ["FAKE_NODES"].split()
print(json.dumps({"data": {n: base64.b64encode(("k" * 64).encode()).decode() for n in nodes}}))
' ;;
                    gpu-fault-regional-connection)
                        python3 -c '
import base64, json, os
values = {"ca.crt": "ca", "cluster-id": os.environ["FAKE_CLUSTER_ID"], "cluster-token": "tok", "control-plane-url": "https://cp.example"}
print(json.dumps({"data": {k: base64.b64encode(v.encode()).decode() for k, v in values.items()}}))
' ;;
                    *) printf '{}\n' ;;
                esac ;;
            jobs) printf '{"items":[]}\n' ;;
            *) printf '{}\n' ;;
        esac ;;
    create)
        name="${positional[2]}"
        source="${positional[3]#--from-file=job.yaml=}"
        cp "${source}" "${FAKE_TEMPLATE_DIR}/${name}.pending"
        printf 'apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: %s\n' "${name}" ;;
    apply)
        if [[ " ${positional[*]} " == *" --dry-run=server "* ]]; then exit 0; fi
        for pending in "${FAKE_TEMPLATE_DIR}"/*.pending; do
            [[ -e "${pending}" ]] && mv "${pending}" "${pending%.pending}"
        done
        printf 'applied\n' ;;
    *) exit 0 ;;
esac
"""

FAKE_RENDER = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_RENDER_LOG}"
node=""
render="false"
while (($#)); do
    case "$1" in
        --node) node="$2"; shift 2 ;;
        --render-only) render="true"; shift ;;
        *) shift ;;
    esac
done
if [[ "${render}" == "true" ]]; then
    printf 'apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: fake-install-%s\n' "${node}"
fi
"""

FAKE_PROVISION = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${GPU_FAULT_CLUSTER_ID}" >>"${FAKE_PROVISION_LOG}"
printf 'provisioned 2 node action key(s) in %s/%s\n' "${GPU_FAULT_NAMESPACE}" "${GPU_FAULT_NODE_ACTION_KEYS_SECRET}"
"""


class ScriptHarness:
    """The reconciler deploy script with fake kubectl and fake sibling scripts."""

    def __init__(self, tmp_path: Path) -> None:
        tree = tmp_path / "repo"
        node_dir = tree / "deploy" / "node"
        node_dir.mkdir(parents=True)
        (tree / "deploy" / "dataplane").mkdir()
        shutil.copy(SCRIPT, node_dir / SCRIPT.name)
        shutil.copy(MANIFEST, tree / "deploy" / "dataplane" / MANIFEST.name)
        self.script = node_dir / SCRIPT.name
        self._write(node_dir / "run-hyperpod-installer-job.sh", FAKE_RENDER)
        self._write(node_dir / "provision-node-action-keys.sh", FAKE_PROVISION)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self._write(bin_dir / "kubectl", FAKE_KUBECTL)
        self.template_dir = tmp_path / "templates"
        self.template_dir.mkdir()
        self.kubectl_log = tmp_path / "kubectl.log"
        self.render_log = tmp_path / "render.log"
        self.provision_log = tmp_path / "provision.log"
        self.products_file = tmp_path / "products.json"
        master = tmp_path / "fleet-master"
        master.write_text("master\n", encoding="utf-8")
        self.environment = {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_KUBECTL_LOG": str(self.kubectl_log),
            "FAKE_RENDER_LOG": str(self.render_log),
            "FAKE_PROVISION_LOG": str(self.provision_log),
            "FAKE_TEMPLATE_DIR": str(self.template_dir),
            "FAKE_NODES": "node-b node-a",
            "FAKE_CLUSTER_ID": "gpu-a",
            "FAKE_WHEEL_CM": "gpu-fault-executor-wheel-test",
            "GPU_FAULT_KUBECTL_CONTEXT": "gpu-a-context",
            "GPU_FAULT_CLUSTER_ID": "gpu-a",
            "GPU_FAULT_RUNTIME_PROFILE": "hyperpod-v1",
            "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "c" * 64,
            "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_INSTALLER_BUNDLE_SHA256": BUNDLE,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": TEMPLATE,
            "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": "d" * 64,
            "GPU_FAULT_WHEEL_CONFIG_MAP": "gpu-fault-executor-wheel-test",
            "GPU_FAULT_NODE_INSTALLER_IMAGE": "registry.example/installer@sha256:"
            + "1" * 64,
            "GPU_FAULT_FLEET_MASTER_FILE": str(master),
            "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
            "GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT": "false",
            "GPU_FAULT_RECONCILER_PRODUCTS_FILE": str(self.products_file),
        }

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def run(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        for log in (self.kubectl_log, self.render_log):
            log.write_text("", encoding="utf-8")
        self.products_file.unlink(missing_ok=True)
        return subprocess.run(
            [str(self.script)],
            env={**self.environment, **overrides},
            check=False,
            text=True,
            capture_output=True,
            timeout=120,
        )

    def products(self) -> dict[str, Any]:
        return json.loads(self.products_file.read_text(encoding="utf-8"))

    def provision_count(self) -> int:
        if not self.provision_log.exists():
            return 0
        return len(self.provision_log.read_text(encoding="utf-8").splitlines())

    def render_count(self) -> int:
        return sum(
            "--render-only" in line
            for line in self.render_log.read_text(encoding="utf-8").splitlines()
        )

    def kubectl_lines(self) -> list[str]:
        return self.kubectl_log.read_text(encoding="utf-8").splitlines()

    def hints(self) -> dict[str, str]:
        products = self.products()
        return {
            "GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256": products["node_set_sha256"],
            "GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP": products[
                "template_config_map"
            ],
        }


@pytest.fixture
def harness(tmp_path: Path) -> ScriptHarness:
    return ScriptHarness(tmp_path)


def test_the_first_mutating_run_provisions_renders_and_reports(
    harness: ScriptHarness,
) -> None:
    result = harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert harness.provision_count() == 1
    assert harness.render_count() == 1
    products = harness.products()
    rendered = (
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: fake-install-node-b\n"
    )
    content_sha = hashlib.sha256(rendered.encode()).hexdigest()
    assert products == {
        "node_set_sha256": NODE_SET,
        "node_action_keys_provisioned": True,
        "template_config_map": f"gpu-fault-node-installer-template-{content_sha[:12]}",
        "template_content_sha256": content_sha,
        "template_rendered": True,
    }, "the report names the node set, the template and what this run did"
    assert (harness.template_dir / products["template_config_map"]).exists(), (
        "the template ConfigMap was applied, not just rendered"
    )


def test_a_later_run_with_true_hints_reuses_keys_and_template(
    harness: ScriptHarness,
) -> None:
    first = harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    hints = harness.hints()

    second = harness.run(**hints)

    assert second.returncode == 0, second.stdout + second.stderr
    assert harness.provision_count() == 1, "keys were not provisioned a second time"
    assert harness.render_count() == 0, "the template was not rendered a second time"
    assert "reusing node action keys" in second.stdout
    assert "reusing template ConfigMap" in second.stdout
    products = harness.products()
    assert products["node_action_keys_provisioned"] is False
    assert products["template_rendered"] is False
    assert (
        products["template_config_map"]
        == hints["GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP"]
    )
    lines = harness.kubectl_lines()
    assert not any(" create configmap " in f" {line} " for line in lines), (
        "a reused template is not re-created"
    )
    assert any(
        "get secret gpu-fault-node-action-keys -o json" in line for line in lines
    ), "the node-scoped key verification still runs on the live Secret"
    assert any(" apply -f " in f" {line} " for line in lines), (
        "the Reconciler Deployment itself is still applied"
    )


def test_a_changed_node_set_defeats_both_hints(harness: ScriptHarness) -> None:
    first = harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    hints = harness.hints()

    second = harness.run(FAKE_NODES="node-a node-b node-c", **hints)

    assert second.returncode == 0, second.stdout + second.stderr
    assert harness.provision_count() == 2, "a new node needs a key"
    assert harness.render_count() == 1
    assert "reusing" not in second.stdout
    assert harness.products()["node_set_sha256"] != NODE_SET


def test_a_template_whose_content_no_longer_matches_its_name_is_rerendered(
    harness: ScriptHarness,
) -> None:
    """The name is content-addressed; an edited ConfigMap is not what it says."""

    first = harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    hints = harness.hints()
    stored = (
        harness.template_dir / hints["GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP"]
    )
    stored.write_text("kind: Job\nmetadata:\n  name: tampered\n", encoding="utf-8")

    second = harness.run(**hints)

    assert second.returncode == 0, second.stdout + second.stderr
    assert "reusing node action keys" in second.stdout, "the node set still matches"
    assert "reusing template ConfigMap" not in second.stdout
    assert harness.render_count() == 1
    products = harness.products()
    assert products["template_rendered"] is True
    assert (
        products["template_config_map"]
        == hints["GPU_FAULT_INSTALLER_REUSE_TEMPLATE_CONFIG_MAP"]
    ), "the render recreates the same content-addressed ConfigMap"
    assert stored.read_text(encoding="utf-8").startswith("apiVersion: batch/v1"), (
        "the re-applied ConfigMap carries the rendered content again"
    )


def test_the_preflight_never_reuses_and_never_writes(harness: ScriptHarness) -> None:
    first = harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    hints = harness.hints()
    for stored in harness.template_dir.iterdir():
        stored.unlink()
    harness.provision_log.unlink()

    result = harness.run(GPU_FAULT_RECONCILER_PREFLIGHT_ONLY="true", **hints)

    assert result.returncode == 0, result.stdout + result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict["status"] == "PASSED"
    assert verdict["node_count"] == 2
    assert "reusing" not in result.stdout, "a preflight proves the inputs itself"
    assert harness.provision_count() == 0
    assert not harness.products_file.exists(), "only a mutating run reports products"
    applies = [line for line in harness.kubectl_lines() if " apply " in f" {line} "]
    assert applies, "the preflight validates its manifests against the server"
    assert all("--dry-run=server" in line for line in applies), (
        "a preflight applies nothing for real"
    )
    assert not [
        path for path in harness.template_dir.iterdir() if path.suffix != ".pending"
    ], "no template ConfigMap is created by a preflight"


def test_a_malformed_hint_is_refused_before_anything_runs(
    harness: ScriptHarness,
) -> None:
    result = harness.run(GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256="not-a-digest")

    assert result.returncode == 2
    assert "invalid GPU_FAULT_INSTALLER_REUSE_NODE_SET_SHA256" in result.stderr
    assert harness.kubectl_lines() == [], "validation precedes the first kubectl call"
