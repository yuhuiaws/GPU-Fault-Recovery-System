"""A cluster's recorded node scope reaches the data plane and is verified there.

The world modelled is the one the live finding came from: a site whose two
GPU clusters were converged to the ownership label in ``site.yaml`` while their
connection Secrets still served the legacy selectors they were joined with
(``flag-legacy`` on the plain EKS cluster, ``provider-default-legacy`` on the
HyperPod one). The recorded scope is a release input, the served scope a
classification-time drift probe, the consumers of ``GPU_FAULT_NODE_SELECTOR``
carry the digest of the scope they were rendered for, and the verify refuses a
cluster whose Secret or consumers disagree with the site.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_gpu_rollout as GPU
from gpu_fault_release import regional_release_node_scope as SCOPE
from gpu_fault_release import regional_release_rollback_target as ROLLBACK
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
    diff_from_changed,
)

ROOT = Path(__file__).resolve().parents[2]
SITE_ID = "us-west-2-site-985fd370"
HP_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
HP_LABEL_SELECTOR = f"{OWNERSHIP_LABEL}={SITE_ID},{HP_SELECTOR}"
EKS_LEGACY_SELECTOR = "eks.amazonaws.com/nodegroup=gpu-p5en-efa-spot-node-group"
EKS_LABEL_SELECTOR = f"{OWNERSHIP_LABEL}={SITE_ID}"
LEGACY_HP = {"selector": HP_SELECTOR, "source": "provider-default-legacy"}
CONVERGED_HP = {"selector": HP_LABEL_SELECTOR, "source": "label-convention"}
LEGACY_EKS = {"selector": EKS_LEGACY_SELECTOR, "source": "flag-legacy"}
CONVERGED_EKS = {"selector": EKS_LABEL_SELECTOR, "source": "label-convention"}
#: The keys a legacy HyperPod Secret records (``_data`` spells them with dashes).
LEGACY_HP_KEYS = {
    "node_selector": HP_SELECTOR,
    "node_selector_source": LEGACY_HP["source"],
}


def _target(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = dict(
        cluster_id="gpu-a",
        context="gpu-a-context",
        provider="hyperpod",
        hyperpod_cluster_name="hp-gpu-a",
        node_selector="",
        node_selector_source="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _eks_target(**overrides: Any) -> SimpleNamespace:
    return _target(
        cluster_id="gpu-eks-cluster",
        context="gpu-eks-context",
        provider="eks-managed",
        hyperpod_cluster_name="",
        node_selector=EKS_LEGACY_SELECTOR,
        **overrides,
    )


def _converged(target: SimpleNamespace, selector: str) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            **vars(target),
            "node_selector": selector,
            "node_selector_source": "label-convention",
        }
    )


def _data(**values: str) -> dict[str, str]:
    """A Secret's ``data`` as kubectl returns it, for the keys given."""

    return {
        key.replace("_", "-"): base64.b64encode(value.encode()).decode()
        for key, value in values.items()
    }


# --- recorded and served scope -----------------------------------------------------


def test_recorded_scope_follows_the_site_entry() -> None:
    assert SCOPE.recorded_node_scope(_target()) == LEGACY_HP
    assert SCOPE.recorded_node_scope(_converged(_target(), HP_LABEL_SELECTOR)) == (
        CONVERGED_HP
    )
    assert SCOPE.recorded_node_scope(_eks_target()) == LEGACY_EKS
    assert SCOPE.recorded_node_scope(_converged(_eks_target(), EKS_LABEL_SELECTOR)) == (
        CONVERGED_EKS
    )
    assert SCOPE.recorded_node_scope(SimpleNamespace(cluster_id="bare")) == {
        "selector": "",
        "source": "provider-default-legacy",
    }, "a partial double without a resolvable selector records an empty one"


def test_served_scope_reads_the_secret_and_defaults_as_the_data_plane_does() -> None:
    target = _target()
    assert SCOPE.served_node_scope(_data(**LEGACY_HP_KEYS), target) == LEGACY_HP
    # A Secret rendered before either key existed: the HyperPod label with the
    # legacy source, which a legacy cluster agrees with.
    served = SCOPE.served_node_scope({}, target)
    assert served == LEGACY_HP
    assert not SCOPE.scopes_disagree(LEGACY_HP, served, {}), (
        "a legacy entry agrees with a Secret that predates the keys"
    )
    # The selector key alone (rendered between the two keys appearing) derives
    # the source the way the data plane reads the selector.
    assert (
        SCOPE.served_node_scope(_data(node_selector=HP_SELECTOR), target) == LEGACY_HP
    )
    assert SCOPE.served_node_scope(_data(node_selector=HP_LABEL_SELECTOR), target) == (
        CONVERGED_HP
    )
    assert SCOPE.served_node_scope(_data(node_selector="node-type=gpu"), target) == {
        "selector": "node-type=gpu",
        "source": "flag-legacy",
    }
    # A plain EKS cluster has no default: a Secret without the key serves an
    # empty selector, which never agrees.
    eks = SCOPE.served_node_scope({}, _eks_target())
    assert eks["selector"] == ""
    assert SCOPE.scopes_disagree(LEGACY_EKS, eks, {}), (
        "an eks-managed Secret without the key never agrees"
    )


def test_the_live_finding_is_a_disagreement_on_both_clusters() -> None:
    """Both Secrets still served the join-time selector after the entries were
    converged; the source key is recorded and disagrees too."""

    eks_data = _data(
        **{"node_selector": EKS_LEGACY_SELECTOR, "node_selector_source": "flag-legacy"}
    )
    eks_served = SCOPE.served_node_scope(
        eks_data, _converged(_eks_target(), EKS_LABEL_SELECTOR)
    )
    assert eks_served == LEGACY_EKS
    assert SCOPE.scopes_disagree(CONVERGED_EKS, eks_served, eks_data), (
        "the plain EKS Secret still serves the node-group selector"
    )
    hp_data = _data(**LEGACY_HP_KEYS)
    hp_served = SCOPE.served_node_scope(
        hp_data, _converged(_target(), HP_LABEL_SELECTOR)
    )
    assert hp_served == LEGACY_HP
    assert SCOPE.scopes_disagree(CONVERGED_HP, hp_served, hp_data), (
        "the HyperPod Secret still serves the cluster-name selector"
    )


def test_the_source_is_judged_only_when_the_secret_records_it() -> None:
    converged = _converged(_target(), HP_LABEL_SELECTOR)
    without_source = _data(node_selector=HP_LABEL_SELECTOR)
    served = SCOPE.served_node_scope(without_source, converged)
    assert not SCOPE.scopes_disagree(CONVERGED_HP, served, without_source), (
        "a Secret without the audit key is judged on the selector it serves"
    )
    wrong_source = _data(
        node_selector=HP_LABEL_SELECTOR, node_selector_source="flag-legacy"
    )
    assert SCOPE.scopes_disagree(
        CONVERGED_HP, SCOPE.served_node_scope(wrong_source, converged), wrong_source
    ), "a recorded source that disagrees is a disagreement"


def test_digests_name_the_scope_and_the_site_wide_record() -> None:
    assert SCOPE.node_scope_digest(LEGACY_HP) == SCOPE.node_scope_digest(
        dict(LEGACY_HP)
    )
    assert SCOPE.node_scope_digest(LEGACY_HP) != SCOPE.node_scope_digest(CONVERGED_HP)
    legacy = SimpleNamespace(clusters=[_eks_target(), _target()])
    converged = SimpleNamespace(
        clusters=[_target(), _converged(_eks_target(), EKS_LABEL_SELECTOR)]
    )
    assert SCOPE.cluster_node_scopes(legacy) == {
        "gpu-a": LEGACY_HP,
        "gpu-eks-cluster": LEGACY_EKS,
    }, "recorded per cluster, in cluster-id order"
    assert SCOPE.cluster_node_scopes_digest(legacy) == SCOPE.cluster_node_scopes_digest(
        SimpleNamespace(clusters=list(reversed(legacy.clusters)))
    ), "the digest does not depend on the declaration order"
    assert SCOPE.cluster_node_scopes_digest(legacy) != (
        SCOPE.cluster_node_scopes_digest(converged)
    ), "one converged entry changes the release input"
    assert SCOPE.cluster_node_scopes(SimpleNamespace()) == {}, "no clusters, no scopes"


def test_the_consumers_are_the_manifests_that_read_the_selector_key() -> None:
    readers: set[str] = set()
    for path in sorted((ROOT / "deploy/dataplane").glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if not isinstance(document, dict) or document.get("kind") != "Deployment":
                continue
            containers = document["spec"]["template"]["spec"]["containers"]
            for container in containers:
                for item in container.get("env") or []:
                    reference = (item.get("valueFrom") or {}).get("secretKeyRef") or {}
                    if reference.get("key") == "node-selector":
                        readers.add(document["metadata"]["name"])
    assert readers == set(SCOPE.NODE_SCOPE_CONSUMERS), (
        "every Deployment reading node-selector from the Secret is a consumer, "
        "and only those"
    )
    assert inventory.GPU_COLLECTOR_DEPLOYMENT not in SCOPE.NODE_SCOPE_CONSUMERS


# --- drift probe -------------------------------------------------------------------


class ProbedRelease:
    def __init__(self, targets: list[SimpleNamespace], secrets: dict[str, Any]) -> None:
        self.runner = SimpleNamespace(dry_run=False)
        self.config = SimpleNamespace(
            clusters=targets,
            namespace="gpu-fault-system",
            delivery_component_digests={"cpu": "a" * 64},
        )
        self.secrets = secrets
        self.reads: list[list[str]] = []

    def _gpu(self, target: Any, *arguments: str) -> list[str]:
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, command: list[str]) -> dict[str, Any]:
        self.reads.append(command)
        assert "secret" in command and "--ignore-not-found" in command
        value = self.secrets.get(command[command.index("--context") + 1])
        return {"data": value} if value is not None else {}


def test_the_probe_names_the_clusters_whose_secret_serves_another_scope() -> None:
    hp = _converged(_target(), HP_LABEL_SELECTOR)
    eks = _converged(_eks_target(), EKS_LABEL_SELECTOR)
    agreeing = _converged(
        _target(cluster_id="gpu-c", context="gpu-c-context"), HP_LABEL_SELECTOR
    )
    not_bootstrapped = _target(cluster_id="gpu-d", context="gpu-d-context")
    release = ProbedRelease(
        [hp, eks, agreeing, not_bootstrapped],
        {
            "gpu-a-context": _data(**LEGACY_HP_KEYS),
            "gpu-eks-context": _data(
                node_selector=EKS_LEGACY_SELECTOR, node_selector_source="flag-legacy"
            ),
            "gpu-c-context": _data(
                node_selector=HP_LABEL_SELECTOR, node_selector_source="label-convention"
            ),
        },
    )

    drift = SCOPE.node_scope_drift(release)

    assert drift == {
        "gpu-a": {"recorded": CONVERGED_HP, "served": LEGACY_HP},
        "gpu-eks-cluster": {"recorded": CONVERGED_EKS, "served": LEGACY_EKS},
    }, "the two converged clusters still serve what they were joined with"
    assert len(release.reads) == 4, "one Secret read per cluster, no other read"


@pytest.mark.parametrize("gate", ["dry-run", "no-control-plane", "bare-double"])
def test_the_probe_never_reads_a_cluster_it_must_not(gate: str) -> None:
    release: Any = ProbedRelease(
        [_target()], {"gpu-a-context": _data(**LEGACY_HP_KEYS)}
    )
    if gate == "dry-run":
        release.runner.dry_run = True
    elif gate == "no-control-plane":
        release.config.delivery_component_digests = {}
    else:
        release = SimpleNamespace(config=release.config, runner=release.runner)
    assert SCOPE.node_scope_drift(release) == {}
    if gate != "bare-double":
        assert release.reads == []


# --- stamps ------------------------------------------------------------------------


def test_the_stamp_is_the_scope_digest_on_the_pod_template() -> None:
    document: dict[str, Any] = {"kind": "Deployment", "metadata": {"name": "x"}}
    SCOPE.stamp_node_scope_annotation(document, CONVERGED_HP)
    annotations = document["spec"]["template"]["metadata"]["annotations"]
    assert annotations == {
        SCOPE.NODE_SCOPE_ANNOTATION: SCOPE.node_scope_digest(CONVERGED_HP)
    }
    assert SCOPE.consumer_node_scope_annotation(document) == (
        SCOPE.node_scope_digest(CONVERGED_HP)
    )
    assert SCOPE.consumer_node_scope_annotation({"spec": {"template": {}}}) is None, (
        "a Deployment rendered before the stamp existed carries none"
    )


def test_the_reconciler_stamp_is_a_merge_patch_of_the_pod_template() -> None:
    calls: list[list[str]] = []
    release = SimpleNamespace(
        runner=SimpleNamespace(
            run=lambda arguments, **_k: calls.append(list(arguments))
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda target, *arguments: [
            "kubectl",
            "--context",
            target.context,
            *arguments,
        ],
    )

    SCOPE.stamp_reconciler_node_scope(release, _target(), CONVERGED_HP)

    (command,) = calls
    assert command[:9] == [
        "kubectl",
        "--context",
        "gpu-a-context",
        "-n",
        "gpu-fault-system",
        "patch",
        "deployment",
        inventory.GPU_RECONCILER_DEPLOYMENT,
        "--type",
    ]
    assert command[9] == "merge"
    patch = json.loads(command[command.index("-p") + 1])
    assert patch == {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        SCOPE.NODE_SCOPE_ANNOTATION: SCOPE.node_scope_digest(
                            CONVERGED_HP
                        )
                    }
                }
            }
        }
    }


def test_deploy_reconciler_stamps_the_scope_after_the_script_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def run(
        arguments: list[str], *, env: dict[str, str] | None = None, **_k: Any
    ) -> str:
        calls.append(list(arguments))
        if env is not None:
            environments.append(dict(env))
        return ""

    def environment(_release: Any, _target: Any, **keywords: Any) -> dict[str, str]:
        return {
            FLEET.INSTALLER_BUNDLE_ENV: "b" * 64,
            FLEET.INSTALLER_TEMPLATE_ENV: "e" * 64,
            "GPU_FAULT_NODE_SELECTOR": (
                keywords["node_scope"]["selector"]
                if keywords.get("node_scope")
                else HP_SELECTOR
            ),
        }

    monkeypatch.setattr(FLEET, "build_reconciler_environment", environment)
    monkeypatch.setattr(FLEET, "wait_deployment_rollout", lambda *_a, **_k: None)
    monkeypatch.setattr(
        FLEET, "reconciler_container_env", lambda *_a, **_k: environments[-1]
    )
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False, run=run),
        state={},
        release_id="release-a",
        config=SimpleNamespace(upgrade_max_unavailable=1, namespace="gpu-fault-system"),
        _gpu=lambda target, *arguments: [
            "kubectl",
            "--context",
            target.context,
            *arguments,
        ],
        _settle_installer_jobs=lambda _target: None,
    )
    target = _converged(_target(fleet_master_file=None), HP_LABEL_SELECTOR)

    FLEET.deploy_reconciler(
        release,
        target,
        wheel_cm="w",
        bundle_cm="b",
        artifact_sha="a" * 64,
        config_digest="c" * 64,
    )
    restored = {"selector": HP_SELECTOR, "source": "provider-default-legacy"}
    FLEET.deploy_reconciler(
        release,
        target,
        wheel_cm="w",
        bundle_cm="b",
        artifact_sha="a" * 64,
        config_digest="c" * 64,
        node_scope=restored,
    )

    script, stamp, rollback_script, rollback_stamp = calls
    assert Path(script[0]).name == "deploy-node-installer-reconciler.sh"
    assert stamp[5:8] == ["patch", "deployment", inventory.GPU_RECONCILER_DEPLOYMENT]
    assert SCOPE.node_scope_digest(CONVERGED_HP) in stamp[-1], (
        "the site's recorded scope is stamped after the script's apply"
    )
    assert Path(rollback_script[0]).name == "deploy-node-installer-reconciler.sh"
    assert SCOPE.node_scope_digest(restored) in rollback_stamp[-1], (
        "a rollback stamps the scope the restored Secret serves"
    )
    assert environments[1]["GPU_FAULT_NODE_SELECTOR"] == HP_SELECTOR, (
        "and enumerates nodes on it"
    )


# --- the CONNECTION component ------------------------------------------------------


def test_a_connection_change_re_renders_the_secret_without_the_endpoint_gate() -> None:
    events: list[str] = []
    progress: list[tuple[Any, ...]] = []
    release = SimpleNamespace(
        _ensure_connection_secret=lambda _target: events.append("connect"),
        _verify_gpu_control_plane_endpoint=lambda _target: events.append("gate"),
    )
    target = SimpleNamespace(cluster_id="gpu-a")

    GPU.upgrade_gpu_target(
        release,
        target,
        diff_from_changed({SCOPE.RECORDED_SCOPES_INPUT}),
        ReleaseExecutionPlan((ReleaseComponent.CONNECTION,)),
        progress=lambda *args: progress.append(args),
    )
    assert events == ["connect"], "the Secret is re-rendered; no DNS/TLS probe Pod"
    assert progress == [
        ((ReleaseComponent.CONNECTION,), "STARTED", None),
        ((ReleaseComponent.CONNECTION,), "COMPLETED", None),
    ]

    events.clear()
    progress.clear()
    GPU.upgrade_gpu_target(
        release,
        target,
        diff_from_changed({"endpoint", SCOPE.RECORDED_SCOPES_INPUT}),
        ReleaseExecutionPlan((ReleaseComponent.ENDPOINT, ReleaseComponent.CONNECTION)),
        progress=lambda *args: progress.append(args),
    )
    assert events == ["connect", "gate"], "the gate step covers both; one render"
    assert progress == [
        ((ReleaseComponent.ENDPOINT, ReleaseComponent.CONNECTION), "STARTED", None),
        ((ReleaseComponent.ENDPOINT, ReleaseComponent.CONNECTION), "COMPLETED", None),
    ]


def _rollback(
    release: Any, target: Any, components: frozenset[ReleaseComponent]
) -> None:
    ROLLBACK.rollback_target(
        release,
        target,
        previous={
            "clusters": {"gpu-a": {"wheel": "previous-wheel", "wheel_key": "e.whl"}},
            "secret_backups": {
                "clusters": {
                    "gpu-a": {
                        "source": SCOPE.CONNECTION_SECRET,
                        "backup": f"{SCOPE.CONNECTION_SECRET}-rollback-release-a",
                    }
                }
            },
        },
        artifact="1" * 64,
        config_digest="2" * 64,
        runtime_profile_version="previous-profile",
        executor_artifact="3" * 64,
        executor_compatibility="4" * 64,
        node_compatibility="5" * 64,
        runtime_image="registry.example/executor@sha256:" + "6" * 64,
        node_installer_image="registry.example/installer@sha256:" + "7" * 64,
        components=components,
    )


def test_a_rollback_re_stamps_the_consumers_for_the_scope_it_restores() -> None:
    restored: list[tuple[str, str]] = []
    applied: list[dict[str, Any]] = []
    release: Any = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system", executor_wheel=Path("e.whl")
        ),
        _gpu=lambda _target, *arguments: list(arguments),
        _restore_secret=lambda _kubectl, *, source, backup: (
            restored.append((source, backup)) or _data(**LEGACY_HP_KEYS)
        ),
        _apply_gpu_deployments=lambda _target, _wheel, **kwargs: applied.append(kwargs),
        _verify_gpu_control_plane_endpoint=lambda _target: None,
    )
    target = _converged(_target(), HP_LABEL_SELECTOR)

    _rollback(
        release,
        target,
        frozenset({ReleaseComponent.CONNECTION, ReleaseComponent.EXECUTOR}),
    )
    assert restored == [
        (SCOPE.CONNECTION_SECRET, f"{SCOPE.CONNECTION_SECRET}-rollback-release-a")
    ], "a CONNECTION compensation restores the Secret backup, like ENDPOINT"
    assert applied[-1]["node_scope"] == LEGACY_HP, (
        "the executor is re-rendered for the scope the restored Secret serves, "
        "so its Pods restart and read it"
    )

    _rollback(release, target, frozenset({ReleaseComponent.EXECUTOR}))
    assert len(restored) == 1, "an executor-only compensation leaves the Secret"
    assert applied[-1]["node_scope"] is None, (
        "and renders the consumers for the site's scope, which the Secret serves"
    )


def test_a_rollback_refuses_without_the_secret_backup() -> None:
    release: Any = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *arguments: list(arguments),
    )
    with pytest.raises(ReleaseError, match="connection Secret backup is missing"):
        ROLLBACK.rollback_target(
            release,
            _target(),
            previous={"clusters": {"gpu-a": {}}},
            artifact="1" * 64,
            config_digest="2" * 64,
            runtime_profile_version="p",
            executor_artifact="3" * 64,
            executor_compatibility="4" * 64,
            node_compatibility="5" * 64,
            runtime_image="r@sha256:" + "6" * 64,
            node_installer_image="i@sha256:" + "7" * 64,
            components=frozenset({ReleaseComponent.CONNECTION}),
        )


# --- verify ------------------------------------------------------------------------


class VerifiedRelease:
    def __init__(
        self,
        *,
        secret: dict[str, str] | None,
        stamps: dict[str, str | None] | None = None,
    ) -> None:
        self.config = SimpleNamespace(namespace="gpu-fault-system")
        self.secret = secret
        self.stamps = stamps or {}
        self.reads: list[list[str]] = []

    def _gpu(self, target: Any, *arguments: str) -> list[str]:
        return ["kubectl", "--context", target.context, *arguments]

    def _get_json(self, command: list[str]) -> dict[str, Any]:
        self.reads.append(command)
        if "secret" in command:
            return {"data": self.secret} if self.secret is not None else {}
        name = command[command.index("deployment") + 1]
        stamp = self.stamps.get(name)
        template: dict[str, Any] = {"spec": {"containers": []}}
        if stamp:
            template["metadata"] = {"annotations": {SCOPE.NODE_SCOPE_ANNOTATION: stamp}}
        return {"kind": "Deployment", "spec": {"template": template}}


def test_verify_reports_an_agreeing_cluster_with_what_it_serves_and_runs() -> None:
    target = _converged(_target(), HP_LABEL_SELECTOR)
    digest = SCOPE.node_scope_digest(CONVERGED_HP)
    release = VerifiedRelease(
        secret=_data(
            node_selector=HP_LABEL_SELECTOR, node_selector_source="label-convention"
        ),
        stamps={
            inventory.GPU_EXECUTOR_DEPLOYMENT: digest,
            inventory.GPU_RECONCILER_DEPLOYMENT: digest,
        },
    )

    details = SCOPE.check_node_scope_agreement(release, target, recorded=CONVERGED_HP)

    assert details == {
        "connection_secret": {**CONVERGED_HP, "selector_recorded": True},
        "consumers": {
            inventory.GPU_EXECUTOR_DEPLOYMENT: digest,
            inventory.GPU_WATCHER_DEPLOYMENT: None,
            inventory.GPU_RECONCILER_DEPLOYMENT: digest,
        },
    }, "the watcher predates the stamp and is reported, not judged"
    assert [command[command.index("get") + 1] for command in release.reads] == [
        "secret",
        "deployment",
        "deployment",
        "deployment",
    ]


@pytest.mark.parametrize(
    ("target", "recorded", "secret", "served"),
    [
        (
            _converged(_eks_target(), EKS_LABEL_SELECTOR),
            CONVERGED_EKS,
            _data(
                node_selector=EKS_LEGACY_SELECTOR, node_selector_source="flag-legacy"
            ),
            LEGACY_EKS,
        ),
        (
            _converged(_target(), HP_LABEL_SELECTOR),
            CONVERGED_HP,
            _data(
                **{
                    "node_selector": HP_SELECTOR,
                    "node_selector_source": "provider-default-legacy",
                }
            ),
            LEGACY_HP,
        ),
    ],
    ids=["plain-eks", "hyperpod"],
)
def test_verify_refuses_a_secret_that_still_serves_the_legacy_scope(
    target: SimpleNamespace,
    recorded: dict[str, str],
    secret: dict[str, str],
    served: dict[str, str],
) -> None:
    release = VerifiedRelease(secret=secret)
    with pytest.raises(ReleaseError) as refusal:
        SCOPE.check_node_scope_agreement(release, target, recorded=recorded)
    message = str(refusal.value)
    assert message.startswith(f"{target.cluster_id} node scope disagreement"), message
    for value in (
        recorded["selector"],
        recorded["source"],
        served["selector"],
        served["source"],
    ):
        assert value in message, f"the refusal names {value!r}"
    assert "gpu-fault-admin deploy" in message, "and the remedy"
    assert len(release.reads) == 1, "refused on the Secret; the consumers are not read"


def test_verify_refuses_a_consumer_rendered_for_another_scope() -> None:
    target = _converged(_target(), HP_LABEL_SELECTOR)
    release = VerifiedRelease(
        secret=_data(
            node_selector=HP_LABEL_SELECTOR, node_selector_source="label-convention"
        ),
        stamps={inventory.GPU_WATCHER_DEPLOYMENT: SCOPE.node_scope_digest(LEGACY_HP)},
    )
    with pytest.raises(
        ReleaseError, match=f"{inventory.GPU_WATCHER_DEPLOYMENT} Pods were rendered"
    ):
        SCOPE.check_node_scope_agreement(release, target, recorded=CONVERGED_HP)


def test_verify_accepts_a_secret_predating_the_keys_on_a_legacy_cluster() -> None:
    release = VerifiedRelease(secret={})
    details = SCOPE.check_node_scope_agreement(release, _target(), recorded=LEGACY_HP)
    assert details["connection_secret"] == {**LEGACY_HP, "selector_recorded": False}, (
        "the defaults the data plane runs are reported, and agree"
    )
