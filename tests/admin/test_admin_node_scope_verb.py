"""``gpu-fault-admin node-scope``: show the recorded scope; converge a legacy cluster.

The world modelled: the site ``test-site`` manages the HyperPod cluster
``hp-gpu-a`` as every site did before the ownership convention (no recorded
selector, every node of the HyperPod cluster), and optionally a plain EKS
cluster recorded through the retired ``--gpu-nodegroup`` flag. The operator
labels the instance groups / node groups, and ``--converge`` records the label
selector only when the label-derived node set is the current node set.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import node_scope_command as verb
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.admin.site import load_site
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin.test_admin_site import site_file

SITE_ID = "test-site"
HP_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks-cluster"
HP_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
HP_LABEL_SELECTOR = f"{OWNERSHIP_LABEL}={SITE_ID},{HP_SELECTOR}"
EKS_LABEL_SELECTOR = f"{OWNERSHIP_LABEL}={SITE_ID}"
#: What a HyperPod cluster's connection Secret serves before any convergence.
LEGACY_HP_SECRET = {
    "node-selector": HP_SELECTOR,
    "node-selector-source": "provider-default-legacy",
}


def _node(name: str, labels: dict[str, str]) -> dict:
    return {"metadata": {"name": name, "labels": labels}}


class ScopeRunner:
    """``aws`` and ``kubectl`` answers for one HyperPod and one plain EKS cluster."""

    def __init__(
        self,
        *,
        hyperpod_groups: list[dict] | None = None,
        nodegroups: dict[str, dict] | None = None,
        nodes: dict[str, list[dict]] | None = None,
        secrets: dict[str, dict[str, str]] | None = None,
    ):
        self.hyperpod_groups = hyperpod_groups or []
        self.nodegroups = nodegroups or {}
        # Per kube context: every node of the cluster, with its labels.
        self.nodes = nodes or {}
        # Per kube context: the decoded selector keys the connection Secret
        # serves; a context without an entry has no Secret (not bootstrapped).
        self.secrets = secrets or {}
        self.commands: list[list[str]] = []

    def aws_json(self, region, service, operation, *arguments, **_options):
        self.commands.append(["aws", service, operation, *arguments])
        if (service, operation) == ("sagemaker", "describe-cluster"):
            return {"ClusterName": "hp-gpu-a", "InstanceGroups": self.hyperpod_groups}
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": sorted(self.nodegroups)}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            return {"nodegroup": {"nodegroupName": name, **self.nodegroups[name]}}
        raise AssertionError(f"unexpected aws call {service} {operation}")

    def run(self, arguments, **options):
        self.commands.append(list(arguments))
        assert arguments[0] == "kubectl" and "get" in arguments
        context = arguments[arguments.index("--context") + 1]
        if "secret" in arguments:
            assert options.get("sensitive") is True, "the Secret read never prints"
            if context not in self.secrets:
                return ""
            return json.dumps(
                {
                    "data": {
                        key: base64.b64encode(value.encode()).decode()
                        for key, value in self.secrets[context].items()
                    }
                }
            )
        assert "nodes" in arguments
        selector = arguments[arguments.index("-l") + 1]
        from gpu_fault.cluster_scope import selector_matches

        return json.dumps(
            {
                "items": [
                    item
                    for item in self.nodes.get(context, [])
                    if selector_matches(item["metadata"]["labels"], selector)
                ]
            }
        )


def _hp_node(name: str, group: str, labelled: bool, instance_type="ml.p5en.48xlarge"):
    labels = {
        "sagemaker.amazonaws.com/cluster-name": "hp-gpu-a",
        "sagemaker.amazonaws.com/instance-group-name": group,
        "node.kubernetes.io/instance-type": instance_type,
    }
    if labelled:
        labels[OWNERSHIP_LABEL] = SITE_ID
    return _node(name, labels)


def _hp_group(name: str, instance_type: str, labelled: bool) -> dict:
    group = {"InstanceGroupName": name, "InstanceType": instance_type}
    if labelled:
        group["KubernetesConfig"] = {"CurrentLabels": {OWNERSHIP_LABEL: SITE_ID}}
    return group


def _arguments(*extra: str, state_dir: Path):
    return admin_cli.parser().parse_args(
        ["node-scope", "--state-dir", str(state_dir), *extra]
    )


def test_node_scope_is_read_only_unless_it_converges_for_real() -> None:
    show = admin_cli.parser().parse_args(["node-scope", "--state-dir", "/tmp/s"])
    dry = admin_cli.parser().parse_args(
        [
            "node-scope",
            "--state-dir",
            "/tmp/s",
            "--gpu-cluster-arn",
            HP_ARN,
            "--converge",
            "--reference",
            "CHG-1",
            "--dry-run",
        ]
    )
    converge = admin_cli.parser().parse_args(
        [
            "node-scope",
            "--state-dir",
            "/tmp/s",
            "--gpu-cluster-arn",
            HP_ARN,
            "--converge",
            "--reference",
            "CHG-1",
        ]
    )
    readonly = admin_cli.READONLY_COMMANDS
    assert is_readonly_command(show, readonly_commands=readonly), (
        "node-scope without --converge reads only"
    )
    assert is_readonly_command(dry, readonly_commands=readonly), (
        "--converge --dry-run reports the decision and writes nothing"
    )
    assert not is_readonly_command(converge, readonly_commands=readonly), (
        "--converge rewrites the site entry: a mutation"
    )


def test_show_reports_a_legacy_hyperpod_cluster_ready_to_converge(
    tmp_path: Path, capsys
) -> None:
    runner = ScopeRunner(
        hyperpod_groups=[
            _hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True),
            _hp_group("accelerated-g6", "ml.g6.12xlarge", labelled=False),
        ],
        nodes={
            "gpu-a": [
                _hp_node("hyperpod-i-1", "spot-p5en", labelled=True),
                _hp_node("hyperpod-i-2", "spot-p5en", labelled=True),
            ]
        },
        secrets={"gpu-a": LEGACY_HP_SECRET},
    )
    site = load_site(site_file(tmp_path))

    code = verb.run_node_scope_command(
        _arguments(state_dir=tmp_path), site=site, runner=runner
    )

    assert code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["ownership_label"] == OWNERSHIP_LABEL
    (cluster,) = report["clusters"]
    assert cluster["cluster_id"] == "gpu-a" and cluster["provider"] == "hyperpod"
    assert cluster["selector"] == HP_SELECTOR, "the legacy cluster-name selector"
    assert cluster["source"] == "provider-default-legacy"
    assert cluster["label"]["selector"] == HP_LABEL_SELECTOR
    assert cluster["label"]["owned_groups"] == ["spot-p5en"]
    assert cluster["nodes"] == {
        "current": ["hyperpod-i-1", "hyperpod-i-2"],
        "label": ["hyperpod-i-1", "hyperpod-i-2"],
        "would_join": [],
        "would_leave": [],
        "owned_group_nodes_without_label": [],
    }
    assert cluster["convergence"] == "ready"
    assert cluster["data_plane"] == {
        "selector": HP_SELECTOR,
        "source": "provider-default-legacy",
        "state": "converged",
    }, "the Secret serves what the legacy entry records: the data plane agrees"
    assert "data plane converged" in captured.err
    secret_reads = [command for command in runner.commands if "secret" in command]
    assert len(secret_reads) == 1 and "--ignore-not-found" in secret_reads[0], (
        "one read of the connection Secret, tolerant of a cluster without one"
    )
    assert any("describe-cluster" in command for command in runner.commands), (
        "the instance groups are read from the provider API"
    )
    assert all("--converge" not in command for command in runner.commands), (
        "show runs no admin verb"
    )
    assert not (tmp_path / "node-scope").exists(), "show writes nothing"


def test_converge_records_the_label_selector_and_journals_it(
    tmp_path: Path, capsys
) -> None:
    runner = ScopeRunner(
        hyperpod_groups=[_hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True)],
        nodes={"gpu-a": [_hp_node("hyperpod-i-1", "spot-p5en", labelled=True)]},
        secrets={"gpu-a": LEGACY_HP_SECRET},
    )
    path = site_file(tmp_path)
    site = load_site(path)

    dry = verb.run_node_scope_command(
        _arguments(
            "--gpu-cluster-arn",
            HP_ARN,
            "--converge",
            "--reference",
            "CHG-7",
            "--dry-run",
            state_dir=tmp_path,
        ),
        site=site,
        runner=runner,
    )
    assert dry == 0
    dry_report = json.loads(capsys.readouterr().out)
    assert dry_report["status"] == "DRY_RUN"
    assert dry_report["would_record"] == {
        "nodeSelector": HP_LABEL_SELECTOR,
        "nodeSelectorSource": "label-convention",
    }
    assert "nodeSelector" not in yaml.safe_load(path.read_text())["spec"]["clusters"][0]

    code = verb.run_node_scope_command(
        _arguments(
            "--gpu-cluster-arn",
            HP_ARN,
            "--converge",
            "--reference",
            "CHG-7",
            state_dir=tmp_path,
        ),
        site=site,
        runner=runner,
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "CONVERGED"
    assert "deploy --state-dir" in report["next_step"], "the deploy rolls the selector"
    assert "DATA_PLANE_COMPATIBLE" in report["next_step"], (
        "and the verb says how the deploy classifies the change"
    )
    entry = yaml.safe_load(path.read_text())["spec"]["clusters"][0]
    assert entry["nodeSelector"] == HP_LABEL_SELECTOR
    assert entry["nodeSelectorSource"] == "label-convention"
    assert entry["hyperpodClusterName"] == "hp-gpu-a", "nothing else in the entry moves"
    journal = json.loads(
        (tmp_path / "node-scope" / "gpu-a" / "journal.json").read_text(encoding="utf-8")
    )
    (record,) = journal
    assert record["reference"] == "CHG-7"
    assert record["previous"] == {"selector": "", "source": "provider-default-legacy"}
    assert record["converged"] == {
        "selector": HP_LABEL_SELECTOR,
        "source": "label-convention",
    }
    assert record["nodes"] == ["hyperpod-i-1"] and record["owned_groups"] == [
        "spot-p5en"
    ]
    assert record["site_sha256_before"] != record["site_sha256_after"]
    backup = Path(record["site_backup"])
    assert backup.is_file() and backup.parent == tmp_path, (
        "the previous site.yaml is kept"
    )
    reloaded = load_site(path)
    (rendered,) = reloaded.release_config["clusters"]
    assert rendered["node_selector"] == HP_LABEL_SELECTOR
    assert rendered["node_selector_source"] == "label-convention"

    again = verb.run_node_scope_command(
        _arguments(
            "--gpu-cluster-arn",
            HP_ARN,
            "--converge",
            "--reference",
            "CHG-8",
            state_dir=tmp_path,
        ),
        site=reloaded,
        runner=runner,
    )
    assert again == 0
    again_report = json.loads(capsys.readouterr().out)
    assert again_report["status"] == "NOOP", "a converged cluster converges to itself"
    assert again_report["clusters"][0]["data_plane"] == {
        "selector": HP_SELECTOR,
        "source": "provider-default-legacy",
        "state": "pending deploy",
    }, "the Secret still serves the legacy selector until the deploy rolls it"


def test_converge_refuses_when_a_current_node_lacks_the_label(
    tmp_path: Path, capsys
) -> None:
    """HyperPod may apply KubernetesConfig labels to nodes created afterwards
    only; a current node of a labelled group without the label would leave the
    managed set silently, so convergence names it and stops."""

    runner = ScopeRunner(
        hyperpod_groups=[_hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True)],
        nodes={
            "gpu-a": [
                _hp_node("hyperpod-i-1", "spot-p5en", labelled=True),
                _hp_node("hyperpod-i-2", "spot-p5en", labelled=False),
            ]
        },
    )
    site = load_site(site_file(tmp_path))

    with pytest.raises(BootstrapError, match="cannot be converged") as refusal:
        verb.run_node_scope_command(
            _arguments(
                "--gpu-cluster-arn",
                HP_ARN,
                "--converge",
                "--reference",
                "CHG-9",
                state_dir=tmp_path,
            ),
            site=site,
            runner=runner,
        )

    message = str(refusal.value)
    assert "do not carry the label yet: hyperpod-i-2" in message
    assert "KubernetesConfig labels through its own reconciliation" in message
    report = json.loads(capsys.readouterr().out)
    (cluster,) = report["clusters"]
    assert cluster["nodes"]["owned_group_nodes_without_label"] == ["hyperpod-i-2"]
    assert cluster["convergence"].startswith("blocked: "), cluster["convergence"]
    assert (
        "nodeSelector"
        not in yaml.safe_load(site.source.read_text())["spec"]["clusters"][0]
    ), "a refused convergence writes nothing"
    assert not (tmp_path / "node-scope").exists(), (
        "no journal for a refused convergence"
    )


def test_converge_refuses_when_membership_would_change(tmp_path: Path, capsys) -> None:
    """A mixed legacy HyperPod cluster: the non-GPU instance group's node is
    managed today (cluster-name label) but not labelled; converging would drop
    it, so the step refuses and names the detach-and-rejoin path."""

    runner = ScopeRunner(
        hyperpod_groups=[
            _hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True),
            _hp_group("accelerated-g6", "ml.g6.12xlarge", labelled=False),
        ],
        nodes={
            "gpu-a": [
                _hp_node("hyperpod-i-1", "spot-p5en", labelled=True),
                _hp_node("hyperpod-i-9", "accelerated-g6", False, "ml.g6.12xlarge"),
            ]
        },
    )
    site = load_site(site_file(tmp_path))

    code = verb.run_node_scope_command(
        _arguments(state_dir=tmp_path), site=site, runner=runner
    )

    assert code == 0, "show reports the blocked convergence without refusing"
    (cluster,) = json.loads(capsys.readouterr().out)["clusters"]
    assert cluster["nodes"]["would_leave"] == ["hyperpod-i-9"]
    assert cluster["nodes"]["owned_group_nodes_without_label"] == []
    assert "would leave the managed set: hyperpod-i-9" in cluster["convergence"]
    assert "detach the cluster and re-join it" in cluster["convergence"]


def test_converge_reports_the_label_refusal_when_no_group_is_labelled(
    tmp_path: Path, capsys
) -> None:
    runner = ScopeRunner(
        hyperpod_groups=[_hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=False)],
        nodes={"gpu-a": [_hp_node("hyperpod-i-1", "spot-p5en", labelled=False)]},
    )
    site = load_site(site_file(tmp_path))

    with pytest.raises(BootstrapError, match="has no instance group labelled"):
        verb.run_node_scope_command(
            _arguments(
                "--gpu-cluster-arn",
                HP_ARN,
                "--converge",
                "--reference",
                "CHG-1",
                state_dir=tmp_path,
            ),
            site=site,
            runner=runner,
        )
    (cluster,) = json.loads(capsys.readouterr().out)["clusters"]
    assert cluster["label"]["selector"] is None
    assert "spot-p5en instanceTypes=ml.p5en.48xlarge" in cluster["label"]["refusal"]


def test_converge_handles_a_plain_eks_legacy_cluster(tmp_path: Path, capsys) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["clusters"].append(
        {
            **{k: v for k, v in document["spec"]["clusters"][0].items()},
            "clusterId": "gpu-eks-cluster",
            "context": "gpu-eks",
            "provider": "eks-managed",
            "nodeSelector": "eks.amazonaws.com/nodegroup=gpu-ng",
            "eksClusterArn": EKS_ARN,
        }
    )
    document["spec"]["clusters"][1].pop("hyperpodClusterName")
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    eks_labels = {"eks.amazonaws.com/nodegroup": "gpu-ng", OWNERSHIP_LABEL: SITE_ID}
    runner = ScopeRunner(
        nodegroups={
            "cpu-ng": {"instanceTypes": ["m7i-flex.xlarge"], "labels": {}},
            "gpu-ng": {
                "instanceTypes": ["p5en.48xlarge"],
                "labels": {OWNERSHIP_LABEL: SITE_ID},
            },
        },
        nodes={
            "gpu-eks": [
                _node("ip-10-0-38-180", eks_labels),
                _node("ip-10-0-42-179", eks_labels),
                _node("ip-10-0-30-189", {"eks.amazonaws.com/nodegroup": "cpu-ng"}),
            ]
        },
    )
    site = load_site(path)

    code = verb.run_node_scope_command(
        _arguments(
            "--gpu-cluster-arn",
            EKS_ARN,
            "--converge",
            "--reference",
            "CHG-2",
            state_dir=tmp_path,
        ),
        site=site,
        runner=runner,
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    (cluster,) = report["clusters"]
    assert cluster["source"] == "flag-legacy"
    assert cluster["nodes"]["current"] == ["ip-10-0-38-180", "ip-10-0-42-179"]
    assert report["status"] == "CONVERGED"
    entry = next(
        item
        for item in yaml.safe_load(path.read_text())["spec"]["clusters"]
        if item["clusterId"] == "gpu-eks-cluster"
    )
    assert entry["nodeSelector"] == EKS_LABEL_SELECTOR
    assert entry["nodeSelectorSource"] == "label-convention"
    assert all(
        "--context gpu-a" not in " ".join(command) for command in runner.commands
    ), "only the named cluster is assessed under --converge"


def test_converge_argument_rules(tmp_path: Path) -> None:
    site = load_site(site_file(tmp_path))
    with pytest.raises(BootstrapError, match="--converge needs --gpu-cluster-arn"):
        verb.run_node_scope_command(
            _arguments("--converge", "--reference", "x", state_dir=tmp_path),
            site=site,
            runner=ScopeRunner(),
        )
    with pytest.raises(BootstrapError, match="--converge needs --reference"):
        verb.run_node_scope_command(
            _arguments("--converge", "--gpu-cluster-arn", HP_ARN, state_dir=tmp_path),
            site=site,
            runner=ScopeRunner(),
        )
    with pytest.raises(BootstrapError, match="only used with --converge"):
        verb.run_node_scope_command(
            _arguments("--reference", "x", state_dir=tmp_path),
            site=site,
            runner=ScopeRunner(),
        )
    with pytest.raises(BootstrapError, match="is not a managed GPU cluster"):
        verb.run_node_scope_command(
            _arguments("--gpu-cluster-arn", EKS_ARN, state_dir=tmp_path),
            site=site,
            runner=ScopeRunner(),
        )


def test_show_reports_a_cluster_without_a_connection_secret_as_absent(
    tmp_path: Path, capsys
) -> None:
    runner = ScopeRunner(
        hyperpod_groups=[_hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True)],
        nodes={"gpu-a": [_hp_node("hyperpod-i-1", "spot-p5en", labelled=True)]},
    )
    code = verb.run_node_scope_command(
        _arguments(state_dir=tmp_path),
        site=load_site(site_file(tmp_path)),
        runner=runner,
    )
    assert code == 0
    (cluster,) = json.loads(capsys.readouterr().out)["clusters"]
    assert cluster["data_plane"] == {
        "selector": None,
        "source": None,
        "state": "absent",
    }, "a cluster the bootstrap has not reached yet has nothing to compare"


def test_show_records_each_eks_node_groups_update_strategy_and_advises(
    tmp_path: Path, capsys
) -> None:
    """A plain EKS cluster's node groups show their ``updateStrategy``; a GPU
    group the recorded selector covers that rolls with DEFAULT (surge) gets
    an advisory finding, the CPU group and a HyperPod cluster none."""

    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["clusters"].append(
        {
            **document["spec"]["clusters"][0],
            "clusterId": "gpu-eks-cluster",
            "context": "gpu-eks",
            "provider": "eks-managed",
            "nodeSelector": EKS_LABEL_SELECTOR,
            "nodeSelectorSource": "label-convention",
            "eksClusterArn": EKS_ARN,
        }
    )
    document["spec"]["clusters"][1].pop("hyperpodClusterName")
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    nodegroups = {
        "cpu-ng": {
            "instanceTypes": ["m7i-flex.xlarge"],
            "labels": {},
            "updateConfig": {"maxUnavailable": 1},
        },
        "gpu-ng": {
            "instanceTypes": ["p5en.48xlarge"],
            "labels": {OWNERSHIP_LABEL: SITE_ID},
            "updateConfig": {"maxUnavailable": 1},
        },
    }
    runner = ScopeRunner(
        hyperpod_groups=[_hp_group("spot-p5en", "ml.p5en.48xlarge", labelled=True)],
        nodegroups=nodegroups,
    )

    code = verb.run_node_scope_command(
        _arguments(state_dir=tmp_path), site=load_site(path), runner=runner
    )

    assert code == 0, "advice never fails the command"
    captured = capsys.readouterr()
    hyperpod, eks = json.loads(captured.out)["clusters"]
    advice = (
        "node group gpu-ng updates with DEFAULT (surge): a version update needs one "
        "more GPU instance; set updateStrategy=MINIMAL (administrator operations "
        "§5.0.1)"
    )
    assert eks["update_strategy"] == {
        "node_groups": {"cpu-ng": "DEFAULT", "gpu-ng": "DEFAULT"},
        "advisories": [advice],
    }, "every group is shown; only the selected GPU group is advised on"
    assert "update_strategy" not in hyperpod, "HyperPod has no update strategy"
    assert f"gpu-fault-admin node-scope: finding: gpu-eks-cluster: {advice}" in (
        captured.err
    )

    nodegroups["gpu-ng"]["updateConfig"]["updateStrategy"] = "MINIMAL"
    assert (
        verb.run_node_scope_command(
            _arguments(state_dir=tmp_path), site=load_site(path), runner=runner
        )
        == 0
    )
    captured = capsys.readouterr()
    eks = json.loads(captured.out)["clusters"][1]
    assert eks["update_strategy"]["advisories"] == []
    assert "updates with" not in captured.err, "a MINIMAL group is not advised on"
