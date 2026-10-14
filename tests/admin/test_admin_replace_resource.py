"""``replace-resource``: the audited transition for a really replaced resource.

The registry refuses every identity change, so a resource that really was
replaced left the deploy sync failing for good (batch 79's open point): a
re-subscribed SNS e-mail keeps its key with a new subscription ARN, a recreated
public subnet changes the NLB row's dependencies. These tests run the verb end
to end: the real deploy rebuild, the real in-Pod script against the real API
and store (``PodRegistry``), and the cleaner's own AWS probes against fakes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin import resource_replacement as replacement
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResourceStatus,
    retired_resource_key,
)
from tests.admin._aws_cleanup_support import ACCOUNT, TAGS, Aws, absent
from tests.admin._registry_pod_support import PodRegistry
from tests.admin.test_admin_site import site_file

TOPIC = f"arn:aws:sns:us-east-1:{ACCOUNT}:gpu-fault"
OLD_SUBSCRIPTION = f"{TOPIC}:0f1e2d3c-old"
NEW_SUBSCRIPTION = f"{TOPIC}:4b5a6978-new"
ENDPOINT = "ops@example.com"


def _monitoring(subscription: str) -> dict[str, Any]:
    return {
        "sns_topic_arn": TOPIC,
        "sns_topic_ownership": "CREATED",
        "email_subscription_arn": subscription,
        "email_subscription_endpoint": ENDPOINT,
        "email_subscription_status": "Confirmed",
    }


def _network(*subnets: str) -> dict[str, Any]:
    return {
        "subnet_resources": [
            {
                "subnet_id": f"subnet-{name}",
                "availability_zone": f"us-east-1{name}",
                "ownership": "CREATED",
                "vpc_id": "vpc-1",
                "route_table_id": f"rtb-{name}",
                "route_table_association_id": f"rtbassoc-{name}",
            }
            for name in subnets
        ],
        "security_group": "sg-123",
        "security_group_ownership": "CREATED",
        "vpc_id": "vpc-1",
    }


class Site:
    """The test site: its registry behind the real API, AWS behind fakes."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = load_site(site_file(tmp_path))
        self.pod = PodRegistry(monkeypatch)
        self.resources: dict[str, Any] = {
            "monitoring_resources": _monitoring(OLD_SUBSCRIPTION),
            "nlb_network": _network("a", "b"),
        }
        self.live: set[str] = set()
        self.untagged: set[str] = set()
        monkeypatch.setattr(
            registry,
            "find_bootstrap_state",
            lambda _site: {"site_id": "test-site", "resources": self.resources},
        )
        monkeypatch.setattr(registry, "discover_runtime_resources", lambda _site: {})
        self.aws = Aws(
            {
                ("sns", "get-subscription-attributes"): self._subscription,
                ("ec2", "describe-subnets"): self._subnets,
                ("ec2", "describe-route-tables"): self._route_tables,
            }
        )
        monkeypatch.setattr(aws_commands, "bounded_command", self.aws)
        registry.sync_installation_resource_registry(self.site)

    @property
    def state_dir(self) -> Path:
        return self.site.source.parent

    def rows(self) -> dict[str, Any]:
        return {
            row.resource_key: row
            for row in self.pod.store.list_installation_resources("test-site")
        }

    def _tags(self, identifier: str) -> list[dict[str, str]]:
        return [] if identifier in self.untagged else TAGS

    def _subscription(self, arguments: list[str]) -> Any:
        arn = arguments[arguments.index("--subscription-arn") + 1]
        if arn not in self.live:
            return absent("NotFound")
        return {
            "Attributes": {
                "SubscriptionArn": arn,
                "TopicArn": TOPIC,
                "Owner": ACCOUNT,
                "Endpoint": ENDPOINT,
                "Protocol": "email",
            }
        }

    def _subnets(self, arguments: list[str]) -> Any:
        subnet = arguments[arguments.index("--subnet-ids") + 1]
        if subnet not in self.live:
            return absent("InvalidSubnetID.NotFound")
        return {
            "Subnets": [
                {"SubnetId": subnet, "OwnerId": ACCOUNT, "Tags": self._tags(subnet)}
            ]
        }

    def _route_tables(self, arguments: list[str]) -> Any:
        if "--route-table-ids" in arguments:
            table = arguments[arguments.index("--route-table-ids") + 1]
            if table not in self.live:
                return absent("InvalidRouteTableID.NotFound")
            return {"RouteTables": [self._table(table)]}
        association = arguments[arguments.index("--filters") + 1].rsplit("=", 1)[1]
        if association not in self.live:
            return {"RouteTables": []}
        return {"RouteTables": [self._table(association.replace("rtbassoc", "rtb"))]}

    def _table(self, table: str) -> dict[str, Any]:
        association = table.replace("rtb", "rtbassoc")
        return {
            "RouteTableId": table,
            "OwnerId": ACCOUNT,
            "Tags": self._tags(table),
            "Associations": [{"RouteTableAssociationId": association, "Main": False}],
        }

    def run(self, *keys: str, confirm: str | None = None) -> int:
        arguments = argparse.Namespace(
            resource_key=list(keys), reference="CHG-replace-1", confirm=confirm
        )
        return replacement.run_replace_resource_command(arguments, site=self.site)


def _email_key() -> str:
    digest = hashlib.sha256(ENDPOINT.encode()).hexdigest()[:16]
    return f"aws/sns/email-subscription/{digest}"


def _printed(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return dict(json.loads(capsys.readouterr().out))


def test_a_resubscribed_email_moves_its_key_to_the_new_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    key = _email_key()
    recorded = harness.rows()[key]
    harness.resources["monitoring_resources"] = _monitoring(NEW_SUBSCRIPTION)
    harness.live = {NEW_SUBSCRIPTION}
    with pytest.raises(BootstrapError) as refused:
        registry.sync_installation_resource_registry(harness.site)
    assert f"{key} (resource_id, resource_arn)" in str(refused.value)
    assert "replace-resource" in str(refused.value), "the refusal names the remedy"

    assert harness.run(key) == 0
    plan = _printed(capsys)
    assert plan["refusals"] == []
    assert plan["transitions"] == [
        {
            "resource_key": key,
            "resource_type": "sns_subscription",
            "recorded": {
                "resource_id": OLD_SUBSCRIPTION,
                "resource_arn": OLD_SUBSCRIPTION,
            },
            "replacement": {
                "resource_id": NEW_SUBSCRIPTION,
                "resource_arn": NEW_SUBSCRIPTION,
            },
            "changed": ["resource_id", "resource_arn"],
            "retired_key": retired_resource_key(recorded),
            "proof": {
                "recorded_present": False,
                "replacement_present": True,
                "replacement_owned": True,
            },
        }
    ]
    assert harness.rows()[key] == recorded, "a plan changes nothing"

    assert harness.run(key, confirm="REPLACE_RESOURCE") == 0
    result = _printed(capsys)
    assert result["status"] == "COMMITTED" and result["replaced"] == [key]
    rows = harness.rows()
    assert rows[key].resource_id == NEW_SUBSCRIPTION
    assert rows[key].status is InstallationResourceStatus.ACTIVE
    retired = rows[retired_resource_key(recorded)]
    assert retired.resource_id == OLD_SUBSCRIPTION
    assert retired.status is InstallationResourceStatus.DETACHED
    assert retired.attributes["replaced_by"] == NEW_SUBSCRIPTION
    assert retired.attributes["replacement_reference"] == "CHG-replace-1"
    journal = json.loads(Path(result["journal"]).read_text(encoding="utf-8"))
    assert journal["phase"] == "COMMITTED" and journal["operator"]
    registry.sync_installation_resource_registry(harness.site)


def test_a_recreated_subnet_is_found_from_the_nlb_conflict_and_replaced_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    harness.resources["nlb_network"] = _network("c", "b")
    harness.live = {
        "subnet-b",
        "subnet-c",
        "rtb-b",
        "rtb-c",
        "rtbassoc-b",
        "rtbassoc-c",
    }
    with pytest.raises(BootstrapError, match=r"aws/nlb \(dependencies\)"):
        registry.sync_installation_resource_registry(harness.site)
    subnet, table, association = (
        f"aws/network/public-{kind}/1"
        for kind in ("subnet", "route-table", "route-association")
    )

    assert harness.run("aws/nlb") == 1
    plan = _printed(capsys)
    assert plan["dependency_hints"] == {"aws/nlb": [subnet]}
    assert plan["refusals"] == [
        "aws/nlb: the rebuild changes more than its physical resource "
        "(dependencies); that is a defect to fix, not a replacement"
    ]

    assert harness.run(subnet) == 1
    assert _printed(capsys)["remaining_conflicts"] == [
        f"{table} (resource_id)",
        f"{association} (resource_id)",
    ], "a partial transition must name what is still to replace"

    assert harness.run(subnet, table, association, confirm="REPLACE_RESOURCE") == 0
    assert _printed(capsys)["status"] == "COMMITTED"
    rows = harness.rows()
    assert (rows[subnet].resource_id, rows[table].resource_id) == ("subnet-c", "rtb-c")
    assert rows[association].resource_id == "rtbassoc-c"
    assert rows["aws/nlb"].dependencies == sorted(
        [subnet, "aws/network/public-subnet/2", "aws/nlb/security-group"]
    ), "the NLB keeps its dependencies: the replacement took the key"
    registry.sync_installation_resource_registry(harness.site)


@pytest.mark.parametrize(
    ("live", "untagged", "refusal"),
    [
        (
            {OLD_SUBSCRIPTION, NEW_SUBSCRIPTION},
            set(),
            "still exists -- it was not replaced, or is not gone yet",
        ),
        (set(), set(), f"the replacement {NEW_SUBSCRIPTION} is absent"),
    ],
    ids=["old-still-live", "new-absent"],
)
def test_a_replacement_without_its_proofs_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    live: set[str],
    untagged: set[str],
    refusal: str,
) -> None:
    harness = Site(tmp_path, monkeypatch)
    harness.resources["monitoring_resources"] = _monitoring(NEW_SUBSCRIPTION)
    harness.live, harness.untagged = live, untagged
    before = harness.rows()

    assert harness.run(_email_key()) == 1
    refusals = _printed(capsys)["refusals"]
    assert any(refusal in item for item in refusals), refusals
    with pytest.raises(BootstrapError, match="replace-resource refused"):
        harness.run(_email_key(), confirm="REPLACE_RESOURCE")
    assert harness.rows() == before
    journals = list((harness.state_dir / "resource-replacement").glob("*/journal.json"))
    assert [json.loads(path.read_text())["phase"] for path in journals] == ["REFUSED"]


def test_a_replacement_must_carry_the_sites_ownership_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    harness.resources["nlb_network"] = _network("c", "b")
    harness.live = {
        "subnet-b",
        "subnet-c",
        "rtb-b",
        "rtb-c",
        "rtbassoc-b",
        "rtbassoc-c",
    }
    harness.untagged = {"subnet-c"}
    keys = [
        f"aws/network/public-{kind}/1"
        for kind in ("subnet", "route-table", "route-association")
    ]
    assert harness.run(*keys) == 1
    refusals = _printed(capsys)["refusals"]
    assert refusals == [
        "aws/network/public-subnet/1: the replacement is not proven this site's: "
        "aws/network/public-subnet/1 exists without gpu-fault:site-id; refusing to "
        "share it"
    ]


@pytest.mark.parametrize(
    ("key", "refusal"),
    [
        ("aws/unknown", "the registry records no such key"),
        ("cluster/cpu-eks", "replace-resource does not replace a cpu_eks row"),
        ("aws/amp/workspace", "replace-resource does not replace a amp_workspace row"),
        ("aws/network/public-subnet/1", "the rebuild records the same resource"),
    ],
)
def test_only_reviewed_rows_this_site_created_and_really_replaced_qualify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str,
    refusal: str,
) -> None:
    harness = Site(tmp_path, monkeypatch)
    harness.live = {"subnet-a"}
    assert harness.run(key) == 1
    (refused,) = _printed(capsys)["refusals"]
    assert refused.startswith(f"{key}: {refusal}"), refused


def test_a_row_the_site_did_not_create_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    (external,) = [
        row.resource_key
        for row in harness.rows().values()
        if row.resource_type in replacement.REPLACEABLE_RESOURCE_TYPES
        and row.ownership.value == "EXTERNAL"
    ]
    assert harness.run(external) == 1
    (refused,) = _printed(capsys)["refusals"]
    assert refused.startswith(f"{external}: the row is EXTERNAL"), refused


def test_an_interrupted_attempt_is_finished_by_the_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    key = _email_key()
    harness.resources["monitoring_resources"] = _monitoring(NEW_SUBSCRIPTION)
    harness.live = {NEW_SUBSCRIPTION}
    real = registry.replace_installation_resources

    def lost(site: Any, request: Any) -> Any:
        real(site, request)
        raise BootstrapError("kubectl exec lost its connection")

    monkeypatch.setattr(registry, "replace_installation_resources", lost)
    with pytest.raises(BootstrapError, match="lost its connection"):
        harness.run(key, confirm="REPLACE_RESOURCE")
    (journal,) = (harness.state_dir / "resource-replacement").glob("*/journal.json")
    assert json.loads(journal.read_text())["phase"] == "PLANNED"
    monkeypatch.setattr(registry, "replace_installation_resources", real)
    harness.live = set()  # the plan's proofs no longer hold; the resume needs none

    with pytest.raises(BootstrapError, match="rerun with exactly those keys"):
        harness.run(key, "aws/sns/topic", confirm="REPLACE_RESOURCE")
    assert harness.run(key, confirm="REPLACE_RESOURCE") == 0
    result = _printed(capsys)
    assert result["resumed"] is True and result["status"] == "COMMITTED"
    assert json.loads(journal.read_text())["phase"] == "COMMITTED"
    assert harness.rows()[key].resource_id == NEW_SUBSCRIPTION


def test_an_open_attempt_that_no_longer_applies_is_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = Site(tmp_path, monkeypatch)
    key = _email_key()
    harness.resources["monitoring_resources"] = _monitoring(NEW_SUBSCRIPTION)
    harness.live = {NEW_SUBSCRIPTION}
    real = registry.replace_installation_resources

    def unreachable(*_args: Any) -> Any:
        raise BootstrapError("kubectl exec failed before the API")

    monkeypatch.setattr(registry, "replace_installation_resources", unreachable)
    with pytest.raises(BootstrapError, match="failed before the API"):
        harness.run(key, confirm="REPLACE_RESOURCE")
    monkeypatch.setattr(registry, "replace_installation_resources", real)
    changed = harness.rows()[key].model_copy(update={"error": "a later write"})
    harness.pod.store.save_installation_resource(changed)

    with pytest.raises(BootstrapError, match="nothing of it landed") as abandoned:
        harness.run(key, confirm="REPLACE_RESOURCE")
    assert f"{key} (error)" in str(abandoned.value)
    (journal,) = (harness.state_dir / "resource-replacement").glob("*/journal.json")
    assert json.loads(journal.read_text())["phase"] == "ABANDONED"
    assert harness.rows()[key] == changed

    assert harness.run(key, confirm="REPLACE_RESOURCE") == 0, "a fresh plan applies"
    assert _printed(capsys)["status"] == "COMMITTED"
    assert harness.rows()[key].resource_id == NEW_SUBSCRIPTION


def test_replace_resource_refuses_while_a_deletion_verb_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Site(tmp_path, monkeypatch)
    journal = harness.state_dir / "remove-cluster" / "gpu-a" / "state.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"phase": "AWS_DETACHED"}), encoding="utf-8")
    with pytest.raises(BootstrapError, match="while remove-cluster/gpu-a is open"):
        harness.run(_email_key(), confirm="REPLACE_RESOURCE")


def test_a_control_plane_without_the_route_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Site(tmp_path, monkeypatch)
    harness.resources["monitoring_resources"] = _monitoring(NEW_SUBSCRIPTION)
    harness.live = {NEW_SUBSCRIPTION}
    real = harness.pod.run

    def legacy(arguments: list[str], **options: Any) -> Any:
        script = arguments[-1].replace(
            'require_route("/v1/installation-resources/replacements")',
            'require_route("/v1/a-route-an-older-release-has")',
        )
        return real([*arguments[:-1], script], **options)

    monkeypatch.setattr(registry, "run_command", legacy)
    with pytest.raises(BootstrapError, match="predates the installation resource"):
        harness.run(_email_key(), confirm="REPLACE_RESOURCE")
    assert harness.rows()[_email_key()].resource_id == OLD_SUBSCRIPTION


def test_the_cli_dispatches_replace_resource_with_the_managed_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_fault.admin import cli

    path = site_file(tmp_path)
    calls: list[tuple[list[str], Path]] = []

    def run(arguments: argparse.Namespace, *, site: Any) -> int:
        calls.append((arguments.resource_key, site.source))
        return 0

    monkeypatch.setattr(cli, "run_replace_resource_command", run)
    arguments = cli.parser().parse_args(
        [
            "replace-resource",
            "--state-dir",
            str(tmp_path),
            "--resource-key",
            "aws/sns/topic",
            "--reference",
            "CHG-fixture",
        ]
    )
    assert cli.run(arguments) == 0
    assert calls == [(["aws/sns/topic"], path.resolve())]


def test_an_uninstall_settles_a_retired_row_by_its_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retired row is an ordinary registry row to uninstall: its resource is
    gone, so it verifies terminal; nothing keys on the ``retired/`` prefix."""

    from gpu_fault.admin.uninstall import uninstall
    from tests.admin.test_uninstall_lifecycle import Harness

    harness = Harness(tmp_path, monkeypatch)
    nlb = next(
        row for row in harness.snapshot.resources if row.resource_key == "aws/nlb"
    )
    replaced = nlb.model_copy(update={"resource_id": "nlb-replaced-earlier"})
    retired = replaced.model_copy(
        update={
            "resource_key": retired_resource_key(replaced),
            "status": InstallationResourceStatus.DELETED,
        }
    )
    snapshot = harness.snapshot.model_copy(
        update={"resources": [*harness.snapshot.resources, retired]}
    )
    harness.snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})

    result = uninstall(harness.request(), runner=harness)

    assert result["delete_policy_residuals"] == 0
    final = registry.load_installation_resource_snapshot(
        harness.site.source.parent / "uninstall" / "installation-resources-final.json"
    )
    statuses = {row.resource_key: row.status for row in final.resources}
    assert statuses[retired.resource_key] is InstallationResourceStatus.DELETED
