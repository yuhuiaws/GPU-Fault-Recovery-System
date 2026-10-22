"""The BOOT-036/037 verdicts, one wrong shape at a time.

The pass shapes come from the real verb (``tests/regional/_eks_nodegroup_site``):
a prepared group's plan, its refusals and an apply. Each test bends one field
of such a document and names the reason the verdict must give.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import eks_nodegroup_case as case
from scripts.e2e.regional import eks_nodegroup_verdicts as verdicts
from tests.regional._eks_nodegroup_site import (
    CPU_EKS_ARN,
    GPU_GROUP,
    NOW,
    SITE_ID,
    Site,
)


@pytest.fixture(name="site")
def prepared(tmp_path: Path) -> Site:
    site = Site(tmp_path)
    site.prepare()
    return site


def target(site: Site) -> case.SiteTarget:
    return case.site_target(site.state_dir, "gpu-eks-cluster")


def plan(site: Site, *flags: str) -> dict[str, Any]:
    run = site.tool(
        target(site).tool_arguments("--nodegroup", GPU_GROUP, *flags), timeout=60
    )
    assert run.document is not None, run.stderr
    return run.document


def facts(site: Site) -> dict[str, verdicts.GroupFacts]:
    instances = tuple(site.world.instances[GPU_GROUP])
    return {GPU_GROUP: verdicts.GroupFacts(instances, instances, 2)}


def group(document: dict[str, Any]) -> dict[str, Any]:
    return document["nodegroups"][0]


def test_the_prepared_plan_is_the_pass_shape(site: Site) -> None:
    errors = verdicts.compliant_plan_errors(
        plan(site), target=target(site), facts=facts(site), mode="host", pin="4.6.1-1"
    )

    assert errors == []


def test_no_document_is_named_first(site: Site) -> None:
    assert verdicts.compliant_plan_errors(
        None, target=target(site), facts=facts(site), mode="host"
    ) == ["the tool printed no plan document"]


FOREIGN_CLUSTER = {"arn": CPU_EKS_ARN, "provider": "hyperpod", "site_managed": False}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"verb": "other"}, "not prepare-eks-nodegroup mode plan"),
        ({"site_id": "other"}, "is not the site's"),
        ({"ownership_label": {"value": "other"}}, "label value is not the site id"),
        ({"gpu_cluster": FOREIGN_CLUSTER}, "not the target member"),
        ({"gpu_cluster": FOREIGN_CLUSTER}, "not a site-managed eks-managed member"),
        ({"recorded_site": {"matches": False}}, "the state directory records site"),
        ({"status": "CHANGES"}, "the plan status is CHANGES"),
        ({"nodegroups": []}, "the plan covers []"),
    ],
)
def test_a_document_for_another_target_fails(
    site: Site, change: dict[str, Any], reason: str
) -> None:
    document = plan(site)
    document.update(copy.deepcopy(change))

    errors = verdicts.compliant_plan_errors(
        document, target=target(site), facts=facts(site), mode="host"
    )

    assert any(reason in item for item in errors), errors


def bent(document: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    result = copy.deepcopy(document)
    entry = group(result)
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("verdict",), "CHANGES", "verdict CHANGES"),
        (("refusal",), "x", "verdict COMPLIANT (x)"),
        (("steps",), [{"step": "roll"}], "the plan would run ['roll']"),
        (("node_repair",), "enabled", "EKS node auto repair is enabled"),
        (("labels", "gpu-fault.io/control-plane"), None, "is not labelled"),
        (("update_strategy", "current"), "DEFAULT", "updateStrategy DEFAULT"),
        (("update_strategy", "finding"), "kept", "(kept)"),
        (("roll", "needed"), True, "still needs a roll"),
        (("dcgm_hostengine", "mode"), "embedded", "DCGM mode embedded"),
        (("labels", "gpu-fault.io/dcgm-hostengine"), None, "does not declare"),
        (("dcgm_hostengine", "pin"), "4.7.0-2", "expected 4.6.1-1"),
        (("dcgm_hostengine", "exporter_pairing"), "host-newer", "not paired"),
        (("dcgm_hostengine", "restart_dropin"), {}, "no nvidia-dcgm restart drop-in"),
        (("user_data", "action"), "replace", "the user data would replace"),
        (("user_data", "parts"), [], "exactly one managed part"),
        (
            ("user_data", "parts"),
            [{"index": 1, "managed": True, "contradictions": ["unpinned"]}],
            "part 1: unpinned",
        ),
    ],
)
def test_each_unprepared_field_is_named(
    site: Site, path: tuple[str, ...], value: Any, reason: str
) -> None:
    errors = verdicts.compliant_plan_errors(
        bent(plan(site), path, value),
        target=target(site),
        facts=facts(site),
        mode="host",
        pin="4.6.1-1",
    )

    assert any(reason in item for item in errors), errors


def test_an_embedded_group_needs_no_dcgm(site: Site) -> None:
    document = bent(plan(site), ("dcgm_hostengine", "mode"), "embedded")

    errors = verdicts.compliant_group_errors(
        GPU_GROUP, group(document), target=target(site), mode="embedded"
    )

    assert errors == []


# --- the update strategy of a single-node group ------------------------------------

#: The finding of the deploy-host tool before the single-node remedy: the
#: verdict must read the plan's fields, not these words.
OLD_TOOL_FINDING = (
    "updateStrategy MINIMAL with maxUnavailable 1: a roll takes 1 of 1 GPU node "
    "(the whole group) out at a time; lower maxUnavailable below the group size to "
    "keep GPU nodes up during a roll"
)


def single_node_facts(site: Site) -> dict[str, verdicts.GroupFacts]:
    instances = tuple(site.world.instances[GPU_GROUP])
    return {GPU_GROUP: verdicts.GroupFacts(instances, instances, 1)}


@pytest.mark.parametrize("finding", [None, OLD_TOOL_FINDING], ids=["tool", "old-tool"])
def test_a_single_node_groups_whole_group_roll_is_a_limitation(
    site: Site, finding: str | None
) -> None:
    site.scale_in(1)
    document = plan(site)
    strategy = group(document)["update_strategy"]
    assert "a single-node group keeps no GPU node up" in strategy["finding"], strategy
    if finding is not None:
        strategy["finding"] = finding

    errors = verdicts.compliant_plan_errors(
        document,
        target=target(site),
        facts=single_node_facts(site),
        mode="host",
        pin="4.6.1-1",
    )

    assert errors == [], "no maxUnavailable avoids it: not a failure"
    (note,) = verdicts.strategy_limitations(document)
    assert note.startswith(
        f"{GPU_GROUP} runs a single node (desiredSize 1): a MINIMAL roll takes its "
        "only GPU node out until the replacement joins"
    ), note
    assert note.endswith(f"(the tool's finding: {strategy['finding']})"), note


@pytest.mark.parametrize(
    ("change", "reason", "noted"),
    [
        ({"current": "DEFAULT"}, "updateStrategy DEFAULT, planned MINIMAL", True),
        ({"planned": "DEFAULT"}, "updateStrategy MINIMAL, planned DEFAULT", False),
        ({"kept": True}, "updateStrategy MINIMAL (kept)", False),
    ],
    ids=["current", "planned", "kept"],
)
def test_a_single_node_group_off_minimal_still_fails(
    site: Site, change: dict[str, Any], reason: str, noted: bool
) -> None:
    site.scale_in(1)
    document = plan(site)
    group(document)["update_strategy"].update(change)

    errors = verdicts.compliant_plan_errors(
        document, target=target(site), facts=single_node_facts(site), mode="host"
    )

    assert any(reason in item for item in errors), errors
    assert bool(verdicts.strategy_limitations(document)) is noted, (
        "the note follows the planned roll: MINIMAL and not kept"
    )


def test_a_whole_group_roll_of_two_nodes_still_fails(site: Site) -> None:
    site.world.nodegroups[GPU_GROUP]["updateConfig"] = {
        "maxUnavailable": 2,
        "updateStrategy": "MINIMAL",
    }
    document = plan(site)

    errors = verdicts.compliant_plan_errors(
        document, target=target(site), facts=facts(site), mode="host"
    )

    assert errors == [
        f"{GPU_GROUP}: updateStrategy MINIMAL (updateStrategy MINIMAL with "
        "maxUnavailable 2: a roll takes 2 of 2 GPU nodes (the whole group) out at a "
        "time; lower maxUnavailable below the group size to keep GPU nodes up during "
        "a roll)"
    ], "a lower maxUnavailable fixes it: still a failure"


def test_only_a_minimal_roll_of_exactly_one_node_is_the_limitation() -> None:
    def shaped(desired: object, **strategy: object) -> dict[str, Any]:
        return {
            "scaling": {"desiredSize": desired},
            "update_strategy": {"planned": "MINIMAL", "kept": False, **strategy},
        }

    assert verdicts.single_node_roll(shaped(1)) is True
    assert not verdicts.single_node_roll(shaped(2)), "two nodes can keep one up"
    assert not verdicts.single_node_roll(shaped(True)), "a bool is no node count"
    assert not verdicts.single_node_roll(shaped(None)), "an unknown size is not one"
    assert not verdicts.single_node_roll(shaped(1, planned="DEFAULT")), "surge"
    assert not verdicts.single_node_roll(shaped(1, kept=True)), "a kept strategy"
    assert not verdicts.single_node_roll({}), "no fields, no limitation"


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("current_nodes", "outcome"), "unavailable", "inventory is unavailable"),
        (("current_nodes", "reason"), "AccessDenied", "(AccessDenied)"),
        (("current_nodes", "findings"), ["drift"], "node findings: drift"),
        (("current_nodes", "nodes"), [], "checked no node"),
    ],
)
def test_an_inventory_that_did_not_pass_is_named(
    site: Site, path: tuple[str, ...], value: Any, reason: str
) -> None:
    document = bent(plan(site), path, value)
    if path[-1] == "reason":
        group(document)["current_nodes"]["outcome"] = "failed"

    errors = verdicts.compliant_plan_errors(
        document, target=target(site), facts=facts(site), mode="host"
    )

    assert any(reason in item for item in errors), errors


def test_a_failing_node_check_and_a_node_finding_are_named(site: Site) -> None:
    document = plan(site)
    node = group(document)["current_nodes"]["nodes"][0]
    node["checks"]["strace"] = {"ok": False, "observed": "missing", "expected": "ok"}
    node["passed"] = False
    node["findings"] = ["a host engine is active"]

    errors = verdicts.compliant_plan_errors(
        document, target=target(site), facts=facts(site), mode="host"
    )

    assert any("fails strace missing (expected ok)" in item for item in errors), errors
    assert any("findings: a host engine is active" in item for item in errors), errors


def test_the_inventory_must_cover_exactly_the_groups_instances(site: Site) -> None:
    document = plan(site)
    instances = ("i-a", "i-b", "i-c")

    errors = verdicts.compliant_plan_errors(
        document,
        target=target(site),
        facts={GPU_GROUP: verdicts.GroupFacts(instances, ("i-a",), 4)},
        mode="host",
    )

    assert any("the group runs ['i-a', 'i-b', 'i-c']" in item for item in errors), (
        errors
    )
    assert any("Kubernetes Nodes are ['i-a']" in item for item in errors), errors
    assert any("3 running instances, desired 4" in item for item in errors), errors


@pytest.mark.parametrize(
    ("findings", "reason"),
    [
        ({"available": False, "reason": "no access entry"}, "unavailable: no access"),
        (
            {"available": True, "namespaces": {"missing": ["training"]}},
            "namespaces missing",
        ),
        (
            {
                "available": True,
                "device_plugins": {"nvidia": {"running": False}, "efa": {}},
            },
            "nvidia device plugin not running",
        ),
        (
            {"available": True, "quiesce_tolerating_gpu_daemonsets": [{"d": 1}]},
            "tolerate the quiesce taint",
        ),
    ],
)
def test_kubernetes_findings_must_be_clear(
    findings: dict[str, Any], reason: str
) -> None:
    assert any(reason in item for item in verdicts.kubernetes_errors(findings)), (
        verdicts.kubernetes_errors(findings)
    )


# --- refusals ---------------------------------------------------------------------


def run(returncode: int, stdout: str = "", stderr: str = "") -> case.ToolRun:
    return case.ToolRun((), returncode, stdout, stderr)


def test_a_refusal_must_exit_two_say_the_fact_and_stop_early() -> None:
    expected = verdicts.Refusal(
        "unknown", "has no managed node group x", forbidden_calls=verdicts.NODE_READS
    )
    stderr = (
        "+ aws ssm send-command --document-name AWS-RunShellScript\n"
        "gpu-fault-admin: logging to /state/logs/readonly/x.log\n"
        "gpu-fault-admin: something else\n"
    )

    errors = verdicts.refusal_errors(run(0, '{\n"verb": "x"\n}\n', stderr), expected)

    assert any("exit 0, expected 2" in item for item in errors), errors
    assert any(
        "did not say 'has no managed node group x'" in item for item in errors
    ), errors
    assert any("still reached ['aws ssm send-command']" in item for item in errors), (
        errors
    )


def test_a_group_refusal_must_say_nothing_was_changed(site: Site) -> None:
    document = plan(site)
    document["status"] = "REFUSED"
    group(document)["refusal"] = "runs amiType X"

    expected = verdicts.Refusal("unsupported", "amiType X", document="REFUSED")
    errors = verdicts.refusal_errors(
        run(2, json.dumps(document, indent=2) + "\n"), expected
    )

    assert errors == ["unsupported: the refusal does not say nothing was changed"]


def test_a_refusal_with_the_wrong_plan_status_is_named() -> None:
    expected = verdicts.Refusal("adopt", "nothing to adopt", document="COMPLIANT")

    errors = verdicts.refusal_errors(
        run(2, "", "gpu-fault-admin: nothing to adopt\n"), expected
    )

    assert errors == ["adopt: plan status None, expected COMPLIANT"]


def test_snapshot_drift_names_every_difference() -> None:
    before = {
        "nodegroups": {"a": {"status": "ACTIVE", "instances": ["i-1"]}},
        "journals": {"j": "1"},
    }
    after = {
        "nodegroups": {
            "a": {"status": "UPDATING", "instances": ["i-1"]},
            "b": {"x": 1},
        },
        "journals": {"j": "2"},
    }

    errors = verdicts.snapshot_drift(before, after)

    assert errors == [
        "a: status changed from ACTIVE to UPDATING",
        "b: x changed from None to 1",
        "the tool's journals changed: {'j': '1'} -> {'j': '2'}",
    ]
    assert verdicts.snapshot_drift(before, copy.deepcopy(before)) == []


# --- the roll ---------------------------------------------------------------------


def roll_errors(
    site: Site, document: dict[str, Any] | None, **options: Any
) -> list[str]:
    values: dict[str, Any] = {"nodegroup": GPU_GROUP, "mode": "host", "desired": 2}
    values.update(options)
    return verdicts.roll_plan_errors(document, target=target(site), **values)


def test_the_repin_plan_is_a_minimal_roll(site: Site) -> None:
    document = plan(site, "--dcgm-hostengine", "host", "--dcgm-version", "4.6.1")

    assert roll_errors(site, document) == []
    assert roll_errors(site, None) == ["the tool printed no plan document"]


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("COMPLIANT", "already runs the target configuration"),
        ("IN_PROGRESS", "of an earlier apply is still in progress"),
        ("REFUSED", "is refused"),
        ("SOMETHING", "the plan status is SOMETHING"),
    ],
)
def test_a_plan_that_does_not_roll_says_why(
    site: Site, status: str, reason: str
) -> None:
    document = plan(site, "--dcgm-hostengine", "host", "--dcgm-version", "4.6.1")
    document["status"] = status

    errors = roll_errors(site, document)

    assert any(reason in item for item in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("steps",), [{"step": "labels"}], "not a roll"),
        (("steps",), [{"step": "roll"}, {"step": "replace"}], "not a roll"),
        (("update_strategy", "planned"), "DEFAULT", "updateStrategy DEFAULT"),
        (("dcgm_hostengine", "mode"), "embedded", "DCGM mode embedded"),
        (("dcgm_hostengine", "exporter_pairing"), "client-newer", "not paired"),
        (("current_nodes", "outcome"), "unavailable", "unreadable"),
        (("current_nodes", "nodes"), [{}], "read 1 nodes, desired 2"),
    ],
)
def test_each_unsafe_roll_field_is_named(
    site: Site, path: tuple[str, ...], value: Any, reason: str
) -> None:
    document = plan(site, "--dcgm-hostengine", "host", "--dcgm-version", "4.6.1")

    errors = roll_errors(site, bent(document, path, value))

    assert any(reason in item for item in errors), errors


def test_a_roll_taking_two_nodes_out_at_once_is_refused(site: Site) -> None:
    site.world.nodegroups[GPU_GROUP]["updateConfig"] = {
        "maxUnavailable": 2,
        "updateStrategy": "MINIMAL",
    }
    document = plan(site, "--dcgm-hostengine", "host", "--dcgm-version", "4.6.1")

    errors = roll_errors(site, document)

    assert errors == [f"{GPU_GROUP}: {group(document)['update_strategy']['finding']}"]
    assert "lower maxUnavailable below the group size" in errors[0], (
        "the replan after it would fail the BOOT-036 verdict on the same finding"
    )


def test_a_single_node_roll_is_not_refused_for_taking_its_node_out(site: Site) -> None:
    site.scale_in(1)
    document = plan(site, "--dcgm-hostengine", "host", "--dcgm-version", "4.6.1")
    assert group(document)["update_strategy"]["finding"], "the tool names it"

    assert roll_errors(site, document, desired=1) == []


def apply_document(**apply: Any) -> str:
    result = {
        "nodegroup": GPU_GROUP,
        "outcome": "COMPLETED",
        "steps": [{"step": "roll", "update": {"status": "Successful"}}],
        "verification": {"outcome": "passed"},
    }
    result.update(apply.pop("result", {}))
    document = {
        "verb": case.TOOL_VERB,
        "mode": "apply",
        "apply": {"status": "COMPLETED", "results": [result], **apply},
    }
    return json.dumps(document, indent=2) + "\n"


def test_a_completed_apply_is_the_pass_shape() -> None:
    assert verdicts.apply_errors(run(0, apply_document()), nodegroup=GPU_GROUP) == []


def test_every_way_an_apply_stops_is_named() -> None:
    document = apply_document(
        status="STOPPED",
        result={
            "outcome": "ROLL_FAILED",
            "reason": "NodeCreationFailure",
            "steps": [{"step": "roll", "update": {"status": "Failed"}}],
            "verification": {},
        },
    )

    errors = verdicts.apply_errors(
        run(2, document, "gpu-fault-admin: stopped\n"), nodegroup=GPU_GROUP
    )

    assert errors[0] == "the apply exited 2: ['stopped']"
    assert "the apply status is STOPPED" in errors
    assert any("ROLL_FAILED: NodeCreationFailure" in item for item in errors), errors
    assert any("no roll update ended Successful" in item for item in errors), errors
    assert any("verification is None" in item for item in errors), errors


def node(name: str, **fields: Any) -> dict[str, Any]:
    value = {
        "name": name,
        "instance_id": f"i-{name}",
        "ready": "True",
        "ready_since": (NOW - timedelta(hours=1)).isoformat(),
        "unschedulable": False,
        "blocking_taints": [],
        "labels": {
            "gpu-fault.io/control-plane": SITE_ID,
            "gpu-fault.io/dcgm-hostengine": "host",
        },
    }
    value.update(fields)
    return value


def test_replacement_and_label_errors() -> None:
    before = [node("a"), node("b")]

    assert verdicts.replacement_errors(before, [node("c"), node("d")], desired=2) == []
    errors = verdicts.replacement_errors(
        before, [node("a"), node("e", ready="False")], desired=3
    )
    assert errors == [
        "departed nodes are still Nodes of the group: ['a']",
        "the group has 2 Nodes, desired 3",
        "replacement nodes not Ready: ['e']",
    ]
    labels = verdicts.label_errors([node("c", labels={})], site_id=SITE_ID, mode="host")
    assert len(labels) == 2, labels
    assert verdicts.label_errors(
        [node("c", labels={})], site_id="", mode="embedded"
    ) == ["c: no gpu-fault.io/control-plane="]


def test_agent_errors_split_new_and_departed_nodes() -> None:
    store = {
        "agents": [
            {"node_id": "new-1", "lifecycle_state": "ACTIVE"},
            {"node_id": "old-1", "lifecycle_state": "DRAINING"},
            {"node_id": "old-2", "lifecycle_state": "REVOKED"},
            "not a record",
        ]
    }

    errors = verdicts.agent_errors(
        store, active=["new-1", "new-2"], departed=["old-1", "old-2", "old-3"]
    )

    assert errors == [
        "new-2: Agent absent, not ACTIVE",
        "old-1: the departed node's Agent is still DRAINING",
    ]


def test_incident_errors_by_scope() -> None:
    since = NOW
    store = {
        "incidents": [
            {
                "incident_id": "new",
                "created_at": (NOW + timedelta(seconds=1)).isoformat(),
            },
            {
                "incident_id": "far",
                "active": True,
                "node_ids": ["x"],
                "created_at": "bad",
            },
            {"incident_id": "near", "active": False, "node_ids": ["n1"]},
            "not a record",
        ],
        "workflows": [
            {"request_id": "w-far", "node_ids": ["x"]},
            {"request_id": "w-near", "node_ids": ["n1"]},
            "not a record",
        ],
    }

    scoped = verdicts.incident_errors(
        store, since=since, nodes=["n1"], cluster_wide=False
    )
    wide = verdicts.incident_errors(store, nodes=["n1"])

    assert [item.split(":")[0] for item in scoped] == [
        "incident opened during the case",
        "open incident on the group's nodes",
        "executable workflow w-near None on ['n1']",
    ]
    assert [item.split(":")[0] for item in wide] == [
        "active incident",
        "open incident on the group's nodes",
        "executable workflow w-far None on ['x']",
        "executable workflow w-near None on ['n1']",
    ], "without a start only the active, the node-bound and every workflow count"


def test_stability_errors_need_ten_ready_minutes_and_no_hold() -> None:
    errors = verdicts.stability_errors(
        [
            node("a"),
            node("b", ready_since=(NOW - timedelta(minutes=2)).isoformat()),
            node("c", ready="False"),
            node("d", ready_since="2030-01-02T03:04:05"),
            node("e", unschedulable=True, blocking_taints=["gpu-fault.io/q:NoExecute"]),
        ],
        now=NOW,
        ready_seconds=600,
    )

    assert errors == [
        "b: Ready for 120 s only, not 600 s",
        "c: not Ready",
        "d: not Ready",
        "e: cordoned",
        "e: tainted ['gpu-fault.io/q:NoExecute']",
    ]


def test_renames_pair_names_in_order_and_feed_the_profile() -> None:
    renames = verdicts.node_renames([node("b"), node("a")], [node("z")])

    assert renames == [
        {
            "departed": {"name": "a", "instance_id": "i-a"},
            "replacement": {"name": "z", "instance_id": "i-z"},
        },
        {"departed": {"name": "b", "instance_id": "i-b"}, "replacement": None},
    ]
    assert verdicts.profile_updates(
        {"fault-node": "a", "pair": ["b", "c"], "count": 2}, renames
    ) == [
        {"argument": "fault-node", "departed": "a", "replacement": "z"},
        {"argument": "pair", "departed": "b", "replacement": None},
    ]
    assert verdicts.node_renames([], [node("y")])[0]["departed"] is None


def test_case_verdict_needs_stages_and_no_error() -> None:
    assert verdicts.case_verdict({}) == "FAIL"
    assert verdicts.case_verdict({"a": [], "b": ["x"]}) == "FAIL"
    assert verdicts.case_verdict({"a": [], "b": []}) == "PASS"


def test_groups_by_name_tolerates_a_missing_document() -> None:
    assert verdicts.groups_by_name(None) == {}
    assert verdicts.groups_by_name({"nodegroups": ["x", {"nodegroup": "g"}]}) == {
        "g": {"nodegroup": "g"}
    }


def test_the_utc_instant_parser_refuses_naive_and_garbage() -> None:
    aware = datetime(2030, 1, 1, tzinfo=timezone.utc)

    assert (
        verdicts.stability_errors(
            [node("a", ready_since=aware.isoformat())], now=aware, ready_seconds=0
        )
        == []
    )
