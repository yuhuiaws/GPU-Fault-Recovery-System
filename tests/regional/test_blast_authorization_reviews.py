from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_1 as one
from tests.regional.test_blast_acceptance_review import make_runner


def query(runner, resources=("jobsets.jobset.x-k8s.io",), verbs=("get",)):
    return runner.auth_can_i(
        kube_prefix=("--context", "unit-gpu"),
        service_account="system:serviceaccount:training:executor",
        verbs=verbs,
        resources=resources,
        namespace="training",
    )


@pytest.mark.parametrize(
    "resource,group,name",
    [
        ("nodes", "", "nodes"),
        ("pods", "", "pods"),
        ("jobs", "batch", "jobs"),
        ("jobsets.jobset.x-k8s.io", "jobset.x-k8s.io", "jobsets"),
        ("pytorchjobs.kubeflow.org", "kubeflow.org", "pytorchjobs"),
    ],
)
def test_authorization_does_not_depend_on_resource_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resource: str,
    group: str,
    name: str,
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])
    calls = []

    def review(args, **kwargs):
        assert "auth" not in args and "can-i" not in args
        request = json.loads(kwargs["input_text"])
        calls.append(request)
        assert request["spec"]["resourceAttributes"] == {
            "verb": "get",
            "resource": name,
            "group": group,
            "namespace": "training",
        }
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({**request, "status": {"allowed": True}})
        )

    monkeypatch.setattr(one, "command", review)
    assert query(runner, (resource,)) == {"get": {resource: True}}
    assert len(calls) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "json",
        "kind",
        "api",
        "group",
        "namespace",
        "name",
        "subresource",
        "missing-status",
        "boolean",
        "denied",
        "contradiction",
        "evaluation",
    ],
)
def test_unknown_or_misbound_authorization_is_not_denial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])

    def review(args, **kwargs):
        response = json.loads(kwargs["input_text"])
        response["status"] = {"allowed": True}
        if defect == "kind":
            response["kind"] = "SubjectAccessReview"
        elif defect == "api":
            response["apiVersion"] = "wrong/v1"
        elif defect in {"group", "namespace", "name", "subresource"}:
            response["spec"]["resourceAttributes"][defect] = "foreign"
        elif defect == "missing-status":
            del response["status"]
        elif defect == "boolean":
            response["status"]["allowed"] = 1
        elif defect == "denied":
            response["status"]["denied"] = "false"
        elif defect == "contradiction":
            response["status"]["denied"] = True
        elif defect == "evaluation":
            response["status"]["evaluationError"] = "authorization unavailable"
        return SimpleNamespace(
            returncode=0,
            stdout="not JSON" if defect == "json" else json.dumps(response),
        )

    monkeypatch.setattr(one, "command", review)
    with pytest.raises(base.CheckError, match="authorization review is incomplete"):
        query(runner)


def test_authorization_queries_are_bounded_and_keep_supervision_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])
    context: ContextVar[str] = ContextVar("unit-audit-scope", default="missing")
    token = context.set("owned-audit")
    lock = Lock()
    barrier = Barrier(8, timeout=10)
    state: dict[str, Any] = {"active": 0, "maximum": 0, "calls": []}

    def review(args, **kwargs):
        assert context.get() == "owned-audit"
        request = json.loads(kwargs["input_text"])
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            state["calls"].append(request["spec"]["resourceAttributes"])
        try:
            barrier.wait()
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({**request, "status": {"allowed": False}}),
            )
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(one, "command", review)
    resources = tuple(f"resource{i}.unit.invalid" for i in range(16))
    try:
        result = query(runner, resources)
    finally:
        context.reset(token)
    assert result == {"get": dict.fromkeys(resources, False)}
    assert len(state["calls"]) == 16
    assert state["maximum"] == 8
    assert state["active"] == 0


def test_ambiguous_resource_group_does_not_start_a_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])
    calls = []
    monkeypatch.setattr(one, "command", lambda *a, **kw: calls.append(a))
    with pytest.raises(base.CheckError, match="API group is unknown"):
        query(runner, ("unknown",))
    assert calls == []
