from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scripts.e2e.regional import seeded_command_fixture as seeded

RUN_ID = "unit-run"
IMAGE = "registry.example/executor@sha256:" + "a" * 64


def probe(root: Path, **updates: Any) -> seeded.SeededCommandProbe:
    script = root / "unit-probe.py"
    script.write_text("raise RuntimeError('fixture must not execute')\n")
    return seeded.SeededCommandProbe(
        **{
            "case_id": "GF-REGIONAL-CMD-017",
            "run_prefix": "unit",
            "pod": "unit-pod",
            "configmap": "unit-script",
            "owner": "unit-owner",
            "script": script,
            "environment": {"UNIT_MODE": "nonphysical"},
            **updates,
        }
    )


def metadata(kind: str, *, run_id: str = RUN_ID) -> dict[str, Any]:
    return {
        "uid": f"uid-{kind}",
        "resourceVersion": "1",
        "labels": {seeded.RUN_LABEL: run_id},
    }


class ResourceAPI:
    def __init__(self) -> None:
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.created: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.create_error_kind: str | None = None
        self.invalid_receipt_kind: str | None = None
        self.replace_at_ready = False
        self.file_responses = ["present"]
        self.state: dict[str, Any] = {"ready": True}
        self.read_error: Exception | None = None

    def __call__(self, *args: str, **kwargs: Any) -> str:
        self.calls.append((args, copy.deepcopy(kwargs)))
        if args[0] == "get" and args[1] == "deployment":
            return json.dumps(
                {"spec": {"template": {"spec": {"containers": [{"image": IMAGE}]}}}}
            )
        if args[0] == "get":
            if self.read_error is not None:
                raise self.read_error
            value = self.resources.get((args[1], args[2]))
            if args[-1] == "name":
                return f"{args[1]}/{args[2]}" if value else ""
            if args[-1] == "jsonpath={.status.phase}":
                return "Running" if value else ""
            return json.dumps(value) if value else ""
        if args[0] == "create":
            document = json.loads(kwargs["stdin"])
            kind, name = document["kind"].lower(), document["metadata"]["name"]
            value = {**document["metadata"], **metadata(kind)}
            self.resources[(kind, name)] = value
            self.created.append(document)
            if self.create_error_kind == kind:
                raise TimeoutError("synthetic create acknowledgement lost")
            if self.invalid_receipt_kind == kind:
                value = {**value, "uid": None}
            return json.dumps(value)
        if args[0] == "wait":
            if self.replace_at_ready:
                self.resources[("pod", args[2].removeprefix("pod/"))]["uid"] = (
                    "replacement"
                )
            return "Ready"
        if args[0] == "exec":
            if args[3] == "sh":
                return (
                    self.file_responses.pop(0)
                    if len(self.file_responses) > 1
                    else self.file_responses[0]
                )
            if args[3] == "cat":
                return json.dumps(self.state)
            if args[3] in {"touch", "rm"}:
                return ""
        if args[0] == "logs":
            return "unit log"
        if args[0] == "delete":
            plural, name = args[2].rsplit("/", 2)[-2:]
            kind = {
                "pods": "pod",
                "configmaps": "configmap",
                "secrets": "secret",
                "clusterroles": "clusterrole",
            }[plural]
            options = json.loads(kwargs["stdin"])
            current = self.resources[(kind, name)]
            assert options["preconditions"] == {
                "uid": current["uid"],
                "resourceVersion": current["resourceVersion"],
            }
            self.deletes.append(options)
            self.resources.pop((kind, name))
            return ""
        raise AssertionError(f"unconfigured synthetic transport operation: {args[:3]}")
