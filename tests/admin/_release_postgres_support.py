"""Command doubles for the production local release PostgreSQL allocator."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.command_log import child_failure
from gpu_fault.admin.postgres_grant import LOCAL_DOCKER_HOST, OWNER_LABEL

IMAGE = "sha256:" + "a" * 64
CID = "b" * 64
FOREIGN_CID = "c" * 64


class FakePostgresDocker(CommandRunner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.containers: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.responses: dict[str, str] = {}
        self.failures: dict[str, BaseException] = {}
        self.images = True
        self.ready = True
        self.ready_code = 1
        self.readiness_calls = 0
        self.pulls = 0
        self.starts: list[str] = []
        self.removal_attempts: list[str] = []
        self.removed: list[str] = []
        self.keep_after_remove = False
        self.directory: Path | None = None
        self.created: dict[str, Any] | None = None

    def action(self, name: str, item: dict[str, Any]) -> None:
        if name in self.hooks:
            self.hooks[name](item)
        if name in self.failures:
            raise self.failures[name]

    def run(self, arguments: Sequence[str], **options: Any) -> str:
        command = list(arguments)
        assert command[:3] == ["docker", "--host", LOCAL_DOCKER_HOST], (
            "owned PostgreSQL commands must select the local Docker daemon"
        )
        assert options["capture"] is True and options["sensitive"] is True, (
            "local PostgreSQL control output must use the private capture channel"
        )
        assert 0 < options["timeout_seconds"] <= 600
        self.calls.append((command, options))
        args = command[3:]
        operation = ""
        if args[:2] == ["image", "ls"]:
            operation = "image-list"
            output = IMAGE if self.images else ""
        elif args[:2] == ["image", "inspect"]:
            operation = "image-inspect"
            output = json.dumps(IMAGE)
        elif args[0] == "pull":
            operation = "pull"
            self.pulls += 1
            self.images = True
            output = ""
        elif args[0] == "create":
            cidfile = Path(args[args.index("--cidfile") + 1])
            self.directory = cidfile.parent
            owner = args[args.index("--label") + 1].split("=", 1)[1]
            image = args[args.index("--") + 1]
            item = {
                "id": CID,
                "name": "/" + args[args.index("--name") + 1],
                "image": image,
                "reference": image,
                "labels": {OWNER_LABEL: owner},
                "running": False,
                "ports": {},
                "bindings": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]},
            }
            self.created = copy.deepcopy(item)
            self.action("before-create", item)
            self.containers[CID] = item
            cidfile.write_text(CID)
            self.action("after-create", item)
            return self.responses.get("create", CID)
        elif args[:2] == ["container", "ls"]:
            operation = "list"
            self.action("list", {})
            selector = args[args.index("--filter") + 1]
            items = list(self.containers.values())
            if selector.startswith("id="):
                items = [item for item in items if item["id"] == selector[3:]]
            else:
                assert selector.startswith("name=^/") and selector.endswith("$")
                items = [item for item in items if item["name"] == selector[6:-1]]
            return self.responses.get("list", "\n".join(item["id"] for item in items))
        elif args[:2] == ["container", "inspect"]:
            operation = "inspect"
            item = self.containers.get(args[-1])
            if item is None:
                raise child_failure(BootstrapError, command, 1, sensitive=True)
            self.action("inspect", item)
            return self.responses.get("inspect", json.dumps(item))
        elif args[0] == "start":
            operation = "start"
            item = self.containers[args[-1]]
            self.starts.append(args[-1])
            item.update(
                running=True,
                ports={"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "54321"}]},
            )
            self.action("start", item)
            return args[-1]
        elif args[0] == "exec":
            operation = "readiness"
            self.readiness_calls += 1
            self.action("readiness", self.containers[args[1]])
            if not self.ready:
                raise child_failure(
                    BootstrapError, command, self.ready_code, sensitive=True
                )
            return ""
        elif args[:2] == ["container", "rm"]:
            operation = "remove"
            identifier = args[-1]
            self.removal_attempts.append(identifier)
            self.action("before-remove", self.containers.get(identifier, {}))
            if not self.keep_after_remove and identifier in self.containers:
                item = self.containers.pop(identifier)
                self.removed.append(identifier)
            else:
                item = {}
            self.action("after-remove", item)
            return ""
        else:
            raise AssertionError(f"unexpected fake Docker operation: {args[:2]}")
        self.action(operation, {})
        return self.responses.get(operation, output)
