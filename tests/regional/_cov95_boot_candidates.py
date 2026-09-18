from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.admin import bootstrap_common, release_artifacts
from scripts.e2e.regional import boot020_release_candidates as candidates


class CandidateBuild:
    def __init__(self, root: Path, monkeypatch: Any) -> None:
        snapshot = root / "snapshot"
        (snapshot / "dist").mkdir(parents=True)
        executor_path = "src/gpu_fault/cluster_executor/metrics.py"
        node_path = "src/gpu_fault/node_agent/common.py"
        for source, text in (
            (executor_path, "executor = 1\n"),
            (node_path, "node = 1\n"),
        ):
            (snapshot / source).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / source).write_text(text)
        self.base = {
            "release_id": "base",
            "components": {
                "control_plane": {"wheel_sha256": "c" * 64},
                "executor": {
                    "wheel_sha256": hashlib.sha256(
                        (snapshot / executor_path).read_bytes()
                    ).hexdigest()
                },
                "node_runtime": {
                    "wheel_sha256": hashlib.sha256(
                        (snapshot / node_path).read_bytes()
                    ).hexdigest()
                },
                "node_bundle": {
                    "bundle_sha256": hashlib.sha256(
                        (snapshot / node_path).read_bytes()
                    ).hexdigest()
                },
            },
        }
        (snapshot / "dist/current-release.json").write_text(json.dumps(self.base))
        self.arguments = Namespace(
            snapshot_repo=snapshot,
            work_dir=root / "work",
            state_dir=root / "state",
            region="us-east-1",
            runtime_repository="registry.example/runtime",
            cache_repository=None,
            runtime_profile="unit-profile",
            impact_base="unit-base",
            executor_module=executor_path,
            node_module=node_path,
        )
        self.git_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.build_calls: list[dict[str, Any]] = []

        def git(
            arguments: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if arguments[0] == sys.executable:
                assert kwargs["cwd"] == snapshot
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    json.dumps(
                        {
                            "control_plane": [],
                            "executor": ["gpu_fault.cluster_executor.metrics"],
                            "node_runtime": ["gpu_fault.node_agent.common"],
                        }
                    ),
                    "",
                )
            assert arguments[0] == "git"
            if arguments[1] == "rev-parse":
                output = (
                    str(kwargs["cwd"] / ".git")
                    if arguments[-1] == "--git-common-dir"
                    else "a" * 40
                )
                return subprocess.CompletedProcess(arguments, 0, output, "")
            if arguments[1] == "clone":
                (Path(arguments[-1]) / ".git").mkdir(parents=True)
                return subprocess.CompletedProcess(arguments, 0, "", "")
            assert kwargs["cwd"].is_relative_to(self.arguments.work_dir), (
                "candidate git operations must stay inside the isolated work directory"
            )
            self.git_calls.append((arguments, kwargs))
            return subprocess.CompletedProcess(arguments, 0, "", "")

        def build(_runner: Any, **kwargs: Any) -> dict[str, Any]:
            self.build_calls.append(kwargs)
            checkout = kwargs["repository_root"]
            manifest = {
                "release_id": "candidate-" + checkout.name,
                "components": {
                    "control_plane": self.base["components"]["control_plane"],
                    **{
                        name: {
                            "wheel_sha256": hashlib.sha256(
                                (checkout / source).read_bytes()
                            ).hexdigest()
                        }
                        for name, source in (
                            ("executor", executor_path),
                            ("node_runtime", node_path),
                        )
                    },
                    "node_bundle": {
                        "bundle_sha256": hashlib.sha256(
                            (checkout / node_path).read_bytes()
                        ).hexdigest()
                    },
                },
            }
            (checkout / "dist/current-release.json").write_text(json.dumps(manifest))
            return manifest

        monkeypatch.setattr(candidates, "subprocess", SimpleNamespace(run=git))
        monkeypatch.setattr(bootstrap_common, "CommandRunner", lambda: object())
        monkeypatch.setattr(release_artifacts, "build_signed_release", build)
