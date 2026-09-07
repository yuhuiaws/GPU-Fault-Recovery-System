"""Data-plane singletons must state and enforce their replica model (S7).

A ``replicas: 1`` Deployment under the default RollingUpdate strategy still
runs two Pods for the length of a rollout, and nothing in the manifest tells
an operator whether scaling to 2 is safe. Every single-replica data-plane
Deployment therefore has to use ``Recreate`` and carry a comment that says it
is a singleton and why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATAPLANE = ROOT / "deploy" / "dataplane"


def _single_replica_deployments() -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(DATAPLANE.glob("*.yaml")):
        documents = yaml.safe_load_all(path.read_text(encoding="utf-8"))
        for document in documents:
            if not isinstance(document, dict) or document.get("kind") != "Deployment":
                continue
            spec = document.get("spec") or {}
            if spec.get("replicas") == 1:
                found.append((path, document))
    return found


SINGLETONS = _single_replica_deployments()


def test_predicate_finds_the_known_singletons() -> None:
    names = sorted(path.name for path, _ in SINGLETONS)

    assert names == [
        "completion-watcher.yaml",
        "kubernetes-node-resource-collector.yaml",
        "node-installer-reconciler.yaml",
    ]


@pytest.mark.parametrize(
    "path, deployment", SINGLETONS, ids=[path.name for path, _ in SINGLETONS]
)
def test_single_replica_deployment_recreates_and_explains_itself(
    path: Path, deployment: dict[str, Any]
) -> None:
    where = str(path.relative_to(ROOT))
    strategy = (deployment.get("spec") or {}).get("strategy") or {}

    assert strategy.get("type") == "Recreate", (
        f"{where}: a replicas: 1 Deployment must not overlap during a rollout"
    )
    text = path.read_text(encoding="utf-8")
    comment = _comment_above_strategy(text)
    assert comment, f"{where}: explain the Recreate strategy in a comment above it"
    assert "singleton" in comment.lower(), (
        f"{where}: state in the comment above strategy that this component is a "
        "singleton"
    )


def _comment_above_strategy(text: str) -> str:
    lines = text.splitlines()
    index = next(
        (i for i, line in enumerate(lines) if line.strip() == "strategy:"), None
    )
    if index is None:
        return ""
    collected: list[str] = []
    for line in reversed(lines[:index]):
        if not line.strip().startswith("#"):
            break
        collected.append(line.strip().lstrip("#").strip())
    return "\n".join(reversed(collected))
