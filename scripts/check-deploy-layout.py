#!/usr/bin/env python3
"""Enforce the production deployment-tree boundary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY = ROOT / "deploy"
FORBIDDEN_NAME = re.compile(
    r"(?:^|[-_.])(e2e|canary|smoke|inject|probe)(?:[-_.]|$)",
    re.IGNORECASE,
)
REMOVED_DIRECTORIES = {
    "iam",
    "kubernetes",
    "lambda",
    "optional",
    "regional",
}


def _generated_contract(deploy: Path) -> list[str]:
    generated = deploy / "control-plane" / "regional" / "generated"
    failures: list[str] = []
    index = generated / "manifest-list.txt"
    for marker in (
        generated / "README.md",
        generated / ".generated",
    ):
        if not marker.is_file():
            failures.append(f"missing generated marker: {marker}")
    if not index.is_file():
        failures.append(f"missing generated allowlist: {index}")
        return failures

    expected = [
        line.strip()
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    actual = sorted(path.name for path in generated.glob("gpu-fault-*.yaml"))
    if expected != sorted(expected):
        failures.append("generated manifest-list.txt is not sorted")
    if len(expected) != len(set(expected)):
        failures.append("generated manifest-list.txt has duplicates")
    if expected != actual:
        failures.append(
            "generated manifest set differs from manifest-list.txt: "
            f"expected={expected}, actual={actual}"
        )
    return failures


def check(deploy: Path) -> list[str]:
    failures: list[str] = []
    if not (deploy / "README.md").is_file():
        failures.append("deploy/README.md is missing")
    if deploy.resolve() == DEFAULT_DEPLOY.resolve():
        ignore = deploy.parent / ".gitignore"
        ignored = (
            set(ignore.read_text(encoding="utf-8").splitlines())
            if ignore.is_file()
            else set()
        )
        for pattern in ("__pycache__/", "*.py[cod]"):
            if pattern not in ignored:
                failures.append(f".gitignore is missing Python cache rule: {pattern}")

    for name in sorted(REMOVED_DIRECTORIES):
        if (deploy / name).exists():
            failures.append(f"obsolete deploy directory exists: {name}")

    root_files = sorted(path.name for path in deploy.iterdir() if path.is_file())
    if root_files != ["README.md"]:
        failures.append(
            "deploy root may contain only README.md: " + ", ".join(root_files)
        )

    for path in sorted(deploy.rglob("*")):
        relative = path.relative_to(deploy).as_posix()
        if path.name == "__pycache__" or path.suffix == ".pyc":
            failures.append(f"Python cache artifact under deploy: {relative}")
            continue
        if path.name == "kustomization.yaml":
            failures.append(f"deployable kustomization trap exists: {relative}")
        if path.is_file() and path.suffix == ".py" and "-" in path.name:
            failures.append(f"non-importable Python filename under deploy: {relative}")
        if path.is_file() and FORBIDDEN_NAME.search(path.name):
            failures.append(f"test-like filename under deploy: {relative}")
        if path.suffix not in {".yaml", ".yml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # ``/dev/kmsg`` at all, not ``/dev/kmsg`` in a top-level ``kind: Pod``.
        # There is no ``kind: Pod`` document anywhere under deploy -- the tree
        # is Deployments, DaemonSets, Jobs and ConfigMaps -- so the old
        # kind-scoped form never reached its append. The injection manifests it
        # is meant to keep out (scripts/e2e/regional/manifests/fault-injection/*.yaml) would
        # still be caught, but so would the same payload pasted into a
        # DaemonSet template or a Job, which is the likelier way it arrives.
        # No false positive to trade away: the production kmsg consumer is a
        # systemd unit (deploy/systemd/gpu-fault-kernel-collector.service),
        # not a manifest, so no YAML under deploy has any reason to name it.
        if "/dev/kmsg" in text:
            failures.append(f"fault-injection manifest under deploy: {relative}")
        try:
            for _ in yaml.load_all(text, Loader=yaml.BaseLoader):
                pass
        except yaml.YAMLError as exc:
            failures.append(f"invalid YAML {relative}: {exc}")

    failures.extend(_generated_contract(deploy))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=DEFAULT_DEPLOY,
    )
    args = parser.parse_args()

    failures = check(args.deploy_root)
    if failures:
        for failure in failures:
            print(f"deploy layout check failed: {failure}")
        return 1
    print("deploy layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
