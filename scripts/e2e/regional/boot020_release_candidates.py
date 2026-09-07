"""Produce the five chained release configs GF-REGIONAL-BOOT-020 consumes.

The BOOT-020 driver (``run_boot020_release_rolling.py``) only takes finished
config files; the case text says "prepare five chained configs" and stops
there. This tool is the missing preparation step, built the way the live run
on 2026-09-07 was actually done against the production site:

``build``
    Copy the *deployed* source snapshot three times, append a comment to one
    executor-only module (candidate B), additionally to one node-runtime-only
    module (C) and once more to both (D), commit each copy locally and build
    it with ``gpu_fault.admin.release_artifacts.build_signed_release`` -- the
    same staging path ``gpu-fault-admin`` uses -- so every candidate is a
    signed release whose control-plane wheel is byte-identical to the live one.

``configs``
    Materialise the site's release config and derive the chain: ``noop`` points
    at the deployed snapshot's manifest; ``control-plane`` raises
    ``capacity.control_worker_replicas`` by one with the admin-config digests
    recomputed; ``data-plane`` swaps in candidate B; ``agent`` candidate C;
    ``full`` candidate D plus a Runtime Profile version suffix. The rollout
    resolves manifest wheel paths (``dist/<release_id>/...``) relative to the
    repository root of the deploy module, so each candidate's ``dist``
    directory is linked there as well.

``check``
    Load every config the way the rollout does and classify it against the
    live state, printing the kind and changed keys -- run this before
    ``--execute``.

``clean``
    Remove the ``dist/<release_id>`` symlinks ``configs`` left in the
    repository so the work tree is back to what it was before the run.

Nothing site-specific is hard-coded: ECR repositories, paths and the site
file all come from the command line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.config import AdminConfig  # noqa: E402

CANDIDATES = ("B", "C", "D")
CONFIG_NAMES = ("noop", "control-plane", "data-plane", "agent", "full")
DRIVER_KEYS = {
    "noop": "noop",
    "control-plane": "control_plane",
    "data-plane": "executor",
    "agent": "agent",
    "full": "full",
}
DEFAULT_EXECUTOR_MODULE = "src/gpu_fault/transport_errors.py"
DEFAULT_NODE_MODULE = "src/gpu_fault/node_agent/common.py"


def candidate_edits(
    executor_module: str,
    node_module: str,
) -> dict[str, list[tuple[str, str]]]:
    """Which module gets which appended comment for each candidate."""

    b_exec = "# BOOT-020 candidate B: executor-only physical change\n"
    c_node = "# BOOT-020 candidate C: node-runtime physical change\n"
    d_exec = b_exec + "# BOOT-020 candidate D: second executor change\n"
    d_node = c_node + "# BOOT-020 candidate D: second node-runtime change\n"
    return {
        "B": [(executor_module, b_exec)],
        "C": [(executor_module, b_exec), (node_module, c_node)],
        "D": [(executor_module, d_exec), (node_module, d_node)],
    }


def edits_digest(edits: list[tuple[str, str]]) -> str:
    """The sha256 of a candidate's module edits, path and text included.

    Recorded beside each candidate so a later ``build`` can tell whether the
    build sitting in ``wt-<name>`` came from the same edits; a different
    ``--executor-module`` or edit text with the same base release must not be
    mistaken for a reusable build.
    """

    digest = hashlib.sha256()
    for relative, text in edits:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def reusable_candidate(
    manifest_path: Path,
    *,
    base_release_id: str,
    recorded: dict[str, Any] | None,
    expected_edits_digest: str,
) -> bool:
    """Whether an existing candidate build can stand in for a fresh one.

    It can only when its manifest exists and names a release other than the
    base (so it is a real rebuild), and ``candidates.json`` recorded the same
    edits digest for it -- otherwise the checkout is rebuilt.
    """

    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("release_id") in (None, base_release_id):
        return False
    if not isinstance(recorded, dict):
        return False
    return recorded.get("edits_sha256") == expected_edits_digest and recorded.get(
        "release_id"
    ) == manifest.get("release_id")


def component_summary(manifest: dict[str, Any]) -> dict[str, str]:
    components = manifest.get("components") or {}
    return {
        name: str(value.get("wheel_sha256") or value.get("bundle_sha256") or "")[:12]
        for name, value in components.items()
    }


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def build_candidates(arguments: argparse.Namespace) -> dict[str, Any]:
    from gpu_fault.admin.bootstrap_common import CommandRunner
    from gpu_fault.admin.release_artifacts import build_signed_release

    snapshot = arguments.snapshot_repo.resolve()
    base = json.loads((snapshot / "dist/current-release.json").read_text("utf-8"))
    work = arguments.work_dir.resolve()
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "base_release_id": base["release_id"],
        "base_components": component_summary(base),
        "candidates": {},
    }
    summary_path = work / "candidates.json"
    previous: dict[str, Any] = {}
    if summary_path.exists():
        try:
            previous = dict(json.loads(summary_path.read_text("utf-8")))
        except (OSError, json.JSONDecodeError):
            previous = {}
    recorded_candidates = (
        previous.get("candidates")
        if previous.get("base_release_id") == base["release_id"]
        else {}
    ) or {}
    edits = candidate_edits(arguments.executor_module, arguments.node_module)
    for name in CANDIDATES:
        checkout = work / f"wt-{name}"
        manifest_path = checkout / "dist/current-release.json"
        digest = edits_digest(edits[name])
        if reusable_candidate(
            manifest_path,
            base_release_id=base["release_id"],
            recorded=recorded_candidates.get(name),
            expected_edits_digest=digest,
        ):
            print(f"[{name}] reusing existing build in {checkout}", flush=True)
        else:
            if checkout.exists():
                shutil.rmtree(checkout)
            shutil.copytree(snapshot, checkout, symlinks=True)
            for relative, text in edits[name]:
                target = checkout / relative
                target.write_text(target.read_text("utf-8") + "\n" + text, "utf-8")
            _git(checkout, "add", "-A")
            _git(
                checkout,
                "-c",
                "user.name=boot020",
                "-c",
                "user.email=boot020@acceptance.invalid",
                "commit",
                "-q",
                "-m",
                f"BOOT-020 candidate {name}",
            )
            started = time.monotonic()
            release = build_signed_release(
                CommandRunner(),
                repository_root=checkout,
                state_dir=arguments.state_dir.resolve(),
                region=arguments.region,
                runtime_repository=arguments.runtime_repository,
                cache_repository=arguments.cache_repository,
                runtime_profile=arguments.runtime_profile,
                staging_only=True,
                impact_base=arguments.impact_base,
            )
            print(
                f"[{name}] built in {time.monotonic() - started:.0f}s: "
                f"release_id={release.get('release_id')}",
                flush=True,
            )
        manifest = json.loads(manifest_path.read_text("utf-8"))
        summary["candidates"][name] = {
            "release_id": manifest["release_id"],
            "components": component_summary(manifest),
            "manifest": str(manifest_path),
            "edits_sha256": digest,
        }
        summary_path.write_text(json.dumps(summary, indent=1), "utf-8")
    return summary


def with_control_worker_replicas(config: dict[str, Any], delta: int) -> dict[str, Any]:
    """Return a copy whose worker admin-config digest differs from the input."""

    value = copy.deepcopy(config)
    admin = value["admin_config"]["config"]
    admin["capacity"]["control_worker_replicas"] = (
        int(admin["capacity"]["control_worker_replicas"]) + delta
    )
    parsed = AdminConfig.from_mapping(admin)
    value["admin_config"] = {
        **value["admin_config"],
        "config": parsed.as_dict(),
        "config_sha256": parsed.sha256(),
        "role_sha256": parsed.role_sha256(),
    }
    return value


def with_manifest(config: dict[str, Any], manifest: str) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["release"]["manifest"] = manifest
    return value


def chain_configs(
    base: dict[str, Any],
    *,
    live_manifest: str,
    candidate_manifests: dict[str, str],
    replicas_delta: int = 1,
    profile_suffix: str = "-boot020",
    full_agent_config_digest: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The five configs, each one step past the previous.

    ``full_agent_config_digest`` is the Agent config digest for the suffixed
    Runtime Profile version. Agents digest their allowed operations *and* the
    profile version they run, and the control plane waits for heartbeats to
    report the pinned digest, so a FULL config that changes the version but
    keeps the old digest can never converge (live 2026-09-07: the first FULL
    wave installed fine and then waited 15 minutes for a digest that no agent
    would ever report). The CLI derives it with the same function
    gpu-fault-admin uses; tests may pass any string.
    """

    noop = with_manifest(base, live_manifest)
    control = with_control_worker_replicas(noop, replicas_delta)
    data_plane = with_manifest(control, candidate_manifests["B"])
    agent = with_manifest(data_plane, candidate_manifests["C"])
    full = with_manifest(agent, candidate_manifests["D"])
    full["runtime_profile"]["version"] = (
        str(base["runtime_profile"]["version"]) + profile_suffix
    )
    if full_agent_config_digest:
        full["release"]["agent_config_digest"] = full_agent_config_digest
    return {
        "noop": noop,
        "control-plane": control,
        "data-plane": data_plane,
        "agent": agent,
        "full": full,
    }


def link_release_dist(repository_root: Path, manifest_path: Path) -> Path:
    """Make ``<repo>/dist/<release_id>`` resolve to the manifest's artifacts."""

    manifest = json.loads(manifest_path.read_text("utf-8"))
    release_id = str(manifest["release_id"])
    source = manifest_path.parent / release_id
    link = repository_root / "dist" / release_id
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == source.resolve():
            return link
        if link.is_symlink():
            link.unlink()
        else:
            raise SystemExit(f"{link} exists and is not a symlink; refusing")
    link.symlink_to(source.resolve())
    return link


def unlink_release_dist(repository_root: Path, manifest_path: Path) -> Path | None:
    """Remove the ``dist/<release_id>`` link ``link_release_dist`` created.

    Only a symlink is removed, and only when it points at the manifest's own
    artifact directory; a real directory or a link to somewhere else is left
    alone and reported, since neither was made by this tool.
    """

    manifest = json.loads(manifest_path.read_text("utf-8"))
    release_id = str(manifest["release_id"])
    source = manifest_path.parent / release_id
    link = repository_root / "dist" / release_id
    if not link.is_symlink():
        if link.exists():
            print(f"{link} is not a symlink; left in place", flush=True)
        return None
    if link.resolve() != source.resolve():
        print(f"{link} points elsewhere; left in place", flush=True)
        return None
    link.unlink()
    return link


def clean_release_dist(arguments: argparse.Namespace) -> int:
    """``clean``: drop every symlink ``configs`` planted under ``<repo>/dist``."""

    candidates = json.loads(
        (arguments.work_dir.resolve() / "candidates.json").read_text("utf-8")
    )["candidates"]
    manifests = [
        arguments.snapshot_repo.resolve() / "dist/current-release.json",
        *(Path(c["manifest"]) for c in candidates.values()),
    ]
    removed = [
        link
        for link in (unlink_release_dist(ROOT, manifest) for manifest in manifests)
        if link is not None
    ]
    print("removed:", ", ".join(str(link) for link in removed) or "nothing", flush=True)
    return 0


def write_configs(arguments: argparse.Namespace) -> None:
    from gpu_fault.admin.site import load_site

    site = load_site(arguments.site.resolve(), repository_root=ROOT)
    base = json.loads(json.dumps(site.release_config, default=str))
    candidates = json.loads(
        (arguments.work_dir.resolve() / "candidates.json").read_text("utf-8")
    )["candidates"]
    live_manifest = arguments.snapshot_repo.resolve() / "dist/current-release.json"
    from gpu_fault.admin.bootstrap_common import (
        CommandRunner,
        compute_agent_config_digest,
    )

    full_version = str(base["runtime_profile"]["version"]) + arguments.profile_suffix
    full_agent_config_digest = compute_agent_config_digest(
        CommandRunner(),
        repository_root=Path(candidates["D"]["manifest"]).parent.parent,
        runtime_profile_version=full_version,
    )
    configs = chain_configs(
        base,
        live_manifest=str(live_manifest),
        candidate_manifests={name: candidates[name]["manifest"] for name in CANDIDATES},
        replicas_delta=arguments.replicas_delta,
        profile_suffix=arguments.profile_suffix,
        full_agent_config_digest=full_agent_config_digest,
    )
    out = arguments.out_dir.resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, config in configs.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True), "utf-8")
        path.chmod(0o600)
    for manifest in (
        live_manifest,
        *(Path(c["manifest"]) for c in candidates.values()),
    ):
        link_release_dist(ROOT, manifest)
    print("written:", ", ".join(f"{name}.json" for name in configs), flush=True)


def check_configs(arguments: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "scripts/e2e/regional"))
    os.environ.setdefault("KUBECONFIG", str(arguments.gpu_kubeconfig.resolve()))
    from run_boot020_release_rolling import LiveReleaseRollingBackend

    out = arguments.out_dir.resolve()
    backend = LiveReleaseRollingBackend(
        {DRIVER_KEYS[name]: out / f"{name}.json" for name in CONFIG_NAMES}
    )
    failures = 0
    for name in CONFIG_NAMES:
        try:
            diff = backend.classify(DRIVER_KEYS[name])
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            failures += 1
            print(f"{name} -> ERROR {type(exc).__name__}: {str(exc)[:300]}")
            continue
        print(f"{name} -> {diff['kind']} {sorted(diff.get('changed') or [])}")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = value.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build candidates B, C and D")
    build.add_argument("--snapshot-repo", type=Path, required=True)
    build.add_argument("--work-dir", type=Path, required=True)
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--region", required=True)
    build.add_argument("--runtime-repository", required=True)
    build.add_argument("--cache-repository", default=None)
    build.add_argument("--runtime-profile", default="hyperpod-v1")
    build.add_argument("--impact-base", default="origin/main")
    build.add_argument("--executor-module", default=DEFAULT_EXECUTOR_MODULE)
    build.add_argument("--node-module", default=DEFAULT_NODE_MODULE)

    configs = commands.add_parser("configs", help="write the five chained configs")
    configs.add_argument("--site", type=Path, required=True)
    configs.add_argument("--snapshot-repo", type=Path, required=True)
    configs.add_argument("--work-dir", type=Path, required=True)
    configs.add_argument("--out-dir", type=Path, required=True)
    configs.add_argument("--replicas-delta", type=int, default=1)
    configs.add_argument("--profile-suffix", default="-boot020")

    check = commands.add_parser("check", help="classify every config against live")
    check.add_argument("--out-dir", type=Path, required=True)
    check.add_argument("--gpu-kubeconfig", type=Path, required=True)

    clean = commands.add_parser("clean", help="remove the dist/<release_id> links")
    clean.add_argument("--snapshot-repo", type=Path, required=True)
    clean.add_argument("--work-dir", type=Path, required=True)
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "build":
        print(json.dumps(build_candidates(arguments), indent=1))
        return 0
    if arguments.command == "configs":
        write_configs(arguments)
        return 0
    if arguments.command == "clean":
        return clean_release_dist(arguments)
    return check_configs(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
