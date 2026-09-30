"""BOOT-020's release-candidate precondition as code the runner can satisfy.

The case needs five chained release configs built against the *deployed*
source snapshot (``boot020_release_candidates.py build`` / ``configs`` /
``check``). Until 2026-09-30 they were an operator input: the runner refused
when a path was missing, the campaign driver could only pre-check the files,
and every redeploy of the site silently made yesterday's candidates stale
(round 4 lost BOOT-020 and BOOT-023 to exactly that). Everything the tool
needs is in the administrator state directory, so the runner derives it and
builds when the candidates are missing or bound to another release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.e2e.regional import boot020_release_candidates as candidates
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CANDIDATES_DIR_NAME = "boot020-releases"
WORK_DIR_NAME = "boot020-work"
# The build summary ``boot020_release_candidates.py build`` keeps in the work
# directory, copied beside the five configs so a later run that reuses them can
# read the per-candidate release ids and edits digests without the work dir.
CANDIDATES_METADATA_NAME = "candidates.json"
# Keys of the ``ensure_release_candidates`` record that describe *this* run's
# work rather than the candidates themselves; never part of the evidence
# identity (see ``candidate_identity``).
VOLATILE_RECORD_KEYS = frozenset({"action", "work_dir", "replicas_delta"})
CONFIG_FILES = {
    "noop": "noop.json",
    "control_plane": "control-plane.json",
    "executor": "data-plane.json",
    "agent": "agent.json",
    "full": "full.json",
}
CACHE_REPOSITORY_INFIX = ("/gpu-fault/runtime-", "/gpu-fault/runtime-cache-")


@dataclass(frozen=True)
class CandidateInputs:
    snapshot_repo: Path
    region: str
    runtime_repository: str
    cache_repository: str | None
    runtime_profile: str
    site_file: Path


def site_spec(state_dir: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped,unused-ignore]

    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        raise RegionalFixtureError(f"administrator state has no site.yaml: {state_dir}")
    document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    spec = document.get("spec") if isinstance(document, dict) else None
    if not isinstance(spec, dict):
        raise RegionalFixtureError(f"site.yaml has no spec: {site_file}")
    return spec


def live_release_id(state_dir: Path) -> str:
    record = state_dir / "source-deploy-success.json"
    if not record.is_file():
        raise RegionalFixtureError(
            f"administrator state has no source-deploy-success.json: {state_dir}"
        )
    live = json.loads(record.read_text(encoding="utf-8")).get("live") or {}
    release_id = str(live.get("release_id") or "")
    if not release_id:
        raise RegionalFixtureError("source-deploy-success.json names no live release")
    return release_id


def live_snapshot_repository(state_dir: Path) -> Path:
    """The deployed source snapshot's repository root.

    ``source-deploy-success.json`` records ``prepared_repository_root`` -- the
    isolated copy the last deploy built from -- and it must still hold
    ``dist/current-release.json``. Without that record, ``spec.repositoryRoot``
    in site.yaml is the repository itself (or, for older sites, the snapshot
    directory whose single ``repository-*`` child is). The live release id is
    deliberately not compared: ``gpu-fault-admin config`` releases move it
    while the deployed source stays the same.
    """

    record = state_dir / "source-deploy-success.json"
    candidates: list[Path] = []
    if record.is_file():
        prepared = json.loads(record.read_text(encoding="utf-8")).get(
            "prepared_repository_root"
        )
        if isinstance(prepared, str) and prepared:
            candidates.append(Path(prepared).expanduser())
    root = Path(str(site_spec(state_dir).get("repositoryRoot") or "")).expanduser()
    if root.is_dir():
        candidates.append(root)
        candidates.extend(sorted(root.glob("repository-*")))
    for candidate in candidates:
        if (candidate / "dist" / "current-release.json").is_file():
            return candidate.resolve()
    raise RegionalFixtureError(
        "no deployed source snapshot with dist/current-release.json found for "
        f"{state_dir} (looked at {', '.join(str(c) for c in candidates) or 'nothing'})"
    )


def snapshot_release_id(snapshot_repo: Path) -> str:
    manifest = json.loads(
        (snapshot_repo / "dist" / "current-release.json").read_text(encoding="utf-8")
    )
    return str(manifest.get("release_id") or "")


def candidate_inputs(state_dir: Path) -> CandidateInputs:
    from gpu_fault.admin.site import load_site

    spec = site_spec(state_dir)
    images = spec.get("images") if isinstance(spec.get("images"), dict) else {}
    runtime_image = str(images.get("runtime") or "")
    if "@sha256:" not in runtime_image:
        raise RegionalFixtureError("site.yaml spec.images.runtime is not digest-pinned")
    repository = runtime_image.split("@", 1)[0]
    cache = (
        repository.replace(*CACHE_REPOSITORY_INFIX, 1)
        if CACHE_REPOSITORY_INFIX[0] in repository
        else None
    )
    site_file = state_dir / "site.yaml"
    site = load_site(site_file, repository_root=candidates.ROOT)
    profile = str(site.release_config["runtime_profile"]["version"])
    region = str(spec.get("awsRegion") or "")
    if not region:
        raise RegionalFixtureError("site.yaml spec.awsRegion is missing")
    return CandidateInputs(
        snapshot_repo=live_snapshot_repository(state_dir),
        region=region,
        runtime_repository=repository,
        cache_repository=cache,
        runtime_profile=profile,
        site_file=site_file,
    )


def candidate_paths(out_dir: Path) -> dict[str, Path]:
    return {key: out_dir / name for key, name in CONFIG_FILES.items()}


def site_inputs(state_dir: Path) -> dict[str, str]:
    """Digests of the site inputs the candidate derivation reads.

    ``configs`` materialises the site through ``load_site``: ``site.yaml`` and
    the desired AdminConfig (``admin-config/desired.json`` beside it, or the
    release defaults before one exists). A candidate set is only as current as
    those two; the live release id says nothing about them (live 2026-09-30
    19:01Z: candidates built while the desired config still carried a
    candidate's 5 workers were reused after the operator re-applied the site's
    6, and the ``noop`` candidate classified CONTROL_PLANE_ONLY).
    """

    from gpu_fault.admin.config import load_desired_admin_config

    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        raise RegionalFixtureError(f"administrator state has no site.yaml: {state_dir}")
    return {
        "site_yaml_sha256": hashlib.sha256(site_file.read_bytes()).hexdigest(),
        "admin_config_sha256": load_desired_admin_config(state_dir).sha256(),
    }


def recorded_site_inputs(out_dir: Path) -> dict[str, str] | None:
    """The ``site_inputs`` the candidates directory's metadata recorded, if any."""

    path = out_dir / CANDIDATES_METADATA_NAME
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    recorded = summary.get("site_inputs") if isinstance(summary, dict) else None
    if not isinstance(recorded, dict):
        return None
    return {str(key): str(value) for key, value in recorded.items()}


def candidates_bound_to(
    out_dir: Path,
    snapshot_repo: Path,
    site: dict[str, str] | None = None,
) -> bool:
    """Whether the candidates may stand in for a fresh derivation.

    The five configs must exist and the NOOP one must name this snapshot's
    manifest. With ``site`` (``site_inputs``) given, the metadata written at
    build time must also record exactly those digests: a directory without the
    record (built before it existed) or with other digests is rebuilt.
    """

    paths = candidate_paths(out_dir)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        noop = json.loads(paths["noop"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    manifest = str((noop.get("release") or {}).get("manifest") or "")
    expected = str((snapshot_repo / "dist" / "current-release.json").resolve())
    if not manifest or str(Path(manifest).resolve()) != expected:
        return False
    return site is None or recorded_site_inputs(out_dir) == site


def ensure_release_candidates(
    state_dir: Path,
    out_dir: Path,
    *,
    gpu_kubeconfig: Path,
    work_dir: Path | None = None,
    replicas_delta: int = -1,
    build: Callable[[argparse.Namespace], Any] = candidates.build_candidates,
    write: Callable[[argparse.Namespace], Any] = candidates.write_configs,
    check: Callable[[argparse.Namespace], int] = candidates.check_configs,
    inputs: Callable[[Path], CandidateInputs] = candidate_inputs,
) -> dict[str, Any]:
    """Make the five chained configs exist for the live release; return a record.

    Reuses configs already bound to the live snapshot; otherwise runs the
    candidate tool's ``build`` -> ``configs`` -> ``check`` in-process with the
    inputs derived from the administrator state. The record goes into the
    plan/evidence so a reader can tell reuse from a fresh build.
    """

    derived = inputs(state_dir)
    site = site_inputs(state_dir)
    record: dict[str, Any] = {
        "candidates_dir": str(out_dir),
        "snapshot_repo": str(derived.snapshot_repo),
        "live_release_id": live_release_id(state_dir),
        "snapshot_release_id": snapshot_release_id(derived.snapshot_repo),
        "runtime_repository": derived.runtime_repository,
        "runtime_profile": derived.runtime_profile,
        "replicas_delta": replicas_delta,
    }
    if candidates_bound_to(out_dir, derived.snapshot_repo, site):
        return {**record, "action": "reused"}
    work = work_dir or (state_dir / WORK_DIR_NAME)
    build(
        argparse.Namespace(
            snapshot_repo=derived.snapshot_repo,
            work_dir=work,
            state_dir=state_dir,
            region=derived.region,
            runtime_repository=derived.runtime_repository,
            cache_repository=derived.cache_repository,
            runtime_profile=derived.runtime_profile,
            impact_base="HEAD",
            executor_module=candidates.DEFAULT_EXECUTOR_MODULE,
            node_module=candidates.DEFAULT_NODE_MODULE,
        )
    )
    write(
        argparse.Namespace(
            site=derived.site_file,
            snapshot_repo=derived.snapshot_repo,
            work_dir=work,
            out_dir=out_dir,
            replicas_delta=replicas_delta,
            profile_suffix="-boot020",
        )
    )
    # The metadata carries the site inputs this build read; written before the
    # final binding check so that check judges the directory a later run sees.
    write_candidate_metadata(work, out_dir, site=site)
    failures = check(argparse.Namespace(out_dir=out_dir, gpu_kubeconfig=gpu_kubeconfig))
    if failures:
        raise RegionalFixtureError(
            "freshly built BOOT-020 release candidates do not classify against live"
        )
    if not candidates_bound_to(out_dir, derived.snapshot_repo, site):
        raise RegionalFixtureError(
            "BOOT-020 release candidates were built but do not name the live snapshot"
        )
    return {**record, "action": "built", "work_dir": str(work)}


def write_candidate_metadata(
    work: Path, out_dir: Path, *, site: dict[str, str]
) -> Path:
    """Keep the build summary and the site inputs next to the configs.

    ``build`` writes ``candidates.json`` (release id, edits digest and component
    summary per candidate) into the work directory, which a later run does not
    know about and an operator may clean. The copy in the candidates directory,
    extended with ``site_inputs`` (``site_inputs``), is what the reuse decision
    and ``candidate_identity`` read on every run, built or reused.
    """

    summary: dict[str, Any] = {}
    source = work / CANDIDATES_METADATA_NAME
    if source.is_file():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            summary = loaded
    target = out_dir / CANDIDATES_METADATA_NAME
    target.write_text(
        json.dumps({**summary, "site_inputs": site}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    target.chmod(0o600)
    return target


def _config_manifest_release_id(name: str, path: Path) -> str:
    """The release id of the manifest a chained config points at (fail closed)."""

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegionalFixtureError(
            f"BOOT-020 {name} config is unreadable: {path} ({exc})"
        ) from exc
    manifest_path = str((config.get("release") or {}).get("manifest") or "")
    if not manifest_path:
        raise RegionalFixtureError(f"BOOT-020 {name} config names no manifest: {path}")
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegionalFixtureError(
            f"BOOT-020 {name} config points at an unreadable manifest: "
            f"{manifest_path} ({exc}); rebuild the candidates or restore the "
            "work directory before planning again"
        ) from exc
    release_id = str(manifest.get("release_id") or "")
    if not release_id:
        raise RegionalFixtureError(
            f"BOOT-020 {name} manifest names no release_id: {manifest_path}"
        )
    return release_id


def _recorded_candidate_metadata(candidates_dir: Path) -> dict[str, dict[str, str]]:
    """``{B: {release_id, edits_sha256}, ...}`` from the candidates directory."""

    path = candidates_dir / CANDIDATES_METADATA_NAME
    if not path.is_file():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    recorded = summary.get("candidates") if isinstance(summary, dict) else None
    if not isinstance(recorded, dict):
        return {}
    return {
        str(name): {
            "release_id": str(value.get("release_id") or ""),
            "edits_sha256": str(value.get("edits_sha256") or ""),
        }
        for name, value in sorted(recorded.items())
        if isinstance(value, dict)
    }


def candidate_identity(
    state_dir: Path, configs: dict[str, Path], record: dict[str, Any]
) -> dict[str, Any]:
    """What identifies the candidates a run consumes, built or reused alike.

    The evidence document's ``inputs`` must be equal on every attempt of one
    run, or ``EvidenceRecorder`` refuses to resume. The ``ensure`` record is
    not that: it says ``built`` with a ``work_dir`` on the first attempt and
    ``reused`` on the next (live 2026-09-30, fifth regional run: every
    ``--resume`` died on exactly this). This identity is computed from the
    administrator state and the candidate files themselves, never from what
    this attempt had to do for them:

    - ``source``: ``explicit`` when the five paths were given, else ``derived``.
      Explicit configs are the operator's and opaque to the runner beyond
      existing and passing the admin-target check (the entrypoint contract);
      their identity is ``config_sha256`` alone and nothing below is read;
    - ``candidates_dir`` and ``snapshot_repo`` (derived candidates);
    - ``live_release_id`` (``source-deploy-success.json``, written only by
      ``gpu-fault-admin deploy``, so a BOOT-020 release does not move it) and
      ``snapshot_release_id`` (the snapshot's ``dist/current-release.json``);
    - ``release_ids``: per config, the release id of the manifest it names;
    - ``candidates``: per candidate B/C/D, release id and edits digest from the
      ``candidates.json`` copied into the candidates directory when present;
    - ``site_inputs``: the ``site.yaml`` and desired-AdminConfig digests the
      derivation read (``site_inputs``), so a resume after the operator changed
      either is refused rather than run against candidates that no longer
      describe the site;
    - ``runtime_profile``: the noop config's Runtime Profile version.

    The config bytes themselves are bound separately (``config_sha256``).
    """

    if record.get("action") == "explicit":
        return {"source": "explicit"}
    identity: dict[str, Any] = {
        "source": "derived",
        "live_release_id": live_release_id(state_dir),
    }
    if record.get("candidates_dir"):
        identity["candidates_dir"] = str(record["candidates_dir"])
    snapshot = record.get("snapshot_repo")
    if snapshot:
        identity["snapshot_repo"] = str(snapshot)
        identity["snapshot_release_id"] = snapshot_release_id(Path(str(snapshot)))
    identity["release_ids"] = {
        name: _config_manifest_release_id(name, path)
        for name, path in sorted(configs.items())
    }
    if record.get("candidates_dir"):
        identity["candidates"] = _recorded_candidate_metadata(
            Path(str(record["candidates_dir"]))
        )
    identity["site_inputs"] = site_inputs(state_dir)
    noop = json.loads(configs["noop"].read_text(encoding="utf-8"))
    identity["runtime_profile"] = str(
        (noop.get("runtime_profile") or {}).get("version") or ""
    )
    return identity


def candidate_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """The per-attempt facts the identity deliberately leaves out.

    ``action`` (``built`` / ``reused`` / ``explicit``), the work directory and
    the replicas delta this attempt would have built with; recorded as an
    evidence note so the reader still learns what each attempt did.
    """

    return {key: record[key] for key in sorted(VOLATILE_RECORD_KEYS) if key in record}


def resolve_release_configs(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """The five config paths and what the run must still do for them.

    Explicit paths win (the documented manual route). Otherwise the candidates
    directory is judged against the live snapshot: ``reused`` when it already
    holds configs bound to it, ``build-at-execute`` when not. Nothing is built
    here -- a plan must not sign and push images, and an execute run builds
    only after ``authorize_execution`` admitted it (see ``ensure_for_execute``).
    """

    explicit = {
        "noop": arguments.noop_config,
        "control_plane": arguments.control_plane_config,
        "executor": arguments.data_plane_config,
        "agent": arguments.agent_config,
        "full": arguments.full_config,
    }
    if all(path is not None for path in explicit.values()):
        return {name: path.resolve() for name, path in explicit.items()}, {
            "action": "explicit"
        }
    state_dir = arguments.admin_state_dir.resolve()
    out_dir = (
        arguments.release_candidates_dir.resolve()
        if arguments.release_candidates_dir is not None
        else state_dir / CANDIDATES_DIR_NAME
    )
    derived = candidate_paths(out_dir)
    snapshot = live_snapshot_repository(state_dir)
    record = {
        "candidates_dir": str(out_dir),
        "snapshot_repo": str(snapshot),
        "action": (
            "reused"
            if candidates_bound_to(out_dir, snapshot, site_inputs(state_dir))
            else "build-at-execute"
        ),
    }
    return {
        name: (path.resolve() if path is not None else derived[name])
        for name, path in explicit.items()
    }, record


def ensure_for_execute(
    arguments: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    """After authorization: build the pending candidates and return the record."""

    if record.get("action") != "build-at-execute":
        return record
    return ensure_release_candidates(
        arguments.admin_state_dir.resolve(),
        Path(record["candidates_dir"]),
        gpu_kubeconfig=(
            arguments.gpu_kubeconfig.resolve()
            if arguments.gpu_kubeconfig is not None
            else Path(os.environ["KUBECONFIG"])
        ),
        replicas_delta=arguments.replicas_delta,
    )
