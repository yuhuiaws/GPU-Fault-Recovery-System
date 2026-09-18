"""Physical and file identity contract for the sacrificial uninstall case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gpu_fault.admin import deploy_host_binding
from gpu_fault.admin.bootstrap_common import Arn
from gpu_fault.admin.site import RenderedSite, load_site
from scripts.e2e.regional.live_driver_guard import details_sha256

CASE_ID = "GF-REGIONAL-BOOT-032"
CONFIRMATION = "BOOT032_DELETE_SACRIFICIAL_CONTROL_PLANE"
RESTART_EXIT = 75
CASE_TAG = "gpu-fault:acceptance-case"
FIXTURE_TAG = "gpu-fault:acceptance-fixture"
ROOT = Path(__file__).resolve().parents[3]


class UninstallCaseError(RuntimeError):
    """Unproved scope, ownership, cleanup or restart evidence."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise UninstallCaseError(message)


def mapping(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UninstallCaseError(message)
    return cast(dict[str, Any], value)


def checked_path(path: Path, *, private: bool = True, directory: bool = False) -> Path:
    path = path.expanduser().absolute()
    require(".." not in path.parts, "path contains parent traversal")
    require(
        not any(item.is_symlink() for item in (path, *path.parents)),
        "symbolic links cannot identify uninstall inputs",
    )
    metadata = path.stat()
    require(
        stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode),
        "uninstall input has an unexpected file type",
    )
    require(metadata.st_uid == os.geteuid(), "uninstall input belongs to another user")
    require(
        not metadata.st_mode & (0o077 if private else 0o022),
        "uninstall input permissions are not sufficiently restricted",
    )
    return path.resolve()


def file_digest(path: Path, *, private: bool = True) -> str:
    with checked_path(path, private=private).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_document(path: Path) -> dict[str, Any]:
    value = json.loads(checked_path(path).read_text(encoding="utf-8"))
    return mapping(value, "uninstall document must be a JSON object")


def kubeconfig(site: RenderedSite, plane: str) -> Path:
    key = "cpu_kubeconfig" if plane == "cpu" else "gpu_kubeconfig"
    value = site.release_config.get(key)
    if plane == "gpu" and not value:
        value = site.environment.get("KUBECONFIG")
    if not isinstance(value, str) or not value:
        raise UninstallCaseError(f"{plane} kubeconfig is not explicit")
    return checked_path(Path(value))


def cluster_specs(site: RenderedSite) -> list[dict[str, str]]:
    config = site.release_config
    rows = [
        {
            "plane": "cpu",
            "context": "cpu",
            "eks_arn": config["cpu_eks_arn"],
            "hyperpod_name": config["cpu_hyperpod_cluster_name"],
        }
    ]
    clusters = config.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise UninstallCaseError("site requires explicit GPU clusters")
    for item in clusters:
        require(isinstance(item, dict), "GPU cluster configuration is malformed")
        rows.append(
            {
                "plane": "gpu",
                "context": item["context"],
                "cluster_id": item["cluster_id"],
                "eks_arn": item["eks_cluster_arn"],
                "hyperpod_name": item["hyperpod_cluster_name"],
            }
        )
    require(
        all(isinstance(value, str) and value for row in rows for value in row.values()),
        "cluster identity is incomplete",
    )
    require(
        len({row["context"] for row in rows}) == len(rows), "cluster contexts alias"
    )
    for row in rows:
        arn = Arn.parse(row["eks_arn"])
        require(
            arn.service == "eks"
            and arn.resource.startswith("cluster/")
            and arn.region == config["aws_region"],
            "cluster ARN is not an EKS cluster in the site Region",
        )
        row["eks_name"] = arn.resource_name
        row["account"] = arn.account
        row["region"] = arn.region
    require(
        len({row["eks_arn"] for row in rows}) == len(rows)
        and len({row["hyperpod_name"] for row in rows}) == len(rows)
        and len({row["account"] for row in rows}) == 1,
        "CPU/GPU physical cluster identities overlap or cross accounts",
    )
    return rows


def site_inputs(site: RenderedSite) -> dict[str, Any]:
    return {
        "path": str(checked_path(site.source)),
        "site_sha256": file_digest(site.source),
        "site_id": site.metadata_name,
        "repository_root": str(site.repository_root.resolve()),
        "config_sha256": details_sha256(site.release_config),
        "environment_sha256": details_sha256(site.environment),
        "cpu_kubeconfig_sha256": file_digest(kubeconfig(site, "cpu")),
        "gpu_kubeconfig_sha256": file_digest(kubeconfig(site, "gpu")),
        "manifest_sha256": file_digest(
            Path(site.release_config["release"]["manifest"]), private=False
        ),
        "clusters": cluster_specs(site),
    }


@dataclass(frozen=True)
class Settings:
    run_dir: Path
    fixture_id: str
    target: RenderedSite
    protected: RenderedSite
    arguments: argparse.Namespace | None = None
    protected_cluster_id: str = ""

    @property
    def case_dir(self) -> Path:
        return self.run_dir / "cases" / CASE_ID

    @property
    def native_dir(self) -> Path:
        return self.target.source.parent / "uninstall"

    def native_binding(self) -> dict[str, Any]:
        deploy_host_binding.enforce_deploy_host_state_dir(
            argparse.Namespace(
                command="uninstall",
                state_dir=self.target.source.parent,
            )
        )
        bound = deploy_host_binding.bound_deploy_host_state_dir()
        return {
            "state_dir": str(self.target.source.parent),
            "installed_state_dir": str(bound) if bound is not None else None,
            "python_prefix": str(Path(sys.prefix).resolve()),
        }

    def environment(self) -> dict[str, str]:
        return {
            "BOOT032_TARGET_SITE": str(self.target.source),
            "BOOT032_PROTECTED_SITE": str(self.protected.source),
            "BOOT032_PROTECTED_CLUSTER_ID": self.protected_cluster_id,
            "BOOT032_NATIVE_STATE": str(self.native_dir),
            "BOOT032_FIXTURE_ID": self.fixture_id,
            **{
                f"BOOT032_{label}_{plane}_KUBECONFIG": str(
                    kubeconfig(site, plane.lower())
                )
                for label, site in (
                    ("TARGET", self.target),
                    ("PROTECTED", self.protected),
                )
                for plane in ("CPU", "GPU")
            },
        }

    def inputs(self) -> dict[str, Any]:
        current = Settings(
            self.run_dir,
            self.fixture_id,
            load_site(checked_path(self.target.source)),
            load_site(checked_path(self.protected.source)),
            self.arguments,
            protected_cluster_id=self.protected_cluster_id,
        )
        validate_sites(current)
        require(
            all(
                fresh.source_sha256 == previous.source_sha256
                and fresh.release_config == previous.release_config
                and fresh.environment == previous.environment
                for fresh, previous in (
                    (current.target, self.target),
                    (current.protected, self.protected),
                )
            ),
            "site changed after runner configuration",
        )
        bootstrap = current.target.source.parent / "bootstrap-state.json"
        document = read_document(bootstrap)
        require(
            document.get("site_id") == current.target.metadata_name
            and document.get("phase") == "site-ready",
            "sacrificial site lacks completed matching bootstrap provenance",
        )
        resources = mapping(
            document.get("resources"), "bootstrap resources are missing"
        )
        discovered = mapping(
            resources.get("initial_deploy_target"), "bootstrap target is missing"
        )
        cpu = mapping(discovered.get("cpu"), "bootstrap CPU target is missing")
        require(
            cpu.get("eks_arn") == current.target.release_config["cpu_eks_arn"]
            and cpu.get("hyperpod_name")
            == current.target.release_config["cpu_hyperpod_cluster_name"],
            "bootstrap provenance targets a different CPU cluster",
        )
        gpu = discovered.get("gpu_clusters")
        require(
            isinstance(gpu, list)
            and len(gpu) == len(current.target.release_config["clusters"])
            and {(item["eks_arn"], item["hyperpod_name"]) for item in gpu}
            == {
                (item["eks_arn"], item["hyperpod_name"])
                for item in cluster_specs(current.target)[1:]
            },
            "bootstrap provenance targets different GPU clusters",
        )
        return {
            "fixture_id": self.fixture_id,
            "protected_cluster_id": self.protected_cluster_id,
            "native_binding": self.native_binding(),
            "run_dir": str(self.run_dir),
            "native_dir": str(self.native_dir),
            "bootstrap_sha256": file_digest(bootstrap),
            "target": site_inputs(current.target),
            "protected": site_inputs(current.protected),
        }


def validate_sites(settings: Settings) -> None:
    require(
        re.fullmatch(r"[a-f0-9]{12}", settings.fixture_id) is not None,
        "fixture ID must be twelve lowercase hexadecimal characters",
    )
    expected = settings.case_dir / "sacrificial" / "site.yaml"
    require(
        settings.target.source == expected.resolve()
        and settings.target.source != settings.protected.source
        and not settings.protected.source.is_relative_to(settings.case_dir),
        "target must be the separate BOOT-032 sacrificial site, never the accepted site",
    )
    checked_path(expected.parent, directory=True)
    require(
        settings.target.repository_root.resolve() == ROOT,
        "sacrificial site must use the runner's tested source checkout",
    )
    target = cluster_specs(settings.target)
    protected = cluster_specs(settings.protected)
    require(
        settings.protected_cluster_id in {row["cluster_id"] for row in protected[1:]},
        "selected protected cluster is not an explicit member of the accepted site",
    )
    prefix = f"boot032-{settings.fixture_id}-"
    for row in target:
        suffix = r"cpu" if row["plane"] == "cpu" else r"gpu-[0-9]+"
        require(
            re.fullmatch(re.escape(prefix) + suffix, row["eks_name"]) is not None
            and row["hyperpod_name"] == row["eks_name"],
            "ordinary cluster names cannot be used as sacrificial targets",
        )
    require(
        settings.target.metadata_name != settings.protected.metadata_name,
        "target and protected site identities alias",
    )
    for key in ("eks_arn", "hyperpod_name"):
        require(
            not {row[key] for row in target} & {row[key] for row in protected},
            "sacrificial physical targets overlap the accepted site",
        )
