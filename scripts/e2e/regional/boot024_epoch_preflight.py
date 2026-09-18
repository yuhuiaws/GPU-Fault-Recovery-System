"""Produce a separate, target-bound readiness receipt for the BOOT-024 epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.site import load_site, materialized_release_config
from gpu_fault_release import regional_release_config, regional_release_diff, rollout
from scripts.e2e.regional.boot_acceptance_common import SiteFixture
from scripts.e2e.regional.boot_acceptance_lifecycle import (
    admin_command,
    parse_status_report,
)
from scripts.e2e.regional.boot_membership_observation import membership_observation
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence

CASE_ID = "GF-REGIONAL-BOOT-024"


def isolated_epoch_scope(
    lifecycle: dict[str, Any], protected: dict[str, Any]
) -> dict[str, Any]:
    gpu = lifecycle.get("clusters")
    other_gpu = protected.get("clusters")
    if (
        not isinstance(gpu, list)
        or len(gpu) != 1
        or not isinstance(other_gpu, list)
        or not other_gpu
    ):
        raise ValueError(
            "BOOT-024 needs one disposable GPU and a separate protected fleet"
        )
    own = [lifecycle.get("cpu_eks_arn"), gpu[0].get("eks_cluster_arn")]
    other = [
        protected.get("cpu_eks_arn"),
        *(item.get("eks_cluster_arn") for item in other_gpu),
    ]
    if (
        any(
            not isinstance(item, str) or ":eks:" not in item or ":cluster/" not in item
            for item in [*own, *other]
        )
        or len(set(own)) != 2
        or set(own) & set(other)
    ):
        raise ValueError("BOOT-024 physical cluster scopes overlap or are incomplete")
    return {
        "lifecycle_cpu_eks_arn": own[0],
        "lifecycle_gpu_eks_arn": own[1],
        "protected_cpu_eks_arn": other[0],
        "protected_gpu_eks_arns": sorted(other[1:]),
    }


def produce_epoch_receipt(
    site_path: Path, protected_path: Path, previous_path: Path
) -> dict[str, Any]:
    site, protected = load_site(site_path), load_site(protected_path)
    scope = isolated_epoch_scope(site.release_config, protected.release_config)
    original_site = hashlib.sha256(site_path.read_bytes()).hexdigest()
    original_protected = hashlib.sha256(protected_path.read_bytes()).hexdigest()
    protected_target = SiteFixture(
        protected_path, protected.release_config["clusters"][0]["cluster_id"]
    )
    protected_identity = protected_target.regional.evidence_identity()
    previous = predecessor_evidence(
        previous_path, "GF-REGIONAL-BOOT-023", **protected_identity
    )
    if previous.get("valid") is not True:
        raise ValueError("original protected-site BOOT-023 sequence proof is not valid")
    target = SiteFixture(site_path, site.release_config["clusters"][0]["cluster_id"])
    identity = target.regional.evidence_identity()
    status = admin_command(
        "status", "--state-dir", str(site_path.parent), "--full", timeout=1800
    )
    if status.returncode:
        raise ValueError("disposable BOOT-024 target is not healthy")
    report = parse_status_report(status.stdout)
    with materialized_release_config(site) as config:
        release = rollout.RegionalRelease(
            regional_release_config.ReleaseConfig.load(config), rollout.Runner()
        )
        state = release._load_state()
        difference = regional_release_diff.classify_release(release, state)
        if (
            difference.kind is not regional_release_diff.ReleaseChangeKind.NOOP
            or state.get("transaction_committed") is not True
            or state.get("phase") != "complete"
            or state.get("release_id") != identity["release_id"]
        ):
            raise ValueError(
                "disposable BOOT-024 target is not a committed NOOP baseline"
            )
    publication = membership_observation(site)
    if (
        hashlib.sha256(site_path.read_bytes()).hexdigest() != original_site
        or hashlib.sha256(protected_path.read_bytes()).hexdigest() != original_protected
        or target.regional.evidence_identity() != identity
        or protected_target.regional.evidence_identity() != protected_identity
    ):
        raise ValueError("BOOT-024 target identity changed during preflight")
    return {
        "schema_version": 1,
        "case_id": CASE_ID,
        "report_type": "epoch-prerequisite",
        "status": "READY",
        "observed_at": datetime.now(UTC).isoformat(),
        "target": identity,
        "protected_target": protected_identity,
        "scope": scope,
        "site_sha256": original_site,
        "protected_site_sha256": original_protected,
        "sequence_predecessor_sha256": hashlib.sha256(
            previous_path.read_bytes()
        ).hexdigest(),
        "sequence_predecessor_case_id": "GF-REGIONAL-BOOT-023",
        "status_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True).encode()
        ).hexdigest(),
        "membership_publication": publication,
        "limitations": [
            "Read-only admission proof, not BOOT-024 PASS or mutation authorization.",
            "Protected-site sequence proof is retained as-is, never relabelled to the disposable cluster.",
        ],
    }


def validate_epoch_receipt(
    receipt: dict[str, Any],
    *,
    site_path: Path,
    protected_path: Path,
    previous_path: Path,
    now: datetime | None = None,
) -> None:
    observed = datetime.fromisoformat(receipt["observed_at"])
    current = now or datetime.now(UTC)
    if (
        observed.tzinfo is None
        or not 0 <= (current - observed).total_seconds() <= 900
        or receipt.get("schema_version") != 1
        or receipt.get("case_id") != CASE_ID
        or receipt.get("report_type") != "epoch-prerequisite"
        or receipt.get("status") != "READY"
        or receipt.get("site_sha256")
        != hashlib.sha256(site_path.read_bytes()).hexdigest()
        or receipt.get("protected_site_sha256")
        != hashlib.sha256(protected_path.read_bytes()).hexdigest()
        or receipt.get("sequence_predecessor_sha256")
        != hashlib.sha256(previous_path.read_bytes()).hexdigest()
        or receipt.get("scope")
        != isolated_epoch_scope(
            load_site(site_path).release_config,
            load_site(protected_path).release_config,
        )
    ):
        raise ValueError(
            "BOOT-024 epoch prerequisite is stale or belongs to another target"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--protected-site", type=Path, required=True)
    parser.add_argument("--predecessor-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate_epoch_receipt(
            json.loads(args.output.read_text(encoding="utf-8")),
            site_path=args.site,
            protected_path=args.protected_site,
            previous_path=args.predecessor_evidence,
        )
        print(json.dumps({"case_id": CASE_ID, "prerequisite": "READY"}))
    else:
        write_json_atomic(
            args.output,
            produce_epoch_receipt(
                args.site, args.protected_site, args.predecessor_evidence
            ),
        )
        print(
            json.dumps(
                {
                    "case_id": CASE_ID,
                    "prerequisite": "READY",
                    "output": str(args.output),
                }
            )
        )


if __name__ == "__main__":
    main()
