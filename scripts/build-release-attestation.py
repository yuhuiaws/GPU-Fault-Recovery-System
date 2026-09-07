from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_attestation import build_attestation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-only", action="store_true")
    parser.add_argument("--impact-base")
    parser.add_argument("--impact-plan", type=Path)
    parser.add_argument("--ci-gate", type=Path)
    parser.add_argument(
        "--sbom-dir",
        type=Path,
        help=(
            "directory of *.cdx.json CycloneDX documents (make sbom); each file's "
            "path and SHA-256 is bound into the signed attestation"
        ),
    )
    arguments = parser.parse_args()
    manifest = ROOT / "dist/current-release.json"
    attestation = build_attestation(
        ROOT,
        manifest,
        staging_only=arguments.staging_only,
        impact_base=arguments.impact_base,
        impact_plan_path=arguments.impact_plan,
        ci_gate_path=arguments.ci_gate,
        sbom_dir=arguments.sbom_dir,
    )
    release_id = attestation["subject"]["release_id"]
    path = ROOT / "dist" / release_id / "attestation.json"
    path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current = ROOT / "dist/current-attestation.json"
    current.write_bytes(path.read_bytes())
    print(path)


if __name__ == "__main__":
    main()
