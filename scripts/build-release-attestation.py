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
    arguments = parser.parse_args()
    manifest = ROOT / "dist/current-release.json"
    attestation = build_attestation(
        ROOT,
        manifest,
        staging_only=arguments.staging_only,
        impact_base=arguments.impact_base,
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
