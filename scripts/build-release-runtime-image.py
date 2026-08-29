from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_image import build_runtime_image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist/release-runtime-image.json",
    )
    arguments = parser.parse_args()
    descriptor = build_runtime_image(
        ROOT,
        repository=arguments.repository,
        platform=arguments.platform,
        push=arguments.push,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
