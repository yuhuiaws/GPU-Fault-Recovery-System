from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from release_attestation import ReleaseAttestationError, verify_attestation
from signing_transparency import ALLOW_UNLOGGED_FLAG


ROOT = Path(__file__).resolve().parents[1]


def _live_release_reader(site_file: Path):
    """Read the cluster's release record only if an unlogged bundle needs it."""

    def read() -> dict[str, Any]:
        from release_live_state import read_live_release_state

        return read_live_release_state(site_file)

    return read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--cosign-key")
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--certificate-identity")
    parser.add_argument("--certificate-oidc-issuer")
    parser.add_argument("--allow-staging-release", action="store_true")
    parser.add_argument(
        ALLOW_UNLOGGED_FLAG,
        dest="allow_unlogged_signature",
        action="store_true",
        help=(
            "this command carries the operator's consent: accept a bundle "
            "without a transparency-log entry whose signed attestation declares it"
        ),
    )
    parser.add_argument(
        "--live-release-site",
        type=Path,
        help=(
            "managed site.yaml; a bundle without a transparency-log entry is "
            "accepted when the cluster's committed live release is this release"
        ),
    )
    args = parser.parse_args()
    try:
        value = _verify(args)
    except ReleaseAttestationError as exc:
        print(f"verify-release-attestation: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(value["subject"]["release_id"])


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    return verify_attestation(
        ROOT,
        args.attestation.resolve(),
        signature=(args.signature.resolve() if args.signature is not None else None),
        bundle=args.bundle.resolve() if args.bundle is not None else None,
        cosign_key=args.cosign_key,
        certificate=(
            args.certificate.resolve() if args.certificate is not None else None
        ),
        certificate_identity=args.certificate_identity,
        certificate_oidc_issuer=args.certificate_oidc_issuer,
        allow_staging=args.allow_staging_release,
        allow_unlogged=args.allow_unlogged_signature,
        live_release=(
            _live_release_reader(args.live_release_site.resolve())
            if args.live_release_site is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
