from __future__ import annotations

import argparse
from pathlib import Path

from release_attestation import verify_attestation


ROOT = Path(__file__).resolve().parents[1]


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
    args = parser.parse_args()
    value = verify_attestation(
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
    )
    print(value["subject"]["release_id"])


if __name__ == "__main__":
    main()
