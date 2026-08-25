#!/usr/bin/env bash
set -euo pipefail

control_plane_url="${1:-}"
ca_bundle="${2:-}"
minimum_seconds="${3:-2592000}"

[[ -n "${control_plane_url}" ]] || {
    printf 'ERROR: control plane URL is required\n' >&2
    exit 2
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bundle_checker="${script_dir}/verify-certificate-bundle"
if [[ ! -x "${bundle_checker}" ]]; then
    bundle_checker="${script_dir}/verify-certificate-bundle.sh"
fi
"${bundle_checker}" \
    "${ca_bundle}" "${minimum_seconds}" >/dev/null

read -r hostname port < <(
    python3 - "${control_plane_url}" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if parsed.scheme != "https" or not parsed.hostname:
    raise SystemExit("control plane URL must be HTTPS")
print(parsed.hostname, parsed.port or 443)
PY
)

scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
timeout 20 openssl s_client \
    -connect "${hostname}:${port}" \
    -servername "${hostname}" \
    -CAfile "${ca_bundle}" \
    -verify_return_error \
    -showcerts </dev/null >"${scratch}/chain.pem" 2>"${scratch}/tls.log" || {
    cat "${scratch}/tls.log" >&2
    printf 'ERROR: control plane TLS handshake failed\n' >&2
    exit 1
}
awk '
  /-----BEGIN CERTIFICATE-----/ { capture = 1 }
  capture { print }
  /-----END CERTIFICATE-----/ { exit }
' "${scratch}/chain.pem" >"${scratch}/leaf.pem"
openssl x509 -in "${scratch}/leaf.pem" -noout >/dev/null || {
    printf 'ERROR: control plane returned no leaf certificate\n' >&2
    exit 1
}
if ! openssl x509 -in "${scratch}/leaf.pem" \
    -checkend "${minimum_seconds}" -noout; then
    printf 'ERROR: control plane leaf certificate expires within %s seconds: ' \
        "${minimum_seconds}" >&2
    openssl x509 -in "${scratch}/leaf.pem" \
        -noout -subject -enddate >&2
    exit 1
fi

openssl x509 -in "${scratch}/leaf.pem" \
    -noout -subject -issuer -enddate
