#!/usr/bin/env bash
set -euo pipefail

certificate_file="${1:-}"
minimum_seconds="${2:-2592000}"

[[ -n "${certificate_file}" && -r "${certificate_file}" ]] || {
    printf 'ERROR: certificate bundle is not readable: %s\n' \
        "${certificate_file:-EMPTY}" >&2
    exit 2
}
[[ "${minimum_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: minimum certificate validity must be positive\n' >&2
    exit 2
}

scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
awk -v directory="${scratch}" '
  /-----BEGIN CERTIFICATE-----/ {
    count += 1
    output = sprintf("%s/cert-%04d.pem", directory, count)
  }
  output != "" { print > output }
  /-----END CERTIFICATE-----/ {
    close(output)
    output = ""
  }
  END { if (count == 0) exit 2 }
' "${certificate_file}" || {
    printf 'ERROR: certificate bundle contains no PEM certificate\n' >&2
    exit 2
}

count=0
for certificate in "${scratch}"/cert-*.pem; do
    [[ -f "${certificate}" ]] || continue
    count=$((count + 1))
    openssl x509 -in "${certificate}" -noout >/dev/null || {
        printf 'ERROR: invalid PEM certificate in %s\n' \
            "${certificate_file}" >&2
        exit 1
    }
    if ! openssl x509 -in "${certificate}" \
        -checkend "${minimum_seconds}" -noout; then
        printf 'ERROR: certificate expires within %s seconds: ' \
            "${minimum_seconds}" >&2
        openssl x509 -in "${certificate}" \
            -noout -subject -enddate >&2
        exit 1
    fi
done
[[ "${count}" -gt 0 ]] || {
    printf 'ERROR: certificate bundle contains no complete certificate\n' >&2
    exit 2
}

printf 'certificate bundle valid: certificates=%s minimum_seconds=%s\n' \
    "${count}" "${minimum_seconds}"
