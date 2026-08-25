#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:-}"
VERSION="$(
    python3 -c '
import pathlib
import tomllib

root = pathlib.Path(__import__("sys").argv[1])
print(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"])
' "${REPO_DIR}"
)"
WHEEL="${GPU_FAULT_WHEEL:-}"
if [[ -z "${WHEEL}" ]]; then
    WHEEL="$(
        python3 - "${REPO_DIR}" "$(
            printf '%s' \
                "${GPU_FAULT_RELEASE_MANIFEST:-${REPO_DIR}/dist/current-release.json}"
        )" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
value = json.loads(manifest.read_text(encoding="utf-8"))["wheel"]
path = pathlib.Path(value)
print(path if path.is_absolute() else root / path)
PY
    )"
fi
[[ -n "${WHEEL}" && -f "${WHEEL}" ]] || {
    printf 'ERROR: build the %s wheel first\n' "${VERSION}" >&2
    exit 1
}
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${WHEEL}")}"

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT
BUNDLE="gpu-fault-node-installer-${VERSION}"
ROOT="${STAGING}/${BUNDLE}"
install -d \
    "${ROOT}/deploy/control-plane/regional" \
    "${ROOT}/deploy/control-plane/tools" \
    "${ROOT}/deploy/node" \
    "${ROOT}/deploy/systemd" \
    "${ROOT}/deploy/dataplane" \
    "${ROOT}/dist"
install -m 0755 "${SCRIPT_DIR}"/*.sh "${ROOT}/deploy/node/"
install -m 0755 \
    "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
    "${ROOT}/deploy/control-plane/tools/"
install -m 0644 \
    "${REPO_DIR}/deploy/control-plane/regional/cleanup-inventory.json" \
    "${ROOT}/deploy/control-plane/regional/"
install -m 0644 "${REPO_DIR}/deploy/systemd/"*.service \
    "${ROOT}/deploy/systemd/"
install -m 0644 "${REPO_DIR}/deploy/systemd/"*.timer \
    "${ROOT}/deploy/systemd/"
install -m 0644 "${REPO_DIR}/deploy/dataplane/dcgm-counters.csv" \
    "${ROOT}/deploy/dataplane/dcgm-counters.csv"
install -m 0644 "${WHEEL}" "${ROOT}/dist/"
install -d "${OUTPUT_DIR}"
tar -C "${STAGING}" -czf "${OUTPUT_DIR}/${BUNDLE}.tar.gz" "${BUNDLE}"
printf '%s\n' "${OUTPUT_DIR}/${BUNDLE}.tar.gz"
