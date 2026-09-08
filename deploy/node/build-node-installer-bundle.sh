#!/usr/bin/env bash
set -euo pipefail
umask 0022

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
WHEEL="${GPU_FAULT_NODE_WHEEL:-${GPU_FAULT_WHEEL:-}}"
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
document = json.loads(manifest.read_text(encoding="utf-8"))
value = (
    document.get("components", {})
    .get("node_runtime", {})
    .get("wheel", document["wheel"])
)
path = pathlib.Path(value)
print(path if path.is_absolute() else root / path)
PY
    )"
fi
[[ -n "${WHEEL}" && -f "${WHEEL}" ]] || {
    printf 'ERROR: build the %s node-runtime wheel first\n' "${VERSION}" >&2
    exit 1
}
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${WHEEL}")}"

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT
BUNDLE="gpu-fault-node-installer-${VERSION}"
ROOT="${STAGING}/${BUNDLE}"
install -d -m 0755 \
    "${ROOT}/deploy/control-plane/regional" \
    "${ROOT}/deploy/control-plane/tools" \
    "${ROOT}/deploy/node" \
    "${ROOT}/deploy/systemd" \
    "${ROOT}/deploy/dataplane" \
    "${ROOT}/dist" \
    "${ROOT}/requirements"
install -m 0755 "${SCRIPT_DIR}"/*.sh "${ROOT}/deploy/node/"
# The node installer installs the dependency closure with
# `pip install --require-hashes -r requirements/node-runtime.lock`, so the
# narrow node-runtime hash lock ships inside the bundle next to the wheel.
# This lock pins only the node collector/agent's true runtime closure, not
# the broad control-plane runtime.lock.
install -m 0644 "${REPO_DIR}/requirements/node-runtime.lock" \
    "${ROOT}/requirements/node-runtime.lock"
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
tar -C "${STAGING}" \
    --sort=name \
    --mtime='UTC 1970-01-01' \
    --owner=0 --group=0 --numeric-owner \
    -cf - "${BUNDLE}" |
    gzip -n >"${OUTPUT_DIR}/${BUNDLE}.tar.gz"
printf '%s\n' "${OUTPUT_DIR}/${BUNDLE}.tar.gz"
