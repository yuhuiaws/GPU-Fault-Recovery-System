#!/usr/bin/env bash
# Ship the AWS RDS global CA bundle into the cluster as a ConfigMap.
#
# The control-plane Postgres DSN uses sslmode=verify-full (see
# gpu_fault.admin.bootstrap._aurora_dsn): the connection is refused unless a
# trusted RDS CA bundle is mounted in the Pod and GPU_FAULT_RDS_CA_BUNDLE
# points at it. Without this ConfigMap the control-plane Deployment, the
# Aurora migration Job and the credential-refresh CronJob all mount a missing
# volume and stay in CreateContainerConfigError -- fail closed, never an
# unverified `require` connection.
#
# The bundle installed is the copy vendored next to this script
# (rds-global-bundle.pem), admitted only if its SHA-256 matches the pin below,
# exactly as the wheel and node-bundle digest gates elsewhere in this tree do;
# a mismatch aborts the deploy. Neither a deploy nor `gpu-fault-admin config`
# downloads the bundle it trusts: AWS republishes the global bundle whenever an
# RDS region launches, and while this script installed the download every such
# rotation failed every deploy and every `config` until a release shipped a
# new pin.
#
# After the install the published bundle is compared with the pin for a
# warning only: the download is never installed, and neither a failed download
# nor a mismatch fails this step. `gpu-fault-admin status` reports the drift in
# full (added, removed and changed CAs by subject and SHA-256 fingerprint, and
# whether the CA of the Aurora cluster's certificates is in each bundle).
#
# Rolling the vendored copy after an AWS rotation is a reviewed source change
# that ships in a release:
#   curl -fsS --proto '=https' \
#       https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
#       -o deploy/control-plane/tools/rds-global-bundle.pem
#   sha256sum deploy/control-plane/tools/rds-global-bundle.pem
# then review the added and removed CAs (`gpu-fault-admin status` lists them),
# replace RDS_CA_BUNDLE_SHA256 below with the new digest and add a provenance
# line, and deploy. A CA that AWS removed and the Aurora certificates still
# chain to must stay trusted: move the cluster to another CA first.
#
# Point GPU_FAULT_RDS_CA_BUNDLE_SOURCE at another local PEM to install it
# instead of the vendored copy; the same digest gate applies, so it must be
# byte-identical to the pinned bundle.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
CONFIGMAP_NAME="${GPU_FAULT_RDS_CA_BUNDLE_CONFIGMAP:-gpu-fault-rds-ca-bundle}"
# The single key inside the ConfigMap. It becomes the file name under the
# mount directory, so the in-Pod path is <mount dir>/ca-bundle.pem, which is
# the value every consumer sets GPU_FAULT_RDS_CA_BUNDLE to.
BUNDLE_KEY="ca-bundle.pem"
VENDORED_BUNDLE="${SCRIPT_DIR}/rds-global-bundle.pem"
# Read only by the drift warning after the install; never installed.
RDS_CA_BUNDLE_URL="${GPU_FAULT_RDS_CA_BUNDLE_URL:-https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem}"
# Pinned SHA-256 of the vendored rds-global-bundle.pem. Provenance:
#   curl -fsS https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem | sha256sum
# Fetched 2026-09-07; 108 certificates.
# 2026-09-30: AWS rotated the global bundle (108 -> 111 CAs; the three
# "Amazon RDS me-west-1 Root CA" G1 roots were added, nothing removed).
# 2026-10-14: AWS rotated the global bundle (111 -> 114 CAs; the three
# "Amazon RDS sa-west-1 Root CA" G1 roots ECC384, RSA2048 and RSA4096 were
# added, nothing removed); vendored as rds-global-bundle.pem from then on.
RDS_CA_BUNDLE_SHA256="90a6f1a7978a1b1b096dca6c57511ca48d3b8cf052f8e704baf358f34ea2f21b"

for command in kubectl sha256sum; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "required command not found: ${command}" >&2
        exit 2
    }
done

[[ "${RDS_CA_BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "RDS_CA_BUNDLE_SHA256 must be a lowercase SHA-256" >&2
    exit 2
}

kubectl_args=()
if [[ -n "${KUBECONFIG:-}" ]]; then
    kubectl_args+=(--kubeconfig "${KUBECONFIG}")
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
BUNDLE_FILE="${WORK_DIR}/${BUNDLE_KEY}"

SOURCE="${GPU_FAULT_RDS_CA_BUNDLE_SOURCE:-${VENDORED_BUNDLE}}"
[[ -f "${SOURCE}" ]] || {
    echo "RDS CA bundle source is not a file: ${SOURCE}" >&2
    exit 2
}
cp "${SOURCE}" "${BUNDLE_FILE}"

# Fail closed: a bundle whose digest does not match the pin never reaches the
# cluster. `sha256sum -c` returns non-zero on mismatch, and set -e aborts.
printf '%s  %s\n' "${RDS_CA_BUNDLE_SHA256}" "${BUNDLE_FILE}" |
    sha256sum -c - >/dev/null || {
    echo "RDS CA bundle digest does not match the pinned SHA-256; refusing" >&2
    echo "  source:   ${SOURCE}" >&2
    echo "  expected: ${RDS_CA_BUNDLE_SHA256}" >&2
    echo "  actual:   $(sha256sum "${BUNDLE_FILE}" | awk '{print $1}')" >&2
    exit 1
}

kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" create configmap \
    "${CONFIGMAP_NAME}" \
    --from-file="${BUNDLE_KEY}=${BUNDLE_FILE}" \
    --dry-run=client -o yaml |
    kubectl "${kubectl_args[@]}" apply -f -

echo "applied ConfigMap ${CONFIGMAP_NAME} in ${NAMESPACE} (${RDS_CA_BUNDLE_SHA256})"

certificate_count() {
    grep -c -- '-----BEGIN CERTIFICATE-----' "$1" || true
}

# The drift warning: report, never trust, never fail.
report_published_drift() {
    local published="${WORK_DIR}/published.pem" error digest
    if ! command -v curl >/dev/null 2>&1; then
        echo "note: curl not found; the published RDS CA bundle was not compared with the pin" >&2
        return 0
    fi
    if ! error="$(curl --fail --silent --show-error --location --proto '=https' \
        --connect-timeout 5 --max-time 20 "${RDS_CA_BUNDLE_URL}" \
        -o "${published}" 2>&1)"; then
        echo "note: could not download ${RDS_CA_BUNDLE_URL} to compare it with the pin (${error:-curl failed}); the pinned copy is installed" >&2
        return 0
    fi
    digest="$(sha256sum "${published}" | awk '{print $1}')"
    [[ "${digest}" != "${RDS_CA_BUNDLE_SHA256}" ]] || return 0
    echo "WARNING: the published AWS RDS CA bundle changed: ${RDS_CA_BUNDLE_URL} has sha256 ${digest} ($(certificate_count "${published}") certificates), the pinned copy installed here has ${RDS_CA_BUNDLE_SHA256} ($(certificate_count "${BUNDLE_FILE}") certificates). The download was not installed; gpu-fault-admin status lists the added, removed and changed CAs and the steps to roll the vendored copy." >&2
}
report_published_drift || true
