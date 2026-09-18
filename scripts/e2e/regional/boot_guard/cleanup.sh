#!/bin/bash
# Remove the fixed-name BOOT fixture only after Pod absence is verified.
set -euo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
FIXTURE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DROP_DB=0
case "${1:-}" in
  "") ;;
  --drop-database) DROP_DB=1 ;;
  *) echo "unknown cleanup option" >&2; exit 2 ;;
esac

k() { kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" "$@"; }

"${FIXTURE_DIR}/reset.sh"
errors=0
for secret in gpu-fault-aurora-guardprobe \
              gpu-fault-regional-clusters-probe \
              gpu-fault-regional-clusters-bad; do
  if ! k delete secret "${secret}" --ignore-not-found --wait=true --timeout=60s; then
    errors=1
    continue
  fi
  if ! remaining="$(k get secret "${secret}" --ignore-not-found -o name)"; then
    errors=1
  elif [[ -n "${remaining}" ]]; then
    echo "FAIL: probe Secret remains" >&2
    errors=1
  fi
done

if (( DROP_DB )); then
  inventory="$(k get pod -l app=gpu-fault-api-ha -o json)"
  pod="$(jq -er '
    [.items[]
     | select(.metadata.deletionTimestamp == null and .status.phase == "Running")
     | select([.status.conditions[]? | select(.type == "Ready") | .status] == ["True"])
     | select((.spec.containers | length) > 0)
     | select(([.spec.containers[].name] | sort) == ([.status.containerStatuses[]? | select(.ready == true) | .name] | sort))]
    | sort_by(.metadata.name) | first | .metadata.name
    | select(type == "string" and length > 0)
  ' <<<"${inventory}")"
  if ! k exec -i "${pod}" -- /opt/gpu-fault/control-plane/bin/python - drop \
    <"${FIXTURE_DIR}/../boot_guard_isolation.py"; then
    errors=1
  fi
fi

for service in gpu-fault-api gpu-fault-api-nlb; do
  endpoints="$(k get endpoints "${service}" -o json)"
  if ! jq -e '
    [.subsets[]? | (.addresses[]?, .notReadyAddresses[]?) | .targetRef.name // ""]
    | all(contains("gpu-fault-api-guard-probe") | not)
  ' <<<"${endpoints}" >/dev/null; then
    echo "FAIL: probe remains in production Service endpoints" >&2
    errors=1
  fi
done
exit "${errors}"
