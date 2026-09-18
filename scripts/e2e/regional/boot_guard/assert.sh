#!/bin/bash
# 判定负向 BOOT 用例：Pod 不进入 Ready，且日志出现指定的错误文本。
#
# 用法: assert.sh "<期望的错误文本>" [超时秒数]
# 前置: 必须先跑 reset.sh，保证 label 下只有本轮的 Pod。
#
# 两条设计约束，都是踩过的坑：
#  1. 不用 `rollout status` 判定——startupProbe 是
#     failureThreshold=120 × periodSeconds=5 = 600 秒，会白等 10 分钟。
#  2. 只读**唯一** Pod 的日志；label 下有多个 Pod 时直接拒绝判定（exit 2），
#     否则文本与 readiness 可能来自不同 Pod，产生假 PASS / 假 FAIL。
set -euo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
expect="${1:?需要期望的错误文本}"
deadline=$(( ${2:-180} ))
SEL="app=gpu-fault-api-guard-probe"
elapsed=0

while (( elapsed < deadline )); do
  inventory="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" \
    -n "${NAMESPACE}" get pod -l "${SEL}" -o json)"
  count="$(jq -er 'if (.items | type) == "array" then .items | length else error("invalid Pod inventory") end' <<<"${inventory}")"
  if (( count > 1 )); then
    echo "FAIL: probe Pod identity is ambiguous; reset is required"
    exit 2
  fi
  if (( count == 1 )); then
    pod="$(jq -er '.items[0].metadata.name | select(type == "string" and length > 0)' <<<"${inventory}")"
    uid="$(jq -er '.items[0].metadata.uid | select(type == "string" and length > 0)' <<<"${inventory}")"
    logs="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      logs "${pod}" --tail=400)"
    restarts="$(jq -er '[.items[0].status.containerStatuses[]?.restartCount] | if length > 0 then add else 0 end' <<<"${inventory}")"
    if (( restarts > 0 )); then
      previous="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
        logs "${pod}" --previous --tail=400)"
      logs="${logs}"$'\n'"${previous}"
    fi
    if grep -qF "${expect}" <<<"${logs}"; then
      current="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
        get pod "${pod}" -o json)"
      if ! jq -e --arg uid "${uid}" \
        '.metadata.uid == $uid and .metadata.deletionTimestamp == null' \
        <<<"${current}" >/dev/null; then
        echo "FAIL: probe Pod changed during observation"
        exit 1
      fi
      ready="$(jq -er '[.status.conditions[]? | select(.type == "Ready") | .status] | if length == 1 then .[0] else error("Ready condition missing or ambiguous") end' <<<"${current}")"
      echo "pod=${pod}; MATCHED expected text; ready=${ready}"
      if [[ "${ready}" == "False" ]]; then
        echo "PASS"
        exit 0
      fi
      echo "FAIL: probe readiness is True or unknown"
      exit 1
    fi
  fi
  sleep 10
  elapsed=$(( elapsed + 10 ))
done

echo "FAIL: 超时 ${deadline}s 未见期望文本"
kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
  get pod -l "${SEL}" -o wide
exit 1
