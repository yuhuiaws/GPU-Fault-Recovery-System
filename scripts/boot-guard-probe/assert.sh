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
set -uo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
expect="${1:?需要期望的错误文本}"
deadline=$(( ${2:-180} ))
SEL="app=gpu-fault-api-guard-probe"
elapsed=0

while (( elapsed < deadline )); do
  mapfile -t pods < <(kubectl --kubeconfig "${CPU_KUBECONFIG}" \
    -n "${NAMESPACE}" get pod -l "${SEL}" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)
  if (( ${#pods[@]} > 1 )); then
    echo "FAIL: label 下有 ${#pods[@]} 个 Pod，证据归属不明（先跑 reset.sh）"
    printf '  %s\n' "${pods[@]}"
    exit 2
  fi
  if (( ${#pods[@]} == 1 )) && [[ -n "${pods[0]}" ]]; then
    pod="${pods[0]}"
    logs="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      logs "${pod}" --tail=400 2>/dev/null)
$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
      logs "${pod}" --previous --tail=400 2>/dev/null)"
    if grep -qF "${expect}" <<<"${logs}"; then
      ready="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
        get pod "${pod}" \
        -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')"
      state="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
        get pod "${pod}" -o jsonpath='{.status.containerStatuses[0].state}')"
      echo "pod=${pod}; MATCHED expected text; ready=${ready}; state=${state}"
      if [[ "${ready}" != "True" ]]; then
        echo "PASS"
        exit 0
      fi
      echo "FAIL: pod became Ready despite the guard"
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
