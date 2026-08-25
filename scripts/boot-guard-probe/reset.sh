#!/bin/bash
# 删除探针，并**等到该 label 下的 Pod 全部消失**。
#
# 为什么不能只用 `kubectl delete deployment --wait=true`：
# 它只等 Deployment 对象消失，不等 Pod 消失。2026-08-04 的执行中，
# BOOT-008 分支 1 因此报了个假 FAIL——assert 用 `logs -l` 跨 Pod 聚合
# 匹配文本、readiness 却只读 .items[0]，于是文本命中来自新 Pod、
# readiness 读到上一轮那个已 Ready 的旧 Pod。同一缺陷反向能产生假 PASS。
set -uo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
SEL="app=gpu-fault-api-guard-probe"

kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
  delete deployment gpu-fault-api-guard-probe \
  --ignore-not-found --wait=true >/dev/null

for _ in $(seq 1 60); do
  n="$(kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" \
    get pod -l "${SEL}" --no-headers 2>/dev/null | wc -l)"
  if [[ "${n}" == "0" ]]; then
    echo "probe reset: 0 pods"
    exit 0
  fi
  sleep 2
done

echo "probe reset FAILED: 仍有 Pod 残留"
kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" get pod -l "${SEL}"
exit 1
