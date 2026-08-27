#!/bin/bash
# BOOT-001~010 跑完后的清理，并复核生产回到基线。
#
# 这个脚本**刻意不读任何 /tmp 里的文件**：2026-08-04 的执行中机器重启、
# /tmp 被清空，夹具与派生清单全丢，而集群上的探针还在 CrashLoopBackOff
# 里空转了 8 小时（重启 102 次）。清理流程必须在那种状态下仍然能跑完，
# 所以它只按固定名字删对象。
#
# 用法:
#   cleanup.sh                     删探针与临时 Secret，复核生产（默认不删库）
#   cleanup.sh --drop-database     额外 DROP DATABASE gpu_fault_guardprobe
#
# DROP DATABASE 不可逆，必须显式加参数；脚本内另有三道防护：
# 断言当前连的是生产库、断言目标 != 当前库、不存在则跳过。
set -uo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
PROBE=gpu-fault-api-guard-probe
DROP_DB=0
[[ "${1:-}" == "--drop-database" ]] && DROP_DB=1

k() { kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" "$@"; }

echo "===== 1. 删探针 Deployment"
k delete deployment "${PROBE}" --ignore-not-found --wait=true
for _ in $(seq 1 60); do
  n="$(k get pod -l "app=${PROBE}" --no-headers 2>/dev/null | wc -l)"
  [[ "${n}" == "0" ]] && break
  sleep 2
done
echo "剩余探针 Pod: $(k get pod -l "app=${PROBE}" --no-headers 2>/dev/null | wc -l)"

echo "===== 2. 删临时 Secret"
for s in gpu-fault-aurora-guardprobe \
         gpu-fault-regional-clusters-probe \
         gpu-fault-regional-clusters-bad; do
  k delete secret "${s}" --ignore-not-found
done
echo "残留 probe 相关 Secret:"
k get secret --no-headers 2>/dev/null | awk '{print $1}' | grep -E 'guardprobe|clusters-probe|clusters-bad' \
  || echo "  （无）"

if (( DROP_DB )); then
  echo "===== 3. DROP DATABASE gpu_fault_guardprobe"
  pod="$(k get pod -l app=gpu-fault-api-ha \
    -o jsonpath='{.items[0].metadata.name}')"
  k exec -i "${pod}" -- python - <<'PY'
import os, urllib.parse, psycopg
TARGET = "gpu_fault_guardprobe"
parts = urllib.parse.urlsplit(os.environ["GPU_FAULT_STORE_URL"])
assert parts.path == "/gpu_fault", f"当前不是生产库: {parts.path}"   # 防护 1
assert parts.path.lstrip("/") != TARGET, "目标库不能是当前连接的库"   # 防护 2
admin = urllib.parse.urlunsplit(parts._replace(path="/postgres"))
with psycopg.connect(admin, autocommit=True) as conn:
    exists = conn.execute(
        "SELECT 1 FROM pg_database WHERE datname=%s", (TARGET,)).fetchone()
    if not exists:                                                   # 防护 3
        print(f"{TARGET} 不存在，跳过")
    else:
        conn.execute(f'DROP DATABASE "{TARGET}"')
        print(f"dropped {TARGET}")
    print("剩余 gpu_fault* 库:", [r[0] for r in conn.execute(
        "SELECT datname FROM pg_database WHERE datname LIKE 'gpu_fault%' "
        "ORDER BY datname")])
PY
else
  echo "===== 3. 跳过 DROP DATABASE（需显式 --drop-database）"
fi

echo "===== 4. 复核生产回到基线"
k get deployment gpu-fault-api-ha \
  -o jsonpath='generation={.metadata.generation} readyReplicas={.status.readyReplicas}{"\n"}'
k get pod -l app=gpu-fault-api-ha \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount \
  --no-headers
echo "--- 生产 Service endpoints（不得出现 guard-probe）"
for s in gpu-fault-api gpu-fault-api-nlb; do
  echo "-- ${s}"
  k get endpoints "${s}" \
    -o jsonpath='{range .subsets[*].addresses[*]}{.targetRef.name}{"\n"}{end}' 2>/dev/null
done
