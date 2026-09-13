#!/bin/bash
# 从生产 Deployment 派生探针基线清单（手册 §4.0 步骤 P4）。
#
# 用 Python 做结构化改写而不是 sed：label/selector/env 都是嵌套结构，
# sed 改不动；而且反亲和与 topologySpread 里的 labelSelector 必须跟着改，
# 否则探针会被已占满 3 个节点的生产 Pod 挤成 Pending——
# "Pod 不 Ready" 是负向用例的判定条件，Pending 会被误判为 PASS。
set -euo pipefail

: "${CPU_KUBECONFIG:?}"
: "${NAMESPACE:?}"
OUT="${1:-/tmp/guard-probe-base.json}"

CPU_KUBECONFIG="${CPU_KUBECONFIG}" NAMESPACE="${NAMESPACE}" \
python3 - "${OUT}" <<'PY'
import json
import os
import subprocess
import sys


def kubectl_json(*args):
    return json.loads(
        subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                os.environ["CPU_KUBECONFIG"],
                "-n",
                os.environ["NAMESPACE"],
                *args,
                "-o",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


d = kubectl_json("get", "deployment", "gpu-fault-api-ha")
probe = "gpu-fault-api-guard-probe"
spec = d["spec"]
out = {"apiVersion": "apps/v1", "kind": "Deployment",
       "metadata": {"name": probe, "namespace": d["metadata"]["namespace"],
                    "labels": {"app": probe}},
       "spec": spec}
spec["replicas"] = 1
spec["selector"]["matchLabels"]["app"] = probe
spec["template"]["metadata"]["labels"]["app"] = probe
spec["template"]["metadata"].pop("annotations", None)
container = spec["template"]["spec"]["containers"][0]
# The production role split now reserves 4 CPU per ingress. A second
# full-size Pod cannot coexist on the three-node production control
# plane, but startup guard semantics do not depend on CPU reservation.
# Lower requests only; keep limits, command and every behaviour knob.
requests = container.setdefault("resources", {}).setdefault("requests", {})
requests["cpu"] = "250m"
requests["memory"] = "1Gi"

# Role split moved most settings into envFrom ConfigMaps. Materialize the
# two values whose presence makes every negative guard meaningful; keep
# envFrom unchanged so the rest of the effective environment is identical.
effective = {}
for source in container.get("envFrom", []):
    ref = source.get("configMapRef") or {}
    if ref.get("name"):
        effective.update(
            kubectl_json("get", "configmap", ref["name"]).get(
                "data", {}
            )
        )
direct = {entry["name"] for entry in container.get("env", [])}
for name in (
    "GPU_FAULT_EXECUTOR_MODE",
    "GPU_FAULT_DEPLOYMENT_MODE",
):
    if name not in direct:
        container.setdefault("env", []).append(
            {"name": name, "value": effective[name]}
        )

overrides = {
    "GPU_FAULT_ALLOW_EMAIL": "false",
    "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED": "false",
    "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": "true",
    # 停用挂载的 DSN 文件（CP-3 凭据轮转把 DSN 从 Secret 文件按连接读取，
    # StoreCredentials 里文件优先于 GPU_FAULT_STORE_URL 环境变量）。生产
    # Deployment 通过 GPU_FAULT_STORE_URL_FILE=/etc/gpu-fault/aurora/postgres-url
    # + aurora-credentials 卷把线上 gpu_fault DSN 投影成文件；若不清空该变量，
    # 下面对 GPU_FAULT_STORE_URL 的独立库改写会被文件覆盖，探针实际连到线上
    # gpu_fault 库，BOOT-002 的空注册表（[]）断言就会读到线上真实集群而失败。
    # 置空后 StoreCredentials.path=None，环境变量指向的独立 guardprobe 库生效。
    "GPU_FAULT_STORE_URL_FILE": "",
}
by_name = {
    entry["name"]: index
    for index, entry in enumerate(container.get("env", []))
}
for name, value in overrides.items():
    entry = {"name": name, "value": value}
    if name in by_name:
        container["env"][by_name[name]] = entry
    else:
        container.setdefault("env", []).append(entry)

for e in container["env"]:
    if e["name"] == "GPU_FAULT_STORE_URL":
        e["valueFrom"]["secretKeyRef"]["name"] = "gpu-fault-aurora-guardprobe"
for c in spec["template"]["spec"].get("topologySpreadConstraints", []):
    c["labelSelector"]["matchLabels"]["app"] = probe
aff = spec["template"]["spec"].get("affinity", {}).get("podAntiAffinity", {})
for key in ("requiredDuringSchedulingIgnoredDuringExecution",
            "preferredDuringSchedulingIgnoredDuringExecution"):
    for term in aff.get(key, []):
        t = term.get("podAffinityTerm", term)
        t["labelSelector"]["matchLabels"]["app"] = probe
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2)
PY

python3 - "${OUT}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["spec"]
assert d["metadata"]["name"] == "gpu-fault-api-guard-probe"
assert s["replicas"] == 1
assert s["selector"]["matchLabels"]["app"] == "gpu-fault-api-guard-probe"
assert s["template"]["metadata"]["labels"]["app"] == "gpu-fault-api-guard-probe"
env = {e["name"]: e for e in s["template"]["spec"]["containers"][0]["env"]}
requests = s["template"]["spec"]["containers"][0]["resources"]["requests"]
assert requests["cpu"] == "250m"
assert requests["memory"] == "1Gi"
assert env["GPU_FAULT_STORE_URL"]["valueFrom"]["secretKeyRef"]["name"] \
    == "gpu-fault-aurora-guardprobe", "探针必须挂独立库（手册 §4.0 风险 3）"
# 文件优先级：GPU_FAULT_STORE_URL_FILE 非空时，StoreCredentials 会从挂载的
# 线上 Secret 文件读 DSN 并覆盖上面的独立库改写，探针会连到线上 gpu_fault 库。
# 置空后文件路径失效，环境变量指向的 guardprobe 库才真正生效。
assert env["GPU_FAULT_STORE_URL_FILE"]["value"] == "", \
    "探针必须停用挂载 DSN 文件，否则文件优先级会让探针连回线上库"
# EXECUTOR_MODE 必须保持 active：api.py:331-332 在 config.enabled 为假时
# 直接 return，后面所有区域守卫都不执行，9 个负向用例会集体假绿。
assert env["GPU_FAULT_EXECUTOR_MODE"]["value"] == "active"
assert env["GPU_FAULT_DEPLOYMENT_MODE"]["value"] == "regional"
assert env["GPU_FAULT_ALLOW_EMAIL"]["value"] == "false"
assert env["GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED"]["value"] == "false"
assert env["GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"]["value"] == "true"
blob = (json.dumps(s["template"]["spec"].get("affinity"))
        + json.dumps(s["template"]["spec"].get("topologySpreadConstraints")))
assert "gpu-fault-api-ha" not in blob, \
    f"反亲和/topologySpread 残留生产 label，探针会 Pending: {blob}"
print(f"guard-probe base OK -> {sys.argv[1]}")
PY
