# BOOT-001~010 一次性守卫探针夹具

`docs/区域模式端到端验收测试用例.md` §4.0 的可执行实现。

**为什么这些脚本在仓库里、而不是像早期那样临时写进 `/tmp`**：
2026-08-04 的执行中，机器在跑完 `BOOT-009` 之后重启，`/tmp` 被整体清空。
夹具脚本、派生清单、kubeconfig 全部丢失，而集群上的探针 Deployment
与临时 Secret **仍然留着**——探针在 `CrashLoopBackOff` 里空转了 8 小时、
重启 102 次。清理流程如果依赖只存在于 `/tmp` 的脚本，
就会在最需要它的时候不可用。所以：

- 判定与清理脚本一律放仓库，随代码走；
- 只有**派生产物**（`guard-probe-base.json`）和 kubeconfig 允许放 `/tmp`，
  因为它们都能用 `derive.sh` / `aws eks update-kubeconfig` 一条命令重建；
- `cleanup.sh` **不读任何 `/tmp` 里的文件**，只按名字删集群对象，
  因此在夹具全丢之后仍然能独立跑完。

## 文件

| 文件 | 作用 |
| --- | --- |
| `derive.sh` | 从生产 Deployment 派生探针基线清单到 `/tmp/guard-probe-base.json`，并做 6 项断言 |
| `mutate.py` | 单变量变形器（`del` / `set` / `sref`） |
| `registry.py` | 生成探针用的 registry JSON（token 长度、`enabled`、缺字段、拼错字段） |
| `reset.sh` | 删探针并**轮询到 Pod 数归零**（`delete --wait` 不等 Pod） |
| `assert.sh` | 轮询判定"不 Ready + 日志含指定文本"，label 下 >1 个 Pod 时拒绝判定 |
| `cleanup.sh` | 全组跑完后的清理；不依赖 `/tmp`，可独立执行 |

## 用法

```bash
export AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export CPU_KUBECONFIG=/tmp/gpu-fault-control-plane.kubeconfig
export NAMESPACE=gpu-fault-system
export CPU_EKS_CLUSTER_NAME='<cpu-eks-cluster>'
aws eks update-kubeconfig --region us-west-2 \
  --name "${CPU_EKS_CLUSTER_NAME}" \
  --kubeconfig "${CPU_KUBECONFIG}"

cd scripts/boot-guard-probe
./derive.sh                       # P4
./reset.sh                        # 每个用例之前
python3 mutate.py del GPU_FAULT_REGIONAL_CLUSTERS_JSON \
  | kubectl --kubeconfig "${CPU_KUBECONFIG}" -n "${NAMESPACE}" apply -f -
./assert.sh 'regional mode requires GPU_FAULT_REGIONAL_CLUSTERS_JSON'
...
./cleanup.sh                      # 全组跑完
```

P2（建独立库 `gpu_fault_guardprobe` 并执行 schema ensure）与 P3
（写探针 DSN Secret）见手册 §4.0。它们要在生产 Pod 内执行以免口令进
本地 argv，不适合脚本化成可能被误用的形式。
