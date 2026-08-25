# Examples

本目录只保存客户训练任务输入样例，不是生产部署入口。生产组件从
`deploy/` 部署；当前唯一生产形态见
[`docs/部署和运维手册.md`](../docs/部署和运维手册.md)。

## HyperPod 训练任务

`hyperpod/three-node-pytorchjob.yaml` 是 SOURCE MANIFEST：可运行，但尚未带
`gpu-fault.io/*` 托管元数据。必须通过 `gpu-training-submit` 提交，或先运行
`gpu-fault-workload-annotate` 再 `kubectl apply`。直接 apply 会启动一个
**不受 GPU 故障控制面管理**的训练任务。

| 文件 | 类型 | 用途 |
|---|---|---|
| `hyperpod/three-node-pytorchjob.yaml` | SOURCE | 客户 YAML 的三节点训练提交示例；`GF-REGIONAL-WORKLOAD-001/002` |

这些文件使用固定 digest 的 AWS PyTorch Training DLC 作为仓库验收基线。客户可以换成
自己的训练镜像，但镜像必须提供清单内命令使用的 PyTorch、NCCL、EFA 运行时和 shell
工具。

该清单按 p5en 级节点编写：每个训练 Pod 请求
`nvidia.com/gpu: 8` 和 `vpc.amazonaws.com/efa: 16`。其他实例类型必须先修改
GPU/EFA 数量、CPU、内存和 `--nproc-per-node`，否则 Pod 可能长期 Pending。

## 相关 E2E 夹具

故障注入、hung、warm-spare、已注解任务和 q118 场景均属于测试资产，位于
`tests/manifests/training/` 或 `scripts/e2e/manifests/`。它们只能从对应测试步骤
使用，不属于客户示例。
