# GPU 多机多卡训练故障自动化处理

本仓库实现面向大规模 GPU 训练集群的故障采集、策略判定、隔离、恢复编排、
训练任务恢复和验收工具。

## 当前支持范围

当前唯一生产交付形态是：

```text
区域 CPU EKS 控制面 + Aurora PostgreSQL + NLB
                    |
                    +-- HyperPod EKS GPU 数据集群 A
                    +-- HyperPod EKS GPU 数据集群 B
                    +-- ...
```

不应从仓库中的历史代码或示例推断额外支持范围：

- HyperPod 单集群一体化只用于迁移、Canary 和历史环境维护。
- 通用 Kubernetes 部署当前未实现为生产交付能力。
- HyperPod Slurm 编排当前不支持。
- GPU 节点上的 systemd Agent 属于区域数据面，不是独立部署架构。

## 方案硬约束

以下约束不是可调默认值：

1. 受管 HyperPod GPU 集群必须设置 `NodeRecovery=None`。
2. 本方案永不调用 `BatchReplaceClusterNodes`；节点替换只使用已纳管 warm spare。
3. HyperPod Job Auto Restart、EKS auto-resume 和 Slurm auto-resume 必须禁用。
4. 训练任务恢复由本方案的 Kubernetes Adapter 和 restart budget 独占管理。
5. 未知策略、缺少证据、版本不一致或 workload 状态未知时，破坏性动作必须
   fail closed。

## 核心链路

```text
Kernel / Fabric Manager / DCGM / Host / Workload
                         |
                         v
              Regional ingress and queue
                         |
                         v
          Policy + Incident + Workflow orchestration
                         |
                         v
       Per-cluster Executor + signed Node Action Agent
                         |
                         v
          Validation + scheduling/workload recovery
```

主要能力：

- NVIDIA XID Catalog 和 Fabric Manager SXID 策略。
- GPU UUID、节点、attempt、workload 和 fabric partition 关联。
- 多副本 processor、lane lease、fencing token 和幂等 workflow。
- GPU 服务 quiesce、GPU reset、节点 reboot 和 warm-spare 故障转移。
- Completion Watcher、训练重启预算和 GPU 数量一致性门禁。
- SES 通知、AMP/Alertmanager/SNS 观测和长期证据归档。

## 文档入口

- [文档索引](docs/README.md)
- [概要设计](docs/概要设计.md)
- [详细设计](docs/详细设计.md)
- [NVIDIA 策略供应链与实现](docs/components/nvidia-policy.md)
- [部署和运维手册](docs/部署和运维手册.md)
- [部署和运维手册逐章解读](docs/部署和运维手册逐章解读.md)
- [环境变量参考](docs/环境变量参考.md)
- [扩展指南](docs/扩展指南.md)
- [Collector 说明](COLLECTORS.md)
- [训练任务示例](examples/README.md)
- [部署目录](deploy/README.md)
- [脚本与工具](scripts/README.md)
- [贡献指南](CONTRIBUTING.md)

## 构建和验证

要求 Python 3.12。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev,collectors,postgres,performance]'
make check
```

常用门禁：

```bash
make docs-check
make architecture-check
make mypy-check
make test-parallel
make artifact-check
```

`make artifact-check` 会重新构建唯一 wheel 和节点安装 bundle，并验证二者与
`src/gpu_fault` 的模块摘要一致。发布时必须使用该次构建产生的同一份内容寻址制品，
不能在上传或 pin 之间再次构建。

## 部署

生产部署从[部署和运维手册](docs/部署和运维手册.md)的区域全新部署顺序开始。
不要直接执行目录级 `kubectl apply -f deploy/`，也不要直接 apply renderer 输入或
测试清单。

区域发布、升级和回滚入口：

```bash
deploy/control-plane/regional/rollout-regional-release.sh
```

部署前必须显式确认目标 Kubernetes context、AWS Region、release metadata、区域共享
Runtime Profile、数据库连接、集群注册信息和节点安装制品。
AWS Region 必须由部署操作者填写，不能从当前 shell、kubectl context 或示例文件推断；
区域编排器会在任何 apply 前校验 CPU/GPU EKS ARN 与 HyperPod 归属。

## 训练任务提交

支持两种受管提交方式：

```bash
gpu-training-submit customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1
```

或先注入托管元数据，再由客户执行 apply：

```bash
gpu-fault-workload-annotate customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1 \
  --training-container trainer \
  -o /tmp/customer-job.managed.yaml

kubectl apply --dry-run=server -f /tmp/customer-job.managed.yaml
kubectl apply -f /tmp/customer-job.managed.yaml
```

客户训练镜像不是本方案控制面镜像。训练镜像只需满足对应框架、NCCL/EFA、
checkpoint 和训练命令要求。

## 安全

- 不得在生产 GPU 上制造真实硬件损伤。
- `/dev/kmsg` 注入只能在批准的隔离测试节点和维护窗口执行。
- execution token、cluster token、Node Action key、数据库密码和私有 CA
  不得写入源码、文档、命令历史或普通 artifact。
- 测试、Canary、probe 和 fault-injection 清单不属于生产部署资产。
- 任何节点动作都必须经过目标节点、boot/incarnation、fencing token、workload
  和维护窗口门禁。
