# GPU多节点分布式训练故障自动化处理

本仓库实现面向大规模 GPU 训练集群的故障采集、策略判定、隔离、恢复编排、训练任务恢复和验收工具。

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

## 部署建议与已验证环境

- **集群创建**：建议使用 AWS 管理控制台的 SageMaker HyperPod UI 分别创建 GPU 数据面集群和 CPU 控制面集群；创建完成后再用 `gpu-fault-admin` 部署本方案。
- **测试基线**：当前故障采集、策略、隔离、恢复和验收主要在 AWS `ml.p5en.48xlarge` H200 GPU 实例上完成；其他实例类型需重新验证 GPU/EFA、驱动、DCGM 和互联拓扑。
- **训练性能**：当前配置通常不影响已有训练性能，组件不使用 GPU 算力且 CPU/内存占用较少；节点默认每 15 秒进行 GPU/Host 轻量采样、每 60 秒上报 GPU inventory、每 5 秒检查 Fabric Manager 日志。

## 方案硬约束

以下约束不是可调默认值：

1. 受管 HyperPod GPU 集群必须设置 `NodeRecovery=None`。
2. 本方案永不调用 `BatchReplaceClusterNodes`；节点替换只使用已纳管的健康的 warm spare 节点。
3. HyperPod Job Auto Restart、EKS auto-resume 必须禁用。
4. 训练任务恢复由本方案的 Kubernetes Adapter 和 restart budget 独占管理。
5. 未知策略、缺少证据、版本不一致或 workload 状态未知时，破坏性动作必须
   fail closed。

## 核心链路

```text
Collectors / Watcher -> Regional ingress and queue -> Policy / Incident / Workflow
                     -> Per-cluster Executor / signed Node Agent
                     -> Validation / scheduling and workload recovery
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
- [管理员快速部署](docs/管理员快速部署.md)
- [管理员日常运维](docs/管理员日常运维.md)
- [安全与参数参考](docs/安全与参数参考.md)
- [概要设计](docs/概要设计.md) / [概要设计 v2](docs/概要设计-v2.md)
- [详细设计](docs/详细设计.md) / [详细设计 v2](docs/详细设计-v2.md)
- [NVIDIA 策略供应链与实现](docs/components/nvidia-policy.md)
- [部署和运维详细参考](docs/部署和运维手册.md)
- [部署和运维手册逐章解读](docs/部署和运维手册逐章解读.md)
- [开发者部署实现](docs/开发者部署实现.md)
- [环境变量参考](docs/管理员环境变量参考.md)
- [性能压测验收方案](docs/性能压测验收方案.md)
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

需要手工逐项执行时，建议按以下顺序：

```bash
make mypy-check
make architecture-check
make docs-check
make artifact-check
make test-parallel
```

`make check` 的最终全量测试同样使用4个xdist worker。隔离PostgreSQL 16测试库不参与
并行；设置`GPU_FAULT_TEST_POSTGRES_URL`后单独运行`make test-postgres`。

`make artifact-check` 构建三个独立 wheel、Node bundle 和内容寻址
`dist/current-release.json`，并验证模块边界、摘要与重复构建确定性。

## 部署：先区分角色

所有生产部署开始前必须满足以下共同前提：

1. 一个已有且至少有3个Ready节点的CPU EKS/HyperPod集群。
2. 至少一个已有GPU EKS/HyperPod集群；GPU HyperPod必须为`NodeRecovery=None`。
3. GPU VPC已有NAT出口。
4. 执行身份具有所需AWS、EKS和Kubernetes管理权限。

不要直接 apply `deploy/`。正式入口只有`make release-build`和
`make release-deploy`两个连续步骤。

### 开发者：修改代码或Profile后发布

无论由同一操作者执行，还是由CI代执行第一步，正式入口都只有以下两步：

```bash
COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<registry/repository>

make PYTHON=.venv/bin/python release-deploy \
  SITE=/secure/gpu-fault/site.yaml \
  COSIGN_KEY=/secure/release/cosign.pub
```

`release-build`以Dockerfile、固定base digest、runtime lock、platform、build args、
wheel SHA和module digest计算image input digest；registry平台和labels完全匹配才复用
OCI，否则build/push。远端cache只加速layer。它签名固定的`dist/current-*`制品，
`release-deploy`默认读取；CI keyless签名时改用固定certificate identity和issuer。

修改Profile策略时，在上述`release-deploy`命令前追加审批引用：

```bash
PROFILE_APPROVAL=CHG-12345 \
make PYTHON=.venv/bin/python release-deploy ...
```

如果尚未提供审批引用，首次运行只生成
`<site目录>/release-deploy/profile-plan.json`并停止。CI已完成`make check`、OCI构建、
Manifest v3、attestation和签名；部署机验签后执行
`deploy -> verify -> stability -> release-summary`，保存`verification-report.json`和
`stability-report.json`，不重新build或跑全量pytest。
实际变更按组件DAG滚动；Node Runtime使用Fleet waves。Profile变化在finalize前若仍有
旧Profile workload/workflow会fail closed。`NOOP`记录`SKIPPED_NOOP`并跳过稳定窗口。
普通代码发布不需要`PROFILE_APPROVAL`，也不得手工修改generated Manifest、artifact摘要或Profile版本。
完整实现见[开发者部署实现](docs/开发者部署实现.md)。

### 管理员：首次部署和日常管理

管理员仍按上述`release-build -> release-deploy`两步执行；职责分离时第一步可由受信
Release CI代执行。管理员不编辑Profile版本或artifact摘要。Region由管理员选择的集群
ARN或`site.yaml`中的`spec.awsRegion`明确给出，不从shell或当前kubectl context猜测。
完整权限和网络要求见[管理员快速部署](docs/管理员快速部署.md)。

#### 1. 首次部署

先用`deploy/aws/regional-foundation/` Terraform模块创建Aurora、AMP/SNS/SQS、
runtime/cache ECR和方案安全组，再根据输出准备`site.yaml`：

```bash
terraform -chdir=deploy/aws/regional-foundation apply
gpu-fault-admin deploy -f /secure/gpu-fault/site.yaml
```

ARN-only Python foundation bootstrap只用于迁移，必须显式追加
`--allow-legacy-python-foundation`；它也只接受CI已生成的deployable Manifest v3。

#### 2. 已有站点的单命令操作

以下每项都是独立的管理员入口：

```bash
SITE=/secure/gpu-fault/site.yaml

# 预置条件检查：只读，不修改AWS或Kubernetes
gpu-fault-admin preflight -f "${SITE}"

# 首次应用部署或部署升级：自动preflight并按release差异最小滚动
gpu-fault-admin deploy -f "${SITE}"

# 独立验收：只读验证CPU/GPU、Profile、TLS/NLB、Agent和Aurora
gpu-fault-admin verify -f "${SITE}"

# 查看当前健康、release、Profile和各集群状态
gpu-fault-admin status -f "${SITE}"

# 注册一个已有GPU集群
gpu-fault-admin join-cluster -f "${SITE}" --gpu-cluster-arn <gpu-arn>

# 注销一个GPU集群；保留该GPU EKS/HyperPod和其他集群
gpu-fault-admin remove-cluster -f "${SITE}" \
  --cluster-id <cluster-id> --confirm REMOVE_GPU_CLUSTER

# 卸载整个方案控制面和数据面，但保留底层CPU/GPU集群
gpu-fault-admin uninstall -f "${SITE}" \
  --cpu-cluster keep --confirm UNINSTALL_GPU_FAULT
```

若永久退役并连底层CPU EKS/HyperPod集群一起删除，使用更强确认：

```bash
gpu-fault-admin uninstall -f "${SITE}" \
  --cpu-cluster delete \
  --aurora-final-snapshot retain \
  --confirm DELETE_CPU_CONTROL_PLANE
```

GPU EKS/HyperPod始终保留。`deploy`失败或自动回滚后使用相同命令重跑；`join-cluster`和
`remove-cluster`也通过持久状态幂等续跑。已有site的`deploy -f`不会重新构建当前
checkout，只应用site声明的release；代码或Profile变更必须先进入开发者发布入口。
详细升级、排障和退役规则见
[管理员日常运维](docs/管理员日常运维.md)，逐对象审计和break-glass见
[部署和运维详细参考](docs/部署和运维手册.md)。

## 训练任务提交

支持两种受管提交方式：

```bash
gpu-training-submit --site /path/to/site.yaml customer-job.yaml \
  --job-id customer-job-001 \
  --attempt-number 1
```

或先注入托管元数据，再由客户执行 apply：

```bash
gpu-fault-workload-annotate --site /path/to/site.yaml customer-job.yaml \
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

## License

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
