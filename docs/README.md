# 文档索引

本目录按“现状事实源、操作入口、验收证据、开发规范、历史材料”区分职责。
同一事实冲突时，优先采用更靠前的类别；`docs/history/` 中的内容只用于追溯，
不得作为当前部署或实现依据。

## 管理员入口

| 任务 | 首选文档 |
|---|---|
| 初始化或检查部署机 | [CI 发布流程](CI发布流程.md)的部署机bundle章节 |
| 在单台EC2首次部署或继续升级staging | [EC2源码Staging统一部署流程](EC2源码Staging复现流程.md) |
| 首次区域部署或重建应用层 | [管理员快速部署](管理员快速部署.md) |
| 审批待发布Runtime Profile计划 | [Runtime Profile变更审批](管理员Profile变更审批.md) |
| 调整control-worker、remediation budget或telemetry spool | [管理员容量配置](管理员容量配置.md) |
| 升级、扩集群、回滚、轮换、按症状排障或退役 | [管理员日常运维](管理员日常运维.md) |
| 审批破坏性能力和查询配置边界 | [安全与参数参考](安全与参数参考.md) |
| 按类别查询环境变量含义、区域值和应用默认 | [管理员环境变量参考](管理员环境变量参考.md) |
| 逐条审计、基础设施准备和 break-glass | [部署和运维手册](部署和运维手册.md) |

正常路径不要求管理员线性阅读完整部署手册，也不要求逐段复制 CPU/REG 命令。

## 当前权威文档

| 文档 | 回答的问题 |
|---|---|
| [概要设计](概要设计.md) | 系统是什么、生产边界和硬约束是什么 |
| [概要设计 v2（闭环视角）](概要设计-v2.md) | 发现到重新纳管的每一环由谁做、能力边界在哪、缺口是什么 |
| [详细设计](详细设计.md) | 当前代码模块、协议和状态机如何实现 |
| [详细设计 v2（可编码视角）](详细设计-v2.md) | 每个模块怎么跑、数据怎么流、异常怎么处理，够据此编码联调 |
| [NVIDIA 策略供应链与实现](components/nvidia-policy.md) | Catalog 固定来源、生成摘要、动作语义和运行时门禁 |
| [CI 发布流程](CI发布流程.md) | Release workflow如何构建、签名、上传并交付部署制品 |
| [EC2源码Staging统一部署流程](EC2源码Staging复现流程.md) | 首次、dirty迭代和后续升级均用四参数`gpu-fault-admin deploy` |
| [管理员快速部署](管理员快速部署.md) | 如何用四参数`gpu-fault-admin deploy`完成首次和后续部署 |
| [Runtime Profile变更审批](管理员Profile变更审批.md) | 管理员如何审核计划SHA、批准、续跑和保存证据 |
| [管理员容量配置](管理员容量配置.md) | 如何用plan/apply或首次部署配置文件生成config-only release |
| [管理员日常运维](管理员日常运维.md) | 按任务或症状选择升级、集群变更、轮换、排障和退役入口 |
| [安全与参数参考](安全与参数参考.md) | 不可放宽的约束、能力阶段和配置事实源 |
| [管理员环境变量参考](管理员环境变量参考.md) | 管理员常用变量的分类含义、区域清单值和应用默认 |
| [部署和运维手册](部署和运维手册.md) | 详细命令、逐条审计、基础设施和 break-glass |
| [逐章解读](部署和运维手册逐章解读.md) | 如何选择并执行手册章节 |

当前唯一生产形态是区域 CPU 控制面与 HyperPod EKS GPU 数据面分离。历史
单集群一体化内容只用于迁移和兼容验证，不构成新的生产部署入口。

## 验收与测试

| 文档 | 类型 | 维护规则 |
|---|---|---|
| [区域模式端到端验收测试用例](区域模式端到端验收测试用例.md) | 规格 | 随当前实现修订；用例 ID 和标题锚点保持稳定 |
| [区域用例索引](区域用例索引.md) | 生成物 | 只展示执行顺序与规格入口，不包含内部执行状态 |
| [变更影响与测试选择](变更影响与测试选择.md) | 开发门禁 | 从代码路径反向选择pytest和受影响区域用例 |
| [故障模拟测试手册](故障模拟测试手册.md) | 操作手册 | 与故障目录和注入清单同步 |
| [性能压测验收方案](性能压测验收方案.md) | 容量验收 | 只定义模型、门槛和证据格式；真实结果保存在私有证据库 |

## 生成物

- [环境变量参考](环境变量参考.md)：完整机器清单；
  [管理员环境变量参考](管理员环境变量参考.md)：精选分类视图。两者均由
  `scripts/generate-env-reference.py` 从同一次源码提取生成。
- [区域用例索引](区域用例索引.md)：由
  `scripts/build-regional-case-index.py` 生成。
- `GPU_FAILURE_AUTOMATION_DESIGN.html`：由 `build-html.sh` 从当前文档集生成，
  作为 Release、Pages 或 CI artifact 发布，不进入源码版本控制。

生成物不得手改。修改事实源后运行对应生成器的 `--check`，并运行
`scripts/check-doc-references.py` 验证代码和测试引用。

## 代码与文档同步

`code-doc-contracts.yaml` 定义需要文档影响审阅的代码范围及其相关文档：

- `src/**/*.py`：运行时、协议、配置与模块设计；
- `tests/**/*.py`：测试行为、验收目录与限制说明；
- `scripts/**/*.sh`、`deploy/**/*.sh`：管理员执行的命令和部署流程。

`scripts/check-doc-impact.py` 在 Pull Request 中读取 Git diff。命中契约的代码
发生变化时，至少一份对应文档也必须变化。纯内部重构确实不影响文档时，必须在
Pull Request 正文中填写：

```text
Documentation-Impact: none
Documentation-Impact-Reason: 仅重构内部实现，公共行为、命令和验收契约均未变化。
```

本地 `make check` 可使用等价的 `GPU_FAULT_DOC_IMPACT=none` 和
`GPU_FAULT_DOC_IMPACT_REASON='具体原因'`。该声明不会替代 CI 中的 Pull Request
正文审计记录。

无 Git 元数据的源码包仍会执行契约结构和覆盖范围校验，但不会伪造差异结论。

## 开发规范

- [CI 发布流程](CI发布流程.md)：GitHub Release workflow、签名制品、artifact交付和
  部署消费边界。离线deploy-host安装隔离工作区Python路径、强制安装bundle项目wheel，
  并自动重建同摘要但不完整的venv；签名密码只进入最终cosign进程，不进入测试或构建。
- [EC2源码统一部署流程](EC2源码Staging复现流程.md)：代码修改后的最短验证、
  staging部署、非破坏性测试和生产晋级闭环。
- [开发者部署实现](开发者部署实现.md)：内部site、生产 Manifest、renderer、
  release pin、三层资源注册表和管理员接口要求。
- [扩展指南](扩展指南.md)：新增 operation、channel、Store、adapter、handler 或
  metrics 时的代码登记点；生产交付继续进入开发者部署实现。
- [Validation limitations](validation-limitations.md)：当前验证边界。

贡献入口见仓库根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 历史与证据

- 历史审阅输入、整改矩阵、架构批次、安全事件、真实环境执行史和未脱敏报告
  保存在仓库外的私有证据库，不属于公开文档集。
- `evidence/`：需要长期保留的精简测试证据。
  故障证据的规范索引见
  [evidence/fault/README.md](evidence/fault/README.md)；从公共规格迁出的
  非权威历史叙述见
  [evidence/regional-history/README.md](evidence/regional-history/README.md)。
- `reference/`：不能直接提交给运行时 API 的说明性参考材料。

## 机器守卫

```bash
make docs-check
```
