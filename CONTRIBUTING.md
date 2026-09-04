# Contributing

修改前按变化类型选择文档：

| 变化 | 必读文档 |
|---|---|
| 新增 operation、channel、Store、路由、插件或指标 | [GPU Fault 扩展指南](docs/扩展指南.md) |
| 修改 Manifest、renderer、systemd、AWS 资源、配置模型或管理员 CLI | [开发者部署实现](docs/开发者部署实现.md) |
| 修改 Release CI、签名、`dist/`制品或部署机离线bundle | [CI 发布流程](docs/CI发布流程.md)和[开发者部署实现](docs/开发者部署实现.md) |
| 完成代码修改并部署到staging或生产验证 | [EC2源码统一部署流程](docs/EC2源码Staging复现流程.md) |
| 同时新增代码能力和生产资源 | 两份都读，两套门禁都执行 |
| 只部署已有 release | [管理员快速部署](docs/管理员快速部署.md) |

operation/channel registry、显式授权、生产资源生命周期和管理员入口都不能只在调用侧
或线上增加字面量。扩展指南决定能力登记点，开发者部署实现决定生成、发布、验证、
回滚和卸载方式。

文档职责和权威顺序见 [docs/README.md](docs/README.md)。历史材料不得作为当前
实现依据。

`src/**/*.py`、`tests/**/*.py`、`scripts/**/*.sh` 和生产部署脚本受
[代码与文档影响契约](docs/code-doc-contracts.yaml)约束。修改这些文件时：

- 同时修改契约列出的相关文档；或
- 在 Pull Request 正文中填写 `Documentation-Impact: none`，并在
  `Documentation-Impact-Reason` 中说明为什么公共行为、命令和验收契约均未变化。

不能用空值、`N/A` 或 `TODO` 代替原因。CI 会基于目标分支和当前提交的 Git diff
执行 `scripts/check-doc-impact.py`。

本地确认属于纯内部改动时，可以显式运行：

```bash
GPU_FAULT_DOC_IMPACT=none \
GPU_FAULT_DOC_IMPACT_REASON='仅重构内部实现，公共行为和命令未变化' \
make PYTHON=.venv/bin/python check
```

日常开发先按相对目标分支的实际差异运行反向影响选择：

```bash
make test-impact BASE=origin/main
make regional-impact-plan BASE=origin/main
```

`test-impact`执行相关静态检查和pytest；`regional-impact-plan`只输出受影响的区域用例，
不会自动执行live或destructive case。无法匹配的文件、共享契约、Python/依赖/runtime
image、schema/事务变化或跨越三个以上影响域时会fail closed并升级为完整门禁。
统一staging release内部只计算一次带摘要的影响计划，测试执行和regional计划共同消费；
只有计划要求PostgreSQL时才启动隔离PostgreSQL 16。
受信本地production候选在个人commit或main候选不可用时运行等价本地完整门禁；该门禁把
static、普通pytest和PostgreSQL stress并行执行，static内部再把Ruff、mypy、compile、
架构、契约、文档、部署配置、YAML、Shell和安全检查拆成独立组；全部通过后再构建artifact。
干净`HEAD == origin/main`可先验签并消费同commit main CI候选，GitHub Release同样只晋级
已经由签名main CI gate证明等价static、coverage、artifact和PostgreSQL stress门禁的
候选，不重复测试。dirty源码和个人clean commit不查询GitHub。完整区域验收只跟随首次
上线、重大架构变化或影响计划明确要求执行。规则与说明见
[变更影响与测试选择](docs/变更影响与测试选择.md)。

管理员首次部署由一个ARN命令完成基础资源、release build和应用部署：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

首次建站需要覆盖容量默认值时，可以额外传入严格、权限`0600`的
`--config <AdminConfig.yaml>`；不传仍是原四参数默认路径。已有站点的容量变化使用
`gpu-fault-admin config --state-dir ... --reference ...`，详见
[管理员容量配置](docs/管理员容量配置.md)，不得通过shell环境变量或`kubectl set env`
替代。

开发者修改代码后也使用同一四参数命令；dirty/clean等级、影响测试、构建、验签和部署
由内部流程选择：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

Runtime Profile策略变化时，首次deploy生成
`<state-dir>/release-deploy/profile-plan.json`并停止。审核后运行：

```bash
gpu-fault-admin approve-profile \
  --state-dir /secure/gpu-fault-staging \
  --plan-sha256 "$(jq -er '.plan_sha256' \
    /secure/gpu-fault-staging/release-deploy/profile-plan.json)" \
  --reference CHG-12345
```

随后重跑原四参数deploy。不得通过隐藏参数或环境变量注入审批引用。
完整管理员流程见[Runtime Profile变更审批](docs/管理员Profile变更审批.md)。

GitHub Actions从触发、OIDC/ECR、质量门禁、制品签名到`gpu-fault-release` artifact
交付的完整顺序见[CI 发布流程](docs/CI发布流程.md)。
开发者修改代码后的最短测试、staging部署和同制品生产晋级顺序见
[EC2源码统一部署流程](docs/EC2源码Staging复现流程.md)。

`release-build`、`release-deploy`、artifact路径、site生成路径和release-ref属于内部
发布实现，不是普通开发者或管理员参数。

`gpu-fault-admin`由独立deploy-host distribution交付，不属于Control Plane Runtime
wheel。只修改管理员部署代码时必须验证deploy-host bundle和deployment影响域；不得因此
重建或滚动未变化的应用Runtime组件。内容寻址deploy-host payload与Git commit授权分离；
已有站点仅deploy-host变化时只更新部署机环境并执行只读preflight。

`make check`的最终全量测试和`make test-parallel`默认使用4个worker、
`--dist=worksteal`且不连接外部PostgreSQL。`make coverage`仍是本地单进程入口：
先采集非PostgreSQL覆盖率，再串行追加隔离PostgreSQL 16测试库覆盖率，最后统一强制
当前78%的floor，并对`config/ci-unit-gate.json`的`coverage.module_floors`执行
per-module floor。文档和CI契约测试由`make docs-check`、`make ci-tooling-check`
独立执行，不重复计入coverage。

单一仓库floor可以被覆盖良好的多数模块抬起来，让整族模块贴近零覆盖也照样通过；
deployment-only的管理员模块正是这种形状，因为它们被每个runtime shard排除，
只有合并报告知道它们的真实覆盖率。因此`coverage.module_floors`为每个
deployment-only族同时声明group floor（整族不得被仓库其余部分抬起来）和file
floor（族内某个覆盖良好的模块不得替兄弟模块背书）。新增的deployment-only源码
文件必须落在某个group的glob内，否则`tests/test_ci_unit_gate.py`失败；某个group
匹配不到任何被测文件同样是失败，避免模块改名后floor被静默作废。

main CI把fresh门禁分成`runtime`、`deployment`、`fault_runner`和`postgres`四个逻辑
域；其中runtime按稳定pytest nodeid哈希拆成`runtime_0..2`，因此共有六个并行物理
shard。每个shard按自己的源码、测试、依赖和Runner环境计算内容身份，独立恢复、验签、
重签和上传；聚合`unit` job最后执行`coverage combine`并统一强制78% floor与
per-module floor，再生成
fault report和签名unit gate。deploy-host-only管理员源码只进入deployment shard；
文档或`.github/`变化由当前static验证，可复用六个历史shard。所有pytest shard保留
`--durations`结构化证据。仓库已配置较高规格Runner时可设置`CI_TEST_RUNNER`及匹配的
`CI_PYTEST_WORKERS`，未设置时保持`ubuntu-latest`和4个worker。

覆盖率可以提高，不能通过调低`COVERAGE_FLOOR`、调低`coverage.module_floors`、
跳过shard或丢弃PostgreSQL stress掩盖未测试的新分支。`file_floor`为0、group的
globs为空或`file_floor`高于`group_floor`都会被配置校验直接拒绝，因为这种floor
读起来像保证却永远不会失败。
测试质量本身也有ratchet。`make private-test-coupling-check`限制测试跨public边界
访问私有成员；`make test-source-assertion-check`限制测试把被测代码当文本断言，
也就是`inspect.getsource(...)`或读取`.py`文件后grep字符串。这类断言只要实现继续
用同样的写法就通过，行为坏掉时依然green，纯改名却red，应改成调用序列spy或可观察
结果。少数文件确实在审计静态文本（CI门禁声明的source root、gate列表、worker上限），
它们连同理由和site数记录在`test-source-assertion-baseline.json`，只能减不能增；
新增文件直接失败，条目降到0也失败，避免baseline被静默作废。

本地`make check`先运行并行static DAG，随后并行执行普通pytest与artifact构建；
`tests/test_artifact_consistency.py`只在artifact分支执行一次，避免与尚未生成的`dist/`
竞争。
Make在checkout中检测到`.venv/bin/python`时会自动使用该解释器；源码包没有`.venv`
时回退到`python3`，显式`PYTHON=...`始终优先。

只修改文档时，仍必须运行：

```bash
make docs-check
```
