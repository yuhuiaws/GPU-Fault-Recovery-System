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

开发checkout统一执行`make deploy-host-setup-online`：先验证路径和版本配置，再用系统Python 3.12
并行准备锁定Python/admin CLI及隔离的供应链/PromQL工具环境；工具包安装与promtool下载也可重叠，
同一venv内的pip写入仍串行。任一步失败均等待已启动任务结束后返回失败。版本、依赖和固定archive
摘要验证通过的工具可复用；在线Python初始化仍可能联网。CI保留独立工具目标，
签名离线`deploy-host-setup`不增加联网或本地测试工具要求。

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
受信本地production候选在个人commit或main候选不可用时运行等价本地完整门禁；该门禁先
运行static，再按部署机CPU预算并行执行普通pytest、PostgreSQL stress和source-only
artifact构建。static内部把Ruff、mypy、compile、架构、契约、文档、部署配置、YAML、
Shell和安全检查拆成独立组；全部通过后才构建运行镜像并签名发布。
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
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault-staging \
  --approve-profile-plan <停止信息里打印的 plan_sha256> \
  --reference CHG-12345
```

审批与发布在同一条命令里完成；只有 `profile-plan.json` 已存在且摘要一致时该参数才被接受。不得通过隐藏参数或环境变量注入审批引用。
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

`make check`的最终全量测试和`make test-parallel`按CPU数量选择4到16个worker，
可用`PYTEST_XDIST_WORKERS=16`显式指定；使用`--dist=worksteal`且不连接外部
PostgreSQL。`make test-postgres`固定以`-n 0`串行执行，不继承普通测试的worker预算；
stress内部的并发竞争仍由其专属参数控制，不等于并发运行共享数据库的pytest用例。
本机release门禁具有有效隔离授权时，`test-postgres-stress`通过
`POSTGRES_TEST_PARALLEL=1`委托独立实例分片：默认按核数四分之一取4至16组（与`PYTEST_XDIST_WORKERS`
同一推导），`POSTGRES_TEST_WORKERS`允许1至16组，每组独立PostgreSQL 16、私有授权和`-n 0`进程。所有分片必须具有一致的
完整发现清单，实际选择不重复、不遗漏，且stress没有skip，才算整个门禁通过。
外部测试连接和自定义impact命令保持原路径；直接运行本机并行入口可使用：

```bash
make test-postgres-stress-parallel POSTGRES_TEST_WORKERS=8
```

该入口不把已有外部数据库当作分片，也不允许用未经验证的继承连接代替本机隔离授权。
CI的既有PostgreSQL shard和下面的coverage追加路径仍保持串行，不混用并行入口的回执。
native测试夹具必须在schema初始化和配置模式的Store启动前清理上一用例的数据，
包括首次使用共享schema缓存的情况；legacy用例的残留不能依赖其他用例先完成回填。
数据库扩展等环境变化应限定在用例拥有的临时数据库，不改变同分片后续用例的前提。
不得为修复测试顺序依赖而放宽生产Store的schema或dedicated启动校验。
`make coverage`先并行采集非PostgreSQL覆盖率，再串行追加隔离PostgreSQL 16
测试库覆盖率，最后统一强制
生产范围的78%组合floor，并对`config/ci-unit-gate.json`的`coverage.module_floors`执行
per-module floor；production、runner 两个完整范围还分别强制语句和分支达到95%。
runner 的高覆盖不能抬高生产范围的比例。文档和CI契约测试由`make docs-check`、`make ci-tooling-check`
独立执行，不重复计入coverage。

per-module 比例严格使用 `(covered_lines + covered_branches) /
(num_statements + num_branches)`；`num_partial_branches` 只是部分覆盖的分支行诊断，
不能代替全部未覆盖分支。缺失、负数或与总量不一致的计数会直接拒绝。
每组还会对照当前工作树的完整源码清单；漏报组内文件或添加不存在的高覆盖文件都不能
通过门禁，不能只检查报告中恰好出现的文件。
95% 门禁分别衡量语句和分支，使用
`python -m tools.coverage_objectives --coverage-json <report> --scope production`
或 `--scope runner` 查看固定范围的结果；独立报告命令加 `--require-target` 才返回失败。
正常 CI 合并与 `make coverage` 已强制两个范围的95%门禁，不依赖手工执行报告命令，
也不会降低或替代原有 CI floor。

场景覆盖按照 [独立需求矩阵](docs/components/scenario-coverage.md) 统计，
不等于代码覆盖率。矩阵区分设计、实现、本地验证和未测量的 LIVE 验证；
不得用本地模拟替代真机结论，未完成项也不得从分母移除。

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
重签和上传；聚合`unit` job最后执行`coverage combine`，同时强制生产78%组合floor、
per-module floor 和两范围的95%语句/分支门禁，再生成
fault report和签名unit gate。deploy-host-only管理员源码只进入deployment shard；
文档或`.github/`变化由当前static验证，可复用六个历史shard。所有pytest shard保留
`--durations`结构化证据。仓库已配置较高规格Runner时可设置`CI_TEST_RUNNER`及匹配的
`CI_PYTEST_WORKERS`，未设置时保持`ubuntu-latest`和4个worker。

分片回执必须具有完整 discovery、实际选择清单、成功的 setup/call/teardown 和相符的
内容身份；必跑 PostgreSQL stress 不允许 skip。历史复用保留首次执行的 producer、
源码身份及 session，schema-2 聚合报告只声明当前验证身份，不把旧回执改写为新执行。
普通分片的整数worker预算必须匹配实际及已记录的请求数量；`auto`/`logical`须同时
记录原请求模式和解析出的实际进程数。非法、空值或被缩小的整数预算不能通过回执校验，
PostgreSQL仍显式串行`-n 0`，不继承普通测试的进程数。
签名与TLS夹具使用的`cryptography`显式属于`dev`及部署机测试依赖锁，不进入CPU/GPU
运行时依赖。新测试不得依赖开发机偶然已安装、但干净锁定环境缺少的包。
时间戳等动态参数须提供稳定的pytest ID，不能让同一源码的收集清单随启动时刻变化；
不因此固定或删减被测的数据值。可选PostgreSQL驱动边界同时验证完整导入图的收集，
而不只扫描直接import。缺少驱动时离线I/O守卫仍执行，真正SQL操作仍要求真实依赖；
收集成功不代表运行了PostgreSQL测试。
测试默认清除从部署进程继承的`KUBECONFIG`，需要Kubernetes配置的用例必须显式设置
自己的本地夹具。该隔离不替代集群命令守卫，也不授权测试访问部署机上的真实集群。
所有分片都可能通过跨域测试执行 runner，因此 runner 源码变化保守失效六个分片；
纯 deploy-host 修改仍只失效 deployment，不扩大应用 wheel 的包含范围。

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
