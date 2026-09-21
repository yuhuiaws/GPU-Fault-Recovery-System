# Workflow And Remote Command State Tables

本文保全 2026-09-07 Store F2 拆表设计，并记录后续实现约束。原稿来自
`store-review-fixes` 工作树中被忽略的 `docs/review/fragments/`；原文件保留不动。
本文件位于正常文档树，不再依赖该工作树的生命周期。

本次实现从 schema v14 追加 v15（remote_command）与 v16（workflow），保留原有唤醒语义。
后续 v17 修复 dual 期间原始 JSON 与专表投影之间的条件删除兼容性。
v18 修复 legacy 到 dual 的非权威副本清理锁序，保留读写协议与两阶段迁移边界。
以下原稿的 v12、索引数量和源码行号属于历史设计背景，不是当前实现事实。
新增 DDL 必须追加连续 migration，不改写历史 migration 的 checksum。

## Migration Implementation

`gpu-fault-store-migrate` keeps schema creation explicit. Status, backfill,
counter-state changes, legacy cleanup and logical copy operations open
`PostgresStore` with `initialize_schema=False`; an incompatible source or target
schema fails validation rather than being changed as a constructor side effect.
Prepare the schema through the schema release Jobs first. Only the explicit
schema/index/diagnostic operations may perform their declared schema work.
Copy helpers close an already-open source even when destination initialization
or cleanup fails; the SQLite compatibility reader uses an encoded read-only URI.
Logical copy is not a full PostgreSQL database backup. A nonempty dedicated
telemetry table on either side is now refused before any insert, because the
control-record view does not contain that state. Use a reviewed native PostgreSQL
backup/restore for such databases; do not purge data or change production modes
to bypass the refusal. Imported legacy hot-state records are backfilled into
their dedicated tables in the same destination transaction before success.

第一阶段新增 v15：`gpu_fault_remote_commands`、数据库迁移模式记录、读视图、
双写与旧 writer 屏障。默认模式为 `legacy`。普通 deploy 只由独立 schema Job
创建结构，不启用双写、不回填、不切换数据源，也不清理历史数据。
例外只有全新数据库：schema Job 以 `--fresh-control-state-mode dedicated` 运行，在没有任何
schema 的库上建表并记录迁移历史后，直接把两类模式播种为 `dedicated`（revision 1、
`backfill_complete=true`、`legacy_purged=false`），因为空库没有可迁移的数据；实现位于
`store/postgres/fresh_control_state.py`，不改任何 DDL，因此不涉及迁移校验和与 schema 版本。
已存在 `gpu_fault_schema_migrations` 的库无论该参数如何都保持已记录的模式，仍须走下面的
显式 legacy → dual → dedicated。播种前重新核对两行都是刚建好的默认值且三张相关表为空，否则
报错停止而不是静默切换。

模式的唯一事实源是 `gpu_fault_control_state_modes`，不是各 Pod 的环境变量。
新 Store 通过数据库路由写入，通过模式感知视图读取；因此不会出现多个进程使用不同
模式缓存。`dual` 的权威写入仍在旧表，AFTER trigger 在同一事务镜像实际成功的
写入和删除；`dedicated` 只写专表，并拒绝旧客户端继续写旧表。

专表把状态、身份和租约提成列。远程命令的大 `snapshot` 不参与续租写入；
`lease_expires_at`、lease owner/token 和 `updated_at` 不建索引，以保留 HOT update。
时间戳保存为 `TIMESTAMPTZ`，附带无时区标记；兼容视图恢复原有固定六位微秒 JSON，
避免改变无时区时间、整行 CAS 和已有游标的语义。状态变更继续发送 v14 的 wakeup，
租约续期保持静默。

以下维护命令优先使用受控会话中的当前`GPU_FAULT_STORE_URL_FILE`投影凭据；
未配置文件时才使用`GPU_FAULT_STORE_URL`。文件不可读、空或编码错误立即拒绝，
不能退回启动时旧密码。显式DSN与投影目标冲突也拒绝；两PostgreSQL间逻辑复制
须使用没有进程级投影覆盖的独立受控进程，避免源和目标都被重定向到同一数据库。
不要把含密码的DSN写进命令参数或日志。先完成对应release的CPU滚动和验收，再执行迁移：

```bash
gpu-fault-store-migrate --state-table-status --state-table-kind remote_command
gpu-fault-store-migrate --set-state-table-mode dual \
  --state-table-kind remote_command --expected-state-table-mode legacy
gpu-fault-store-migrate --backfill-state-table --state-table-kind remote_command
gpu-fault-store-migrate --state-table-status --state-table-kind remote_command
```

回填默认每批 100 行、每次最多 25 批，游标和完成状态与该批写入一起提交；
尚未完成时重跑同一命令。锁冲突或无效记录使当前批回滚，不越过未处理记录。
新写入即使落在游标之前也由双写覆盖。`--restart-state-table-backfill` 显式从头复核。
未完成批次只返回进度和行数，`verification_performed=false`，不会反复扫描全部大快照。
回填完成、显式 `--state-table-status` 和 dedicated 切换前仍执行完整验证。
验证使用服务端游标，不把全部大快照一次加载到部署机内存；SQL 等待和游标读取共用
30 秒验证预算，超时不返回成功报告。连接限时 10 秒，维护锁等待限时 5 秒。
维护命令先持有 schema 共享屏障，检查当前 release 的 schema 版本、完整迁移历史与
状态表定义；不匹配或触发器停用时拒绝改变模式、回填和退役。该会话屏障不会串行化业务 DML，
但能阻止验证后发生并发 schema ensure。

旧记录可能缺少可选字段或使用早期时间字符串。条件写在同一事务锁住权威行，比较当前模型，
再以原始 JSON 执行 CAS；真实的字段变化仍会拒绝。v17 的 dual 条件删除可以匹配完整原始
JSON 或同一行的完整专表投影，不会因为标准化时间、补出可选列而错误返回未删除。

稳定期后，在排空 remote command 和 workflow lease 的维护窗口切换：

```bash
gpu-fault-store-migrate --set-state-table-mode dedicated \
  --state-table-kind remote_command --expected-state-table-mode dual \
  --confirm-state-table-change DEDICATED
```

切换持数据库屏障锁，验证回填已完成、没有缺失/额外/不一致记录，且记录是当前模型的
规范 JSON。失败不改模式；验证超时同样拒绝。等待锁的旧 READ COMMITTED 写入会读取
新模式并被拒绝，旧 REPEATABLE READ/serializable 写入会因快照过期而失败。
`dedicated` 不能退回 `legacy/dual`。`dual` 取消后再启用时会清空非权威专表副本，
重新回填，避免复用过期副本。

v18 的模式切换不再排队等待独占模式屏障；存在活动 writer 时立即拒绝，调用者在
原维护流程重试。非权威副本使用行锁与 `DELETE` 清理，避免 `TRUNCATE` 的表锁
与先读取兼容视图、后请求写屏障的业务事务形成互等。只在 legacy、显式指定 kind
且仅 DELETE 的受限路径允许副本清理；跳过的锁定行只要仍存在，就回滚整个模式变更。
该操作产生普通 MVCC dead tuples，不像 TRUNCATE 立即释放空间；保留 30 秒语句预算，
很大的旧副本仍需安排维护窗口，不以部分清理冒充 dual 已完成。

完成 dedicated 稳定期后，显式退役旧行与旧索引：

```bash
gpu-fault-store-migrate --purge-legacy-state-table \
  --state-table-kind remote_command --confirm-state-table-change PURGE_LEGACY
```

旧行分批删除，旧索引使用 `DROP INDEX CONCURRENTLY`。中断可重跑；已退役索引不再被
schema ensure 重建，也不再被业务启动校验要求。控制记录归档和跨 PostgreSQL 导出读取
逻辑数据源，不能只读取剩余的 `gpu_fault_objects`。
在线建索引在读取必需索引清单前取得同一 schema 维护屏障的排他会话锁，与 ensure、
退役和模式切换互斥；竞争时返回可重试错误，不使用过期清单重新建回已退役索引。
该锁不串行化业务 DML，成功或异常退出都会释放。

第二阶段 v16 将 workflow 的状态、版本守卫、租约和调度时间列接入同一机制，
`gpu_fault_workflow_records` 保留 dispatcher 的时间字符串与分页游标语义。

孤立命令清理由 `WorkflowStore.cancel_orphaned_remote_commands` 在同一事务中
完成命令状态更新与 workflow 审计事件写入。它按既有锁序取得命令和 workflow 的锁，
并在锁内校验完整终态快照，审计只包含实际改变的命令。该契约覆盖 `legacy`、`dual`
和 `dedicated`；审计失败不得留下已经取消、但重试不再记录的命令。
`LEASED` 命令只收到取消请求，不代表远端动作已经停止。
workflow 续租只更新租约列，仍保留“剩余租约超过一半不写”的优化；预算、merge、
predecessor、reconcile 和归档继续使用原有业务规则。
模式过滤在 UNION 外层统一执行，让 PostgreSQL 能保留各分支的有序索引路径；
`dispatch_eligible_at` 在分支中投影原有 GREATEST 表达式，分页不会因兼容视图失去
索引顺序。dedicated 的未启用 legacy 分支仍被一次性条件裁剪。

两类模式独立。remote_command 验收和稳定后，workflow 使用相同命令族，把
`--state-table-kind` 改为 `workflow`，单独执行 dual、回填、校验、dedicated 和稳定后的
退役。两个阶段及后续修复均使用连续 migration，但创建结构并不启用任何一类数据迁移。
函数、触发器和视图发生漂移时，业务启动只拒绝，不擅自修复生产结构。
触发器校验绑定当前 schema 与目标表，不接受其他 schema 的同名对象替代；
函数校验包括签名、返回类型、默认参数、STRICT、并行属性及函数正文。
迁移模式行或已建立的 registry 丢失时，ensure 也拒绝重新播种 legacy，避免把专表数据
隐藏在错误模式后；无变化 ensure 不执行模式 INSERT，不取得该表的 RowExclusiveLock。
已切 dedicated 的专表丢失也必须先恢复，ensure 不会创建空表冒充原有权威数据。

过期 fencing 的远程命令仍可保存迟到的执行证据，但结果必须匹配已发出的 lease；
不同 lease 或未领取命令的结果不能借此覆盖状态。归档必须证明相关命令和 workflow
已经终结，未知、缺失或空状态不构成终结证明。
外部后继的未知状态会保留其前驱记录，空的 `failure_handled_at` 也不能证明失败已经处理。
三份 incident-audit SQL 脚本通过 `ON_ERROR_STOP` 与显式 SQL 异常拒绝缺失参数、
不匹配确认或不满足清理条件的请求，不依赖旧版 `psql` 不支持的带退出码 `\quit`。
preview 也会列出未知状态的远程命令，purge 的任一条件删除失败都会回滚整个事务。
三份incident-audit脚本先用workflow/remote-command视图的类型化incident及predecessor
列筛选，再为选中行重建payload。非状态对象的OR分支明确排除这两类，避免外层OR再次
让全部大快照提前投影；计数查询不重建状态JSON。legacy fallback仍保留，不能据此
声称所有模式的查询都同样加速。legacy后继记录即使缺失或含NULL的incident归属，只要
引用目标workflow，也会出现在preview并阻止purge，不再被SQL的NULL比较略过。

热状态、legacy Observation 和原始证据的过期清理在选取候选时执行
`FOR UPDATE SKIP LOCKED`，刷新中的行留给后续轮次，不按旧 key 删除刚更新的版本。
原始证据仍先排除被开放incident固定的记录，LIMIT只消耗在可删除的未锁定候选上；
刷新事务回滚后，旧过期版本可以在下一轮再次被选取，不改变节点证据硬上限的既有规则。
PostgreSQL Collector状态批量更新按排序后的状态键取得事务级锁，再读取、合并并写入，
不存在的键也受保护。较新观察仍按共享合并契约保留已有成功/错误时间和未解除的
`rejected-event:`记录；不能只靠时间戳UPSERT守卫覆盖并发提交的历史。

Processor认领窗口满但被观测/租约阻塞时，后续调用以索引tuple继续查找，而不是
反复扫描同一前缀。每次仍保留头部与aging窗口，及时看到新高优先级和刚解除阻塞的请求；
继续查找时也检查更早的已就绪STRICT前驱，防止越过同lane的重试。
窗口和claim限额不增加，不在一次调用里无界重扫；每Store最多保存64个过滤作用域的
可丢弃进度提示，提交后才更新，租约与fencing仍由数据库裁决。
scope交集谓词与已有GIN表达式对齐，避免逐scope的路径扫描；是否选择GIN仍取决于
维护状态、pending list及优化器成本，不能把本机EXPLAIN当作Aurora固定计划或时延承诺。
归档候选在 LIMIT 之前排除开放的 remote command 与外部后继，避免稳定的旧记录
阻塞用完每轮配额；上传后的 SERIALIZABLE 复验仍保留。归档 withheld 计数表示实际
尝试后遭拒绝的次数，不包含在候选 SQL 中已被排除的记录。
每次 `ControlRecordArchiver.archive_one` 固定同一个保留期截止时刻，在初次打包和
上传后的最终事务中都核对实际 incident 的 `updated_at`；直接调用也不能绕过保留期。
候选选取后发生刷新，或时间戳缺失、畸形时拒绝删除，不能仅凭旧候选 ID 继续归档。

本机 PostgreSQL 验证不代表 Aurora 已建表、回填或切换；生产两阶段仍需要各自的发布、
维护审批、验收与稳定周期。

## Implementation Boundaries

- 先实现 remote_command，再实现 workflow；两个迁移面各自保留 legacy、dual、
  dedicated 状态，不在应用启动或普通 deploy 中自动回填、切换或清理旧数据。
- 保留三种 Store 后端的公共契约、锁顺序、CAS、fencing、取消和幂等语义。
- PostgreSQL 专表把频繁变化的状态与租约字段提成列；续租不能重新写大 JSON 快照。
  HOT update 还取决于更新列是否被索引、页内空间和实际执行计划，不能只凭 fillfactor
  声称性能已经改善。
- 回填必须有界、可续跑，不覆盖并发新写；切换 dedicated 前证明完整性并阻止旧 writer
  在切换后继续写回 legacy。旧索引只能在受检退役流程中删除。
- 当前版本的 wakeup、archive、dispatcher 游标、reconcile、admin 与 SQL 审计读取
  均属于迁移范围；不得只改 `_get`/`_put` 而留下直接 SQL 旁路。
- 仅在隔离 PostgreSQL 验证 SQL、并发、EXPLAIN 与 HOT 行为。线上两个阶段仍需独立的
  schema 发布、维护审批、回填验收和稳定期；本次代码实现不代表 Aurora 已迁移。

## Original F2 Design

### 为什么没有在原轮次做

审阅项 F 的完整形态是把 `workflow` 和 `remote_command` 从
`gpu_fault_objects(kind, key, payload)` 搬进各自的专表，把状态/租约/时间戳提成列，
payload 只留不变部分。这是一次带数据回填的 schema 迁移，按仓库先例
（processor queue 的 `GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE`
legacy/dual/dedicated 三态，hot-state 的 `--backfill-hot-state`）需要：新表 DDL、
双写、回填 CLI、状态 CLI、启动校验、三种后端的读写路径改写、EXPLAIN 回归测试、
以及所有直接读 `kind='workflow'` / `kind='remote_command'` 的脚本改写。
原稿的读点清单（2026-09-07 统计）：

| 文件 | workflow 引用 | remote_command 引用 |
|---|---|---|
| `src/gpu_fault/store/postgres/workflows.py` | 33 | 无 |
| `src/gpu_fault/store/sqlite/workflows.py` | 29 | 无 |
| `src/gpu_fault/store/shared/transactional_workflows.py` | 18 | 无 |
| `src/gpu_fault/store/postgres/ddl.py` | 6 | 4 |
| `src/gpu_fault/control_record_archive.py` | 6 | 2 |
| admin config 模块 | 10 | 无 |
| `src/gpu_fault/store/postgres/control_records.py` | 3 | 无 |
| `src/gpu_fault/store/sqlite/remote_commands.py` | 2 | 22 |
| `src/gpu_fault/store/postgres/remote_commands.py` | 2 | 18 |
| regional 性能套件及 cleanup | 四个文件 | 无 |
| stuck-workflow、node-health atomicity 审计 | 2 | 无 |
| incident-audit export/preview/purge SQL | 3 | 无 |

原轮次认为同时改这些点、且只能对本机 postgres:16 验证而不能对 Aurora 演练回填，
出错面太大；改为交付 F1（减少租约整行重写）、B（守卫写）、G/H（索引）三项直接缓解。

### 写放大来源

- `WorkflowRequest` 的 `step_executions`、`events`、`official_steps`、
  `completed_operations` 只增不减；原租约续期每步前后各一次整行重写，F1 后只在
  剩余租约不超过一半时写。
- `RemoteActionCommand` 内嵌 workflow 和 incident 快照；每次续租都把整份快照与
  `lease_expires_at` 一起重写。
- legacy partial expression index 以 `payload` 为输入列；payload 更新导致新
  heap tuple、TOAST 及匹配索引的写入，不能依靠 HOT 消除此成本。

### 原始目标 DDL

```sql
CREATE TABLE gpu_fault_workflows (
    request_id                 TEXT PRIMARY KEY,
    incident_id                TEXT NOT NULL,
    status                     TEXT NOT NULL,
    blocked_kind               TEXT,
    fencing_token              BIGINT NOT NULL,
    execution_epoch            BIGINT NOT NULL DEFAULT 0,
    execution_owner_id         TEXT,
    execution_lease_expires_at TIMESTAMPTZ,
    merge_revision             BIGINT NOT NULL DEFAULT 0,
    predecessor_workflow_id    TEXT,
    preempt_predecessor        BOOLEAN NOT NULL DEFAULT FALSE,
    not_before                 TIMESTAMPTZ,
    failure_handled_at         TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL,
    updated_at                 TIMESTAMPTZ NOT NULL,
    payload                    JSONB NOT NULL
) WITH (fillfactor = 70);

CREATE TABLE gpu_fault_remote_commands (
    command_id                TEXT PRIMARY KEY,
    cluster_id                TEXT NOT NULL,
    workflow_request_id       TEXT NOT NULL,
    incident_id               TEXT NOT NULL,
    fencing_token             BIGINT NOT NULL,
    execution_owner           TEXT NOT NULL,
    status                    TEXT NOT NULL,
    status_source             TEXT,
    lease_owner               TEXT,
    last_lease_owner          TEXT,
    lease_token               TEXT,
    lease_expires_at          TIMESTAMPTZ,
    cancellation_requested_at TIMESTAMPTZ,
    cancellation_reason       TEXT,
    error                     TEXT,
    created_at                TIMESTAMPTZ NOT NULL,
    updated_at                TIMESTAMPTZ NOT NULL,
    snapshot                  JSONB NOT NULL,
    result_details            JSONB NOT NULL DEFAULT '{}'::jsonb
) WITH (fillfactor = 70);
```

索引原则：租约列不建索引，尤其 `execution_lease_expires_at`、`lease_expires_at`、
`lease_token`、`lease_owner`。claim 对过期租约的条件通过 cluster/status/created_at
索引缩小范围后过滤。原稿建议把相关 legacy 表达式索引改为列索引。
snapshot 用来保留 step、workflow、incident、restart authorization 等大对象；
实际实现必须按当前模型核对所有字段与可变性，不能假定整个 workflow payload 不变。

### 原始迁移顺序

1. 新增空专表和列索引；在线索引由 index-build Job CONCURRENTLY 创建。
2. `GPU_FAULT_WORKFLOW_STATE_MODE` / `GPU_FAULT_REMOTE_COMMAND_STATE_MODE`：
   legacy（读写 objects）到 dual（双写，读专表，缺失回退）再到 dedicated。
3. 提供 backfill/status CLI，分批 UPSERT；缺失或不一致归零才允许切 dedicated，
   启动校验使用 EXISTS，而不是每个进程重复 count 全表。
4. dedicated 稳定一个发布周期后，受检清理旧行和旧 partial indexes。
5. schema、启用双写、回填验证、最终切换和旧状态清理作为独立运维阶段处理。

先做 remote_command：读点少、快照大、续租频繁，claim/complete/renew/cancel
沿用同一把 per-command 锁。再做 workflow：需同时覆盖 dispatcher 分页、
reconcile 三件套、admin 导出及 SQL 审计。

### 原轮次已交付缓解

- F1：workflow 剩余租约超过一半时不续写，三个后端一致。
- B：`save_workflow` 守卫 merge_revision、execution_epoch、fencing_token，
  `expected=` 保持整行 CAS。
- G/H：指标聚合和热查询增加覆盖索引，减少回表读取 payload。
