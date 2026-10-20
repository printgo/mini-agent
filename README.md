# research-agent

本地 Python 研究助手，支持普通聊天、长期记忆，以及需要人工审批的研究工作流。

V18 增加离线报告导出与交付说明。主程序横幅及新任务 checkpoint 协议仍为 V16.0；V17 验收和 V18 导出是独立工具，不迁移已有任务。

## 安装与配置

当前验证环境：Windows PowerShell、Python 3.13.15。
依赖仅精确固定两个 LangGraph 包，其余依赖尚未完整锁定。

在 PowerShell 执行：

```powershell
Set-Location 'E:\agent-learn\research-agent'

# 已存在虚拟环境时，跳过创建。
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.13 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 保留已有配置。
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

编辑项目根目录的 .env，替换 API Key 和两个模型名称的占位符。
模型名称应使用账号实际可用的值。
DEEPSEEK_MEMORY_MODEL 不要留空；删除整个变量时才会继承主模型。
当前进程环境变量优先于 .env；修改后重新启动。

- DEEPSEEK_API_KEY：主程序必填，缺失会在导入阶段失败。
- DEEPSEEK_BASE_URL：接口地址；客户端还会携带 thinking disabled 参数，需服务支持。
- DEEPSEEK_MODEL、DEEPSEEK_MEMORY_MODEL：主模型和记忆模型。
- LLM_TIMEOUT_SECONDS：有限正数，默认 30 秒。
- LLM_MAX_RETRIES：非负整数，默认 1。
- LLM_MAX_OUTPUT_TOKENS：正整数，默认 4096。
- TAVILY_API_KEY：web 搜索所需；缺失不阻止启动，但搜索会失败。
- GITHUB_TOKEN：可选，公开仓库也可匿名访问。

离线测试、验收、导出均不需要 .env 或模型服务。

## 启动和研究流程

```powershell
.\.venv\Scripts\python.exe -B main.py
```

普通输入用于聊天；研究输入示例：

```text
/research 比较 ReAct 论文摘要与指定 GitHub 仓库 README 的实现范围
```

规划后会显示任务 ID 和查询。输入 yes 批准，no 拒绝；
其他输入暂停并保留 checkpoint，之后使用 /resume ID。

| 命令 | 用途 |
|---|---|
| /tasks | 最近 10 项研究任务 |
| /research 问题 | 新建计划并进入审批 |
| /report ID | 状态与报告 |
| /state ID | checkpoint、待审批与恢复条件 |
| /resume ID、/retry ID | 恢复任务，两者执行同一恢复逻辑 |
| /cancel ID | 取消 planned 或可恢复任务，取消后不能恢复 |
| /failures ID | 最近 20 项失败记录 |
| /reviews ID | 各轮草稿与审查 |
| /documents ID、/chunks ID | 文档与分块预览 |
| /chunk ID D1C1 | 指定任务内的完整分块 |
| /evidence ID | 旧版证据表；新任务主要看 documents/chunks |
| /collections ID、/generations ID | 采集和生成缓存 |
| /usage ID | V16 任务逻辑调用量 |
| /limits ID 模型上限 查询上限 | 调整 planned 或可恢复 V16 任务额度 |
| /cache ID | 缓存行数与 JSON 字节数 |
| /cache-clear ID [天数] | 清理已终止或已取消任务的缓存 |
| /cache-drop ID | 失效当前待恢复 write/review 的精确缓存 |
| exit | 退出 |

取消操作需要取得任务锁，不能强行打断另一个持锁进程。
锁文件被刻意保留，文件存在不代表任务仍在运行。

## 离线验收

```powershell
$taskId = '替换为真实任务ID'

.\.venv\Scripts\python.exe -B -m research.quality `
    --db 'E:\agent-learn\research-agent\data\memory.db' `
    --task $taskId
```

退出码：0 为规则通过，1 为规则未通过，2 为读取或参数错误。
验收包括最新轮审查、引用范围、来源覆盖、正文长度及报告一致性。
正文长度按 Python 字符数计算，不包含标题和参考来源。

## V18 导出

先创建目标目录，再运行独立导出命令。目标必须是尚不存在的 ZIP 文件。

```powershell
New-Item -ItemType Directory -Path 'E:\agent-learn\research-agent\exports' -Force | Out-Null
$taskId = '替换为真实任务ID'

.\.venv\Scripts\python.exe -B -m research.export `
    --db 'E:\agent-learn\research-agent\data\memory.db' `
    --task $taskId `
    --output "E:\agent-learn\research-agent\exports\report-$taskId.zip"
```

默认仅导出 completed 且通过 V17 验收的任务。
需要保留未通过验收的终态资料时，显式使用快照模式并选择新文件名：

```powershell
.\.venv\Scripts\python.exe -B -m research.export `
    --db 'E:\agent-learn\research-agent\data\memory.db' `
    --task $taskId `
    --output "E:\agent-learn\research-agent\exports\snapshot-$taskId.zip" `
    --snapshot
```

快照模式只允许 completed、partial、cancelled、failed；
即使验收通过，显式快照也始终标记为 snapshot。
规划中、运行中和待审批任务不能导出。

ZIP 内固定包含：

- report.md：报告、任务状态与验收说明。
- evidence.json：任务、全部审查记录、本任务全部已保存分块及来源元数据。
- quality.json：同一业务快照的 V17 验收结果。
- manifest.json：格式版本、导出时间、类型及三个数据文件的 SHA256。

全部分块不等于全部被引用。审查 created_at 是该轮记录首次创建时间。
哈希校验不包含 manifest 自身，也不提供数字签名或来源认证。
文件使用 UTF-8；已有输出绝不覆盖，重复导出请换文件名。
Windows 使用同目录重命名发布；其他平台要求文件系统支持硬链接。
这不构成断电持久性保证。

导出退出码：0 为成功，1 为状态或验收拒绝，2 为读取、参数或文件写入错误。
exports 目录已忽略；选择其他目录时自行管理导出文件。

## 测试

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py" -v
```

当前共 66 个离线测试，外部模型和采集接口使用测试夹具。
覆盖额度、互斥、取消、恢复、来源覆盖、审批、验收和导出。
尚未以真实模型及搜索服务验证端到端研究质量。

## 额度、来源和数据边界

默认每任务最多 3 轮写作审查、2 项补充查询；
V16 逻辑额度为 12 次写作/审查调用、8 次采集查询尝试。
规划、普通聊天、记忆处理、SDK 内部重试和正文等内部 HTTP 请求不包含在内。
失败预占不退还，缓存命中不重复计量；额度不是 Token 或实际费用预算。
修改模型配置可能改变缓存哈希，导致再次调用模型。

研究 web 正文只读取代码白名单中的 HTTPS 主机。
其他结果或正文失败会回退搜索摘要并记录 warning；
warning 会进入 errors，即使 Critic pass 也可能最终为 partial。
普通聊天网页工具不采用同一套研究白名单。
论文只读取 arXiv 摘要；GitHub 只读取公开仓库元数据和 README，
元数据不能替代 README 满足来源覆盖。单文档最多保存 12000 字符。

规则通过、Critic pass、合法引用均不证明语义真实性。
验收和导出只检查业务数据库快照，不确认 checkpoint 已经终止。
外部响应到缓存提交之间的崩溃窗口仍可能导致重复请求。
修改报告格式或固定局限说明时，必须同步更新验收基线。

业务数据位于 E:\agent-learn\research-agent\data\memory.db，
恢复状态位于 E:\agent-learn\research-agent\data\research_checkpoints.db。
备份前退出所有相关进程，再一起备份两个数据库；
不要单独删除 checkpoint 数据库。