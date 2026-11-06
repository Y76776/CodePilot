# CodePilot

CodePilot 基于 LangGraph 实现编码 agent，使用 Docker 执行任务、收集补丁和完整轨迹。主 agent 提供 `read`、`edit`、`write`、`shell`、`finish`、`think`、`delegate` 七个工具；子 agent 提供前六个工具。

`think` 保存当前模型生成的计划。`delegate` 并行执行独立子任务，子任务使用相同仓库快照和独立上下文。子 agent 的所有仓库写入，包括 shell，必须通过文件授权检查。

## 安装

需要 Python 3.10+、本地 Docker 守护进程，以及 Linux FUSE 3（`/dev/fuse`、`fusermount3`、`libfuse3`）。多子 agent 使用宿主机 FUSE 代理，因此不支持远程 Docker 守护进程；缺少执行层权限机制时会拒绝子任务。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluation]'
```


## 运行

默认开发入口连接 `http://localhost:8000/v1`，可以使用自定义API：

```bash
python -m scripts.run_dev --limit 1 --max-model-calls 20 --max-parallel-subagents 4
python -m scripts.analyze_dev
```

`--max-model-calls` 是模型调用阈值。已启动子任务不会因为达到阈值而停止；整批返回后，主 agent 进行一次不提供工具的最终分析并结束。因此实际模型调用数可能超过阈值。`finish` 表示 agent 主动结束。

## 轨迹

所有 agent 轨迹默认位于 `trajectory/`，可用 `--trajectory-dir` 指定另一个根目录：

```text
trajectory/
  failures.jsonl
  evaluation.jsonl
  <任务ID>/
    attempt-<次数>-<唯一ID>/
      events.jsonl
      plan.md
      subagents/<子任务ID>/
        events.jsonl
        plan.md
        report.json
        changes.json
        patch.diff
```

计划只在模型调用 think 时创建或更新。`changes.json` 保存可精确恢复的内容增量；`patch.diff` 供人阅读。子模型失败时，部分修改留档且不自动接纳。主子轨迹通过运行 ID、子任务 ID 和父工具调用 ID 关联。实际模型输入输出始终记录。

预测及评测结果默认位于 `outputs/dev/`。分析脚本汇总单次运行及所有子 agent 的实际成本。

## 验证

```bash
python -m pytest tests -q
python -m mypy codepilot scripts
```

文件权限测试会创建临时 FUSE 挂载，需要主机允许 `/dev/fuse`。Docker 集成测试默认跳过；指定一个已安装、包含 Python 和 Git 的本地镜像可执行实际容器验证：

```bash
CODEPILOT_DOCKER_TEST_IMAGE=<本地镜像> python -m pytest tests/test_docker_subagents.py -q
```


## 交互 CLI（离线模型替身）

CLI 与实验入口共享 `AgentCore`、工具和 Docker 文件授权机制。CLI 不设置模型决策次数阈值；`finish` 结束当前请求并返回输入。实验入口的单任务、补丁收集和软预算语义保持不变。

本阶段仅提供 **明确标识为 `[STUB]` 的离线模型替身**。替身默认只结束请求，不会理解或完成编码任务；可以用 JSON 显式指定工具批次来演示修改。没有配置或请求 DeepSeek API，没有展示真实模型结果，也没有开展记忆正确性或编码效果对照评测。

CLI 需要 **Python 3.11+**：当前 LangMem 0.0.30 在 Python 3.10 下无法导入；实验入口仍支持 Python 3.10。需要本地 Docker，子 agent 另需上文的 Linux FUSE 前提；记忆存储不需要额外服务。

```bash
python3.12 -m venv .venv-cli
source .venv-cli/bin/activate
python -m pip install -e '.[cli,dev]'
docker build -f containers/cli.Dockerfile -t codepilot-cli:local .
codepilot --repo /absolute/path/to/repository
# 等价入口：python -m codepilot.cli --repo /absolute/path/to/repository
```

镜像需有 `python`、Git、bash 和 coreutils；可用 `--image` 指定预先安装的本地镜像，CLI 不自动拉取。默认镜像只提供基础运行环境；项目自身的测试依赖需提前加入自定义镜像。容器执行时禁用网络。

启动时打印本地仓库、会话 ID 和 Docker 工作副本。初始副本包含当前 Git 已跟踪文件及未忽略的未跟踪文件，保留本地未提交内容；不复制原仓库 `.git` 或被忽略的未跟踪文件，不支持子模块。容器内创建独立 Git 基线，不挂载本地仓库。启动后的本地编辑不会自动同步进去。

### 修改交付演示

创建 `/tmp/codepilot-stub.json`：

```json
[
  [{"name":"write","args":{"path":"codepilot-demo.txt","content":"offline stub change\n"}}],
  [{"name":"finish","args":{"summary":"[STUB] Scripted write completed; correctness unevaluated."}}]
]
```

```bash
codepilot --repo /absolute/path/to/repository --stub-script /tmp/codepilot-stub.json
```

提交一条请求后，替身按脚本执行 `write`，展示工具状态并返回输入。然后输入：

```text
/diff
/apply
/exit
```

`/diff` 展示相对会话初始快照的变化。只有显式 `/apply` 才修改本地；应用前检查所有相关文件的内容、类型和权限，任一冲突会阻止整批应用。无冲突时复用现有事务式文件增量应用，后续 `/apply` 支持增量交付；不会自动合并或覆盖本地后续编辑。二进制文件及权限变化也属于文件增量。

### 多轮、恢复与中断

普通输入提交一轮请求；多行输入使用 `/begin` 开始、独占一行的 `/end` 提交：

```text
/begin
/save 保留现有公共接口。
新功能应遵循项目命名习惯。
/end
继续检查测试。
/sessions
/exit
```

`/save` 只在输入开头且后接空白或结尾时生效，其后全部原文（包括开头空格、换行）既保存为固定指令又作为当前请求。正文中的 `/save` 不会被解析为保存命令。

```bash
codepilot --list
codepilot --resume <会话ID>
codepilot --repo /absolute/path/to/repository --request '单次请求后退出'
```

默认状态目录为 `~/.local/state/codepilot`，可用 `--state-dir` 指定。每条轨迹的 `state.sqlite` 保存消息、计划、固定指令、工作记忆、归档事件、压缩进度、上下文配置和工作副本关联。容器文件系统单独保留，退出会停止容器而不删除它。恢复先验证原容器身份和 Git 基线；副本已被删除或损坏时明确报告无法完整恢复，不重建一个空副本冒充恢复。

会话内 `/new` 从本地仓库当前状态建立新轨迹，不继承旧消息或记忆；`/resume ID` 切换到已有轨迹。恢复、列出会话本身不会调用模型或重放工具。替身脚本是显式演示输入，每次 CLI 启动从头加载，只有提交新请求或显式重试才会使用；不要将脚本当成恢复日志。

执行期间 `Ctrl+C` 会停止后续模型和工具调用，并尝试停止容器中的运行命令，返回输入状态。已完成的文件修改保留；正在执行和排队但未完成的调用记录为中断，结果未知，不承诺回滚 shell 副作用。恢复也会补记进程退出时未完成的调用，绝不自动重放。输入期间 `Ctrl+C` 只取消输入。

### 记忆、冲突、压缩与回查

```text
/memory
/memory edit <条目ID> 更新后的原文
/memory delete <条目ID>
/conflicts
/policy newer
/policy prompt
/context
/archive {"keyword":"write","offset":0,"limit":5}
/archive {"event_id":12,"content_offset":0}
```

`/memory` 列出有效和已失效条目及其来源，编辑与删除均留档；新轨迹不继承旧记忆。有效固定原文在每次主、子 agent 决策中提供，自动摘要不能将工作记忆升级为固定指令。默认同优先级冲突采用来源时间较新的条目，固定指令优先于工作记忆；摘要重写时间不改变事实时间。`prompt` 策略暂停并展示涉及条目与理由，编辑或删除后可 `/retry` 执行保存时的请求。被取代或编辑的固定指令在历史消息输入中替换为归档引用，原文仍保留在归档。压缩冲突的候选条目标记为 `proposed`，编辑候选后 `/retry`，旧的有效记忆在新状态通过检查前保持原状。

语义冲突检测可注入；离线默认替身不做语义判断。可通过 `--stub-conflicts /tmp/conflicts.json` 演示指定原文对的冲突：

```json
[{"first":" use tabs","second":" use spaces","reason":"indentation fixture"}]
```

依次输入 `/save use tabs`、`/save use spaces` 可观察替身冲突策略。理由标注 `[STUB]`；这不是模型理解能力的演示。

上下文用量是 UTF-8 字节启发式**估计**，包含系统指令、工具定义、有效固定指令、工作记忆及近期交互。服务报告用量单独记录；替身没有实际服务用量。默认触发阈值 320,000、压缩目标 128,000、单次输出上限 32,000 token，可通过 `--context-config` JSON 文件或 `/config` 修改：

```text
/config {"trigger_tokens":9000,"target_tokens":6500,"retain_interactions":1}
```

其余配置项为 `window_tokens`、`summary_window_tokens`、`max_output_tokens`、`retries`、`tool_output_chars`；默认保留最近 2 个完整交互、最多尝试摘要 2 次、工具输出上限 12,000 字符。这些是工程默认值，未证明能改善效果。文件 `read` 本身支持 `offset`/`limit` 分页；过长工具输出明确截断，完整输出保留归档。

达到阈值时使用 LangMem 短期摘要 API（离线摘要模型替身），生成目标、发现、失败尝试、验证结果和待办五类工作记忆，并检查来源引用与压缩后输入体积。保留最近完整工具批次和固定原文，移出的旧交互仍留在 SQLite 归档。首次单条超大输入或近期交互无法压缩时会暂停，需缩短输入或调整参数。摘要输入另行检查窗口，不先截掉待摘要历史。失败、无效引用或压缩后仍超目标时保留旧记忆和压缩位置，有限重试后暂停；修正参数或记忆后 `/retry` 只从下一次模型决策继续，不重放之前的工具。

归档结果包含轨迹 ID、事件 ID、`next_offset`；单个长事件会带 `next_content_offset`，用其继续读取。CLI 和 agent 的 `archive` 工具共享同一分页查询，隔离到所属轨迹，没有 embedding 或向量数据库。来源引用存在只表示原始证据可追溯，不代表内容已经证实。

子 agent 仍使用独立副本和执行层文件授权，继承有效固定指令，只接收明确委派上下文，各自管理压缩和归档。返回简短报告、子轨迹 ID 与来源事件 ID；子摘要不自动进入主工作记忆。完整子轨迹位于该会话目录的 `subagents/<子轨迹ID>/state.sqlite`，报告和补丁保留为文件。

### 扩展与验证边界

`Session.create/resume/request/retry/snapshot/archive/diff/apply` 是会话与后续评测的高层入口，可注入模型、仓库沙箱、冲突识别和摘要模型，`observer` 接收实际模型输入输出、工具、来源和压缩事件。`snapshot()` 提供输入 token 估计、服务报告用量（如有）、耗时、压缩次数；`record_resolution()` 接收外部解决判定，`finish` 不等于任务成功。本阶段不新增记忆正确性或编码效果对照实验。

`--model-config` 仅保存未来服务配置（`name`、`base_url`、`window_tokens`、`max_output_tokens`）；当前 CLI 始终使用替身，不读取 API 密钥。真实模型需通过模型注入边界接入；DeepSeek 是后续目标，需要届时重新核对模型标识、窗口与工具能力，再实现和验证真实 token 流。当前只验证事件输出通路，不能把替身事件称为真实流式模型结果。

```bash
python -m mypy codepilot scripts
python -m pytest tests/test_cli_session.py -q
CODEPILOT_DOCKER_TEST_IMAGE=<已有的Python和Git镜像> python -m pytest tests/test_cli_docker.py -q
# 完整回归还需旧实验入口的 evaluation 依赖；安装依赖不会启动效果评测。
python -m pip install -e '.[cli,dev,evaluation]'
python -m pytest tests -q
```

会话测试验证功能契约；Docker 集成测试验证真实副本、恢复、应用冲突、缺失副本和真实命令中断。它们都不验证模型解题效果。用完后可手工删除自己的 `codepilot.cli.*` 容器与相应会话目录；删除容器后该轨迹不能完整恢复。

本轮验证：Python 3.12 下完整回归 **110 项通过**，包含以本机已有 `ice-verl:base` 镜像运行的真实 Docker/FUSE 检查；类型检查与新增代码 lint 通过。默认 Dockerfile 的构建尝试因 Docker Hub 拉取 `python:3.12-slim` 网络超时而未完成，可按 `--image` 使用满足工具前提的已有本地镜像。LangMem 依赖有一条上游弃用警告。以上均为功能验证，不是记忆或编码效果评测。
