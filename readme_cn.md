# CodePilot

CodePilot 是基于 LangGraph 的终端编码助手。进入项目目录即可与模型连续对话，让它阅读代码、修改文件、执行命令和测试；修改在 Docker 工作副本中完成，由你审查后应用到本地仓库。项目也保留了用于编码任务评测的实验入口。

## Quick Start

### 1. 安装

需要 **Python 3.11+、Git 和已启动的本机 Docker**。使用并行子 agent 还需 Linux FUSE 3（`/dev/fuse`、`fusermount3`、`libfuse3`）。以下命令在 CodePilot 源码目录执行：

```bash
python3.12 -m venv .venv-cli
source .venv-cli/bin/activate
python -m pip install -e '.[cli]'
docker build -f containers/cli.Dockerfile -t codepilot-cli:local .
```

上例使用 Python 3.12；也可替换为已安装的 Python 3.11+。

### 2. 打开项目

保持上述虚拟环境已激活，进入要操作的 Git 仓库：

```bash
cd /你的项目路径
codepilot
```

默认使用当前 Git 仓库作为工作区，在子目录启动时自动定位仓库根目录。也可显式指定：

```bash
codepilot --repo /其他项目路径
```

### 3. 登录并开始对话

在 `codepilot>` 提示符后输入 `\login` 或 `/login`，依次填写：

1. **API Key**：隐藏输入。
2. **Base URL**：服务的 API 基础地址，例如 `https://api.deepseek.com` 或网关提供的 `https://你的服务/v1`，不要填写完整的 `/chat/completions` 路径。
3. **模型名**：服务端提供的模型 ID，须支持 OpenAI 兼容 Chat Completions 和工具调用。

登录会保存配置，下一条消息才实际调用模型。之后直接输入需求：

```text
介绍一下这个项目的结构。
找到登录逻辑，解释它的执行流程。
为这个函数补充参数校验，并运行相关测试。
```

切换模型或思考模式：

```text
/model deepseek-flash
/effort on
/effort off
/effort default
```

`/model`、`/effort` 不带参数时会提示输入；留空查看当前值。切换保留对话，从下一次请求生效，并保存供后续会话使用。思考开关须由所选模型支持，`default` 恢复服务端默认设置。

### 4. 审查并应用修改

```text
/diff
/apply
/exit
```

### 5. 配置AGENT.md
每一轮对话


`/diff` 查看工作副本的修改，`/apply` 才将修改应用到本地仓库；若相关本地文件已被其他操作修改，会停止应用并提示冲突。执行过程中可按 `Ctrl+C` 中断当前请求，已完成的工作副本修改保留。

## 主要功能

- **项目指令**：创建会话时自动读取工作区根目录的 `agent.md`（缺省时兼容 `AGENT.md`），仅在首次模型调用时提供全文，后续调用不再注入；子 agent 也只在自己的首次调用中接收。恢复会话不会重复发送已提供的指令；修改文件后通过 `/new` 加载新版本。
- **多轮对话与编码**：连续追问、阅读和编辑文件、执行 shell 命令及测试，显示工具执行状态。
- **模型配置**：交互登录，随时切换模型和思考模式。
- **工作区隔离**：基于本地仓库创建 Docker 副本，保留初始未提交内容，支持 diff 审查和显式应用。
- **会话恢复**：通过 SQLite 保存对话、计划和记忆，退出后可恢复原会话及其容器工作副本。
- **上下文与记忆**：保存会话内固定指令，长对话自动压缩为工作记忆，按需查询完整归档。
- **并行子 agent**：将独立子任务分配给不同 agent，在独立副本中执行，并限制其可修改文件。
- **实验与评测**：单任务运行、执行轨迹记录、补丁收集和 SWE-bench 评测入口。

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `\login` 或 `/login` | 配置或更换 API Key、URL 和模型 |
| `/model [模型名]` | 切换模型 |
| `/effort [on\|off\|default]` | 开启、关闭思考或使用服务端默认值 |
| `/begin` … `/end` | 输入多行请求，结束标记须独占一行 |
| `/save 指令内容` | 保存当前会话的固定指令，并作为本轮请求执行 |
| `/memory` | 查看记忆；支持 `/memory edit ID TEXT` 和 `/memory delete ID` |
| `/context`、`/config {JSON}` | 查看上下文用量、调整压缩配置 |
| `/archive {"keyword":"关键词","limit":5}` | 查询当前会话归档 |
| `/sessions`、`/new`、`/resume ID` | 列出、新建、恢复会话 |
| `/diff`、`/apply` | 查看修改、应用到本地仓库 |
| `/retry` | 继续已暂停的请求 |
| `/help`、`/exit` | 查看帮助、退出 |

也可在终端列出或恢复会话，或执行一次请求后退出：

```bash
codepilot --list
codepilot --resume 会话ID
codepilot --request '解释项目入口'
```

恢复会话使用原工作区，不受当前启动目录影响；恢复依赖原容器副本仍然存在。`/new` 从本地仓库当前状态创建新会话，不继承旧会话记忆。

## 配置与使用说明

- **配置存储**：默认在 `~/.local/state/codepilot/`，可用 `--state-dir PATH` 更改。密钥以本机明文单独保存在 `credentials.json`，文件权限为 `0600`，不写入会话历史。
- **运行环境**：默认镜像只有基础工具。运行项目测试所需的依赖应提前加入自定义镜像，再用 `--image 镜像名` 启动；容器执行时禁用网络。启动后的本地编辑不会自动同步到工作副本。
- **模型上下文**：默认预算面向 1M 窗口，压缩阈值为 320k、目标为 128k、单次输出上限为 32k。换用较小窗口的模型时，用 `--context-config FILE` 或 `/config {JSON}` 调整相应预算。
- **网关适配**：`--model-config FILE` 可配置 `name`、`base_url`、`effort` 和 `thinking_parameter`。思考参数自动适配 DeepSeek、Qwen/DashScope 和 OpenAI；自定义网关可指定 `thinking_parameter` 为 `thinking`、`enable_thinking`、`reasoning_effort` 或 `chat_template_kwargs`。
- **当前边界**：输出按完整响应显示，尚无逐 token 流式输出；仅支持 Git 仓库，不支持子模块。模型声明完成不代表测试或外部评测通过。

## 实验入口

在 CodePilot 源码目录装实验依赖后运行：

```bash
python -m pip install -e '.[dev,evaluation]'
python -m scripts.run_dev --limit 1 --max-model-calls 20 --max-parallel-subagents 4
python -m scripts.analyze_dev
```

实验入口需要另行准备模型服务与任务镜像，默认连接 `http://localhost:8000/v1`，与交互 CLI 分开配置。执行轨迹默认在 `trajectory/`，预测和评测结果默认在 `outputs/dev/`。
