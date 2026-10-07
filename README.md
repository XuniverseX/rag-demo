# 诊间：医疗客服 Agent 面试项目

面向 AI 应用开发 / Agent 工程师的可运行项目：**LangGraph 工作流 + 有来源的 RAG + 真实 MCP 协议 + 持久化记忆**。重点展示确认、重启恢复、幂等执行、工具超时对账和用户隔离。

所有内置医疗知识与医院业务均为合成演示。离线模式不是大模型，也不是医学能力评测；在线模式需要自行配置模型服务。

## 先跑起来

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```bash
uv sync --python 3.12
cp .env.example .env
uv run uvicorn medical_agent.api:app --host 127.0.0.1 --port 18765 --workers 1
```

打开 [聊天界面](http://127.0.0.1:18765/) 或 [API 文档](http://127.0.0.1:18765/docs)。离线模式不需要 API key。端口被占用时换一个端口。仅用一个 API worker；当前并发锁是进程内锁。

默认自动登录本地 Alice 演示账号；可切换 Bob 验证隔离。这不是现实患者认证。令牌不保存在浏览器持久化存储，只有会话 ID 保存在 localStorage；页面刷新重新获得演示令牌后恢复会话。

## 建议阅读顺序

1. [五分钟演示与学习路线](docs/demo-and-study.md)：如何现场讲与操作。
2. [技术细节手册](docs/technical-guide.md)：数据流、源码入口、关键机制、失败场景和扩展。
3. [面试追问与参考回答](docs/interview.md)：从 90 秒介绍到深入追问。
4. [检索评测报告](docs/evaluation-report.json)：真实脚本输出，含逐题结果。
5. [设计文档](docs/superpowers/specs/2026-10-01-medical-agent-design.md)、[实施计划](docs/superpowers/plans/2026-10-01-medical-agent.md)。

## 可以演示什么

| 输入 / 操作 | 核心机制 |
|---|---|
| “预约挂号需要什么证件” | 关键词 / 向量混合检索、引用 |
| “我咳嗽” → “三天” | 槽位、多轮状态、有限规则导诊 |
| “帮我预约内科” → 点击确认 | MCP 查号源、interrupt、确认、事务写入 |
| 核对卡片出现后重启服务、刷新页面 | 持久化 checkpoint 和确认恢复 |
| “取消预约” → 确认 | 归属检查、取消、号源释放 |
| “记住我偏好内科” → 新会话 → “帮我预约，按上次偏好” | 跨会话长期偏好 |
| 切换 Bob，使用 Alice 会话 ID 调 API | 用户隔离与 403 |
| “我现在胸痛，帮我挂内科” | 紧急优先分支；有限演示规则 |
| “转人工” → “帮我预约内科” → “返回助手” | 接管状态暂停自动办理；没有真实客服连接 |
| 无关知识问题 | 无证据退出 |

## 测试和评测

```bash
uv run pytest -q
uv run ruff check medical_agent scripts tests
uv run python scripts/evaluate.py
```

测试使用临时数据库与真实 MCP 子进程，无需模型凭据。评测含 17 个有标注文档的问题与 3 个域外问题，比较 lexical / vector / hybrid；这是 12 条合成知识的小样本检索评测，不代表临床效果或在线模型准确率。具体通过数量以最新测试输出为准。

## 在线 Agentic RAG

在线咨询使用有界状态图：检索 → 证据评估 → 生成 → 答案自查。模型通过结构化动作决定直接回答、改写后重新检索、根据反馈重写答案或退出；评估时直接返回新查询，改写节点不额外调用模型。默认最多检索 3 次、改写 2 次、生成 3 次，次数是上限，不是固定轮数。

多轮证据按文档 ID 合并。候选答案通过自查且引用 ID 合法后才发布；无证据、模型输出无效或预算耗尽时返回不足说明。循环状态沿用 Checkpoint 持久化，`rag` 响应提供次数、查询历史、评审动作和退出原因，候选草稿不通过 API 返回。前端执行轨迹展示评估和自查阶段。离线模式仍是固定流程，不进行模型评审。

实现与测试说明见 [Agentic RAG](docs/agentic-rag.md)。状态路由、API 和 SSE 已通过模型服务替身验证；未调用真实模型验证答案效果。模型自查不能保证事实正确，在线模型和 embedding 调用会增加费用与延迟。

## 在线模型

在 `.env` 设置：

```dotenv
MODEL_MODE=online
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
CHAT_MODEL=your-chat-model
EMBEDDING_MODEL=BAAI/bge-m3
```

服务必须支持 chat tool calling / 结构化输出与 embedding。不兼容的提供方需要修改 `model.py` / `rag.py` 的适配器。缺配置或调用失败会明确失败，不会静默转成离线模式。

在线 embedding 默认模型为 `BAAI/bge-m3`，可通过 `EMBEDDING_MODEL` 覆盖为服务端实际模型 ID。需要由配置的服务提供该模型，项目不会自动下载或部署 BGE-M3。当前 chat 与 embedding 共用 `OPENAI_BASE_URL` 和 API key，服务需同时提供两类接口。embedding 请求发送原始文本，长度须符合服务端模型限制；当前不使用客户端的 OpenAI tokenizer 切分。更换模型后重启会重新计算文档向量，检索相似度阈值应按新模型重新评测。

在线路径调用真实模型完成意图提取、必要时查询改写与证据回答；导诊与写操作约束仍由代码执行。没有配置真实凭据时，此路径只通过外部模型替身验证接口边界，不能声称已做真实模型效果评测。

```bash
uv run python scripts/evaluate.py --mode online --output docs/evaluation-online.json
```

该命令会发送合成语料给配置的 embedding 服务，并产生调用费用。首次服务启动也会对全部语料计算 embedding；当前没有增量 embedding 缓存。

## PostgreSQL + pgvector

Docker daemon 启动后：

```bash
docker compose up --build
```

打开 [Docker 聊天界面](http://127.0.0.1:8000/)。业务表、长期记忆、checkpoint 与向量保存在 PostgreSQL；向量通过 pgvector `<=>` 余弦距离计算。默认账号密码和 secret 只用于本地演示，端口只绑定 127.0.0.1。

本次交付的机器 Docker daemon 未启动，因此 Docker / PostgreSQL 运行路径未做实际运行验证；本地 SQLite 和两种 MCP 传输已有实际测试。

## 独立 HTTP MCP 服务

默认 API 自动启动 stdio MCP 子进程。若要展示服务拆分，先在另一个终端启动 HTTP server：

```bash
mkdir -p runtime
export DATABASE_URL="sqlite:///$PWD/runtime/business.db"
export MCP_SERVICE_SECRET="local-demo-shared-secret"
export MCP_TRANSPORT="streamable-http"
export MCP_PORT=8001
uv run python -m medical_agent.mcp_server
```

API 终端使用同一个 DATABASE_URL 和 MCP_SERVICE_SECRET，并设置：

```bash
export MCP_HTTP_URL="http://127.0.0.1:8001/mcp"
uv run uvicorn medical_agent.api:app --host 127.0.0.1 --port 18765 --workers 1
```

独立 server 默认监听 0.0.0.0，仅适合受控演示环境。共享服务 secret 验证工具调用方，患者归属仍由 API 的可信身份和服务端 SQL 条件控制；真实多租户应使用强身份体系，不能把 MCP 等同于授权系统。

## 提交成功但响应异常

`.env` 中设置 `MCP_FAULT_AFTER_COMMIT=true`，重启 stdio 模式 API。确认挂号后 MCP 先提交事务，再返回模拟超时错误。client 调用 `operation_status`，返回已确认的原预约并标注 `reconciled`，数据库只出现一条预约。这是确定性的故障注入，不是网络丢包模拟。

## 导入自己的知识

```bash
uv run python scripts/ingest.py /absolute/path/to/authorized-markdown --output data/imported-knowledge.json
```

设置 `.env` 中 `KNOWLEDGE_PATH=/absolute/path/to/data/imported-knowledge.json` 后重启。支持 Markdown 按标题层级切分，超长章节使用窗口和重叠。默认内置语料已按章节预切分。PDF / OCR / 表格解析未实现。

## 项目结构

```text
medical_agent/
  api.py             FastAPI、鉴权依赖、SSE、记忆入口
  auth.py            本地演示令牌
  service.py         生命周期、请求去重、锁、图执行与恢复
  graph.py           主图 / 咨询子图 / 预约子图
  consultation.py    在线 RAG 评估、重检索、reflection 与预算控制
  model.py           离线规则与在线模型适配、引用校验
  rag.py             词项、哈希向量、混合检索、pgvector
  ingestion.py       Markdown 章节切分
  tools.py           真实 MCP client、超时状态查询
  mcp_server.py      真实 MCP server、模拟医院工具
  store.py           事务、幂等、预约、记忆、归属
  observability.py   ContextVar 记录请求内 chat token 用量
data/                合成知识与固定评测集
scripts/             导入与检索评测
tests/               单元 / API / 工作流 / MCP 集成测试
web/index.html       无前端构建步骤的聊天界面
docs/                技术讲解、面试材料、演示、报告、截图
```

## 能力边界

这是工程展示，不是可直接上线的医疗服务：没有真实 HIS、OIDC、客服、支付、代家属授权或医疗审核。每科只有一个合成号源；紧急识别是有限关键词规则，存在误报与漏报。上下文使用近期消息、结构化槽位和有界服务摘要；未实现自由文本病史语义摘要或健康资料长期存储。SSE 实时推送节点阶段，回答经过校验后一次返回；不是 token 级流式回答。

当前单机构数据集、进程内锁、同步短事务和全语料向量计算适合小型演示。生产化扩展与必要取舍见技术手册，不把这些扩展写成已经实现的能力。
