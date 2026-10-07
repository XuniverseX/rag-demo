# 医疗客服 Agent 面试项目设计

用户已确认全部推荐方案，面向 AI 应用开发 / Agent 工程师。交付代码、测试、可复现评测、可视化聊天界面、详细中文技术手册与面试追问。

## 方案与边界
选择确定性业务工作流与受约束模型节点结合的 LangGraph，而非纯 ReAct（操作确认和恢复难控制），也不使用多个自治 Agent（本项目没有收益足以抵消协调成本）。主图路由，咨询与预约子图负责各自流程。真实 MCP stdio 默认，支持独立 HTTP server。模拟医院业务和合成知识明确标注；不可声称临床可用。

本地默认 SQLite、离线确定性模型及哈希向量；Docker PostgreSQL + pgvector 与 PostgreSQL checkpointer，在线模式使用兼容 OpenAI 的 chat / embedding API。离线模式用于验证工程路径，不能作为真实模型和医学能力的评测证据。不加入未经评测证明的 reranker。

## 流程
优先规则识别紧急线索（仅演示有限集合）；随后意图路由。普通咨询：查询改写、混合检索、证据门控、引用回答。导诊：收集症状 / 病程，有限规则推荐科室，用户自行决定预约。预约：收集科室，通过 MCP 查医生 / 号源，结构化生成操作，interrupt 展示确认。确认绑定 operation_id，执行节点使用 operation_id 作幂等键。取消预约同样确认且验证归属。工具超时后查询操作状态，禁止直接重复写。

## 状态与身份
TypedDict 状态：messages (add_messages)、intent、slots、evidence、answer、pending_operation、user_id、thread_id、trace。消息入图前由应用校验。每用户拥有多个 thread；请求不能自行指定其他身份。演示登录只在 DEMO_AUTH=true 下可用，令牌签名与过期校验；这是本地演示身份，真实 OIDC 不在范围内。每会话并发锁，单 API worker；多 worker 分布式锁明确作为扩展边界。

## 持久化
checkpoint 保存执行；业务表保存会话归属、请求响应去重、预约和幂等操作；记忆表按用户分区保存确认的偏好与来源。上下文限制：近期消息 + 结构化槽位 + 按任务读取的记忆。允许查看、修改及删除记忆；删除用户数据同时删除该用户业务表和 checkpoint。日志仅包含业务阶段 / 文档 ID / 操作 ID，不输出病情原文。

## RAG
合成语料按章节组织，记录 institution、source、version、synthetic。离线字符 ngram 哈希向量与词项相关度融合；在线 embedding 可替换，PostgreSQL 用 pgvector 距离查询。稳定 ID 支持引用检查；低于阈值 / 无匹配时明确拒答。在线生成只用当前证据，验证引用存在但不声称证明语义完全正确。检索内容不具备工具执行权限。

## 验收
真实 MCP tools/list 与 tools/call；挂号 / 取消确认；数据库唯一幂等约束；interrupt 重启恢复；紧急优先；知识无证据退出；跨会话记忆及用户隔离 / 删除；SSE 阶段事件；工具提交后超时故障注入；JSON 评测报告包含模式、Recall@k、MRR、延迟与样本数。在线真实模型无凭据时标注未验证。
