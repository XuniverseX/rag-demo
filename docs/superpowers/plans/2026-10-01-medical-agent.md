# Medical Agent Implementation Plan

> Execute inline with test-driven development and verification-before-completion. User has authorized the design and implementation; no additional approval gate.

**Goal:** 可运行且可解释的 LangGraph + RAG + MCP + 记忆面试项目。
**Architecture:** FastAPI 调用持久化主图 / 子图，MCP 独立业务服务。SQLAlchemy 数据层兼容 SQLite / PostgreSQL。
**Tech Stack:** Python 3.12+, FastAPI, LangGraph 1.x, MCP 1.x, SQLAlchemy 2, pgvector.
**Spec:** ../specs/2026-10-01-medical-agent-design.md

## Global Constraints
- 合成医疗语料 / 模拟业务明确标注。
- 离线默认，无 API key 可启动；在线模式失败不能静默降级。
- 写操作先确认，用户隔离与幂等在代码 / 数据库执行。
- 单 worker 会话锁；在线模型与 Docker 没有实际运行不得宣称已验证。

## Task 1: 业务持久化与 MCP
Files: medical_agent/store.py, mcp_server.py, tools.py; tests/test_store.py, test_mcp.py.
Interfaces: Store(url), create_appointment(user_id, slot_id, operation_id), operation_status(user_id, operation_id); MCPGateway.call(name, arguments).
- [x] 编写重复操作、跨用户访问、取消归属、工具超时状态查询行为测试并运行失败。
- [x] 实现 SQL 事务与唯一操作约束，FastMCP 工具、真实 stdio 客户端。
- [x] 运行 pytest tests/test_store.py tests/test_mcp.py。

## Task 2: 可评测 RAG 与模型边界
Files: rag.py, model.py, data/knowledge.json, scripts/evaluate.py; tests/test_rag.py.
Interfaces: Retriever.search(query, mode, k); Brain.classify(text), Brain.answer(query, evidence).
- [x] 先测试挂号 / 退号检索与无证据退出、非法引用过滤。
- [x] 实现字符分块语料、离线向量、在线 embedding / pgvector、词项融合及来源元数据。
- [x] 运行检索测试，保存检索方法对比报告。

## Task 3: 图、记忆与 API
Files: graph.py, service.py, api.py, auth.py; tests/test_workflow.py, test_api.py.
Interfaces: build_graph(checkpointer, gateway, retriever, store, brain); service.chat / confirm.
- [x] 测试未确认不写、真实 MCP 预约、服务重启恢复、取消、拒绝确认、越权、请求去重和记忆删除。
- [x] 实现主图 + 两个子图、interrupt、持久化 saver、最近消息截断、用户记忆和 SSE。
- [x] 运行全部 pytest；核对 trace 与错误路径。

## Task 4: 展示与文档
Files: web/index.html, Dockerfile, compose.yaml, README.md, docs/technical-guide.md, docs/interview.md.
- [x] 实现聊天 / 操作确认 / 引用 / 记忆 / 轨迹界面。
- [x] 编写从快速运行到原理与失败机制的中文讲解，所有代码定位对应真实文件。
- [x] lint、测试、评测、HTTP/SSE smoke test；记录实际通过与未验证项。

## 执行结果
完成代码、文档、30 个测试和离线评测。Docker / PostgreSQL 与真实模型未运行，参见 ../../verification.md。
