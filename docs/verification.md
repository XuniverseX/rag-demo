# 交付验证记录

验证环境：macOS、本地 Python 3.12.10、SQLite、offline 模式，2026-10-01。所有测试数据均为临时或合成数据。

## 已实际验证

- `uv run pytest -q`：**30 passed in 13.77s**，无 warning。涵盖真实 MCP stdio / Streamable HTTP、提交后异常对账、业务幂等、过期号源、持久化重启、确认、跨用户隔离、多轮取消、跨会话偏好与删除等。
- `uv run ruff check medical_agent scripts tests`：All checks passed。
- `uv run python scripts/evaluate.py`：20 个问题、12 条合成语料，三种检索方法报告在 evaluation-report.json。
- `docker compose config --quiet`：退出码 0，仅证明配置语法，不证明镜像可构建或服务可启动。
- `node --check` 前端 script：退出码 0。
- 正在运行的 Uvicorn 服务：真实 HTTP / SSE 返回 7 个事件、registration 引用、offline 零模型调用统计。
- 浏览器：知识咨询显示来源，确认前展示具体操作卡片，确认后出现模拟预约，刷新 / 新建会话可正常使用。
- 移动宽度 390px：document.scrollWidth=390、clientWidth=390，无横向溢出；桌面与移动截图已保存。
- 独立只读代码审查：发现并修正多个预约 ID 续接、明确偏好优先级、预约列表变化后不可替换取消目标；均有失败后修正通过的回归测试。

## 未做实际运行验证

- Docker build / PostgreSQL / pgvector / PostgreSQL checkpointer：本机 Docker daemon 未运行；提供配置与代码，但没有真实启动验证。
- 真实在线 chat 与 embedding：没有提供真实模型凭据；结构化模型边界使用 provider 替身验证，不能视为真实模型效果。
- 真实医院系统、客服平台、支付、患者认证：不在本演示范围。
- 高并发压测、自动 TTL、备份数据清理、临床导诊效果：未实施。

## 评测结论

关键词检索和 hybrid 在小合成集上 MRR@3 均为 1.0，哈希向量为 0.9706。三者 Recall@3 和 3 题域外拒绝率均为 1.0。没有依据声称混合检索显著超过关键词基线；也没有依据声称临床、真实用户或在线 LLM 准确率。

用户可按照 README 重跑；用量、耗时和测试时间会随环境变化。30 是当前测试数量，不是运行时质量分数。
