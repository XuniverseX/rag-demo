import asyncio
import os
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from medical_agent.consultation import public_rag
from medical_agent.graph import build_graph, emergency_signal
from medical_agent.model import Brain
from medical_agent.observability import collect_usage
from medical_agent.rag import Retriever
from medical_agent.store import Store
from medical_agent.tools import MCPGateway


class Service:
    """One API worker. Per-user lock serializes confirmation, chat and data deletion."""

    def __init__(self, root=None, mode=None, database_url=None):
        self.root = Path(root or os.getenv("RUNTIME_DIR", "runtime")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode = mode or os.getenv("MODEL_MODE", "offline")
        if self.mode not in {"offline", "online"}:
            raise ValueError("MODEL_MODE must be offline or online")
        self.url = database_url or os.getenv("DATABASE_URL") or f"sqlite:///{self.root / 'business.db'}"
        self.stack = AsyncExitStack()
        self.locks = defaultdict(asyncio.Lock)

    async def __aenter__(self):
        try:
            self.store = Store(self.url)
            self.stack.callback(self.store.close)
            if self.url.startswith("postgresql"):
                checkpoint_url = self.url.replace("postgresql+psycopg://", "postgresql://", 1)
                self.saver = await self.stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(checkpoint_url)
                )
            else:
                self.saver = await self.stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(str(self.root / "checkpoints.db"))
                )
            await self.saver.setup()
            self.gateway = await self.stack.enter_async_context(
                MCPGateway(
                    self.url,
                    timeout_after_commit=os.getenv("MCP_FAULT_AFTER_COMMIT") == "true",
                    http_url=os.getenv("MCP_HTTP_URL"),
                    service_secret=os.getenv("MCP_SERVICE_SECRET"),
                )
            )
            self.retriever = await Retriever.create(mode=self.mode, store=self.store)
            self.graph = build_graph(self.saver, self.gateway, self.retriever, self.store, Brain(self.mode))
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        await self.stack.aclose()

    def authorize(self, user_id, thread_id):
        if not self.store.owns_thread(user_id, thread_id):
            raise PermissionError("session unavailable to this user")

    def config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}

    async def state(self, user_id, thread_id):
        self.authorize(user_id, thread_id)
        snapshot = await self.graph.aget_state(self.config(thread_id))
        values = snapshot.values
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return {
            "thread_id": thread_id,
            "answer": "请核对并确认操作详情。" if pending else values.get("answer", ""),
            "pending_operation": pending,
            "appointment": values.get("appointment"),
            "citations": values.get("citations", []),
            "slots": values.get("slots", {}),
            "waiting_for": values.get("waiting_for"),
            "trace": values.get("trace", [])
            + (["prepare_operation", "awaiting_confirmation"] if pending else []),
            "mode": self.mode,
            "handoff": values.get("handoff", False),
            "service_summary": values.get("service_summary", []),
            "rag": public_rag(values.get("rag", {})),
        }

    async def _run(self, input_value, thread_id, emit=None):
        started = time.perf_counter()
        with collect_usage() as usage:
            async for chunk in self.graph.astream(
                input_value, self.config(thread_id), stream_mode="updates", subgraphs=True
            ):
                namespace, updates = chunk
                for node, update in updates.items():
                    if node != "__interrupt__" and emit:
                        await emit(
                            {
                                "type": "stage",
                                "node": node,
                                "scope": list(namespace),
                                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                            }
                        )
        return {
            "graph_ms": round((time.perf_counter() - started) * 1000, 2),
            "chat_usage": dict(usage),
            "scope": "chat token usage only; embedding billing excluded",
        }

    async def chat(self, user_id, thread_id, message, request_id, emit=None):
        async with self.locks[user_id]:
            self.authorize(user_id, thread_id)
            payload = dict(user_id=user_id, thread_id=thread_id, message=message, action="chat")
            cache_id = f"{user_id}:{thread_id}:{request_id}"
            cached = self.store.cached_request(cache_id, payload)
            if cached is not None:
                return cached
            current = await self.state(user_id, thread_id)
            if current["pending_operation"]:
                if emergency_signal(message) or message.strip() == "取消" or "改成" in message:
                    operation_id = current["pending_operation"]["operation_id"]
                    await self._run(
                        Command(resume={"operation_id": operation_id, "approved": False}), thread_id, emit
                    )
                    if message.strip() == "取消":
                        result = await self.state(user_id, thread_id)
                        self.store.save_request(cache_id, user_id, payload, result)
                        return result
                    if "改成" in message and not emergency_signal(message):
                        message = "帮我预约" + message
                else:
                    raise ValueError("请先确认或放弃当前操作；可以输入“取消”或“改成皮肤科”。")
            input_value = dict(
                messages=[HumanMessage(content=message, id=f"{thread_id}:{request_id}")],
                user_id=user_id,
                thread_id=thread_id,
                evidence=[],
                citations=[],
                pending_operation=None,
                appointment=None,
                answer="",
                trace=[],
                rag={},
            )
            metrics = await self._run(input_value, thread_id, emit)
            result = await self.state(user_id, thread_id)
            result["metrics"] = metrics
            self.store.save_request(cache_id, user_id, payload, result)
            return result

    async def confirm(self, user_id, thread_id, operation_id, approved, request_id, emit=None):
        async with self.locks[user_id]:
            self.authorize(user_id, thread_id)
            payload = dict(
                user_id=user_id,
                thread_id=thread_id,
                operation_id=operation_id,
                approved=approved,
                action="confirm",
            )
            cache_id = f"{user_id}:{thread_id}:{request_id}"
            cached = self.store.cached_request(cache_id, payload)
            if cached is not None:
                return cached
            state = await self.state(user_id, thread_id)
            pending = state["pending_operation"]
            if not pending or pending["operation_id"] != operation_id:
                raise ValueError("no matching pending operation")
            metrics = await self._run(
                Command(resume={"operation_id": operation_id, "approved": approved}), thread_id, emit
            )
            result = await self.state(user_id, thread_id)
            result["metrics"] = metrics
            self.store.save_request(cache_id, user_id, payload, result)
            return result

    async def delete_user(self, user_id):
        async with self.locks[user_id]:
            for thread in self.store.user_threads(user_id):
                await self.saver.adelete_thread(thread)
            self.store.delete_user(user_id)
