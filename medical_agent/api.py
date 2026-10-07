import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager, suppress
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from medical_agent.auth import issue_token, verify_token
from medical_agent.rag import ROOT
from medical_agent.service import Service
from medical_agent.tools import ToolFailure

load_dotenv()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    user: Literal["alice", "bob"] = "alice"


class Chat(Input):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    request_id: str = Field(min_length=1, max_length=100)


class Confirm(Input):
    thread_id: str = Field(min_length=1, max_length=100)
    operation_id: str = Field(min_length=1, max_length=100)
    approved: StrictBool
    request_id: str = Field(min_length=1, max_length=100)


class Memory(Input):
    key: Literal["preferred_department", "language"]
    value: str = Field(min_length=1, max_length=30)


def create_app(root=None):
    secret = os.getenv("AUTH_SECRET") or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        async with Service(root=root) as service:
            app.state.service = service
            yield

    app = FastAPI(title="Medical Agent · Interview Lab", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not credentials:
            raise HTTPException(401, "需要登录演示账号。")
        try:
            return verify_token(credentials.credentials, secret)
        except PermissionError:
            raise HTTPException(401, "令牌无效或过期，请重新登录。") from None

    @app.exception_handler(PermissionError)
    async def permission_error(request, exception):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=403, content={"detail": "无权访问该会话或数据。"})

    @app.exception_handler(ValueError)
    async def value_error(request, exception):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(exception)})

    @app.exception_handler(ToolFailure)
    async def tool_error(request, exception):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=502, content={"detail": "工具调用失败，请稍后查询或转人工。"})

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "mode": app.state.service.mode,
            "synthetic": True,
            "storage": app.state.service.store.engine.dialect.name,
            "tools": await app.state.service.gateway.names(),
        }

    @app.post("/api/demo-login")
    async def login(value: Login):
        if os.getenv("DEMO_AUTH", "true") != "true":
            raise HTTPException(403, "演示登录已关闭；此项目尚未实现外部身份提供方。")
        return {"token": issue_token(value.user, secret), "user": value.user, "expires_in": 3600}

    @app.post("/api/threads")
    async def create_thread(user_id=Depends(user)):
        return {"thread_id": app.state.service.store.create_thread(user_id)}

    @app.get("/api/threads/{thread_id}")
    async def state(thread_id: str, user_id=Depends(user)):
        return await app.state.service.state(user_id, thread_id)

    @app.post("/api/chat")
    async def chat(value: Chat, user_id=Depends(user)):
        return await app.state.service.chat(user_id, value.thread_id, value.message, value.request_id)

    @app.post("/api/confirm")
    async def confirm(value: Confirm, user_id=Depends(user)):
        return await app.state.service.confirm(
            user_id, value.thread_id, value.operation_id, value.approved, value.request_id
        )

    @app.post("/api/chat/stream")
    async def stream(value: Chat, user_id=Depends(user)):
        service = app.state.service
        service.authorize(user_id, value.thread_id)

        async def events():
            queue = asyncio.Queue()

            async def run():
                try:
                    result = await service.chat(
                        user_id, value.thread_id, value.message, value.request_id, queue.put
                    )
                    await queue.put({"type": "result", "data": result})
                except (ValueError, PermissionError) as exc:
                    await queue.put({"type": "error", "message": str(exc)})
                except Exception:
                    await queue.put({"type": "error", "message": "处理失败，请检查服务终端或稍后重试。"})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(run())
            try:
                while (event := await queue.get()) is not None:
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            finally:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/memory")
    async def get_memory(user_id=Depends(user)):
        return app.state.service.store.memories(user_id)

    @app.put("/api/memory")
    async def put_memory(value: Memory, user_id=Depends(user)):
        if value.key == "preferred_department" and value.value not in {"内科", "皮肤科", "外科", "儿科"}:
            raise HTTPException(422, "科室不在演示范围内。")
        if value.key == "language" and value.value not in {"中文", "English"}:
            raise HTTPException(422, "语言偏好只能是中文或 English；当前回复仍以中文为主。")
        async with app.state.service.locks[user_id]:
            app.state.service.store.set_memory(user_id, value.key, value.value, "explicit user edit")
        return {"status": "saved"}

    @app.delete("/api/memory")
    async def delete_memory(user_id=Depends(user)):
        async with app.state.service.locks[user_id]:
            app.state.service.store.delete_memory(user_id)
        return {"status": "deleted"}

    @app.get("/api/appointments")
    async def appointments(user_id=Depends(user)):
        return app.state.service.store.appointments(user_id)

    @app.delete("/api/user-data")
    async def delete_user(user_id=Depends(user)):
        await app.state.service.delete_user(user_id)
        return {"status": "deleted"}

    return app


app = create_app()
