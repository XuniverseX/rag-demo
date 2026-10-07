import asyncio
import json
import os
import secrets
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


class ToolFailure(RuntimeError):
    pass


class MCPGateway:
    """Trusted service-to-service client. User IDs must originate in authenticated API code."""

    def __init__(self, database_url, timeout_after_commit=False, http_url=None, service_secret=None):
        self.database_url = database_url
        self.fault = timeout_after_commit
        self.http_url = http_url
        self.secret = service_secret or secrets.token_urlsafe(32)
        self.stack = AsyncExitStack()

    async def __aenter__(self):
        try:
            if self.http_url:
                reader, writer, _ = await self.stack.enter_async_context(
                    streamable_http_client(self.http_url)
                )
            else:
                env = {
                    **os.environ,
                    "DATABASE_URL": self.database_url,
                    "MCP_SERVICE_SECRET": self.secret,
                    "MCP_FAULT_AFTER_COMMIT": str(self.fault).lower(),
                    "MCP_TRANSPORT": "stdio",
                }
                reader, writer = await self.stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=sys.executable, args=["-m", "medical_agent.mcp_server"], env=env
                        )
                    )
                )
            self.session = await self.stack.enter_async_context(ClientSession(reader, writer))
            await self.session.initialize()
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        await self.stack.aclose()

    async def names(self):
        return [tool.name for tool in (await self.session.list_tools()).tools]

    async def call(self, name, arguments):
        result = await asyncio.wait_for(
            self.session.call_tool(name, {**arguments, "service_token": self.secret}), timeout=10
        )
        if result.isError:
            raise ToolFailure("MCP tool returned error")
        if result.structuredContent is not None:
            data = result.structuredContent
            return data.get("result", data)
        content = "".join(item.text for item in result.content if item.type == "text")
        return json.loads(content)

    async def write(self, name, arguments):
        try:
            return await self.call(name, arguments)
        except (ToolFailure, TimeoutError):
            status = await self.call(
                "operation_status",
                {"user_id": arguments["user_id"], "operation_id": arguments["operation_id"]},
            )
            if status:
                return {**status, "reconciled": True}
            raise ToolFailure("操作结果未能确认，请查询预约或转人工；系统不会自动重复提交。") from None
