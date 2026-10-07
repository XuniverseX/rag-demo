import pytest

from medical_agent.tools import MCPGateway


@pytest.mark.asyncio
async def test_real_mcp_discovery_and_timeout_reconciliation(tmp_path):
    url = f"sqlite:///{tmp_path / 'mcp.db'}"
    async with MCPGateway(url, timeout_after_commit=True) as gateway:
        names = await gateway.names()
        assert {"query_slots", "create_appointment", "cancel_appointment"} <= set(names)
        slots = await gateway.call("query_slots", {"department": "内科"})
        assert slots[0]["id"] == "internal-1"
        result = await gateway.write(
            "create_appointment", {"user_id": "alice", "slot_id": "internal-1", "operation_id": "op-timeout"}
        )
        assert result["status"] == "booked"
        assert result["reconciled"] is True
        appointments = await gateway.call("get_appointment", {"user_id": "alice"})
        assert len(appointments) == 1


@pytest.mark.asyncio
async def test_streamable_http_and_service_auth(tmp_path):
    import asyncio
    import os
    import socket
    import sys

    import httpx

    from medical_agent.tools import ToolFailure

    with socket.socket() as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        port = socket_handle.getsockname()[1]
    url = f"sqlite:///{tmp_path / 'http-mcp.db'}"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "medical_agent.mcp_server",
        env={
            **os.environ,
            "DATABASE_URL": url,
            "MCP_SERVICE_SECRET": "test-internal-secret",
            "MCP_TRANSPORT": "streamable-http",
            "MCP_PORT": str(port),
        },
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        endpoint = f"http://127.0.0.1:{port}/mcp"
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    await client.get(endpoint)
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.05)
            else:
                raise AssertionError("HTTP MCP did not start")
        async with MCPGateway(url, http_url=endpoint, service_secret="test-internal-secret") as gateway:
            assert "query_slots" in await gateway.names()
            assert (await gateway.call("query_slots", {"department": "皮肤科"}))[0]["id"] == "skin-1"
        async with MCPGateway(url, http_url=endpoint, service_secret="wrong-secret") as gateway:
            with pytest.raises(ToolFailure):
                await gateway.call("query_slots", {"department": "内科"})
    finally:
        process.terminate()
        await process.wait()
