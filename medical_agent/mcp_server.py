"""Real MCP service with simulated hospital data. stdout belongs to JSON-RPC."""

import hmac
import os

from mcp.server.fastmcp import FastMCP

from medical_agent.store import Store

mcp = FastMCP("Demo hospital", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8001")))
store = Store(os.environ["DATABASE_URL"])
secret = os.environ["MCP_SERVICE_SECRET"]


def authorize(service_token):
    if not hmac.compare_digest(service_token, secret):
        raise PermissionError("invalid service identity")


@mcp.tool()
def list_departments(service_token: str) -> list[str]:
    authorize(service_token)
    return ["内科", "皮肤科", "外科", "儿科"]


@mcp.tool()
def search_doctors(department: str, service_token: str) -> list[dict]:
    authorize(service_token)
    return [{"name": row["doctor"], "department": department} for row in store.query_slots(department)]


@mcp.tool()
def query_slots(department: str, service_token: str) -> list[dict]:
    authorize(service_token)
    return store.query_slots(department)


@mcp.tool()
def create_appointment(user_id: str, slot_id: str, operation_id: str, service_token: str) -> dict:
    authorize(service_token)
    result = store.create_appointment(user_id, slot_id, operation_id)
    if os.getenv("MCP_FAULT_AFTER_COMMIT") == "true":
        raise TimeoutError("simulated response timeout after committed transaction")
    return result


@mcp.tool()
def get_appointment(user_id: str, service_token: str) -> list[dict]:
    authorize(service_token)
    return store.appointments(user_id)


@mcp.tool()
def cancel_appointment(user_id: str, appointment_id: str, operation_id: str, service_token: str) -> dict:
    authorize(service_token)
    return store.cancel_appointment(user_id, appointment_id, operation_id)


@mcp.tool()
def operation_status(user_id: str, operation_id: str, service_token: str) -> dict:
    authorize(service_token)
    return store.operation_status(user_id, operation_id) or {}


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
