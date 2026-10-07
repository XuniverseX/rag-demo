"""Explicit medical service workflow. Deterministic side effects never depend on model discretion."""

import re
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import interrupt

from medical_agent.consultation import build_consultation
from medical_agent.tools import ToolFailure


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    thread_id: str
    intent: str
    slots: dict
    waiting_for: str | None
    evidence: list[dict]
    citations: list[dict]
    answer: str
    pending_operation: dict | None
    appointment: dict | None
    approved: bool
    emergency: bool
    trace: list[str]
    handoff: bool
    service_summary: list[dict]
    rag: dict


def emergency_signal(value):
    # Deliberately limited demonstration rules; not a clinically validated triage classifier.
    return any(word in value for word in ["胸痛", "呼吸困难", "意识不清", "意识异常", "大量出血"])


def step(state, name, **values):
    return {**values, "trace": state.get("trace", []) + [name]}


def build_graph(checkpointer, gateway, retriever, store, brain):
    async def guard(state):
        messages = state["messages"][-12:]
        return step(
            state,
            "risk_and_context",
            emergency=emergency_signal(messages[-1].content),
            messages=[RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages],
        )

    async def route(state):
        value = state["messages"][-1].content
        if state.get("handoff") and value.strip() not in {"返回助手", "恢复助手"}:
            return step(state, "handoff_wait", intent="human")
        if value.strip() in {"返回助手", "恢复助手"}:
            return step(state, "handoff_return", intent="clarify", handoff=False)
        decision = await brain.classify(value)
        slot_values = dict(state.get("slots", {}))
        for key in ["department", "symptom", "duration"]:
            field = getattr(decision, key)
            if field:
                slot_values[key] = field
        intent = decision.intent
        waiting = state.get("waiting_for")
        if intent == "consult" and waiting == "department" and decision.department:
            intent = "book"
        if intent == "consult" and waiting == "duration" and decision.duration:
            intent = "triage"
        if (
            intent == "consult"
            and waiting == "appointment_id"
            and value.strip() in slot_values.get("cancel_candidates", [])
        ):
            intent = "cancel"
        if intent == "memory" and any(word in value for word in ["帮我预约", "帮我挂", "挂个号"]):
            intent = "book"
        if intent == "book" and not decision.department and "偏好" in value:
            memory = next(
                (item for item in store.memories(state["user_id"]) if item["key"] == "preferred_department"),
                None,
            )
            if memory:
                slot_values["department"] = memory["value"]
            else:
                slot_values.pop("department", None)
        summary = state.get("service_summary", []) + [
            {"intent": intent, "department": slot_values.get("department")}
        ]
        return step(
            state,
            "intent_route",
            intent=intent,
            slots=slot_values,
            waiting_for=None,
            service_summary=summary[-8:],
        )

    async def emergency(state):
        return step(
            state,
            "emergency",
            answer="如当前出现胸痛、呼吸困难、意识异常或大量出血，"
            "请立即联系当地急救或前往急诊；中国大陆可拨打 120。不要等待普通预约。"
            "本助手不能诊断。",
            waiting_for=None,
            pending_operation=None,
        )

    async def triage(state):
        values = dict(state.get("slots", {}))
        if not values.get("symptom"):
            return step(
                state,
                "triage_slots",
                answer="请描述主要症状。本演示只能做有限的科室推荐。",
                waiting_for="symptom",
            )
        if not values.get("duration"):
            return step(
                state,
                "triage_slots",
                answer="症状持续多久了？例如三天。若症状严重或迅速恶化，请及时就医。",
                waiting_for="duration",
            )
        values["department"] = (
            "皮肤科"
            if values["symptom"] in {"皮疹", "瘙痒"}
            else "外科"
            if values["symptom"] in {"外伤", "伤口"}
            else "内科"
        )
        evidence = await retriever.search(values["department"])
        return step(
            state,
            "triage_recommendation",
            slots=values,
            evidence=evidence,
            citations=evidence[:1],
            answer=f"有限规则示例建议咨询{values['department']}。"
            "这不是诊断；严重、持续或恶化时应及时就医。如需预约，请明确提出。",
            waiting_for=None,
        )

    async def retrieve(state):
        query = await brain.rewrite(state["messages"][-1].content, state["messages"])
        evidence = await retriever.search(query)
        return step(state, "retrieve", evidence=evidence)

    async def grounded_answer(state):
        response = await brain.answer(state["messages"][-1].content, state["evidence"])
        return step(
            state,
            "grounded_answer",
            answer=response.answer,
            citations=[item for item in state["evidence"] if item["id"] in response.citations],
        )

    consultation = StateGraph(AgentState)
    consultation.add_node("retrieve", retrieve)
    consultation.add_node("answer", grounded_answer)
    consultation.add_edge(START, "retrieve")
    consultation.add_edge("retrieve", "answer")
    consultation.add_edge("answer", END)
    consultation_graph = (
        build_consultation(AgentState, brain, retriever)
        if brain.mode == "online" else consultation.compile()
    )

    async def prepare(state):
        if state["intent"] == "cancel":
            booked = [
                item
                for item in await gateway.call("get_appointment", {"user_id": state["user_id"]})
                if item["status"] == "booked"
            ]
            if not booked:
                return step(state, "prepare_cancel", pending_operation=None, answer="没有可取消的预约。")
            # Explicit selection remains authoritative even when the active list changed meanwhile.
            requested_ids = re.findall(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                state["messages"][-1].content,
            )
            if len(booked) > 1 or requested_ids:
                target = next(
                    (
                        item
                        for item in booked
                        if len(requested_ids) == 1 and item["id"] == requested_ids[0].lower()
                    ),
                    None,
                )
                if not target:
                    return step(
                        state,
                        "prepare_cancel",
                        pending_operation=None,
                        waiting_for="appointment_id",
                        slots={
                            **state.get("slots", {}),
                            "cancel_candidates": [item["id"] for item in booked],
                        },
                        answer=(
                            "指定预约已不可取消，请重新选择预约 ID："
                            if requested_ids
                            else "有多个预约，请指定要取消的预约 ID："
                        )
                        + ", ".join(item["id"] for item in booked),
                    )
            else:
                target = booked[0]
            operation = {
                **target,
                "action": "cancel",
                "appointment_id": target["id"],
                "operation_id": str(uuid4()),
            }
        else:
            department = state.get("slots", {}).get("department")
            if not department:
                return step(
                    state,
                    "collect_department",
                    waiting_for="department",
                    pending_operation=None,
                    answer="需要预约哪个科室？演示支持内科、皮肤科、外科和儿科。",
                )
            available = await gateway.call("query_slots", {"department": department})
            if not available:
                return step(
                    state,
                    "query_slots",
                    pending_operation=None,
                    answer="当前没有可用号源，请换科室或转人工。",
                )
            # One deterministic slot per department in this synthetic dataset; details require confirmation.
            slot = available[0]
            operation = {**slot, "action": "book", "slot_id": slot["id"], "operation_id": str(uuid4())}
        return step(
            state,
            "prepare_operation",
            pending_operation=operation,
            waiting_for=None,
            answer="请核对科室、医生及日期，确认后才会执行。",
        )

    async def confirm(state):
        operation = state["pending_operation"]
        response = interrupt(operation)
        approved = (
            isinstance(response, dict)
            and response.get("operation_id") == operation["operation_id"]
            and response.get("approved") is True
        )
        return step(
            state,
            "confirm_operation",
            approved=approved,
            answer="正在执行已确认的操作。" if approved else "已放弃本次操作，没有提交。",
        )

    async def execute(state):
        operation = state["pending_operation"]
        if not state["approved"]:
            return step(state, "decline_operation", pending_operation=None)
        action = operation["action"]
        arguments = {"user_id": state["user_id"], "operation_id": operation["operation_id"]}
        arguments["slot_id" if action == "book" else "appointment_id"] = operation[
            "slot_id" if action == "book" else "appointment_id"
        ]
        try:
            result = await gateway.write(
                "create_appointment" if action == "book" else "cancel_appointment", arguments
            )
            answer = (
                ("预约成功：" if action == "book" else "预约已取消：")
                + result["department"]
                + "，"
                + result["date"]
            )
            if result.get("reconciled"):
                answer += "。提交后响应异常，已通过操作状态查询确认结果。"
            return step(state, "execute_operation", appointment=result, answer=answer, pending_operation=None)
        except (ToolFailure, TimeoutError):
            return step(
                state,
                "operation_uncertain",
                answer="未能确认操作结果，请查询我的预约或转人工。系统没有自动重复提交。",
                pending_operation=None,
            )

    booking = StateGraph(AgentState)
    for name, node in [("prepare", prepare), ("confirm", confirm), ("execute", execute)]:
        booking.add_node(name, node)
    booking.add_edge(START, "prepare")
    booking.add_conditional_edges(
        "prepare",
        lambda state: "confirm" if state.get("pending_operation") else END,
        {"confirm": "confirm", END: END},
    )
    booking.add_edge("confirm", "execute")
    booking.add_edge("execute", END)

    async def memory(state):
        value = state["messages"][-1].content
        department = next((item for item in ["内科", "皮肤科", "外科", "儿科"] if item in value), None)
        if "记住" not in value or not department:
            return step(state, "memory", answer="仅支持保存明确确认的科室偏好，例如“记住我偏好内科”。")
        store.set_memory(state["user_id"], "preferred_department", department, "explicit user confirmation")
        return step(state, "memory", answer=f"已记住科室偏好：{department}。可在记忆面板查看、更正和删除。")

    async def list_appointments(state):
        result = await gateway.call("get_appointment", {"user_id": state["user_id"]})
        return step(
            state,
            "appointments",
            answer="暂无预约记录。"
            if not result
            else "\n".join(
                f"{item['id']}｜{item['department']}｜{item['date']}｜{item['status']}" for item in result
            ),
        )

    async def human(state):
        return step(
            state,
            "human_handoff",
            handoff=True,
            answer="已进入人工接管演示状态，自动办理已暂停。实际客服系统尚未接入。"
            "输入“返回助手”可恢复自动服务；紧急症状仍会优先给出就医提示。",
        )

    async def clarify(state):
        return step(state, "clarify", answer="你需要院务咨询、症状导诊，还是预约办理？")

    async def finish(state):
        return {"messages": [AIMessage(content=state.get("answer", "请补充问题。"))]}

    graph = StateGraph(AgentState)
    for name, node in [
        ("guard", guard),
        ("route", route),
        ("emergency", emergency),
        ("triage", triage),
        ("consult", consultation_graph),
        ("book", booking.compile()),
        ("memory", memory),
        ("appointments", list_appointments),
        ("human", human),
        ("clarify", clarify),
        ("finish", finish),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges(
        "guard",
        lambda state: "emergency" if state["emergency"] else "route",
        {"emergency": "emergency", "route": "route"},
    )
    graph.add_conditional_edges(
        "route",
        lambda state: "book" if state["intent"] == "cancel" else state["intent"],
        {name: name for name in ["consult", "triage", "book", "memory", "appointments", "human", "clarify"]},
    )
    for name in ["emergency", "triage", "consult", "book", "memory", "appointments", "human", "clarify"]:
        graph.add_edge(name, "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
