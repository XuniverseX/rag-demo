import pytest

from medical_agent.service import Service


@pytest.mark.asyncio
async def test_booking_confirmation_survives_restart_and_is_idempotent(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        response = await service.chat("alice", thread, "帮我预约内科", "request-1")
        operation = response["pending_operation"]
        assert operation["department"] == "内科"
        assert service.store.appointments("alice") == []
    async with Service(root=tmp_path) as service:
        restored = await service.state("alice", thread)
        assert restored["pending_operation"]["operation_id"] == operation["operation_id"]
        with pytest.raises(ValueError):
            await service.confirm("alice", thread, "wrong-operation", True, "wrong-request")
        response = await service.confirm("alice", thread, operation["operation_id"], True, "confirm-1")
        assert response["appointment"]["status"] == "booked"
        again = await service.confirm("alice", thread, operation["operation_id"], True, "confirm-1")
        assert again == response
        assert len(service.store.appointments("alice")) == 1


@pytest.mark.asyncio
async def test_declined_operation_never_writes(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, "帮我预约内科", "book")
        await service.confirm("alice", thread, result["pending_operation"]["operation_id"], False, "decline")
        assert service.store.appointments("alice") == []


@pytest.mark.asyncio
async def test_emergency_preempts_booking_and_interrupt(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, "我现在胸痛，帮我挂内科", "emergency")
        assert "120" in result["answer"]
        assert result["pending_operation"] is None
        await service.chat("alice", thread, "帮我预约内科", "prepare")
        result = await service.chat("alice", thread, "现在呼吸困难", "emergency-2")
        assert "120" in result["answer"]
        assert result["pending_operation"] is None
        assert service.store.appointments("alice") == []


@pytest.mark.asyncio
async def test_triage_slots_and_citation(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        first = await service.chat("alice", thread, "我咳嗽", "first")
        assert first["waiting_for"] == "duration"
        second = await service.chat("alice", thread, "三天", "second")
        assert second["slots"]["department"] == "内科"
        assert second["waiting_for"] is None
        assert second["citations"][0]["id"] == "internal"


@pytest.mark.asyncio
async def test_memory_cross_session_and_user_isolation(tmp_path):
    async with Service(root=tmp_path) as service:
        first = service.store.create_thread("alice")
        await service.chat("alice", first, "记住我偏好内科", "remember")
        second = service.store.create_thread("alice")
        result = await service.chat("alice", second, "帮我预约，按上次偏好", "use-memory")
        assert result["pending_operation"]["department"] == "内科"
        with pytest.raises(PermissionError):
            await service.state("bob", first)
        bob = service.store.create_thread("bob")
        result = await service.chat("bob", bob, "帮我预约，按上次偏好", "bob-memory")
        assert result["pending_operation"] is None
        assert result["waiting_for"] == "department"
        await service.delete_user("alice")
        assert service.store.memories("alice") == []
        assert not (await service.graph.aget_state({"configurable": {"thread_id": first}})).values


@pytest.mark.asyncio
async def test_cancel_requires_confirmation(tmp_path):
    async with Service(root=tmp_path) as service:
        appointment = service.store.create_appointment("alice", "internal-1", "existing")
        thread = service.store.create_thread("alice")
        response = await service.chat("alice", thread, "取消预约", "cancel")
        assert service.store.appointments("alice")[0]["status"] == "booked"
        await service.confirm(
            "alice", thread, response["pending_operation"]["operation_id"], True, "cancel-confirm"
        )
        assert service.store.appointments("alice")[0]["id"] == appointment["id"]
        assert service.store.appointments("alice")[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_human_handoff_stops_ai_work_until_explicit_return(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        await service.chat("alice", thread, "转人工", "human")
        result = await service.chat("alice", thread, "帮我预约内科", "during-handoff")
        assert result["pending_operation"] is None
        assert result["handoff"] is True
        await service.chat("alice", thread, "返回助手", "return")
        result = await service.chat("alice", thread, "帮我预约内科", "after-return")
        assert result["pending_operation"]


@pytest.mark.asyncio
async def test_memory_does_not_infer_preference_from_old_slots(tmp_path):
    async with Service(root=tmp_path) as service:
        thread = service.store.create_thread("alice")
        await service.chat("alice", thread, "内科看什么", "department")
        await service.chat("alice", thread, "记住我担心有糖尿病", "unconfirmed-health")
        assert service.store.memories("alice") == []


@pytest.mark.asyncio
async def test_multiple_appointments_can_be_cancelled_by_replying_with_id(tmp_path):
    async with Service(root=tmp_path) as service:
        first = service.store.create_appointment("alice", "internal-1", "existing-one")
        second = service.store.create_appointment("alice", "skin-1", "existing-two")
        thread = service.store.create_thread("alice")
        response = await service.chat("alice", thread, "取消预约", "ask-cancel")
        assert response["waiting_for"] == "appointment_id"
        response = await service.chat("alice", thread, second["id"], "choose-cancel")
        assert response["pending_operation"]["appointment_id"] == second["id"]
        await service.confirm(
            "alice", thread, response["pending_operation"]["operation_id"], True, "cancel-selected"
        )
        items = {item["id"]: item["status"] for item in service.store.appointments("alice")}
        assert items[first["id"]] == "booked"
        assert items[second["id"]] == "cancelled"


@pytest.mark.asyncio
async def test_explicit_memory_request_overrides_unrelated_session_department(tmp_path):
    async with Service(root=tmp_path) as service:
        service.store.set_memory("alice", "preferred_department", "皮肤科", "explicit user edit")
        thread = service.store.create_thread("alice")
        await service.chat("alice", thread, "内科看什么", "unrelated-question")
        response = await service.chat("alice", thread, "帮我预约，按上次偏好", "use-explicit-memory")
        assert response["pending_operation"]["department"] == "皮肤科"


@pytest.mark.asyncio
async def test_stale_cancellation_selection_never_substitutes_other_appointment(tmp_path):
    async with Service(root=tmp_path) as service:
        first = service.store.create_appointment("alice", "internal-1", "first-booking")
        second = service.store.create_appointment("alice", "skin-1", "second-booking")
        thread = service.store.create_thread("alice")
        await service.chat("alice", thread, "取消预约", "choose")
        service.store.cancel_appointment("alice", second["id"], "cancelled-elsewhere")
        response = await service.chat("alice", thread, second["id"], "stale-choice")
        assert response["pending_operation"] is None
        items = {item["id"]: item["status"] for item in service.store.appointments("alice")}
        assert items[first["id"]] == "booked"
