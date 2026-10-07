import pytest

from medical_agent.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'business.db'}")


def test_duplicate_operation_has_one_appointment(store):
    first = store.create_appointment("alice", "internal-1", "op-1")
    second = store.create_appointment("alice", "internal-1", "op-1")
    assert first["id"] == second["id"]
    assert len(store.appointments("alice")) == 1


def test_operation_id_cannot_be_reused_for_different_parameters(store):
    store.create_appointment("alice", "internal-1", "op-1")
    with pytest.raises(ValueError, match="different"):
        store.create_appointment("alice", "skin-1", "op-1")


def test_ownership_checked_for_reads_and_cancel(store):
    appointment = store.create_appointment("alice", "internal-1", "op-1")
    assert store.appointments("bob") == []
    assert store.operation_status("bob", "op-1") is None
    with pytest.raises(PermissionError):
        store.cancel_appointment("bob", appointment["id"], "op-2")
    assert store.appointments("alice")[0]["status"] == "booked"


def test_cancel_is_idempotent(store):
    appointment = store.create_appointment("alice", "internal-1", "op-1")
    first = store.cancel_appointment("alice", appointment["id"], "op-2")
    second = store.cancel_appointment("alice", appointment["id"], "op-2")
    assert first == second
    assert store.appointments("alice")[0]["status"] == "cancelled"


def test_memory_isolated_and_deletion_clears_user_data(store):
    store.set_memory("alice", "preferred_department", "内科", "explicit user confirmation")
    assert store.memories("bob") == []
    assert store.memories("alice")[0]["value"] == "内科"
    store.create_appointment("alice", "internal-1", "op-1")
    store.delete_user("alice")
    assert store.memories("alice") == []
    assert store.appointments("alice") == []
    assert store.operation_status("alice", "op-1") is None


def test_expired_slot_is_never_booked(store):
    from sqlalchemy import update

    from medical_agent.store import slots

    with store.engine.begin() as connection:
        connection.execute(update(slots).where(slots.c.id == "internal-1").values(date="2000-01-01"))
    assert store.query_slots("内科") == []
    with pytest.raises(ValueError, match="unavailable"):
        store.create_appointment("alice", "internal-1", "expired")
    assert store.operation_status("alice", "expired") is None
