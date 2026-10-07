"""Transactional business store. User identity is supplied by trusted application code."""

import hashlib
import json
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

metadata = MetaData()
slots = Table(
    "slots",
    metadata,
    Column("id", String, primary_key=True),
    Column("department", String),
    Column("doctor", String),
    Column("date", String),
    Column("remaining", Integer),
)
appointments = Table(
    "appointments",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, index=True),
    Column("slot_id", String),
    Column("department", String),
    Column("doctor", String),
    Column("date", String),
    Column("status", String),
)
operations = Table(
    "operations",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String),
    Column("fingerprint", String),
    Column("result", JSON),
)
memories = Table(
    "memories",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String),
    Column("key", String),
    Column("value", String),
    Column("source", String),
    Column("updated_at", String),
)
threads = Table("threads", metadata, Column("id", String, primary_key=True), Column("user_id", String))
requests = Table(
    "requests",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String),
    Column("fingerprint", String),
    Column("result", JSON),
)


class Store:
    def __init__(self, url: str):
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.engine = create_engine(
            url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {}
        )
        metadata.create_all(self.engine)
        day = (datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(days=1)).date().isoformat()
        with self.engine.begin() as connection:
            if not connection.execute(select(slots.c.id).limit(1)).first():
                connection.execute(
                    insert(slots),
                    [
                        dict(id="internal-1", department="内科", doctor="演示医生甲", date=day, remaining=20),
                        dict(id="skin-1", department="皮肤科", doctor="演示医生乙", date=day, remaining=20),
                        dict(id="surgery-1", department="外科", doctor="演示医生丙", date=day, remaining=20),
                        dict(id="child-1", department="儿科", doctor="演示医生丁", date=day, remaining=20),
                    ],
                )

    @staticmethod
    def fingerprint(payload):
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def query_slots(self, department):
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(slots).where(
                        slots.c.department == department, slots.c.remaining > 0, slots.c.date >= today
                    )
                ).mappings()
            ]

    def _write(self, user_id, operation_id, payload, action):
        fingerprint = self.fingerprint(payload)
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(operations).values(
                        id=operation_id, user_id=user_id, fingerprint=fingerprint, result=None
                    )
                )
                result = action(connection)
                connection.execute(
                    update(operations).where(operations.c.id == operation_id).values(result=result)
                )
                return result
        except IntegrityError:
            with self.engine.connect() as connection:
                existing = (
                    connection.execute(select(operations).where(operations.c.id == operation_id))
                    .mappings()
                    .first()
                )
            if not existing:
                raise
            if existing["user_id"] != user_id:
                raise PermissionError("operation belongs to another user")
            if existing["fingerprint"] != fingerprint:
                raise ValueError("operation reused with different parameters")
            return existing["result"]

    def create_appointment(self, user_id, slot_id, operation_id):
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()

        def action(connection):
            slot = (
                connection.execute(
                    update(slots)
                    .where(slots.c.id == slot_id, slots.c.remaining > 0, slots.c.date >= today)
                    .values(remaining=slots.c.remaining - 1)
                    .returning(slots)
                )
                .mappings()
                .first()
            )
            if slot is None:
                raise ValueError("slot unavailable")
            result = dict(
                id=str(uuid4()),
                user_id=user_id,
                slot_id=slot_id,
                department=slot["department"],
                doctor=slot["doctor"],
                date=slot["date"],
                status="booked",
            )
            connection.execute(insert(appointments).values(**result))
            return result

        return self._write(user_id, operation_id, {"action": "book", "slot_id": slot_id}, action)

    def cancel_appointment(self, user_id, appointment_id, operation_id):
        def action(connection):
            result = (
                connection.execute(
                    select(appointments)
                    .where(appointments.c.id == appointment_id, appointments.c.user_id == user_id)
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if result is None:
                raise PermissionError("appointment unavailable to this user")
            result = dict(result)
            if result["status"] == "booked":
                connection.execute(
                    update(appointments).where(appointments.c.id == appointment_id).values(status="cancelled")
                )
                connection.execute(
                    update(slots)
                    .where(slots.c.id == result["slot_id"])
                    .values(remaining=slots.c.remaining + 1)
                )
            result["status"] = "cancelled"
            return result

        return self._write(
            user_id, operation_id, {"action": "cancel", "appointment_id": appointment_id}, action
        )

    def appointments(self, user_id):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(appointments).where(appointments.c.user_id == user_id)
                ).mappings()
            ]

    def operation_status(self, user_id, operation_id):
        with self.engine.connect() as connection:
            row = connection.execute(
                select(operations.c.result).where(
                    operations.c.id == operation_id, operations.c.user_id == user_id
                )
            ).first()
            return row[0] if row else None

    def set_memory(self, user_id, key, value, source):
        if key not in {"preferred_department", "language"}:
            raise ValueError("only confirmed service preferences are supported")
        memory_id = f"{user_id}:{key}"
        values = dict(
            id=memory_id,
            user_id=user_id,
            key=key,
            value=value,
            source=source,
            updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        )
        with self.engine.begin() as connection:
            connection.execute(delete(memories).where(memories.c.id == memory_id))
            connection.execute(insert(memories).values(**values))

    def memories(self, user_id):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(memories).where(memories.c.user_id == user_id)
                ).mappings()
            ]

    def delete_memory(self, user_id, key=None):
        with self.engine.begin() as connection:
            statement = delete(memories).where(memories.c.user_id == user_id)
            connection.execute(statement.where(memories.c.key == key) if key else statement)

    def create_thread(self, user_id):
        thread_id = str(uuid4())
        with self.engine.begin() as connection:
            connection.execute(insert(threads).values(id=thread_id, user_id=user_id))
        return thread_id

    def owns_thread(self, user_id, thread_id):
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(threads.c.id).where(threads.c.id == thread_id, threads.c.user_id == user_id)
                ).first()
            )

    def user_threads(self, user_id):
        with self.engine.connect() as connection:
            return list(
                connection.execute(select(threads.c.id).where(threads.c.user_id == user_id)).scalars()
            )

    def cached_request(self, request_id, payload):
        with self.engine.connect() as connection:
            row = connection.execute(select(requests).where(requests.c.id == request_id)).mappings().first()
        if row and row["fingerprint"] != self.fingerprint(payload):
            raise ValueError("request id reused with different payload")
        return row["result"] if row else None

    def save_request(self, request_id, user_id, payload, result):
        with self.engine.begin() as connection:
            connection.execute(
                insert(requests).values(
                    id=request_id, user_id=user_id, fingerprint=self.fingerprint(payload), result=result
                )
            )

    def delete_user(self, user_id):
        with self.engine.begin() as connection:
            for row in connection.execute(
                select(appointments).where(
                    appointments.c.user_id == user_id, appointments.c.status == "booked"
                )
            ).mappings():
                connection.execute(
                    update(slots).where(slots.c.id == row["slot_id"]).values(remaining=slots.c.remaining + 1)
                )
            for table in (appointments, operations, memories, requests, threads):
                connection.execute(delete(table).where(table.c.user_id == user_id))

    def close(self):
        self.engine.dispose()
