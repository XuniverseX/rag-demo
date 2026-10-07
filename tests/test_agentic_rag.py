"""Exercise the real graph, checkpoints and model parser with scripted provider I/O."""

import json
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from medical_agent.graph import build_graph
from medical_agent.model import Brain
from medical_agent.service import Service
from medical_agent.store import Store

QUESTION = "挂号需要哪些材料？"
DOC = {"id": "registration", "title": "挂号材料", "text": "携带身份证。", "source": "guide.md"}


def review(action, **kwargs):
    return ("RagReview", {"action": action, "reason": "证据检查", **kwargs})


def draft(text="携带身份证。", citations=None):
    return ("GroundedAnswer", {"answer": text, "citations": ["registration"] if citations is None else citations})


class Provider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.inputs = []

    def with_structured_output(self, schema, **kwargs):
        provider = self

        class Call:
            async def ainvoke(self, messages):
                expected, payload = provider.replies.pop(0)
                assert schema.__name__ == expected
                provider.inputs.append((expected, messages))
                malformed = payload.get("_malformed", False)
                return {
                    "parsed": None if malformed else schema(**payload),
                    "raw": AIMessage(
                        content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    ),
                    "parsing_error": ValueError("invalid provider output") if malformed else None,
                }

        return Call()


class Knowledge:
    def __init__(self, results):
        self.results = iter(results)
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return next(self.results)


@asynccontextmanager
async def session(tmp_path, provider, knowledge):
    service = Service(root=tmp_path, mode="online")
    service.store = Store(service.url)
    brain = Brain()
    brain.mode, brain.chat = "online", provider
    try:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
            await saver.setup()
            service.saver = saver
            service.graph = build_graph(saver, None, knowledge, service.store, brain)
            yield service
    finally:
        service.store.close()


def classify():
    return ("Decision", {"intent": "consult"})


@pytest.mark.asyncio
async def test_accept_once_and_restore_without_exposing_draft(tmp_path):
    provider = Provider([classify(), review("generate"), draft(), review("accept")])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["answer"] == "携带身份证。"
        assert result["rag"]["retrieval_count"] == 1
        assert result["rag"]["generation_count"] == 1
        assert result["rag"]["status"] == "accepted"
        assert "draft_answer" not in result["rag"]
        assert result["metrics"]["chat_usage"]["model_calls"] == 4
        assert await service.chat("alice", thread, QUESTION, "one") == result
    async with session(tmp_path, Provider([]), Knowledge([])) as service:
        restored = await service.state("alice", thread)
        assert restored["rag"] == result["rag"]
        assert restored["answer"] == result["answer"]


@pytest.mark.asyncio
async def test_retrieve_again_and_merge_evidence(tmp_path):
    second = {"id": "card", "title": "就诊卡", "text": "携带就诊卡。"}
    provider = Provider([
        classify(), review("rewrite", next_query="身份证 就诊卡"),
        review("generate"), draft("携带身份证和就诊卡。", ["registration", "card"]), review("accept"),
    ])
    knowledge = Knowledge([[DOC], [DOC, second]])
    async with session(tmp_path, provider, knowledge) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert knowledge.queries == [QUESTION, "身份证 就诊卡"]
        assert result["rag"]["rewrite_count"] == 1
        assert [item["id"] for item in result["citations"]] == ["registration", "card"]


@pytest.mark.asyncio
async def test_reflection_passes_feedback_and_keeps_draft_private(tmp_path):
    provider = Provider([
        classify(), review("generate"), draft("携带身份证，免费挂号。"),
        review("revise", feedback="删除没有依据的免费挂号"), draft(), review("accept"),
    ])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        events = []

        async def emit(event):
            events.append(event)
            current = await service.state("alice", thread)
            assert "免费挂号" not in current["answer"]
            assert "draft_answer" not in current["rag"]

        result = await service.chat("alice", thread, QUESTION, "one", emit)
        assert result["answer"] == "携带身份证。"
        assert result["rag"]["generation_count"] == 2
        generations = [messages for name, messages in provider.inputs if name == "GroundedAnswer"]
        assert "删除没有依据的免费挂号" in generations[1][-1].content
        assert any(event["node"] == "reflect" for event in events)


@pytest.mark.asyncio
async def test_reflection_can_request_new_evidence(tmp_path):
    provider = Provider([
        classify(), review("generate"), draft("携带身份证，免费挂号。"),
        review("rewrite", next_query="挂号证件要求", feedback="删除免费挂号声明"),
        review("generate"), draft(), review("accept"),
    ])
    async with session(tmp_path, provider, Knowledge([[DOC], [DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["retrieval_count"] == 2
        assert result["rag"]["generation_count"] == 2
        generations = [messages for name, messages in provider.inputs if name == "GroundedAnswer"]
        assert "删除免费挂号声明" in json.loads(generations[1][-1].content)["revision_feedback"]


@pytest.mark.asyncio
async def test_retrieval_budget_exits_without_answering(tmp_path):
    provider = Provider([
        classify(), review("rewrite", next_query="材料一"), review("rewrite", next_query="材料二"),
        review("rewrite", next_query="材料三"),
    ])
    async with session(tmp_path, provider, Knowledge([[], [], []])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["retrieval_count"] == 3
        assert result["rag"]["rewrite_count"] == 2
        assert result["rag"]["status"] == "abstained"
        assert result["citations"] == []


@pytest.mark.asyncio
async def test_generation_budget_never_publishes_rejected_draft(tmp_path):
    provider = Provider([classify(), review("generate")] + [
        reply for _ in range(3) for reply in (draft("不受支持的草稿"), review("revise", feedback="删除无依据内容"))
    ])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["generation_count"] == 3
        assert result["rag"]["status"] == "abstained"
        assert "不受支持的草稿" not in result["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("citations", [["invented"], []])
async def test_accept_cannot_override_invalid_citations(tmp_path, citations):
    provider = Provider([classify(), review("generate"), draft("无效引用的草稿", citations), review("accept")])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["status"] == "abstained"
        assert "无效引用的草稿" not in result["answer"]
        assert result["citations"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [
    review("accept"), review("rewrite", next_query=" "), review("rewrite", next_query=QUESTION),
    review("generate"), review("abstain"),
])
async def test_invalid_or_unanswerable_evaluation_stops(tmp_path, decision):
    provider = Provider([classify(), decision])
    async with session(tmp_path, provider, Knowledge([[]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["status"] == "abstained"
        assert result["rag"]["generation_count"] == 0


@pytest.mark.asyncio
async def test_new_question_resets_rag_state(tmp_path):
    provider = Provider([
        classify(), review("rewrite", next_query="所需材料"), review("generate"), draft(), review("accept"),
        classify(), review("generate"), draft(), review("accept"),
    ])
    async with session(tmp_path, provider, Knowledge([[DOC], [DOC], [DOC]])) as service:
        thread = service.store.create_thread("alice")
        await service.chat("alice", thread, QUESTION, "one")
        result = await service.chat("alice", thread, "挂号材料有哪些？", "two")
        assert result["rag"]["query_history"] == ["挂号材料有哪些？"]
        assert result["rag"]["rewrite_count"] == 0
        assert result["rag"]["retrieval_count"] == 1


@pytest.mark.asyncio
async def test_longest_path_fits_graph_limit(tmp_path):
    provider = Provider([
        classify(), review("generate"), draft(), review("rewrite", next_query="材料一"),
        review("generate"), draft(), review("rewrite", next_query="材料二"),
        review("generate"), draft(), review("accept"),
    ])
    async with session(tmp_path, provider, Knowledge([[DOC], [DOC], [DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["status"] == "accepted"
        assert result["rag"]["retrieval_count"] == 3
        assert result["rag"]["generation_count"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("replies", [
    [("RagReview", {"_malformed": True})],
    [review("generate"), ("GroundedAnswer", {"_malformed": True})],
    [review("generate"), draft(), ("RagReview", {"_malformed": True})],
])
async def test_malformed_model_output_exits_safely(tmp_path, replies):
    provider = Provider([classify(), *replies])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert result["rag"]["status"] == "abstained"
        assert result["citations"] == []


def test_online_api_stream_contains_reviews_and_final_result(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import medical_agent.service as service_module
    from medical_agent.api import create_app

    provider = Provider([classify(), review("generate"), draft(), review("accept")])
    brain = Brain()
    brain.mode, brain.chat = "online", provider
    knowledge = Knowledge([[DOC]])

    async def create_retriever(**kwargs):
        return knowledge

    monkeypatch.setenv("MODEL_MODE", "online")
    monkeypatch.setattr(service_module, "Brain", lambda mode: brain)
    monkeypatch.setattr(service_module.Retriever, "create", create_retriever)
    with TestClient(create_app(root=tmp_path)) as client:
        token = client.post("/api/demo-login", json={"user": "alice"}).json()["token"]
        headers = {"Authorization": "Bearer " + token}
        thread = client.post("/api/threads", headers=headers).json()["thread_id"]
        response = client.post("/api/chat/stream", headers=headers, json={
            "thread_id": thread, "message": QUESTION, "request_id": "stream",
        })
        assert response.status_code == 200
        events = [json.loads(frame[6:]) for frame in response.text.strip().split("\n\n")]
        assert {"evaluate", "reflect", "publish"} <= {
            event["node"] for event in events if event["type"] == "stage"
        }
        result = events[-1]
        assert result["type"] == "result"
        assert result["data"]["rag"]["status"] == "accepted"
        assert result["data"]["citations"][0]["id"] == "registration"
        assert "draft_answer" not in result["data"]["rag"]


@pytest.mark.asyncio
async def test_rejected_draft_cannot_leak_through_review_prose(tmp_path):
    rejected = "没有证据支持的敏感草稿"
    provider = Provider([
        classify(), review("generate"), draft(rejected),
        review("abstain", reason=f"拒绝草稿：{rejected}", feedback=f"删除：{rejected}"),
    ])
    async with session(tmp_path, provider, Knowledge([[DOC]])) as service:
        thread = service.store.create_thread("alice")
        result = await service.chat("alice", thread, QUESTION, "one")
        assert rejected not in json.dumps(result, ensure_ascii=False)
        restored = await service.state("alice", thread)
        assert rejected not in json.dumps(restored, ensure_ascii=False)
        assert result["rag"]["reviews"][-1]["action"] == "abstain"
