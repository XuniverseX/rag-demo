import json

import httpx
import numpy as np
import pytest

from medical_agent.rag import Retriever


@pytest.mark.asyncio
async def test_retrieval_and_source_metadata():
    retriever = await Retriever.create()
    found = await retriever.search("预约挂号需要什么证件")
    assert found[0]["id"] == "registration"
    assert found[0]["synthetic"] is True
    assert found[0]["source"]


@pytest.mark.asyncio
async def test_no_evidence_for_unsupported_question():
    retriever = await Retriever.create()
    assert await retriever.search("量子计算机超导芯片如何制造") == []


@pytest.mark.asyncio
async def test_methods_can_be_evaluated_independently():
    retriever = await Retriever.create()
    for mode in ["lexical", "vector", "hybrid"]:
        results = await retriever.search("怎么取消预约退号", mode=mode)
        assert "cancellation" in [item["id"] for item in results]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_model", [None, "BAAI/bge-m3"])
async def test_online_retrieval_with_text_only_bge_service(monkeypatch, configured_model):
    import langchain_openai

    real_embeddings = langchain_openai.OpenAIEmbeddings
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding.test/v1")
    monkeypatch.delenv("KNOWLEDGE_PATH", raising=False)
    if configured_model is None:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_MODEL", configured_model)

    def embedding_endpoint(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        if payload["model"] != "BAAI/bge-m3" or not all(
            isinstance(item, str) for item in payload["input"]
        ):
            return httpx.Response(400, json={"error": {"message": "Expected BGE model and text input"}})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "BAAI/bge-m3",
                "data": [
                    {"object": "embedding", "index": index, "embedding": [3.0, 4.0]}
                    for index in range(len(payload["input"]))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(embedding_endpoint)) as client:
        monkeypatch.setattr(
            langchain_openai,
            "OpenAIEmbeddings",
            lambda **kwargs: real_embeddings(http_async_client=client, **kwargs),
        )
        retriever = await Retriever.create(mode="online")
        np.testing.assert_allclose(retriever.vectors[0], [0.6, 0.8])
        found = await retriever.search("预约挂号需要什么证件")
        assert "registration" in [item["id"] for item in found]
