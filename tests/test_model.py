import pytest
from langchain_core.messages import AIMessage

from medical_agent.model import Brain, GroundedAnswer
from medical_agent.observability import collect_usage


class ModelReply:
    """External LLM double; only replaces provider I/O, keeping answer validation real."""

    def __init__(self, citation):
        self.citation = citation

    def with_structured_output(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages):
        parsed = GroundedAnswer(answer="unsupported provider answer", citations=[self.citation])
        return {
            "parsed": parsed,
            "raw": AIMessage(
                content="", usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}
            ),
            "parsing_error": None,
        }


@pytest.mark.asyncio
async def test_invalid_model_citation_is_not_shown():
    brain = Brain()
    brain.mode, brain.chat = "online", ModelReply("invented-id")
    result = await brain.answer("挂号", [{"id": "registration", "text": "携带证件。"}])
    assert result.citations == []
    assert "unsupported provider answer" not in result.answer


@pytest.mark.asyncio
async def test_token_accounting_isolated_between_concurrent_requests():
    import asyncio

    async def request(calls):
        brain = Brain()
        brain.mode, brain.chat = "online", ModelReply("registration")
        with collect_usage() as usage:
            for _ in range(calls):
                await asyncio.sleep(0)
                await brain.answer("挂号", [{"id": "registration", "text": "携带证件。"}])
        return usage

    first, second = await asyncio.gather(request(1), request(2))
    assert first["input_tokens"] == 12
    assert first["total_tokens"] == 17
    assert second["total_tokens"] == 34
    assert second["model_calls"] == 2
