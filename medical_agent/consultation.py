"""Bounded online RAG: model decisions control retrieval and answer revision."""

from langgraph.graph import END, START, StateGraph

LIMITS = {"retrieval": 3, "rewrite": 2, "generation": 3}
PUBLIC_FIELDS = (
    "original_question", "search_query", "query_history", "retrieval_count",
    "rewrite_count", "generation_count", "status", "stop_reason", "reviews",
)


def public_rag(value):
    """Drafts stay in checkpoints and never appear in API results."""
    public = {key: value[key] for key in PUBLIC_FIELDS if key in value}
    # Model prose can quote a rejected draft; expose decisions, not that prose.
    if "reviews" in public:
        public["reviews"] = [
            {key: review[key] for key in ("phase", "action", "next_node")}
            for review in public["reviews"]
        ]
    return public


def build_consultation(state_schema, brain, retriever):
    def update(state, node, rag, **values):
        return {"rag": rag, "trace": state.get("trace", []) + [node], **values}

    async def init(state):
        question = state["messages"][-1].content
        query = (await brain.rewrite(question, state["messages"])).strip() or question
        rag = {
            "original_question": question, "search_query": query, "query_history": [],
            "retrieval_count": 0, "rewrite_count": 0, "generation_count": 0,
            "draft_answer": "", "draft_citation_ids": [], "feedback": "",
            "status": "running", "stop_reason": "", "reviews": [],
        }
        return update(state, "rag_init", rag, evidence=[], citations=[], answer="")

    async def retrieve(state):
        rag = dict(state["rag"])
        found = await retriever.search(rag["search_query"])
        # At most 3 searches of 3 documents: bounded evidence, deduplicated by ID.
        evidence = {item["id"]: item for item in state.get("evidence", [])}
        evidence.update({item["id"]: item for item in found})
        rag["retrieval_count"] += 1
        rag["query_history"] = [*rag["query_history"], rag["search_query"]]
        return update(state, "rag_retrieve", rag, evidence=list(evidence.values()))

    def next_node(phase, decision, rag, evidence):
        allowed = {"generate", "rewrite", "abstain"} if phase == "evaluate" else {
            "accept", "revise", "rewrite", "abstain",
        }
        action = decision.action
        if action not in allowed:
            return "fallback", "invalid_action"
        if action == "abstain":
            return "fallback", "model_abstained"
        if action == "rewrite":
            if rag["retrieval_count"] >= LIMITS["retrieval"] or rag["rewrite_count"] >= LIMITS["rewrite"]:
                return "fallback", "retrieval_budget_exhausted"
            query = (decision.next_query or "").strip()
            if not query or query in rag["query_history"]:
                return "fallback", "invalid_or_repeated_query"
            return "rewrite", ""
        if action in {"generate", "revise"}:
            if not evidence:
                return "fallback", "no_evidence"
            if rag["generation_count"] >= LIMITS["generation"]:
                return "fallback", "generation_budget_exhausted"
            if action == "revise" and not (decision.feedback or "").strip():
                return "fallback", "missing_revision_feedback"
            return "generate", ""
        ids = set(rag["draft_citation_ids"])
        if not rag["draft_answer"].strip() or not ids or not ids <= {item["id"] for item in evidence}:
            return "fallback", "invalid_answer_or_citations"
        return "publish", ""

    async def assess(state, phase):
        rag = dict(state["rag"])
        context = {
            "original_question": rag["original_question"],
            "search_query": rag["search_query"], "query_history": rag["query_history"],
            "evidence": state["evidence"], "previous_feedback": rag["feedback"],
            "remaining_budget": {
                key: limit - rag[f"{key}_count"] for key, limit in LIMITS.items()
            },
        }
        if phase == "reflect":
            context.update(answer=rag["draft_answer"], citations=rag["draft_citation_ids"])
        try:
            decision = await brain.review(phase, context)
        except ValueError:
            rag.update(next_node="fallback", stop_reason="invalid_model_review")
            return update(state, phase, rag)
        destination, reason = next_node(phase, decision, rag, state["evidence"])
        feedback = rag["feedback"]
        if decision.feedback:
            feedback = (
                feedback + "\n" + decision.feedback
                if phase == "evaluate" and feedback else decision.feedback
            )
        elif not feedback:
            feedback = decision.reason
        rag.update(
            next_node=destination, stop_reason=reason,
            next_query=(decision.next_query or "").strip(),
            feedback=feedback,
            reviews=[*rag["reviews"], {"phase": phase, **decision.model_dump(), "next_node": destination}],
        )
        return update(state, phase, rag)

    async def evaluate(state):
        return await assess(state, "evaluate")

    async def reflect(state):
        return await assess(state, "reflect")

    async def rewrite(state):
        rag = dict(state["rag"])
        rag["search_query"] = rag["next_query"]
        rag["rewrite_count"] += 1
        return update(state, "rag_rewrite", rag)

    async def generate(state):
        rag = dict(state["rag"])
        rag["generation_count"] += 1
        try:
            response = await brain.draft(
                rag["original_question"], state["evidence"], rag["draft_answer"], rag["feedback"],
            )
            rag.update(draft_answer=response.answer, draft_citation_ids=response.citations)
            rag["next_node"] = "reflect"
        except ValueError:
            rag.update(next_node="fallback", stop_reason="invalid_model_answer")
        return update(state, "rag_generate", rag)

    async def publish(state):
        rag = dict(state["rag"])
        rag["status"] = "accepted"
        return update(
            state, "rag_publish", rag, answer=rag["draft_answer"],
            citations=[item for item in state["evidence"] if item["id"] in rag["draft_citation_ids"]],
        )

    async def fallback(state):
        rag = dict(state["rag"])
        rag.update(status="abstained", draft_answer="", draft_citation_ids=[])
        return update(
            state, "rag_fallback", rag, citations=[],
            answer="当前证据不足或回答未通过校验，无法可靠回答。请补充具体问题或转人工。",
        )

    graph = StateGraph(state_schema)
    for name, node in [
        ("init", init), ("retrieve", retrieve), ("evaluate", evaluate), ("rewrite", rewrite),
        ("generate", generate), ("reflect", reflect), ("publish", publish), ("fallback", fallback),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "init")
    graph.add_edge("init", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("rewrite", "retrieve")
    for name, destinations in [
        ("evaluate", ["generate", "rewrite", "fallback"]),
        ("generate", ["reflect", "fallback"]),
        ("reflect", ["publish", "generate", "rewrite", "fallback"]),
    ]:
        graph.add_conditional_edges(name, lambda state: state["rag"]["next_node"], {
            destination: destination for destination in destinations
        })
    graph.add_edge("publish", END)
    graph.add_edge("fallback", END)
    return graph.compile()
