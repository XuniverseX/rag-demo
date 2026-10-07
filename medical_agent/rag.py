"""Inspectable lexical / vector / hybrid retrieval, with synthetic source metadata."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"退号": "取消预约", "发烧": "发热", "红疹": "皮疹", "小孩": "儿童", "开门": "门诊时间"}


def terms(value):
    for old, new in ALIASES.items():
        value = value.replace(old, new)
    for stop in ["怎么", "如何", "什么", "需要", "可以", "请问", "告诉我", "一下", "的", "吗", "？"]:
        value = value.replace(stop, " ")
    pieces = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", value.lower())
    return [piece[i : i + 2] for piece in pieces for i in range(max(1, len(piece) - 1))]


def hash_embedding(value, dimensions=128):
    vector = np.zeros(dimensions)
    for word, count in Counter(terms(value)).items():
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "big") % dimensions] += count * (1 if digest[4] % 2 else -1)
    return vector / (np.linalg.norm(vector) or 1)


class Retriever:
    @classmethod
    async def create(cls, mode="offline", store=None):
        self = cls()
        self.mode, self.store = mode, store
        knowledge_path = Path(os.getenv("KNOWLEDGE_PATH", str(ROOT / "data/knowledge.json")))
        self.documents = json.loads(knowledge_path.read_text())
        if not self.documents:
            raise ValueError("knowledge corpus must not be empty")
        for item in self.documents:
            item.setdefault("source", f"合成示例资料 / {item['title']}")
            item.setdefault("version", "2026-10-01")
            item.setdefault("institution", "demo-hospital")
            item.setdefault("synthetic", True)
        bodies = [item["title"] + " " + item["keywords"] + " " + item["text"] for item in self.documents]
        self.counts = [Counter(terms(body)) for body in bodies]
        self.avg_length = sum(sum(counter.values()) for counter in self.counts) / len(self.counts)
        self.model_id = "offline-hash-128"
        if mode == "online":
            from langchain_openai import OpenAIEmbeddings

            self.model_id = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
            self.embedder = OpenAIEmbeddings(
                model=self.model_id,
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                max_retries=1,
                # Compatible embedding services expect text, not OpenAI tokenizer IDs.
                check_embedding_ctx_length=False,
            )
            self.vectors = np.array(await self.embedder.aembed_documents(bodies))
        else:
            self.vectors = np.array([hash_embedding(body) for body in bodies])
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-12)
        self.pg = bool(store and store.engine.dialect.name == "postgresql")
        if self.pg:
            with store.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS knowledge_vectors "
                        "(id TEXT, model_id TEXT, payload JSONB, embedding vector, PRIMARY KEY(id, model_id))"
                    )
                )
                conn.execute(
                    text("DELETE FROM knowledge_vectors WHERE model_id=:model"), {"model": self.model_id}
                )
                for item, vector in zip(self.documents, self.vectors):
                    conn.execute(
                        text(
                            "INSERT INTO knowledge_vectors VALUES "
                            "(:id, :model, CAST(:payload AS JSONB), CAST(:embedding AS vector))"
                        ),
                        {
                            "id": item["id"],
                            "model": self.model_id,
                            "payload": json.dumps(item),
                            "embedding": json.dumps(vector.tolist()),
                        },
                    )
        return self

    def lexical(self, query):
        scores = []
        for counter in self.counts:
            score = 0.0
            for word in set(terms(query)):
                frequency = counter[word]
                if not frequency:
                    continue
                df = sum(word in candidate for candidate in self.counts)
                idf = math.log(1 + (len(self.counts) - df + 0.5) / (df + 0.5))
                length = sum(counter.values())
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.avg_length))
            scores.append(score)
        return np.array(scores)

    async def search(self, query, mode="hybrid", k=3):
        if mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("unsupported retrieval mode")
        lexical = self.lexical(query)
        vector = (
            np.array(await self.embedder.aembed_query(query))
            if self.mode == "online"
            else hash_embedding(query)
        )
        vector = vector / (np.linalg.norm(vector) or 1)
        if self.pg:
            with self.store.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT id, 1 - (embedding <=> CAST(:v AS vector)) AS score "
                        "FROM knowledge_vectors WHERE model_id=:model ORDER BY score DESC"
                    ),
                    {"v": json.dumps(vector.tolist()), "model": self.model_id},
                ).mappings()
                indexed = {row["id"]: float(row["score"]) for row in rows}
            semantic = np.array([indexed[item["id"]] for item in self.documents])
        else:
            semantic = self.vectors @ vector
        # Offline hashing cannot establish semantic relevance; require lexical evidence too.
        if max(lexical, default=0) < 0.6 and (self.mode == "offline" or max(semantic, default=0) < 0.45):
            return []
        lex_order, vec_order = np.argsort(-lexical), np.argsort(-semantic)
        if mode == "hybrid":
            scores = np.zeros(len(self.documents))
            for rank, index in enumerate(lex_order):
                if lexical[index] > 0:
                    scores[index] += 1 / (10 + rank + 1)
            for rank, index in enumerate(vec_order):
                scores[index] += 1 / (10 + rank + 1)
        else:
            scores = lexical if mode == "lexical" else semantic
        order = np.argsort(-scores)
        return [
            {**self.documents[index], "score": round(float(scores[index]), 5)}
            for index in order
            if lexical[index] > 0 or (self.mode == "online" and semantic[index] >= 0.45)
        ][:k]
