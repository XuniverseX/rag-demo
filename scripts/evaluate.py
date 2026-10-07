"""Reproducible retrieval ablation. Does NOT measure clinical or LLM answer quality."""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv

from medical_agent.rag import ROOT, Retriever


async def evaluate(mode, output):
    retriever = await Retriever.create(mode=mode)
    dataset = json.loads((ROOT / "data/evaluation.json").read_text())
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_mode": mode,
        "embedding": retriever.model_id,
        "corpus_size": len(retriever.documents),
        "samples": len(dataset),
        "scope": "synthetic retrieval only; no clinical or LLM-quality claims",
        "methods": {},
    }
    for method in ["lexical", "vector", "hybrid"]:
        rows, latencies, recalls, reciprocal_ranks, rejections = [], [], [], [], []
        for case in dataset:
            started = time.perf_counter()
            found = await retriever.search(case["query"], mode=method, k=3)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            ids = [item["id"] for item in found]
            relevant = set(case["relevant"])
            if relevant:
                recalls.append(len(relevant.intersection(ids)) / len(relevant))
                reciprocal_ranks.append(
                    next((1 / (i + 1) for i, doc_id in enumerate(ids) if doc_id in relevant), 0)
                )
            else:
                rejections.append(not ids)
            rows.append(
                {
                    "query": case["query"],
                    "relevant": case["relevant"],
                    "retrieved": ids,
                    "latency_ms": round(elapsed, 3),
                }
            )
        report["methods"][method] = {
            "recall_at_3": round(float(np.mean(recalls)), 4),
            "mrr_at_3": round(float(np.mean(reciprocal_ranks)), 4),
            "out_of_domain_rejection_rate": float(np.mean(rejections)),
            "p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "cases": rows,
        }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                key: {name: value for name, value in result.items() if name != "cases"}
                for key, result in report["methods"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report: {path.resolve()}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--output", default="docs/evaluation-report.json")
    args = parser.parse_args()
    asyncio.run(evaluate(args.mode, args.output))
