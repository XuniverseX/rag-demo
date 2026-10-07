import argparse
import json
from pathlib import Path

from medical_agent.ingestion import chunk_markdown

parser = argparse.ArgumentParser(description="Import authorized Markdown documents as retrievable sections.")
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=Path, default=Path("data/imported-knowledge.json"))
parser.add_argument("--synthetic", action="store_true")
args = parser.parse_args()
chunks = []
for path in sorted(args.directory.rglob("*.md")):
    chunks.extend(
        chunk_markdown(
            path.read_text(), source=str(path.relative_to(args.directory)), synthetic=args.synthetic
        )
    )
if not chunks:
    raise SystemExit("No Markdown content found; output was not modified.")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
print(f"Wrote {len(chunks)} chunks to {args.output}")
