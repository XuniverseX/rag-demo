"""Markdown section-aware ingestion. Oversized paragraphs use bounded overlap windows."""

import hashlib
import re


def chunk_markdown(content, source, max_chars=500, overlap=80, synthetic=False):
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    headings, paragraphs, output = {}, [], []

    def flush():
        if not paragraphs:
            return
        title = " / ".join(headings[level] for level in sorted(headings)) or source
        body = "\n\n".join(paragraphs).strip()
        for offset in range(0, len(body), max_chars - overlap):
            part = body[offset : offset + max_chars]
            identity = hashlib.sha256(f"{source}:{title}:{offset}:{part}".encode()).hexdigest()[:16]
            output.append(
                {
                    "id": identity,
                    "title": title,
                    "keywords": title,
                    "text": part,
                    "source": source,
                    "version": "manual-import",
                    "synthetic": synthetic,
                    "institution": "demo-hospital",
                }
            )
            if offset + max_chars >= len(body):
                break
        paragraphs.clear()

    for line in content.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            flush()
            level = len(match[1])
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = match[2].strip()
        else:
            paragraphs.append(line)
    flush()
    return output
