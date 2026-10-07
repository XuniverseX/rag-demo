from medical_agent.ingestion import chunk_markdown


def test_sections_keep_heading_and_source():
    chunks = chunk_markdown(
        "# 就诊\n\n## 挂号\n携带身份证。\n\n## 退号\n确认取消预约。", source="official.md"
    )
    assert len(chunks) == 2
    assert chunks[0]["title"] == "就诊 / 挂号"
    assert chunks[0]["text"] == "携带身份证。"
    assert chunks[1]["title"] == "就诊 / 退号"
    assert chunks[1]["source"] == "official.md"


def test_long_paragraph_is_bounded_and_overlap_preserved():
    paragraph = "甲" * 100 + "乙" * 100 + "丙" * 100
    chunks = chunk_markdown("# 长文\n" + paragraph, source="x.md", max_chars=120, overlap=20)
    assert all(len(item["text"]) <= 120 for item in chunks)
    assert chunks[0]["text"][-20:] == chunks[1]["text"][:20]
    assert "".join([chunks[0]["text"]] + [item["text"][20:] for item in chunks[1:]]) == paragraph
