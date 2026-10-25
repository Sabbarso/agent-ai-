from rpe.ui.conversations import ConversationStore, conversation_markdown, make_title


def test_new_empty_conversation_is_not_saved(tmp_path):
    store = ConversationStore(tmp_path / "conv")
    store.save(store.new())
    assert store.list() == []


def test_save_list_load_delete(tmp_path):
    store = ConversationStore(tmp_path / "conv")
    a = store.new()
    a["messages"] += [
        {"role": "user", "content": "How can retrieval reduce hallucinations in LLMs?"},
        {
            "role": "assistant",
            "content": "[link]",
            "answer": "By grounding [arXiv:2401.00001].",
            "comparison": {"question": "q", "comparisons": []},
        },
    ]
    store.save(a)
    b = store.new()
    b["messages"].append({"role": "user", "content": "Second question"})
    store.save(b)

    listed = store.list()
    assert [c["id"] for c in listed] == [b["id"], a["id"]] or {c["id"] for c in listed} == {
        a["id"],
        b["id"],
    }
    assert {c["title"] for c in listed} == {
        "How can retrieval reduce hallucinations in LLMs?",
        "Second question",
    }
    loaded = store.load(a["id"])
    assert loaded["messages"][1]["comparison"] == {"question": "q", "comparisons": []}

    store.delete(a["id"])
    assert [c["id"] for c in store.list()] == [b["id"]] and store.load(a["id"]) is None


def test_title_truncated_and_markdown_export():
    assert make_title("word " * 40).endswith("…") and len(make_title("word " * 40)) <= 60
    conv = {
        "title": "T",
        "updated_at": "2026-10-25T10:00:00",
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "[arXiv:1](link)", "answer": "A [arXiv:2401.00001]."},
        ],
    }
    md = conversation_markdown(conv)
    assert "**Question**" in md and "A [arXiv:2401.00001]." in md


def test_corrupt_file_is_skipped(tmp_path):
    store = ConversationStore(tmp_path)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert store.list() == []
