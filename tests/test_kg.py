import pytest

from rpe.kg.build_graph import build_graph
from rpe.kg.extract import extract_batch, load_cache, select_papers
from rpe.kg.graph_store import NetworkXGraphStore
from rpe.kg.graphrag import (
    answer_graph_question,
    author_communities,
    central_authors,
    find_entity,
    graph_query,
    parse_question_rules,
)
from rpe.kg.normalize import DEFAULT_ALIASES, EntityNormalizer, clean_surface, comparison_key
from rpe.llm.client import LLMClient
from rpe.llm.schemas import PaperExtraction
from tests.fakes import ScriptedLLM

EXTRACTIONS = {
    "2301.00001": PaperExtraction(
        tasks=["question answering"],
        methods=["RAG", "dense retrieval"],
        datasets=["SQuAD v1.1", "Natural Questions"],
        metrics=["EM"],
        contribution_type="new_method",
    ),
    "2302.00002": PaperExtraction(
        tasks=["machine translation", "low-resource MT"],
        methods=["back-translation"],
        datasets=["WMT"],
        metrics=["BLEU score"],
    ),
    "2405.00005": PaperExtraction(
        tasks=["Machine Translation"],
        methods=["COMET", "human evaluation"],
        metrics=["BLEU", "COMET"],
        contribution_type="survey",
    ),
    "2506.00006": PaperExtraction(
        tasks=["Question-Answering"],
        methods=["Dense Retrieval", "contrastive learning"],
        datasets=["SQuAD"],
        metrics=["accuracy"],
    ),
}


@pytest.fixture
def normalizer():
    return EntityNormalizer(DEFAULT_ALIASES)


@pytest.fixture
def graph(papers_df, normalizer):
    return build_graph(papers_df, EXTRACTIONS, normalizer, max_coauthors=30)


# ----------------------------------------------------------- normalize


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        ("SQuAD v1.1", "dataset", "SQuAD"),
        ("SQuAD (v2)", "dataset", "SQuAD"),
        ('"GLUE benchmark"', "dataset", "GLUE"),
        ("BERT model", "method", "BERT"),
        ("language model", "method", "language model"),
        ("GPT-3.5", "method", "GPT-3.5"),
        ("YOLOv5", "method", "YOLOv5"),
        ("F1 score", "metric", "F1"),
    ],
)
def test_clean_surface(raw, kind, expected):
    assert clean_surface(raw, kind) == expected


def test_comparison_key_insensitive():
    assert comparison_key("Question-Answering") == comparison_key("question answering")
    assert comparison_key("Large Language Models") == comparison_key("large language model")


def test_aliases_and_fuzzy_merge(normalizer):
    normalizer.fit(
        {
            "dataset": [
                "SQuAD",
                "SQuAD v1.1",
                "Stanford Question Answering Dataset",
                "MultiWOZ 2.1",
                "MultiWOZ2.1",
                "MultiWoz",
            ],
            "task": ["MT", "machine translation", "Machine Translation"],
        }
    )
    assert normalizer.normalize("SQuAD v1.1", "dataset") == "SQuAD"
    assert normalizer.normalize("Stanford Question Answering Dataset", "dataset") == "SQuAD"
    assert normalizer.normalize("MT", "task") == "Machine Translation"
    assert normalizer.normalize("MultiWOZ2.1", "dataset") == normalizer.normalize(
        "MultiWOZ 2.1", "dataset"
    )


def test_alias_file_created(tmp_path):
    path = tmp_path / "aliases.yaml"
    n = EntityNormalizer.load(path)
    assert path.exists() and n.normalize("QA", "task") == "Question Answering"


# --------------------------------------------------------------- graph


def test_graph_nodes_and_edges(graph):
    types = graph.count_by_type()
    assert types["Paper"] == 6 and types["Dataset"] >= 3 and types["Category"] == 4
    rels = graph.count_by_relation()
    for rel in (
        "WRITTEN_BY",
        "ADDRESSES",
        "PROPOSES_OR_USES",
        "EVALUATED_ON",
        "MEASURED_BY",
        "IN_CATEGORY",
        "CO_AUTHOR",
    ):
        assert rels.get(rel, 0) > 0, rel
    # "Ada Lovelace" co-authors every paper with "Author i": weight = number of shared papers.
    assert graph.edge_weight("author:adalovelace", "author:alanturing", "CO_AUTHOR") == 3


def test_graph_save_load(tmp_path, graph):
    path = tmp_path / "kg.graphml"
    graph.save(path)
    assert path.exists() and path.with_suffix(".pkl").exists()
    again = NetworkXGraphStore.load(path)
    assert again.count_by_relation() == graph.count_by_relation()


def test_which_papers_use_squad(graph, normalizer):
    res = graph_query(graph, "SQuAD", "EVALUATED_ON", normalizer=normalizer)
    assert res.matched and set(res.papers) == {"2301.00001", "2506.00006"}


def test_methods_for_machine_translation(graph, normalizer):
    res = graph_query(graph, "MT", "ADDRESSES", target_type="Method", normalizer=normalizer)
    names = [n for n, _ in res.entities]
    assert set(res.papers) == {"2302.00002", "2405.00005"}
    assert "back-translation" in names and "COMET" in names


def test_query_falls_back_on_other_types(graph, normalizer):
    res = graph_query(graph, "dense retrieval", "EVALUATED_ON", normalizer=normalizer)
    assert res.entity_type == "Method" and set(res.papers) == {"2301.00001", "2506.00006"}


def test_fuzzy_entity_lookup(graph):
    assert find_entity(graph, "Squad", "Dataset") == "dataset:squad"
    assert find_entity(graph, "unknown thing", "Dataset") is None


def test_coauthor_query(graph):
    res = graph_query(
        graph, "Ada Lovelace", "CO_AUTHOR", entity_type="Author", target_type="Author"
    )
    assert res.entities[0] == ("Alan Turing", 3)


def test_rules_parse_questions():
    q = parse_question_rules("Which papers use SQuAD?")
    assert (q.entity, q.entity_type, q.relation) == ("SQuAD", "Dataset", "EVALUATED_ON")
    q = parse_question_rules("What methods are used for machine translation?")
    assert (q.entity, q.target_type) == ("machine translation", "Method")
    q = parse_question_rules("Quelles méthodes pour la traduction automatique ?")
    assert q.entity == "traduction automatique" and q.target_type == "Method"


def test_answer_graph_question_with_citations(graph, paper_store, normalizer):
    def router(messages):
        if "translate a question" in messages[0]["content"]:
            return {
                "entity": "SQuAD",
                "entity_type": "Dataset",
                "relation": "EVALUATED_ON",
                "target_type": "Paper",
            }
        assert "arXiv:2506.00006" in messages[-1]["content"]
        return "Two papers use SQuAD [arXiv:2506.00006, arXiv:2301.00001] and [arXiv:1234.5678]."

    llm = LLMClient("m", completion_fn=ScriptedLLM(router))
    ans = answer_graph_question("Which papers use SQuAD?", graph, paper_store, llm, normalizer)
    assert set(ans.citations) == {"2506.00006", "2301.00001"} and ans.invalid_citations == [
        "1234.5678"
    ]


def test_answer_graph_question_without_llm(graph, paper_store, normalizer):
    ans = answer_graph_question("Which papers use SQuAD?", graph, paper_store, None, normalizer)
    assert "[arXiv:2301.00001]" in ans.answer


def test_author_stats(graph):
    top = central_authors(graph, 2)
    assert top[0]["author"] == "Ada Lovelace"
    assert author_communities(graph, min_size=2)


# ---------------------------------------------------------- extraction


def test_select_papers_prioritises_accepted_or_code(papers_df):
    sel = select_papers(papers_df, 3)
    prio = set(papers_df[papers_df.is_accepted | papers_df.has_code].arxiv_id)
    assert set(sel) <= prio and len(sel) == 3


def test_extract_batch_with_missing_item(paper_store):
    papers = paper_store.many(["2301.00001", "2302.00002"])

    def router(messages):
        user = messages[-1]["content"]
        if "EACH paper" in user:  # batch reply forgets the second paper
            return {
                "papers": [{"arxiv_id": "arXiv:2301.00001", "tasks": ["QA"], "datasets": ["SQuAD"]}]
            }
        return {"tasks": ["machine translation"], "metrics": ["BLEU"]}

    fake = ScriptedLLM(router)
    out = extract_batch(papers, LLMClient("m", completion_fn=fake))
    assert out["2301.00001"].datasets == ["SQuAD"]
    assert out["2302.00002"].metrics == ["BLEU"] and len(fake.calls) == 2


def test_load_cache_skips_corrupt_lines(tmp_path):
    p = tmp_path / "ex.jsonl"
    p.write_text('{"arxiv_id": "1", "extraction": {"tasks": ["QA"]}}\nnot json\n', encoding="utf-8")
    assert list(load_cache(p)) == ["1"]
