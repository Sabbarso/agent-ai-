import pandas as pd
import pytest

from rpe.data.enrich_comments import (
    enrich_dataframe,
    enrich_record,
    extract_github_url,
    extract_n_figures,
    extract_n_pages,
    extract_venue,
)

# (comments, journal_ref, expected venue, expected year, expected is_accepted)
VENUE_CASES = [
    ("Accepted at ACL 2024", None, "ACL", 2024, True),
    ("To appear in NeurIPS 2023", None, "NeurIPS", 2023, True),
    ("Accepted to EMNLP 2023 Findings", None, "EMNLP Findings", 2023, True),
    ("Findings of ACL 2023, 15 pages", None, "ACL Findings", 2023, True),
    ("NAACL 2024 main conference, camera-ready", None, "NAACL", 2024, True),
    ("EACL 2024", None, "EACL", 2024, True),
    ("Camera-ready version for ICLR 2024; 9 pages", None, "ICLR", 2024, True),
    ("Under review at ICLR 2025", None, "ICLR", 2025, False),
    ("Submitted to ACL Rolling Review", None, "ACL", None, False),
    ("NeurIPS'23 spotlight", None, "NeurIPS", 2023, True),
    ("Accepted by AAAI-24", None, "AAAI", 2024, True),
    (
        "Published in Transactions of the Association for Computational Linguistics (TACL)",
        None,
        "TACL",
        None,
        True,
    ),
    (
        "Accepted at the 2nd Workshop on Fact Verification at COLING 2025",
        None,
        "COLING Workshop",
        2025,
        True,
    ),
    ("Accepted at WWW '24 (The Web Conference)", None, "WWW", 2024, True),
    ("Accepted to TMLR", None, "TMLR", None, True),
    ("Code: https://github.com/acl-org/acl-anthology", None, None, None, False),
    (
        None,
        "Proceedings of the 2023 Conference on Empirical Methods in Natural Language "
        "Processing, pages 100-120",
        "EMNLP",
        2023,
        True,
    ),
    ("12 pages", "Journal of Machine Learning Research 24 (2023)", "JMLR", None, True),
    ("Accepted for publication in a journal", None, None, None, True),
    (None, None, None, None, False),
]


@pytest.mark.parametrize(("comments", "jref", "venue", "year", "accepted"), VENUE_CASES)
def test_extract_venue(comments, jref, venue, year, accepted):
    info = extract_venue(comments, jref)
    assert (info.venue, info.venue_year, info.is_accepted) == (venue, year, accepted)


def test_venue_from_abstract_requires_cue():
    abstract = "We propose X. This paper has been accepted to ACL 2023. Code is released."
    assert extract_venue(None, None, abstract).venue == "ACL"
    assert extract_venue(None, None, "We improve over prior ACL work.").venue is None


def test_url_does_not_trigger_www_venue():
    assert extract_venue("Demo at www.example.com/demo", None).venue is None


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("Code: https://github.com/x/y",), "https://github.com/x/y"),
        (("Project and code at github.com/foo/bar-baz.",), "https://github.com/foo/bar-baz"),
        (("See https://github.com/org/repo.git for details",), "https://github.com/org/repo"),
        ((None, "Our code is available at https://github.com/a/b."), "https://github.com/a/b"),
        (("No code here",), None),
    ],
)
def test_extract_github_url(texts, expected):
    assert extract_github_url(*texts) == expected


@pytest.mark.parametrize(
    ("comments", "pages", "figures"),
    [
        ("12 pages, 3 figures", 12, 3),
        ("8+2 pages, 4 figures, 5 tables", 10, 4),
        ("One figure, 5 pages", 5, 1),
        ("10pp, 2 figs", 10, 2),
        ("Accepted at ACL 2024", None, None),
        (None, None, None),
    ],
)
def test_pages_and_figures(comments, pages, figures):
    assert extract_n_pages(comments) == pages
    assert extract_n_figures(comments) == figures


def test_enrich_record_full():
    e = enrich_record(
        "Accepted at ACL 2024. 9 pages, 2 figures. Code: https://github.com/x/y", None, ""
    )
    assert e.venue == "ACL" and e.venue_year == 2024 and e.is_accepted
    assert e.github_url == "https://github.com/x/y" and e.has_code
    assert (e.n_pages, e.n_figures) == (9, 2)


def test_has_code_with_gitlab():
    assert enrich_record("Code at https://gitlab.com/team/proj", None, None).has_code


def test_enrich_dataframe_dtypes():
    df = pd.DataFrame(
        {
            "comments": ["Accepted at ICML 2024, 8 pages", None],
            "journal_ref": [None, None],
            "abstract": ["abc", "def"],
        }
    )
    out = enrich_dataframe(df)
    assert out.loc[0, "venue"] == "ICML" and pd.isna(out.loc[1, "venue"])
    assert out["n_pages"].dtype.name == "Int64"
    assert out["has_code"].dtype == bool
