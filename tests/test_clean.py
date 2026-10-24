from datetime import date

import orjson
import pandas as pd

from rpe.config import Config, DataConfig
from rpe.data.clean import (
    authors_from_record,
    clean_record,
    has_wanted_category,
    in_date_range,
    make_raw_filter,
    normalize_whitespace,
    parse_categories,
    parse_published_date,
    reservoir_sample,
    strip_latex_accents,
)
from rpe.data.load import StreamStats, stream_records
from rpe.data.prepare import run_prepare


def make_record(arxiv_id: str, categories: str, created: str, **extra) -> dict:
    rec = {
        "id": arxiv_id,
        "authors": "A. One, B. Two",
        "title": "A  title\n  on two lines",
        "comments": None,
        "journal-ref": None,
        "doi": None,
        "categories": categories,
        "abstract": "  An abstract\nwith   spaces. ",
        "versions": [{"version": "v1", "created": created}],
        "update_date": "2024-02-01",
        "authors_parsed": [["One", "A.", ""], ["Two", "B.", ""]],
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------- dates


def test_parse_published_date_rfc2822():
    versions = [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}]
    assert parse_published_date(versions) == date(2007, 4, 2)


def test_parse_published_date_uses_first_version():
    versions = [
        {"version": "v1", "created": "Tue, 3 Jan 2023 10:00:00 GMT"},
        {"version": "v2", "created": "Fri, 1 Mar 2024 10:00:00 GMT"},
    ]
    assert parse_published_date(versions) == date(2023, 1, 3)


def test_parse_published_date_invalid():
    assert parse_published_date([]) is None
    assert parse_published_date(None) is None
    assert parse_published_date([{"created": "not a date"}]) is None


def test_in_date_range():
    start, end = date(2023, 1, 1), date(2023, 12, 31)
    assert in_date_range(date(2023, 1, 1), start, end)
    assert in_date_range(date(2023, 12, 31), start, end)
    assert not in_date_range(date(2022, 12, 31), start, end)
    assert in_date_range(date(2030, 1, 1), start, None)
    assert not in_date_range(None, None, None)


# ----------------------------------------------------------- categories


def test_categories():
    cats = parse_categories("cs.LG cs.CL")
    assert cats == ["cs.LG", "cs.CL"]
    assert has_wanted_category(cats, ["cs.CL"])
    assert not has_wanted_category(["hep-ph"], ["cs.CL"])
    assert not has_wanted_category(["cs.CLX"], ["cs.CL"])
    assert has_wanted_category(["cs.AI"], ["cs.*"])
    assert parse_categories(None) == []


def test_raw_filter():
    f = make_raw_filter(["cs.CL"])
    assert f(b'{"categories":"cs.CL"}') and not f(b'{"categories":"hep-ph"}')


# ---------------------------------------------------------------- text


def test_normalize_whitespace():
    assert normalize_whitespace("  a\n  b\tc  ") == "a b c"
    assert normalize_whitespace(None) == ""


def test_latex_accents():
    assert strip_latex_accents(r"Bal\'azs") == "Balázs"
    assert strip_latex_accents(r"M{\"u}ller") == "Müller"
    assert strip_latex_accents(r"Fran\c{c}ois") == "François"


def test_authors_prefers_parsed():
    rec = {"authors_parsed": [["Balázs", "C.", ""], ["Yuan", "C. -P.", "Jr"]], "authors": "x"}
    assert authors_from_record(rec) == ["C. Balázs", "C. -P. Yuan Jr"]


def test_authors_fallback_to_raw():
    assert authors_from_record({"authors": r"A. Bal\'azs, B. Two and C. Three"}) == [
        "A. Balázs",
        "B. Two",
        "C. Three",
    ]


def test_clean_record_columns():
    row = clean_record(make_record("2301.00001", "cs.CL cs.LG", "Tue, 3 Jan 2023 10:00:00 GMT"))
    assert row["title"] == "A title on two lines"
    assert row["abstract"] == "An abstract with spaces."
    assert row["authors_list"] == ["A. One", "B. Two"] and row["n_authors"] == 2
    assert row["primary_category"] == "cs.CL"
    assert (row["year"], row["month"]) == (2023, 1)
    assert row["update_date"] == date(2024, 2, 1)
    assert row["n_versions"] == 1 and row["journal_ref"] is None


def test_clean_record_missing_id():
    assert clean_record({"title": "x"}) is None


# ------------------------------------------------------------- sampling


def test_reservoir_sample_deterministic():
    a = reservoir_sample(range(1000), 50, seed=42)
    b = reservoir_sample(range(1000), 50, seed=42)
    assert a == b and len(a) == 50 and len(set(a)) == 50
    assert reservoir_sample(range(10), 50) == list(range(10))
    assert reservoir_sample(range(10), None) == list(range(10))


# --------------------------------------------------------- stream + e2e


def _write_jsonl(path, records, extra_lines=()):
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec) + b"\n")
        for line in extra_lines:
            fh.write(line + b"\n")


def test_stream_records_counts_invalid(tmp_path):
    p = tmp_path / "raw.json"
    _write_jsonl(p, [{"id": "1"}, {"id": "2"}], [b"{not json", b"[1, 2]"])
    stats = StreamStats()
    records = list(stream_records(p, stats=stats, progress=False))
    assert [r["id"] for r in records] == ["1", "2"]
    assert (stats.n_lines, stats.n_invalid) == (4, 2)


def test_run_prepare_end_to_end(tmp_path):
    raw = tmp_path / "raw.json"
    records = [
        make_record(
            "2301.00001",
            "cs.CL",
            "Tue, 3 Jan 2023 10:00:00 GMT",
            comments="Accepted at ACL 2023. Code: https://github.com/a/b",
        ),
        make_record("2301.00002", "cs.LG cs.CL", "Wed, 4 Jan 2023 10:00:00 GMT"),
        make_record("2212.00003", "cs.CL", "Thu, 1 Dec 2022 10:00:00 GMT"),  # too old
        make_record("2301.00004", "hep-ph", "Tue, 3 Jan 2023 10:00:00 GMT"),  # wrong cat
    ]
    _write_jsonl(raw, records, [b"garbage cs.CL"])
    cfg = Config(
        data=DataConfig(
            raw_path=str(raw),
            processed_dir="processed",
            categories=["cs.CL"],
            start_date=date(2023, 1, 1),
            max_papers=10,
            sample_dev=1,
        ),
        root=tmp_path,
    )
    out, df, report = run_prepare(cfg, progress=False)
    assert out.exists() and list(df["arxiv_id"]) == ["2301.00001", "2301.00002"]
    assert report["scan"]["n_invalid"] == 1 and report["scan"]["n_kept"] == 2
    assert df.loc[0, "venue"] == "ACL" and bool(df.loc[0, "has_code"])
    reloaded = pd.read_parquet(out)
    assert list(reloaded["categories"].iloc[1]) == ["cs.LG", "cs.CL"]

    # Resumable: second call reuses the file; --dev writes a separate, smaller file.
    assert run_prepare(cfg, progress=False)[2]["reused"]
    out_dev, df_dev, _ = run_prepare(cfg, dev=True, progress=False)
    assert out_dev.name == "papers_dev.parquet" and len(df_dev) == 1
