from mindgraph.timeline import concept_timeline


def _raw(tmp_path):
    raw = tmp_path / "raw" / "devdiary"
    raw.mkdir(parents=True)
    (raw / "2026-01-10.md").write_text(
        "---\ndate: 2026-01-10\n---\n# Day\n\nStarted on the caching layer design.\n",
        encoding="utf-8",
    )
    (raw / "2026-03-15.md").write_text(
        "# Mid\n\nRefined the caching layer with TTL expiry.\n", encoding="utf-8"
    )
    (raw / "unrelated.md").write_text("nothing relevant here\n", encoding="utf-8")
    return tmp_path / "raw"


def test_timeline_orders_by_date(tmp_path):
    out = concept_timeline(_raw(tmp_path), "caching")
    assert out["concept"] == "caching"
    assert out["mentions"] == 2
    dates = [e["date"] for e in out["timeline"]]
    assert dates == sorted(dates)
    assert dates[0] == "2026-01-10"   # from frontmatter date
    assert "2026-03-15" in dates      # from YYYY-MM-DD filename fallback


def test_timeline_no_match(tmp_path):
    out = concept_timeline(_raw(tmp_path), "nonexistent-xyz")
    assert out["mentions"] == 0 and out["timeline"] == []


def test_timeline_missing_raw_dir(tmp_path):
    out = concept_timeline(tmp_path / "raw", "anything")
    assert out["mentions"] == 0


def test_timeline_snippet_contains_concept(tmp_path):
    out = concept_timeline(_raw(tmp_path), "caching")
    assert all("caching" in e["snippet"].lower() for e in out["timeline"])
